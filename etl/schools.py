"""Public school outcomes, 2024-25, at school level and by community district.

Source: NYC DOE School Quality Reports (SQR), long format: one row per
school x metric.

Watch out: the SQR metric named `chronic_absent_*` is displayed as
"Percentage of Students with >90% Attendance". It is the share of students
who were NOT chronically absent. The chronic absence rate is 1 - value.
(Citywide K-8 mean is ~0.67, i.e. ~33% chronically absent, matching DOE's
published figure.) Tests in tests/test_schools.py pin this down.

Schools are placed in community districts with the DOE school locations file
(DBN -> community_district). Economic Need Index (2021-22) and Students in
Temporary Housing (2020-21) are the latest school-level releases on NYC
Open Data and are used as lagged context, not as same-year measures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import soda
from src.config import DATASETS, RAW, SQR_SCHOOL_YEAR
from src.geo import normalize_cd

METRICS = {
    "chronic_absent_ems_all": "pct90_ems",
    "chronic_absent_hs_all": "pct90_hs",
    "attendance_k8_all": "attendance_rate",
    "attendance_hs_all": "attendance_rate",
    "grad_pct_4_all": "grad_rate_4yr",
}


def sqr_metrics() -> pd.DataFrame:
    names = ", ".join(f"'{m}'" for m in METRICS)
    df = soda.fetch_all(
        DATASETS["sqr"],
        select="dbn, school_name, report_type, school_type, metric_variable_name, "
               "number_of_students, metric_value",
        where=f"school_year = '{SQR_SCHOOL_YEAR}' AND metric_variable_name in ({names})",
        order="dbn, report_type, metric_variable_name",
    )
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
    df["number_of_students"] = pd.to_numeric(df.get("number_of_students"), errors="coerce")
    df["measure"] = df["metric_variable_name"].map(METRICS)
    return df.dropna(subset=["metric_value"])


def to_school_level(sqr: pd.DataFrame) -> pd.DataFrame:
    """One row per school, one column per measure.

    A 6-12 school reports an EMS (grades 6-8) row and an HS row. Elementary
    and middle schools mostly serve their own neighborhood, while high school
    students commute across the city, so the elementary/middle chronic absence
    rate is kept separately as the main neighborhood measure. The all-grades
    rate combines both, weighted by the number of students behind each value.
    """
    out = []
    for measure, grp in sqr.groupby("measure"):
        g = grp.assign(w=grp["number_of_students"].fillna(0))
        agg = g.groupby("dbn").apply(
            lambda x: pd.Series({
                measure: np.average(x["metric_value"], weights=x["w"]) if x["w"].sum() > 0
                else x["metric_value"].mean(),
                f"n_{measure}": x["w"].sum(),
            }), include_groups=False)
        out.append(agg)
    schools = pd.concat(out, axis=1).reset_index()
    for m in set(METRICS.values()):            # a measure absent from a pull -> all NaN
        for col in (m, f"n_{m}"):
            if col not in schools:
                schools[col] = np.nan
    names = sqr.drop_duplicates("dbn").set_index("dbn")[["school_name", "school_type"]]
    schools = schools.merge(names, left_on="dbn", right_index=True, how="left")

    # SQR "chronic_absent" = share attending >90% of days -> invert
    schools["chronic_absence_ems"] = 1 - schools["pct90_ems"]
    schools["chronic_absence_hs"] = 1 - schools["pct90_hs"]
    w_e = schools["n_pct90_ems"].fillna(0).where(schools["pct90_ems"].notna(), 0)
    w_h = schools["n_pct90_hs"].fillna(0).where(schools["pct90_hs"].notna(), 0)
    num = (schools["chronic_absence_ems"].fillna(0) * w_e
           + schools["chronic_absence_hs"].fillna(0) * w_h)
    schools["n_chronic"] = w_e + w_h
    schools["chronic_absence_rate"] = (num / schools["n_chronic"]).where(schools["n_chronic"] > 0)
    schools = schools.rename(columns={"n_pct90_ems": "n_chronic_ems", "n_pct90_hs": "n_chronic_hs"})
    return schools.drop(columns=["pct90_ems", "pct90_hs"])


def school_locations() -> pd.DataFrame:
    df = soda.query(DATASETS["school_locations"],
                    "SELECT system_code, community_district, latitude, longitude LIMIT 5000")
    df = df.rename(columns={"system_code": "dbn"})
    df["dbn"] = df["dbn"].str.strip()
    df["cd"] = df["community_district"].map(normalize_cd)
    return df.drop_duplicates("dbn")[["dbn", "cd", "latitude", "longitude"]]


def economic_need() -> pd.DataFrame:
    df = soda.query(DATASETS["demographics"], """
        SELECT dbn, total_enrollment, economic_need_index, poverty_1
        WHERE year = '2021-22' LIMIT 5000""")
    pct = lambda s: pd.to_numeric(s.astype(str).str.rstrip("%").replace({"Above 95": "95"}),
                                  errors="coerce") / 100
    df["eni_2122"] = pct(df["economic_need_index"])
    df["poverty_2122"] = pct(df["poverty_1"])
    df["enrollment_2122"] = pd.to_numeric(df["total_enrollment"], errors="coerce")
    return df[["dbn", "eni_2122", "poverty_2122", "enrollment_2122"]]


def temporary_housing() -> pd.DataFrame:
    df = soda.query(DATASETS["sth"], """
        SELECT dbn, total_students, students_in_temporary_housing,
               students_residing_in_shelter LIMIT 5000""")
    for c in ("total_students", "students_in_temporary_housing", "students_residing_in_shelter"):
        df[c] = pd.to_numeric(df[c], errors="coerce")   # small counts are suppressed as 's'
    return df.rename(columns={"total_students": "sth_total_students",
                              "students_in_temporary_housing": "sth_students",
                              "students_residing_in_shelter": "sth_shelter"})


def weighted(df: pd.DataFrame, value: str, weight: str, by: str = "cd") -> pd.Series:
    d = df.dropna(subset=[value, weight])
    d = d[d[weight] > 0]
    return d.groupby(by).apply(lambda x: np.average(x[value], weights=x[weight]),
                               include_groups=False)


def build() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sqr = sqr_metrics()
    schools = to_school_level(sqr)
    schools = (schools.merge(school_locations(), on="dbn", how="left")
                      .merge(economic_need(), on="dbn", how="left")
                      .merge(temporary_housing(), on="dbn", how="left"))
    quality = {
        "schools_in_sqr": int(len(schools)),
        "schools_matched_to_cd": int(schools["cd"].notna().sum()),
        "schools_with_eni": int(schools["eni_2122"].notna().sum()),
    }
    located = schools.dropna(subset=["cd"]).copy()
    located["cd"] = located["cd"].astype(int)

    by_cd = pd.DataFrame({
        "chronic_absence_ems": weighted(located, "chronic_absence_ems", "n_chronic_ems"),
        "chronic_absence_rate": weighted(located, "chronic_absence_rate", "n_chronic"),
        "grad_rate_4yr": weighted(located, "grad_rate_4yr", "n_grad_rate_4yr"),
        "eni_2122": weighted(located, "eni_2122", "enrollment_2122"),
    })
    sth = (located.dropna(subset=["sth_students", "sth_total_students"])
                  .groupby("cd")[["sth_students", "sth_total_students"]].sum())
    by_cd["sth_rate_2021"] = sth["sth_students"] / sth["sth_total_students"]
    by_cd["n_schools"] = located.groupby("cd").size()
    by_cd["students_measured"] = located.groupby("cd")["n_chronic"].sum()
    by_cd = by_cd.reset_index()

    schools.to_csv(RAW / "schools_school_level.csv", index=False)
    by_cd.to_csv(RAW / "schools_cd.csv", index=False)
    return schools, by_cd, quality
