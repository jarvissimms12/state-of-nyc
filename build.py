"""Pull every source from NYC Open Data and build the processed tables.

    python build.py            # ~3-5 minutes; all heavy aggregation runs server-side

Outputs (data/processed/):
    cd_profile.csv              one row per community district (59)
    schools.csv                 one row per public school, with its district
    job_postings.csv            Jobs NYC postings with resolved borough
    payroll_agencies.csv        FY2025 headcount/pay by agency + borough coding
    community_districts.geojson simplified district boundaries
    data_quality.json           every count dropped, unmatched or unresolved
    state_of_nyc.db             all of the above as SQLite tables
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from etl import geography, housing, jobs, schools
from src.config import BOROUGHS, FY_END, FY_START, PROCESSED, RAW, SQR_SCHOOL_YEAR

HOUSING_RATES = ["heat_hw_per_1k", "class_c_per_1k", "evictions_per_1k"]


def housing_distress_index(df: pd.DataFrame, cols=HOUSING_RATES) -> pd.Series:
    """Mean of z-scores of log rates. Logs because the rates are right-skewed
    (heat complaints range 6-275 per 1,000 units); z-scores so each measure
    counts equally. 0 = typical district, +1 = one SD worse than typical."""
    z = []
    for c in cols:
        x = np.log(df[c].clip(lower=1e-6))
        z.append((x - x.mean()) / x.std(ddof=0))
    return pd.concat(z, axis=1).mean(axis=1)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    quality: dict = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "window": {"start": FY_START, "end": FY_END, "school_year": f"{SQR_SCHOOL_YEAR}-{int(SQR_SCHOOL_YEAR) % 100 + 1}"}}

    print("[1/4] Geography");  geography.boundaries(); names = geography.names()
    print("[2/4] Housing");    house, q = housing.build(); quality["housing"] = q
    print("[3/4] Schools");    school_rows, school_cd, q = schools.build(); quality["schools"] = q
    print("[4/4] Jobs");       postings, payroll, q = jobs.build(); quality["jobs"] = q

    # --- community district profile -------------------------------------
    cd = (house.merge(school_cd, on="cd", how="left").merge(names, on="cd", how="left"))
    cd["boro_code"] = cd["cd"] // 100
    cd["borough"] = cd["boro_code"].map(BOROUGHS)
    cd["district"] = cd["borough"] + " CD " + (cd["cd"] % 100).astype(str)
    cd["housing_distress_index"] = housing_distress_index(cd)
    lead = ["cd", "district", "borough", "boro_code", "cd_name"]
    cd = cd[lead + [c for c in cd.columns if c not in lead]].sort_values("cd")
    cd.to_csv(PROCESSED / "cd_profile.csv", index=False)

    keep = ["dbn", "school_name", "school_type", "cd", "latitude", "longitude",
            "chronic_absence_ems", "n_chronic_ems", "chronic_absence_hs", "n_chronic_hs",
            "chronic_absence_rate", "n_chronic", "attendance_rate", "grad_rate_4yr",
            "n_grad_rate_4yr", "eni_2122", "poverty_2122", "enrollment_2122",
            "sth_students", "sth_total_students"]
    school_rows[keep].to_csv(PROCESSED / "schools.csv", index=False)

    pkeep = ["job_id", "agency", "business_title", "civil_service_title", "job_category",
             "career_level", "full_time_part_time_indicator", "number_of_positions",
             "salary_range_from", "salary_range_to", "salary_frequency", "annual_salary_mid",
             "work_location", "boro_code", "geocode_method", "posting_date", "process_date"]
    postings[pkeep].to_csv(PROCESSED / "job_postings.csv", index=False)

    nyc = payroll[payroll["work_location_borough"].isin(
        ["MANHATTAN", "BRONX", "BROOKLYN", "QUEENS", "RICHMOND"])]
    agency = payroll.assign(w=payroll["headcount"] * payroll["mean_base_salary"]).groupby("agency_name").agg(
        headcount=("headcount", "sum"), salary_total=("w", "sum"))
    agency["mean_base_salary"] = agency.pop("salary_total") / agency["headcount"]
    man = nyc[nyc["work_location_borough"] == "MANHATTAN"].groupby("agency_name")["headcount"].sum()
    agency["share_coded_manhattan"] = (man / nyc.groupby("agency_name")["headcount"].sum()).reindex(agency.index)
    agency = agency.sort_values("headcount", ascending=False).reset_index()
    agency.to_csv(PROCESSED / "payroll_agencies.csv", index=False)

    with sqlite3.connect(PROCESSED / "state_of_nyc.db") as con:
        for name, frame in {"cd_profile": cd, "schools": school_rows[keep],
                            "job_postings": postings[pkeep], "payroll_agencies": agency}.items():
            frame.to_sql(name, con, if_exists="replace", index=False)

    (PROCESSED / "data_quality.json").write_text(json.dumps(quality, indent=2, default=int))
    print(json.dumps(quality, indent=2, default=int))


if __name__ == "__main__":
    main()
