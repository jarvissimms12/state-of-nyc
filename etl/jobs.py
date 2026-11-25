"""City government jobs: where the openings are, and what the payroll can and
can't tell us about geography.

1. Jobs NYC postings (live snapshot): every posting's free-text work address
   is resolved to a borough by
     a) explicit place names in the text ('Bronx', 'L I City', 'N.Y.'),
     b) exact match of the normalized street address against PLUTO tax lots,
        breaking ties between boroughs by building floor area,
     c) if no lot matches, the street name, when it exists in only one borough,
     d) NYC addressing convention: hyphenated house numbers ('30-30 47th Ave')
        are Queens addresses.
   Anything else is left unresolved and reported, not guessed.

2. Citywide Payroll FY2025: headcount and pay by agency. Its
   work_location_borough field records where an agency *codes* staff, not
   where they work: all ~94,000 DOE teachers are coded Manhattan. So payroll
   is used for workforce size and pay, never for geography.
"""
from __future__ import annotations

import re

import pandas as pd

from src import soda
from src.config import DATASETS, FY, RAW
from src.geo import BORO_NAME_TO_CODE, borough_from_address_text, street_address_key, street_name


def postings() -> pd.DataFrame:
    df = soda.fetch_all(
        DATASETS["job_postings"],
        select="job_id, agency, posting_type, number_of_positions, business_title, "
               "civil_service_title, job_category, career_level, full_time_part_time_indicator, "
               "salary_range_from, salary_range_to, salary_frequency, work_location, "
               "posting_date, post_until, process_date",
        where=None, order="job_id, posting_type")
    for c in ("number_of_positions", "salary_range_from", "salary_range_to"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Each job is often listed twice (Internal + External). Count each job once.
    return df.sort_values("posting_type").drop_duplicates("job_id", keep="first")


def pluto_lookup(keys: list[str], batch: int = 40) -> pd.DataFrame:
    rows = []
    for i in range(0, len(keys), batch):
        chunk = [k.replace("'", "''") for k in keys[i:i + batch]]
        inlist = ", ".join(f"'{k}'" for k in chunk)
        rows.append(soda.query(DATASETS["pluto"],
                               f"SELECT address, borough, bldgarea WHERE address in ({inlist}) LIMIT 5000"))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["address", "borough", "bldgarea"])
    df["bldgarea"] = pd.to_numeric(df.get("bldgarea"), errors="coerce").fillna(0)
    df["boro_code"] = df["borough"].map(BORO_NAME_TO_CODE)
    return df


def street_boroughs(streets: list[str]) -> dict[str, set[int]]:
    """Which boroughs have at least one tax lot on each street name."""
    found: dict[str, set[int]] = {}
    for st in streets:
        esc = st.replace("'", "''")
        df = soda.query(DATASETS["pluto"],
                        f"SELECT borough, count(*) AS n WHERE address like '% {esc}' GROUP BY borough")
        found[st] = {BORO_NAME_TO_CODE[b] for b in df.get("borough", []) if b in BORO_NAME_TO_CODE}
    return found


def resolve_pluto(matches: pd.DataFrame) -> tuple[int | None, str]:
    if matches.empty:
        return None, "no_pluto_match"
    by_boro = matches.groupby("boro_code")["bldgarea"].max().sort_values(ascending=False)
    if len(by_boro) == 1:
        return int(by_boro.index[0]), "pluto_exact"
    top, second = by_boro.iloc[0], by_boro.iloc[1]
    if top >= 5 * max(second, 1):
        return int(by_boro.index[0]), "pluto_largest_building"
    return None, "pluto_ambiguous"


def geocode_postings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_result = df["work_location"].map(borough_from_address_text)
    df["street_key"] = df["work_location"].map(street_address_key)
    keys = sorted(df.loc[text_result.isna() & df["street_key"].notna(), "street_key"].unique())
    pluto = pluto_lookup(keys)
    unmatched = [k for k in keys if k not in set(pluto["address"])]
    streets = street_boroughs(sorted({street_name(k) for k in unmatched if street_name(k)}))

    boro, method = [], []
    for txt, key in zip(text_result, df["street_key"]):
        if isinstance(txt, int):
            boro.append(txt); method.append("address_text"); continue
        if txt in ("multiple", "outside"):
            boro.append(None); method.append(txt); continue
        if isinstance(key, str):
            b, m = resolve_pluto(pluto[pluto["address"] == key])
            if b is None and m == "no_pluto_match":
                on_street = streets.get(street_name(key), set())
                if len(on_street) == 1:
                    b, m = next(iter(on_street)), "street_in_one_borough"
                elif len(on_street) > 1:
                    m = "street_in_several_boroughs"
            if b is None and re.match(r"^\d+-\d+\s", key):
                b, m = 4, "queens_hyphen_rule"
            boro.append(b); method.append(m); continue
        boro.append(None); method.append("unparseable")
    df["boro_code"] = pd.array(boro, dtype="Int64")
    df["geocode_method"] = method
    return df


def annual_salary(df: pd.DataFrame) -> pd.Series:
    """Midpoint of the posted range, annualized (hourly x 1,820 h, daily x 260 d)."""
    mid = (df["salary_range_from"] + df["salary_range_to"]) / 2
    factor = df["salary_frequency"].map({"Annual": 1, "Hourly": 1820, "Daily": 260}).fillna(1)
    return mid * factor


def payroll_by_agency() -> pd.DataFrame:
    df = soda.query(DATASETS["payroll"], f"""
        SELECT agency_name, work_location_borough, count(*) AS headcount,
               avg(base_salary) AS mean_base_salary
        WHERE fiscal_year = '{FY}' AND leave_status_as_of_june_30 = 'ACTIVE'
              AND pay_basis = 'per Annum'
        GROUP BY agency_name, work_location_borough LIMIT 20000""")
    df["headcount"] = df["headcount"].astype(int)
    df["mean_base_salary"] = df["mean_base_salary"].astype(float)
    return df


def build() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    jobs = geocode_postings(postings())
    jobs["annual_salary_mid"] = annual_salary(jobs)
    payroll = payroll_by_agency()
    quality = {
        "postings_unique_jobs": int(len(jobs)),
        "postings_resolved": int(jobs["boro_code"].notna().sum()),
        "postings_by_method": jobs["geocode_method"].value_counts().to_dict(),
        "postings_snapshot_date": str(jobs["process_date"].max())[:10],
        "payroll_active_annual_headcount": int(payroll["headcount"].sum()),
    }
    jobs.to_csv(RAW / "job_postings_geocoded.csv", index=False)
    payroll.to_csv(RAW / "payroll_fy2025_agency_borough.csv", index=False)
    return jobs, payroll, quality
