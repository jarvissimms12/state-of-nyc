"""Shared settings: the analysis window, dataset IDs and file locations.

Everything is aligned to NYC fiscal year 2025 (July 1, 2024 - June 30, 2025),
which is also the 2024-25 school year, so housing, schools and city jobs
describe the same twelve months.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"

FY = 2025
FY_START = "2024-07-01T00:00:00"
FY_END = "2025-06-30T23:59:59"
# School Quality Reports label a school year by its fall year: 2024 = 2024-25
SQR_SCHOOL_YEAR = "2024"

SODA_DOMAIN = "https://data.cityofnewyork.us"

# NYC Open Data dataset IDs (https://data.cityofnewyork.us/d/<id>)
DATASETS = {
    # Housing
    "311": "erm2-nwe9",               # 311 Service Requests from 2010 to Present
    "hpd_violations": "wvxf-dwi5",    # HPD Housing Maintenance Code Violations
    "evictions": "6z8x-wfk4",         # Evictions executed by City Marshals
    "pluto": "64uk-42ks",             # Primary Land Use Tax Lot Output (PLUTO)
    # Schools
    "sqr": "dnpx-dfnc",               # School Quality Reports Data (2015-16 to 2024-25)
    "school_locations": "wg9x-4ke6",  # 2019-20 School Locations (DBN -> community district)
    "demographics": "c7ru-d68s",      # 2017-18 to 2021-22 Demographic Snapshot (Economic Need Index)
    "sth": "3wtp-43m9",               # 2021 Students in Temporary Housing (school level)
    # Jobs
    "job_postings": "kpav-sd4t",      # Jobs NYC Postings (current openings)
    "payroll": "k397-673e",           # Citywide Payroll Data (Fiscal Year)
    # Geography
    "cd_boundaries": "5crt-au7u",     # Community Districts (boundaries)
}

BOROUGHS = {
    1: "Manhattan",
    2: "Bronx",
    3: "Brooklyn",
    4: "Queens",
    5: "Staten Island",
}
