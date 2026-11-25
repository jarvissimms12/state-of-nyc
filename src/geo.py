"""Geography helpers: community district codes and borough lookups.

A community district (CD) code is borough * 100 + district number, e.g.
Brooklyn CD 7 = 307. NYC has 59 CDs. Codes like 164 (Central Park) or
480-484 (airports, large parks) are "joint interest areas" with almost no
residents, so they are excluded from per-district rates.
"""
from __future__ import annotations

import re

import pandas as pd

CD_COUNTS = {1: 12, 2: 12, 3: 18, 4: 14, 5: 3}
VALID_CDS = frozenset(b * 100 + n for b, k in CD_COUNTS.items() for n in range(1, k + 1))

BORO_NAME_TO_CODE = {
    "MANHATTAN": 1, "NEW YORK": 1, "MN": 1,
    "BRONX": 2, "THE BRONX": 2, "BX": 2,
    "BROOKLYN": 3, "KINGS": 3, "BK": 3,
    "QUEENS": 4, "QN": 4,
    "STATEN ISLAND": 5, "RICHMOND": 5, "SI": 5,
}


def cd_code(borough: int | str | None, district: int | str | None) -> int | None:
    """Combine a borough (code or name) and district number into a valid CD code."""
    if borough is None or district is None or pd.isna(borough) or pd.isna(district):
        return None
    if isinstance(borough, str) and not borough.strip().isdigit():
        b = BORO_NAME_TO_CODE.get(borough.strip().upper())
    else:
        b = int(borough)
    try:
        d = int(str(district).strip())
    except ValueError:
        return None
    code = b * 100 + d if b else None
    return code if code in VALID_CDS else None


def parse_311_board(value: str | None) -> int | None:
    """311 stores community boards as text like '07 BROOKLYN' or '0 Unspecified'."""
    if not isinstance(value, str):
        return None
    m = re.match(r"^\s*(\d{1,2})\s+([A-Z ]+?)\s*$", value.upper())
    if not m:
        return None
    return cd_code(m.group(2), m.group(1))


def normalize_cd(value) -> int | None:
    """Accept a 3-digit CD code given as int, float or string ('307', 307.0)."""
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return None
    return code if code in VALID_CDS else None


# ---------------------------------------------------------------------------
# Borough from a free-text work address (Jobs NYC postings)
# ---------------------------------------------------------------------------
_E = r"(?![A-Z])"          # end of a place name (word boundary that also works after '.')
_BORO_PATTERNS = [
    # Landmarks without a street address (legal borough of each site)
    (2, r"RIKERS\s+ISLAND|JEROME\s+PARK"),
    (1, r"CITY\s+HALL|WORLD\s+TRADE\s+CENTER|UNITED\s+NATIONS|RANDALLS?\s+ISLAND|WARDS\s+ISLAND"),
    # Explicit borough names win over neighborhood or street names
    (4, r"(?<![A-Z])(QUEENS|QNS)" + _E),
    (2, r"(?<![A-Z])(BRONX|BX)" + _E),
    (3, r"(?<![A-Z])(BROOKLYN|BKLYN|BK)" + _E + r"|METRO\s*TECH"),
    (5, r"(?<![A-Z])(STATEN\s+IS(LAND)?|S\.\s?I\.|ST\.?\s+GEORGE)"),
    # Queens addresses usually name the neighborhood instead of the borough
    (4, r"(?<![A-Z])(L\.?\s?I\.?\s?CITY|LONG\s+IS(LAND)?(\s+CITY)?|JAMAICA|FLUSHING|ASTORIA|"
        r"CORONA|WOODSIDE|ELMHURST|RIDGEWOOD|FAR\s+ROCKAWAY|KEW\s+GARDENS|REGO\s+PARK|FOREST\s+HILLS|"
        r"COLLEGE\s+POINT|BAYSIDE|SUNNYSIDE|MASPETH|RICHMOND\s+HILL|OZONE\s+PARK)" + _E),
    # "New York, N.Y." is the USPS city name for Manhattan. A bare "NY" or "NYC"
    # is just the state/city and says nothing about the borough.
    (1, r"(?<![A-Z])(MANHATTAN|NEW\s+YORK|N\.\s?Y\.|NY\s*,?\s*NY)" + _E),
]
_OUTSIDE_NYC = re.compile(r"\b(VALHALLA|YONKERS|KINGSTON|ALBANY|GRAHAMSVILLE|DOWNSVILLE|"
                          r"WESTCHESTER|ULSTER|DELAWARE|SULLIVAN|PUTNAM|CARMEL|MAHOPAC)\b")
_ALL_BOROUGHS = re.compile(r"ALL\s+BOROUGHS|CITYWIDE|VARIOUS|MULTIPLE|QUEENS\s*-\s*S\.?\s?I", re.I)


def borough_from_address_text(address: str | None) -> int | str | None:
    """Borough code from explicit place names in an address string.

    Returns 'outside' for upstate watershed sites, 'multiple' for postings
    covering several boroughs, or None when the text alone doesn't say.
    """
    if not isinstance(address, str) or not address.strip():
        return None
    text = address.upper()
    if _ALL_BOROUGHS.search(text):
        return "multiple"
    if _OUTSIDE_NYC.search(text):
        return "outside"
    for code, pat in _BORO_PATTERNS:
        if re.search(pat, text):
            return code
    return None


_SUFFIXES = {
    "ST": "STREET", "ST.": "STREET", "AVE": "AVENUE", "AVE.": "AVENUE", "AV": "AVENUE",
    "BLVD": "BOULEVARD", "BLVD.": "BOULEVARD", "PL": "PLACE", "PL.": "PLACE",
    "RD": "ROAD", "DR": "DRIVE", "EXPWY": "EXPRESSWAY", "PKWY": "PARKWAY",
    "E": "EAST", "W": "WEST", "N": "NORTH", "S": "SOUTH", "PLZ": "PLAZA",
}


def street_address_key(address: str | None) -> str | None:
    """Normalize '100 Gold St., N.Y.' -> '100 GOLD STREET' to match PLUTO addresses."""
    if not isinstance(address, str):
        return None
    text = address.upper().split(",")[0]
    text = re.sub(r"(\d+)(ST|ND|RD|TH)\b", r"\1", text)          # 28TH -> 28
    text = re.sub(r"\b(\d+)\s*-\s*(\d+)\b", r"\1-\2", text)       # 30 - 30 -> 30-30
    text = re.sub(r"[^A-Z0-9\- ]", " ", text)
    tokens = text.split()
    if len(tokens) < 2 or not re.match(r"^\d", tokens[0]):
        return None
    # drop trailing city words that sometimes follow without a comma
    stop = {"NY", "N", "NYC", "NEW", "BRONX", "BROOKLYN", "QUEENS", "MANHATTAN", "STATEN",
            "LONG", "L", "LIC", "JAMAICA", "FLUSHING", "CORONA", "QNS", "ASTORIA", "FL", "FLOOR"}
    out = [tokens[0]]
    for tok in tokens[1:]:
        if tok in stop and len(out) >= 3:
            break
        out.append(_SUFFIXES.get(tok, tok))
    return " ".join(out)


def street_name(key: str | None) -> str | None:
    """'125 WORTH STREET' -> 'WORTH STREET' (drop the house number)."""
    if not isinstance(key, str) or " " not in key:
        return None
    return key.split(" ", 1)[1]
