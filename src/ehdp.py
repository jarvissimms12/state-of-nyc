"""NYC Environment & Health Data Portal indicators (NYC DOHMH).

The Health Department publishes every portal indicator as JSON on GitHub
(github.com/nychealth/EHDP-data, branch `production`). Files are cached in
data/raw/ehdp/ so repeated runs work offline.

    from src import ehdp
    asthma = ehdp.measure(1196, geo="CD")     # long DataFrame: geo_id, period, value
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from .config import RAW

BASE = "https://raw.githubusercontent.com/nychealth/EHDP-data/production"
CACHE = RAW / "ehdp"


def _get(rel: str) -> object:
    path = CACHE / rel
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(4):
        try:
            r = requests.get(f"{BASE}/{rel}", timeout=120)
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(r.text)
            return r.json()
        except (requests.RequestException, ValueError):
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"could not download {rel}")


def time_periods() -> dict[int, str]:
    return {t["TimePeriodID"]: t["TimePeriod"] for t in _get("indicators/metadata/TimePeriods.json")}


def metadata() -> list[dict]:
    return _get("indicators/metadata/metadata.json")


def indicator_of_measure(measure_id: int) -> int:
    for ind in metadata():
        if any(m["MeasureID"] == measure_id for m in ind["Measures"]):
            return ind["IndicatorID"]
    raise KeyError(measure_id)


def measure_name(measure_id: int) -> str:
    for ind in metadata():
        for m in ind["Measures"]:
            if m["MeasureID"] == measure_id:
                return m["MeasureName"]
    raise KeyError(measure_id)


def measure(measure_id: int, geo: str | None = None) -> pd.DataFrame:
    """All values of one measure: columns geo_type, geo_id, period, value."""
    raw = _get(f"indicators/data/{indicator_of_measure(measure_id)}.json")
    df = pd.DataFrame(raw)
    df = df[df["MeasureID"] == measure_id]
    if geo:
        df = df[df["GeoType"] == geo]
    tp = time_periods()
    out = pd.DataFrame({
        "geo_type": df["GeoType"].values,
        "geo_id": pd.to_numeric(df["GeoID"], errors="coerce").values,
        "period": df["TimePeriodID"].map(tp).values,
        "value": pd.to_numeric(df["Value"], errors="coerce").values,
    })
    return out.dropna(subset=["geo_id", "value"]).astype({"geo_id": int})


def latest(measure_id: int, geo: str) -> pd.DataFrame:
    """Most recent period of a measure for one geography type."""
    df = measure(measure_id, geo)
    last = df["period"].iloc[-1] if len(df) else None
    periods = sorted(df["period"].unique(), key=lambda p: (len(str(p)), str(p)))
    # periods like "2019-23" vs "2023": choose the one with the latest end year
    def end_year(p: str) -> int:
        digits = [int(x) for x in str(p).replace("-", " ").split() if x.isdigit()]
        if not digits:
            return 0
        y = digits[-1]
        return y + 2000 if y < 100 else y
    last = max(periods, key=end_year) if periods else last
    return df[df["period"] == last].assign(period=last)


def uhf42_geojson() -> dict:
    return _get("geography/UHF42.geojson")


def geo_lookup() -> pd.DataFrame:
    path = CACHE / "geography/GeoLookup.csv"
    if not path.exists():
        r = requests.get(f"{BASE}/geography/GeoLookup.csv", timeout=120)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
    return pd.read_csv(path)


def cdta_to_cd() -> dict[int, int]:
    """CDTA2020 GeoIDs (e.g. 501 = Bronx CD 1) -> community district codes (201)."""
    import re
    from .geo import cd_code
    g = geo_lookup()
    g = g[g["GeoType"] == "CDTA2020"]
    out = {}
    for gid, name, boro in zip(g["GeoID"], g["Name"], g["Borough"]):
        m = re.search(r"\(CD\s*(\d+)\)", str(name))
        code = cd_code(boro, m.group(1)) if m else None
        if code:
            out[int(gid)] = code
    return out


def by_community_district(measure_id: int) -> pd.DataFrame:
    """Latest value per community district, preferring 2020 boundaries (CDTA2020)."""
    cdta = latest(measure_id, "CDTA2020") if "CDTA2020" in set(measure(measure_id)["geo_type"]) else pd.DataFrame()
    if len(cdta):
        cdta = cdta.assign(cd=cdta["geo_id"].map(cdta_to_cd())).dropna(subset=["cd"])
        return cdta.astype({"cd": int})[["cd", "period", "value"]]
    cd = latest(measure_id, "CD")
    return cd.rename(columns={"geo_id": "cd"})[["cd", "period", "value"]]
