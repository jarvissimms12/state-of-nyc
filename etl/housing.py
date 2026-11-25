"""Housing conditions by community district, FY2025.

Measures (all per 1,000 residential units from PLUTO):
  * heat_hw_per_1k      311 heat / hot-water complaints
  * class_c_per_1k      HPD Class C ("immediately hazardous") violations issued
  * evictions_per_1k    residential evictions executed by City Marshals

All counts are aggregated on the server with SoQL GROUP BY, so the full
row-level datasets (millions of rows) never need to be downloaded.
"""
from __future__ import annotations

import pandas as pd

from src import soda
from src.config import DATASETS, FY_END, FY_START, RAW
from src.geo import cd_code, normalize_cd, parse_311_board

WINDOW = f"between '{FY_START}' and '{FY_END}'"


def residential_units() -> pd.DataFrame:
    df = soda.query(DATASETS["pluto"],
                    "SELECT cd, sum(unitsres) AS units, count(*) AS lots GROUP BY cd LIMIT 500")
    df["cd"] = df["cd"].map(normalize_cd)
    df = df.dropna(subset=["cd"])
    df["units"] = df["units"].astype(float)
    df["lots"] = df["lots"].astype(int)
    return df.groupby("cd", as_index=False)[["units", "lots"]].sum()


def heat_complaints() -> tuple[pd.DataFrame, dict]:
    df = soda.query(DATASETS["311"], f"""
        SELECT community_board, count(*) AS n
        WHERE complaint_type = 'HEAT/HOT WATER' AND created_date {WINDOW}
        GROUP BY community_board LIMIT 500""")
    df["n"] = df["n"].astype(int)
    df["cd"] = df["community_board"].map(parse_311_board)
    quality = {"heat_total": int(df["n"].sum()),
               "heat_unassigned": int(df.loc[df["cd"].isna(), "n"].sum())}
    out = df.dropna(subset=["cd"]).groupby("cd", as_index=False)["n"].sum()
    return out.rename(columns={"n": "heat_hw_complaints"}), quality


def hpd_violations() -> tuple[pd.DataFrame, dict]:
    df = soda.query(DATASETS["hpd_violations"], f"""
        SELECT boroid, communityboard, class, count(*) AS n
        WHERE novissueddate {WINDOW}
        GROUP BY boroid, communityboard, class LIMIT 5000""")
    df["n"] = df["n"].astype(int)
    df["cd"] = [cd_code(b, c) for b, c in zip(df["boroid"], df.get("communityboard"))]
    quality = {"hpd_total": int(df["n"].sum()),
               "hpd_unassigned": int(df.loc[df["cd"].isna(), "n"].sum())}
    wide = (df.dropna(subset=["cd"])
              .pivot_table(index="cd", columns="class", values="n", aggfunc="sum", fill_value=0)
              .reset_index())
    wide.columns.name = None
    wide = wide.rename(columns={"A": "hpd_class_a", "B": "hpd_class_b", "C": "hpd_class_c"})
    keep = ["cd"] + [c for c in ("hpd_class_a", "hpd_class_b", "hpd_class_c") if c in wide]
    return wide[keep], quality


def evictions() -> tuple[pd.DataFrame, dict]:
    df = soda.query(DATASETS["evictions"], f"""
        SELECT borough, community_board, count(*) AS n
        WHERE residential_commercial_ind = 'Residential' AND executed_date {WINDOW}
        GROUP BY borough, community_board LIMIT 500""")
    df["n"] = df["n"].astype(int)
    df["cd"] = [cd_code(b, c) for b, c in zip(df["borough"], df.get("community_board"))]
    quality = {"evictions_total": int(df["n"].sum()),
               "evictions_unassigned": int(df.loc[df["cd"].isna(), "n"].sum())}
    out = df.dropna(subset=["cd"]).groupby("cd", as_index=False)["n"].sum()
    return out.rename(columns={"n": "residential_evictions"}), quality


def build() -> tuple[pd.DataFrame, dict]:
    units = residential_units()
    heat, q1 = heat_complaints()
    hpd, q2 = hpd_violations()
    ev, q3 = evictions()
    df = units
    for part in (heat, hpd, ev):
        df = df.merge(part, on="cd", how="left")
    count_cols = [c for c in df.columns if c not in ("cd", "units", "lots")]
    df[count_cols] = df[count_cols].fillna(0).astype(int)
    per1k = 1000 / df["units"]
    df["heat_hw_per_1k"] = df["heat_hw_complaints"] * per1k
    df["class_c_per_1k"] = df["hpd_class_c"] * per1k
    df["evictions_per_1k"] = df["residential_evictions"] * per1k
    df["cd"] = df["cd"].astype(int)
    df.to_csv(RAW / "housing_cd.csv", index=False)
    return df, {**q1, **q2, **q3}
