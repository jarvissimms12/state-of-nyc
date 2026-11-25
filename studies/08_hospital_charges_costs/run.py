"""Study 8 — What NYC hospitals charge vs what care costs them.

Data: NYS SPARCS de-identified inpatient discharges (Health Data NY), the
same file as Study 3. Every row has Total Charges (the hospital's list price,
what it bills) and Total Costs (SPARCS's estimate of what the stay cost the
hospital to provide, from charges and each hospital's cost reports).

    python studies/08_hospital_charges_costs/run.py --data Hospital_Inpatient_Discharges.csv

For hospitals in the New York City health service area (newborns excluded):
  1. Markup: total charges / total costs, per hospital.
  2. Case-mix-adjusted cost index: a hospital's costs / what the same mix of
     APR-DRG x severity would cost at the citywide average. 1.0 = average;
     1.2 = 20% more expensive for the same kinds of patients. 99.8% funnel
     limits, as in Study 3, so a hospital is flagged only when the gap isn't noise.
     The benchmark is built in two passes so outlier hospitals don't distort it.
  3. The same adjusted index for charges, and markup by payer.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, RED, GRAY, INK, INK_2, GRID = "#257de0", "#ec4647", "#9c9c9c", "#191919", "#666666", "#e5e5e5"
COLUMNS = {
    "service_area": ["hospital_service_area", "health_service_area"],
    "facility_id": ["permanent_facility_id", "facility_id"], "facility": ["facility_name"],
    "admission_type": ["type_of_admission"], "drg": ["apr_drg_code"], "drg_desc": ["apr_drg_description"],
    "severity": ["apr_severity_of_illness_code"], "payer": ["payment_typology_1"],
    "los": ["length_of_stay"], "charges": ["total_charges"], "costs": ["total_costs"],
}
Z = 3.09   # 99.8% limits


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def money(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")


def load(path: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    lookup = {norm(c): c for c in header}
    pick = {}
    for canon, options in COLUMNS.items():
        hit = next((lookup[o] for o in options if o in lookup), None)
        if hit:
            pick[hit] = canon
    missing = {"service_area", "facility_id", "drg", "severity", "charges", "costs"} - set(pick.values())
    if missing:
        raise SystemExit(f"Missing SPARCS columns: {sorted(missing)}. Found: {list(header)}")
    df = pd.read_csv(path, usecols=list(pick), dtype=str, low_memory=False).rename(columns=pick)
    for c in set(COLUMNS) - set(df.columns):
        df[c] = "Unknown"
    df["charges"], df["costs"] = money(df["charges"]), money(df["costs"])
    return df


def index_table(df: pd.DataFrame, value: str, expected: str, by: str, min_n: int = 200) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by):
        if len(g) < min_n:
            continue
        o, e = g[value].sum(), g[expected].sum()
        se = (g[value] - g[expected]).std(ddof=1) * np.sqrt(len(g)) / e
        r = o / e
        rows.append({by: key, "stays": len(g), "index": r, "lo": r - Z * se, "hi": r + Z * se})
    t = pd.DataFrame(rows)
    if len(t):
        t["flag"] = np.where(t["lo"] > 1, "above", np.where(t["hi"] < 1, "below", "within"))
    return t


def main(data: str, out: str | None = None) -> dict:
    out_dir = Path(out) if out else HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load(data)
    q = {"rows_in_file": int(len(df))}
    df = df[df["service_area"].str.strip().str.lower() == "new york city"]
    df = df[~df["admission_type"].str.contains("newborn", case=False, na=False)]
    df = df[(df["charges"] > 0) & (df["costs"] > 0)].copy()
    q["nyc_stays_with_charges_and_costs"] = int(len(df))
    df["cell"] = df["drg"].astype(str) + "|" + df["severity"].astype(str)
    cell_n = df.groupby("cell")["costs"].transform("size")
    df = df[cell_n >= 20].copy()
    q["stays_in_cells_with_20plus"] = int(len(df))
    # Two-pass benchmark: outlier hospitals with many stays drag the citywide
    # average toward themselves and make everyone else look cheap or dear.
    # Pass 1 flags them; pass 2 rebuilds each DRG x severity average from the
    # remaining hospitals and scores everyone against that.
    for value, exp in (("costs", "expected_cost"), ("charges", "expected_charge")):
        df[exp] = df.groupby("cell")[value].transform("mean")
        first = index_table(df, value, exp, "facility_id")
        outliers = set(first.loc[first["flag"] != "within", "facility_id"]) if len(first) else set()
        clean = df[~df["facility_id"].isin(outliers)].groupby("cell")[value].mean()
        df[exp] = df["cell"].map(clean).fillna(df[exp])
        q[f"{value}_pass1_outliers"] = len(outliers)

    names = df.groupby("facility_id")["facility"].agg(lambda s: s.mode().iat[0])
    hosp = df.groupby("facility_id").agg(stays=("costs", "size"), charges=("charges", "sum"), costs=("costs", "sum"))
    hosp = hosp[hosp["stays"] >= 200]
    hosp["markup"] = hosp["charges"] / hosp["costs"]
    cost_ix = index_table(df, "costs", "expected_cost", "facility_id").set_index("facility_id")
    charge_ix = index_table(df, "charges", "expected_charge", "facility_id").set_index("facility_id")
    hosp = hosp.join(cost_ix.add_prefix("cost_")).join(charge_ix[["index", "flag"]].add_prefix("charge_"))
    hosp["facility"] = names
    hosp = hosp.sort_values("markup", ascending=False)
    hosp.round(4).to_csv(out_dir / "hospital_charges_costs.csv")

    payer = df.groupby("payer").agg(stays=("costs", "size"), charges=("charges", "sum"), costs=("costs", "sum"))
    payer = payer[payer["stays"] >= 500]
    payer["markup"] = payer["charges"] / payer["costs"]
    payer_cost = index_table(df, "costs", "expected_cost", "payer", 500).set_index("payer")["index"]
    payer["cost_index"] = payer_cost

    results = {
        "data_quality": q,
        "hospitals": int(len(hosp)),
        "citywide_markup": round(float(df["charges"].sum() / df["costs"].sum()), 3),
        "markup_range": [round(float(hosp["markup"].min()), 2), round(float(hosp["markup"].max()), 2)],
        "markup_quartiles": hosp["markup"].quantile([0.25, 0.5, 0.75]).round(2).tolist(),
        "cost_index_flags": hosp["cost_flag"].value_counts().to_dict(),
        "highest_markup": hosp.head(8)[["facility", "stays", "markup", "cost_index", "charge_index"]].round(3).reset_index().to_dict(orient="records"),
        "lowest_markup": hosp.tail(5)[["facility", "stays", "markup", "cost_index", "charge_index"]].round(3).reset_index().to_dict(orient="records"),
        "costliest_for_case_mix": hosp.sort_values("cost_index", ascending=False).head(8)[
            ["facility", "stays", "cost_index", "cost_lo", "cost_hi", "cost_flag", "markup"]].round(3).reset_index().to_dict(orient="records"),
        "by_payer": payer.round(3).reset_index().to_dict(orient="records"),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    color = hosp["cost_flag"].map({"above": RED, "below": BLUE, "within": GRAY})
    ax.scatter(hosp["cost_index"], hosp["markup"], s=np.clip(hosp["stays"] / 150, 15, 200), c=color,
               edgecolor="white", linewidth=1.2, alpha=0.9, zorder=3)
    notable = hosp[(hosp["markup"] > 1.25 * results["citywide_markup"]) | (hosp["cost_flag"] != "within")]
    for _, r in notable.head(8).iterrows():
        ax.annotate(str(r["facility"])[:30], (r["cost_index"], r["markup"]), xytext=(5, 3),
                    textcoords="offset points", fontsize=7.5, color=INK)
    ax.axvline(1, color=INK_2, linewidth=1)
    ax.axhline(results["citywide_markup"], color=INK_2, linewidth=1, linestyle="--")
    ax.set_xlabel("Cost index: costs vs. citywide average for the same case mix (1.0 = average)")
    ax.set_ylabel("Markup: total charges ÷ total costs")
    ax.set_title("NYC hospitals: what care costs vs. what they charge", loc="left", fontweight="bold")
    for lbl, c in (("Costlier than case mix predicts (99.8%)", RED), ("Cheaper than predicted", BLUE), ("Within range", GRAY)):
        ax.scatter([], [], color=c, label=lbl, s=40)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(color=GRID); ax.set_axisbelow(True)
    fig.savefig(out_dir / "charges_vs_costs.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(json.dumps({k: results[k] for k in ("data_quality", "hospitals", "citywide_markup", "markup_range",
                                             "cost_index_flags")}, indent=2, default=float))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.data, a.out)
