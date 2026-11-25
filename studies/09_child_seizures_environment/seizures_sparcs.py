"""Study 9, part B — Pediatric seizure hospitalizations in NYC (SPARCS).

Uses the same SPARCS de-identified inpatient file as Study 3 (download from
https://health.data.ny.gov, see studies/03_hospital_length_of_stay/README.md):

    python studies/09_child_seizures_environment/seizures_sparcs.py --data Hospital_Inpatient_Discharges.csv

What it measures, for children 0-17 who live in NYC:
  * seizure-related stays: principal diagnosis group whose description
    mentions seizure, epilepsy or convulsion (CCSR "Seizure" / older CCS
    "Epilepsy; convulsions")
  * by borough of residence (from the 3-digit ZIP: 100-102 Manhattan,
    103 Staten Island, 104 Bronx, 112 Brooklyn, 111/113/114/116 Queens;
    110 is excluded because it is mostly Nassau County)
  * share of all pediatric stays, length of stay, severity, payer, race and
    ethnicity, set beside each borough's child blood-lead rate (part A)

Limits, stated in the README: the public file has no patient ID (repeat stays
for one child can't be linked), location is only the 3-digit ZIP (about a
borough), and the diagnosis group can't say what caused a seizure.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SEIZURE = re.compile(r"seizure|epilep|convuls", re.I)
ZIP3_BOROUGH = {"100": "Manhattan", "101": "Manhattan", "102": "Manhattan", "103": "Staten Island",
                "104": "Bronx", "112": "Brooklyn", "111": "Queens", "113": "Queens", "114": "Queens",
                "116": "Queens"}
COLUMNS = {
    "age_group": ["age_group"], "zip3": ["zip_code_3_digits", "zip_code_3_digit"],
    "dx_desc": ["ccsr_diagnosis_description", "ccs_diagnosis_description"],
    "los": ["length_of_stay"], "severity": ["apr_severity_of_illness_description", "apr_severity_of_illness_code"],
    "payer": ["payment_typology_1"], "race": ["race"], "ethnicity": ["ethnicity"],
    "admission_type": ["type_of_admission"], "ed": ["emergency_department_indicator"],
}


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def load(path: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    lookup = {norm(c): c for c in header}
    pick = {}
    for canon, options in COLUMNS.items():
        hit = next((lookup[o] for o in options if o in lookup), None)
        if hit:
            pick[hit] = canon
    missing = {"age_group", "zip3", "dx_desc", "los"} - set(pick.values())
    if missing:
        raise SystemExit(f"Missing SPARCS columns: {sorted(missing)}. Found: {list(header)}")
    df = pd.read_csv(path, usecols=list(pick), dtype=str, low_memory=False).rename(columns=pick)
    for c in set(COLUMNS) - set(df.columns):
        df[c] = "Unknown"
    df["los"] = pd.to_numeric(df["los"].str.replace("+", "", regex=False).str.strip(), errors="coerce")
    return df


def main(data: str, out: str | None = None) -> dict:
    out_dir = Path(out) if out else HERE / "results_sparcs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load(data)
    q = {"rows_in_file": int(len(df))}
    kids = df[df["age_group"].str.strip().str.startswith("0 to 17")].copy()
    kids = kids[~kids["admission_type"].str.contains("newborn", case=False, na=False)]
    q["pediatric_stays_statewide_excl_newborn"] = int(len(kids))
    kids["borough"] = kids["zip3"].str.strip().str[:3].map(ZIP3_BOROUGH)
    nyc = kids.dropna(subset=["borough"]).copy()
    q["pediatric_stays_nyc_residents"] = int(len(nyc))
    nyc["seizure"] = nyc["dx_desc"].fillna("").str.contains(SEIZURE)
    q["seizure_dx_groups_matched"] = sorted(nyc.loc[nyc["seizure"], "dx_desc"].unique().tolist())

    s = nyc[nyc["seizure"]]
    by_boro = nyc.groupby("borough").agg(pediatric_stays=("seizure", "size"), seizure_stays=("seizure", "sum"))
    by_boro["seizure_share_of_pediatric_stays"] = by_boro["seizure_stays"] / by_boro["pediatric_stays"]
    by_boro["seizure_mean_los"] = s.groupby("borough")["los"].mean()
    by_boro["seizure_medicaid_share"] = s.groupby("borough")["payer"].apply(
        lambda p: p.str.contains("medicaid", case=False, na=False).mean())
    by_boro["seizure_via_ed_share"] = s.groupby("borough")["ed"].apply(lambda e: (e.str.upper() == "Y").mean())

    lead_path = HERE / "results.json"
    if lead_path.exists():
        try:
            from src import ehdp
            bl = ehdp.latest(777, "Borough")
            names = ehdp.geo_lookup().query("GeoType == 'Borough'").set_index("GeoID")["Name"]
            by_boro["child_bll5_per_1k"] = bl.assign(borough=bl["geo_id"].map(names)).set_index("borough")["value"]
        except Exception:                      # offline: skip the lead column
            pass

    def share_table(col):
        t = nyc.groupby(col).agg(pediatric_stays=("seizure", "size"), seizure_stays=("seizure", "sum"))
        t["seizure_share"] = t["seizure_stays"] / t["pediatric_stays"]
        return t[t["pediatric_stays"] >= 200].sort_values("seizure_share", ascending=False)

    results = {
        "data_quality": q,
        "nyc_pediatric_seizure_stays": int(len(s)),
        "nyc_seizure_share_of_pediatric_stays": round(float(len(s) / max(len(nyc), 1)), 4),
        "seizure_los": {"mean": round(float(s["los"].mean()), 2), "median": float(s["los"].median())},
        "by_borough": by_boro.round(4).reset_index().to_dict(orient="records"),
        "by_race": share_table("race").round(4).reset_index().to_dict(orient="records"),
        "by_ethnicity": share_table("ethnicity").round(4).reset_index().to_dict(orient="records"),
        "by_payer": share_table("payer").round(4).reset_index().to_dict(orient="records"),
        "severity_mix": s["severity"].value_counts(normalize=True).round(3).to_dict(),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    by_boro.round(4).to_csv(out_dir / "seizure_stays_by_borough.csv")
    print(json.dumps({k: results[k] for k in ("data_quality", "nyc_pediatric_seizure_stays",
                                             "nyc_seizure_share_of_pediatric_stays", "by_borough")}, indent=2, default=float))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.data, a.out)
