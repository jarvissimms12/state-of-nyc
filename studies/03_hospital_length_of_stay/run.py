"""Study 3 — Hospital length of stay in NYC: prediction, benchmarking, equity.

Data: NYS SPARCS "Hospital Inpatient Discharges (SPARCS De-Identified)",
public use file on Health Data NY (https://health.data.ny.gov). One row per
inpatient stay, ~2.1M rows a year statewide. Download the CSV for the latest
year and pass its path:

    python studies/03_hospital_length_of_stay/run.py --data Hospital_Inpatient_Discharges_2023.csv

Scope: hospitals in the "New York City" health service area; newborn
admissions excluded (their stays are driven by birth, not illness).

Three questions:
  1. Prediction. How well can length of stay (LOS) be predicted from the
     patient's diagnosis group, severity, admission type and demographics?
     Baseline = median LOS for the same APR-DRG x severity. Model = gradient boosting. Also a classifier for
     "extended stay" (LOS above the 75th percentile for the DRG x severity).
  2. Benchmarking. Observed / expected LOS for every NYC hospital, where
     expected = the DRG x severity benchmark, with 95% confidence intervals,
     so a hospital is flagged only when the difference isn't just noise.
  3. Equity. Risk-adjusted O/E LOS by race, ethnicity and payer, and whether
     the model's errors are similar across groups.

Note on timing: APR-DRG and severity are assigned at discharge, so this model
describes stays rather than predicting them at the door; an admission-time
version would use the admitting diagnosis instead.
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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

SEED = 42
BLUE, RED, INK, INK_2, GRID = "#257de0", "#ec4647", "#191919", "#666666", "#e5e5e5"

# canonical name -> accepted normalized column names across SPARCS years
COLUMNS = {
    "service_area": ["hospital_service_area", "health_service_area"],
    "facility_id": ["permanent_facility_id", "facility_id"],
    "facility": ["facility_name"],
    "age_group": ["age_group"],
    "gender": ["gender"],
    "race": ["race"],
    "ethnicity": ["ethnicity"],
    "los": ["length_of_stay"],
    "admission_type": ["type_of_admission"],
    "drg": ["apr_drg_code"],
    "drg_desc": ["apr_drg_description"],
    "severity": ["apr_severity_of_illness_code"],
    "mortality_risk": ["apr_risk_of_mortality"],
    "med_surg": ["apr_medical_surgical_description"],
    "payer": ["payment_typology_1"],
    "ed": ["emergency_department_indicator"],
    "dx_group": ["ccsr_diagnosis_code", "ccs_diagnosis_code", "ccsr_diagnosis_description"],
}
FEATURES = ["age_group", "gender", "race", "ethnicity", "admission_type", "drg", "severity",
            "mortality_risk", "med_surg", "payer", "ed", "dx_group", "facility_id"]


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def load(path: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    lookup = {norm(c): c for c in header}
    pick, missing = {}, []
    for canon, options in COLUMNS.items():
        hit = next((lookup[o] for o in options if o in lookup), None)
        if hit is None:
            missing.append(canon)
        else:
            pick[hit] = canon
    required = {"service_area", "los", "drg", "severity", "admission_type", "facility_id"}
    if required & set(missing):
        raise SystemExit(f"Missing required SPARCS columns: {sorted(required & set(missing))}. "
                         f"Found: {list(header)}")
    df = pd.read_csv(path, usecols=list(pick), dtype=str, low_memory=False).rename(columns=pick)
    for c in set(COLUMNS) - set(df.columns):
        df[c] = "Unknown"
    df["los"] = pd.to_numeric(df["los"].str.replace("+", "", regex=False).str.strip(), errors="coerce")
    return df


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    q = {"rows_in_file": int(len(df))}
    df = df[df["service_area"].str.strip().str.lower() == "new york city"]
    q["rows_nyc"] = int(len(df))
    df = df[~df["admission_type"].str.contains("newborn", case=False, na=False)]
    q["rows_nyc_excl_newborn"] = int(len(df))
    df = df.dropna(subset=["los", "drg", "severity"]).copy()
    df = df[df["los"] >= 1]
    q["rows_modeled"] = int(len(df))
    q["los_capped_at_120"] = int((df["los"] >= 120).sum())
    for c in FEATURES:
        df[c] = df[c].fillna("Unknown").astype(str)
    df["cell"] = df["drg"] + "|" + df["severity"]
    return df, q


def benchmark(train: pd.DataFrame, min_n: int = 20) -> dict:
    """Per DRG x severity cell, from training data only:
       mean   -> expected LOS for observed/expected ratios. O/E must use the
                 mean: LOS is right-skewed, so a median makes everyone look long.
       median -> the point benchmark for prediction error (minimizes MAE)
       p75    -> threshold for an "extended stay" """
    g = train.groupby("cell")["los"]
    ok = g.size() >= min_n
    return {"mean": g.mean()[ok], "median": g.median()[ok], "p75": g.quantile(0.75)[ok],
            "overall": {"mean": float(train["los"].mean()), "median": float(train["los"].median()),
                        "p75": float(train["los"].quantile(0.75))}}


def encode(df: pd.DataFrame, cats: dict[str, list]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in FEATURES:
        out[c] = pd.Categorical(df[c], categories=cats[c]).codes.astype(float)
        out.loc[out[c] < 0, c] = np.nan
    return out


def oe_table(df: pd.DataFrame, by: str, min_n: int = 200, z: float = 1.96) -> pd.DataFrame:
    """Observed/expected LOS with a confidence interval from per-stay O-E variation.
    Hospitals use z = 3.09 (99.8% funnel limits, the usual choice when dozens of
    providers are compared at once); a handful of equity groups use 95%."""
    rows = []
    for key, g in df.groupby(by):
        if len(g) < min_n:
            continue
        o, e = g["los"].sum(), g["expected"].sum()
        diff_sd = (g["los"] - g["expected"]).std(ddof=1)
        se = diff_sd * np.sqrt(len(g)) / e
        ratio = o / e
        rows.append({by: key, "stays": len(g), "observed_mean": g["los"].mean(),
                     "expected_mean": g["expected"].mean(), "oe_ratio": ratio,
                     "ci_low": ratio - z * se, "ci_high": ratio + z * se})
    t = pd.DataFrame(rows)
    if len(t):
        t["flag"] = np.where(t["ci_low"] > 1, "longer than expected",
                             np.where(t["ci_high"] < 1, "shorter than expected", "as expected"))
    return t


def main(data: str, out: str | None = None) -> dict:
    out_dir = Path(out) if out else HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df, quality = prepare(load(data))

    train, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["severity"])
    bm = benchmark(train)
    for part in (train, test):
        part["expected"] = part["cell"].map(bm["mean"]).fillna(bm["overall"]["mean"])
        part["benchmark_median"] = part["cell"].map(bm["median"]).fillna(bm["overall"]["median"])
        part["long_cut"] = part["cell"].map(bm["p75"]).fillna(bm["overall"]["p75"])
        part["extended"] = (part["los"] > part["long_cut"]).astype(int)

    cats = {c: sorted(train[c].unique()) for c in FEATURES}
    Xtr, Xte = encode(train, cats), encode(test, cats)
    cat_mask = [True] * len(FEATURES)
    # log target: LOS is heavily right-skewed
    reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                        categorical_features=cat_mask, random_state=SEED,
                                        early_stopping=True, validation_fraction=0.1)
    reg.fit(Xtr, np.log1p(train["los"]))
    pred = np.expm1(reg.predict(Xte))
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                         categorical_features=cat_mask, random_state=SEED,
                                         early_stopping=True, validation_fraction=0.1)
    clf.fit(Xtr, train["extended"])
    p_ext = clf.predict_proba(Xte)[:, 1]

    y = test["los"].to_numpy()
    results = {
        "data_quality": quality,
        "test_stays": int(len(test)),
        "los": {"mean": round(float(df["los"].mean()), 2), "median": float(df["los"].median()),
                "p90": float(df["los"].quantile(0.9))},
        "prediction": {
            "mae_days_benchmark_drg_x_severity_median": round(mean_absolute_error(y, test["benchmark_median"]), 3),
            "mae_days_gradient_boosting": round(mean_absolute_error(y, pred), 3),
            "extended_stay_rate": round(float(test["extended"].mean()), 3),
            "extended_stay_auc": round(roc_auc_score(test["extended"], p_ext), 3),
            "extended_stay_pr_auc": round(average_precision_score(test["extended"], p_ext), 3),
        },
    }

    # Benchmarking & equity use all NYC stays against the training benchmark
    full = pd.concat([train, test])
    hosp = oe_table(full, "facility_id", z=3.09)
    names = full.groupby("facility_id")["facility"].agg(lambda s: s.mode().iat[0])
    if len(hosp):
        hosp["facility"] = hosp["facility_id"].map(names)
        hosp = hosp.sort_values("oe_ratio", ascending=False)
        hosp.round(4).to_csv(out_dir / "hospital_oe.csv", index=False)
    results["hospitals"] = {"scored": int(len(hosp)),
                            "longer_than_expected": int((hosp["flag"] == "longer than expected").sum()) if len(hosp) else 0,
                            "shorter_than_expected": int((hosp["flag"] == "shorter than expected").sum()) if len(hosp) else 0}

    test = test.assign(pred=pred, abs_err=np.abs(y - pred))
    equity = {}
    for col in ("race", "ethnicity", "payer"):
        t = oe_table(full, col, min_n=500)
        err = test.groupby(col).agg(stays=("los", "size"), model_mae=("abs_err", "mean"),
                                    mean_pred=("pred", "mean"), mean_obs=("los", "mean"))
        err = err[err["stays"] >= 200]
        equity[col] = {"risk_adjusted_oe": t.round(3).to_dict(orient="records"),
                       "model_error_by_group": err.round(3).reset_index().to_dict(orient="records")}
    results["equity"] = equity
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))

    # Figure: hospital funnel plot (O/E vs volume with 95% limits)
    if len(hosp):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        color = hosp["flag"].map({"longer than expected": RED, "shorter than expected": BLUE,
                                  "as expected": "#9c9c9c"})
        ax.scatter(hosp["stays"], hosp["oe_ratio"], c=color, s=40, edgecolor="white", linewidth=1.2, zorder=3)
        ax.axhline(1, color=INK_2, linewidth=1)
        wide = hosp["stays"].max() / hosp["stays"].min() >= 10
        if wide:
            ax.set_xscale("log")
        ax.set_xlabel("Inpatient stays" + (" (log scale)" if wide else ""))
        ax.set_ylabel("Observed / expected length of stay")
        ax.set_title("NYC hospitals: length of stay vs. the DRG × severity benchmark",
                     loc="left", fontweight="bold")
        for lbl, c in (("Longer than expected (99.8% limit above 1)", RED),
                       ("Shorter than expected (99.8% limit below 1)", BLUE), ("Within expected range", "#9c9c9c")):
            ax.scatter([], [], color=c, label=lbl, s=40)
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        if wide:
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(color=GRID); ax.set_axisbelow(True)
        fig.savefig(out_dir / "hospital_funnel.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    print(json.dumps({k: results[k] for k in ("data_quality", "los", "prediction", "hospitals")}, indent=2))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="SPARCS de-identified inpatient CSV from Health Data NY")
    ap.add_argument("--out", default=None, help="output folder (default: studies/03_hospital_length_of_stay/results)")
    a = ap.parse_args()
    main(a.data, a.out)
