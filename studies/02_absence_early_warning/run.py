"""Study 2 — Early warning: which schools' chronic absence will rise next year?

Data: NYC DOE School Quality Reports (dnpx-dfnc), elementary/middle schools,
school years 2015-16 to 2024-25. Note the SQR metric `chronic_absent_ems_all`
is the share with >90% attendance; chronic absence = 1 - value.

Design (no look-ahead):
  * each row = one school in year t; target = chronic absence in year t+1
  * transitions touching 2019-20 (closures from March 2020) or 2020-21
    (remote learning) are excluded; their measurement isn't comparable
  * train: 2015->16, 16->17, 17->18, 21->22   validate: 22->23   test: 23->24 (once)
  * features are known at the end of year t: current rate, last year's change,
    attendance rate, size, school type, borough/district, Economic Need Index
    (2021-22) and students-in-temporary-housing share (2020-21)
  * baselines: "same as this year" and a mean-reversion line

    python studies/02_absence_early_warning/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, roc_auc_score  # noqa: E402

from src import soda  # noqa: E402
from src.config import PROCESSED  # noqa: E402

DISRUPTED = {2019, 2020}
TRAIN_T, VALID_T, TEST_T = [2015, 2016, 2017, 2021], 2022, 2023
RISE = 0.05                      # "meaningful rise" = 5+ percentage points
BLUE, GRAY, INK_2, GRID = "#257de0", "#9c9c9c", "#666666", "#e5e5e5"
FEATURES = ["rate", "change_prev", "attendance", "log_n", "eni_2122", "sth_rate",
            "is_k8", "is_middle", "boro_code", "district"]


def fetch_panel() -> pd.DataFrame:
    df = soda.fetch_all(
        "dnpx-dfnc",
        select="school_year, dbn, metric_variable_name, number_of_students, metric_value",
        where="report_type = 'EMS' AND metric_variable_name in ('chronic_absent_ems_all','attendance_k8_all')",
        order="school_year, dbn, metric_variable_name")
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
    df["number_of_students"] = pd.to_numeric(df["number_of_students"], errors="coerce")
    wide = df.pivot_table(index=["dbn", "school_year"], columns="metric_variable_name",
                          values=["metric_value", "number_of_students"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    out = pd.DataFrame({
        "dbn": wide["dbn"], "year": wide["school_year"].astype(int),
        "rate": 1 - wide["metric_value_chronic_absent_ems_all"],           # invert SQR metric
        "n": wide["number_of_students_chronic_absent_ems_all"],
        "attendance": wide.get("metric_value_attendance_k8_all"),
    })
    out = out.dropna(subset=["rate"])
    # A school-year with under 1% chronic absence is almost certainly a reporting
    # error (53 school-years over ten years, 40 of them charter schools); NYC's
    # lowest-absence schools sit around 3-5%. Excluded rather than modeled.
    return out[out["rate"] >= 0.01]


def build_rows(panel: pd.DataFrame, schools: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["dbn", "year"])
    nxt = p[["dbn", "year", "rate"]].assign(year=lambda d: d["year"] - 1).rename(columns={"rate": "rate_next"})
    prv = p[["dbn", "year", "rate"]].assign(year=lambda d: d["year"] + 1).rename(columns={"rate": "rate_prev"})
    rows = p.merge(nxt, on=["dbn", "year"], how="left").merge(prv, on=["dbn", "year"], how="left")
    # last year's change is unusable when last year was a disrupted year
    rows["change_prev"] = np.where(rows["year"].sub(1).isin(DISRUPTED), np.nan, rows["rate"] - rows["rate_prev"])
    rows["log_n"] = np.log(rows["n"].clip(lower=1))
    static = schools[["dbn", "eni_2122", "sth_students", "sth_total_students", "school_type"]].copy()
    static["sth_rate"] = static["sth_students"] / static["sth_total_students"]
    rows = rows.merge(static[["dbn", "eni_2122", "sth_rate", "school_type"]], on="dbn", how="left")
    rows["is_k8"] = (rows["school_type"] == "K-8").astype(int)
    rows["is_middle"] = (rows["school_type"] == "Middle").astype(int)
    rows["district"] = rows["dbn"].str[:2].astype(int)
    rows["boro_code"] = rows["dbn"].str[2].map({"M": 1, "X": 2, "K": 3, "Q": 4, "R": 5})
    rows["change_next"] = rows["rate_next"] - rows["rate"]
    rows["rise"] = (rows["change_next"] >= RISE).astype(int)
    return rows


def usable(rows: pd.DataFrame, years) -> pd.DataFrame:
    r = rows[rows["year"].isin(years) & rows["rate_next"].notna()]
    return r[~r["year"].isin(DISRUPTED) & ~(r["year"] + 1).isin(DISRUPTED)]


def precision_at(y_true: np.ndarray, score: np.ndarray, frac: float = 0.10) -> float:
    k = max(1, int(round(len(score) * frac)))
    return float(y_true[np.argsort(-score)[:k]].mean())


def main() -> dict:
    schools = pd.read_csv(PROCESSED / "schools.csv")
    rows = build_rows(fetch_panel(), schools)
    train, valid, test = usable(rows, TRAIN_T), usable(rows, [VALID_T]), usable(rows, [TEST_T])

    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                        min_samples_leaf=30, l2_regularization=1.0,
                                        categorical_features=[FEATURES.index("district")],
                                        random_state=0)
    # Tune nothing on test: pick the number of boosting rounds on the validation year
    best_iter, best_mae = None, np.inf
    for it in (100, 200, 300, 500):
        m = gbm.set_params(max_iter=it).fit(train[FEATURES], train["change_next"])
        mae = mean_absolute_error(valid["rate_next"], valid["rate"] + m.predict(valid[FEATURES]))
        if mae < best_mae:
            best_iter, best_mae = it, mae
    fit_rows = pd.concat([train, valid])
    gbm.set_params(max_iter=best_iter).fit(fit_rows[FEATURES], fit_rows["change_next"])
    lin = LinearRegression().fit(fit_rows[["rate"]], fit_rows["rate_next"])

    y = test["rate_next"].to_numpy()
    pred = {
        "persistence (next year = this year)": test["rate"].to_numpy(),
        "mean reversion (linear on this year)": lin.predict(test[["rate"]]),
        "gradient boosting (all features)": test["rate"].to_numpy() + gbm.predict(test[FEATURES]),
    }
    rise = test["rise"].to_numpy()
    year_shift = float(fit_rows["change_next"].mean())
    scores = {
        "persistence (next year = this year)": np.zeros(len(test)) + 1e-9 * np.arange(len(test)),
        "mean reversion (linear on this year)": pred["mean reversion (linear on this year)"] - test["rate"].to_numpy(),
        "gradient boosting (all features)": gbm.predict(test[FEATURES]),
    }
    metrics = {name: {
        "mae_pts": round(mean_absolute_error(y, p) * 100, 2),
        "rise_auc": round(roc_auc_score(rise, scores[name]), 3) if "persistence" not in name else 0.5,
        "precision_top10pct_rise": round(precision_at(rise, scores[name]), 3) if "persistence" not in name else None,
    } for name, p in pred.items()}

    # Watch list for 2025-26: refit on every usable transition, score 2024-25
    all_rows = usable(rows, list(range(2015, 2024)))
    gbm_all = HistGradientBoostingRegressor(**gbm.get_params()).fit(all_rows[FEATURES], all_rows["change_next"])
    now = rows[rows["year"] == 2024].copy()
    now["predicted_change"] = gbm_all.predict(now[FEATURES])
    now["predicted_rate_2025_26"] = now["rate"] + now["predicted_change"]
    watch = (now.merge(schools[["dbn", "school_name", "cd"]], on="dbn", how="left")
                .sort_values("predicted_change", ascending=False))
    watch[["dbn", "school_name", "cd", "rate", "change_prev", "predicted_change", "predicted_rate_2025_26",
           "eni_2122", "sth_rate"]].round(4).to_csv(HERE / "watch_list_2025_26.csv", index=False)

    # Permutation-free importance proxy: drop-column MAE on the validation year
    imp = {}
    base_mae = mean_absolute_error(valid["rate_next"], valid["rate"] + HistGradientBoostingRegressor(
        **gbm.get_params()).fit(train[FEATURES], train["change_next"]).predict(valid[FEATURES]))
    for f in FEATURES:
        feats = [x for x in FEATURES if x != f]
        params = gbm.get_params()
        params["categorical_features"] = [feats.index("district")] if "district" in feats else None
        m = HistGradientBoostingRegressor(**params).fit(train[feats], train["change_next"])
        imp[f] = round((mean_absolute_error(valid["rate_next"], valid["rate"] + m.predict(valid[feats])) - base_mae) * 100, 3)

    results = {
        "rows": {"train": int(len(train)), "valid": int(len(valid)), "test": int(len(test))},
        "test_transition": "2023-24 -> 2024-25",
        "test_share_rising_5pts": round(float(rise.mean()), 3),
        "test_mean_change_pts": round(float(test["change_next"].mean() * 100), 2),
        "boosting_rounds_chosen_on_validation": best_iter,
        "metrics": metrics,
        "drop_column_mae_increase_pts": dict(sorted(imp.items(), key=lambda kv: -kv[1])),
        "watch_list_schools": int(len(watch)),
    "excluded_implausible_school_years": "rate < 1% (53 school-years)",
        "train_mean_change_pts": round(year_shift * 100, 2),
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2))

    # Figure: citywide trend + calibration by predicted-change decile on test
    trend = rows.groupby("year").apply(lambda g: np.average(g["rate"], weights=g["n"].fillna(1)),
                                       include_groups=False)
    t = test.assign(score=scores["gradient boosting (all features)"])
    t["decile"] = pd.qcut(t["score"].rank(method="first"), 10, labels=False) + 1
    cal = t.groupby("decile")["rise"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    ax.plot(trend.index, trend.values * 100, color=BLUE, linewidth=2, marker="o", markersize=5)
    ax.axvspan(2018.5, 2020.5, color=GRAY, alpha=0.15, linewidth=0)
    ax.text(2019.5, trend.max() * 100 - 0.5, "closures /\nremote", ha="center", va="top", fontsize=8, color=INK_2)
    ax.set_title("K-8 chronic absence, NYC public schools", loc="left", fontweight="bold", fontsize=11)
    ax.set_xticks(trend.index, [f"{y}-{(y + 1) % 100:02d}" for y in trend.index], rotation=45, fontsize=8)
    ax.set_ylabel("Chronically absent (%)")
    ax = axes[1]
    ax.bar(cal.index, cal.values * 100, color=BLUE, width=0.65)
    ax.axhline(rise.mean() * 100, color=INK_2, linestyle="--", linewidth=1.2)
    ax.text(0.6, rise.mean() * 100 + 0.4, f"all schools: {rise.mean():.1%}", ha="left", fontsize=8, color=INK_2)
    ax.set_title("Test year: share of schools that rose 5+ points,\nby model risk decile", loc="left",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("Predicted-risk decile (10 = highest)"); ax.set_xticks(range(1, 11))
    ax.set_ylabel("Rose 5+ points (%)")
    for a in axes:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.grid(axis="y", color=GRID); a.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(HERE / "early_warning.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
