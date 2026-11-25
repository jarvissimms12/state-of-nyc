"""Study 7 — Children's asthma and the homes they live in.

Data, all by community district (59), calendar year 2023 unless noted:
  * Asthma emergency-department visits per 10,000 children, ages 5-17 and
    0-4; asthma hospitalizations ages 5-17 (NYC Health Department,
    Environment & Health Data Portal, via github.com/nychealth/EHDP-data)
  * HPD "asthma trigger" violations issued in 2023: mold, roaches, mice and
    rats (order numbers below), per 1,000 residential units (PLUTO)
  * Fine particulate air pollution (PM2.5, annual mean 2023; NYC Open Data
    c3uy-2p5r, NYC Community Air Survey)
  * Neighborhood poverty and very old housing stock (ACS 2019-23, EHDP)
  * K-8 chronic absence 2024-25 (main build): asthma is a leading cause of
    missed school days

    python studies/07_asthma_and_housing/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402

from src import ehdp, soda  # noqa: E402
from src.config import PROCESSED  # noqa: E402
from src.geo import cd_code, normalize_cd  # noqa: E402

YEAR = "2023"
# HPD order numbers for indoor asthma triggers (checked against novdescription text)
TRIGGERS = {"mold": ["550", "553", "853"], "roaches": ["566", "568", "868"],
            "mice_rats": ["567", "569", "867", "869"]}
MEASURES = {"asthma_ed_5_17": 1196, "asthma_ed_0_4": 460, "asthma_hosp_5_17": 1201,
            "poverty_pct": 221, "very_old_homes_pct": 27}
BLUE, INK, INK_2, GRID = "#257de0", "#191919", "#666666", "#e5e5e5"


def trigger_violations() -> pd.DataFrame:
    codes = ", ".join(f"'{c}'" for cs in TRIGGERS.values() for c in cs)
    df = soda.query("wvxf-dwi5", f"""
        SELECT boroid, communityboard, ordernumber, count(*) AS n
        WHERE ordernumber in ({codes})
          AND novissueddate between '{YEAR}-01-01' and '{YEAR}-12-31T23:59:59'
        GROUP BY boroid, communityboard, ordernumber LIMIT 50000""")
    df["n"] = df["n"].astype(int)
    df["cd"] = [cd_code(b, c) for b, c in zip(df["boroid"], df["communityboard"])]
    kind = {c: k for k, cs in TRIGGERS.items() for c in cs}
    df["kind"] = df["ordernumber"].map(kind)
    wide = df.dropna(subset=["cd"]).pivot_table(index="cd", columns="kind", values="n", aggfunc="sum", fill_value=0)
    wide["trigger_violations"] = wide.sum(axis=1)
    return wide.reset_index().astype({"cd": int})


def pm25() -> pd.DataFrame:
    df = soda.query("c3uy-2p5r", f"""
        SELECT geo_join_id, data_value WHERE name = 'Fine particles (PM 2.5)'
          AND geo_type_name = 'CD' AND measure = 'Annual mean' AND time_period = '{YEAR}' LIMIT 100""")
    df["cd"] = df["geo_join_id"].map(normalize_cd)
    df["pm25"] = pd.to_numeric(df["data_value"], errors="coerce")
    return df.dropna(subset=["cd"]).astype({"cd": int})[["cd", "pm25"]]


def main() -> dict:
    cd = pd.read_csv(PROCESSED / "cd_profile.csv")
    d = cd[["cd", "district", "cd_name", "borough", "units", "housing_distress_index",
            "heat_hw_per_1k", "class_c_per_1k", "chronic_absence_ems", "absence_gap"]].copy()
    periods = {}
    for col, mid in MEASURES.items():
        m = ehdp.by_community_district(mid)
        periods[col] = str(m["period"].iloc[0])
        d = d.merge(m[["cd", "value"]].rename(columns={"value": col}), on="cd", how="left")
    d = d.merge(trigger_violations(), on="cd", how="left").merge(pm25(), on="cd", how="left")
    for c in ("mold", "roaches", "mice_rats", "trigger_violations"):
        d[c] = d[c].fillna(0)
        d[f"{c}_per_1k"] = d[c] / d["units"] * 1000

    rho = lambda a, b: [round(float(x), 3) for x in stats.spearmanr(d[a], d[b], nan_policy="omit")]
    z = lambda s: (s - s.mean()) / s.std()
    m = d.dropna(subset=["asthma_ed_5_17", "poverty_pct", "pm25"]).copy()
    m["log_asthma"] = np.log(m["asthma_ed_5_17"])
    for c in ("trigger_violations_per_1k", "poverty_pct", "pm25", "very_old_homes_pct"):
        m[f"z_{c}"] = z(np.log1p(m[c]) if c == "trigger_violations_per_1k" else m[c])
    full = smf.ols("log_asthma ~ z_trigger_violations_per_1k + z_poverty_pct + z_pm25 + z_very_old_homes_pct",
                   m).fit(cov_type="HC3")
    pov_only = smf.ols("log_asthma ~ z_poverty_pct", m).fit(cov_type="HC3")
    eff = lambda t: {"pct_per_sd": round(float(np.expm1(full.params[t]) * 100), 1),
                     "ci95": [round(float(np.expm1(x) * 100), 1) for x in full.conf_int().loc[t]],
                     "p": round(float(full.pvalues[t]), 4)}

    # absence link: K-8 chronic absence vs child asthma ED, with and without poverty
    a = m.dropna(subset=["chronic_absence_ems"]).copy()
    a["z_log_asthma"] = z(a["log_asthma"])
    abs_model = smf.ols("chronic_absence_ems ~ z_log_asthma + z_poverty_pct", a).fit(cov_type="HC3")

    top = d.sort_values("asthma_ed_5_17", ascending=False)
    citywide = ehdp.latest(1196, "Citywide")["value"].iloc[0]
    results = {
        "periods": periods | {"trigger_violations": YEAR, "pm25": YEAR},
        "citywide_asthma_ed_5_17_per_10k": float(citywide),
        "range_asthma_ed_5_17": [float(d["asthma_ed_5_17"].min()), float(d["asthma_ed_5_17"].max())],
        "ratio_highest_to_lowest": round(float(d["asthma_ed_5_17"].max() / d["asthma_ed_5_17"].min()), 1),
        "trigger_violations_total": int(d["trigger_violations"].sum()),
        "trigger_breakdown": {k: int(d[k].sum()) for k in ("mold", "roaches", "mice_rats")},
        "spearman_with_child_asthma_ed": {
            "trigger_violations_per_1k": rho("trigger_violations_per_1k", "asthma_ed_5_17"),
            "mold_per_1k": rho("mold_per_1k", "asthma_ed_5_17"),
            "roaches_per_1k": rho("roaches_per_1k", "asthma_ed_5_17"),
            "mice_rats_per_1k": rho("mice_rats_per_1k", "asthma_ed_5_17"),
            "housing_distress_index": rho("housing_distress_index", "asthma_ed_5_17"),
            "poverty_pct": rho("poverty_pct", "asthma_ed_5_17"),
            "pm25": rho("pm25", "asthma_ed_5_17"),
            "very_old_homes_pct": rho("very_old_homes_pct", "asthma_ed_5_17"),
            "chronic_absence_ems": rho("chronic_absence_ems", "asthma_ed_5_17"),
        },
        "model_log_child_asthma_ed": {
            "n": int(len(m)), "r2": round(float(full.rsquared), 3),
            "r2_poverty_only": round(float(pov_only.rsquared), 3),
            "trigger_violations": eff("z_trigger_violations_per_1k"), "poverty": eff("z_poverty_pct"),
            "pm25": eff("z_pm25"), "very_old_homes": eff("z_very_old_homes_pct")},
        "absence_model": {"asthma_coef_pts_per_sd": round(float(abs_model.params["z_log_asthma"] * 100), 2),
                          "ci95_pts": [round(float(x * 100), 2) for x in abs_model.conf_int().loc["z_log_asthma"]],
                          "p": round(float(abs_model.pvalues["z_log_asthma"]), 4),
                          "poverty_coef_pts_per_sd": round(float(abs_model.params["z_poverty_pct"] * 100), 2)},
        "highest_child_asthma_ed": top.head(8)[["district", "cd_name", "asthma_ed_5_17", "trigger_violations_per_1k",
                                                "poverty_pct", "pm25"]].round(2).to_dict(orient="records"),
        "lowest_child_asthma_ed": top.tail(5)[["district", "cd_name", "asthma_ed_5_17", "trigger_violations_per_1k",
                                               "poverty_pct", "pm25"]].round(2).to_dict(orient="records"),
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=float))
    d.round(4).to_csv(HERE / "district_asthma_housing.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, col, label in ((axes[0], "trigger_violations_per_1k", "HPD mold, roach and rodent violations per 1,000 homes (2023, log scale)"),
                           (axes[1], "pm25", "Fine particle pollution, PM2.5 annual mean (µg/m³, 2023)")):
        ax.scatter(d[col], d["asthma_ed_5_17"], s=40, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
        if col == "trigger_violations_per_1k":
            ax.set_xscale("log")
            ax.set_xticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        for _, r in top.head(3).iterrows():
            ax.annotate(r["district"], (r[col], r["asthma_ed_5_17"]), xytext=(5, 2),
                        textcoords="offset points", fontsize=7.5, color=INK)
        r_, p_ = stats.spearmanr(d[col], d["asthma_ed_5_17"], nan_policy="omit")
        ax.set_title(f"ρ = {r_:.2f}", loc="right", fontsize=9, color=INK_2)
        ax.set_xlabel(label, fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(color=GRID); ax.set_axisbelow(True)
    axes[0].set_ylabel("Asthma ER visits per 10,000 children\nages 5-17 (2023)")
    fig.suptitle("Child asthma ER visits by community district: indoor triggers vs outdoor air",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "asthma_vs_triggers_and_air.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(json.dumps({k: v for k, v in results.items() if not k.startswith(("highest", "lowest"))}, indent=2, default=float))
    return results


if __name__ == "__main__":
    main()
