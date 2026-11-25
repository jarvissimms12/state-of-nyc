"""Questions this answers, in order:

1. Where are housing conditions worst?            (housing distress index by district)
2. Do schools in those districts have more chronic absence?
3. Is that the housing, or just poverty?           (school-level models with
                                                    Economic Need Index as a control)
4. Where are the city's job openings, and do they reach the districts under
   the most housing strain?

    python analysis/run_analysis.py   ->  docs/results.json + docs/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.patches import Polygon
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import BOROUGHS, DOCS, PROCESSED  # noqa: E402

# Reference palette (dataviz skill): one sequential blue ramp, recessive inks
BLUE_RAMP = ["#cce3fd", "#84b9f3", "#338ced", "#2370cd", "#0f4a96"]
BLUE, INK, INK_2, GRID = "#257de0", "#191919", "#666666", "#e5e5e5"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
})


def load():
    cd = pd.read_csv(PROCESSED / "cd_profile.csv")
    schools = pd.read_csv(PROCESSED / "schools.csv")
    jobs = pd.read_csv(PROCESSED / "job_postings.csv")
    payroll = pd.read_csv(PROCESSED / "payroll_agencies.csv")
    geo = json.loads((PROCESSED / "community_districts.geojson").read_text())
    quality = json.loads((PROCESSED / "data_quality.json").read_text())
    return cd, schools, jobs, payroll, geo, quality


def ci(model, term):
    lo, hi = model.conf_int().loc[term]
    return [round(float(lo), 4), round(float(hi), 4)]


def school_models(schools: pd.DataFrame, cd: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Elementary/middle chronic absence, weighted by students measured,
    standard errors clustered by community district."""
    d = schools.merge(cd[["cd", "housing_distress_index"]], on="cd")
    d = d.dropna(subset=["chronic_absence_ems", "eni_2122"])
    d = d[d["n_chronic_ems"] > 0].copy()
    d["eni_10"] = d["eni_2122"] * 10                 # per 10 points of ENI
    d["sth_rate"] = d["sth_students"] / d["sth_total_students"]
    d["sth_10"] = d["sth_rate"] * 10                 # per 10 points of STH share

    def fit(formula, data):
        return smf.wls(formula, data, weights=data["n_chronic_ems"]).fit(
            cov_type="cluster", cov_kwds={"groups": data["cd"]})

    m_h = fit("chronic_absence_ems ~ housing_distress_index", d)
    m_he = fit("chronic_absence_ems ~ housing_distress_index + eni_10", d)
    s = d.dropna(subset=["sth_rate"])
    m_e_s = fit("chronic_absence_ems ~ eni_10", s)
    m_es = fit("chronic_absence_ems ~ eni_10 + sth_10", s)
    m_esh = fit("chronic_absence_ems ~ eni_10 + sth_10 + housing_distress_index", s)

    out = {
        "n_schools": int(len(d)), "n_districts": int(d["cd"].nunique()),
        "housing_only": {"coef_per_sd": round(m_h.params["housing_distress_index"], 4),
                         "ci95": ci(m_h, "housing_distress_index"),
                         "p": round(m_h.pvalues["housing_distress_index"], 4), "r2": round(m_h.rsquared, 3)},
        "housing_plus_eni": {"housing_coef_per_sd": round(m_he.params["housing_distress_index"], 4),
                             "housing_ci95": ci(m_he, "housing_distress_index"),
                             "housing_p": round(m_he.pvalues["housing_distress_index"], 4),
                             "eni_coef_per_10pts": round(m_he.params["eni_10"], 4),
                             "r2": round(m_he.rsquared, 3)},
        "temporary_housing": {"n_schools": int(len(s)),
                              "r2_eni_only": round(m_e_s.rsquared, 3),
                              "sth_coef_per_10pts": round(m_es.params["sth_10"], 4),
                              "sth_ci95": ci(m_es, "sth_10"), "sth_p": round(m_es.pvalues["sth_10"], 4),
                              "eni_coef_per_10pts": round(m_es.params["eni_10"], 4),
                              "r2": round(m_es.rsquared, 3),
                              "housing_coef_when_added": round(m_esh.params["housing_distress_index"], 4),
                              "housing_p_when_added": round(m_esh.pvalues["housing_distress_index"], 4)},
    }
    return out, d


def district_findings(cd: pd.DataFrame) -> dict:
    rho = lambda a, b: round(float(stats.spearmanr(cd[a], cd[b], nan_policy="omit")[0]), 2)
    # Districts with more chronic absence than their schools' economic need predicts
    fit = smf.ols("chronic_absence_ems ~ eni_2122", cd).fit()
    cd = cd.assign(expected_absence=fit.fittedvalues, absence_gap=cd["chronic_absence_ems"] - fit.fittedvalues)
    cols = ["district", "cd_name", "chronic_absence_ems", "expected_absence", "absence_gap", "housing_distress_index"]
    r = lambda x: x[cols].round(3).to_dict(orient="records")
    top = cd.sort_values("housing_distress_index", ascending=False)
    return {
        "spearman": {
            "housing_vs_chronic_absence_ems": rho("housing_distress_index", "chronic_absence_ems"),
            "housing_vs_eni": rho("housing_distress_index", "eni_2122"),
            "housing_vs_sth_2021": rho("housing_distress_index", "sth_rate_2021"),
            "housing_vs_grad_rate": rho("housing_distress_index", "grad_rate_4yr"),
            "eni_vs_chronic_absence_ems": rho("eni_2122", "chronic_absence_ems"),
        },
        "top10_housing_distress": top.head(10)[["district", "cd_name", "housing_distress_index",
                                                 "heat_hw_per_1k", "class_c_per_1k", "evictions_per_1k",
                                                 "chronic_absence_ems"]].round(3).to_dict(orient="records"),
        "top10_share_of_class_c": round(float(top.head(10)["hpd_class_c"].sum() / cd["hpd_class_c"].sum()), 3),
        "top10_share_of_units": round(float(top.head(10)["units"].sum() / cd["units"].sum()), 3),
        "absence_above_expected": r(cd.sort_values("absence_gap", ascending=False).head(8)),
        "absence_below_expected": r(cd.sort_values("absence_gap").head(5)),
    }, cd


def borough_table(cd: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    w = lambda g, c, wt: np.average(g[c], weights=g[wt])
    b = cd.groupby("borough").apply(lambda g: pd.Series({
        "residential_units": g["units"].sum(),
        "heat_hw_per_1k": g["heat_hw_complaints"].sum() / g["units"].sum() * 1000,
        "class_c_per_1k": g["hpd_class_c"].sum() / g["units"].sum() * 1000,
        "evictions_per_1k": g["residential_evictions"].sum() / g["units"].sum() * 1000,
        "chronic_absence_ems": w(g, "chronic_absence_ems", "students_measured"),
        "share_units_in_top_quartile_distress":
            g.loc[g["housing_distress_index"] >= cd["housing_distress_index"].quantile(0.75), "units"].sum() / g["units"].sum(),
    }), include_groups=False)
    j = jobs.dropna(subset=["boro_code"]).copy()
    j["borough"] = j["boro_code"].astype(int).map(BOROUGHS)
    jb = j.groupby("borough").agg(postings=("job_id", "count"), positions=("number_of_positions", "sum"),
                                  median_salary=("annual_salary_mid", "median"))
    entry = j[j["career_level"].isin(["Entry-Level", "Student"])].groupby("borough")["number_of_positions"].sum()
    b = b.join(jb)
    b["entry_level_positions"] = entry.reindex(b.index).fillna(0)
    b["positions_per_10k_units"] = b["positions"] / b["residential_units"] * 10_000
    # A single posting can carry hundreds of positions (e.g. 600 Correction
    # Officers), so distinct postings are the steadier measure of where hiring is.
    b["postings_per_10k_units"] = b["postings"] / b["residential_units"] * 10_000
    b["share_of_city_positions"] = b["positions"] / b["positions"].sum()
    b["share_of_city_units"] = b["residential_units"] / b["residential_units"].sum()
    order = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
    return b.reindex(order).reset_index()


def jobs_findings(jobs: pd.DataFrame, payroll: pd.DataFrame, quality: dict) -> dict:
    j = jobs.dropna(subset=["boro_code"])
    big = j.sort_values("number_of_positions", ascending=False).head(5)
    doe = payroll[payroll["agency_name"].str.contains("DEPT OF ED", na=False)]
    return {
        "snapshot_date": quality["jobs"]["postings_snapshot_date"],
        "unique_jobs": int(len(jobs)), "resolved_jobs": int(len(j)),
        "resolved_share": round(len(j) / len(jobs), 3),
        "largest_postings": big[["agency", "business_title", "number_of_positions", "boro_code"]].to_dict(orient="records"),
        "payroll_fy2025_active_annual": int(payroll["headcount"].sum()),
        "payroll_doe_coded_manhattan": doe[["agency_name", "headcount", "share_coded_manhattan"]].round(3).to_dict(orient="records"),
    }


# ---------------------------------------------------------------- figures
def _bins(values, k=5):
    edges = np.quantile(values.dropna(), np.linspace(0, 1, k + 1))
    return edges, np.clip(np.searchsorted(edges, values, side="right") - 1, 0, k - 1)


def choropleth(cd, geo, col, title, fmt, path, note=None):
    edges, idx = _bins(cd[col])
    color = dict(zip(cd["cd"], [BLUE_RAMP[i] for i in idx]))
    fig, ax = plt.subplots(figsize=(7, 7.4))
    for f in geo["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ax.add_patch(Polygon(poly[0], closed=True, facecolor=color.get(f["id"], "#efefef"),
                                 edgecolor="white", linewidth=0.8))
    ax.autoscale(); ax.set_aspect(1.32); ax.axis("off")
    ax.set_title(title, loc="left", pad=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in BLUE_RAMP]
    labels = [f"{fmt(edges[i])} – {fmt(edges[i + 1])}" for i in range(5)]
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=9, title="Quintile",
              title_fontsize=9)
    if note:
        fig.text(0.02, 0.02, note, fontsize=8, color=INK_2)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def scatter(cd, path):
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.scatter(cd["housing_distress_index"], cd["chronic_absence_ems"] * 100, s=46,
               color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
    slope, intercept = np.polyfit(cd["housing_distress_index"], cd["chronic_absence_ems"] * 100, 1)
    xs = np.linspace(cd["housing_distress_index"].min(), cd["housing_distress_index"].max(), 50)
    ax.plot(xs, intercept + slope * xs, color=INK_2, linewidth=1.5, linestyle="--", zorder=2)
    label = pd.concat([cd.nlargest(3, "housing_distress_index"), cd.nsmallest(2, "housing_distress_index"),
                       cd.nlargest(2, "chronic_absence_ems")]).drop_duplicates("cd")
    placed = []
    for _, r in label.sort_values("chronic_absence_ems", ascending=False).iterrows():
        x, y = r["housing_distress_index"], r["chronic_absence_ems"] * 100
        dy = 4
        if any(abs(x - px) < 0.5 and abs(y - py) < 1.2 for px, py in placed):
            dy = -12                                   # nudge below a crowded neighbor
        placed.append((x, y))
        ax.annotate(r["district"], (x, y), xytext=(6, dy), textcoords="offset points",
                    fontsize=8, color=INK)
    ax.set_xlabel("Housing distress index (0 = typical district; +1 = one SD worse)")
    ax.set_ylabel("Chronically absent, grades K-8 (%)")
    ax.set_title("Districts with worse housing have more chronic absence")
    fig.text(0.01, -0.02, "Each dot is one of 59 community districts. FY2025 housing data; 2024-25 school year. "
             "Dashed line: least-squares fit.", fontsize=8, color=INK_2)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def borough_bars(b, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    y = np.arange(len(b))
    for ax, col, title, fmt in [
        (axes[0], "share_units_in_top_quartile_distress", "Homes in the city's most-distressed districts", lambda v: f"{v:.0%}"),
        (axes[1], "postings_per_10k_units", "Open city-government job postings per 10,000 homes", lambda v: f"{v:.1f}"),
    ]:
        ax.barh(y, b[col], color=BLUE, height=0.6)
        for yi, v in zip(y, b[col]):
            ax.text(v, yi, "  " + fmt(v), va="center", fontsize=9, color=INK)
        ax.set_title(title, fontsize=11); ax.grid(axis="y", visible=False)
        ax.set_xticks([])
        ax.set_xlim(0, b[col].max() * 1.25)
    axes[0].set_yticks(y, b["borough"]); axes[0].invert_yaxis()
    fig.text(0.01, -0.04, "Left: share of each borough's residential units in the top quartile of the housing distress index. "
             "Right: distinct live Jobs NYC postings, by resolved work borough.", fontsize=8, color=INK_2)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    cd, schools, jobs, payroll, geo, quality = load()
    district, cd = district_findings(cd)
    models, _ = school_models(schools, cd)
    boroughs = borough_table(cd, jobs)
    boroughs.to_csv(PROCESSED / "borough_summary.csv", index=False)
    cd.to_csv(PROCESSED / "cd_profile.csv", index=False)       # adds expected_absence, absence_gap

    results = {"data_quality": quality, "districts": district, "school_models": models,
               "boroughs": boroughs.round(4).to_dict(orient="records"),
               "jobs": jobs_findings(jobs, payroll, quality)}
    (DOCS / "results.json").write_text(json.dumps(results, indent=2, default=float))

    choropleth(cd, geo, "housing_distress_index", "Housing distress index, FY2025",
               lambda v: f"{v:+.1f}", DOCS / "map_housing_distress.png",
               "Mean z-score of log rates per 1,000 homes: 311 heat/hot-water complaints, HPD Class C violations, residential evictions.")
    choropleth(cd, geo, "chronic_absence_ems", "Chronic absence, grades K-8, 2024-25",
               lambda v: f"{v:.0%}", DOCS / "map_chronic_absence.png",
               "Share of students missing 10%+ of school days. NYC DOE School Quality Reports, enrollment-weighted by district.")
    scatter(cd, DOCS / "housing_vs_absence.png")
    borough_bars(boroughs, DOCS / "boroughs_housing_jobs.png")
    print(json.dumps({k: results[k] for k in ("school_models",)}, indent=2, default=float))
    print(boroughs.round(3).to_string())


if __name__ == "__main__":
    main()
