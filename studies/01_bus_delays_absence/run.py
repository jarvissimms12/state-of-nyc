"""Study 1 — Do school bus breakdowns and delays explain chronic absence?

Follow-up to the main finding that some districts (the Rockaways most of all)
have far more K-8 chronic absence than their schools' economic need predicts.

Data: NYC DOE Office of Pupil Transportation "Bus Breakdown and Delays"
(NYC Open Data ez4e-fazm), 2024-25 school year, joined to the 2024-25 School
Quality Reports already built by build.py.

Linking buses to schools: `schools_serviced` holds a 5-digit code = 2-digit
geographic district + 3-digit school number (e.g. 15879 -> 15K879). A DBN is
district + borough letter + number, so the code matches a DBN on
(district, number). One incident can serve several schools; each school
served is counted.

    python studies/01_bus_delays_absence/run.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from src import soda  # noqa: E402
from src.config import PROCESSED  # noqa: E402

DATASET = "ez4e-fazm"
SCHOOL_YEAR = "2024-2025"
DELAY_MINUTES = {"0-15 Min": 7.5, "16-30 Min": 23, "31-45 Min": 38, "46-60 Min": 53, "61-90 Min": 75.5}
BLUE, INK_2, GRID = "#257de0", "#666666", "#e5e5e5"


def school_key(code: str) -> tuple[str, str] | None:
    """'15879' -> ('15', '879'); DBN '15K879' -> ('15', '879')."""
    code = code.strip().upper()
    if re.fullmatch(r"\d{5}", code):
        return code[:2], code[2:]
    if re.fullmatch(r"\d{2}[MXKQR]\d{3}", code):
        return code[:2], code[3:]
    return None


def fetch_incidents() -> pd.DataFrame:
    df = soda.fetch_all(
        DATASET,
        select="busbreakdown_id, run_type, reason, boro, schools_serviced, how_long_delayed, "
               "breakdown_or_running_late, school_age_or_prek, occurred_on",
        where=f"school_year = '{SCHOOL_YEAR}'", order="busbreakdown_id")
    df["delay_min"] = df["how_long_delayed"].map(DELAY_MINUTES)
    return df


def explode_schools(inc: pd.DataFrame) -> pd.DataFrame:
    rows = inc.assign(code=inc["schools_serviced"].fillna("").str.split(r"[,\s]+")).explode("code")
    rows["key"] = rows["code"].map(lambda c: school_key(c) if isinstance(c, str) and c else None)
    return rows


def main() -> dict:
    inc = fetch_incidents()
    school_age = inc[inc["school_age_or_prek"] == "School-Age"]
    rows = explode_schools(school_age)

    schools = pd.read_csv(PROCESSED / "schools.csv")
    cd = pd.read_csv(PROCESSED / "cd_profile.csv")
    schools["key"] = list(zip(schools["dbn"].str[:2], schools["dbn"].str[3:6]))
    keys = set(schools["key"])

    matched = rows[rows["key"].isin(keys)]
    am = matched[matched["run_type"].str.contains("AM", na=False)]
    per_school = matched.groupby("key").agg(
        bus_incidents=("busbreakdown_id", "count"),
        delay_minutes=("delay_min", "sum"),
        gen_ed_incidents=("run_type", lambda s: s.str.startswith("General Ed").sum()),
        sped_incidents=("run_type", lambda s: s.str.startswith("Special Ed").sum()),
    )
    per_school["am_incidents"] = am.groupby("key").size().reindex(per_school.index).fillna(0)

    # Students with disabilities share (2021-22) - special-ed busing drives most incidents
    demo = soda.query("c7ru-d68s", "SELECT dbn, students_with_disabilities_1 WHERE year = '2021-22' LIMIT 5000")
    demo["swd_share"] = pd.to_numeric(demo["students_with_disabilities_1"], errors="coerce")

    d = (schools.merge(per_school, left_on="key", right_index=True, how="left")
                .merge(demo[["dbn", "swd_share"]], on="dbn", how="left")
                .merge(cd[["cd", "district", "cd_name", "housing_distress_index"]], on="cd", how="left"))
    for c in ("bus_incidents", "am_incidents", "delay_minutes", "gen_ed_incidents", "sped_incidents"):
        d[c] = d[c].fillna(0)

    k8 = d.dropna(subset=["chronic_absence_ems", "eni_2122", "cd", "swd_share"])
    k8 = k8[k8["n_chronic_ems"] > 0].copy()
    k8["am_per_100"] = k8["am_incidents"] / k8["n_chronic_ems"] * 100
    k8["log_am"] = np.log1p(k8["am_per_100"])
    k8["eni_10"] = k8["eni_2122"] * 10
    k8["swd_10"] = k8["swd_share"] * 10
    k8["rockaways"] = (k8["cd"] == 414).astype(int)

    fit = lambda f: smf.wls(f, k8, weights=k8["n_chronic_ems"]).fit(
        cov_type="cluster", cov_kwds={"groups": k8["cd"]})
    base = fit("chronic_absence_ems ~ eni_10 + swd_10")
    bus = fit("chronic_absence_ems ~ eni_10 + swd_10 + log_am")
    rock0 = fit("chronic_absence_ems ~ eni_10 + swd_10 + rockaways")
    rock1 = fit("chronic_absence_ems ~ eni_10 + swd_10 + log_am + rockaways")

    # Exposure comparison: Rockaways vs everywhere else
    grp = k8.assign(area=np.where(k8["rockaways"] == 1, "Rockaways (Queens CD 14)", "Rest of NYC"))
    exposure = grp.groupby("area").apply(lambda g: pd.Series({
        "schools": len(g),
        "students": g["n_chronic_ems"].sum(),
        "am_incidents_per_100_students": g["am_incidents"].sum() / g["n_chronic_ems"].sum() * 100,
        "share_schools_with_any_am_incident": (g["am_incidents"] > 0).mean(),
        "chronic_absence_k8": np.average(g["chronic_absence_ems"], weights=g["n_chronic_ems"]),
    }), include_groups=False)
    delay_rock = matched.merge(d[["key", "cd"]], on="key")
    mean_delay = delay_rock.groupby(delay_rock["cd"] == 414)["delay_min"].mean()

    ci = lambda m, t: [round(float(x), 4) for x in m.conf_int().loc[t]]
    results = {
        "school_year": SCHOOL_YEAR,
        "incidents_total": int(len(inc)),
        "incidents_school_age": int(len(school_age)),
        "school_links": int(len(rows)), "school_links_matched": int(len(matched)),
        "match_rate": round(len(matched) / max(len(rows.dropna(subset=['key'])), 1), 3),
        "share_special_ed_runs": round(float(school_age["run_type"].str.startswith("Special Ed").mean()), 3),
        "share_delays_over_60_min": round(float((school_age["delay_min"] > 60).mean()), 3),
        "top_reasons": school_age["reason"].value_counts(normalize=True).head(5).round(3).to_dict(),
        "n_schools_model": int(len(k8)),
        "bus_coef_per_log_unit": round(bus.params["log_am"], 4), "bus_ci95": ci(bus, "log_am"),
        "bus_p": round(bus.pvalues["log_am"], 4),
        "r2_without_bus": round(base.rsquared, 3), "r2_with_bus": round(bus.rsquared, 3),
        "rockaways_gap_controls_only": round(rock0.params["rockaways"], 4),
        "rockaways_gap_ci95": ci(rock0, "rockaways"),
        "rockaways_gap_with_bus": round(rock1.params["rockaways"], 4),
        "rockaways_gap_with_bus_ci95": ci(rock1, "rockaways"),
        "exposure": exposure.round(4).reset_index().to_dict(orient="records"),
        "mean_delay_minutes": {"rockaways": round(float(mean_delay.get(True, np.nan)), 1),
                               "rest_of_nyc": round(float(mean_delay.get(False, np.nan)), 1)},
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=float))
    d.drop(columns=["key"]).to_csv(HERE / "schools_with_bus_incidents.csv", index=False)

    # Figure: AM incidents per 100 students, by quintile, vs chronic absence
    k8["bus_band"] = pd.cut(k8["am_per_100"], [-0.01, 0, 1, 3, 8, np.inf],
                            labels=["None", "≤1", "1–3", "3–8", ">8"])
    band = k8.groupby("bus_band", observed=True).apply(lambda g: pd.Series({
        "absence": np.average(g["chronic_absence_ems"], weights=g["n_chronic_ems"]),
        "eni": np.average(g["eni_2122"], weights=g["n_chronic_ems"]), "n": len(g)}), include_groups=False)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(range(len(band)), band["absence"] * 100, color=BLUE, width=0.6)
    for i, (a, n) in enumerate(zip(band["absence"], band["n"])):
        ax.text(i, a * 100 + 0.6, f"{a:.0%}\n{int(n)} schools", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(band)), band.index)
    ax.set_xlabel("Morning bus breakdowns/delays per 100 K-8 students, 2024-25")
    ax.set_ylabel("Chronically absent, K-8 (%)")
    ax.set_title("Among schools with bus service, absence rises with bus disruptions", loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True); ax.set_ylim(0, band["absence"].max() * 100 + 8)
    fig.text(0.01, -0.03, "Unadjusted. Schools with more bus incidents also serve more students with disabilities "
             "and higher economic need; see the model in README.", fontsize=8, color=INK_2)
    fig.savefig(HERE / "bus_vs_absence.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(json.dumps(results, indent=2, default=float))
    return results


if __name__ == "__main__":
    main()
