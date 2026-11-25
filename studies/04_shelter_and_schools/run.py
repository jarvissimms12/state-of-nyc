"""Study 4 — Where NYC's shelter population lives, and what it means for schools.

Follow-up to the main finding that a school's share of students in temporary
housing predicts its chronic absence beyond poverty.

Data (NYC Open Data):
  * DHS Daily Report, historical (dwrg-kzni, 2013-08 to 2021-02) + current
    (k46n-sa2m, 2021-03 onward): citywide census of people in DHS shelter
  * Individual Census by Borough, Community District and Facility Type
    (veav-vj3r): monthly, by community district, since July 2018

Blank cells in the district census are treated as zero (DHS leaves a
facility-type column blank when a district has none of that type).

    python studies/04_shelter_and_schools/run.py
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
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402

from src import soda  # noqa: E402
from src.config import PROCESSED  # noqa: E402
from src.geo import cd_code  # noqa: E402

BLUE, ORANGE, INK, INK_2, GRID = "#257de0", "#f47230", "#191919", "#666666", "#e5e5e5"
FAMILY_COLS = ["family_with_chidren_shelter", "family_with_children_comm", "family_cluster"]  # sic: DHS spelling
ADULT_COLS = ["adult_shelter", "adult_shelter_comm_hotel", "adult_family_shelter", "adult_family_comm_hotel"]


def daily_census() -> pd.DataFrame:
    cols = ("date_of_census, total_individuals_in_shelter, total_children_in_shelter, "
            "total_single_adults_in_shelter")
    # the two files name the family-count column differently
    hist = soda.fetch_all("dwrg-kzni", select=cols + ", families_with_children_in AS families_with_children",
                          where=None, order="date_of_census")
    cur = soda.fetch_all("k46n-sa2m", select=cols + ", families_with_children_in_shelter AS families_with_children",
                         where=None, order="date_of_census")
    df = pd.concat([hist, cur], ignore_index=True)
    df["date"] = pd.to_datetime(df.pop("date_of_census"))
    for c in df.columns.drop("date"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def district_census() -> pd.DataFrame:
    df = soda.fetch_all("veav-vj3r", select="*", where=None,
                        order="report_date, borough, community_districts")
    df["date"] = pd.to_datetime(df["report_date"])
    for c in FAMILY_COLS + ADULT_COLS:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0)
    df["cd"] = [cd_code(b, n) for b, n in zip(df["borough"], df["community_districts"])]
    df["families_with_children"] = df[FAMILY_COLS].sum(axis=1)
    df["adults"] = df[ADULT_COLS].sum(axis=1)
    return df


def fy_mean(df: pd.DataFrame, start: str, end: str, col: str) -> pd.Series:
    m = df[(df["date"] >= start) & (df["date"] <= end)]
    return m.groupby("cd")[col].mean()


def main() -> dict:
    daily = daily_census()
    dist = district_census()
    cd = pd.read_csv(PROCESSED / "cd_profile.csv")

    unassigned_share = float(dist.loc[dist["cd"].isna(), "families_with_children"].sum()
                             / dist["families_with_children"].sum())
    d = dist.dropna(subset=["cd"])
    fam25 = fy_mean(d, "2024-07-01", "2025-06-30", "families_with_children")
    fam19 = fy_mean(d, "2018-07-01", "2019-06-30", "families_with_children")
    prof = cd.set_index("cd").join(fam25.rename("family_shelter_fy25")).join(fam19.rename("family_shelter_fy19"))
    prof[["family_shelter_fy25", "family_shelter_fy19"]] = prof[["family_shelter_fy25", "family_shelter_fy19"]].fillna(0)
    prof["family_shelter_per_1k"] = prof["family_shelter_fy25"] / prof["units"] * 1000
    prof["log_shelter"] = np.log1p(prof["family_shelter_per_1k"])
    prof = prof.reset_index()

    rho = lambda a, b: [round(float(x), 3) for x in stats.spearmanr(prof[a], prof[b])]
    m_gap = smf.ols("absence_gap ~ log_shelter", prof).fit(cov_type="HC3")
    m_abs = smf.ols("chronic_absence_ems ~ eni_2122 + log_shelter", prof).fit(cov_type="HC3")

    top = prof.sort_values("family_shelter_fy25", ascending=False)
    share_top10 = float(top.head(10)["family_shelter_fy25"].sum() / prof["family_shelter_fy25"].sum())

    def at(date):
        row = daily.iloc[(daily["date"] - pd.Timestamp(date)).abs().argsort()[:1]].iloc[0]
        return {"date": str(row["date"].date()), "total": int(row["total_individuals_in_shelter"]),
                "children": int(row["total_children_in_shelter"])}
    peak = daily.loc[daily["total_individuals_in_shelter"].idxmax()]
    rock = prof[prof["cd"] == 414].iloc[0]

    results = {
        "daily_range": [str(daily["date"].min().date()), str(daily["date"].max().date())],
        "census_points": {"2014-01-01": at("2014-01-01"), "2019-06-30": at("2019-06-30"),
                          "2022-06-30": at("2022-06-30"), "latest": at(daily["date"].max())},
        "peak": {"date": str(peak["date"].date()), "total": int(peak["total_individuals_in_shelter"]),
                 "children": int(peak["total_children_in_shelter"])},
        "district_census_range": [str(dist["date"].min().date()), str(dist["date"].max().date())],
        "family_shelter_unassigned_share": round(unassigned_share, 4),
        "fy25_family_individuals_total": round(float(prof["family_shelter_fy25"].sum())),
        "fy19_family_individuals_total": round(float(prof["family_shelter_fy19"].sum())),
        "districts_with_any_family_shelter_fy25": int((prof["family_shelter_fy25"] > 0).sum()),
        "top10_districts_share": round(share_top10, 3),
        "top10_districts": top.head(10)[["district", "cd_name", "family_shelter_fy25",
                                         "family_shelter_per_1k", "chronic_absence_ems", "absence_gap"]
                                        ].round(3).to_dict(orient="records"),
        "spearman": {
            "shelter_rate_vs_sth_2021": rho("family_shelter_per_1k", "sth_rate_2021"),
            "shelter_rate_vs_chronic_absence_ems": rho("family_shelter_per_1k", "chronic_absence_ems"),
            "shelter_rate_vs_absence_gap": rho("family_shelter_per_1k", "absence_gap"),
            "shelter_rate_vs_housing_distress": rho("family_shelter_per_1k", "housing_distress_index"),
        },
        "absence_gap_on_log_shelter": {"coef": round(m_gap.params["log_shelter"], 4),
                                       "p": round(m_gap.pvalues["log_shelter"], 4), "r2": round(m_gap.rsquared, 3)},
        "absence_on_eni_and_log_shelter": {"shelter_coef": round(m_abs.params["log_shelter"], 4),
                                           "shelter_p": round(m_abs.pvalues["log_shelter"], 4),
                                           "r2": round(m_abs.rsquared, 3)},
        "rockaways": {"family_shelter_fy25": round(float(rock["family_shelter_fy25"])),
                      "per_1k_units": round(float(rock["family_shelter_per_1k"]), 1),
                      "rank_of_59": int(prof["family_shelter_per_1k"].rank(ascending=False)[prof["cd"] == 414].iloc[0]),
                      "absence_gap": round(float(rock["absence_gap"]), 3)},
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=float))
    prof[["cd", "district", "cd_name", "units", "family_shelter_fy19", "family_shelter_fy25",
          "family_shelter_per_1k", "chronic_absence_ems", "absence_gap", "sth_rate_2021",
          "housing_distress_index"]].round(4).to_csv(HERE / "district_shelter.csv", index=False)

    # Figure 1: citywide census over time
    w = daily.set_index("date")[["total_individuals_in_shelter", "total_children_in_shelter"]].resample("W").mean()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(w.index, w["total_individuals_in_shelter"] / 1000, color=BLUE, linewidth=2, label="All people in DHS shelter")
    ax.plot(w.index, w["total_children_in_shelter"] / 1000, color=ORANGE, linewidth=2, label="Children")
    ax.set_ylabel("Thousands of people"); ax.legend(frameon=False, loc="upper left")
    ax.set_title("NYC Department of Homeless Services shelter census, 2013–2026", loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    fig.text(0.01, -0.02, "Weekly average of the DHS daily census. Counts DHS shelters only; shelters run by other "
             "agencies (HPD, HRA domestic-violence, asylum-seeker emergency sites) are not included.", fontsize=8, color=INK_2)
    fig.savefig(HERE / "shelter_census_trend.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Figure 2: district shelter rate vs absence gap
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(prof["family_shelter_per_1k"], prof["absence_gap"] * 100, s=44, color=BLUE,
               edgecolor="white", linewidth=1.5, zorder=3)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 3, 10, 30, 100], ["0", "1", "3", "10", "30", "100"])
    for _, r in pd.concat([top.head(4), prof[prof["cd"] == 414],
                           prof.nlargest(2, "absence_gap")]).drop_duplicates("cd").iterrows():
        ax.annotate(r["district"], (r["family_shelter_per_1k"], r["absence_gap"] * 100),
                    xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xlabel("People in family shelters per 1,000 homes, FY2025 (log scale)")
    ax.set_ylabel("K-8 chronic absence above (+) or below (−)\nwhat economic need predicts (pts)")
    ax.set_title("Family shelter population and unexplained absence, by district", loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID); ax.set_axisbelow(True)
    fig.savefig(HERE / "shelter_vs_absence_gap.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(json.dumps(results, indent=2, default=float))
    return results


if __name__ == "__main__":
    main()
