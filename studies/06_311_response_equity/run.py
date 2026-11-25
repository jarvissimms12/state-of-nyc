"""Study 6 — Do some neighborhoods wait longer for the city to fix the same problem?

Data: 311 Service Requests (NYC Open Data erm2-nwe9), FY2025 (Jul 2024 - Jun
2025), nine high-volume complaint types across five agencies. Neighborhood
poverty from the NYC Health Department's Environment & Health Data Portal
(ACS 2019-23, by community district).

Resolution time = closed_date - created_date. The comparison is always
*within* a complaint type: a pothole in one district vs a pothole in another,
never a pothole vs a noise complaint.

Per district, a "wait index" = median over complaint types of
log(district median hours / citywide median hours) for that type, so 0 = the
city's typical speed, +0.69 = twice as long.

    python studies/06_311_response_equity/run.py
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
from src.config import FY_END, FY_START, PROCESSED, RAW  # noqa: E402
from src.geo import parse_311_board  # noqa: E402

TYPES = {
    "HEAT/HOT WATER": "HPD", "UNSANITARY CONDITION": "HPD", "PAINT/PLASTER": "HPD",
    "Noise - Residential": "NYPD", "Blocked Driveway": "NYPD",
    "Water System": "DEP", "Street Condition": "DOT", "Street Light Condition": "DOT",
    "Dirty Condition": "DSNY",
}
# Rodent complaints were tested and dropped: 59% close within a minute because
# DOHMH closes the request when it is routed to inspection, so closing time
# says nothing about when the problem was dealt with.
BLUE, RED, INK, INK_2, GRID = "#257de0", "#ec4647", "#191919", "#666666", "#e5e5e5"
MIN_PER_CELL = 50


def fetch_type(ctype: str) -> pd.DataFrame:
    cache = RAW / "311" / f"{ctype.replace('/', '_').replace(' ', '_')}_FY2025.csv.gz"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["created_date", "closed_date"])
    esc = ctype.replace("'", "''")
    df = soda.fetch_all(
        "erm2-nwe9", select="unique_key, created_date, closed_date, status, community_board",
        where=f"complaint_type = '{esc}' AND created_date between '{FY_START}' and '{FY_END}'",
        order="unique_key")
    df["created_date"] = pd.to_datetime(df["created_date"])
    df["closed_date"] = pd.to_datetime(df.get("closed_date"), errors="coerce")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def prepare(df: pd.DataFrame, ctype: str) -> tuple[pd.DataFrame, dict]:
    q = {"requests": int(len(df))}
    df = df.assign(cd=df["community_board"].map(parse_311_board), complaint_type=ctype)
    q["no_district"] = int(df["cd"].isna().sum())
    df["hours"] = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600
    q["still_open"] = int(df["closed_date"].isna().sum())
    q["closed_before_created"] = int((df["hours"] < 0).sum())
    # closed within a minute of being opened = closed on intake (duplicate,
    # referral, no access), not resolved; excluded from wait times
    q["closed_within_1_min"] = int((df["hours"].between(0, 1 / 60)).sum())
    ok = df.dropna(subset=["cd", "hours"])
    ok = ok[ok["hours"] > 1 / 60]
    q["used"] = int(len(ok))
    return ok[["cd", "complaint_type", "hours"]].astype({"cd": int}), q


def main() -> dict:
    frames, quality = [], {}
    for ctype in TYPES:
        part, q = prepare(fetch_type(ctype), ctype)
        frames.append(part)
        quality[ctype] = q
        print(f"{ctype:24s} {q}")
    d = pd.concat(frames, ignore_index=True)

    city = d.groupby("complaint_type")["hours"].median().rename("city_median_h")
    cell = (d.groupby(["cd", "complaint_type"])["hours"]
              .agg(n="size", median_h="median", p90_h=lambda s: s.quantile(0.9)).reset_index())
    cell = cell[cell["n"] >= MIN_PER_CELL].merge(city, on="complaint_type")
    cell["log_ratio"] = np.log(cell["median_h"].clip(lower=0.05) / cell["city_median_h"].clip(lower=0.05))

    cd = pd.read_csv(PROCESSED / "cd_profile.csv")
    pov = ehdp.by_community_district(221).rename(columns={"value": "poverty_pct", "period": "poverty_period"})
    wait = cell.groupby("cd").agg(wait_index=("log_ratio", "median"), types=("complaint_type", "size"))
    prof = (cd[["cd", "district", "cd_name", "borough", "housing_distress_index"]]
              .merge(wait.reset_index(), on="cd").merge(pov[["cd", "poverty_pct"]], on="cd", how="left"))

    # Within-type models: does a district's poverty predict slower service for the same problem?
    cell = cell.merge(prof[["cd", "poverty_pct", "borough"]], on="cd")
    cell["pov_10"] = cell["poverty_pct"] / 10
    per_type = {}
    for ctype, g in cell.groupby("complaint_type"):
        r, p = stats.spearmanr(g["poverty_pct"], g["median_h"])
        per_type[ctype] = {"agency": TYPES[ctype], "districts": int(len(g)),
                           "city_median_hours": round(float(g["city_median_h"].iloc[0]), 1),
                           "fastest_district_median_h": round(float(g["median_h"].min()), 1),
                           "slowest_district_median_h": round(float(g["median_h"].max()), 1),
                           "spearman_poverty_vs_median": round(float(r), 3), "p": round(float(p), 4)}
    pooled = smf.wls("log_ratio ~ pov_10 + C(complaint_type)", cell, weights=np.sqrt(cell["n"])).fit(
        cov_type="cluster", cov_kwds={"groups": cell["cd"]})
    boro = smf.wls("log_ratio ~ pov_10 + C(complaint_type) + C(borough)", cell, weights=np.sqrt(cell["n"])).fit(
        cov_type="cluster", cov_kwds={"groups": cell["cd"]})

    rho, p_rho = stats.spearmanr(prof["poverty_pct"], prof["wait_index"])
    slow = prof.sort_values("wait_index", ascending=False)
    results = {
        "window": "FY2025 (Jul 2024 - Jun 2025)",
        "complaint_types": TYPES,
        "data_quality": quality,
        "requests_used": int(len(d)),
        "per_type": dict(sorted(per_type.items(), key=lambda kv: kv[1]["spearman_poverty_vs_median"], reverse=True)),
        "pooled_within_type": {
            "pct_change_in_median_wait_per_10pts_poverty": round(float(np.expm1(pooled.params["pov_10"]) * 100), 1),
            "ci95_pct": [round(float(np.expm1(x) * 100), 1) for x in pooled.conf_int().loc["pov_10"]],
            "p": round(float(pooled.pvalues["pov_10"]), 4),
            "with_borough_controls_pct": round(float(np.expm1(boro.params["pov_10"]) * 100), 1),
            "with_borough_controls_ci95": [round(float(np.expm1(x) * 100), 1) for x in boro.conf_int().loc["pov_10"]],
            "with_borough_controls_p": round(float(boro.pvalues["pov_10"]), 4)},
        "district_wait_index_vs_poverty": {"spearman": round(float(rho), 3), "p": round(float(p_rho), 4)},
        "slowest_districts": slow.head(8)[["district", "cd_name", "wait_index", "poverty_pct"]].round(3).to_dict(orient="records"),
        "fastest_districts": slow.tail(5)[["district", "cd_name", "wait_index", "poverty_pct"]].round(3).to_dict(orient="records"),
        "borough_wait_index": prof.groupby("borough")["wait_index"].median().round(3).to_dict(),
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=float))
    cell.round(4).to_csv(HERE / "district_by_type.csv", index=False)
    prof.round(4).to_csv(HERE / "district_wait_index.csv", index=False)

    # Figure: per-type spread (fastest/median/slowest district) as a dot plot
    pt = pd.DataFrame(per_type).T.sort_values("city_median_hours")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    y = np.arange(len(pt))
    for i, (_, r) in enumerate(pt.iterrows()):
        ax.plot([r["fastest_district_median_h"], r["slowest_district_median_h"]], [i, i], color="#cfcfcf", linewidth=3, zorder=1)
    ax.scatter(pt["fastest_district_median_h"], y, color=BLUE, s=30, zorder=3, label="Fastest district")
    ax.scatter(pt["city_median_hours"], y, color=INK, s=30, marker="|", zorder=3, label="Citywide median")
    ax.scatter(pt["slowest_district_median_h"], y, color=RED, s=30, zorder=3, label="Slowest district")
    ax.set_xscale("log"); ax.set_yticks(y, [f"{t} ({TYPES[t]})" for t in pt.index], fontsize=8.5)
    ax.set_xlabel("Median hours to close (log scale)")
    ax.set_title("Same problem, different wait", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax = axes[1]
    ax.scatter(prof["poverty_pct"], np.expm1(prof["wait_index"]) * 100, color=BLUE, s=40,
               edgecolor="white", linewidth=1.2, zorder=3)
    for _, r in pd.concat([slow.head(3), slow.tail(2)]).iterrows():
        ax.annotate(r["district"], (r["poverty_pct"], np.expm1(r["wait_index"]) * 100),
                    xytext=(5, 3), textcoords="offset points", fontsize=7.5, color=INK)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xlabel("Residents below poverty line (%, ACS 2019-23)")
    ax.set_ylabel("Typical wait vs. city, same complaint (%)")
    ax.set_title("Wait index by district", loc="left", fontweight="bold")
    for a in axes:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.grid(color=GRID); a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(HERE / "response_equity.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(json.dumps({k: results[k] for k in ("requests_used", "pooled_within_type", "district_wait_index_vs_poverty",
                                             "borough_wait_index")}, indent=2))
    for k, v in results["per_type"].items():
        print(f"  {k:24s} {v}")
    return results


if __name__ == "__main__":
    import matplotlib.ticker  # noqa: F401
    main()
