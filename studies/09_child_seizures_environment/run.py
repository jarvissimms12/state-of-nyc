"""Study 9 — Children's environment and neurological risk: lead, old buildings,
schools (part A, real data) and pediatric seizure hospitalizations (part B,
SPARCS; see seizures_sparcs.py).

Why lead: lead is the best-documented environmental cause of seizures in
children. At very high blood levels it causes lead encephalopathy, with
seizures; at the much lower levels seen in NYC today it is linked to learning
and attention problems rather than seizures. Most childhood seizures (febrile
seizures, epilepsy) have other causes, so this study maps *exposure risk*, not
seizure causes.

Part A, all by the Health Department's 42 UHF neighborhoods:
  * children under 6 with elevated blood lead (5+ mcg/dL and 3.5+), 2024, and
    the citywide trend since 2005 (NYC Health Department, EHDP)
  * residential units in buildings built before 1960 (PLUTO, every tax lot,
    placed in its neighborhood by coordinates). NYC Local Law 1 presumes paint
    in pre-1960 apartments contains lead.
  * HPD lead-based-paint hazard violations, FY2025 (order numbers 616, 617, 624:
    peeling or deteriorated lead paint), by coordinates
  * poverty (ACS 2019-23, EHDP)
  * schools: K-8 chronic absence 2024-25 and share of students with
    disabilities (2021-22), schools placed by coordinates

    python studies/09_child_seizures_environment/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import shape  # noqa: E402
from scipy import stats  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from src import ehdp, soda  # noqa: E402
from src.config import FY_END, FY_START, PROCESSED, RAW  # noqa: E402

LEAD_HAZARD_ORDERS = ("616", "617", "624")
BLUE, ORANGE, INK, INK_2, GRID = "#257de0", "#f47230", "#191919", "#666666", "#e5e5e5"


def neighborhoods() -> tuple[list, pd.DataFrame]:
    gj = ehdp.uhf42_geojson()
    polys, rows = [], []
    for f in gj["features"]:
        code = int(f["properties"]["GEOCODE"])
        if code == 0:          # unassigned area in the source file
            continue
        polys.append((code, shape(f["geometry"])))
        rows.append({"uhf": code, "name": f["properties"]["GEONAME"], "borough": f["properties"]["BOROUGH"]})
    return polys, pd.DataFrame(rows)


def assign(lon: np.ndarray, lat: np.ndarray, polys: list) -> np.ndarray:
    out = np.zeros(len(lon), dtype=int)
    ok = np.isfinite(lon) & np.isfinite(lat)
    for code, poly in polys:
        shapely.prepare(poly)
        hit = ok & (out == 0)
        idx = np.where(hit)[0]
        inside = shapely.contains_xy(poly, lon[idx], lat[idx])
        out[idx[inside]] = code
    return out


def pluto_lots() -> pd.DataFrame:
    cache = RAW / "pluto_residential_lots.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    df = soda.fetch_all("64uk-42ks", select="bbl, unitsres, yearbuilt, latitude, longitude",
                        where="unitsres > 0", order="bbl")
    for c in ("unitsres", "yearbuilt", "latitude", "longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_csv(cache, index=False)
    return df


def lead_violations() -> pd.DataFrame:
    codes = ", ".join(f"'{c}'" for c in LEAD_HAZARD_ORDERS)
    df = soda.fetch_all("wvxf-dwi5", select="violationid, ordernumber, latitude, longitude, bbl",
                        where=f"ordernumber in ({codes}) AND novissueddate between '{FY_START}' and '{FY_END}'",
                        order="violationid")
    for c in ("latitude", "longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> dict:
    polys, nb = neighborhoods()

    lots = pluto_lots()
    lots["uhf"] = assign(lots["longitude"].to_numpy(), lots["latitude"].to_numpy(), polys)
    lots["pre1960_units"] = np.where(lots["yearbuilt"].between(1, 1959), lots["unitsres"], 0)
    housing = lots[lots["uhf"] > 0].groupby("uhf").agg(units=("unitsres", "sum"), pre1960_units=("pre1960_units", "sum"))
    housing["pre1960_share"] = housing["pre1960_units"] / housing["units"]

    viol = lead_violations()
    viol["uhf"] = assign(viol["longitude"].to_numpy(), viol["latitude"].to_numpy(), polys)
    lead_v = viol[viol["uhf"] > 0].groupby("uhf").size().rename("lead_paint_violations")

    schools = pd.read_csv(PROCESSED / "schools.csv")
    demo = soda.query("c7ru-d68s", "SELECT dbn, students_with_disabilities_1 WHERE year = '2021-22' LIMIT 5000")
    demo["swd_share"] = pd.to_numeric(demo["students_with_disabilities_1"], errors="coerce")
    schools = schools.merge(demo[["dbn", "swd_share"]], on="dbn", how="left")
    schools["uhf"] = assign(schools["longitude"].to_numpy(dtype=float), schools["latitude"].to_numpy(dtype=float), polys)
    s = schools[(schools["uhf"] > 0) & schools["chronic_absence_ems"].notna() & (schools["n_chronic_ems"] > 0)]
    sch = s.groupby("uhf").apply(lambda g: pd.Series({
        "chronic_absence_k8": np.average(g["chronic_absence_ems"], weights=g["n_chronic_ems"]),
        "swd_share": np.average(g["swd_share"].fillna(g["swd_share"].mean()), weights=g["n_chronic_ems"]),
        "k8_schools": len(g)}), include_groups=False)

    bll5 = ehdp.latest(777, "UHF42").rename(columns={"geo_id": "uhf", "value": "bll5_per_1k"})
    bll35 = ehdp.latest(1409, "UHF42").rename(columns={"geo_id": "uhf", "value": "bll35_per_1k"})
    tested = ehdp.latest(795, "UHF42").rename(columns={"geo_id": "uhf", "value": "children_tested"})
    pov = ehdp.latest(221, "UHF42").rename(columns={"geo_id": "uhf", "value": "poverty_pct"})
    old50 = ehdp.latest(27, "UHF42").rename(columns={"geo_id": "uhf", "value": "pre1950_pct_acs"})

    d = nb.set_index("uhf").join([housing, lead_v, sch])
    for part in (bll5, bll35, tested, pov, old50):
        d = d.join(part.set_index("uhf")[[c for c in part.columns if c not in ("geo_type", "period", "uhf")]])
    d["lead_paint_violations"] = d["lead_paint_violations"].fillna(0)
    d["lead_viol_per_1k_old_units"] = d["lead_paint_violations"] / d["pre1960_units"].clip(lower=1) * 1000
    d = d.reset_index()

    rho = lambda a, b: [round(float(x), 3) for x in stats.spearmanr(d[a], d[b], nan_policy="omit")]
    m = d.dropna(subset=["bll5_per_1k", "poverty_pct", "pre1960_share"]).copy()
    z = lambda x: (x - x.mean()) / x.std()
    m["z_old"], m["z_pov"], m["z_viol"] = z(m["pre1960_share"]), z(m["poverty_pct"]), z(np.log1p(m["lead_viol_per_1k_old_units"]))
    fit = smf.ols("bll5_per_1k ~ z_old + z_pov + z_viol", m).fit(cov_type="HC3")
    term = lambda t: {"per_sd": round(float(fit.params[t]), 2), "p": round(float(fit.pvalues[t]), 4),
                      "ci95": [round(float(x), 2) for x in fit.conf_int().loc[t]]}

    trend = ehdp.measure(777, "Citywide").sort_values("period")
    trend_n = ehdp.measure(774, "Citywide").sort_values("period")
    top = d.sort_values("bll5_per_1k", ascending=False)
    results = {
        "periods": {"blood_lead": str(bll5["period"].iloc[0]), "poverty": str(pov["period"].iloc[0]),
                    "lead_paint_violations": "FY2025", "pluto": "current", "schools": "2024-25 (absence), 2021-22 (disability share)"},
        "citywide_bll5_per_1k": {str(p): float(v) for p, v in zip(trend["period"], trend["value"])},
        "citywide_bll5_children": {str(p): int(v) for p, v in zip(trend_n["period"], trend_n["value"])},
        "residential_units_mapped": int(housing["units"].sum()),
        "pre1960_share_citywide": round(float(housing["pre1960_units"].sum() / housing["units"].sum()), 3),
        "lead_paint_violations_fy2025": int(len(viol)), "lead_paint_violations_mapped": int((viol["uhf"] > 0).sum()),
        "spearman_with_child_bll5": {
            "pre1960_share": rho("pre1960_share", "bll5_per_1k"),
            "pre1950_pct_acs": rho("pre1950_pct_acs", "bll5_per_1k"),
            "lead_viol_per_1k_old_units": rho("lead_viol_per_1k_old_units", "bll5_per_1k"),
            "poverty_pct": rho("poverty_pct", "bll5_per_1k"),
            "chronic_absence_k8": rho("chronic_absence_k8", "bll5_per_1k"),
            "swd_share": rho("swd_share", "bll5_per_1k"),
        },
        "model_bll5": {"n": int(len(m)), "r2": round(float(fit.rsquared), 3),
                       "pre1960_share": term("z_old"), "poverty": term("z_pov"), "lead_violations": term("z_viol")},
        "highest_bll5": top.head(8)[["name", "borough", "bll5_per_1k", "children_tested", "pre1960_share",
                                     "lead_viol_per_1k_old_units", "poverty_pct"]].round(3).to_dict(orient="records"),
        "lowest_bll5": top.tail(5)[["name", "borough", "bll5_per_1k", "pre1960_share", "poverty_pct"]].round(3).to_dict(orient="records"),
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=float))
    d.round(4).to_csv(HERE / "neighborhood_lead_profile.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    yrs = [int(str(p)[:4]) for p in trend["period"]]
    ax.plot(yrs, trend["value"], color=BLUE, linewidth=2, marker="o", markersize=4)
    ax.set_title("Children under 6 with blood lead 5+ µg/dL\nper 1,000 tested, NYC", loc="left", fontweight="bold", fontsize=11)
    ax.set_ylim(0, trend["value"].max() * 1.1)
    ax.set_xticks(range(min(yrs), max(yrs) + 1, 3))
    ax = axes[1]
    ax.scatter(d["pre1960_share"] * 100, d["bll5_per_1k"], s=np.clip(d["children_tested"].fillna(0) / 60, 15, 160),
               color=BLUE, alpha=0.85, edgecolor="white", linewidth=1.2, zorder=3)
    for _, r in top.head(4).iterrows():
        ax.annotate(r["name"], (r["pre1960_share"] * 100, r["bll5_per_1k"]), xytext=(5, 3),
                    textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xlabel("Homes in buildings built before 1960 (%)")
    ax.set_ylabel(f"Children <6 with blood lead 5+ per 1,000 tested ({results['periods']['blood_lead']})")
    r_, _ = stats.spearmanr(d["pre1960_share"], d["bll5_per_1k"], nan_policy="omit")
    ax.set_title(f"42 neighborhoods (dot size = children tested)   ρ = {r_:.2f}", loc="left", fontweight="bold", fontsize=11)
    for a in axes:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.grid(color=GRID); a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(HERE / "lead_exposure.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(json.dumps({k: v for k, v in results.items() if k not in ("highest_bll5", "lowest_bll5")}, indent=2, default=float))
    for r in results["highest_bll5"]:
        print(r)
    return results


if __name__ == "__main__":
    main()
