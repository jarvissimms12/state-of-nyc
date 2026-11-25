"""Study 5 — How long do NYC government job postings stay open?

Data: Jobs NYC Postings (NYC Open Data kpav-sd4t). The dataset only holds
postings that are open *today*; closed postings disappear. So there are two
parts:

  1. Today's snapshot: how long each open posting has been up
     (process date - posting date), by agency, job category, career level and
     civil-service classification. Long-open postings are a hard-to-fill signal.
  2. History: every run saves a dated snapshot to data/snapshots/. Once two or
     more exist, a posting that was present and then disappears is treated as
     closed, and its time open = last day seen - posting date. The monthly
     GitHub Actions refresh builds this history automatically.

Caveat: the open-today view over-represents long-running postings (a posting
open for a year is more likely to be caught in any snapshot than one open
for a week). That's why it's reported as "age of open postings", not
time-to-fill, until the snapshot history exists.

    python studies/05_city_job_postings/run.py
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

from src import soda  # noqa: E402

SNAPSHOTS = ROOT / "data" / "snapshots"
BLUE, INK, INK_2, GRID = "#257de0", "#191919", "#666666", "#e5e5e5"
KEEP = ["job_id", "agency", "posting_type", "number_of_positions", "business_title",
        "civil_service_title", "title_classification", "job_category", "career_level",
        "full_time_part_time_indicator", "salary_range_from", "salary_range_to",
        "salary_frequency", "posting_date", "post_until", "posting_updated", "process_date"]


ACRONYMS = {"Nyc": "NYC", "Hra": "HRA", "Nypd": "NYPD", "Fdny": "FDNY", "Cuny": "CUNY", "Doe": "DOE"}


def pretty(name: str) -> str:
    import string
    words = string.capwords(name.lower()).split()
    return " ".join(ACRONYMS.get(w.split("/")[0], w) if "/" not in w else
                    "/".join(ACRONYMS.get(p.capitalize(), p.capitalize()) for p in w.split("/")) for w in words)


def fetch() -> pd.DataFrame:
    df = soda.fetch_all("kpav-sd4t", select=", ".join(KEEP), where=None, order="job_id, posting_type")
    for c in ("posting_date", "posting_updated", "process_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["number_of_positions"] = pd.to_numeric(df["number_of_positions"], errors="coerce").fillna(1)
    # internal + external listings of the same job count once
    return df.sort_values("posting_type").drop_duplicates("job_id", keep="first")


def save_snapshot(df: pd.DataFrame) -> Path:
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    day = df["process_date"].max().date().isoformat()
    path = SNAPSHOTS / f"job_postings_{day}.csv.gz"
    df[["job_id", "agency", "business_title", "job_category", "career_level",
        "number_of_positions", "posting_date", "process_date"]].to_csv(path, index=False)
    return path


def closed_from_snapshots() -> pd.DataFrame:
    """Postings seen in one snapshot and gone from the next = closed."""
    files = sorted(SNAPSHOTS.glob("job_postings_*.csv.gz"))
    if len(files) < 2:
        return pd.DataFrame()
    snaps = {f.stem.split("_")[-1].replace(".csv", ""): pd.read_csv(f, parse_dates=["posting_date"]) for f in files}
    days = sorted(snaps)
    closed = []
    for prev, nxt in zip(days, days[1:]):
        gone = snaps[prev][~snaps[prev]["job_id"].isin(snaps[nxt]["job_id"])].copy()
        gone["last_seen"] = pd.Timestamp(prev)
        gone["closed_by"] = pd.Timestamp(nxt)
        closed.append(gone)
    out = pd.concat(closed, ignore_index=True)
    out["days_open_at_least"] = (out["last_seen"] - out["posting_date"]).dt.days
    return out


def created_by_bound(df: pd.DataFrame) -> pd.Series:
    """Lower bound on when each job's requisition existed, from job ID order.

    Job IDs are assigned in sequence. If job A was posted on date d, every job
    with a lower ID already existed on d. So for each month-end m, let L_m be
    the highest ID among postings dated on or before m; any job with ID <= L_m
    existed by m, however recent its current posting date is."""
    ids = pd.to_numeric(df["job_id"], errors="coerce")
    ends = pd.date_range(df["posting_date"].min().normalize(), df["process_date"].max(), freq="ME")
    bound = df["posting_date"].copy()
    for m in ends[::-1]:                              # latest first, so earlier months overwrite
        dated = ids[df["posting_date"] <= m]
        if dated.empty:
            continue
        bound = bound.where(~(ids <= dated.max()), np.minimum(bound, m))
    return bound


def summarize(df: pd.DataFrame, by: str, min_n: int = 15) -> pd.DataFrame:
    g = df.groupby(by)
    t = pd.DataFrame({
        "postings": g.size(),
        "positions": g["number_of_positions"].sum(),
        "median_days_since_posted": g["age_days"].median(),
        "median_days_in_pipeline_min": g["pipeline_days_min"].median(),
        "share_reposted": g["reposted"].mean(),
        "share_in_pipeline_180plus": g["pipeline_days_min"].apply(lambda s: (s >= 180).mean()),
    })
    return t[t["postings"] >= min_n].sort_values("median_days_in_pipeline_min", ascending=False)


def main() -> dict:
    df = fetch()
    snap = save_snapshot(df)
    df["age_days"] = (df["process_date"] - df["posting_date"]).dt.days
    df = df[df["age_days"].between(0, 3650)].copy()
    df["existed_by"] = created_by_bound(df)
    df["pipeline_days_min"] = (df["process_date"] - df["existed_by"]).dt.days
    df["reposted"] = df["existed_by"] < df["posting_date"] - pd.Timedelta(days=31)
    df["category"] = df["job_category"].fillna("Unknown").str.split(",").str[0].str.strip()
    df["annual_mid"] = ((pd.to_numeric(df["salary_range_from"], errors="coerce")
                         + pd.to_numeric(df["salary_range_to"], errors="coerce")) / 2
                        * df["salary_frequency"].map({"Annual": 1, "Hourly": 1820, "Daily": 260}).fillna(1))
    df["salary_band"] = pd.cut(df["annual_mid"], [0, 60_000, 80_000, 100_000, 130_000, np.inf],
                               labels=["<$60k", "$60–80k", "$80–100k", "$100–130k", "$130k+"])

    by_agency = summarize(df, "agency")
    closed = closed_from_snapshots()
    results = {
        "snapshot_date": str(df["process_date"].max().date()),
        "snapshot_saved": str(snap.relative_to(ROOT)),
        "snapshots_available": len(list(SNAPSHOTS.glob("job_postings_*.csv.gz"))),
        "open_postings": int(len(df)), "open_positions": int(df["number_of_positions"].sum()),
        "median_days_since_posted": float(df["age_days"].median()),
        "p99_days_since_posted": float(df["age_days"].quantile(0.99)),
        "share_reposted": round(float(df["reposted"].mean()), 3),
        "median_days_in_pipeline_min": float(df["pipeline_days_min"].median()),
        "share_in_pipeline_90plus": round(float((df["pipeline_days_min"] >= 90).mean()), 3),
        "share_in_pipeline_180plus": round(float((df["pipeline_days_min"] >= 180).mean()), 3),
        "by_agency_longest": by_agency.head(10).round(3).reset_index().to_dict(orient="records"),
        "by_agency_shortest": by_agency.tail(5).round(3).reset_index().to_dict(orient="records"),
        "by_category": summarize(df, "category").round(3).reset_index().to_dict(orient="records"),
        "by_career_level": summarize(df, "career_level", 5).round(3).reset_index().to_dict(orient="records"),
        "by_classification": summarize(df, "title_classification", 5).round(3).reset_index().to_dict(orient="records"),
        "by_salary_band": summarize(df, "salary_band", 5).round(3).reset_index().astype({"salary_band": str}).to_dict(orient="records"),
        "closed_postings_from_history": int(len(closed)),
        "closed_median_days_open": float(closed["days_open_at_least"].median()) if len(closed) else None,
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=float))
    by_agency.round(3).to_csv(HERE / "postings_by_agency.csv")
    if len(closed):
        closed.to_csv(HERE / "closed_postings.csv", index=False)

    top = by_agency.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    y = np.arange(len(top))
    ax.barh(y + 0.18, top["median_days_in_pipeline_min"], color=BLUE, height=0.34,
            label="At least this long in the hiring pipeline (from job ID)")
    ax.barh(y - 0.18, top["median_days_since_posted"], color="#a3cbf7", height=0.34,
            label="Since current posting date")
    for i, (v, r) in enumerate(zip(top["median_days_in_pipeline_min"], top["share_reposted"])):
        ax.text(v, i + 0.18, f"  {v:.0f} d · {r:.0%} re-posted", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [pretty(a)[:38] for a in top.index], fontsize=8.5)
    ax.set_xlabel("Median days (agencies with 15+ open postings)")
    ax.set_xlim(0, top["median_days_in_pipeline_min"].max() * 1.45)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"NYC job postings: posting dates hide how long roles stay open ({results['snapshot_date']})",
                 loc="left", fontweight="bold", fontsize=11)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    fig.savefig(HERE / "postings_age_by_agency.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(json.dumps({k: v for k, v in results.items() if not k.startswith("by_")}, indent=2, default=float))
    return results


if __name__ == "__main__":
    main()
