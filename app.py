"""State of NYC — interactive dashboard.

    streamlit run app.py

Reads only the committed files in data/processed/ and docs/results.json,
so it runs anywhere (including Streamlit Community Cloud) without API calls.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
P = ROOT / "data" / "processed"
BLUES = ["#cce3fd", "#84b9f3", "#338ced", "#2370cd", "#0f4a96"]
BLUE = "#257de0"

st.set_page_config(page_title="State of NYC", layout="wide")


@st.cache_data
def load():
    cd = pd.read_csv(P / "cd_profile.csv")
    schools = pd.read_csv(P / "schools.csv")
    jobs = pd.read_csv(P / "job_postings.csv")
    boroughs = pd.read_csv(P / "borough_summary.csv")
    payroll = pd.read_csv(P / "payroll_agencies.csv")
    geo = json.loads((P / "community_districts.geojson").read_text())
    results = json.loads((ROOT / "docs" / "results.json").read_text())
    return cd, schools, jobs, boroughs, payroll, geo, results


cd, schools, jobs, boroughs, payroll, geo, R = load()
BORO = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}

METRICS = {
    "Housing distress index": ("housing_distress_index", "{:+.2f}"),
    "311 heat/hot-water complaints per 1,000 homes": ("heat_hw_per_1k", "{:.0f}"),
    "HPD Class C violations per 1,000 homes": ("class_c_per_1k", "{:.0f}"),
    "Residential evictions per 1,000 homes": ("evictions_per_1k", "{:.1f}"),
    "Chronic absence, grades K-8": ("chronic_absence_ems", "{:.0%}"),
    "Chronic absence beyond what poverty predicts": ("absence_gap", "{:+.1%}"),
    "School Economic Need Index (2021-22)": ("eni_2122", "{:.0%}"),
    "Students in temporary housing (2020-21)": ("sth_rate_2021", "{:.1%}"),
}

st.title("State of NYC")
st.caption("Housing conditions, public schools and city jobs across NYC's 59 community districts. "
           "Fiscal year 2025 (July 2024 – June 2025) and the 2024-25 school year. All data from NYC Open Data.")

m = R["school_models"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Heat/hot-water complaints", f"{R['data_quality']['housing']['heat_total']:,}")
c2.metric("HPD violations issued", f"{R['data_quality']['housing']['hpd_total']:,}")
k8 = schools.dropna(subset=["chronic_absence_ems", "n_chronic_ems"])
c3.metric("Chronic absence, K-8", f"{(k8['chronic_absence_ems'] * k8['n_chronic_ems']).sum() / k8['n_chronic_ems'].sum():.1%}")
c4.metric("Open city job postings", f"{R['jobs']['unique_jobs']:,}", help=f"Snapshot {R['jobs']['snapshot_date']}")

tab_map, tab_link, tab_jobs, tab_district, tab_data = st.tabs(
    ["Map", "Housing ↔ schools", "City jobs", "District profile", "Data & methods"])

with tab_map:
    label = st.selectbox("Measure", list(METRICS))
    col, fmt = METRICS[label]
    fig = px.choropleth(cd, geojson=geo, locations="cd", featureidkey="id", color=col,
                        color_continuous_scale=BLUES, hover_name="district",
                        hover_data={"cd_name": True, col: ":.3f", "cd": False})
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=10, b=0), coloraxis_colorbar_title="")
    st.plotly_chart(fig, width="stretch")
    top = cd.sort_values(col, ascending=False)[["district", "cd_name", col]].head(10)
    top[col] = top[col].map(fmt.format)
    st.markdown(f"**Highest 10 — {label}**")
    st.dataframe(top, hide_index=True, width="stretch")

with tab_link:
    st.subheader("Do districts with worse housing have more chronic absence?")
    fig = px.scatter(cd, x="housing_distress_index", y="chronic_absence_ems", hover_name="district",
                     hover_data={"cd_name": True}, trendline="ols", color_discrete_sequence=[BLUE],
                     labels={"housing_distress_index": "Housing distress index",
                             "chronic_absence_ems": "Chronic absence, K-8"})
    fig.update_yaxes(tickformat=".0%"); fig.update_layout(height=480)
    st.plotly_chart(fig, width="stretch")
    t = m["temporary_housing"]; h = m["housing_plus_eni"]
    st.markdown(f"""
**Yes, but neighborhood buildings aren't the whole story.** Across districts the housing index and K-8 chronic
absence move together (Spearman ρ = {R['districts']['spearman']['housing_vs_chronic_absence_ems']}).
At the school level ({m['n_schools']:,} schools, standard errors clustered by district):

| Model | Effect | 95% CI | p |
|---|---|---|---|
| Neighborhood housing index alone (per 1 SD) | {m['housing_only']['coef_per_sd']:+.1%} | {m['housing_only']['ci95'][0]:+.1%} to {m['housing_only']['ci95'][1]:+.1%} | {m['housing_only']['p']:.3f} |
| Housing index, controlling for school Economic Need Index | {h['housing_coef_per_sd']:+.1%} | {h['housing_ci95'][0]:+.1%} to {h['housing_ci95'][1]:+.1%} | {h['housing_p']:.2f} |
| School's share of students in temporary housing (per 10 pts), controlling for ENI | {t['sth_coef_per_10pts']:+.1%} | {t['sth_ci95'][0]:+.1%} to {t['sth_ci95'][1]:+.1%} | {t['sth_p']:.3f} |

Once a school's own poverty level is known, the condition of the surrounding buildings adds nothing.
What still matters is how many of the school's **own students** are homeless or doubled up.
""")
    st.markdown("**Districts with more chronic absence than their schools' economic need predicts**")
    gap = cd.sort_values("absence_gap", ascending=False)[["district", "cd_name", "chronic_absence_ems",
                                                          "expected_absence", "absence_gap"]].head(10)
    st.dataframe(gap.style.format({"chronic_absence_ems": "{:.1%}", "expected_absence": "{:.1%}",
                                   "absence_gap": "{:+.1%}"}), hide_index=True, width="stretch")

with tab_jobs:
    st.subheader("Where is the city hiring?")
    st.caption(f"Live Jobs NYC postings as of {R['jobs']['snapshot_date']}. "
               f"{R['jobs']['resolved_share']:.0%} of postings resolved to a work borough.")
    b = boroughs.copy()
    fig = px.bar(b, x="postings_per_10k_units", y="borough", orientation="h",
                 color_discrete_sequence=[BLUE], text_auto=".1f",
                 labels={"postings_per_10k_units": "Postings per 10,000 homes", "borough": ""})
    fig.update_layout(height=320, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")
    st.dataframe(b[["borough", "postings", "positions", "entry_level_positions", "median_salary",
                    "share_units_in_top_quartile_distress"]].style.format(
        {"median_salary": "${:,.0f}", "share_units_in_top_quartile_distress": "{:.0%}",
         "positions": "{:,.0f}", "entry_level_positions": "{:,.0f}"}),
        hide_index=True, width="stretch")
    j = jobs.dropna(subset=["boro_code"]).assign(borough=lambda d: d["boro_code"].astype(int).map(BORO))
    pick = st.multiselect("Filter postings by borough", list(BORO.values()), default=["Bronx"])
    view = j[j["borough"].isin(pick)] if pick else j
    st.dataframe(view[["agency", "business_title", "career_level", "number_of_positions",
                       "annual_salary_mid", "work_location", "borough"]]
                 .sort_values("number_of_positions", ascending=False),
                 hide_index=True, width="stretch")
    st.markdown("**Why payroll isn't used for geography**")
    st.dataframe(payroll.head(12)[["agency_name", "headcount", "mean_base_salary", "share_coded_manhattan"]]
                 .style.format({"headcount": "{:,}", "mean_base_salary": "${:,.0f}",
                                "share_coded_manhattan": "{:.0%}"}),
                 hide_index=True, width="stretch")
    st.caption("FY2025 Citywide Payroll codes 100% of DOE teachers and paraprofessionals to Manhattan, "
               "so its work-borough field reflects administrative coding, not where people work.")

with tab_district:
    name = st.selectbox("Community district", cd.sort_values("cd")["district"] + " — " + cd.sort_values("cd")["cd_name"])
    row = cd[cd["district"] == name.split(" — ")[0]].iloc[0]
    a, b_, c = st.columns(3)
    a.metric("Housing distress index", f"{row['housing_distress_index']:+.2f}",
             help="0 = typical district; +1 = one standard deviation worse")
    b_.metric("Chronic absence, K-8", f"{row['chronic_absence_ems']:.1%}",
              f"{row['absence_gap']:+.1%} vs. expected", delta_color="inverse")
    c.metric("Residential units", f"{row['units']:,.0f}")
    d1, d2, d3 = st.columns(3)
    d1.metric("Heat/hot-water complaints per 1k", f"{row['heat_hw_per_1k']:.0f}")
    d2.metric("Class C violations per 1k", f"{row['class_c_per_1k']:.0f}")
    d3.metric("Evictions per 1k", f"{row['evictions_per_1k']:.1f}")
    s = schools[schools["cd"] == row["cd"]].sort_values("chronic_absence_rate", ascending=False)
    st.markdown(f"**{len(s)} public schools in this district**")
    st.dataframe(s[["dbn", "school_name", "school_type", "chronic_absence_rate", "eni_2122"]]
                 .style.format({"chronic_absence_rate": "{:.1%}", "eni_2122": "{:.0%}"}, na_rep="—"),
                 hide_index=True, width="stretch")

with tab_data:
    st.markdown((ROOT / "docs" / "METHODS.md").read_text())
    st.json(R["data_quality"])
