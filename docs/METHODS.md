## Data & methods

**Time window.** NYC fiscal year 2025, July 1, 2024 to June 30, 2025. This matches the 2024-25 school year, so every housing and school measure describes the same twelve months. Job postings are a live snapshot taken on the build date.

**Unit of analysis.** NYC's 59 community districts (CDs), plus individual public schools for the school-level models. Joint interest areas (Central Park, the airports, large parks) have almost no residents and are excluded.

### Sources (all NYC Open Data)

| Measure | Dataset | ID |
|---|---|---|
| Heat / hot-water complaints | 311 Service Requests | `erm2-nwe9` |
| Class A/B/C housing violations | HPD Housing Maintenance Code Violations | `wvxf-dwi5` |
| Residential evictions | Evictions (City Marshals) | `6z8x-wfk4` |
| Residential units (denominator) | PLUTO | `64uk-42ks` |
| Chronic absence, attendance, graduation | DOE School Quality Reports | `dnpx-dfnc` |
| School → community district | 2019-20 School Locations | `wg9x-4ke6` |
| Economic Need Index (2021-22) | 2017-18 to 2021-22 Demographic Snapshot | `c7ru-d68s` |
| Students in temporary housing (2020-21) | 2021 Students in Temporary Housing | `3wtp-43m9` |
| City job openings | Jobs NYC Postings | `kpav-sd4t` |
| City workforce & pay | Citywide Payroll Data (FY2025) | `k397-673e` |
| Boundaries | Community Districts | `5crt-au7u` |

### Measures

- **Rates per 1,000 residential units** (PLUTO `unitsres`). Units are the right denominator for building conditions; population would penalize districts with large households.
- **Housing distress index:** the mean of the z-scores of the log of three rates (heat/hot-water complaints, Class C violations, residential evictions). Logs because the rates are right-skewed (heat complaints range 6–275 per 1,000 units); z-scores so each counts equally. 0 is a typical district, +1 is one standard deviation worse.
- **Chronic absence** is missing 10% or more of enrolled school days. **Data trap:** the School Quality Reports field `chronic_absent_*` is labeled "Percentage of Students with >90% Attendance". It is the share *not* chronically absent, so the rate is `1 − value`. Citywide K-8 comes out at 30.0% and all grades at 31.7%, which is the right magnitude.
- **K-8 vs. all grades.** Elementary and middle schools mostly serve their neighborhood, but high school students commute citywide. District-level comparisons use K-8 chronic absence; high school figures are kept in `schools.csv`.

### Models

School-level weighted least squares, weighted by the number of students behind each school's rate, with standard errors clustered by community district (schools in the same district share the same housing measure):

1. `chronic_absence ~ housing_index`
2. `chronic_absence ~ housing_index + ENI`
3. `chronic_absence ~ ENI + temporary_housing_share` (1,044 schools with unsuppressed counts)

District "absence gap": the difference between a district's K-8 chronic absence and what a district-level regression on ENI predicts.

### Job posting geocoding

Jobs NYC gives a free-text work address. Each unique job (internal and external listings deduplicated) is assigned a borough by, in order: explicit borough or neighborhood names; exact match to a PLUTO tax-lot address (ties between boroughs broken by building floor area when one is 5x larger); a street name that exists in only one borough; and the NYC convention that hyphenated house numbers (`30-30 47th Ave`) are Queens addresses. Unresolved, multi-borough and out-of-city postings are counted in `data_quality.json`, never guessed.

### Limitations

- **Association, not causation.** Districts are large (≈60,000 homes); this is ecological data.
- **ENI and temporary housing are lagged** (2021-22 and 2020-21, the latest school-level releases on Open Data). A school's composition shifts slowly, but this understates any recent change, such as shelter placements from 2022 on.
- **Suppressed counts.** Schools with small temporary-housing counts are suppressed (`s`) and drop out of model 3. These are mostly small schools.
- **Complaints measure reporting.** 311 rates reflect willingness and ability to call, as well as conditions.
- **School locations are from 2019-20.** 37 of 1,680 schools (2.2%) opened later or moved and couldn't be placed in a district.
- **Payroll geography is administrative.** FY2025 payroll codes every DOE teacher and paraprofessional to Manhattan, so it is used for headcount and pay only.
- **Job postings are a snapshot** and count city-government openings only, not private employment.
