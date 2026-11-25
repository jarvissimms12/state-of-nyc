# Study 3 — Hospital length of stay in NYC

> **Status: pipeline built and verified; real-data run pending.** The SPARCS file lives on
> New York State's Health Data NY portal, which the build environment used for this repo
> couldn't reach. Running it yourself takes a few minutes (see below), and `results/`
> will then hold the real numbers.

## Summary

| | |
|---|---|
| **Question** | How long do patients stay in NYC hospitals, which hospitals run consistently longer or shorter than their case mix predicts, and do stays differ by race, ethnicity or payer for the same condition and severity? |
| **Data** | NYS SPARCS de-identified inpatient discharges (Health Data NY): every inpatient stay in the state, ~2.1M a year. Filtered to the New York City health service area; newborn admissions excluded. |
| **Method** | (1) Predict length of stay with gradient boosting vs. the standard benchmark (median stay for the same APR-DRG × severity), plus a classifier for "extended stays" (above the 75th percentile for the case type). (2) **Hospital benchmarking:** observed ÷ expected days per hospital (indirect standardization), with 99.8% funnel limits so a hospital is flagged only when the gap isn't noise. (3) **Equity:** risk-adjusted observed ÷ expected days by race, ethnicity and payer, and the model's error for each group. |
| **Result** | Pending the real-data run. On synthetic data with known effects, the pipeline flags exactly the 2 hospitals planted to run 30% long and the 1 planted to run 25% short, none of the 40 normal hospitals, finds the planted Medicaid effect, and shows no race effect where none was planted. |
| **Takeaway** | Hospital operations, quality and population-health analysts do exactly this work: case-mix-adjusted length-of-stay benchmarking with honest uncertainty, plus an equity check. |

## Run it

1. Go to [health.data.ny.gov](https://health.data.ny.gov) and search **"Hospital Inpatient Discharges (SPARCS De-Identified)"**.
2. Open the most recent year and choose **Export → CSV** (about 2.1 million rows, several hundred MB).
3. From the repo root:

```bash
pip install -r requirements.txt
python studies/03_hospital_length_of_stay/run.py --data path/to/Hospital_Inpatient_Discharges.csv
```

Outputs go to `studies/03_hospital_length_of_stay/results/`: `results.json`,
`hospital_oe.csv` (every NYC hospital's observed/expected ratio and limits) and
`hospital_funnel.png`. Commit the `results/` folder and fill the Result row above
from `results.json`.

Column names are matched flexibly across SPARCS years (e.g. CCS vs. CCSR diagnosis
codes). If a required column is missing, the script stops and lists what it found.

## Two methodology bugs the synthetic test caught

The synthetic data (`make_synthetic.py`) has **known** effects, so it shows whether the
method is right, not just whether the code runs:

1. **Median vs. mean for "expected."** Using the median length of stay for each case
   type as "expected" made *every* hospital and *every* racial group look ~15% long,
   because stays are right-skewed (mean > median). Observed/expected ratios now use
   the mean; the median is kept only as the prediction baseline, where it's the right
   choice.
2. **Multiple comparisons.** Testing 43 hospitals at 95% falsely flagged 7 normal ones.
   Hospital benchmarking now uses 99.8% funnel limits, the usual standard when many
   providers are compared at once, and flags only the 3 planted hospitals.

`tests/test_studies.py` reruns this check on every push.

## Caveats to state when results are in

- **APR-DRG and severity are assigned at discharge**, so this describes stays rather
  than predicting them at admission. An admission-time model would use the admitting
  diagnosis instead.
- **Observed/expected doesn't measure quality.** A long stay can mean sicker patients
  than the codes capture, or slow discharge because patients have nowhere safe to go,
  which links back to the housing findings in this repo.
- **De-identification** removes some detail (e.g. 3-digit ZIPs, top-coded stays at 120+ days).

## Files

- `run.py`: loads SPARCS, models, benchmarks hospitals, runs the equity checks, draws the funnel plot
- `make_synthetic.py`: SPARCS-shaped test data with planted effects
