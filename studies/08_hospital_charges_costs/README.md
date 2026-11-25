# Study 8 — What NYC hospitals charge vs. what care costs them

> **Status: pipeline built and verified; real-data run pending.** It uses the same
> SPARCS file as Study 3, from New York State's Health Data NY portal, which the build
> environment couldn't reach. Download it once and run Studies 3, 8 and 9B.

## Summary

| | |
|---|---|
| **Question** | How much do NYC hospitals mark up their list prices over the cost of care, and which hospitals spend more than their patient mix explains? |
| **Data** | NYS SPARCS de-identified inpatient discharges: every stay has **Total Charges** (the hospital's billed list price) and **Total Costs** (SPARCS's estimate of what providing the stay cost, from each hospital's cost reports). Filtered to the New York City health service area, newborns excluded. |
| **Method** | (1) **Markup** = total charges ÷ total costs, per hospital and per payer. (2) **Case-mix-adjusted cost index**: a hospital's costs ÷ what the same APR-DRG × severity mix would cost at the citywide average, with 99.8% funnel limits. (3) The same index for charges. The benchmark is built in **two passes** so outlier hospitals don't distort it. |
| **Result** | Pending the real-data run. On synthetic data with planted effects, the pipeline finds the hospital planted with a 2× markup (6.0 vs 3.0), flags the one planted to cost 40% more per case (index 1.41), correctly flags the long- and short-stay hospitals from Study 3 (cost tracks length of stay), and falsely flags 1 of 40 normal hospitals. |
| **Takeaway** | Charges are list prices that few patients pay, so markup mostly tells you about a hospital's chargemaster. The cost index is the efficiency signal: who spends more for the same kinds of patients. Hospital finance and health policy teams use both. |

## Run it

```bash
python studies/08_hospital_charges_costs/run.py --data path/to/Hospital_Inpatient_Discharges.csv
```

Outputs in `results/`: `results.json`, `hospital_charges_costs.csv` (each hospital's
markup, cost index with limits, charge index) and `charges_vs_costs.png`.

## A methodology bug the synthetic test caught

With a one-pass benchmark, the planted expensive hospitals (about a tenth of all stays) pulled
the citywide average cost up, so **5 normal hospitals** were flagged as cheaper than
expected (index 0.93–0.95). The two-pass benchmark flags outliers, rebuilds the
average from everyone else, then scores all hospitals. That cut the false flags to
1 of 40, which is what 99.8% limits should allow. `tests/test_studies.py` reruns the
check.

## Caveats to state with the real results

- **Costs are estimates.** SPARCS derives them from charges and each hospital's
  cost-to-charge ratios in its cost reports, so markup partly reflects how those
  reports are prepared.
- **Charges aren't prices paid.** Insurers pay negotiated rates and Medicaid and
  Medicare pay set rates. For what hospitals actually receive, see hospital
  price-transparency files.
- **Case mix is APR-DRG × severity.** Teaching hospitals and trauma centers may
  treat sicker patients within the same group, which pushes their cost index up.

## Files

- `run.py`: loads SPARCS, computes markups and adjusted indices, writes results and chart
- Synthetic test data: `../03_hospital_length_of_stay/make_synthetic.py`
