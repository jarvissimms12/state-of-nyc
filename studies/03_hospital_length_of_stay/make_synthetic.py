"""Synthetic SPARCS-shaped file with KNOWN effects, used to verify run.py.

Planted truth:
  * LOS depends on DRG and severity (the benchmark should capture most of it)
  * two hospitals run 30% long (H_LONG_1, H_LONG_2), one runs 25% short (H_SHORT)
  * Medicaid stays run 10% long within the same DRG x severity
  * race has NO effect

A correct pipeline should flag exactly those hospitals, show Medicaid O/E
above 1 and race O/E near 1. This is a test of the code, not a finding.

    python studies/03_hospital_length_of_stay/make_synthetic.py synthetic.csv [rows]
"""
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 120_000
hospitals = [f"H{i:03d}" for i in range(40)] + ["H_LONG_1", "H_LONG_2", "H_SHORT"]
h_effect = {h: 1.0 for h in hospitals} | {"H_LONG_1": 1.3, "H_LONG_2": 1.3, "H_SHORT": 0.75}
drgs = {f"{d:03d}": rng.uniform(2, 9) for d in range(1, 61)}
pay = ["Medicare", "Medicaid", "Private Health Insurance", "Self-Pay"]
races = ["White", "Black/African American", "Other Race", "Multi-racial"]

h = rng.choice(hospitals, N, p=np.r_[np.full(40, 0.9 / 40), [0.04, 0.03, 0.03]])
drg = rng.choice(list(drgs), N)
sev = rng.choice(["1", "2", "3", "4"], N, p=[0.35, 0.35, 0.22, 0.08])
payer = rng.choice(pay, N, p=[0.38, 0.32, 0.26, 0.04])
race = rng.choice(races, N, p=[0.4, 0.3, 0.25, 0.05])
mu = (np.array([drgs[d] for d in drg]) * np.array([1, 1.4, 2.1, 3.5])[sev.astype(int) - 1]
      * np.array([h_effect[x] for x in h]) * np.where(payer == "Medicaid", 1.10, 1.0))
los = np.clip(np.round(rng.gamma(shape=2.5, scale=mu / 2.5)), 1, 120).astype(int)

df = pd.DataFrame({
    "Hospital Service Area": np.where(rng.random(N) < 0.85, "New York City", "Long Island"),
    "Permanent Facility Id": h, "Facility Name": [f"Hospital {x}" for x in h],
    "Age Group": rng.choice(["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"], N),
    "Gender": rng.choice(["F", "M"], N), "Race": race,
    "Ethnicity": rng.choice(["Not Span/Hispanic", "Spanish/Hispanic", "Unknown"], N, p=[0.7, 0.25, 0.05]),
    "Length of Stay": np.where(los >= 120, "120 +", los.astype(str)),
    "Type of Admission": rng.choice(["Emergency", "Elective", "Urgent", "Newborn"], N, p=[0.6, 0.2, 0.15, 0.05]),
    "APR DRG Code": drg, "APR DRG Description": [f"DRG {d}" for d in drg],
    "APR Severity of Illness Code": sev,
    "APR Risk of Mortality": rng.choice(["Minor", "Moderate", "Major", "Extreme"], N),
    "APR Medical Surgical Description": rng.choice(["Medical", "Surgical"], N, p=[0.75, 0.25]),
    "Payment Typology 1": payer, "Emergency Department Indicator": rng.choice(["Y", "N"], N),
    "CCSR Diagnosis Code": rng.choice([f"CIR{i:03d}" for i in range(30)], N),
})
df.to_csv(sys.argv[1] if len(sys.argv) > 1 else "synthetic.csv", index=False)
print(f"wrote {len(df):,} rows")
