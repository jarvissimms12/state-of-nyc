"""Offline tests for the follow-up studies (no network)."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "studies"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem + "_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- Study 1: bus school codes -> DBN keys
@pytest.mark.parametrize("code,key", [("15879", ("15", "879")), ("15K879", ("15", "879")),
                                      ("C522", None), ("", None), ("0768", None)])
def test_bus_school_key(code, key):
    bus = load(STUDIES / "01_bus_delays_absence" / "run.py")
    assert bus.school_key(code) == key


def test_bus_incident_serving_two_schools_counts_for_both():
    bus = load(STUDIES / "01_bus_delays_absence" / "run.py")
    inc = pd.DataFrame({"busbreakdown_id": [1], "schools_serviced": ["28402,28600"]})
    rows = bus.explode_schools(inc)
    assert sorted(rows["key"]) == [("28", "402"), ("28", "600")]


# ---- Study 2: no look-ahead, disrupted years excluded
def test_early_warning_excludes_pandemic_transitions():
    ew = load(STUDIES / "02_absence_early_warning" / "run.py")
    rows = pd.DataFrame({"year": [2017, 2018, 2019, 2020, 2021, 2022],
                         "rate_next": [0.2] * 6})
    kept = ew.usable(rows, range(2015, 2024))["year"].tolist()
    assert kept == [2017, 2021, 2022]          # 2018->19, 2019->20, 2020->21 dropped


def test_precision_at_top_fraction():
    ew = load(STUDIES / "02_absence_early_warning" / "run.py")
    import numpy as np
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    s = np.arange(10, dtype=float)              # highest score = last item (a positive)
    assert ew.precision_at(y, s, 0.10) == 1.0


# ---- Study 3: pipeline recovers planted hospital effects on synthetic SPARCS data
def test_hospital_pipeline_finds_planted_outliers(tmp_path):
    d = STUDIES / "03_hospital_length_of_stay"
    csv = tmp_path / "syn.csv"
    subprocess.run([sys.executable, str(d / "make_synthetic.py"), str(csv), "60000"], check=True)
    los = load(d / "run.py")
    res = los.main(str(csv), str(tmp_path / "out"))
    h = pd.read_csv(tmp_path / "out" / "hospital_oe.csv")
    flagged = dict(zip(h.loc[h["flag"] != "as expected", "facility_id"], h.loc[h["flag"] != "as expected", "flag"]))
    assert flagged.get("H_LONG_1") == "longer than expected"
    assert flagged.get("H_LONG_2") == "longer than expected"
    assert flagged.get("H_SHORT") == "shorter than expected"
    assert len(flagged) <= 4                    # at most one false flag among 40 normal hospitals
    race = res["equity"]["race"]["risk_adjusted_oe"]
    assert all(abs(r["oe_ratio"] - 1) < 0.05 for r in race)   # no effect was planted


# ---- committed study outputs are internally consistent
@pytest.mark.parametrize("study,key", [("01_bus_delays_absence", "n_schools_model"),
                                       ("02_absence_early_warning", "rows"),
                                       ("04_shelter_and_schools", "daily_range")])
def test_study_results_exist(study, key):
    p = STUDIES / study / "results.json"
    if not p.exists():
        pytest.skip("run the study first")
    assert key in json.loads(p.read_text())
