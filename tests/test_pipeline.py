"""Offline tests: no network. They pin down the decisions that would silently
change the results if they broke."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build import housing_distress_index  # noqa: E402
from etl.jobs import annual_salary, resolve_pluto  # noqa: E402
from etl.schools import to_school_level, weighted  # noqa: E402
from src.geo import (VALID_CDS, borough_from_address_text, cd_code,  # noqa: E402
                     normalize_cd, parse_311_board, street_address_key, street_name)


# ---------------------------------------------------------------- geography
def test_there_are_59_community_districts():
    assert len(VALID_CDS) == 59
    assert {101, 112, 212, 318, 414, 503} <= VALID_CDS
    assert not {164, 226, 355, 480, 595} & VALID_CDS      # parks / airports


@pytest.mark.parametrize("raw,expected", [
    ("07 BROOKLYN", 307), ("12 MANHATTAN", 112), ("01 STATEN ISLAND", 501),
    ("0 Unspecified", None), ("64 MANHATTAN", None), ("Unspecified QUEENS", None), (None, None),
])
def test_parse_311_board(raw, expected):
    assert parse_311_board(raw) == expected


def test_cd_code_accepts_codes_and_names():
    assert cd_code(2, "5") == 205
    assert cd_code("BRONX", 5) == 205
    assert cd_code("3", "18") == 318
    assert cd_code(3, 19) is None
    assert normalize_cd("307") == 307 and normalize_cd(307.0) == 307 and normalize_cd("164") is None


@pytest.mark.parametrize("addr,expected", [
    ("30-30 Thomson Ave L I City Qns", 4),
    ("59-17 Junction Blvd Corona Ny", 4),
    ("198 E161St Street Bronx N.Y.", 2),
    ("Jamaica Ave, Bronx", 2),                 # explicit borough beats neighborhood
    ("1075 Ralph Avenue Bklyn, N.Y.", 3),      # Brooklyn beats the N.Y. postal city
    ("2 Metro Tech", 3),
    ("1 Bay St., S.I.,Ny", 5),
    ("455 First Ave., N.Y.", 1),
    ("Rikers Island", 2),
    ("NYC - All Boroughs", "multiple"),
    ("Queens-SI Floating Staff", "multiple"),
    ("465 Columbus Ave. Valhalla, Ny", "outside"),
    ("160 West 100Th Street Ny", None),        # bare "NY" says nothing about borough
    ("100 Gold Street", None),
])
def test_borough_from_address_text(addr, expected):
    assert borough_from_address_text(addr) == expected


def test_street_address_key_matches_pluto_style():
    assert street_address_key("100 Gold St., N.Y.") == "100 GOLD STREET"
    assert street_address_key("42-09 28th Street") == "42-09 28 STREET"
    assert street_address_key("30-30 Thomson Ave L I City Qns") == "30-30 THOMSON AVENUE"
    assert street_address_key("City Hall") is None
    assert street_name("125 WORTH STREET") == "WORTH STREET"


def test_resolve_pluto_tie_breaks_by_building_size():
    both = pd.DataFrame({"boro_code": [1, 3], "bldgarea": [1_800_000, 20_000]})
    assert resolve_pluto(both) == (1, "pluto_largest_building")
    close = pd.DataFrame({"boro_code": [1, 3], "bldgarea": [50_000, 40_000]})
    assert resolve_pluto(close) == (None, "pluto_ambiguous")
    assert resolve_pluto(pd.DataFrame({"boro_code": [4], "bldgarea": [10]})) == (4, "pluto_exact")


def test_annual_salary_midpoint():
    df = pd.DataFrame({"salary_range_from": [50000, 20.0], "salary_range_to": [70000, 30.0],
                       "salary_frequency": ["Annual", "Hourly"]})
    assert annual_salary(df).tolist() == [60000, 25.0 * 1820]


# ---------------------------------------------------------------- schools
def _sqr(rows):
    return pd.DataFrame(rows, columns=["dbn", "school_name", "school_type", "measure",
                                       "number_of_students", "metric_value"])


def test_chronic_absence_is_inverted_from_sqr_metric():
    """SQR 'chronic_absent' = share with >90% attendance. 0.70 -> 30% chronically absent."""
    sqr = _sqr([("01M001", "A", "Elementary", "pct90_ems", 100, 0.70),
                ("01M001", "A", "Elementary", "pct90_hs", np.nan, np.nan)]).dropna(subset=["metric_value"])
    s = to_school_level(sqr)
    assert s.loc[0, "chronic_absence_ems"] == pytest.approx(0.30)
    assert s.loc[0, "chronic_absence_rate"] == pytest.approx(0.30)


def test_six_to_twelve_school_combines_ems_and_hs_by_students():
    sqr = _sqr([("02M100", "B", "Secondary School", "pct90_ems", 100, 0.80),   # 20% absent
                ("02M100", "B", "Secondary School", "pct90_hs", 300, 0.60)])   # 40% absent
    s = to_school_level(sqr)
    assert s.loc[0, "chronic_absence_ems"] == pytest.approx(0.20)
    assert s.loc[0, "chronic_absence_hs"] == pytest.approx(0.40)
    assert s.loc[0, "chronic_absence_rate"] == pytest.approx((0.2 * 100 + 0.4 * 300) / 400)
    assert s.loc[0, "n_chronic"] == 400


def test_weighted_district_average():
    df = pd.DataFrame({"cd": [101, 101, 102], "v": [0.1, 0.5, 0.3], "w": [300, 100, 0]})
    out = weighted(df, "v", "w")
    assert out.loc[101] == pytest.approx(0.2)
    assert 102 not in out.index                      # zero weight -> no estimate


# ---------------------------------------------------------------- index
def test_housing_distress_index_is_centered_and_ordered():
    df = pd.DataFrame({"heat_hw_per_1k": [10, 50, 200], "class_c_per_1k": [5, 40, 300],
                       "evictions_per_1k": [1, 3, 12]})
    idx = housing_distress_index(df)
    assert idx.mean() == pytest.approx(0, abs=1e-12)
    assert list(idx.rank()) == [1, 2, 3]


def test_housing_distress_index_weights_measures_equally():
    """Scaling one rate by 100x must not change the index (z-scores of logs)."""
    df = pd.DataFrame({"heat_hw_per_1k": [10, 50, 200], "class_c_per_1k": [5, 40, 300],
                       "evictions_per_1k": [1, 3, 12]})
    scaled = df.assign(heat_hw_per_1k=df["heat_hw_per_1k"] * 100)
    assert np.allclose(housing_distress_index(df), housing_distress_index(scaled))


# ---------------------------------------------------------------- outputs
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


@pytest.mark.skipif(not (PROCESSED / "cd_profile.csv").exists(), reason="run build.py first")
def test_processed_outputs_are_consistent():
    cd = pd.read_csv(PROCESSED / "cd_profile.csv")
    assert len(cd) == 59 and cd["cd"].is_unique and set(cd["cd"]) == set(VALID_CDS)
    assert cd["chronic_absence_ems"].between(0, 1).all()
    assert (cd["units"] > 0).all()
    schools = pd.read_csv(PROCESSED / "schools.csv")
    k8 = schools.dropna(subset=["chronic_absence_ems", "n_chronic_ems"])
    citywide = np.average(k8["chronic_absence_ems"], weights=k8["n_chronic_ems"])
    assert 0.2 < citywide < 0.45        # inverted metric would land near 0.7
