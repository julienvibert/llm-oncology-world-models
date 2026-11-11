"""Small-cell masking of Source Data (scripts/release/mask_small_cells.py): units, complements, partitions."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "release"))
import mask_small_cells as MS  # noqa: E402


def test_percentage_unit_is_declared_not_inferred():
    # 0.1% of 1,000 patients is one patient: must be masked although the value is below 1
    d = pd.DataFrame({"n": [1000, 1000], "deaths_pct": [0.1, 20.0]})
    spec = MS.Spec(counts=("n",), shares={"deaths_pct": ("pct", "n")})
    out, k = MS.mask(d, spec)
    assert k == 1 and pd.isna(out.loc[0, "deaths_pct"]) and out.loc[1, "deaths_pct"] == 20.0
    assert MS.verify(d, out, spec) == []
    # the same table released unmasked fails verification
    assert MS.verify(d, d, spec)


def test_undeclared_share_column_fails():
    d = pd.DataFrame({"n": [100], "share_alive": [0.5]})
    with pytest.raises(ValueError, match="not declared"):
        MS.mask(d, MS.Spec(counts=("n",)))


def test_unit_mismatch_fails():
    d = pd.DataFrame({"n": [100], "share": [45.0]})
    with pytest.raises(ValueError, match="unit"):
        MS.mask(d, MS.Spec(counts=("n",), shares={"share": ("frac", "n")}))


def test_complement_is_masked():
    d = pd.DataFrame({"n_group": [50, 50], "n_alive": [48, 30]})
    spec = MS.Spec(counts=("n_alive", "n_group"), denoms=("n_group",))
    out, k = MS.mask(d, spec)
    assert k == 1 and pd.isna(out.loc[0, "n_alive"])
    assert MS.verify(d, out, spec) == []


def test_single_blank_in_partition_is_detected_and_prevented():
    # total 20 split [1, 5, 14]: blanking only the 1 leaves it recoverable as 20 - 5 - 14
    d = pd.DataFrame({"g": ["a"] * 3, "state": ["x", "y", "z"], "n_group": [20] * 3, "n_patients": [1, 5, 14]})
    spec = MS.Spec(counts=("n_patients", "n_group"), denoms=("n_group",), partition=(("g",), ("n_patients",)))
    naive = d.astype(object).copy()
    naive.loc[0, "n_patients"] = np.nan
    assert any("subtraction" in p for p in MS.verify(d, naive, spec))
    out, k = MS.mask(d, spec)
    assert k == 2 and out.n_patients.isna().sum() == 2
    assert MS.verify(d, out, spec) == []


def test_ribbon_partition_and_implied_counts():
    months = ["3", "6"]
    shown = pd.DataFrame({"3": [np.nan, 50.0, 30.0], "6": [40.0, 40.0, 20.0], "group": "arm"},
                         index=["CR or PR", "Stable", "Progressive"])
    den = pd.DataFrame({"3": [20.0, 20.0, 20.0], "6": [10.0, 10.0, 10.0], "group": "arm"}, index=shown.index)
    problems = MS.verify_ribbon(shown, den, ["arm"], ("CR or PR", "Stable", "Progressive"))
    assert any("subtraction" in p for p in problems)  # month 3: one blank, recoverable
    assert any("implying 1 to 4" in p for p in problems)  # month 6: 40% and 20% of 10 are 4 and 2 patients
    assert set(months) <= set(shown.columns)


def test_share_checked_against_its_own_denominator():
    # occupancy table: share = state count / n_patients (followed that month), not / n_group (starting cohort).
    # CRC month 24, responding: 0.030534 x 131 = 4 patients (x 149 would suggest 4.55 and pass)
    d = pd.DataFrame({"g": ["a", "a"], "state": ["responding", "stable"], "share": [0.030534, 0.969466],
                      "n_patients": [131, 131], "n_censored": [18, 18], "n_group": [149, 149]})
    right = MS.Spec(counts=("n_patients", "n_censored", "n_group"), denoms=("n_group",),
                    shares={"share": ("frac", "n_patients")}, partition=(("g",), ("share",)))
    assert MS.verify(d, d, right)  # unmasked: flagged
    out, k = MS.mask(d, right)
    assert pd.isna(out.loc[0, "share"]) and out.loc[0, "suppressed_columns"] == "share"
    assert MS.verify(d, out, right) == []


def test_only_rows_of_real_patients_are_masked():
    d = pd.DataFrame({"source": ["observed", "observed", "model", "model"], "g": ["a", "a", "b", "b"],
                      "share": [0.02, 0.98, 0.02, 0.98], "n_patients": [200, 200, 200, 200]})
    spec = MS.Spec(counts=("n_patients",), shares={"share": ("frac", "n_patients")},
                   partition=(("source", "g"), ("share",)), real_rows=("source", "observed"))
    out, k = MS.mask(d, spec)
    assert k == 2 and out.share.isna().tolist() == [True, True, False, False]
    assert MS.verify(d, out, spec) == []
