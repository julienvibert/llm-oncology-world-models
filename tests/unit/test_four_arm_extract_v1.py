"""Gate-4 cohort orchestration: response binding, eligibility masks, pairing, fixed
adjustment families (step-3 gates R2-R3). No patient data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
sys.path.insert(0, str(ROOT / "src"))

import four_arm_extract_v1 as X  # noqa: E402
import score_four_arm_final_v1 as F  # noqa: E402


def _rec(first_ok, used, accepted, first_raw, final_raw):
    detail = [{"accepted": first_ok, "raw_response": first_raw}]
    detail += [{"accepted": accepted, "raw_response": final_raw}] * (used - 1)
    return {"attempts_used": used, "accepted": accepted, "raw_response": final_raw,
            "attempts_detail": detail}


def test_valid_first_attempt_must_be_the_saved_response() -> None:
    assert F.first_attempt_bound(_rec(True, 1, True, "x", "x")) == "valid"
    with pytest.raises(F.InputError):
        F.first_attempt_bound(_rec(True, 1, True, "x", "y"))          # different text
    with pytest.raises(F.InputError):
        F.first_attempt_bound(_rec(True, 2, True, "x", "x"))          # valid but retried


def test_retry_success_and_missing_metadata_are_not_first_valid() -> None:
    assert F.first_attempt_bound(_rec(False, 2, True, "x", "y")) == "invalid"
    missing_flag = {"attempts_used": 1, "accepted": True, "raw_response": "x",
                    "attempts_detail": [{"raw_response": "x"}]}
    assert F.first_attempt_bound(missing_flag) == "unavailable"


def test_holm_family_keeps_its_size_with_unavailable_slots() -> None:
    slots = [(h, c) for h in (3.0, 6.0, 12.0, 24.0) for c in ("2-1", "3-2", "4-3", "4-1")]
    pv = {s: 0.01 for s in slots[:3]} | {slots[3]: None}           # 13 slots absent, 1 None
    adj = X.adjust_family(pv, slots)
    assert len(adj) == 16 and adj[slots[3]] is None and adj[slots[5]] is None
    assert adj[slots[0]] == pytest.approx(0.16)                     # m = 16, not 3
    donor = [(c, h) for c in ("generic", "matched") for h in (3.0, 6.0, 12.0, 24.0)]
    adj_d = X.adjust_family({donor[0]: 0.001}, donor)
    assert adj_d[donor[0]] == pytest.approx(0.008)


def test_adjustment_uses_unrounded_p_values() -> None:
    slots = ["a", "b"]
    tiny = X.adjust_family({"a": 3e-9, "b": 0.5}, slots)
    assert tiny["a"] == pytest.approx(6e-9) and tiny["a"] > 0
    assert X._r(3e-9) == 3e-9                                      # no display rounding


def test_eligibility_is_compared_in_both_directions() -> None:
    saved = {("q1", 3.0, "primary"): True, ("q2", 3.0, "primary"): False,
             ("q3", 3.0, "primary"): True}
    derived = {("q1", 3.0, "primary"): True, ("q2", 3.0, "primary"): True,
               ("q3", 3.0, "primary"): False}
    mm = X.eligibility_mismatches(saved, derived)
    assert mm["saved_only"] == [("q3", 3.0, "primary")]
    assert mm["derived_only"] == [("q2", 3.0, "primary")]
    assert X.eligibility_mismatches(saved, saved) == {"saved_only": [], "derived_only": []}


def test_patients_are_paired_across_all_arms_or_refused() -> None:
    qids, mat = X.paired_matrix({"b": [1, 2, 3, 4], "a": [5, 6, 7, 8]})
    assert qids == ["a", "b"] and np.array_equal(mat[0], [5, 6, 7, 8])
    with pytest.raises(ValueError):
        X.paired_matrix({"a": [1, 2, None, 4]})
    with pytest.raises(ValueError):
        X.paired_matrix({"a": [1, 2, 3]})


def _row(h, contrast, p, quantity="equal_size", conv="primary", cancer="all"):
    return {"horizon_months": h, "row": f"contrast:{contrast}", "quantity": quantity,
            "convention": conv, "cancer": cancer, "p_wilcoxon": p}


def test_contrast_rows_feed_the_fixed_holm_family() -> None:
    """From contrast-table rows through Holm: other quantities, conventions and cancers are
    ignored; a missing P stays unavailable but counts in the 16-slot family."""
    slots = [(h, c) for h in (3.0, 6.0, 12.0, 24.0) for c in ("2-1", "3-2", "4-3", "4-1")]
    rows = [_row(h, c, 0.5) for h, c in slots]
    rows[0]["p_wilcoxon"] = 2e-5                      # small, must not be rounded away
    rows[1]["p_wilcoxon"] = ""                         # unavailable
    rows += [_row(3.0, "2-1", 1e-9, quantity="as_delivered"),
             _row(3.0, "2-1", 1e-9, conv="sensitivity"), _row(3.0, "2-1", 1e-9, cancer="BRCA")]
    n = X.assign_joint_holm(rows, slots)
    assert n == 15
    assert rows[0]["p_holm_primary_family"] == pytest.approx(16 * 2e-5)
    assert "p_holm_primary_family" in rows[1] and rows[1]["p_holm_primary_family"] == ""
    assert all("p_holm_primary_family" not in r for r in rows[16:])


def test_precheck_grid_must_be_complete_unique_and_boolean(tmp_path) -> None:
    import pandas as pd
    q = {"q1", "q2"}
    base = pd.DataFrame([{"query_id": a, "horizon_months": h, "convention": c,
                          "four_arm_intersection": True}
                         for a in sorted(q) for h in (3.0, 6.0, 12.0, 24.0)
                         for c in ("primary", "sensitivity")])
    f = tmp_path / "sets.csv"
    base.to_csv(f, index=False)
    assert len(X.load_precheck_sets(f, q)) == 16
    base.iloc[:-1].to_csv(f, index=False)
    with pytest.raises(RuntimeError):
        X.load_precheck_sets(f, q)                                    # incomplete
    bad = base.copy()
    bad["four_arm_intersection"] = bad["four_arm_intersection"].astype(object)
    bad.loc[0, "four_arm_intersection"] = None
    bad.to_csv(f, index=False)
    with pytest.raises(RuntimeError):
        X.load_precheck_sets(f, q)                                    # missing flag
    pd.concat([base, base.iloc[:1]]).to_csv(f, index=False)
    with pytest.raises(RuntimeError):
        X.load_precheck_sets(f, q)                                    # duplicate
