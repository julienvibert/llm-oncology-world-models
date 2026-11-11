from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import build_perturbation_matrix_decision_aligned as dam  # noqa: E402


def _course(days: list[float], pc1: list[float], extent: list[float] | None = None,
            ecog: list[float] | None = None, tox: list[float] | None = None) -> pd.DataFrame:
    n = len(days)
    return pd.DataFrame({
        "days_from_diagnosis": days,
        "snapshot_index": list(range(n)),
        "pc1": pc1,
        "tumour_extent": extent or [0.0] * n,
        "host_ecog": ecog or [0.0] * n,
        "host_active_ae_max_grade": tox or [0.0] * n,
    })


def test_decision_at_day_123_is_measured_from_day_123() -> None:
    # Diagnosis day 0, prefix progression day 122, decision day 123, change day 150.
    course = _course([0, 122, 123, 150], [0.0, 1.0, 1.5, 3.0])
    out = dam.decision_window_readouts(course, 123.0, death_days=2000.0)
    assert out["anchor_day"] == 123.0
    assert out["horizon_day"] == pytest.approx(123.0 + 3 * 30.4)
    assert out["d_severity"] == pytest.approx(1.5)  # 3.0 - 1.5, not 3.0 - 0.0


def test_change_before_the_decision_is_not_counted() -> None:
    # Every change happens before the decision; nothing changes after it.
    course = _course([0, 60, 123], [0.0, 2.0, 2.0])
    out = dam.decision_window_readouts(course, 123.0, death_days=2000.0)
    assert out["d_severity"] == 0.0


def test_diagnosis_window_would_miss_a_post_decision_change() -> None:
    # A change at day 150 lies outside 0..91 days but inside 123..214 days.
    course = _course([0, 123, 150], [0.0, 0.0, 1.0])
    assert dam.decision_window_readouts(course, 0.0, 2000.0)["d_severity"] == 0.0
    assert dam.decision_window_readouts(course, 123.0, 2000.0)["d_severity"] == 1.0


def test_horizon_uses_locf_and_ignores_later_states() -> None:
    course = _course([0, 123, 200, 220], [0.0, 0.0, 1.0, 5.0])  # 220 > 123 + 91.2
    out = dam.decision_window_readouts(course, 123.0, death_days=2000.0)
    assert out["d_severity"] == 1.0
    assert out["horizon_state_day"] == 200.0


def test_same_day_tie_takes_the_last_snapshot() -> None:
    course = _course([0, 123, 123], [0.0, 1.0, 2.0])
    out = dam.decision_window_readouts(course, 123.0, death_days=2000.0)
    assert out["anchor_pc1"] == 2.0


def test_death_within_three_months_of_decision_exits() -> None:
    course = _course([0, 123, 150], [0.0, 0.0, 4.0], tox=[0.0, 0.0, 3.0])
    out = dam.decision_window_readouts(course, 123.0, death_days=180.0)
    assert out["exited"] is True
    assert np.isnan(out["d_severity"]) and np.isnan(out["d_host"])
    assert out["tox"] == 3.0  # toxicity before death is still read


def test_death_after_horizon_does_not_exit() -> None:
    course = _course([0, 123], [0.0, 0.0])
    out = dam.decision_window_readouts(course, 123.0, death_days=123.0 + 92.0)
    assert out["exited"] is False


def test_toxicity_before_the_decision_is_not_counted() -> None:
    course = _course([0, 50, 123, 150], [0.0] * 4, tox=[0.0, 4.0, 0.0, 2.0])
    out = dam.decision_window_readouts(course, 123.0, death_days=2000.0)
    assert out["tox"] == 2.0


def test_missing_anchor_raises() -> None:
    course = _course([10, 20], [0.0, 1.0])
    with pytest.raises(ValueError):
        dam.decision_window_readouts(course, 5.0, death_days=2000.0)


def test_endpoints_are_converted_to_decision_origin() -> None:
    e = dam.post_decision_endpoints(123.0 + 300.0, 123.0, 200.0, 123.0)
    assert e["os"] == pytest.approx(300.0 / 30.4)
    assert e["exit12"] == 1.0
    assert e["pfs1"] == pytest.approx(200.0 / 30.4)
    late = dam.post_decision_endpoints(123.0 + 400.0, 123.0, np.nan, np.nan)
    assert late["exit12"] == 0.0 and np.isnan(late["pfs1"])


def test_death_by_12_months_counts_from_the_decision() -> None:
    # Death at day 400 after diagnosis but only 277 days after a day-123 decision.
    assert dam.post_decision_endpoints(400.0, 123.0, np.nan, np.nan)["exit12"] == 1.0
    # Death at day 360 after diagnosis is 360 days after a day-0 decision.
    assert dam.post_decision_endpoints(360.0, 0.0, np.nan, np.nan)["exit12"] == 1.0
    # Death 366 days after the decision.
    assert dam.post_decision_endpoints(489.0, 123.0, np.nan, np.nan)["exit12"] == 0.0


def test_first_line_must_start_on_decision_day() -> None:
    with pytest.raises(ValueError):
        dam.post_decision_endpoints(500.0, 123.0, 100.0, 0.0)


def test_effect_is_zero_ish_without_signal_and_large_with_signal() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for c in ("c1", "c2"):
        for x in ("x1", "x2"):
            for h in ("H0", "H1", "H2"):
                for a in ("a1", "a2"):
                    for _ in range(3):
                        rows.append({"C": c, "X": x, "H": h, "A": a,
                                     "y": (5.0 if h == "H2" else 0.0) + rng.normal()})
    d = pd.DataFrame(rows)
    big, _raw, _n, null = dam.effect(d, "H", "y", np.random.default_rng(1))
    small, *_ = dam.effect(d, "A", "y", np.random.default_rng(1))
    assert len(null) == dam.NPERM
    assert big > 2.0
    assert abs(small) < 1.0
