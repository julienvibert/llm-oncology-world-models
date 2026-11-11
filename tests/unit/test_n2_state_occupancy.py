"""N2 month rule: death, assessment and censoring boundaries, the same on both sides.

Synthetic courses only (no registry data). Covers two boundary cases: a death
exactly at month 24 (dead on one side, stable on the other) and a state change exactly at month 24
(missed when month 24 was read just before the horizon).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[2]
_PATH, _RC = list(sys.path), matplotlib.rcParams.copy()
for p in ("scripts", "scripts/explore", "scripts/figures/panels", "src"):
    sys.path.insert(0, str(ROOT / p))

import four_arm_scores_v1 as S  # noqa: E402
import n2_state_occupancy_v1 as N  # noqa: E402

# the figure kit imported by the script sets global matplotlib style; leave other tests untouched
sys.path[:] = _PATH
matplotlib.rcParams.update(_RC)

H = N.READ_HORIZON


def labelled(path: list, death: float) -> list:
    """A synthetic course read as the scorer reads it (death absorbing)."""
    return S.labelled(path, death, H)


def test_death_exactly_at_horizon_is_dead_on_both_sides() -> None:
    segs = labelled([(0.0, 25.0, "stable")], death=24.0)
    observed = N.month_state(segs, 24.0, 24, follow_end=30.0)
    generated = N.month_state(segs, 24.0, 24)
    assert observed == generated == S.DEAD


def test_alive_at_horizon_reads_last_state_not_none() -> None:
    segs = labelled([(0.0, 25.0, "stable")], death=40.0)
    assert N.month_state(segs, 40.0, 24, follow_end=30.0) == "stable"
    assert N.month_state(segs, 40.0, 24) == "stable"


def test_death_exactly_at_a_month_counts_as_dead_at_that_month() -> None:
    segs = labelled([(0.0, 6.0, "responding")], death=6.0)
    assert N.month_state(segs, 6.0, 5) == "responding"
    assert N.month_state(segs, 6.0, 6) == S.DEAD


def test_state_change_at_an_assessment_month_takes_the_new_state() -> None:
    segs = labelled([(0.0, 3.0, "responding"), (3.0, 25.0, "progressing")], death=99.0)
    assert N.month_state(segs, 99.0, 2) == "responding"
    assert N.month_state(segs, 99.0, 3) == "progressing"


def test_censoring_at_a_month_excludes_it_and_is_never_unresolved() -> None:
    segs = labelled([(0.0, 10.0, "stable")], death=99.0)
    assert N.month_state(segs, 99.0, 9, follow_end=10.0) == "stable"
    assert N.month_state(segs, 99.0, 10, follow_end=10.0) == N.CENSORED


def test_death_before_censoring_wins() -> None:
    segs = labelled([(0.0, 8.0, "progressing")], death=8.0)
    assert N.month_state(segs, 8.0, 12, follow_end=8.0) == S.DEAD


def test_uncovered_living_time_is_unresolved() -> None:
    segs = labelled([(2.3, 25.0, "progressing")], death=99.0)
    assert N.month_state(segs, 99.0, 0) == S.UNRESOLVED
    assert N.month_state(segs, 99.0, 3) == "progressing"


def test_assessment_exactly_at_month_24_is_seen_on_both_sides() -> None:
    segs = labelled([(0.0, 24.0, "stable"), (24.0, 25.0, "progressing")], death=99.0)
    assert N.month_state(segs, 99.0, 23) == "stable"
    assert N.month_state(segs, 99.0, 24, follow_end=30.0) == "progressing"
    assert N.month_state(segs, 99.0, 24) == "progressing"
