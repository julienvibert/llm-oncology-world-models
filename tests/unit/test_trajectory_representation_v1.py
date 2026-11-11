"""The shared trajectory representation, under the repository's own test suite.

External review of the first version found five blocking defects, each reproduced here as a
regression test so "the suite is green" actually covers this estimator -- previously only
the module's own self-test did, and that self-test is written by the same author as the
code it checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
sys.path.insert(0, str(ROOT / "src"))

import trajectory_representation_v1 as T  # noqa: E402

HORIZON = 24.0


@pytest.fixture(scope="module")
def rep() -> T.Representation:
    return T.fit(T._toy_training(np.random.default_rng(0)), HORIZON, n_prototypes=8)


def test_old_gate_blind_spot_is_separated(rep) -> None:
    """Two responses versus none, identical first progression and death: the passage-time
    gate scored this pair at distance exactly 0.0.

    Note the clock block as a whole DOES move here, legitimately -- it now carries first
    RESPONSE, which is 2.0 months in one course and absent in the other. What must be
    identical are the two coordinates the old gate could see.
    """
    a, b = T.FIXTURES["richness"]
    ca, cb = T.components(*a, HORIZON), T.components(*b, HORIZON)
    assert ca["clock"][0] == cb["clock"][0], "first progression is equal by construction"
    assert ca["clock"][1] == cb["clock"][1], "death is equal by construction"
    profile = rep.profile(a, b)
    assert profile["composite"] > 0.1
    assert profile["richness"] > 0.05


@pytest.mark.parametrize("block", list(T.FIXTURES))
def test_each_block_detects_its_own_fixture(rep, block) -> None:
    assert rep.profile(*T.FIXTURES[block])[block] > 0.05


def test_occupancy_fixture_is_isolated_from_clock_and_richness(rep) -> None:
    """The responding-window pair holds first progression, death, first response and every
    episode count equal by construction; only the window length differs."""
    profile = rep.profile(*T.FIXTURES["occupancy"])
    assert profile["occupancy"] > 0.05
    assert profile["clock"] < 1e-9
    assert profile["richness"] < 1e-9


def test_no_horizon_leakage_through_the_prototype_panel() -> None:
    """A 12-month analysis must not consult month-55 events via the panel. The first
    version stored raw training trajectories and did exactly that."""
    late = [([2.0, 30.0, 55.0], ["progression", "progression", "death"])
            for _ in range(12)]
    rep12 = T.fit(late, horizon=12.0, n_prototypes=4)
    assert all(max(t) <= 12.0 for t, _ in rep12.prototypes)


def test_dedup_never_merges_across_a_treatment_event() -> None:
    """Progressing before a new line and progressing on it are different clinical facts.
    The earlier similarity module protected this; the first version here dropped it."""
    _, plain = T.episodes([2.0, 2.8], ["progression", "progression"], HORIZON)
    assert plain.count("progression") == 1
    _, split = T.episodes(
        [2.0, 2.5, 2.8],
        ["progression", "systemic_treatment_start", "progression"], HORIZON)
    assert split.count("progression") == 2


def test_zero_variance_training_coordinate_is_inert() -> None:
    """A feature absent in training must not contribute in test. The first version fell
    back to scale 1.0, letting it contribute arbitrarily."""
    no_resp = [([float(p), float(p) + 3.0], ["progression", "death"])
               for p in range(2, 20)]
    rep_nr = T.fit(no_resp, HORIZON, n_prototypes=4)
    pair = (([5.0, 8.0], ["progression", "death"]),
            ([5.0, 6.0, 7.0, 8.0], ["progression", "response", "response", "death"]))
    assert rep_nr.profile(*pair)["richness"] < 1e-9


def test_each_primitive_has_one_home() -> None:
    """Passage times live in clock only, treatment starts in treatment only. The first
    composite double-counted PFS, OS and scheduling."""
    c = T.components([2.0, 5.0, 9.0, 14.0],
                     ["response", "systemic_treatment_switch", "progression", "death"],
                     HORIZON)
    assert len(c["clock"]) == 3
    assert len(c["richness"]) == 2, "richness is progressions and responses only"
    # occupancy is the responding state, not a redrawn passage time: it starts OFF
    assert c["occupancy"][0] == 0.0 and c["occupancy"].max() == 1.0


def test_vp_is_a_dissimilarity_not_a_metric() -> None:
    """The /max(n,m) normalization breaks the triangle inequality (verified
    counterexample). The violation is asserted so that nobody silently renames the quantity
    a metric while the normalization stands."""
    A = ([1.0, 2.0], ["response", "response"])
    B = ([1.0, 2.0], ["progression", "progression"])
    C = ([1.0, 1.5, 2.0], ["response", "progression", "response"])
    assert T.typed_vp(*A, *B) > T.typed_vp(*A, *C) + T.typed_vp(*C, *B)


def test_vp_identity_cap_and_ontology_grading() -> None:
    t, m = [1.0, 5.0, 9.0], ["response", "progression", "death"]
    assert T.typed_vp(t, m, t, m) == 0.0
    adjacent = T.typed_vp(t, m, t, ["response", "recurrence", "death"])
    unrelated = T.typed_vp(t, m, t, ["response", "systemic_treatment_stop", "death"])
    assert 0 < adjacent < unrelated <= 2.0


def test_vp_is_order_sensitive_at_identical_times() -> None:
    assert T.typed_vp(
        [3.0, 6.0, 9.0], ["response", "systemic_treatment_switch", "death"],
        [3.0, 6.0, 9.0], ["systemic_treatment_switch", "response", "death"]) > 0.1


def test_time_norm_is_prespecified_own_with_horizon_sensitivity() -> None:
    """Own-duration normalization is the primary (shape-only); a common horizon is the
    declared sensitivity. The two must actually differ where durations differ."""
    a = ([2.0, 4.0], ["progression", "death"])
    b = ([10.0, 20.0], ["progression", "death"])
    own = T.typed_vp(*a, *b, time_norm="own")
    common = T.typed_vp(*a, *b, time_norm=24.0)
    assert own < common, (own, common)


def test_scan_cadence_does_not_inflate_richness() -> None:
    """A patient scanned weekly through one progression must not look richer than one
    scanned quarterly -- observation intensity is a confound, not a signal."""
    weekly = ([3.0, 3.25, 3.5, 3.75, 10.0],
              ["progression"] * 4 + ["death"])
    quarterly = ([3.0, 10.0], ["progression", "death"])
    ca = T.components(*weekly, HORIZON)
    cb = T.components(*quarterly, HORIZON)
    np.testing.assert_allclose(ca["richness"], cb["richness"])


def test_treatment_identity_separates_same_schedule_different_drug() -> None:
    """The audit's concrete failure: FOLFOX-then-FOLFIRI and FOLFOX-then-osimertinib were
    indistinguishable under bare on/off. Class-typed marks fix it at two places: grammar
    (via the typed VP) and the distinct-class count in the treatment block."""
    assert T.mark_distance("systemic_treatment_start::platinum",
                           "systemic_treatment_start::targeted TKI") == 0.5
    assert T.mark_distance("systemic_treatment_start::platinum",
                           "systemic_treatment_start::platinum") == 0.0
    a = ([6.0, 7.0, 14.0],
         ["progression", "systemic_treatment_start::antimetabolite", "death"])
    b = ([6.0, 7.0, 14.0],
         ["progression", "systemic_treatment_start::targeted TKI", "death"])
    assert T.typed_vp(*a, *b) > 0.05
    same = T.components(*a, HORIZON)
    # class-typed starts still count as starts and still switch the on/off state
    assert same["treatment"][-3] == pytest.approx(1.0)   # sqrt(n_start)


def test_class_typed_starts_do_not_merge_as_episodes() -> None:
    """Two starts of different classes 0.8 months apart are two clinical facts, not one
    episode -- different marks never merged, and the class suffix keeps them different."""
    _, marks = T.episodes([2.0, 2.8],
                          ["systemic_treatment_start::platinum",
                           "systemic_treatment_start::targeted TKI"], HORIZON)
    assert len(marks) == 2
