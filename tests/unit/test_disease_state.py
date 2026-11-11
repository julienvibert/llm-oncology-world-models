"""The clinical truth table for the canonical disease-state operator.

Each case is a clinical situation stated in words, its expected state, and why. If the
table and the implementation ever disagree, one of them is wrong and the test says which
situation exposed it. The edge cases are the point: partial resection, mixed response,
treatment change without disease change, recurrence, and death.
"""

from __future__ import annotations

import pytest

from synthonccases.disease_state import (
    DiseaseState,
    defined_window,
    from_burden,
    from_genie,
)

# ---------------------------------------------------------------------------------------
# GENERATED TRAJECTORIES: burden is materialized, so every state is directly observed.
# (situation, burden bands, kinetics, expected state, mixed?)
# ---------------------------------------------------------------------------------------
BURDEN_CASES = [
    ("complete resection of a solitary primary",
     ["absent"], ["cleared"], "no_active_disease", False),
    ("metastatic disease cleared by systemic therapy",
     ["absent", "absent"], ["cleared", "cleared"], "no_active_disease", False),
    # PARTIAL RESECTION. Removing some disease is not remission, and it is not
    # progression either: the state is whatever the RESIDUAL disease is doing. The
    # cleared site drops out of the assessment entirely -- it is gone, not shrinking.
    ("partial resection leaving stable residual liver disease",
     ["absent", "moderate"], ["cleared", "stable"], "stable", False),
    ("partial resection leaving residual disease that then grows",
     ["absent", "moderate"], ["cleared", "growing"], "progressing", False),
    ("microscopic residual disease after resection",
     ["microscopic"], ["stable"], "stable", False),
    ("all measurable sites shrinking on treatment",
     ["small", "small"], ["shrinking", "shrinking"], "responding", False),
    ("disease present and unchanging",
     ["moderate"], ["stable"], "stable", False),
    ("established disease growing",
     ["moderate"], ["growing"], "progressing", False),
    # MIXED RESPONSE. Worst-wins: the growing site drives management, and calling this
    # "responding" would report a patient who needs a treatment change as improving.
    ("liver shrinking while a new bone lesion appears",
     ["small", "small"], ["shrinking", "appearing"], "progressing", True),
    # RECURRENCE is not a mixed response. The previously cleared site is gone, not
    # shrinking, so there are no competing kinetics to resolve -- only new disease.
    ("bone metastasis appearing after years of remission",
     ["absent", "small"], ["cleared", "appearing"], "progressing", False),
]


@pytest.mark.parametrize(("situation", "bands", "kinetics", "expected", "mixed"),
                         BURDEN_CASES)
def test_burden_truth_table(situation: str, bands: list[str], kinetics: list[str],
                            expected: str, mixed: bool) -> None:
    got = from_burden(bands, kinetics)
    assert got.state == expected, f"{situation}: expected {expected}, got {got.state}"
    assert got.mixed is mixed, situation
    assert got.evidence == "directly_observed"


def test_microscopic_residual_is_not_remission() -> None:
    """Minimal residual disease and complete remission are different situations, and a
    band that merged them would make them indistinguishable forever after."""
    assert from_burden(["microscopic"], ["stable"]).state != "no_active_disease"
    assert from_burden(["absent"], ["cleared"]).state == "no_active_disease"


def test_treatment_change_alone_cannot_move_the_state() -> None:
    """The operator has no treatment argument at all, so a planned chemotherapy-to-
    endocrine handover CANNOT change the disease state. That is the property, expressed
    as an absence: it is guaranteed by the signature rather than by a rule that could be
    forgotten."""
    before = from_burden(["absent"], ["cleared"])
    after = from_burden(["absent"], ["cleared"])
    assert before == after


# ---------------------------------------------------------------------------------------
# GENIE: two questions answered separately, and nothing inferred.
# ---------------------------------------------------------------------------------------
def genie(**kw) -> DiseaseState:
    base = {"presence": None, "curated_status": None, "new_distant_site": False}
    return from_genie(**{**base, **kw})


def test_presence_is_the_only_route_to_no_active_disease() -> None:
    """GENIE's response vocabulary has no complete-response category, so without the
    oncologist's presence field a patient in remission is invisible."""
    assert genie(presence="no evidence").state == "no_active_disease"
    assert genie(curated_status="responding").state == "responding"


def test_a_current_assessment_is_used_directly() -> None:
    got = genie(presence="evidence", curated_status="progressing")
    assert (got.state, got.evidence) == ("progressing", "directly_observed")


def test_mixed_resolves_to_progressing_and_keeps_the_flag() -> None:
    got = genie(presence="evidence", curated_status="mixed")
    assert got.state == "progressing"
    assert got.mixed is True


def test_cancer_present_with_no_dynamics_is_not_the_same_as_no_information() -> None:
    """The distinction that concealed the worst case of the old inference: documented
    active disease had been labelled remission, and reporting both as bare 'unknown' hid
    it. The basis must say which situation this is."""
    got = genie(presence="evidence")
    assert got.state == "unknown"
    assert "cancer present" in got.basis
    assert "no recent assessment" in genie().basis


def test_conflict_is_named_not_resolved_in_favour_of_the_reassuring_reading() -> None:
    """No evidence of cancer AND a progressing status are the two most opposite readings.
    An earlier version claimed worst-first and tested presence first, so this returned
    remission."""
    got = genie(presence="no evidence", curated_status="progressing")
    assert got.state == "unknown"
    assert got.basis.startswith("CONFLICT")


def test_a_new_distant_site_conflicts_with_no_evidence_too() -> None:
    got = genie(presence="no evidence", new_distant_site=True)
    assert got.state == "unknown"
    assert got.basis.startswith("CONFLICT")


def test_a_new_distant_site_outranks_an_older_response() -> None:
    """A patient with a newly documented metastasis is progressing, whatever the earlier
    assessment said about the lesions it was watching."""
    got = genie(presence="evidence", curated_status="responding", new_distant_site=True)
    assert got.state == "progressing"


def test_equivocal_presence_is_unknown() -> None:
    assert genie(presence="equivocal").state == "unknown"


def test_nothing_is_ever_inferred_from_silence() -> None:
    """The removed inference read the patient's SUBSEQUENT follow-up to decide their state
    at the landmark. There is no argument left that could carry future information."""
    import inspect
    args = set(inspect.signature(from_genie).parameters)
    assert args == {"presence", "curated_status", "new_distant_site"}
    assert genie().evidence == "unknown"


def test_strict_mode_is_a_no_op_now_that_nothing_is_inferred() -> None:
    for st in (genie(presence="no evidence"), genie(presence="evidence"), genie()):
        assert st.strict() == st


# ---------------------------------------------------------------------------------------
# DEATH is vital status, not a disease state.
# ---------------------------------------------------------------------------------------
def test_death_closes_the_window_rather_than_producing_unknown_time() -> None:
    """A patient who dies at month 5 of a 12-month window has a 5-month course, not a
    5-month course plus 7 unknown months. Those 7 would otherwise be masked, imputed, or
    counted as unobserved follow-up -- all wrong in the same direction."""
    start, end, why = defined_window(0.0, 12.0, death=5.0, last_contact=5.0)
    assert (end - start, why) == (5.0, "closed by death")


def test_loss_to_follow_up_is_the_opposite_case() -> None:
    """Here the question still applies and we cannot answer it, so the interval ends and
    the remainder is genuinely unknown."""
    _s, e, why = defined_window(0.0, 12.0, death=float("inf"), last_contact=7.0)
    assert (e, why) == (7.0, "closed by loss to follow-up")


def test_a_fully_followed_survivor_has_the_whole_window() -> None:
    _s, e, why = defined_window(0.0, 12.0, death=float("inf"), last_contact=40.0)
    assert (e, why) == (12.0, "complete")


def test_death_after_the_horizon_does_not_truncate() -> None:
    _s, e, why = defined_window(0.0, 12.0, death=30.0, last_contact=30.0)
    assert (e, why) == (12.0, "complete")
