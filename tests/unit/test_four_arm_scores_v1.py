"""Gate-4 scoring layer: §8 synthetic fixtures and the §10 scoring rows.

These test the scorer, not an LLM hypothesis. No patient data.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
sys.path.insert(0, str(ROOT / "src"))

import four_arm_scores_v1 as S  # noqa: E402

R, P, D = "responding", "progressing", S.DEAD


# ======================================================================================
# Energy score
# ======================================================================================

def test_energy_score_hand_value_and_fair_estimator() -> None:
    z, y = np.array([[0.0], [2.0]]), np.array([0.0])
    assert S.energy_score(z, y) == pytest.approx(1.0 - 0.5 * 1.0)      # 0.5
    assert S.fair_energy_score(z, y) == pytest.approx(1.0 - 0.5 * 4.0 / 2)
    assert S.energy_score(np.array([[3.0, 4.0]]), np.zeros(2)) == pytest.approx(5.0)


def test_energy_score_refuses_unresolved_vectors() -> None:
    with pytest.raises(ValueError):
        S.energy_score(np.array([[np.nan, 0.0]]), np.zeros(2))


def test_correct_dependence_beats_opposite_dependence_with_equal_marginals() -> None:
    """§8 fixture: identical marginals, coupled versus oppositely coupled outcomes."""
    truth = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
    coupled = np.array([[0.0, 0.0], [1.0, 1.0]])
    opposite = np.array([[0.0, 1.0], [1.0, 0.0]])
    for k in range(2):
        assert sorted(coupled[:, k]) == sorted(opposite[:, k])
    es_c = np.mean([S.energy_score(coupled, y) for y in truth])
    es_o = np.mean([S.energy_score(opposite, y) for y in truth])
    assert es_c < es_o


def test_equal_size_subsets_hand_enumerated_and_reused() -> None:
    """§10: a forecast larger than m* scores as the mean over its fixed subsets; the same
    subsets serve the joint and per-block scores."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=(3, S.N_COORDS))
    y = rng.normal(size=S.N_COORDS)
    sub = S.equal_size_subsets(3, 2, "fam", "q1", "arm", 12.0, n=6)
    assert sub.shape == (6, 2) and all(len(set(r)) == 2 for r in sub)
    expected = np.mean([S.energy_score(z[list(r)], y) for r in sub])
    assert S.equal_size_score(z, y, sub) == pytest.approx(expected)
    blocks = S.block_scores(z, y, sub)
    for b, cols in S.BLOCKS.items():
        assert blocks[b] == pytest.approx(np.mean([S.energy_score(z[list(r)][:, cols], y[cols])
                                                   for r in sub]))
    again = S.equal_size_subsets(3, 2, "fam", "q1", "arm", 12.0, n=6)
    other = S.equal_size_subsets(3, 2, "fam", "q2", "arm", 12.0, n=6)
    assert np.array_equal(sub, again) and not np.array_equal(sub, other)
    # every subset is one of the three pairs
    assert {tuple(sorted(r)) for r in sub} <= set(combinations(range(3), 2))


def test_forecast_already_at_m_star_uses_its_complete_score() -> None:
    z = np.array([[0.0] * S.N_COORDS, [1.0] * S.N_COORDS])
    y = np.zeros(S.N_COORDS)
    assert S.equal_size_subsets(2, 2, "k") is None
    assert S.equal_size_score(z, y, None) == pytest.approx(S.energy_score(z, y))


def test_four_arms_share_one_m_star_and_keep_delivered_scores() -> None:
    sizes = [10, 8, 9, 10]
    m_star = S.four_arm_m_star(sizes)
    assert m_star == 8
    rng = np.random.default_rng(1)
    y = rng.normal(size=S.N_COORDS)
    for arm, m in zip(("a", "b", "c", "d"), sizes, strict=True):
        z = rng.normal(size=(m, S.N_COORDS))
        js = S.joint_scores(z, y, m_star, "four_arm", "q", arm, 6.0)
        assert js.m_star == 8 and js.m == m
        assert js.as_delivered == pytest.approx(S.energy_score(z, y))
        assert js.fair is not None
    donor = S.joint_scores(rng.normal(size=(20, S.N_COORDS)), y, 8, "donor", "q", "card", 6.0,
                           llm=False)
    assert donor.fair is None


def test_stable_seed_is_keyed_and_reproducible() -> None:
    assert S.stable_seed("a", 1) == S.stable_seed("a", 1)
    assert S.stable_seed("a", 1) != S.stable_seed("a", 2)


def test_block_layout_matches_the_17_coordinates() -> None:
    assert S.N_COORDS == 17
    assert [S.BLOCKS[b].stop - S.BLOCKS[b].start for b in S.BLOCKS] == [2, 4, 2, 9]


# ======================================================================================
# §6 documented-time diagnostics
# ======================================================================================

def test_identical_courses_have_zero_loss() -> None:
    q = [(0.0, 3.0, R), (3.0, 6.0, P)]
    out = S.course_losses(q, q, [(0.0, 6.0)])
    assert out["occupancy_tv"] == 0 and out["disagreement"] == 0


def test_reversed_order_keeps_occupancy_and_changes_disagreement() -> None:
    """§8 order fixture: equal occupancy, reversed chronology."""
    q = [(0.0, 3.0, R), (3.0, 6.0, P)]
    f = [(0.0, 3.0, P), (3.0, 6.0, R)]
    out = S.course_losses(q, f, [(0.0, 6.0)])
    assert out["occupancy_tv"] == pytest.approx(0.0)
    assert out["disagreement"] == pytest.approx(1.0)


def test_death_is_an_exit_and_post_death_time_is_scored() -> None:
    q = S.labelled([(0.0, 4.0, P)], death=4.0, horizon=12.0)
    f = S.labelled([(0.0, 12.0, P)], death=float("inf"), horizon=12.0)
    out = S.course_losses(q, f, [(0.0, 12.0)])
    assert out["disagreement"] == pytest.approx(8.0 / 12.0)
    assert out["occupancy_tv"] == pytest.approx(8.0 / 12.0)


def test_generated_silence_over_documented_progression_is_charged() -> None:
    """§10: the diagnostic charges the silence; it is reported as unresolved time."""
    q = [(0.0, 6.0, P)]
    f = S.labelled([(0.0, 2.0, P), (2.0, 6.0, S.E.UNKNOWN)], death=float("inf"), horizon=6.0)
    out = S.course_losses(q, f, [(0.0, 6.0)])
    assert out["disagreement"] == pytest.approx(4.0 / 6.0)
    assert out["unresolved_fraction"] == pytest.approx(4.0 / 6.0)


def test_window_outside_query_support_is_refused() -> None:
    q = [(0.0, 2.0, P), (2.0, 6.0, S.UNRESOLVED)]
    with pytest.raises(ValueError):
        S.course_losses(q, q, [(0.0, 6.0)])


def _card(n_usable: int, n_short: int, path) -> list:
    return [(path, [(0.0, 6.0)])] * n_usable + [(path, [(0.0, 0.5)])] * n_short


def test_donor_diagnostic_needs_seven_usable_donors() -> None:
    q = [(0.0, 6.0, P)]
    rep = [[(0.0, 6.0, P)]]
    assert not S.donor_diagnostic(q, _card(6, 14, q), rep)["eligible"]
    out = S.donor_diagnostic(q, _card(7, 13, q), rep)
    assert out["eligible"] and out["n_usable"] == 7 and not out["all_twenty_usable"]
    assert out["scored_months"] == pytest.approx(42.0)


def test_secondary_diagnostic_uses_the_same_reduced_interval_for_both() -> None:
    """§10: donor and repeat use the repeat's resolved part of W; primary unchanged."""
    q = [(0.0, 6.0, P)]
    donor = [(0.0, 3.0, P), (3.0, 6.0, R)]          # wrong only after month 3
    rep = [(0.0, 3.0, P), (3.0, 6.0, S.UNRESOLVED)]  # silent after month 3
    out = S.donor_diagnostic(q, [(donor, [(0.0, 6.0)])] * 7, [rep])
    # primary: both disagree on [3, 6): LLM charged for silence, donor wrong -> difference 0
    assert out["disagreement_llm_minus_donor"] == pytest.approx(0.0)
    # secondary: only [0, 3), where both agree with the query
    assert out["disagreement_secondary_llm_minus_donor"] == pytest.approx(0.0)
    assert out["secondary_pairs_kept"] == 7 and out["secondary_months"] == pytest.approx(21.0)
    assert out["llm_unresolved_fraction"] == pytest.approx(0.5)
    # a repeat with under one month resolved is excluded from the secondary only
    short = [(0.0, 0.5, P), (0.5, 6.0, S.UNRESOLVED)]
    out2 = S.donor_diagnostic(q, [(donor, [(0.0, 6.0)])] * 7, [short])
    assert out2["secondary_pairs_kept"] == 0 and out2["eligible"]
    assert out2["disagreement_secondary_llm_minus_donor"] is None


def test_arm_diagnostic_uses_the_query_window_and_its_floor() -> None:
    q = [(0.0, 6.0, P)]
    assert S.arm_diagnostic(q, [q], [(0.0, 0.5)]) is None
    out = S.arm_diagnostic(q, [q, [(0.0, 6.0, R)]], [(0.0, 6.0)])
    assert out["disagreement"] == pytest.approx(0.5)


# ======================================================================================
# §8 dependence and timing
# ======================================================================================

def test_dependence_check_keeps_marginals_and_survival_block() -> None:
    rng = np.random.default_rng(3)
    z = rng.normal(size=(10, S.N_COORDS))
    z[:, 0] = np.r_[np.ones(6), np.zeros(4)]
    out = S.dependence_check(z, rng.normal(size=S.N_COORDS), "q", "arm", 12.0, n=20)
    assert out["stratum_sizes"] == (6, 4) and out["stratified_permutation_possible"]
    assert np.isfinite(out["unstratified_mean_shift"])
    assert np.isfinite(out["alive_stratified_mean_shift"])


def test_alive_stratified_shuffle_leaves_singleton_strata_unchanged() -> None:
    z = np.zeros((2, S.N_COORDS))
    z[0, 0] = 1.0                          # one alive, one dead: two singleton strata
    z[:, 3] = [0.2, 0.7]
    out = S.dependence_check(z, np.zeros(S.N_COORDS), "q", "a", 3.0, n=10)
    assert not out["stratified_permutation_possible"]
    assert out["alive_stratified_mean_shift"] == pytest.approx(0.0)


def test_progression_to_regimen_timing() -> None:
    seen = [(0.0, 3.0, R), (3.0, 6.0, P), (6.0, 12.0, P)]
    out = S.progression_to_regimen(seen, [5.0], 12.0)
    assert out["both"] and out["months_progression_to_regimen"] == pytest.approx(2.0)
    assert not out["regimen_precedes_any_progression"]
    early = S.progression_to_regimen(seen, [1.0], 12.0)
    assert early["regimen_precedes_any_progression"] and not early["both"]
    none = S.progression_to_regimen([(0.0, 12.0, R)], [], 12.0)
    assert not none["progression"] and none["months_progression_to_regimen"] is None


# ======================================================================================
# §5 survival
# ======================================================================================

def test_reverse_km_left_limits_hand_example() -> None:
    t = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.array([1, 0, 1, 0])
    g = S.reverse_km_left(t, e, np.array([2.0, 3.0, 4.0, 5.0]))
    assert g == pytest.approx([1.0, 2 / 3, 2 / 3, 0.0])


def test_outcome_at_horizon_conventions() -> None:
    t = np.array([12.0, 12.0, 8.0, 20.0, 20.0])
    e = np.array([1, 0, 0, 1, 0])
    y = S.outcome_at(t, e, 12.0)
    assert y[0] == 0 and y[1] == 1 and np.isnan(y[2]) and y[3] == 1 and y[4] == 1


def test_ipcw_brier_without_censoring_is_the_ordinary_brier() -> None:
    t = np.array([2.0, 5.0, 20.0, 30.0])
    e = np.array([1, 1, 1, 0])
    p = np.array([0.2, 0.4, 0.9, 0.7])
    out = S.ipcw_brier(t, e, np.array(["A"] * 4), p, 12.0)
    y = np.array([0, 0, 1, 1])
    assert out["brier_ipcw"] == pytest.approx(np.mean((p - y) ** 2))
    assert not out["unsupported"] and out["n_evaluable"] == 4


def test_ipcw_censored_before_h_contributes_nothing_and_cohort_is_denominator() -> None:
    t = np.array([2.0, 6.0, 20.0, 30.0])
    e = np.array([1, 0, 1, 0])
    p = np.array([0.0, 0.5, 1.0, 1.0])
    out = S.ipcw_brier(t, e, np.array(["A"] * 4), p, 12.0)
    assert out["brier_ipcw"] == pytest.approx(0.0) and out["n_evaluable"] == 3
    # survivors reweighted by 1/G(12-) = 1/(2/3): one censoring at month 6, three at risk
    w, _, _ = S.ipcw_weights(t, e, np.array(["A"] * 4), 12.0)
    assert w == pytest.approx([1.0, 0.0, 1.5, 1.5])


def test_ipcw_flags_unsupported_weights_instead_of_capping() -> None:
    # 20 censorings before H among 21 patients: G(24-) = 1/21 < 0.05
    t = np.r_[np.arange(1.0, 21.0), 30.0]
    e = np.zeros(21, int)
    out = S.ipcw_brier(t, e, np.array(["A"] * 21), np.full(21, 0.5), 24.0)
    assert out["unsupported"] and out["max_weight"] == pytest.approx(21.0)


def test_ipcw_is_stratified_by_cancer() -> None:
    t = np.array([5.0, 20.0, 20.0, 20.0])
    e = np.array([0, 0, 1, 1])
    w, _, _ = S.ipcw_weights(t, e, np.array(["A", "A", "B", "B"]), 12.0)
    assert w == pytest.approx([0.0, 2.0, 1.0, 1.0])


def test_fan_corrected_brier_keeps_negative_values() -> None:
    out = S.ipcw_brier(np.array([20.0]), np.array([0]), np.array(["A"]), np.array([0.9]),
                       12.0, m=np.array([10]))
    assert out["brier_ipcw_fan_corrected"] == pytest.approx(0.01 - 0.09 / 9)


def test_donor_km_does_not_extrapolate_past_its_support() -> None:
    """§10: support ending before H gives no estimate unless the curve reached zero."""
    assert S.km_alive_at(np.array([2.0, 5.0, 8.0]), np.array([1, 0, 0]), 12.0) is None
    assert S.km_alive_at(np.array([2.0, 5.0]), np.array([1, 1]), 12.0) == pytest.approx(0.0)
    assert S.km_alive_at(np.array([2.0, 12.0, 15.0]), np.array([1, 1, 0]), 12.0) == \
        pytest.approx(2 / 3 * 1 / 2)


def test_reverse_km_tie_keeps_the_death_in_the_censoring_risk_set() -> None:
    """§10: a censoring/death tie has a defined limit (death at s still at risk at s)."""
    g = S.reverse_km_left(np.array([2.0, 2.0, 5.0]), np.array([1, 0, 1]), np.array([3.0]))
    assert g == pytest.approx([1 - 1 / 3])


def test_card_with_one_unresolved_member_is_not_rescued() -> None:
    """§10: one unresolved member makes the card unavailable before any subsampling."""
    card = np.zeros((20, S.N_COORDS))
    card[7, 4] = np.nan
    with pytest.raises(ValueError):
        S.joint_scores(card, np.zeros(S.N_COORDS), 10, "donor", "q", "card", 6.0, llm=False)


def test_tie_preserving_bins_never_split_identical_predictions() -> None:
    p = np.array([0.1] * 7 + [0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6])
    bins = S.tie_preserving_bins(p)
    assert len(set(bins)) <= 5
    for v in np.unique(p):
        assert len(set(bins[p == v])) == 1


def test_ece_of_a_perfect_forecast_is_zero() -> None:
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert S.ece(p, np.array([0, 0, 1, 1]))["ece"] == pytest.approx(0.0)


# ======================================================================================
# §9 inference and first attempts
# ======================================================================================

def test_bootstrap_resamples_patient_cohorts_within_cancer() -> None:
    strata = ["NSCLC"] * 5 + ["BRCA"] * 3
    cohorts = S.bootstrap_cohorts(strata, "four_arm", 12.0, n=50)
    assert len(cohorts) == 50
    for idx in cohorts:
        assert len(idx) == 8
        assert sum(strata[i] == "NSCLC" for i in idx) == 5
    assert all(np.array_equal(a, b) for a, b in
               zip(cohorts, S.bootstrap_cohorts(strata, "four_arm", 12.0, n=50), strict=True))


def test_paired_mean_ci_contains_the_mean() -> None:
    d = np.array([0.1, 0.2, 0.3, 0.4])
    out = S.paired_mean_ci(d, S.bootstrap_cohorts(["A"] * 4, "x", n=200))
    assert out["ci_low"] <= out["mean"] <= out["ci_high"]


def test_holm_counts_unavailable_slots() -> None:
    adj = S.holm([0.01, None, 0.04])
    assert adj[0] == pytest.approx(0.03) and adj[1] is None
    assert adj[2] == pytest.approx(0.08)


def test_first_attempt_is_read_from_the_attempt_detail() -> None:
    """§10: a failed first attempt followed by a passing retry is never first-attempt valid."""
    retry = {"attempts_used": 2, "accepted": True,
             "attempts_detail": [{"accepted": False, "raw_response": "x"},
                                 {"accepted": True, "raw_response": "y"}]}
    assert S.first_attempt_status(retry) == "invalid"
    ok = {"attempts_used": 1, "attempts_detail": [{"accepted": True, "raw_response": "y"}]}
    assert S.first_attempt_status(ok) == "valid"
    assert S.first_attempt_status({"attempts_used": 1}) == "unavailable"
    empty = {"attempts_detail": [{"accepted": False, "raw_response": ""}]}
    assert S.first_attempt_status(empty) == "invalid"         # observed empty output
    for missing in ({"accepted": False}, {"accepted": False, "raw_response": None},
                    {"raw_response": "x"}):
        assert S.first_attempt_status({"attempts_detail": [missing]}) == "unavailable"


# ======================================================================================
# Step 1 checks (R1-R5): boundary cases
# ======================================================================================

def test_living_time_never_overrides_death() -> None:
    """R1: an unclipped living segment is clipped at death; death is absorbing."""
    unclipped = S.labelled([(0.0, 12.0, P)], death=4.0, horizon=12.0)
    clipped = S.labelled([(0.0, 4.0, P)], death=4.0, horizon=12.0)
    assert unclipped == clipped == [(0.0, 4.0, P), (4.0, 12.0, D)]
    out = S.course_losses(clipped, unclipped, [(0.0, 12.0)])
    assert out["occupancy_tv"] == 0 and out["disagreement"] == 0


def test_death_at_zero_and_at_h() -> None:
    assert S.labelled([(0.0, 6.0, P)], death=0.0, horizon=6.0) == [(0.0, 6.0, D)]
    assert S.labelled([(0.0, 6.0, P)], death=6.0, horizon=6.0) == [(0.0, 6.0, P)]
    assert S.labelled([(0.0, 9.0, P)], death=float("inf"), horizon=6.0) == [(0.0, 6.0, P)]


def test_near_tied_times_form_one_group_in_both_km_helpers() -> None:
    """R2: one tie rule for groups and risk sets."""
    eps = S.EPS / 2
    g = S.reverse_km_left(np.array([1.0, 1.0 + eps, 5.0]), np.array([0, 0, 1]),
                          np.array([2.0]))
    assert g == pytest.approx([1 / 3])
    s = S.km_alive_at(np.array([1.0, 1.0 + eps, 15.0]), np.array([1, 1, 0]), 12.0)
    assert s == pytest.approx(1 / 3)


def test_ece_without_evaluable_patients_is_unavailable() -> None:
    """R3: missing information is not perfect calibration."""
    for p, y, w in (([0.7], [np.nan], None), ([], [], None), ([0.5], [1.0], [0.0])):
        out = S.ece(np.array(p), np.array(y), None if w is None else np.array(w))
        assert out["ece"] is None and out["n_evaluable"] == 0


def test_wilcoxon_runs_on_tiny_nonzero_differences() -> None:
    """R5: only exactly-zero differences are omitted."""
    assert S.wilcoxon_p(np.array([0.0, 0.0, 0.0])) is None
    assert S.wilcoxon_p(np.array([1e-10, -1e-10, 2e-10])) is not None


def test_secondary_component_means_reconcile_with_the_secondary_difference() -> None:
    """Step-3 gate R1: secondary means come from the retained pairs and reduced windows."""
    q = [(0.0, 6.0, P)]
    donor = [(0.0, 2.0, P), (2.0, 6.0, R)]                     # wrong after month 2
    reps = [[(0.0, 3.0, P), (3.0, 6.0, S.UNRESOLVED)],          # resolved [0, 3)
            [(0.0, 1.0, R), (1.0, 6.0, P)]]                     # fully resolved
    out = S.donor_diagnostic(q, [(donor, [(0.0, 6.0)])] * 7, reps)
    for k in S.LOSSES:
        diff = out[f"{k}_secondary_llm"] - out[f"{k}_secondary_donor"]
        assert diff == pytest.approx(out[f"{k}_secondary_llm_minus_donor"])
    # repeat 1 on [0, 3): LLM 0, donor 1/3; repeat 2 on [0, 6): LLM 1/6, donor 4/6
    assert out["disagreement_secondary_llm"] == pytest.approx((0 + 1 / 6) / 2)
    assert out["disagreement_secondary_donor"] == pytest.approx((1 / 3 + 4 / 6) / 2)
    assert out["secondary_pairs_kept"] == 14 and out["secondary_months"] == pytest.approx(63.0)
    # primary means still reconcile with the primary difference
    for k in S.LOSSES:
        assert out[f"{k}_llm"] - out[f"{k}_donor"] == pytest.approx(out[f"{k}_llm_minus_donor"])
