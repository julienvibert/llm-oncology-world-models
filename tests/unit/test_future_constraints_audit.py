"""Gate A2 — generalized ``future_constraints_audit`` over the materialized ledger.

Tests cover:

- inversion locks for ``must_not_occur`` (zero matches pass; ≥1 fail)
  vs all other constraints (zero matches fail);
- selector ordinal resolution (first / second / third / last; out-of-range);
- ``ordinal_after_cutoff`` filtering + spec invariant (rejects when no prefix);
- ``date_range`` filter (alone and combined with ordinal);
- deterministic tie-breaking on same-date events by original index;
- ``regimen_matches`` semantics (label-superset, drug-component, empty-alias,
  missing-regimen handling);
- ``event_type_is_one_of`` constraint;
- ``must_occur_within_days`` with ``reference: cutoff`` (within / beyond /
  before-cutoff / missing-prefix);
- ``must_occur_within_days`` with ``reference: prior_event`` (deferred-in-A2 error);
- multi-constraint specs return one match per constraint in input order;
- empty future_event_constraints → empty result;
- ``must_not_occur`` exclusivity guard against other semantic keys;
- unknown constraint-key typo guard;
- parity lock: the copied ``_regimen_matches_alias`` matches the runner's
  byte-for-byte on a representative input set.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from synthonccases.oncollm.conditioning_audit import (
    ConditioningSpec,
    ConstraintMatch,
    FutureEventConstraint,
    FutureEventSelector,
    ObservedPrefix,
    _normalize_regimen_str as audit_normalize,
    _regimen_matches_alias as audit_matches,
    future_constraints_audit,
)
from synthonccases.oncollm.poc_runner import (
    _normalize_regimen_str as runner_normalize,
    _regimen_matches_alias as runner_matches,
)
from synthonccases.oncollm.replay import MaterializedLedgerEvent


# ---------------------------------------------------------------------------
# Helpers for materializing test ledgers.
# ---------------------------------------------------------------------------


def _materialized_event(
    *,
    event_id: str,
    event_date: str,
    event_type: str,
    regimen_label: str | None = None,
    regimen_components: list[str] | None = None,
    **extra: Any,
) -> MaterializedLedgerEvent:
    payload: dict[str, Any] = {
        "event_id": event_id,
        "date": event_date,
        "type": event_type,
    }
    if regimen_label is not None:
        payload["regimen"] = {
            "label": regimen_label,
            "components": regimen_components or [regimen_label],
        }
    payload.update(extra)
    return MaterializedLedgerEvent.model_validate(payload)


def _empty_spec_with_constraints(
    *constraints: FutureEventConstraint, prefix: ObservedPrefix | None = None
) -> ConditioningSpec:
    return ConditioningSpec(
        baseline_seed={},
        pinned_paths=(),
        observed_prefix=prefix,
        future_event_constraints=tuple(constraints),
        free_text_operator_constraints="",
    )


def _prefix(cutoff: str) -> ObservedPrefix:
    return ObservedPrefix(events=(), cutoff_date=date.fromisoformat(cutoff))


# ---------------------------------------------------------------------------
# Empty-spec and basic shape.
# ---------------------------------------------------------------------------


def test_audit_returns_empty_for_spec_without_constraints() -> None:
    spec = _empty_spec_with_constraints()
    ledger: list[MaterializedLedgerEvent] = [
        _materialized_event(event_id="e1", event_date="2024-04-15", event_type="response"),
    ]
    assert future_constraints_audit(spec, ledger) == []


def test_audit_returns_one_match_per_constraint_in_input_order() -> None:
    c_a = FutureEventConstraint(
        constraint_id="alpha",
        selector=FutureEventSelector(event_type="response"),
        constraint={"must_not_occur": True},
    )
    c_b = FutureEventConstraint(
        constraint_id="beta",
        selector=FutureEventSelector(event_type="progression"),
        constraint={"must_not_occur": True},
    )
    spec = _empty_spec_with_constraints(c_a, c_b)
    result = future_constraints_audit(spec, [])
    assert [m.constraint_id for m in result] == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# must_not_occur — inversion lock.
# ---------------------------------------------------------------------------


def test_must_not_occur_passes_when_selector_finds_nothing() -> None:
    constraint = FutureEventConstraint(
        constraint_id="no_local",
        selector=FutureEventSelector(event_type="local_intervention"),
        constraint={"must_not_occur": True},
    )
    spec = _empty_spec_with_constraints(constraint)
    ledger = [_materialized_event(event_id="e1", event_date="2024-04-15", event_type="response")]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is True
    assert result.matched_event_indices == ()
    assert result.matched_event_ids == ()
    assert result.failure_reason is None


def test_must_not_occur_fails_when_selector_finds_events() -> None:
    constraint = FutureEventConstraint(
        constraint_id="no_local",
        selector=FutureEventSelector(event_type="local_intervention"),
        constraint={"must_not_occur": True},
    )
    spec = _empty_spec_with_constraints(constraint)
    ledger = [
        _materialized_event(event_id="e1", event_date="2024-04-15", event_type="response"),
        _materialized_event(
            event_id="e2", event_date="2024-05-01", event_type="local_intervention"
        ),
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is False
    assert result.matched_event_indices == (1,)
    assert result.matched_event_ids == ("e2",)
    assert "must_not_occur=true" in (result.failure_reason or "")


def test_normal_constraint_fails_when_selector_finds_nothing() -> None:
    constraint = FutureEventConstraint(
        constraint_id="must_start_THP",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start", ordinal="first"
        ),
        constraint={"regimen_matches": ["THP"]},
    )
    spec = _empty_spec_with_constraints(constraint)
    [result] = future_constraints_audit(spec, [])
    assert result.passed is False
    assert "selector resolved to no events" in (result.failure_reason or "")


def test_must_not_occur_exclusive_with_other_semantic_keys() -> None:
    """must_not_occur=true plus any other constraint key must raise."""
    constraint = FutureEventConstraint(
        constraint_id="conflicting",
        selector=FutureEventSelector(event_type="response"),
        constraint={"must_not_occur": True, "regimen_matches": ["foo"]},
    )
    spec = _empty_spec_with_constraints(constraint)
    with pytest.raises(ValueError, match="exclusive with other constraint keys"):
        future_constraints_audit(spec, [])


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(False, id="literal_false"),
        pytest.param(None, id="none"),
        pytest.param("false", id="string_false"),
        pytest.param("true", id="string_true"),
        pytest.param(1, id="int_one"),
        pytest.param(0, id="int_zero"),
    ],
)
def test_must_not_occur_rejects_non_true_values(bad_value: Any) -> None:
    """must_not_occur is binary by spec: only literal ``True`` is accepted.

    ``False`` would create a no-op constraint (silent always-pass). Non-bool
    truthy values like ``"true"`` or ``1`` could mask operator typos. The
    DSL rejects everything except ``True``.
    """
    constraint = FutureEventConstraint(
        constraint_id="strict",
        selector=FutureEventSelector(event_type="response"),
        constraint={"must_not_occur": bad_value},
    )
    spec = _empty_spec_with_constraints(constraint)
    with pytest.raises(
        ValueError, match="must_not_occur must be exactly True"
    ):
        future_constraints_audit(spec, [])


def test_must_not_occur_false_with_other_keys_rejects_at_value_check() -> None:
    """``{must_not_occur: False, regimen_matches: ...}`` is rejected at the
    value-validation step (not as exclusivity), because ``False`` is itself
    an invalid value regardless of pairing.
    """
    constraint = FutureEventConstraint(
        constraint_id="bad_pairing",
        selector=FutureEventSelector(event_type="response"),
        constraint={"must_not_occur": False, "regimen_matches": ["foo"]},
    )
    spec = _empty_spec_with_constraints(constraint)
    with pytest.raises(
        ValueError, match="must_not_occur must be exactly True"
    ):
        future_constraints_audit(spec, [])


# ---------------------------------------------------------------------------
# Selector ordinals + tie-break.
# ---------------------------------------------------------------------------


def test_selector_ordinal_first_picks_chronologically_earliest() -> None:
    constraint = FutureEventConstraint(
        constraint_id="first_st",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start", ordinal="first"
        ),
        constraint={"regimen_matches": ["THP"]},
    )
    spec = _empty_spec_with_constraints(constraint)
    # Out-of-order ledger; "first" must pick the earliest by date.
    ledger = [
        _materialized_event(
            event_id="late",
            event_date="2024-09-15",
            event_type="systemic_treatment_start",
            regimen_label="THP",
        ),
        _materialized_event(
            event_id="early",
            event_date="2024-04-15",
            event_type="systemic_treatment_start",
            regimen_label="THP",
        ),
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is True
    assert result.matched_event_ids == ("early",)
    # Original index = 1 (the "early" event was second in input order).
    assert result.matched_event_indices == (1,)


def test_selector_ordinal_last_picks_chronologically_latest() -> None:
    constraint = FutureEventConstraint(
        constraint_id="last_st",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start", ordinal="last"
        ),
        constraint={"regimen_matches": ["T-DM1"]},
    )
    spec = _empty_spec_with_constraints(constraint)
    ledger = [
        _materialized_event(
            event_id="early",
            event_date="2024-04-15",
            event_type="systemic_treatment_start",
            regimen_label="THP",
        ),
        _materialized_event(
            event_id="late",
            event_date="2024-09-15",
            event_type="systemic_treatment_start",
            regimen_label="T-DM1",
        ),
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is True
    assert result.matched_event_ids == ("late",)


def test_selector_ordinal_second_returns_empty_when_only_one_match() -> None:
    constraint = FutureEventConstraint(
        constraint_id="must_second",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start", ordinal="second"
        ),
        constraint={"regimen_matches": ["T-DM1"]},
    )
    spec = _empty_spec_with_constraints(constraint)
    ledger = [
        _materialized_event(
            event_id="only",
            event_date="2024-04-15",
            event_type="systemic_treatment_start",
            regimen_label="THP",
        ),
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is False
    assert "selector resolved to no events" in (result.failure_reason or "")


def test_selector_ties_break_by_original_index() -> None:
    """Two events with the same date keep their ledger order."""
    constraint = FutureEventConstraint(
        constraint_id="first_same_day",
        selector=FutureEventSelector(event_type="response", ordinal="first"),
        constraint={},  # any matched event is fine; we just want the index.
    )
    # Workaround: empty constraint means "no recognized keys"; pass an
    # event_type_is_one_of that always holds.
    constraint = FutureEventConstraint(
        constraint_id="first_same_day",
        selector=FutureEventSelector(event_type="response", ordinal="first"),
        constraint={"event_type_is_one_of": ["response"]},
    )
    spec = _empty_spec_with_constraints(constraint)
    ledger = [
        _materialized_event(event_id="A", event_date="2024-06-01", event_type="response"),
        _materialized_event(event_id="B", event_date="2024-06-01", event_type="response"),
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.matched_event_ids == ("A",)  # A was first in ledger order


# ---------------------------------------------------------------------------
# ordinal_after_cutoff + spec invariant.
# ---------------------------------------------------------------------------


def test_ordinal_after_cutoff_filters_post_cutoff_events() -> None:
    constraint = FutureEventConstraint(
        constraint_id="post_cutoff_switch",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start",
            ordinal_after_cutoff="first",
        ),
        constraint={"regimen_matches": ["T-DM1"]},
    )
    spec = _empty_spec_with_constraints(constraint, prefix=_prefix("2024-07-15"))
    ledger = [
        _materialized_event(
            event_id="pre",
            event_date="2024-04-15",
            event_type="systemic_treatment_start",
            regimen_label="THP",
        ),
        _materialized_event(
            event_id="post",
            event_date="2024-09-15",
            event_type="systemic_treatment_start",
            regimen_label="T-DM1",
        ),
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is True
    assert result.matched_event_ids == ("post",)


def test_ordinal_after_cutoff_raises_when_spec_has_no_prefix() -> None:
    constraint = FutureEventConstraint(
        constraint_id="needs_prefix",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start",
            ordinal_after_cutoff="first",
        ),
        constraint={"regimen_matches": ["T-DM1"]},
    )
    spec = _empty_spec_with_constraints(constraint, prefix=None)
    with pytest.raises(ValueError, match="observed_prefix"):
        future_constraints_audit(spec, [])


# ---------------------------------------------------------------------------
# date_range filter.
# ---------------------------------------------------------------------------


def test_date_range_filters_to_window() -> None:
    constraint = FutureEventConstraint(
        constraint_id="window",
        selector=FutureEventSelector(
            event_type="progression",
            date_range=(date(2024, 5, 1), date(2024, 8, 31)),
        ),
        constraint={"must_not_occur": True},
    )
    spec = _empty_spec_with_constraints(constraint)
    ledger = [
        _materialized_event(event_id="before", event_date="2024-03-15", event_type="progression"),
        _materialized_event(event_id="in_window", event_date="2024-07-15", event_type="progression"),
        _materialized_event(event_id="after", event_date="2024-10-15", event_type="progression"),
    ]
    [result] = future_constraints_audit(spec, ledger)
    # Window contains one event → must_not_occur fails (it should not occur).
    assert result.passed is False
    assert result.matched_event_ids == ("in_window",)


def test_date_range_combined_with_ordinal_first_in_window() -> None:
    constraint = FutureEventConstraint(
        constraint_id="first_in_window",
        selector=FutureEventSelector(
            event_type="progression",
            ordinal="first",
            date_range=(date(2024, 5, 1), date(2024, 12, 31)),
        ),
        constraint={"event_type_is_one_of": ["progression"]},
    )
    spec = _empty_spec_with_constraints(constraint)
    ledger = [
        _materialized_event(event_id="before", event_date="2024-03-15", event_type="progression"),
        _materialized_event(event_id="first_in", event_date="2024-06-01", event_type="progression"),
        _materialized_event(event_id="second_in", event_date="2024-09-01", event_type="progression"),
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is True
    assert result.matched_event_ids == ("first_in",)


# ---------------------------------------------------------------------------
# regimen_matches semantics.
# ---------------------------------------------------------------------------


def test_regimen_matches_label_token_superset() -> None:
    constraint = FutureEventConstraint(
        constraint_id="match_label_superset",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start", ordinal="first"
        ),
        constraint={"regimen_matches": ["THP"]},
    )
    spec = _empty_spec_with_constraints(constraint)
    ledger = [
        _materialized_event(
            event_id="e1",
            event_date="2024-04-15",
            event_type="systemic_treatment_start",
            regimen_label="THP combination",  # observed label = superset of alias
            regimen_components=["trastuzumab", "pertuzumab", "docetaxel"],
        )
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is True


def test_regimen_matches_drug_component_membership() -> None:
    constraint = FutureEventConstraint(
        constraint_id="match_via_component",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start", ordinal="first"
        ),
        constraint={"regimen_matches": ["trastuzumab"]},
    )
    spec = _empty_spec_with_constraints(constraint)
    ledger = [
        _materialized_event(
            event_id="e1",
            event_date="2024-04-15",
            event_type="systemic_treatment_start",
            regimen_label="THP",
            regimen_components=["trastuzumab", "pertuzumab", "docetaxel"],
        )
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is True


def test_regimen_matches_fails_when_no_alias_overlap() -> None:
    constraint = FutureEventConstraint(
        constraint_id="must_be_T_DM1",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start", ordinal="first"
        ),
        constraint={"regimen_matches": ["T-DM1"]},
    )
    spec = _empty_spec_with_constraints(constraint)
    ledger = [
        _materialized_event(
            event_id="e1",
            event_date="2024-04-15",
            event_type="systemic_treatment_start",
            regimen_label="THP",
            regimen_components=["trastuzumab", "pertuzumab", "docetaxel"],
        )
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is False
    assert "does not match any alias" in (result.failure_reason or "")


def test_regimen_matches_empty_alias_list_accepts_any_regimen() -> None:
    """Mirror the runner: empty alias list means any regimen is acceptable."""
    constraint = FutureEventConstraint(
        constraint_id="any_regimen",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start", ordinal="first"
        ),
        constraint={"regimen_matches": []},
    )
    spec = _empty_spec_with_constraints(constraint)
    ledger = [
        _materialized_event(
            event_id="e1",
            event_date="2024-04-15",
            event_type="systemic_treatment_start",
            regimen_label="anything",
            regimen_components=["x"],
        )
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is True


def test_regimen_matches_fails_cleanly_when_event_has_no_regimen() -> None:
    """Matched event with regimen=None must produce a clean failure reason."""
    constraint = FutureEventConstraint(
        constraint_id="must_match_THP",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start", ordinal="first"
        ),
        constraint={"regimen_matches": ["THP"]},
    )
    spec = _empty_spec_with_constraints(constraint)
    ledger = [
        _materialized_event(
            event_id="missing_regimen",
            event_date="2024-04-15",
            event_type="systemic_treatment_start",
            # NO regimen_label provided → MaterializedLedgerEvent.regimen is None
        )
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is False
    assert "no regimen" in (result.failure_reason or "")
    assert result.matched_event_ids == ("missing_regimen",)


# ---------------------------------------------------------------------------
# event_type_is_one_of.
# ---------------------------------------------------------------------------


def test_event_type_is_one_of_passes_when_type_in_list() -> None:
    constraint = FutureEventConstraint(
        constraint_id="type_check",
        selector=FutureEventSelector(event_type="response"),
        constraint={"event_type_is_one_of": ["response", "progression"]},
    )
    spec = _empty_spec_with_constraints(constraint)
    ledger = [
        _materialized_event(event_id="e1", event_date="2024-04-15", event_type="response"),
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is True


def test_event_type_is_one_of_fails_when_matched_event_type_not_in_list() -> None:
    """The selector picks events of type ``response``; the constraint allows
    only ``progression`` — so the matched event's type fails the membership
    check and the audit reports a clean failure.
    """
    constraint = FutureEventConstraint(
        constraint_id="type_mismatch",
        selector=FutureEventSelector(event_type="response"),
        constraint={"event_type_is_one_of": ["progression"]},
    )
    spec = _empty_spec_with_constraints(constraint)
    ledger = [
        _materialized_event(
            event_id="resp1", event_date="2024-04-15", event_type="response"
        ),
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is False
    assert result.matched_event_ids == ("resp1",)
    assert "event_type_is_one_of" in (result.failure_reason or "")
    assert "'response'" in (result.failure_reason or "")


# ---------------------------------------------------------------------------
# must_occur_within_days — reference: cutoff.
# ---------------------------------------------------------------------------


def test_must_occur_within_days_passes_when_inside_window() -> None:
    constraint = FutureEventConstraint(
        constraint_id="switch_within_90d",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start",
            ordinal_after_cutoff="first",
        ),
        constraint={
            "must_occur_within_days": {"reference": "cutoff", "days_max": 90}
        },
    )
    spec = _empty_spec_with_constraints(constraint, prefix=_prefix("2024-07-15"))
    ledger = [
        _materialized_event(
            event_id="post",
            event_date="2024-09-01",  # 48 days after cutoff
            event_type="systemic_treatment_start",
            regimen_label="T-DM1",
        )
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is True


def test_must_occur_within_days_fails_when_beyond_window() -> None:
    constraint = FutureEventConstraint(
        constraint_id="switch_within_30d",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start",
            ordinal_after_cutoff="first",
        ),
        constraint={
            "must_occur_within_days": {"reference": "cutoff", "days_max": 30}
        },
    )
    spec = _empty_spec_with_constraints(constraint, prefix=_prefix("2024-07-15"))
    ledger = [
        _materialized_event(
            event_id="post",
            event_date="2024-09-01",  # 48 days after cutoff > 30
            event_type="systemic_treatment_start",
            regimen_label="T-DM1",
        )
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is False
    assert "must_occur_within_days" in (result.failure_reason or "")


def test_must_occur_within_days_rejects_pre_cutoff_event() -> None:
    """Non-negative-timing requirement: pre-cutoff matches must fail.

    A pre-cutoff event having a small absolute delta should NOT accidentally
    satisfy a 'within N days of cutoff' constraint.
    """
    constraint = FutureEventConstraint(
        constraint_id="post_cutoff_only",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start",
            ordinal="first",  # global ordinal, can pick pre-cutoff events
        ),
        constraint={
            "must_occur_within_days": {"reference": "cutoff", "days_max": 365}
        },
    )
    spec = _empty_spec_with_constraints(constraint, prefix=_prefix("2024-07-15"))
    ledger = [
        _materialized_event(
            event_id="pre_cutoff",
            event_date="2024-04-15",  # 91 days BEFORE cutoff
            event_type="systemic_treatment_start",
            regimen_label="THP",
        )
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is False
    # Delta is negative; reason includes the negative day count.
    assert "-" in (result.failure_reason or "")


def test_must_occur_within_days_cutoff_raises_when_no_prefix() -> None:
    constraint = FutureEventConstraint(
        constraint_id="needs_cutoff",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start", ordinal="first"
        ),
        constraint={
            "must_occur_within_days": {"reference": "cutoff", "days_max": 90}
        },
    )
    spec = _empty_spec_with_constraints(constraint, prefix=None)
    ledger = [
        _materialized_event(
            event_id="e1",
            event_date="2024-04-15",
            event_type="systemic_treatment_start",
            regimen_label="THP",
        )
    ]
    with pytest.raises(ValueError, match="observed_prefix"):
        future_constraints_audit(spec, ledger)


def test_must_occur_within_days_prior_event_is_deferred_in_a2() -> None:
    constraint = FutureEventConstraint(
        constraint_id="prior_event_deferred",
        selector=FutureEventSelector(
            event_type="systemic_treatment_switch"
        ),
        constraint={
            "must_occur_within_days": {"reference": "prior_event", "days_max": 90}
        },
    )
    spec = _empty_spec_with_constraints(constraint)
    ledger = [
        _materialized_event(
            event_id="e1",
            event_date="2024-04-15",
            event_type="systemic_treatment_switch",
            regimen_label="T-DM1",
        )
    ]
    with pytest.raises(ValueError, match="not implemented in A2"):
        future_constraints_audit(spec, ledger)


def test_must_occur_within_days_rejects_negative_days_max() -> None:
    constraint = FutureEventConstraint(
        constraint_id="bad_days_max",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start", ordinal="first"
        ),
        constraint={
            "must_occur_within_days": {"reference": "cutoff", "days_max": -1}
        },
    )
    spec = _empty_spec_with_constraints(constraint, prefix=_prefix("2024-07-15"))
    ledger = [
        _materialized_event(
            event_id="e1",
            event_date="2024-08-01",
            event_type="systemic_treatment_start",
            regimen_label="THP",
        )
    ]
    with pytest.raises(ValueError, match="days_max must be a non-negative integer"):
        future_constraints_audit(spec, ledger)


# ---------------------------------------------------------------------------
# Typo guard for unknown constraint keys.
# ---------------------------------------------------------------------------


def test_unknown_constraint_key_raises() -> None:
    constraint = FutureEventConstraint(
        constraint_id="typo_test",
        selector=FutureEventSelector(event_type="response"),
        constraint={"reggimen_matches": ["THP"]},  # deliberate typo
    )
    spec = _empty_spec_with_constraints(constraint)
    with pytest.raises(ValueError, match="unknown constraint key"):
        future_constraints_audit(spec, [])


# ---------------------------------------------------------------------------
# Legacy-shape parity — lowered +1L constraint evaluates correctly.
# ---------------------------------------------------------------------------


def test_legacy_lowered_plus_one_l_constraint_passes_on_matching_first_st() -> None:
    """A `+1L` constraint built by ``condition_lower`` evaluates correctly
    on a materialized ledger whose first systemic_treatment_start matches.
    """
    from synthonccases.oncollm.condition_lower import lower_conditioning_cell

    cell = {
        "baseline_seed": {},
        "first_treatment": {
            "regimen_label": "THP",
            "drugs": ["trastuzumab", "pertuzumab", "docetaxel"],
        },
        "text": "",
    }
    spec = lower_conditioning_cell(condition_payload=cell, policy_payload=cell)
    ledger = [
        _materialized_event(
            event_id="first_line",
            event_date="2024-04-15",
            event_type="systemic_treatment_start",
            regimen_label="THP",
            regimen_components=["trastuzumab", "pertuzumab", "docetaxel"],
        )
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is True


def test_legacy_lowered_plus_one_l_constraint_fails_on_mismatching_first_st() -> None:
    """Use FOLFOX as the "mismatching" first line.

    A naive choice like T-DM1 (ado-trastuzumab emtansine) would actually
    *pass* a `+1L=THP` audit because the runner's ``_regimen_matches_alias``
    treats single-component token-superset matches as valid — and
    "trastuzumab" is a token in "ado-trastuzumab emtansine". That's the
    runner's real semantics, mirrored exactly by the audit. FOLFOX
    (5-FU + oxaliplatin + leucovorin) shares no tokens with THP or any of
    the trastuzumab/pertuzumab/docetaxel aliases, so it is unambiguously a
    miss.
    """
    from synthonccases.oncollm.condition_lower import lower_conditioning_cell

    cell = {
        "baseline_seed": {},
        "first_treatment": {
            "regimen_label": "THP",
            "drugs": ["trastuzumab", "pertuzumab", "docetaxel"],
        },
        "text": "",
    }
    spec = lower_conditioning_cell(condition_payload=cell, policy_payload=cell)
    ledger = [
        _materialized_event(
            event_id="wrong_line",
            event_date="2024-04-15",
            event_type="systemic_treatment_start",
            regimen_label="FOLFOX",
            regimen_components=["5fu", "oxaliplatin", "leucovorin"],
        )
    ]
    [result] = future_constraints_audit(spec, ledger)
    assert result.passed is False
    assert "does not match any alias" in (result.failure_reason or "")


# ---------------------------------------------------------------------------
# Parity lock — audit's copied helpers match runner's helpers byte-for-byte.
# ---------------------------------------------------------------------------


def test_normalize_regimen_str_parity_with_runner() -> None:
    samples = [
        "",
        "THP",
        "THP combination",
        "trastuzumab/pertuzumab/docetaxel",
        "T-DM1",
        "ado-trastuzumab emtansine",
        "   capecitabine + tucatinib + trastuzumab  ",
        "FOLFOX-6",
        "carbo,taxol",
        "  ",
    ]
    for s in samples:
        assert audit_normalize(s) == runner_normalize(s), s


def test_regimen_matches_alias_parity_with_runner() -> None:
    cases: list[tuple[str, list[str], str]] = [
        # (observed_label, observed_components, alias)
        ("THP", ["trastuzumab", "pertuzumab", "docetaxel"], "THP"),
        ("THP combination", ["trastuzumab"], "THP"),
        ("THP", ["trastuzumab"], "trastuzumab"),
        ("THP", ["trastuzumab", "pertuzumab"], "T-DM1"),
        ("", ["trastuzumab"], "trastuzumab"),
        ("", [], "anything"),
        ("anything", [], ""),
        ("trastuzumab/pertuzumab/docetaxel", [], "THP"),
        (
            "trastuzumab/pertuzumab/docetaxel",
            ["trastuzumab", "pertuzumab", "docetaxel"],
            "trastuzumab/pertuzumab/docetaxel",
        ),
    ]
    for observed_label, observed_components, alias in cases:
        assert audit_matches(observed_label, observed_components, alias) == (
            runner_matches(observed_label, observed_components, alias)
        ), (observed_label, observed_components, alias)


# ---------------------------------------------------------------------------
# Smoke: result shape lock.
# ---------------------------------------------------------------------------


def test_constraint_match_is_frozen_dataclass() -> None:
    """The result type is a frozen dataclass — locked for B1 derivation use."""
    cm = ConstraintMatch(
        constraint_id="x",
        passed=True,
        matched_event_indices=(0,),
        matched_event_ids=("e",),
        failure_reason=None,
    )
    with pytest.raises(Exception):
        cm.passed = False  # type: ignore[misc]
