"""Gate A3 — observed-prefix preflight + collision detection.

Tests cover:

- ``validate_observed_prefix``: no-op when no prefix; empty events tuple;
  rejects event after cutoff; rejects event before diagnosis (when given);
  rejects out-of-order dates; allows same-day duplicates (non-decreasing
  is fine); diagnosis-date check is skipped when caller passes ``None``.
- ``assert_no_prefix_collision``: no-op when no prefix; empty emitted
  ledger returns ``[]``; flags emitted event at cutoff (boundary collision);
  flags emitted event before cutoff; flags emitted event semantically
  identical to a prefix event; does NOT flag same-date same-type emitted
  event that differs in other fields; does NOT flag a same-type emitted
  event on a later date; aggregates multiple violations.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from synthonccases.oncollm.conditioning_audit import (
    ConditioningSpec,
    ObservedPrefix,
    ObservedPrefixError,
    PrefixCollision,
    _event_semantic_key,
    assert_no_prefix_collision,
    validate_observed_prefix,
)
from synthonccases.oncollm.replay import LedgerEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evt(date_iso: str, event_type: str, **extra: Any) -> LedgerEvent:
    payload: dict[str, Any] = {"date": date_iso, "type": event_type}
    payload.update(extra)
    return LedgerEvent.model_validate(payload)


def _treatment_event(date_iso: str, *, label: str, components: list[str]) -> LedgerEvent:
    return LedgerEvent.model_validate(
        {
            "date": date_iso,
            "type": "systemic_treatment_start",
            "regimen": {"label": label, "components": components},
        }
    )


def _spec(
    *, prefix: ObservedPrefix | None = None
) -> ConditioningSpec:
    return ConditioningSpec(
        baseline_seed={},
        pinned_paths=(),
        observed_prefix=prefix,
        future_event_constraints=(),
        free_text_operator_constraints="",
    )


# ---------------------------------------------------------------------------
# validate_observed_prefix
# ---------------------------------------------------------------------------


def test_validate_observed_prefix_noop_when_no_prefix() -> None:
    """No prefix on the spec → silent return, no exceptions."""
    spec = _spec(prefix=None)
    validate_observed_prefix(spec, baseline_diagnosis_date=date(2024, 1, 15))


def test_validate_observed_prefix_noop_when_empty_events() -> None:
    spec = _spec(prefix=ObservedPrefix(events=(), cutoff_date=date(2024, 7, 15)))
    validate_observed_prefix(spec, baseline_diagnosis_date=date(2024, 1, 15))


def test_validate_observed_prefix_accepts_event_at_cutoff_boundary() -> None:
    """Event date == cutoff_date is allowed (cutoff is inclusive for prefix)."""
    spec = _spec(
        prefix=ObservedPrefix(
            events=(_evt("2024-07-15", "response"),),
            cutoff_date=date(2024, 7, 15),
        )
    )
    validate_observed_prefix(spec, baseline_diagnosis_date=date(2024, 1, 15))


def test_validate_observed_prefix_accepts_event_at_diagnosis_boundary() -> None:
    """Event date == baseline_diagnosis_date is allowed."""
    spec = _spec(
        prefix=ObservedPrefix(
            events=(_evt("2024-01-15", "response"),),
            cutoff_date=date(2024, 7, 15),
        )
    )
    validate_observed_prefix(spec, baseline_diagnosis_date=date(2024, 1, 15))


def test_validate_observed_prefix_rejects_event_after_cutoff() -> None:
    spec = _spec(
        prefix=ObservedPrefix(
            events=(_evt("2024-08-01", "response"),),
            cutoff_date=date(2024, 7, 15),
        )
    )
    with pytest.raises(ObservedPrefixError) as exc:
        validate_observed_prefix(spec, baseline_diagnosis_date=date(2024, 1, 15))
    assert exc.value.path == "observed_prefix.events[0].date"
    assert "after cutoff_date" in exc.value.reason


def test_validate_observed_prefix_rejects_event_before_diagnosis() -> None:
    spec = _spec(
        prefix=ObservedPrefix(
            events=(_evt("2023-12-31", "response"),),
            cutoff_date=date(2024, 7, 15),
        )
    )
    with pytest.raises(ObservedPrefixError) as exc:
        validate_observed_prefix(spec, baseline_diagnosis_date=date(2024, 1, 15))
    assert exc.value.path == "observed_prefix.events[0].date"
    assert "before baseline diagnosis_date" in exc.value.reason


def test_validate_observed_prefix_rejects_out_of_order_dates() -> None:
    spec = _spec(
        prefix=ObservedPrefix(
            events=(
                _evt("2024-04-15", "systemic_treatment_start", regimen={
                    "label": "THP", "components": ["trastuzumab"],
                }),
                _evt("2024-03-01", "response"),  # earlier than previous
            ),
            cutoff_date=date(2024, 7, 15),
        )
    )
    with pytest.raises(ObservedPrefixError) as exc:
        validate_observed_prefix(spec, baseline_diagnosis_date=date(2024, 1, 15))
    assert exc.value.path == "observed_prefix.events[1].date"
    assert "before the previous event" in exc.value.reason


def test_validate_observed_prefix_allows_same_day_events() -> None:
    """Non-decreasing — ties are fine. Two events recorded on the same date."""
    spec = _spec(
        prefix=ObservedPrefix(
            events=(
                _evt("2024-04-15", "response"),
                _evt("2024-04-15", "patient_state_update"),
            ),
            cutoff_date=date(2024, 7, 15),
        )
    )
    validate_observed_prefix(spec, baseline_diagnosis_date=date(2024, 1, 15))


def test_validate_observed_prefix_skips_diagnosis_check_when_none() -> None:
    """``baseline_diagnosis_date=None`` skips the lower-bound check (e.g.,
    during fixture construction before StudyReference is applied).
    """
    spec = _spec(
        prefix=ObservedPrefix(
            events=(_evt("2020-01-01", "response"),),  # arbitrarily old
            cutoff_date=date(2024, 7, 15),
        )
    )
    validate_observed_prefix(spec, baseline_diagnosis_date=None)


# ---------------------------------------------------------------------------
# _event_semantic_key — identity + sensitivity properties.
# ---------------------------------------------------------------------------


def test_event_semantic_key_identical_events_share_key() -> None:
    e1 = _treatment_event(
        "2024-04-15", label="THP", components=["trastuzumab", "pertuzumab"]
    )
    e2 = _treatment_event(
        "2024-04-15", label="THP", components=["trastuzumab", "pertuzumab"]
    )
    assert _event_semantic_key(e1) == _event_semantic_key(e2)


def test_event_semantic_key_differs_on_regimen_label() -> None:
    e1 = _treatment_event("2024-04-15", label="THP", components=["trastuzumab"])
    e2 = _treatment_event("2024-04-15", label="T-DM1", components=["trastuzumab"])
    assert _event_semantic_key(e1) != _event_semantic_key(e2)


def test_event_semantic_key_drops_irrelevant_none_fields() -> None:
    """Adding a None-valued optional field after the fact should NOT change
    the key — ``exclude_none=True`` drops Nones from the dump.
    """
    e1 = _evt("2024-04-15", "response")
    e2 = _evt("2024-04-15", "response", patient=None)
    assert _event_semantic_key(e1) == _event_semantic_key(e2)


# ---------------------------------------------------------------------------
# assert_no_prefix_collision
# ---------------------------------------------------------------------------


def test_collision_noop_when_no_prefix() -> None:
    spec = _spec(prefix=None)
    emitted = [_evt("2024-09-01", "response")]
    assert assert_no_prefix_collision(spec, emitted) == []


def test_collision_empty_emitted_ledger_returns_empty() -> None:
    spec = _spec(
        prefix=ObservedPrefix(
            events=(_evt("2024-04-15", "response"),),
            cutoff_date=date(2024, 7, 15),
        )
    )
    assert assert_no_prefix_collision(spec, []) == []


def test_collision_clean_ledger_returns_empty() -> None:
    """Emitted events strictly after cutoff, no semantic duplicates → []."""
    spec = _spec(
        prefix=ObservedPrefix(
            events=(_treatment_event(
                "2024-04-15", label="THP", components=["trastuzumab"]
            ),),
            cutoff_date=date(2024, 7, 15),
        )
    )
    emitted = [
        _evt("2024-09-01", "response"),
        _evt("2024-10-01", "progression"),
    ]
    assert assert_no_prefix_collision(spec, emitted) == []


def test_collision_flags_emitted_at_cutoff_boundary() -> None:
    """date == cutoff_date is a collision: emitted ledger must be strictly after."""
    spec = _spec(
        prefix=ObservedPrefix(
            events=(_evt("2024-04-15", "response"),),
            cutoff_date=date(2024, 7, 15),
        )
    )
    emitted = [_evt("2024-07-15", "response")]
    collisions = assert_no_prefix_collision(spec, emitted)
    assert len(collisions) == 1
    assert collisions[0].reason == "emitted_event_before_or_at_cutoff"
    assert collisions[0].emitted_index == 0
    assert collisions[0].prefix_index is None
    assert "not strictly after cutoff_date" in collisions[0].detail


def test_collision_flags_emitted_before_cutoff() -> None:
    spec = _spec(
        prefix=ObservedPrefix(
            events=(_evt("2024-04-15", "response"),),
            cutoff_date=date(2024, 7, 15),
        )
    )
    emitted = [_evt("2024-06-01", "response")]
    collisions = assert_no_prefix_collision(spec, emitted)
    assert len(collisions) == 1
    assert collisions[0].reason == "emitted_event_before_or_at_cutoff"


def test_collision_well_formed_spec_does_not_flag_same_content_different_date() -> None:
    """In a well-formed spec (prefix.dates <= cutoff < emitted.dates), an
    emitted event sharing CONTENT with a prefix event but on a different
    date does NOT collide — the date is part of the semantic key.

    This is the expected normal case for a properly-validated spec: the
    semantic-duplicate rule never fires here because dates always differ
    between prefix and emitted events.
    """
    prefix_event = _treatment_event(
        "2024-04-15", label="THP", components=["trastuzumab"]
    )
    spec = _spec(
        prefix=ObservedPrefix(
            events=(prefix_event,),
            cutoff_date=date(2024, 7, 15),
        )
    )
    # Same type + same regimen content as the prefix event, but on a
    # different (post-cutoff) date.
    emitted = [_treatment_event(
        "2024-09-01", label="THP", components=["trastuzumab"]
    )]
    assert assert_no_prefix_collision(spec, emitted) == []


def test_collision_duplicate_rule_fires_on_malformed_spec_defense_in_depth() -> None:
    """Defense-in-depth: the semantic-duplicate rule fires when the spec is
    malformed (prefix event after cutoff — a case ``validate_observed_prefix``
    would reject at preflight).

    ``assert_no_prefix_collision`` operates independently of preflight, so
    if a future code path constructs a spec without validation, the model
    cannot silently restate a prefix event. The test uses a malformed
    spec where ``prefix_event.date > cutoff_date`` so the emitted event
    with the same date + content passes the date-floor rule but trips
    the duplicate rule.
    """
    prefix_event = _treatment_event(
        "2024-09-01", label="THP", components=["trastuzumab"]
    )
    spec = _spec(
        prefix=ObservedPrefix(
            events=(prefix_event,),
            # Deliberately malformed: prefix event date > cutoff_date.
            # validate_observed_prefix would reject this spec.
            cutoff_date=date(2024, 7, 15),
        )
    )
    emitted = [_treatment_event(
        "2024-09-01", label="THP", components=["trastuzumab"]
    )]
    collisions = assert_no_prefix_collision(spec, emitted)
    assert len(collisions) == 1
    assert collisions[0].reason == "emitted_event_duplicates_prefix"
    assert collisions[0].emitted_index == 0
    assert collisions[0].prefix_index == 0
    assert "semantically identical" in collisions[0].detail


def test_collision_does_not_flag_same_date_different_type() -> None:
    """Emitted event on the same date as a prefix event but with a different
    type produces a different semantic key, so it's not a collision (and the
    date-floor rule is irrelevant when both dates are <= cutoff for the
    prefix and the emitted event has its own valid date)."""
    spec = _spec(
        prefix=ObservedPrefix(
            events=(_evt("2024-04-15", "response"),),
            cutoff_date=date(2024, 7, 15),
        )
    )
    emitted = [_evt("2024-09-01", "progression")]
    assert assert_no_prefix_collision(spec, emitted) == []


def test_collision_does_not_flag_same_type_different_content() -> None:
    """Emitted treatment event with same type but a different regimen does
    not share a semantic key with a prefix treatment event.
    """
    spec = _spec(
        prefix=ObservedPrefix(
            events=(_treatment_event(
                "2024-04-15", label="THP", components=["trastuzumab"]
            ),),
            cutoff_date=date(2024, 7, 15),
        )
    )
    emitted = [_treatment_event(
        "2024-09-01", label="T-DM1", components=["ado-trastuzumab emtansine"]
    )]
    assert assert_no_prefix_collision(spec, emitted) == []


def test_collision_aggregates_multiple_violations() -> None:
    """Returns ALL collisions, not just the first — caller assembles the
    full failure message.
    """
    spec = _spec(
        prefix=ObservedPrefix(
            events=(_evt("2024-04-15", "response"),),
            cutoff_date=date(2024, 7, 15),
        )
    )
    emitted = [
        _evt("2024-06-01", "response"),     # before cutoff
        _evt("2024-09-01", "progression"),  # ok
        _evt("2024-07-15", "patient_state_update"),  # at cutoff
        _evt("2024-10-01", "death"),  # ok
    ]
    collisions = assert_no_prefix_collision(spec, emitted)
    assert len(collisions) == 2
    assert all(
        c.reason == "emitted_event_before_or_at_cutoff" for c in collisions
    )
    assert {c.emitted_index for c in collisions} == {0, 2}


# ---------------------------------------------------------------------------
# Shape lock.
# ---------------------------------------------------------------------------


def test_prefix_collision_is_frozen_dataclass() -> None:
    pc = PrefixCollision(
        reason="emitted_event_before_or_at_cutoff",
        emitted_index=0,
        prefix_index=None,
        detail="...",
    )
    with pytest.raises(Exception):
        pc.reason = "x"  # type: ignore[misc]
