"""Gate B1b — ``spec_to_json_dict`` / ``spec_from_json_dict`` round-trip.

The runner persists ``conditioning_spec.json`` per attempt (accepted +
failed) so the derivation layer can reconstruct the spec from disk and
recompute the four ``spec_*`` hashes plus re-run ``future_constraints_audit``
to recover ``constraint_match_ids`` per event. These tests lock the
round-trip invariant — for any spec, every hash helper produces the same
digest on the original and on the reconstructed spec.
"""

from __future__ import annotations

import json
from datetime import date

from synthonccases.oncollm.conditioning_audit import (
    ConditioningSpec,
    FutureEventConstraint,
    FutureEventSelector,
    ObservedPrefix,
    compute_baseline_hash,
    compute_conditioning_spec_hash,
    compute_future_constraint_hash,
    compute_observed_prefix_hash,
    compute_pinned_paths,
    spec_from_json_dict,
    spec_to_json_dict,
)
from synthonccases.oncollm.replay import LedgerEvent


# ---------------------------------------------------------------------------
# Spec builders (mirror tests/unit/test_conditioning_spec_data.py style)
# ---------------------------------------------------------------------------


def _empty_spec() -> ConditioningSpec:
    return ConditioningSpec(
        baseline_seed={},
        pinned_paths=(),
        observed_prefix=None,
        future_event_constraints=(),
        free_text_operator_constraints="",
    )


def _tumor_only_spec() -> ConditioningSpec:
    seed = {"identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}}
    return ConditioningSpec(
        baseline_seed=seed,
        pinned_paths=compute_pinned_paths(seed),
        observed_prefix=None,
        future_event_constraints=(),
        free_text_operator_constraints="prefer adjuvant chemo if eligible",
    )


def _spec_with_two_constraints() -> ConditioningSpec:
    c_a = FutureEventConstraint(
        constraint_id="a_first_treatment",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start", ordinal="first"
        ),
        constraint={"regimen_matches": ["THP"]},
    )
    c_b = FutureEventConstraint(
        constraint_id="b_no_local",
        selector=FutureEventSelector(event_type="local_intervention"),
        constraint={"must_not_occur": True},
    )
    return ConditioningSpec(
        baseline_seed={},
        pinned_paths=(),
        observed_prefix=None,
        future_event_constraints=(c_a, c_b),
        free_text_operator_constraints="",
    )


def _spec_with_prefix_and_constraint() -> ConditioningSpec:
    prefix_event = LedgerEvent.model_validate(
        {
            "date": "2024-04-15",
            "type": "systemic_treatment_start",
            "regimen": {
                "label": "THP",
                "components": ["trastuzumab", "pertuzumab", "docetaxel"],
            },
        }
    )
    constraint = FutureEventConstraint(
        constraint_id="post_cutoff_progression",
        selector=FutureEventSelector(
            event_type="progression", ordinal_after_cutoff="first"
        ),
        constraint={"event_type_is_one_of": ["progression"]},
    )
    return ConditioningSpec(
        baseline_seed={"identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}},
        pinned_paths=("identity_card.diagnosis.oncotree_code",),
        observed_prefix=ObservedPrefix(
            events=(prefix_event,),
            cutoff_date=date(2024, 7, 15),
        ),
        future_event_constraints=(constraint,),
        free_text_operator_constraints="patient prefers oral therapy",
    )


def _spec_with_date_range_selector() -> ConditioningSpec:
    constraint = FutureEventConstraint(
        constraint_id="in_window",
        selector=FutureEventSelector(
            event_type="response",
            date_range=(date(2024, 6, 1), date(2024, 9, 30)),
        ),
        constraint={"event_type_is_one_of": ["response"]},
    )
    return ConditioningSpec(
        baseline_seed={},
        pinned_paths=(),
        observed_prefix=None,
        future_event_constraints=(constraint,),
        free_text_operator_constraints="",
    )


ALL_FIXTURE_SPECS = [
    _empty_spec,
    _tumor_only_spec,
    _spec_with_two_constraints,
    _spec_with_prefix_and_constraint,
    _spec_with_date_range_selector,
]


# ---------------------------------------------------------------------------
# spec_to_json_dict — pure JSON-types output
# ---------------------------------------------------------------------------


def test_spec_to_json_dict_returns_dict() -> None:
    for builder in ALL_FIXTURE_SPECS:
        spec = builder()
        out = spec_to_json_dict(spec)
        assert isinstance(out, dict)


def test_spec_to_json_dict_is_json_serializable() -> None:
    """The dict must serialize to JSON without `default=str` workarounds —
    no stray ``date``, ``tuple``, ``LedgerEvent``, or dataclass left."""
    for builder in ALL_FIXTURE_SPECS:
        spec = builder()
        out = spec_to_json_dict(spec)
        # Will raise TypeError if any non-JSON type sneaks through.
        json.dumps(out)


def test_spec_to_json_dict_byte_stable_across_emissions() -> None:
    """Same spec emitted twice produces byte-identical JSON. Defends the
    'stable file bytes' contract that the on-disk artifact relies on."""
    for builder in ALL_FIXTURE_SPECS:
        spec = builder()
        a = json.dumps(spec_to_json_dict(spec), sort_keys=True)
        b = json.dumps(spec_to_json_dict(spec), sort_keys=True)
        assert a == b


# ---------------------------------------------------------------------------
# Round-trip — reconstruction preserves all four hashes
# ---------------------------------------------------------------------------


def test_round_trip_preserves_baseline_hash() -> None:
    for builder in ALL_FIXTURE_SPECS:
        spec = builder()
        reconstructed = spec_from_json_dict(spec_to_json_dict(spec))
        assert compute_baseline_hash(reconstructed) == compute_baseline_hash(spec)


def test_round_trip_preserves_observed_prefix_hash() -> None:
    for builder in ALL_FIXTURE_SPECS:
        spec = builder()
        reconstructed = spec_from_json_dict(spec_to_json_dict(spec))
        assert (
            compute_observed_prefix_hash(reconstructed)
            == compute_observed_prefix_hash(spec)
        )


def test_round_trip_preserves_future_constraint_hash() -> None:
    for builder in ALL_FIXTURE_SPECS:
        spec = builder()
        reconstructed = spec_from_json_dict(spec_to_json_dict(spec))
        assert (
            compute_future_constraint_hash(reconstructed)
            == compute_future_constraint_hash(spec)
        )


def test_round_trip_preserves_conditioning_spec_hash() -> None:
    for builder in ALL_FIXTURE_SPECS:
        spec = builder()
        reconstructed = spec_from_json_dict(spec_to_json_dict(spec))
        assert (
            compute_conditioning_spec_hash(reconstructed)
            == compute_conditioning_spec_hash(spec)
        )


# ---------------------------------------------------------------------------
# Round-trip — reconstructed spec has the right Python types
# ---------------------------------------------------------------------------


def test_round_trip_reconstructs_dataclasses_and_pydantic_models() -> None:
    spec = _spec_with_prefix_and_constraint()
    reconstructed = spec_from_json_dict(spec_to_json_dict(spec))
    assert isinstance(reconstructed, ConditioningSpec)
    assert isinstance(reconstructed.observed_prefix, ObservedPrefix)
    assert isinstance(reconstructed.observed_prefix.events, tuple)
    assert all(
        isinstance(e, LedgerEvent) for e in reconstructed.observed_prefix.events
    )
    assert isinstance(reconstructed.observed_prefix.cutoff_date, date)
    assert isinstance(reconstructed.future_event_constraints, tuple)
    assert all(
        isinstance(c, FutureEventConstraint)
        for c in reconstructed.future_event_constraints
    )
    assert all(
        isinstance(c.selector, FutureEventSelector)
        for c in reconstructed.future_event_constraints
    )


def test_round_trip_constraint_mapping_round_trips_values() -> None:
    """Reconstructed constraint mapping preserves the original keys/values.
    (Type is plain ``dict`` to match the original construction path and
    keep the dataclass ``asdict``-friendly.)"""
    spec = _spec_with_two_constraints()
    reconstructed = spec_from_json_dict(spec_to_json_dict(spec))
    for original_c, recon_c in zip(
        spec.future_event_constraints,
        reconstructed.future_event_constraints,
        strict=True,
    ):
        assert dict(recon_c.constraint) == dict(original_c.constraint)


def test_round_trip_preserves_pinned_paths_as_tuple() -> None:
    spec = _spec_with_prefix_and_constraint()
    reconstructed = spec_from_json_dict(spec_to_json_dict(spec))
    assert isinstance(reconstructed.pinned_paths, tuple)
    assert reconstructed.pinned_paths == spec.pinned_paths


def test_round_trip_preserves_date_range_as_tuple_of_dates() -> None:
    spec = _spec_with_date_range_selector()
    reconstructed = spec_from_json_dict(spec_to_json_dict(spec))
    selector = reconstructed.future_event_constraints[0].selector
    assert selector.date_range == (date(2024, 6, 1), date(2024, 9, 30))
    assert isinstance(selector.date_range, tuple)
    assert all(isinstance(d, date) for d in selector.date_range)


# ---------------------------------------------------------------------------
# Idempotence: two round-trips equal one round-trip
# ---------------------------------------------------------------------------


def test_double_round_trip_byte_identical() -> None:
    """spec_to(spec_from(spec_to(spec))) == spec_to(spec). Required for the
    derivation layer to be re-runnable without artifact churn."""
    for builder in ALL_FIXTURE_SPECS:
        spec = builder()
        once = spec_to_json_dict(spec)
        twice = spec_to_json_dict(spec_from_json_dict(once))
        assert json.dumps(once, sort_keys=True) == json.dumps(twice, sort_keys=True)


# ---------------------------------------------------------------------------
# Error surface
# ---------------------------------------------------------------------------


def test_spec_from_json_dict_handles_missing_optional_fields() -> None:
    """Empty / minimal JSON dict reconstructs to the empty spec."""
    minimal = {
        "baseline_seed": {},
        "pinned_paths": [],
        "observed_prefix": None,
        "future_event_constraints": [],
        "free_text_operator_constraints": "",
    }
    spec = spec_from_json_dict(minimal)
    assert isinstance(spec, ConditioningSpec)
    assert spec.observed_prefix is None
    assert spec.future_event_constraints == ()
    assert spec.pinned_paths == ()


def test_spec_from_json_dict_defaults_origin_when_missing() -> None:
    """Older spec dumps may not carry ``origin``; default to
    ``"observed_prefix"`` (the only Literal value today)."""
    data = {
        "baseline_seed": {},
        "pinned_paths": [],
        "observed_prefix": {
            "events": [],
            "cutoff_date": "2024-07-15",
            # no "origin" key
        },
        "future_event_constraints": [],
        "free_text_operator_constraints": "",
    }
    spec = spec_from_json_dict(data)
    assert spec.observed_prefix is not None
    assert spec.observed_prefix.origin == "observed_prefix"
