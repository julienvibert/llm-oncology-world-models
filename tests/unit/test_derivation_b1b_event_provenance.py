"""Gate B1b — ``event_source`` and ``constraint_match_ids`` columns on
event_rows.csv.

Locks: index-based ``event_source`` assignment against
``len(spec.observed_prefix.events)``; ``constraint_match_ids`` inversion
from ``ConstraintMatch.matched_event_ids`` to JSON-encoded sorted list
per ``event_id``; sentinel backfill when no spec is on disk.
"""

from __future__ import annotations

from pathlib import Path

from synthonccases.oncollm.conditioning_audit import (
    ConditioningSpec,
    ConstraintMatch,
    ObservedPrefix,
)
from synthonccases.oncollm.derivation import (
    CellKey,
    EVENT_SOURCE_GENERATED,
    EVENT_SOURCE_OBSERVED_PREFIX,
    TrajectoryArtifact,
    _constraint_match_ids_by_event_id,
    _observed_prefix_event_count,
    build_event_rows,
)
from synthonccases.oncollm.replay import LedgerEvent


def _make_spec_with_prefix(n_prefix_events: int) -> ConditioningSpec:
    events = tuple(
        LedgerEvent.model_validate({
            "date": f"2024-03-{15 + i:02d}",
            "type": "systemic_treatment_start",
            "regimen": {"label": "THP",
                        "components": ["trastuzumab", "pertuzumab", "docetaxel"]},
        })
        for i in range(n_prefix_events)
    )
    from datetime import date
    return ConditioningSpec(
        baseline_seed={}, pinned_paths=(),
        observed_prefix=ObservedPrefix(events=events, cutoff_date=date(2024, 7, 15)),
        future_event_constraints=(),
        free_text_operator_constraints="",
    )


def _make_artifact(
    *,
    ledger: list[dict],
    conditioning_spec: ConditioningSpec | None = None,
    constraint_results: tuple[ConstraintMatch, ...] | None = None,
) -> TrajectoryArtifact:
    cell = CellKey(
        trajectory_id="t1", condition_id="c", policy_id="p",
        model_id="m", model_short="m", future_index=1,
        tumor_oncotree_code="BRCA", conditioning_label="x",
        conditioning_hash="h", accepted=True,
    )
    materialized = {
        "baseline": {}, "ledger": ledger,
        "snapshots": [], "treatment_history": [],
    }
    return TrajectoryArtifact(
        cell=cell, trajectory_dir=Path("/tmp"),
        parsed_ledger=(), materialized=materialized,
        trajectory_result={},
        conditioning_spec=conditioning_spec,
        constraint_results=constraint_results,
    )


# ---------------------------------------------------------------------------
# _observed_prefix_event_count helper
# ---------------------------------------------------------------------------


def test_observed_prefix_event_count_with_prefix() -> None:
    spec = _make_spec_with_prefix(3)
    art = _make_artifact(ledger=[], conditioning_spec=spec)
    assert _observed_prefix_event_count(art) == 3


def test_observed_prefix_event_count_no_spec_is_zero() -> None:
    art = _make_artifact(ledger=[], conditioning_spec=None)
    assert _observed_prefix_event_count(art) == 0


def test_observed_prefix_event_count_no_prefix_is_zero() -> None:
    """Spec with observed_prefix=None → 0."""
    from datetime import date  # noqa: F401
    spec = ConditioningSpec(
        baseline_seed={}, pinned_paths=(), observed_prefix=None,
        future_event_constraints=(), free_text_operator_constraints="",
    )
    art = _make_artifact(ledger=[], conditioning_spec=spec)
    assert _observed_prefix_event_count(art) == 0


# ---------------------------------------------------------------------------
# _constraint_match_ids_by_event_id helper
# ---------------------------------------------------------------------------


def test_constraint_match_ids_by_event_id_inverts_mapping() -> None:
    """ConstraintMatch.matched_event_ids inverted per event_id, sorted."""
    matches = (
        ConstraintMatch(
            constraint_id="zebra_first",
            passed=True,
            matched_event_indices=(0,),
            matched_event_ids=("ev1",),
            failure_reason=None,
        ),
        ConstraintMatch(
            constraint_id="alpha_second",
            passed=True,
            matched_event_indices=(1, 2),
            matched_event_ids=("ev1", "ev2"),  # ev1 matched by BOTH constraints
            failure_reason=None,
        ),
    )
    art = _make_artifact(ledger=[], constraint_results=matches)
    inverted = _constraint_match_ids_by_event_id(art)
    # ev1 matched by both, sorted alphabetically.
    assert inverted["ev1"] == ["alpha_second", "zebra_first"]
    # ev2 matched only by alpha_second.
    assert inverted["ev2"] == ["alpha_second"]


def test_constraint_match_ids_by_event_id_none_is_empty_dict() -> None:
    art = _make_artifact(ledger=[], constraint_results=None)
    assert _constraint_match_ids_by_event_id(art) == {}


# ---------------------------------------------------------------------------
# build_event_rows — event_source + constraint_match_ids columns
# ---------------------------------------------------------------------------


def test_event_rows_event_source_index_based_against_prefix_n() -> None:
    """First N materialized events are observed_prefix (1-indexed
    1..N), the rest are generated."""
    spec = _make_spec_with_prefix(2)  # 2 prefix events
    ledger = [
        {"event_id": "p1", "type": "systemic_treatment_start", "date": "2024-03-15"},
        {"event_id": "p2", "type": "systemic_treatment_start", "date": "2024-03-16"},
        {"event_id": "g1", "type": "response", "date": "2024-08-01"},
        {"event_id": "g2", "type": "progression", "date": "2024-10-01"},
    ]
    art = _make_artifact(ledger=ledger, conditioning_spec=spec)
    df = build_event_rows([art])
    assert list(df["event_source"]) == [
        EVENT_SOURCE_OBSERVED_PREFIX,
        EVENT_SOURCE_OBSERVED_PREFIX,
        EVENT_SOURCE_GENERATED,
        EVENT_SOURCE_GENERATED,
    ]


def test_event_rows_no_spec_all_generated_sentinel() -> None:
    """Legacy sentinel path: no conditioning_spec on artifact → every
    event reports event_source='generated'."""
    ledger = [
        {"event_id": "e1", "type": "response", "date": "2024-08-01"},
        {"event_id": "e2", "type": "progression", "date": "2024-10-01"},
    ]
    art = _make_artifact(ledger=ledger, conditioning_spec=None)
    df = build_event_rows([art])
    assert all(s == EVENT_SOURCE_GENERATED for s in df["event_source"])


def test_event_rows_no_prefix_all_generated() -> None:
    """Spec with no observed_prefix → all events generated."""
    spec = ConditioningSpec(
        baseline_seed={}, pinned_paths=(), observed_prefix=None,
        future_event_constraints=(), free_text_operator_constraints="",
    )
    ledger = [{"event_id": "e1", "type": "response", "date": "2024-08-01"}]
    art = _make_artifact(ledger=ledger, conditioning_spec=spec)
    df = build_event_rows([art])
    assert df.iloc[0]["event_source"] == EVENT_SOURCE_GENERATED


def test_event_rows_constraint_match_ids_json_sorted_list() -> None:
    """constraint_match_ids cell is a JSON-encoded sorted list of
    constraint_ids whose matched_event_ids contains this event."""
    matches = (
        ConstraintMatch(
            constraint_id="zebra",
            passed=True,
            matched_event_indices=(0,),
            matched_event_ids=("e1",),
            failure_reason=None,
        ),
        ConstraintMatch(
            constraint_id="alpha",
            passed=True,
            matched_event_indices=(0,),
            matched_event_ids=("e1",),
            failure_reason=None,
        ),
    )
    ledger = [
        {"event_id": "e1", "type": "response", "date": "2024-08-01"},
        {"event_id": "e2", "type": "progression", "date": "2024-10-01"},
    ]
    art = _make_artifact(ledger=ledger, constraint_results=matches)
    df = build_event_rows([art])
    import json
    # e1 matched by both, sorted alphabetically.
    assert json.loads(df.iloc[0]["constraint_match_ids"]) == ["alpha", "zebra"]
    # e2 matched by none → empty list.
    assert json.loads(df.iloc[1]["constraint_match_ids"]) == []


def test_event_rows_sentinel_constraint_match_ids_when_no_results() -> None:
    """No constraint_results on artifact → '[]' for every event."""
    ledger = [{"event_id": "e1", "type": "response", "date": "2024-08-01"}]
    art = _make_artifact(ledger=ledger, constraint_results=None)
    df = build_event_rows([art])
    assert df.iloc[0]["constraint_match_ids"] == "[]"
