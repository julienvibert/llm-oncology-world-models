"""Legacy event-vocabulary canonicalization in derivation.

Two layers exercised here:

1. ``synthonccases.oncollm.replay.canonicalize_legacy_event_dict`` — the
   shared helper that mirrors ``LedgerEvent._coerce_legacy_event_shape``.
   Covers only the rewrites ``LedgerEvent`` already accepts; unmappable
   legacy types (e.g. ``clinical_decision``) are passed through unchanged
   so the Pydantic validator still rejects them.
2. The derivation-local adapters — ``_canonicalize_event_for_derivation``
   and ``_canonicalize_snapshot_anchor_event_type`` — which apply the
   shared helper AND additionally null-type any type that is not in the
   canonical ``EventType`` enum. The raw original is preserved via
   ``materialized_payload_json`` so provenance is recoverable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from synthonccases.oncollm.derivation import (
    CellKey,
    TrajectoryArtifact,
    _canonicalize_event_for_derivation,
    _canonicalize_snapshot_anchor_event_type,
    build_event_rows,
    build_snapshot_rows,
)
from synthonccases.oncollm.replay import canonicalize_legacy_event_dict


# ---------------------------------------------------------------------------
# Shared replay helper: canonicalize_legacy_event_dict
# ---------------------------------------------------------------------------


def test_canonicalize_maps_legacy_treatment_lifecycle_types() -> None:
    assert canonicalize_legacy_event_dict({"type": "treatment_start"})["type"] == (
        "systemic_treatment_start"
    )
    assert canonicalize_legacy_event_dict({"type": "treatment_switch"})["type"] == (
        "systemic_treatment_switch"
    )
    assert canonicalize_legacy_event_dict({"type": "treatment_stop"})["type"] == (
        "systemic_treatment_stop"
    )


def test_canonicalize_maps_legacy_molecular_and_pro_types() -> None:
    assert canonicalize_legacy_event_dict({"type": "molecular_lead_event"})["type"] == (
        "clonal_evolution_event"
    )
    assert canonicalize_legacy_event_dict({"type": "pro_update"})["type"] == (
        "patient_state_update"
    )


def test_canonicalize_surgery_sets_modality_local_intervention() -> None:
    result = canonicalize_legacy_event_dict({"type": "surgery"})
    assert result["type"] == "local_intervention"
    assert result["modality"] == "surgery"


def test_canonicalize_toxicity_renames_ctcae_grade_and_defaults_transition() -> None:
    result = canonicalize_legacy_event_dict({
        "type": "toxicity",
        "ctcae_grade": 3,
        "resolution_date": "2024-09-01",
        "current_status": "ongoing",
        "attributed_to": "trastuzumab",
    })
    assert result.get("ctcae_v5_grade") == 3
    assert "ctcae_grade" not in result
    assert result.get("toxicity_transition") == "onset"
    assert "resolution_date" not in result
    assert "current_status" not in result
    assert "attributed_to" not in result


def test_canonicalize_canonical_type_passes_through_unchanged() -> None:
    canonical = {"type": "progression", "date": "2024-08-01"}
    result = canonicalize_legacy_event_dict(canonical)
    assert result["type"] == "progression"
    assert result["date"] == "2024-08-01"


def test_canonicalize_unmappable_type_is_passed_through_for_pydantic_to_reject() -> None:
    """`canonicalize_legacy_event_dict` must NOT drop unmappable types — that
    would silently neutralize legitimate LedgerEvent rejections elsewhere.
    The derivation-local adapter is responsible for null-typing in CSVs."""
    result = canonicalize_legacy_event_dict({"type": "clinical_decision"})
    assert result["type"] == "clinical_decision"


def test_canonicalize_does_not_mutate_input() -> None:
    src = {"type": "treatment_start", "date": "2024-08-01"}
    canonicalize_legacy_event_dict(src)
    assert src["type"] == "treatment_start"


# ---------------------------------------------------------------------------
# Derivation-local adapter: _canonicalize_event_for_derivation
# ---------------------------------------------------------------------------


def test_event_adapter_unmappable_type_resolves_to_null() -> None:
    result = _canonicalize_event_for_derivation({"type": "clinical_decision"})
    assert result.get("type") is None


def test_event_adapter_mapped_legacy_type_canonicalizes() -> None:
    result = _canonicalize_event_for_derivation({"type": "molecular_lead_event"})
    assert result.get("type") == "clonal_evolution_event"


def test_event_adapter_canonical_type_passes_through() -> None:
    result = _canonicalize_event_for_derivation({
        "type": "progression", "date": "2024-08-01",
    })
    assert result.get("type") == "progression"


# ---------------------------------------------------------------------------
# Snapshot anchor adapter: _canonicalize_snapshot_anchor_event_type
# ---------------------------------------------------------------------------


def test_snapshot_anchor_baseline_preserved() -> None:
    assert _canonicalize_snapshot_anchor_event_type("baseline") == "baseline"


def test_snapshot_anchor_legacy_type_canonicalizes() -> None:
    assert (
        _canonicalize_snapshot_anchor_event_type("treatment_start")
        == "systemic_treatment_start"
    )


def test_snapshot_anchor_unmappable_type_resolves_to_none() -> None:
    assert _canonicalize_snapshot_anchor_event_type("clinical_decision") is None


def test_snapshot_anchor_unknown_string_resolves_to_none() -> None:
    assert _canonicalize_snapshot_anchor_event_type("not_a_real_type_xyz") is None


def test_snapshot_anchor_non_string_resolves_to_none() -> None:
    assert _canonicalize_snapshot_anchor_event_type(None) is None
    assert _canonicalize_snapshot_anchor_event_type(42) is None


# ---------------------------------------------------------------------------
# Row-level integration: build_event_rows preserves row count + provenance
# ---------------------------------------------------------------------------


def _cell() -> CellKey:
    return CellKey(
        trajectory_id="t1", condition_id="c", policy_id="p",
        model_id="m", model_short="m", future_index=1,
        tumor_oncotree_code="BRCA", conditioning_label="x",
        conditioning_hash="h", accepted=True,
    )


def _artifact_with_ledger(ledger: list[dict]) -> TrajectoryArtifact:
    materialized = {
        "baseline": {}, "ledger": ledger,
        "snapshots": [], "treatment_history": [],
    }
    return TrajectoryArtifact(
        cell=_cell(), trajectory_dir=Path("/tmp"),
        parsed_ledger=(), materialized=materialized,
        trajectory_result={},
        conditioning_spec=None, constraint_results=None,
    )


def test_event_rows_legacy_pilot_ledger_canonicalizes_and_preserves_row_count() -> None:
    ledger = [
        {"type": "treatment_start", "date": "2024-08-01",
         "event_id": "e1", "regimen": {"label": "THP", "components": ["x"]}},
        {"type": "molecular_lead_event", "date": "2024-09-01",
         "event_id": "e2"},
        {"type": "clinical_decision", "date": "2024-10-01",
         "event_id": "e3", "decision_type": "treatment_escalation"},
        {"type": "pro_update", "date": "2024-10-15", "event_id": "e4"},
        {"type": "progression", "date": "2024-11-01", "event_id": "e5"},
    ]
    df = build_event_rows([_artifact_with_ledger(ledger)])
    assert len(df) == 5, "row count must be preserved across all input events"
    # pandas converts None → NaN in object columns; compare non-null values
    # by equality and null position by isna().
    types = df["event_type"]
    assert types.iloc[0] == "systemic_treatment_start"
    assert types.iloc[1] == "clonal_evolution_event"
    assert pd.isna(types.iloc[2])      # clinical_decision → null
    assert types.iloc[3] == "patient_state_update"
    assert types.iloc[4] == "progression"


def test_event_rows_clinical_decision_provenance_recoverable_from_payload() -> None:
    """When event_type is null-typed for clinical_decision, the raw original
    type ('clinical_decision') and any associated fields (e.g. decision_type)
    must remain recoverable from `materialized_payload_json`."""
    ledger = [
        {"type": "clinical_decision", "date": "2024-10-01",
         "event_id": "e_cd", "decision_type": "treatment_escalation",
         "clinical_context": "rising_LDH"},
    ]
    df = build_event_rows([_artifact_with_ledger(ledger)])
    assert len(df) == 1
    assert pd.isna(df["event_type"].iloc[0])
    payload = json.loads(df["materialized_payload_json"].iloc[0])
    assert payload["type"] == "clinical_decision"
    assert payload["decision_type"] == "treatment_escalation"
    assert payload["clinical_context"] == "rising_LDH"


def test_event_rows_legacy_treatment_lifecycle_preserves_emitted_line() -> None:
    """The shared replay canonicalizer drops the legacy `line` field so that
    LedgerEvent validates without it, but derivation must preserve the
    model's emitted line for provenance in event_rows."""
    ledger = [
        {"type": "treatment_start", "date": "2024-08-01",
         "event_id": "e1", "line": 1,
         "regimen": {"label": "THP", "components": ["x"]}},
        {"type": "treatment_switch", "date": "2025-01-01",
         "event_id": "e2", "line": 2,
         "regimen": {"label": "T-DM1", "components": ["y"]}},
        {"type": "treatment_stop", "date": "2025-06-01",
         "event_id": "e3", "line": 2},
    ]
    df = build_event_rows([_artifact_with_ledger(ledger)])
    assert len(df) == 3
    assert df["event_type"].iloc[0] == "systemic_treatment_start"
    assert df["line"].iloc[0] == 1
    assert df["line"].iloc[1] == 2
    assert df["line"].iloc[2] == 2


def test_event_rows_legacy_toxicity_ctcae_grade_promoted_in_canonical_column() -> None:
    ledger = [
        {"type": "toxicity", "date": "2024-09-15", "event_id": "e_tox",
         "ae": "neutropenia", "ctcae_grade": 4, "ae_category": "hematologic"},
    ]
    df = build_event_rows([_artifact_with_ledger(ledger)])
    assert df["ctcae_v5_grade"].iloc[0] == 4
    assert df["toxicity_transition"].iloc[0] == "onset"


# ---------------------------------------------------------------------------
# Row-level integration: build_snapshot_rows anchor_event_type canonicalization
# ---------------------------------------------------------------------------


def _artifact_with_snapshots(snapshots: list[dict]) -> TrajectoryArtifact:
    materialized = {
        "baseline": {}, "ledger": [],
        "snapshots": snapshots, "treatment_history": [],
    }
    return TrajectoryArtifact(
        cell=_cell(), trajectory_dir=Path("/tmp"),
        parsed_ledger=(), materialized=materialized,
        trajectory_result={},
        conditioning_spec=None, constraint_results=None,
    )


def test_snapshot_rows_anchor_event_type_canonicalized_across_legacy_inputs() -> None:
    snapshots = [
        {"event_type": "baseline", "state_id": "s_base",
         "date": "2024-07-15", "patient_state": {}, "disease_state": []},
        {"event_type": "treatment_start", "state_id": "s_tx",
         "date": "2024-08-01", "patient_state": {}, "disease_state": []},
        {"event_type": "molecular_lead_event", "state_id": "s_mol",
         "date": "2024-09-01", "patient_state": {}, "disease_state": []},
        {"event_type": "clinical_decision", "state_id": "s_cd",
         "date": "2024-10-01", "patient_state": {}, "disease_state": []},
        {"event_type": "progression", "state_id": "s_prog",
         "date": "2024-11-01", "patient_state": {}, "disease_state": []},
    ]
    df = build_snapshot_rows([_artifact_with_snapshots(snapshots)])
    assert len(df) == 5
    anchors = df["anchor_event_type"]
    assert anchors.iloc[0] == "baseline"
    assert anchors.iloc[1] == "systemic_treatment_start"
    assert anchors.iloc[2] == "clonal_evolution_event"
    assert pd.isna(anchors.iloc[3])    # clinical_decision → null
    assert anchors.iloc[4] == "progression"
