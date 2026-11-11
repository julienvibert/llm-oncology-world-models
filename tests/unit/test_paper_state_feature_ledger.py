"""Tests for the pre-PCA clinical-state feature ledger."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_paper_state_feature_ledger as ledger  # noqa: E402


def test_semantic_ledger_keeps_non_state_roles_out_of_map() -> None:
    entries = ledger._entries()

    assert any(entry.role == "C" and entry.object_id == "cancer_identity" for entry in entries)
    assert any(entry.role == "H" and entry.object_id == "complete_history" for entry in entries)
    assert any(entry.role == "A" and entry.object_id == "systemic_action" for entry in entries)
    assert any(entry.role == "E" and entry.object_id == "death_exit" for entry in entries)
    assert all(
        "never a PCA coordinate" in entry.map_status
        or entry.role in {"X", "X_or_C"}
        for entry in entries
    )


def test_site_features_distinguish_entries_sites_and_compartments() -> None:
    sites = pd.DataFrame(
        {
            "_gid": ["g", "g", "g", "g"],
            "snapshot_index": [0, 0, 0, 1],
            "site_class_canonical": ["brain", "brain", "liver", "brain"],
            "site_signature": ["brain-a", "brain-a", "liver-a", "brain-a"],
            "burden_band": ["small", "moderate", "bulky", "microscopic"],
        }
    )

    derived, site_columns = ledger.derive_site_features(sites)
    baseline = derived.set_index("snapshot_index").loc[0]

    assert site_columns == ("site__brain", "site__liver")
    assert baseline["n_unique_anatomic_sites"] == 2
    assert baseline["n_site_compartments"] == 2
    assert baseline["site__brain"] == 3
    assert baseline["site__liver"] == 4


def test_state_candidates_use_relative_weight_and_keep_cns() -> None:
    snapshots = pd.DataFrame(
        {
            "_gid": ["g", "g"],
            "snapshot_index": [0, 1],
            "vital_status": ["alive", "alive"],
            "outcome_type": [None, None],
            "total_burden_band": ["small", "moderate"],
            "num_active_disease_sites": [1, 2],
            "num_active_clones": [1, 1],
            "n_acquired_alterations": [0, 1],
            "ecog_ps": [1, 2],
            "symptom_burden": ["mild", "moderate"],
            "dominant_symptoms": ["fatigue", "fatigue;pain"],
            "weight_kg": [80.0, 72.0],
            "organ_renal": ["normal", "normal"],
            "organ_hepatic": ["normal", "mild_impairment"],
            "organ_cardiac": ["normal", "normal"],
            "organ_marrow_reserve": ["normal", "normal"],
            "organ_pulmonary": ["normal", "normal"],
            "n_active_aes": [0, 1],
            "model_short": ["m", "m"],
            "tumor_oncotree_code": ["X", "X"],
        }
    )
    sites = pd.DataFrame(
        {
            "_gid": ["g", "g", "g"],
            "snapshot_index": [0, 1, 1],
            "site_class_canonical": ["brain", "brain", "liver"],
            "site_signature": ["brain-a", "brain-a", "liver-a"],
            "burden_band": ["small", "moderate", "small"],
        }
    )
    active = pd.DataFrame(
        {
            "_gid": ["g", "g"],
            "snapshot_index": [0, 1],
            "active_condition_count": [0.0, 1.0],
            "active_condition_grade_max": [0.0, 3.0],
            "active_condition_grade_sum": [0.0, 3.0],
        }
    )

    state, site_columns = ledger.derive_state_candidates(snapshots, sites, active)

    assert "site__brain" in site_columns
    assert state.loc[1, "site__brain"] == 3
    assert state.loc[1, "weight_change_pct"] == pytest.approx(-10.0)
    assert state.loc[1, "symptom_rank"] == 2
    assert state.loc[1, "organ_hepatic_rank"] == 1


def test_active_condition_summary_retains_system_and_grade() -> None:
    snapshot = {
        "active_aes": [
            {"ae_category": "cardiac", "ctcae_v5_grade": 3},
            {"ae_category": "cardiac", "ctcae_v5_grade": 2},
            {"ae_category": "dermatologic", "ctcae_v5_grade": 1},
        ]
    }

    result = ledger.summarize_active_conditions(snapshot)

    assert result["active_condition_count"] == 3
    assert result["active_condition_grade_max"] == 3
    assert result["active_condition_grade_sum"] == 6
    assert result["active_condition__cardiac"] == 3
    assert result["active_condition__dermatologic"] == 1


def test_symptom_vocabulary_only_normalizes_surface_form() -> None:
    result = ledger.symptom_vocabulary_audit(
        pd.Series(["Abdominal_pain; fatigue", "abdominal pain"])
    )

    assert result["raw_label"].nunique() == 3
    assert result["minimally_normalized_label"].nunique() == 2
    abdominal = result[
        result["minimally_normalized_label"].eq("abdominal pain")
    ]
    assert abdominal["n_mentions"].sum() == 2


def test_symptom_system_candidates_do_not_invent_per_system_severity() -> None:
    frame = pd.DataFrame(
        {
            "dominant_symptoms": ["headache;dyspnea", "fatigue", None],
            "symptom_rank": [3.0, 1.0, 0.0],
        }
    )

    result, columns = ledger.add_symptom_system_candidates(frame)

    assert ledger.classify_symptom_system("abdominal_pain") == (
        "gastrointestinal_hepatobiliary"
    )
    assert result.loc[0, "symptom_system__neurologic"] == 1
    assert result.loc[0, "symptom_system__respiratory"] == 1
    assert result.loc[0, "symptom_intensity__neurologic"] == 3
    assert result.loc[1, "symptom_intensity__constitutional"] == 1
    assert "symptom_intensity__unmapped" in columns


def test_symptom_recording_distinguishes_missing_names_from_absence() -> None:
    state = pd.DataFrame(
        {
            "model_short": ["m", "m", "m"],
            "symptom_rank": [0.0, 2.0, 3.0],
            "dominant_symptoms": ["", "", "dyspnea"],
        }
    )

    by_model, meta = ledger.symptom_recording_deep_audit(state)

    assert by_model.iloc[0]["named_coverage_among_symptomatic_pct"] == 50.0
    assert meta["symptomatic_without_names_pct_pooled"] == pytest.approx(
        100.0 / 3.0
    )
    assert "not evidence" in meta["interpretation"]


def test_site_multiplicity_is_not_nonzero_compartment_count() -> None:
    sites = pd.DataFrame(
        {
            "_gid": ["g", "g", "g"],
            "snapshot_index": [0, 0, 0],
            "site_class_canonical": ["liver", "liver", "brain"],
            "site_signature": ["liver-left", "liver-right", "brain-a"],
            "site_anatomic_label": ["left liver", "right liver", "brain"],
        }
    )
    state = pd.DataFrame(
        {
            "_gid": ["g"],
            "snapshot_index": [0],
            "n_unique_anatomic_sites": [3.0],
            "n_site_compartments": [2.0],
        }
    )

    by_class, _examples, meta = ledger.site_multiplicity_deep_audit(sites, state)
    liver = by_class.set_index("site_class_canonical").loc["liver"]

    assert liver["max_unique_labels"] == 2
    assert meta["living_snapshots_with_more_sites_than_compartments_pct"] == 100


def test_weight_audit_keeps_transform_candidates_explicit() -> None:
    state = pd.DataFrame(
        {
            "_run": ["r", "r"],
            "_gid": ["g", "g"],
            "snapshot_index": [0, 1],
            "model_short": ["m", "m"],
            "tumor_oncotree_code": ["X", "X"],
            "days_from_diagnosis": [0.0, 30.4],
            "baseline_weight_kg": [100.0, 100.0],
            "weight_kg_absolute": [100.0, 60.0],
            "weight_change_pct": [0.0, -40.0],
        }
    )

    transforms, intervals, _outliers, meta = ledger.weight_deep_audit(state)

    clipped = transforms[
        transforms["transform"].eq("clinically_clipped_percent_m30_p20")
    ].iloc[0]
    assert clipped["minimum"] == -30.0
    assert intervals.iloc[0]["weight_jump_pct"] == pytest.approx(-40.0)
    assert meta["n_weight_changes"] == 1


def test_reviewer_table_makes_symptom_and_ae_defaults_explicit() -> None:
    feature_ids = [
        "tumour_burden_max_rank",
        "n_site_compartments",
        "n_unique_anatomic_sites",
        "n_clonal_site_entries",
        "ecog_rank",
        "symptom_rank",
        "organ_renal_rank",
        "organ_hepatic_rank",
        "organ_cardiac_rank",
        "organ_marrow_reserve_rank",
        "organ_pulmonary_rank",
        "weight_change_pct",
        "active_condition_grade_max",
        "active_condition_count",
        "active_condition_grade_sum",
        "n_active_clones",
        "n_acquired_alterations",
        "weight_kg_absolute",
    ]
    audit = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "n_snapshots": 100,
            "missing_pct": 0.0,
            "zero_pct_equal_model": 50.0,
            "median": 1.0,
            "p10": 0.0,
            "p90": 2.0,
            "change_pct_equal_model": 10.0,
        }
    )

    result = ledger.reviewer_feature_decision_table(
        audit,
        n_snapshots=100,
        n_site_classes=32,
        symptom_recording_meta={
            "named_coverage_among_symptomatic_pct_equal_model": 49.1
        },
    ).set_index("candidate_id")

    assert result.loc["global_symptom_burden", "paper_v1_default"] == "INCLUDE"
    assert "not calculated" in result.loc[
        "global_symptom_burden", "exact_measurement"
    ]
    assert result.loc[
        "maximum_active_ae_grade", "paper_v1_default"
    ].startswith("INCLUDE")
    assert result.loc["active_ae_count", "paper_v1_default"].startswith("INCLUDE")
    assert result.loc["summed_active_ae_grades", "paper_v1_default"] == "EXCLUDE"


def test_feature_audit_reports_onset_resolution_and_change() -> None:
    frame = pd.DataFrame(
        {
            "_gid": ["a", "a", "a", "b", "b"],
            "snapshot_index": [0, 1, 2, 0, 1],
            "model_short": ["m1", "m1", "m1", "m2", "m2"],
            "x": [0.0, 2.0, 0.0, 1.0, 1.0],
        }
    )
    spec = ledger.AuditFeature(
        "x", "x", "host phenotype", "test", "ordinal", "test", "test"
    )

    row = ledger.audit_features(frame, [spec]).iloc[0]

    assert row["onset_count"] == 1
    assert row["resolution_count"] == 1
    assert row["change_pct_pooled"] == pytest.approx(100.0 * 2.0 / 3.0)
    assert row["resolution_per_onset"] == pytest.approx(1.0)
