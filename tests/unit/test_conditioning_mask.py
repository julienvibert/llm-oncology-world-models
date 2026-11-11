"""Pinned-paths walker + required-patch-leaves derivation."""

from __future__ import annotations

import json
from pathlib import Path

from synthonccases.oncollm.conditioning_audit import (
    compute_pinned_paths,
    compute_required_patch_leaves,
)


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "rev5_poc" / "her2_brca_baseline_002.json"
)


# ---------- compute_pinned_paths ----------


def test_empty_seed_yields_no_paths() -> None:
    assert compute_pinned_paths({}) == ()


def test_nested_partial_seed_yields_sorted_dotted_paths() -> None:
    seed = {
        "identity_card": {
            "diagnosis": {"oncotree_code": "BRCA"},
            "demographics": {"sex": "female"},
        },
    }
    assert compute_pinned_paths(seed) == (
        "identity_card.demographics.sex",
        "identity_card.diagnosis.oncotree_code",
    )


def test_lists_are_pinned_as_whole_leaves_not_per_index() -> None:
    """Regression for the rev3 list-semantics decision: a list-valued seed
    field appears as one dotted leaf, never `field[0].subfield`."""
    seed = {
        "medical_history": {
            "family_history": [
                {"relation": "mother", "cancer_type": "breast_cancer"},
                {"relation": "father", "cancer_type": "lung_cancer"},
            ],
            "smoking_status": "never",
        },
    }
    paths = compute_pinned_paths(seed)
    assert "medical_history.family_history" in paths
    assert "medical_history.smoking_status" in paths
    assert not any(p.startswith("medical_history.family_history.") for p in paths)
    assert not any("[" in p for p in paths)


def test_empty_nested_dict_contributes_no_paths() -> None:
    assert compute_pinned_paths({"identity_card": {"diagnosis": {}}}) == ()


def test_full_baseline_seed_pins_every_leaf_under_top_level_objects() -> None:
    seed = json.loads(FIXTURE.read_text())
    paths = compute_pinned_paths(seed)
    assert "baseline_id" in paths
    assert "identity_card.diagnosis.oncotree_code" in paths
    assert "identity_card.molecular_profile.alterations" in paths
    assert "baseline_patient_state.weight_kg" in paths


# ---------- compute_required_patch_leaves ----------


def test_required_patch_leaves_for_empty_seed_covers_required_baseline_leaves() -> None:
    required = compute_required_patch_leaves({})
    # Spot-check required leaves under BaselineObject
    assert "baseline_id" in required
    assert "identity_card.demographics.age_at_diagnosis" in required
    assert "identity_card.staging.stage_class" in required
    assert "identity_card.diagnosis.histology_bucket" in required
    assert "identity_card.diagnosis.primary_site_class" in required
    assert "identity_card.molecular_profile.primary_driver_bucket" in required
    # list-valued required field is a whole-leaf entry
    assert "clone_registry" in required
    assert "baseline_disease_burden" in required
    assert not any(p.startswith("clone_registry.") for p in required)


def test_required_patch_leaves_excludes_pinned_paths_and_their_ancestors() -> None:
    seed = {"identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}}
    required = compute_required_patch_leaves(seed)
    assert "identity_card.diagnosis.oncotree_code" not in required
    # Other diagnosis fields the seed did not pin should still be required
    assert "identity_card.diagnosis.histology_class" in required
    assert "identity_card.diagnosis.histology_bucket" in required
    assert "identity_card.diagnosis.primary_site_class" in required


def test_required_patch_leaves_with_full_seed_is_empty() -> None:
    full = json.loads(FIXTURE.read_text())
    required = compute_required_patch_leaves(full)
    # Every required leaf has a pinned ancestor (the leaf itself)
    assert required == ()
