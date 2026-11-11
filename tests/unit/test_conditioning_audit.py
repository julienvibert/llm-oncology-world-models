"""Patch-pin collision rule + resolve-and-validate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthonccases.oncollm.conditioning_audit import (
    BaselineResolutionError,
    assert_no_pin_collision,
    deep_merge,
    resolve_and_validate,
)
from synthonccases.oncollm.replay import BaselineObject


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "rev5_poc" / "her2_brca_baseline_002.json"
)


def _full() -> dict:
    return json.loads(FIXTURE.read_text())


# ---------- assert_no_pin_collision ----------


def test_no_pinned_paths_means_no_collisions() -> None:
    assert assert_no_pin_collision({"identity_card": {"foo": 1}}, ()) == []


def test_equal_paths_collide() -> None:
    pinned = ("identity_card.diagnosis.oncotree_code",)
    patch = {"identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}}
    collisions = assert_no_pin_collision(patch, pinned)
    assert len(collisions) == 1
    assert collisions[0].relation == "equal"


def test_patch_ancestor_of_pinned_collides() -> None:
    pinned = ("identity_card.diagnosis.oncotree_code",)
    # Patch sets a higher-up leaf that subsumes the pinned path
    patch = {"identity_card": {"diagnosis": "REPLACED"}}
    collisions = assert_no_pin_collision(patch, pinned)
    assert len(collisions) == 1
    assert collisions[0].relation == "patch_ancestor_of_pinned"
    assert collisions[0].patch_path == "identity_card.diagnosis"


def test_pinned_ancestor_of_patch_collides() -> None:
    pinned = ("identity_card",)
    patch = {"identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}}
    collisions = assert_no_pin_collision(patch, pinned)
    assert any(c.relation == "pinned_ancestor_of_patch" for c in collisions)


def test_disjoint_sibling_paths_do_not_collide() -> None:
    """Regression: dropped suffix rule means siblings under a common parent
    do NOT collide (e.g. medical_history.smoking_status vs
    identity_card.diagnosis.oncotree_code)."""
    pinned = ("identity_card.diagnosis.oncotree_code",)
    patch = {"medical_history": {"smoking_status": "never"}}
    assert assert_no_pin_collision(patch, pinned) == []


def test_segment_prefix_not_string_prefix() -> None:
    """`identity_card.diagnosis` is NOT an ancestor of
    `identity_card.diagnosis_label` — segment match, not raw string prefix."""
    pinned = ("identity_card.diagnosis_label",)
    patch = {"identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}}
    assert assert_no_pin_collision(patch, pinned) == []


# ---------- deep_merge ----------


def test_deep_merge_merges_disjoint_subtrees() -> None:
    out = deep_merge({"a": {"x": 1}}, {"a": {"y": 2}, "b": 3})
    assert out == {"a": {"x": 1, "y": 2}, "b": 3}


def test_deep_merge_leaf_collision_raises() -> None:
    with pytest.raises(BaselineResolutionError):
        deep_merge({"a": 1}, {"a": 2})


# ---------- resolve_and_validate ----------


def test_full_seed_empty_patch_equals_seed() -> None:
    seed = _full()
    resolved = resolve_and_validate(seed, {})
    assert isinstance(resolved, BaselineObject)


def test_empty_seed_full_patch_equals_patch() -> None:
    full = _full()
    resolved = resolve_and_validate({}, full)
    assert isinstance(resolved, BaselineObject)


def test_partial_seed_completed_by_patch() -> None:
    full = _full()
    diagnosis_seed = {
        "identity_card": {"diagnosis": full["identity_card"]["diagnosis"]},
    }
    patch = {k: v for k, v in full.items() if k != "identity_card"}
    patch["identity_card"] = {
        k: v for k, v in full["identity_card"].items() if k != "diagnosis"
    }
    resolved = resolve_and_validate(diagnosis_seed, patch)
    assert resolved.identity_card.diagnosis.oncotree_code == "BRCA"


def test_resolved_baseline_failing_validation_raises() -> None:
    """An incomplete resolved baseline (missing required leaves) raises
    BaselineResolutionError — which the runner maps to
    failure_kind='baseline_validation'."""
    with pytest.raises(BaselineResolutionError):
        resolve_and_validate({}, {"baseline_id": "x"})
