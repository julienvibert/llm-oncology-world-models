"""Preflight seed validation against the BaselineObject Pydantic tree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthonccases.oncollm.conditioning_audit import (
    BaselineSeedError,
    validate_baseline_seed,
)


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "rev5_poc" / "her2_brca_baseline_002.json"
)


def _full() -> dict:
    return json.loads(FIXTURE.read_text())


def test_empty_seed_validates() -> None:
    validate_baseline_seed({})


def test_full_baseline_seed_validates() -> None:
    validate_baseline_seed(_full())


def test_tumor_only_partial_seed_validates() -> None:
    validate_baseline_seed({
        "identity_card": {"diagnosis": {"oncotree_code": "BRCA"}},
    })


def test_typo_unknown_path_rejected() -> None:
    with pytest.raises(BaselineSeedError) as exc_info:
        validate_baseline_seed({
            "identity_card": {"diagnosis": {"oncotree": "BRCA"}},
        })
    assert exc_info.value.path == "identity_card.diagnosis.oncotree"


def test_unknown_top_level_path_rejected() -> None:
    with pytest.raises(BaselineSeedError) as exc_info:
        validate_baseline_seed({"not_a_field": 1})
    assert exc_info.value.path == "not_a_field"


def test_invalid_enum_value_rejected() -> None:
    with pytest.raises(BaselineSeedError) as exc_info:
        validate_baseline_seed({
            "identity_card": {"staging": {"stage_class": "advanced"}},
        })
    assert exc_info.value.path == "identity_card.staging.stage_class"


def test_type_mismatch_rejected() -> None:
    with pytest.raises(BaselineSeedError) as exc_info:
        validate_baseline_seed({
            "identity_card": {"demographics": {"age_at_diagnosis": "not-an-int"}},
        })
    assert exc_info.value.path == "identity_card.demographics.age_at_diagnosis"


def test_non_mapping_root_rejected() -> None:
    with pytest.raises(BaselineSeedError):
        validate_baseline_seed("not a dict")  # type: ignore[arg-type]


def test_list_valued_seed_field_accepted() -> None:
    """Pinning a whole list (clone_registry / family_history / etc.) works."""
    validate_baseline_seed({
        "medical_history": {
            "family_history": [
                {"relation": "mother", "cancer_type": "breast_cancer"},
            ],
        },
    })


def test_list_with_invalid_inner_item_rejected() -> None:
    with pytest.raises(BaselineSeedError) as exc_info:
        validate_baseline_seed({
            "medical_history": {
                "family_history": [
                    {"relation": "not_a_relation", "cancer_type": "x"},
                ],
            },
        })
    assert exc_info.value.path == "medical_history.family_history"


def test_no_gold_template_import_path_exists() -> None:
    """Regression guard: validate_baseline_seed must not depend on a fixture
    merge. If a future refactor reintroduces a GOLD_TEMPLATE hack, this test
    will fail because the symbol is intentionally absent."""
    from synthonccases.oncollm import conditioning_audit
    assert not hasattr(conditioning_audit, "GOLD_TEMPLATE")
