"""Human-readable conditioning_label derivation."""

from __future__ import annotations

import json
from pathlib import Path

from synthonccases.oncollm.conditioning_audit import (
    compute_pinned_paths,
    derive_conditioning_label,
)


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "rev5_poc" / "her2_brca_baseline_002.json"
)


def _label(seed: dict, *, regimen_label: str | None = None) -> str:
    return derive_conditioning_label(
        baseline_seed=seed,
        pinned_paths=compute_pinned_paths(seed),
        fixed_first_line_regimen_label=regimen_label,
    )


def test_empty_seed_is_spontaneous() -> None:
    assert _label({}) == "cond:spontaneous"


def test_tumor_only_seed() -> None:
    seed = {"identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}}
    assert _label(seed) == "cond:tumor_only"


def test_tumor_only_seed_with_diagnosis_label() -> None:
    seed = {
        "identity_card": {
            "diagnosis": {
                "oncotree_code": "BRCA",
                "diagnosis_label": "Breast Invasive Carcinoma",
            }
        }
    }
    assert _label(seed) == "cond:tumor_only"


def test_study_reference_pin_is_ignored_for_label_shape() -> None:
    seed = {
        "identity_card": {
            "diagnosis": {
                "oncotree_code": "BRCA",
                "diagnosis_date": "2024-01-15",
            },
        },
    }
    assert (
        derive_conditioning_label(
            baseline_seed=seed,
            pinned_paths=compute_pinned_paths(seed),
            fixed_first_line_regimen_label=None,
            label_ignored_pinned_paths=("identity_card.diagnosis.diagnosis_date",),
        )
        == "cond:tumor_only"
    )


def test_only_study_reference_pin_is_spontaneous_for_label_shape() -> None:
    seed = {
        "identity_card": {
            "diagnosis": {"diagnosis_date": "2024-01-15"},
        },
    }
    assert (
        derive_conditioning_label(
            baseline_seed=seed,
            pinned_paths=compute_pinned_paths(seed),
            fixed_first_line_regimen_label=None,
            label_ignored_pinned_paths=("identity_card.diagnosis.diagnosis_date",),
        )
        == "cond:spontaneous"
    )


def test_full_baseline_seed_uses_model_validate() -> None:
    """Full-baseline detection uses BaselineObject.model_validate(seed) success,
    NOT required-leaf counting. A complete migrated fixture validates and
    must produce cond:full_baseline."""
    seed = json.loads(FIXTURE.read_text())
    assert _label(seed) == "cond:full_baseline"


def test_partial_baseline_label_counts_pinned_leaves() -> None:
    seed = {
        "identity_card": {
            "diagnosis": {"oncotree_code": "BRCA"},
            "demographics": {"sex": "female", "age_at_diagnosis": 50},
        },
    }
    # 3 pinned leaves: oncotree_code, sex, age_at_diagnosis
    assert _label(seed) == "cond:partial_baseline_3"


def test_first_line_regimen_label_suffix_appended() -> None:
    assert _label({}, regimen_label="THP") == "cond:spontaneous+1L=THP"
    seed = {"identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}}
    assert _label(seed, regimen_label="THP") == "cond:tumor_only+1L=THP"


def test_no_suffix_when_regimen_label_is_none() -> None:
    seed = {"identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}}
    assert _label(seed, regimen_label=None) == "cond:tumor_only"
