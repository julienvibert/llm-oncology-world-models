"""Tests for the OncoTree roster builder script (overlay v2 / tier-based schema).

Covers:
  - Determinism: rebuild produces byte-identical roster.
  - Strict tier-based overlay validation: bad entries fail-fast with BuildError,
    well-formed entries pass. One pass + one fail per guardrail rule.

The script is in scripts/, not in src/, so we import via sys.path manipulation
(matches the convention used by tests/unit/test_analyze.py).
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_oncotree_roster as builder  # noqa: E402

ROSTER_PATH = REPO_ROOT / "data" / "oncotree" / "roster_v1.yaml"

KNOWN: set[str] = {"LUAD", "LUSC", "BREAST", "GB", "CCRCC", "MNG"}
# Leaf status used across the parametric overlay tests.
LEAVES: dict[str, bool] = {
    "LUAD": True,
    "LUSC": True,
    "BREAST": False,
    "GB": False,     # non-leaf primary exception target
    "CCRCC": False,  # non-leaf primary exception target
    "MNG": True,
}
ALLOWED_REASONS = [
    "non_clinical_grouping_node",
    "pediatric_only_unless_intentional",
    "premalignant_or_in_situ_only_without_invasive_analogue",
    "no_clinical_management_distinction_from_parent",
    "outside_adult_oncology_scope",
]


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _curation(codes: dict) -> dict:
    return {"codes": codes, "exclusion_rule_categories": ALLOWED_REASONS}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_roster_rebuild_is_deterministic():
    """Building twice produces byte-identical YAML — the fetched_at fix is the key."""
    if not ROSTER_PATH.exists():
        pytest.skip("roster_v1.yaml not yet built; run scripts/build_oncotree_roster.py first")
    sha_before = _sha256_bytes(ROSTER_PATH)
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "build_oncotree_roster.py")],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )
    sha_after = _sha256_bytes(ROSTER_PATH)
    assert sha_before == sha_after, "roster rebuild must be deterministic"


# ---------------------------------------------------------------------------
# Overlay validation — unknown code + legacy include rejection
# ---------------------------------------------------------------------------


def test_overlay_unknown_code_fails():
    bad = _curation({"XYZ_NOT_REAL": {"tier": "excluded", "exclusion_reason": "non_clinical_grouping_node"}})
    with pytest.raises(builder.BuildError, match="unknown OncoTree code"):
        builder._validate_overlay(bad, KNOWN, LEAVES)


def test_overlay_legacy_include_field_is_rejected():
    bad = _curation({"LUAD": {"include": True, "tier": "primary"}})
    with pytest.raises(builder.BuildError, match="legacy 'include' field is not permitted"):
        builder._validate_overlay(bad, KNOWN, LEAVES)


# ---------------------------------------------------------------------------
# Tier sanity
# ---------------------------------------------------------------------------


def test_overlay_unknown_tier_fails():
    bad = _curation({"LUAD": {"tier": "bogus"}})
    with pytest.raises(builder.BuildError, match="tier='bogus' not in"):
        builder._validate_overlay(bad, KNOWN, LEAVES)


def test_overlay_tier_null_with_curation_fields_fails():
    bad = _curation({"LUAD": {"tier": None, "exclusion_reason": "non_clinical_grouping_node"}})
    with pytest.raises(builder.BuildError, match="tier=null .pending. must have all curation fields null"):
        builder._validate_overlay(bad, KNOWN, LEAVES)


def test_overlay_tier_null_clean_passes():
    good = _curation({"LUAD": {"tier": None}})
    builder._validate_overlay(good, KNOWN, LEAVES)  # no raise


# ---------------------------------------------------------------------------
# Guardrail: exclusion_reason iff tier == "excluded"
# ---------------------------------------------------------------------------


def test_overlay_excluded_without_reason_fails():
    bad = _curation({"LUAD": {"tier": "excluded", "exclusion_reason": None}})
    with pytest.raises(builder.BuildError, match="non-null exclusion_reason"):
        builder._validate_overlay(bad, KNOWN, LEAVES)


def test_overlay_excluded_with_bad_reason_fails():
    bad = _curation({"LUAD": {"tier": "excluded", "exclusion_reason": "made_up_reason"}})
    with pytest.raises(builder.BuildError, match="not in allowed categories"):
        builder._validate_overlay(bad, KNOWN, LEAVES)


def test_overlay_non_excluded_with_reason_fails():
    bad = _curation({"LUAD": {"tier": "primary", "exclusion_reason": "non_clinical_grouping_node"}})
    with pytest.raises(builder.BuildError, match="must have exclusion_reason=null"):
        builder._validate_overlay(bad, KNOWN, LEAVES)


def test_overlay_excluded_with_valid_reason_passes():
    good = _curation({
        "LUAD": {"tier": "excluded", "exclusion_reason": "pediatric_only_unless_intentional"},
    })
    builder._validate_overlay(good, KNOWN, LEAVES)  # no raise


# ---------------------------------------------------------------------------
# Guardrail: secondary_role iff tier == "secondary"
# ---------------------------------------------------------------------------


def test_overlay_secondary_without_role_fails():
    bad = _curation({"LUAD": {"tier": "secondary", "secondary_role": None}})
    with pytest.raises(builder.BuildError, match="requires a non-null secondary_role"):
        builder._validate_overlay(bad, KNOWN, LEAVES)


def test_overlay_secondary_with_bad_role_fails():
    bad = _curation({"LUAD": {"tier": "secondary", "secondary_role": "bogus_role"}})
    with pytest.raises(builder.BuildError, match="not in allowed roles"):
        builder._validate_overlay(bad, KNOWN, LEAVES)


def test_overlay_non_secondary_with_role_fails():
    bad = _curation({"LUAD": {"tier": "primary", "secondary_role": "pool"}})
    with pytest.raises(builder.BuildError, match="must have secondary_role=null"):
        builder._validate_overlay(bad, KNOWN, LEAVES)


def test_overlay_secondary_with_valid_role_passes():
    good = _curation({"LUAD": {"tier": "secondary", "secondary_role": "targeted_stress"}})
    builder._validate_overlay(good, KNOWN, LEAVES)  # no raise


# ---------------------------------------------------------------------------
# Guardrail: primary_non_leaf_rationale iff tier == "primary" AND not is_leaf
# ---------------------------------------------------------------------------


def test_overlay_primary_leaf_with_rationale_fails():
    bad = _curation({"LUAD": {"tier": "primary", "primary_non_leaf_rationale": "should be null on leaves"}})
    with pytest.raises(builder.BuildError, match="primary_non_leaf_rationale=null"):
        builder._validate_overlay(bad, KNOWN, LEAVES)


def test_overlay_primary_non_leaf_without_rationale_fails():
    bad = _curation({"GB": {"tier": "primary", "primary_non_leaf_rationale": None}})
    with pytest.raises(builder.BuildError, match="requires a non-empty primary_non_leaf_rationale"):
        builder._validate_overlay(bad, KNOWN, LEAVES)


def test_overlay_primary_non_leaf_with_empty_rationale_fails():
    bad = _curation({"GB": {"tier": "primary", "primary_non_leaf_rationale": "   "}})
    with pytest.raises(builder.BuildError, match="requires a non-empty primary_non_leaf_rationale"):
        builder._validate_overlay(bad, KNOWN, LEAVES)


def test_overlay_secondary_with_rationale_fails():
    bad = _curation({
        "LUAD": {
            "tier": "secondary",
            "secondary_role": "pool",
            "primary_non_leaf_rationale": "wrong",
        },
    })
    with pytest.raises(builder.BuildError, match="tier=secondary must have primary_non_leaf_rationale=null"):
        builder._validate_overlay(bad, KNOWN, LEAVES)


def test_overlay_primary_non_leaf_with_rationale_passes():
    good = _curation({
        "GB": {
            "tier": "primary",
            "primary_non_leaf_rationale": (
                "Glioblastoma, IDH-wildtype is a canonical adult-type diffuse glioma entity; "
                "the nested rows are histologic variants rather than replacement disease labels."
            ),
        },
    })
    builder._validate_overlay(good, KNOWN, LEAVES)  # no raise


def test_overlay_primary_leaf_passes():
    good = _curation({"LUAD": {"tier": "primary"}})
    builder._validate_overlay(good, KNOWN, LEAVES)  # no raise


# ---------------------------------------------------------------------------
# Notes-type sanity
# ---------------------------------------------------------------------------


def test_overlay_non_string_notes_fails():
    bad = _curation({"LUAD": {"tier": "primary", "notes": 42}})
    with pytest.raises(builder.BuildError, match="notes must be a string or null"):
        builder._validate_overlay(bad, KNOWN, LEAVES)


def test_overlay_notes_string_passes():
    good = _curation({"LUAD": {"tier": "primary", "notes": "Lung adenocarcinoma; canonical."}})
    builder._validate_overlay(good, KNOWN, LEAVES)  # no raise


# ---------------------------------------------------------------------------
# Empty overlay
# ---------------------------------------------------------------------------


def test_overlay_empty_passes():
    builder._validate_overlay(_curation({}), KNOWN, LEAVES)  # no raise


# ---------------------------------------------------------------------------
# include is derived from tier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tier,expected_include",
    [
        ("primary", True),
        ("secondary", True),
        ("excluded", False),
        (None, None),
    ],
)
def test_tier_to_include_derivation(tier, expected_include):
    assert builder._derive_include(tier) is expected_include
