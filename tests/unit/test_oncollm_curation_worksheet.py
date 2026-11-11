"""Tests for the OncoTree curation worksheet artifact + scaffold script."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSHEET_PATH = REPO_ROOT / "data" / "oncotree" / "curation_worksheet_v1.yaml"


def _load() -> dict:
    with WORKSHEET_PATH.open() as f:
        return yaml.safe_load(f)


_EXPECTED_NODE_KEYS: frozenset[str] = frozenset({
    "code", "name", "level", "parent", "is_leaf", "main_type",
    "suggested_tier", "suggested_exclusion_reason",
    "tier", "exclusion_reason", "secondary_role",
    "primary_non_leaf_rationale", "notes",
})

_EXPECTED_EXCLUSION_CATEGORIES: frozenset[str] = frozenset({
    "non_clinical_grouping_node",
    "pediatric_only_unless_intentional",
    "premalignant_or_in_situ_only_without_invasive_analogue",
    "no_clinical_management_distinction_from_parent",
    "outside_adult_oncology_scope",
})

_EXPECTED_TIERS: frozenset[str] = frozenset({
    "primary",
    "secondary",
    "excluded",
})

_EXPECTED_SECONDARY_ROLES: frozenset[str] = frozenset({
    "targeted_stress",
    "pool",
    "rollup_only",
    "catalog_only",
})

_GENERATION_ELIGIBLE_SECONDARY_ROLES: frozenset[str] = frozenset({
    "targeted_stress",
    "pool",
})


def test_worksheet_file_exists():
    assert WORKSHEET_PATH.exists(), (
        "data/oncotree/curation_worksheet_v1.yaml is missing — re-run "
        "scripts/scaffold_curation_worksheet.py"
    )


def test_worksheet_schema_version_locked():
    data = _load()
    assert data["schema_version"] == "oncotree_curation_worksheet_v1"


def test_worksheet_total_nodes_matches_roster():
    data = _load()
    assert data["total_nodes"] == 897
    all_nodes = [n for g in data["groups"] for n in g["nodes"]]
    assert len(all_nodes) == 897


def test_worksheet_tier_categories_defined():
    data = _load()
    assert set(data["tier_definitions"]) == {"primary", "secondary", "excluded"}


def test_worksheet_exclusion_categories_lockstep_with_overlay():
    """Categories here must match data/oncotree/roster_curation_v1.yaml's."""
    data = _load()
    assert set(data["exclusion_reason_categories"]) == _EXPECTED_EXCLUSION_CATEGORIES


def test_every_node_has_full_column_set():
    data = _load()
    for g in data["groups"]:
        for n in g["nodes"]:
            assert set(n) == _EXPECTED_NODE_KEYS, (
                f"node {n.get('code')} has unexpected key set: {set(n)}"
            )


def test_curated_columns_obey_tier_contract():
    """As curation progresses, filled curation columns must stay valid.

    Primary-on-non-leaf is permitted ONLY when primary_non_leaf_rationale is
    filled (documented-exception rule, locked 2026-05-14 to accommodate
    clinically canonical non-leaf headline entities like CCRCC, DLBCLNOS,
    FL, MCL, CML, HL whose OncoTree children are rare variants rather than
    co-equal subtypes).
    """
    data = _load()
    for g in data["groups"]:
        for n in g["nodes"]:
            if n["tier"] is not None:
                assert n["tier"] in _EXPECTED_TIERS, n["code"]
            if n["tier"] == "excluded":
                assert n["exclusion_reason"] in _EXPECTED_EXCLUSION_CATEGORIES, n["code"]
            else:
                assert n["exclusion_reason"] is None, n["code"]
            if n["tier"] == "primary" and not n["is_leaf"]:
                rationale = n["primary_non_leaf_rationale"]
                assert isinstance(rationale, str) and rationale.strip(), (
                    f"{n['code']}: primary on non-leaf requires a non-empty "
                    "primary_non_leaf_rationale"
                )
            if n["notes"] is not None:
                assert isinstance(n["notes"], str), n["code"]


def test_primary_non_leaf_rationale_only_for_non_leaf_primary():
    """The rationale field is meaningful ONLY for primary non-leaf nodes;
    must stay null on any other configuration."""
    data = _load()
    for g in data["groups"]:
        for n in g["nodes"]:
            r = n["primary_non_leaf_rationale"]
            if r is None:
                continue
            assert n["tier"] == "primary", (
                f"{n['code']}: primary_non_leaf_rationale set but tier={n['tier']!r}"
            )
            assert n["is_leaf"] is False, (
                f"{n['code']}: primary_non_leaf_rationale set but node is a leaf "
                "(rationale only applies to non-leaf primary exceptions)"
            )
            assert isinstance(r, str) and r.strip(), n["code"]


def test_secondary_role_categories_defined_at_top_level():
    """The four secondary_role categories are declared in the worksheet header."""
    data = _load()
    assert set(data["secondary_role_definitions"]) == _EXPECTED_SECONDARY_ROLES


def test_secondary_role_only_non_null_when_tier_secondary():
    """secondary_role is required for secondary rows and forbidden otherwise."""
    data = _load()
    for g in data["groups"]:
        for n in g["nodes"]:
            if n["tier"] == "secondary":
                assert n["secondary_role"] in _EXPECTED_SECONDARY_ROLES, n["code"]
            else:
                assert n["secondary_role"] is None, n["code"]


def test_generation_eligible_subset_locked():
    """Only targeted_stress + pool are generation-eligible; rollup_only +
    catalog_only are documented but not sampled. Lock the membership here."""
    assert _GENERATION_ELIGIBLE_SECONDARY_ROLES == {"targeted_stress", "pool"}
    assert _GENERATION_ELIGIBLE_SECONDARY_ROLES.issubset(_EXPECTED_SECONDARY_ROLES)


def test_level_1_nodes_pre_filled_as_excluded():
    """The 32 tissue grouping nodes (level=1, parent=TISSUE) are mechanically excluded."""
    data = _load()
    level_1 = [n for g in data["groups"] for n in g["nodes"] if n["level"] == 1]
    assert len(level_1) == 32
    for n in level_1:
        assert n["suggested_tier"] == "excluded", n["code"]
        assert n["suggested_exclusion_reason"] == "non_clinical_grouping_node"
        assert n["parent"] == "TISSUE"


def test_non_level_1_nodes_have_no_mechanical_pre_fill():
    """suggested_tier / suggested_exclusion_reason stay null for the 865 non-grouping nodes."""
    data = _load()
    others = [n for g in data["groups"] for n in g["nodes"] if n["level"] != 1]
    assert len(others) == 897 - 32
    for n in others:
        assert n["suggested_tier"] is None, n["code"]
        assert n["suggested_exclusion_reason"] is None, n["code"]


def test_leaf_count_matches_roster_geometry():
    data = _load()
    all_nodes = [n for g in data["groups"] for n in g["nodes"]]
    leaves = [n for n in all_nodes if n["is_leaf"]]
    non_leaves = [n for n in all_nodes if not n["is_leaf"]]
    assert len(leaves) + len(non_leaves) == 897
    # The roster has 727 leaves (sanity-check from build).
    assert len(leaves) == 727


def test_groups_are_in_main_type_order():
    """Groups sort by main_type for predictable navigation."""
    data = _load()
    mts = [g["main_type"] for g in data["groups"]]
    # None (no main_type) is sorted last in the script; ignore for comparison.
    non_null = [mt for mt in mts if mt is not None]
    assert non_null == sorted(non_null)


def test_each_group_carries_node_count():
    data = _load()
    for g in data["groups"]:
        assert g["n_nodes"] == len(g["nodes"])
