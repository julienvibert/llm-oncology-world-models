"""Tests for the worksheet -> overlay adapter (scripts/port_worksheet_to_overlay.py).

Covers:
  - Round-trip: worksheet content lands intact in the overlay.
  - Idempotence: re-running the adapter on the same worksheet is byte-identical.
  - Mechanical-field strip: worksheet-only columns do not bleed into the overlay.
  - Schema conformance: every emitted entry obeys the v2 contract.
  - Validation: malformed worksheet rows raise PortError with a useful message.

The adapter is in scripts/, not in src/, so we import via sys.path manipulation
(matches the convention used by tests/unit/test_oncollm_roster_builder.py).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import port_worksheet_to_overlay as adapter  # noqa: E402

WORKSHEET_PATH = REPO_ROOT / "data" / "oncotree" / "curation_worksheet_v1.yaml"
OVERLAY_PATH = REPO_ROOT / "data" / "oncotree" / "roster_curation_v1.yaml"

OVERLAY_ENTRY_KEYS = frozenset({
    "tier", "exclusion_reason", "secondary_role",
    "primary_non_leaf_rationale", "notes",
})

WORKSHEET_ONLY_KEYS = frozenset({
    "code", "name", "level", "parent", "is_leaf", "main_type",
    "suggested_tier", "suggested_exclusion_reason",
})


def _load_worksheet() -> dict:
    return yaml.safe_load(WORKSHEET_PATH.read_text())


def _load_overlay() -> dict:
    return yaml.safe_load(OVERLAY_PATH.read_text())


# ---------------------------------------------------------------------------
# Smoke / shape
# ---------------------------------------------------------------------------


def test_overlay_file_exists_after_port():
    assert OVERLAY_PATH.exists(), (
        "data/oncotree/roster_curation_v1.yaml missing — run "
        "scripts/port_worksheet_to_overlay.py"
    )


def test_overlay_schema_version_locked():
    assert _load_overlay()["schema_version"] == "oncotree_curation_v2"


def test_overlay_carries_all_897_codes():
    overlay = _load_overlay()
    assert len(overlay["codes"]) == 897


def test_overlay_tier_categories_locked():
    overlay = _load_overlay()
    assert overlay["tier_categories"] == ["primary", "secondary", "excluded"]


def test_overlay_secondary_role_categories_locked():
    overlay = _load_overlay()
    assert overlay["secondary_role_categories"] == [
        "targeted_stress", "pool", "rollup_only", "catalog_only",
    ]


# ---------------------------------------------------------------------------
# Per-entry shape — mechanical fields stripped, locked field order
# ---------------------------------------------------------------------------


def test_overlay_entries_strip_mechanical_fields():
    overlay = _load_overlay()
    for code, entry in overlay["codes"].items():
        assert set(entry.keys()) == OVERLAY_ENTRY_KEYS, (
            f"{code}: entry keys {set(entry.keys())} != {OVERLAY_ENTRY_KEYS}"
        )
        for stray in WORKSHEET_ONLY_KEYS:
            assert stray not in entry, f"{code}: leaked worksheet-only field {stray!r}"


def test_overlay_entry_field_order_is_canonical():
    """Emit order: tier, exclusion_reason, secondary_role, primary_non_leaf_rationale, notes."""
    overlay = _load_overlay()
    expected = ["tier", "exclusion_reason", "secondary_role", "primary_non_leaf_rationale", "notes"]
    for code, entry in overlay["codes"].items():
        assert list(entry.keys()) == expected, (
            f"{code}: field order is {list(entry.keys())}, expected {expected}"
        )


def test_overlay_codes_alphabetically_sorted():
    overlay = _load_overlay()
    codes = list(overlay["codes"].keys())
    assert codes == sorted(codes), "overlay codes must be alphabetically sorted for diff-friendliness"


# ---------------------------------------------------------------------------
# Round-trip: worksheet content survives the port
# ---------------------------------------------------------------------------


def test_round_trip_curation_fields_match_worksheet():
    """For every node in the worksheet, the overlay entry carries identical
    tier / exclusion_reason / secondary_role / primary_non_leaf_rationale / notes."""
    worksheet = _load_worksheet()
    overlay = _load_overlay()
    for group in worksheet["groups"]:
        for n in group["nodes"]:
            entry = overlay["codes"][n["code"]]
            assert entry["tier"] == n["tier"], n["code"]
            assert entry["exclusion_reason"] == n["exclusion_reason"], n["code"]
            assert entry["secondary_role"] == n["secondary_role"], n["code"]
            assert entry["primary_non_leaf_rationale"] == n["primary_non_leaf_rationale"], n["code"]
            assert entry["notes"] == n["notes"], n["code"]


# ---------------------------------------------------------------------------
# Idempotence: re-running produces byte-identical output
# ---------------------------------------------------------------------------


def test_adapter_is_idempotent():
    """serialize(port(worksheet)) is stable across runs."""
    worksheet = _load_worksheet()
    a = adapter.serialize(adapter.port(worksheet))
    b = adapter.serialize(adapter.port(worksheet))
    assert a == b


def test_overlay_on_disk_matches_freshly_ported_worksheet():
    """Whatever is on disk must equal the result of porting the current worksheet
    — otherwise the overlay is stale and someone forgot to re-run the adapter."""
    worksheet = _load_worksheet()
    expected = adapter.serialize(adapter.port(worksheet))
    on_disk = OVERLAY_PATH.read_text()
    assert on_disk == expected, "overlay is stale; re-run scripts/port_worksheet_to_overlay.py"


# ---------------------------------------------------------------------------
# Tier counts (cross-check vs the curation summary)
# ---------------------------------------------------------------------------


def test_tier_counts_match_curation_summary():
    """Locks the post-curation tier balance recorded on 2026-05-15."""
    overlay = _load_overlay()
    counts = {"primary": 0, "secondary": 0, "excluded": 0}
    for entry in overlay["codes"].values():
        counts[entry["tier"]] += 1
    assert counts == {"primary": 148, "secondary": 631, "excluded": 118}


def test_secondary_role_counts_match_curation_summary():
    overlay = _load_overlay()
    roles = {"targeted_stress": 0, "pool": 0, "rollup_only": 0, "catalog_only": 0}
    for entry in overlay["codes"].values():
        if entry["secondary_role"]:
            roles[entry["secondary_role"]] += 1
    assert roles == {"targeted_stress": 220, "pool": 292, "rollup_only": 101, "catalog_only": 18}


# ---------------------------------------------------------------------------
# Validation: malformed worksheet rows fail-fast in the adapter itself
# ---------------------------------------------------------------------------


def _single_node_worksheet(node_overrides: dict) -> dict:
    """Build a minimal worksheet with one node, overriding fields as needed."""
    base = {
        "code": "TEST",
        "name": "Test Node",
        "level": 2,
        "parent": "TISSUE",
        "is_leaf": True,
        "main_type": "Test",
        "suggested_tier": None,
        "suggested_exclusion_reason": None,
        "tier": "primary",
        "exclusion_reason": None,
        "secondary_role": None,
        "primary_non_leaf_rationale": None,
        "notes": None,
    }
    base.update(node_overrides)
    return {"groups": [{"main_type": "Test", "n_nodes": 1, "nodes": [base]}]}


def test_adapter_accepts_pending_tier_clean():
    """tier=null with all-null curation fields is permitted (forward-compat: a
    future curation round may legitimately have pending rows mid-pass)."""
    ws = _single_node_worksheet({"tier": None, "is_leaf": True})
    overlay = adapter.port(ws)
    assert overlay["codes"]["TEST"]["tier"] is None


def test_adapter_rejects_pending_tier_with_curation_fields():
    """tier=null + any non-null curation field is contradictory; reject."""
    ws = _single_node_worksheet({
        "tier": None,
        "exclusion_reason": "non_clinical_grouping_node",
    })
    with pytest.raises(adapter.PortError, match="all curation fields null"):
        adapter.port(ws)


def test_adapter_rejects_excluded_without_reason():
    ws = _single_node_worksheet({"tier": "excluded", "exclusion_reason": None})
    with pytest.raises(adapter.PortError, match="exclusion_reason"):
        adapter.port(ws)


def test_adapter_rejects_secondary_without_role():
    ws = _single_node_worksheet({"tier": "secondary", "secondary_role": None})
    with pytest.raises(adapter.PortError, match="secondary_role"):
        adapter.port(ws)


def test_adapter_rejects_primary_non_leaf_without_rationale():
    ws = _single_node_worksheet({
        "is_leaf": False,
        "tier": "primary",
        "primary_non_leaf_rationale": None,
    })
    with pytest.raises(adapter.PortError, match="primary_non_leaf_rationale"):
        adapter.port(ws)


def test_adapter_rejects_primary_leaf_with_rationale():
    ws = _single_node_worksheet({
        "is_leaf": True,
        "tier": "primary",
        "primary_non_leaf_rationale": "should not be set on leaves",
    })
    with pytest.raises(adapter.PortError, match="primary_non_leaf_rationale"):
        adapter.port(ws)


def test_adapter_rejects_duplicate_code():
    ws = _single_node_worksheet({})
    # Inject a duplicate by appending a second copy of the same node.
    ws["groups"].append(copy.deepcopy(ws["groups"][0]))
    with pytest.raises(adapter.PortError, match="duplicate code"):
        adapter.port(ws)


def test_adapter_happy_path_minimal_worksheet():
    """A tiny well-formed worksheet ports cleanly."""
    ws = {
        "groups": [
            {
                "main_type": "Test",
                "n_nodes": 2,
                "nodes": [
                    {
                        "code": "AAA",
                        "name": "A",
                        "level": 2,
                        "parent": "TISSUE",
                        "is_leaf": True,
                        "main_type": "Test",
                        "suggested_tier": None,
                        "suggested_exclusion_reason": None,
                        "tier": "primary",
                        "exclusion_reason": None,
                        "secondary_role": None,
                        "primary_non_leaf_rationale": None,
                        "notes": "Hello.",
                    },
                    {
                        "code": "BBB",
                        "name": "B",
                        "level": 2,
                        "parent": "TISSUE",
                        "is_leaf": True,
                        "main_type": "Test",
                        "suggested_tier": None,
                        "suggested_exclusion_reason": None,
                        "tier": "secondary",
                        "exclusion_reason": None,
                        "secondary_role": "pool",
                        "primary_non_leaf_rationale": None,
                        "notes": None,
                    },
                ],
            },
        ],
    }
    overlay = adapter.port(ws)
    assert list(overlay["codes"].keys()) == ["AAA", "BBB"]
    assert overlay["codes"]["AAA"]["tier"] == "primary"
    assert overlay["codes"]["BBB"]["secondary_role"] == "pool"
