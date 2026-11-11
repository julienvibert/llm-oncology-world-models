"""Gate B1b — ``_normalize_site_class`` + ``build_disease_site_rows`` semantics.

Locks the 13-value site_class enum (order-sensitive keyword matching),
the clone_id-exclusion of ``site_signature``, the per-trajectory
``is_new_at_snapshot`` semantic (signatures evaluated against PRIOR
snapshots only), and the active-site filter.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synthonccases.oncollm.derivation import (
    CellKey,
    TrajectoryArtifact,
    _normalize_site_class,
    _site_signature,
    build_disease_site_rows,
)


# ---------------------------------------------------------------------------
# _normalize_site_class — 13-value enum coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,expected", [
    # Order-sensitive: bone_marrow must beat bone.
    ("bone marrow", "bone_marrow"),
    ("BM biopsy", "bone_marrow"),
    ("vertebra L2", "bone"),
    ("osseous metastasis", "bone"),
    ("skeletal involvement", "bone"),
    # Order-sensitive: leptomeningeal_cns must beat brain.
    ("leptomeningeal disease", "leptomeningeal_cns"),
    ("LMD progression", "leptomeningeal_cns"),
    ("CSF cytology positive", "leptomeningeal_cns"),
    ("spinal cord compression", "leptomeningeal_cns"),
    ("brain (frontal lobe)", "brain"),
    ("cerebral edema", "brain"),
    ("intracranial mass", "brain"),
    # Visceral
    ("liver lobe IV", "liver"),
    ("hepatic metastasis", "liver"),
    ("left lung apex", "lung"),
    ("pulmonary nodule", "lung"),
    ("pleural effusion right", "pleura"),
    ("pleural plaques", "pleura"),
    ("adrenal gland", "adrenal"),
    ("peritoneal carcinomatosis", "peritoneum"),
    ("ascites", "peritoneum"),
    ("omental cake", "peritoneum"),
    # Lymph
    ("right inguinal lymph node", "lymph_node"),
    ("axillary LN", "lymph_node"),
    # Soft tissue
    ("subcutaneous nodule", "soft_tissue"),
    ("muscle infiltration", "soft_tissue"),
    ("soft tissue mass", "soft_tissue"),
    # Primary tumor
    ("primary tumor breast", "primary_site"),
    # Fallback
    ("unknown organ", "other"),
    ("", "other"),
    (None, "other"),
    (42, "other"),  # non-string falls through to str() then misses
])
def test_normalize_site_class(label, expected) -> None:
    assert _normalize_site_class(label) == expected


# ---------------------------------------------------------------------------
# _site_signature — clone_id deliberately excluded
# ---------------------------------------------------------------------------


def test_site_signature_excludes_clone_id() -> None:
    """Per metastatic-tropism call: same anatomic site under different
    clones must produce the SAME signature."""
    sig_a = _site_signature("liver", "liver lobe IV")
    sig_b = _site_signature("liver", "liver lobe IV")
    assert sig_a == sig_b


def test_site_signature_is_lowercase_invariant() -> None:
    """Site label case must not affect signature (matches metric intent
    that 'Liver' and 'liver' are the same site)."""
    sig_lower = _site_signature("liver", "liver lobe iv")
    sig_upper = _site_signature("liver", "Liver Lobe IV")
    assert sig_lower == sig_upper


def test_site_signature_differs_for_different_classes() -> None:
    """Same anatomic label but different site_class → different signature."""
    sig_a = _site_signature("liver", "metastasis")
    sig_b = _site_signature("lung", "metastasis")
    assert sig_a != sig_b


# ---------------------------------------------------------------------------
# build_disease_site_rows — semantics on synthetic artifacts
# ---------------------------------------------------------------------------


def _make_artifact(snapshots: list[dict]) -> TrajectoryArtifact:
    cell = CellKey(
        trajectory_id="t1", condition_id="c", policy_id="p",
        model_id="m", model_short="m", future_index=1,
        tumor_oncotree_code="BRCA", conditioning_label="x",
        conditioning_hash="h", accepted=True,
    )
    materialized = {
        "baseline": {}, "ledger": [], "treatment_history": [],
        "snapshots": snapshots,
    }
    return TrajectoryArtifact(
        cell=cell, trajectory_dir=Path("/tmp"),
        parsed_ledger=(), materialized=materialized,
        trajectory_result={},
    )


def test_disease_site_rows_filters_absent_burden() -> None:
    """burden == 'absent' or None must NOT produce a row (matches
    snapshot_rows.csv 'active site' semantic)."""
    art = _make_artifact([
        {"date": "2024-01-15", "state_id": "s0", "clonal_state": [
            {"clone": "C0", "site": "liver", "burden": "absent"},
            {"clone": "C0", "site": "lung", "burden": "small"},
            {"clone": "C1", "site": "bone", "burden": None},
        ]},
    ])
    df = build_disease_site_rows([art])
    assert len(df) == 1
    assert df.iloc[0]["site_anatomic_label"] == "lung"


def test_disease_site_rows_is_baseline_only_at_index_zero() -> None:
    """is_baseline true iff snapshot_index == 0."""
    art = _make_artifact([
        {"date": "2024-01-15", "state_id": "s0", "clonal_state": [
            {"clone": "C0", "site": "liver", "burden": "small"},
        ]},
        {"date": "2024-04-15", "state_id": "s1", "clonal_state": [
            {"clone": "C0", "site": "liver", "burden": "moderate"},
        ]},
    ])
    df = build_disease_site_rows([art])
    assert len(df) == 2
    assert df.iloc[0]["is_baseline"] is True or df.iloc[0]["is_baseline"] == 1
    assert df.iloc[1]["is_baseline"] is False or df.iloc[1]["is_baseline"] == 0


def test_disease_site_rows_is_new_at_snapshot_against_prior_snapshots_only() -> None:
    """The semantic: 'not seen in any earlier snapshot'. Within-snapshot
    duplicates (same site_class+label across multiple clones) all share
    their is_new value because the signature is evaluated against the
    SEEN set from prior snapshots only, then signatures from the current
    snapshot are merged in AFTER the row is built."""
    art = _make_artifact([
        # snap 0: two clones at the SAME bone marrow site (same signature
        # — clone_id excluded). Both rows must be is_new=True since
        # nothing seen before.
        {"date": "2024-01-15", "state_id": "s0", "clonal_state": [
            {"clone": "C0", "site": "bone marrow", "burden": "microscopic"},
            {"clone": "C1", "site": "bone marrow", "burden": "small"},
        ]},
        # snap 1: two clones at brain (new site, never seen before — both
        # is_new=True); plus bone marrow again (seen in snap0 — is_new=False).
        {"date": "2024-04-15", "state_id": "s1", "clonal_state": [
            {"clone": "C0", "site": "brain", "burden": "moderate"},
            {"clone": "C2", "site": "brain", "burden": "small"},
            {"clone": "C0", "site": "bone marrow", "burden": "moderate"},
        ]},
    ])
    df = build_disease_site_rows([art])
    assert len(df) == 5
    # snap 0: both bone marrow rows is_new=True
    snap0 = df[df["snapshot_index"] == 0]
    assert all(bool(v) for v in snap0["is_new_at_snapshot"])
    # snap 1: both brain rows is_new=True (new signature in snap1),
    # bone marrow row is_new=False (signature from snap0).
    snap1 = df[df["snapshot_index"] == 1]
    brain_rows = snap1[snap1["site_class"] == "brain"]
    bone_rows = snap1[snap1["site_class"] == "bone_marrow"]
    assert all(bool(v) for v in brain_rows["is_new_at_snapshot"])
    assert all(not bool(v) for v in bone_rows["is_new_at_snapshot"])


def test_disease_site_rows_assigns_within_snapshot_ordinal_site_id() -> None:
    """site_id is a within-snapshot ordinal starting at 0 for active sites."""
    art = _make_artifact([
        {"date": "2024-01-15", "state_id": "s0", "clonal_state": [
            {"clone": "C0", "site": "liver", "burden": "absent"},  # skipped
            {"clone": "C0", "site": "lung", "burden": "small"},    # site_id=0
            {"clone": "C0", "site": "brain", "burden": "small"},   # site_id=1
        ]},
    ])
    df = build_disease_site_rows([art])
    assert list(df["site_id"]) == [0, 1]


def test_disease_site_rows_carries_site_class_via_normalize() -> None:
    """Legacy/null site_class rows still fall back to _normalize_site_class."""
    art = _make_artifact([
        {"date": "2024-01-15", "state_id": "s0", "clonal_state": [
            {"clone": "C0", "site": "bone marrow biopsy", "burden": "small"},
            {"clone": "C1", "site": "leptomeningeal spread", "burden": "small"},
            {"clone": "C2", "site": "unknown sandbox", "burden": "small"},
        ]},
    ])
    df = build_disease_site_rows([art])
    assert list(df["site_class"]) == ["bone_marrow", "leptomeningeal_cns", "other"]
    assert list(df["site_class_canonical"]) == [
        "bone_marrow",
        "leptomeningeal_cns",
        "other",
    ]
    assert all(not bool(v) for v in df["site_class_discordance"])


def test_disease_site_rows_prefers_model_emitted_site_class() -> None:
    """Paper-v1 rows preserve the LLM's bucket instead of re-normalizing text."""
    art = _make_artifact([
        {"date": "2024-01-15", "state_id": "s0", "clonal_state": [
            {
                "clone": "C0",
                "site": "right hepatic lobe",
                "site_class": "liver",
                "burden": "small",
            },
            {
                "clone": "C0",
                "site": "ambiguous treated bed",
                "site_class": "other",
                "burden": "small",
            },
        ]},
    ])
    df = build_disease_site_rows([art])
    assert list(df["site_class"]) == ["liver", "other"]
    assert list(df["site_class_canonical"]) == ["liver", "other"]


def test_disease_site_rows_canonicalizes_primary_site_class_aliases() -> None:
    """v4 preserves raw disease-site aliases but uses canonical signatures."""
    art = _make_artifact([
        {"date": "2024-01-15", "state_id": "s0", "clonal_state": [
            {
                "clone": "C0",
                "site": "right frontal brain lesion",
                "site_class": "central_nervous_system",
                "burden": "small",
            },
            {
                "clone": "C1",
                "site": "bone marrow",
                "site_class": "bone_marrow_blood",
                "burden": "microscopic",
            },
            {
                "clone": "C2",
                "site": "retroperitoneal mass",
                "site_class": "retroperitoneum",
                "burden": "moderate",
            },
        ]},
    ])
    df = build_disease_site_rows([art])
    assert list(df["site_class"]) == [
        "central_nervous_system",
        "bone_marrow_blood",
        "retroperitoneum",
    ]
    assert list(df["site_class_canonical"]) == [
        "brain",
        "bone_marrow",
        "retroperitoneum",
    ]
    assert [bool(v) for v in df["site_class_discordance"]] == [True, True, False]
    assert df.iloc[0]["site_signature"] == _site_signature(
        "brain",
        "right frontal brain lesion",
    )
