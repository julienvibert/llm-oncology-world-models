"""classify_response contract for the baseline_patch + ledger shape."""

from __future__ import annotations

import json
from pathlib import Path

from synthonccases.oncollm.conditioning_audit import compute_pinned_paths
from synthonccases.oncollm.poc_runner import (
    FixedFirstLineTreatment,
    PolicySpec,
    RegimenLine,
    classify_response,
)


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "rev5_poc" / "her2_brca_baseline_002.json"
)


def _full() -> dict:
    return json.loads(FIXTURE.read_text())


def _death_ledger(diagnosis_date: str = "2024-05-15") -> list[dict]:
    return [
        {
            "date": "2026-03-01",
            "type": "death",
            "cause_category": "cancer_progression",
            "cause": "progressive disease",
            "eol_care_pattern": "inpatient_palliative",
        },
    ]


def _response(*, baseline_patch: dict, ledger: list[dict]) -> str:
    return json.dumps({"baseline_patch": baseline_patch, "ledger": ledger})


# ---------- shape checks ----------


def test_missing_baseline_patch_is_parse_failure() -> None:
    raw = json.dumps({"ledger": _death_ledger()})
    kind, _, parsed = classify_response(
        raw, baseline_seed=_full(), pinned_paths=(),
    )
    assert kind == "parse"
    assert parsed is None


def test_missing_ledger_is_parse_failure() -> None:
    raw = json.dumps({"baseline_patch": {}})
    kind, _, _ = classify_response(
        raw, baseline_seed=_full(), pinned_paths=(),
    )
    assert kind == "parse"


# ---------- full-seed contract ----------


def test_full_seed_empty_patch_succeeds() -> None:
    seed = _full()
    pinned = compute_pinned_paths(seed)
    raw = _response(baseline_patch={}, ledger=_death_ledger())
    kind, message, parsed = classify_response(
        raw, baseline_seed=seed, pinned_paths=pinned,
    )
    assert kind is None, message
    assert parsed is not None


def test_full_seed_nonempty_patch_triggers_collision_audit() -> None:
    seed = _full()
    pinned = compute_pinned_paths(seed)
    # Any patch write will collide with some pinned path under the full seed
    raw = _response(
        baseline_patch={"identity_card": {"demographics": {"sex": "male"}}},
        ledger=_death_ledger(),
    )
    kind, message, _ = classify_response(
        raw, baseline_seed=seed, pinned_paths=pinned,
    )
    assert kind == "conditioning_audit"
    assert "demographics.sex" in message  # type: ignore[operator]


# ---------- spontaneous + full patch ----------


def test_spontaneous_with_complete_patch_succeeds() -> None:
    full = _full()
    raw = _response(baseline_patch=full, ledger=_death_ledger())
    kind, message, parsed = classify_response(
        raw, baseline_seed={}, pinned_paths=(),
    )
    assert kind is None, message
    assert parsed is not None


def test_spontaneous_with_incomplete_patch_baseline_validation() -> None:
    raw = _response(baseline_patch={"baseline_id": "x"}, ledger=_death_ledger())
    kind, _, _ = classify_response(
        raw, baseline_seed={}, pinned_paths=(),
    )
    assert kind == "baseline_validation"


# ---------- partial seed ----------


def test_partial_seed_complementary_patch_succeeds() -> None:
    full = _full()
    diagnosis_seed = {
        "identity_card": {"diagnosis": full["identity_card"]["diagnosis"]},
    }
    pinned = compute_pinned_paths(diagnosis_seed)
    # patch covers everything except identity_card.diagnosis
    patch = {k: v for k, v in full.items() if k != "identity_card"}
    patch["identity_card"] = {
        k: v for k, v in full["identity_card"].items() if k != "diagnosis"
    }
    raw = _response(baseline_patch=patch, ledger=_death_ledger())
    kind, message, parsed = classify_response(
        raw, baseline_seed=diagnosis_seed, pinned_paths=pinned,
    )
    assert kind is None, message
    assert parsed is not None


def test_partial_seed_patch_writes_pinned_path_collision() -> None:
    """Pinned tumor type, model emits a different tumor type in its patch."""
    seed = {"identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}}
    pinned = compute_pinned_paths(seed)
    full = _full()
    patch = {k: v for k, v in full.items() if k != "identity_card"}
    # Patch writes the same pinned leaf with a drifted value
    patch["identity_card"] = full["identity_card"]
    raw = _response(baseline_patch=patch, ledger=_death_ledger())
    kind, message, _ = classify_response(
        raw, baseline_seed=seed, pinned_paths=pinned,
    )
    assert kind == "conditioning_audit"
    assert "oncotree_code" in message  # type: ignore[operator]


# ---------- full-baseline empty-patch gate ----------


def test_full_baseline_rejects_patch_on_unpinned_optional_leaf() -> None:
    """A complete valid seed can omit optional leaves (e.g.
    identity_card.diagnosis.diagnosis_label). The model can therefore craft
    a patch that does NOT collide with any pinned path but still violates
    the full-baseline contract. seed_is_complete=True must reject ANY
    non-empty patch, including writes to optional unpinned leaves."""
    full = _full()
    # Sanity: ensure diagnosis_label is NOT in the seed (it's optional).
    assert "diagnosis_label" not in full["identity_card"]["diagnosis"]
    pinned = compute_pinned_paths(full)
    assert "identity_card.diagnosis.diagnosis_label" not in pinned

    raw = _response(
        baseline_patch={"identity_card": {"diagnosis": {"diagnosis_label": "BRCA"}}},
        ledger=_death_ledger(),
    )
    kind, message, _ = classify_response(
        raw,
        baseline_seed=full,
        pinned_paths=pinned,
        seed_is_complete=True,
    )
    assert kind == "conditioning_audit"
    assert "complete BaselineObject" in message  # type: ignore[operator]


def test_full_baseline_with_empty_patch_still_succeeds() -> None:
    full = _full()
    raw = _response(baseline_patch={}, ledger=_death_ledger())
    kind, message, parsed = classify_response(
        raw,
        baseline_seed=full,
        pinned_paths=compute_pinned_paths(full),
        seed_is_complete=True,
    )
    assert kind is None, message
    assert parsed is not None


# ---------- intervention-audit gate (fixed_first_line_treatment) ----------


def _thp_policy_with_ffl() -> PolicySpec:
    return PolicySpec(
        id="A_THP",
        text="Start first-line THP.",
        regimen_sequence=(
            RegimenLine(line=1, aliases=("THP", "trastuzumab/pertuzumab/docetaxel")),
        ),
        unconstrained_after_line=None,
        fixed_first_line_treatment=FixedFirstLineTreatment(
            regimen_label="trastuzumab_pertuzumab_docetaxel",
            drugs=("trastuzumab", "pertuzumab", "docetaxel"),
        ),
    )


def _ledger_with_first_regimen(label: str, components: list[str]) -> list[dict]:
    class_by_component = {
        "trastuzumab": "anti_her2",
        "pertuzumab": "anti_her2",
        "docetaxel": "cytotoxic_chemotherapy",
        "trastuzumab_emtansine": "antibody_drug_conjugate",
    }
    return [
        {
            "date": "2024-02-01",
            "type": "systemic_treatment_start",
            "regimen": {
                "label": label,
                "components": components,
                "component_classes": [
                    class_by_component.get(component, "other")
                    for component in components
                ],
            },
        },
        {
            "date": "2026-03-01",
            "type": "death",
            "cause_category": "cancer_progression",
            "cause": "progressive disease",
            "eol_care_pattern": "inpatient_palliative",
        },
    ]


def test_intervention_audit_rejects_first_line_drift() -> None:
    full = _full()
    raw = _response(
        baseline_patch={},
        ledger=_ledger_with_first_regimen("T-DM1", ["trastuzumab_emtansine"]),
    )
    kind, message, _ = classify_response(
        raw,
        baseline_seed=full,
        pinned_paths=compute_pinned_paths(full),
        seed_is_complete=True,
        policy=_thp_policy_with_ffl(),
    )
    assert kind == "intervention_audit"
    assert "fixed_first_line_treatment" in message or "T-DM1" in message  # type: ignore[operator]


def test_intervention_audit_accepts_first_line_match_by_label() -> None:
    full = _full()
    raw = _response(
        baseline_patch={},
        ledger=_ledger_with_first_regimen("THP", ["trastuzumab", "pertuzumab", "docetaxel"]),
    )
    kind, message, parsed = classify_response(
        raw,
        baseline_seed=full,
        pinned_paths=compute_pinned_paths(full),
        seed_is_complete=True,
        policy=_thp_policy_with_ffl(),
    )
    assert kind is None, message
    assert parsed is not None


def test_intervention_audit_accepts_first_line_match_by_full_spelling() -> None:
    """The alias 'trastuzumab/pertuzumab/docetaxel' covers the long-form
    label spelling. The matcher normalizes punctuation; case-insensitive."""
    full = _full()
    raw = _response(
        baseline_patch={},
        ledger=_ledger_with_first_regimen(
            "trastuzumab/pertuzumab/docetaxel",
            ["trastuzumab", "pertuzumab", "docetaxel"],
        ),
    )
    kind, message, parsed = classify_response(
        raw,
        baseline_seed=full,
        pinned_paths=compute_pinned_paths(full),
        seed_is_complete=True,
        policy=_thp_policy_with_ffl(),
    )
    assert kind is None, message
    assert parsed is not None


def test_intervention_audit_rejects_missing_systemic_treatment_start() -> None:
    """When FFL is set but the ledger has no systemic_treatment_start
    event at all, intervention_audit fires."""
    full = _full()
    raw = _response(
        baseline_patch={},
        ledger=[
            {
                "date": "2025-01-01",
                "type": "local_intervention",
                "modality": "surgery",
                "procedure": "mastectomy",
            },
            {
                "date": "2026-03-01",
                "type": "death",
                "cause_category": "cancer_progression",
                "cause": "x",
                "eol_care_pattern": "hospice",
            },
        ],
    )
    kind, message, _ = classify_response(
        raw,
        baseline_seed=full,
        pinned_paths=compute_pinned_paths(full),
        seed_is_complete=True,
        policy=_thp_policy_with_ffl(),
    )
    assert kind == "intervention_audit"
    assert "no systemic_treatment_start" in message  # type: ignore[operator]


def test_no_intervention_gate_when_policy_has_no_fixed_first_line() -> None:
    """An unconstrained policy (no FFL) does not run the intervention
    gate — any first regimen is accepted."""
    full = _full()
    policy_no_ffl = PolicySpec(
        id="unconstrained",
        text="Choose any.",
        regimen_sequence=(),
        unconstrained_after_line=None,
        fixed_first_line_treatment=None,
    )
    raw = _response(
        baseline_patch={},
        ledger=_ledger_with_first_regimen("anything_goes", ["random_drug"]),
    )
    kind, message, parsed = classify_response(
        raw,
        baseline_seed=full,
        pinned_paths=compute_pinned_paths(full),
        seed_is_complete=True,
        policy=policy_no_ffl,
    )
    assert kind is None, message
    assert parsed is not None


# ---------- failure-kind enumeration ----------


def test_failure_kinds_includes_new_audit_kinds() -> None:
    from synthonccases.oncollm.poc_runner import FAILURE_KINDS
    assert "conditioning_audit" in FAILURE_KINDS
    assert "baseline_validation" in FAILURE_KINDS
    assert "intervention_audit" in FAILURE_KINDS
