"""Gate A1 — ``lower_conditioning_cell`` legacy + generalized equivalence.

Covers the five conditioning regimes currently shipping in
``configs/paper_v1_*.yaml`` (spontaneous / tumor_only / partial_baseline /
full_baseline / +1L) and the new generalized ``conditioning:`` shape. Each
case asserts the lowered ``ConditioningSpec`` carries the right axes — not
that hashes match across regimes, which is independent.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from synthonccases.oncollm.condition_lower import (
    LEGACY_FIRST_TREATMENT_CONSTRAINT_ID,
    lower_conditioning_cell,
)
from synthonccases.oncollm.conditioning_audit import (
    EMPTY_CONSTRAINTS_SENTINEL,
    EMPTY_PREFIX_SENTINEL,
    compute_future_constraint_hash,
    compute_observed_prefix_hash,
)
from synthonccases.oncollm.replay import BaselineObject


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "rev5_poc" / "her2_brca_baseline_002.json"
)


# ---------------------------------------------------------------------------
# Legacy regimes — five current paper-v1 configs lower into equivalent specs.
# ---------------------------------------------------------------------------


def test_lower_spontaneous_legacy() -> None:
    spec = lower_conditioning_cell(
        condition_payload={"baseline_seed": {}},
        policy_payload={"text": ""},
    )
    assert spec.baseline_seed == {}
    assert spec.pinned_paths == ()
    assert spec.observed_prefix is None
    assert spec.future_event_constraints == ()
    assert spec.free_text_operator_constraints == ""
    assert compute_observed_prefix_hash(spec) == EMPTY_PREFIX_SENTINEL
    assert compute_future_constraint_hash(spec) == EMPTY_CONSTRAINTS_SENTINEL


def test_lower_tumor_only_legacy() -> None:
    spec = lower_conditioning_cell(
        condition_payload={
            "baseline_seed": {
                "identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}
            }
        },
        policy_payload={"text": "Treat per oncologist's clinical judgment."},
    )
    assert spec.pinned_paths == ("identity_card.diagnosis.oncotree_code",)
    assert spec.observed_prefix is None
    assert spec.future_event_constraints == ()
    assert spec.free_text_operator_constraints.startswith("Treat per oncologist")


def test_lower_partial_baseline_legacy() -> None:
    seed = {
        "identity_card": {
            "demographics": {"age_at_diagnosis": 62, "sex": "female"},
            "diagnosis": {
                "oncotree_code": "BRCA",
                "diagnosis_date": "2024-01-15",
            },
        }
    }
    spec = lower_conditioning_cell(
        condition_payload={"baseline_seed": seed},
        policy_payload={"text": ""},
    )
    # Expected leaves: demographics.age_at_diagnosis, demographics.sex,
    # diagnosis.oncotree_code, diagnosis.diagnosis_date — 4 total.
    assert set(spec.pinned_paths) == {
        "identity_card.demographics.age_at_diagnosis",
        "identity_card.demographics.sex",
        "identity_card.diagnosis.oncotree_code",
        "identity_card.diagnosis.diagnosis_date",
    }
    assert spec.observed_prefix is None
    assert spec.future_event_constraints == ()


def test_lower_full_baseline_legacy() -> None:
    seed = json.loads(FIXTURE.read_text(encoding="utf-8"))
    spec = lower_conditioning_cell(
        condition_payload={"baseline_seed": seed},
        policy_payload={"text": ""},
    )
    # Hard lock: a "full baseline" fixture must Pydantic-validate cleanly so
    # downstream code can rely on ``BaselineObject.model_validate(seed)``
    # succeeding for the full-baseline regime.
    BaselineObject.model_validate(spec.baseline_seed)
    assert spec.observed_prefix is None
    assert spec.future_event_constraints == ()


def test_lower_plus_one_l_legacy() -> None:
    seed = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fixed_first = {
        "regimen_label": "THP",
        "drugs": ["trastuzumab", "pertuzumab", "docetaxel"],
        "schedule": "q3w",
    }
    spec = lower_conditioning_cell(
        condition_payload={"baseline_seed": seed},
        policy_payload={
            "text": "First line: THP. Continue until progression.",
            "fixed_first_line_treatment": fixed_first,
        },
    )
    assert len(spec.future_event_constraints) == 1
    constraint = spec.future_event_constraints[0]
    assert constraint.constraint_id == LEGACY_FIRST_TREATMENT_CONSTRAINT_ID
    assert constraint.selector.event_type == "systemic_treatment_start"
    assert constraint.selector.ordinal == "first"
    assert constraint.selector.ordinal_after_cutoff is None
    # regimen_matches preserves order and de-duplicates: label first, then
    # the three drugs.
    assert constraint.constraint["regimen_matches"] == [
        "THP",
        "trastuzumab",
        "pertuzumab",
        "docetaxel",
    ]


def test_lower_plus_one_l_dedupes_alias_when_label_in_drugs() -> None:
    """Drug list with the regimen label itself does not produce a duplicate."""
    fixed_first = {
        "regimen_label": "trastuzumab",
        "drugs": ["trastuzumab"],
    }
    spec = lower_conditioning_cell(
        condition_payload={"baseline_seed": {}},
        policy_payload={"fixed_first_line_treatment": fixed_first, "text": ""},
    )
    assert spec.future_event_constraints[0].constraint["regimen_matches"] == [
        "trastuzumab"
    ]


# ---------------------------------------------------------------------------
# Legacy +1L resolution must match the runner's `_resolve_line1_aliases`
# chain exactly. Three priority levels:
#   1. regimen_sequence[line=1].aliases
#   2. first_treatment.aliases (explicit override)
#   3. (regimen_label, *drugs)
# ---------------------------------------------------------------------------


def test_lower_plus_one_l_cell_shape_with_first_treatment_only() -> None:
    """Cell shape uses ``first_treatment`` (not ``fixed_first_line_treatment``)."""
    cell = {
        "baseline_seed": {},
        "first_treatment": {
            "regimen_label": "THP",
            "drugs": ["trastuzumab", "pertuzumab", "docetaxel"],
        },
        "text": "",
    }
    spec = lower_conditioning_cell(
        condition_payload=cell, policy_payload=cell
    )
    assert len(spec.future_event_constraints) == 1
    assert spec.future_event_constraints[0].constraint["regimen_matches"] == [
        "THP",
        "trastuzumab",
        "pertuzumab",
        "docetaxel",
    ]


def test_lower_plus_one_l_regimen_sequence_wins_over_first_treatment_drugs() -> None:
    """When ``regimen_sequence[line=1].aliases`` is set, it takes precedence
    over the raw ``regimen_label`` + ``drugs`` fields — mirroring the runner's
    ``_resolve_line1_aliases``.
    """
    cell = {
        "baseline_seed": {},
        "first_treatment": {
            "regimen_label": "THP",
            "drugs": ["trastuzumab", "pertuzumab", "docetaxel"],
        },
        "regimen_sequence": [
            {
                "line": 1,
                "aliases": ["THP", "trastuzumab/pertuzumab/docetaxel"],
            },
            {"line": 2, "aliases": ["T-DM1"]},
        ],
        "text": "",
    }
    spec = lower_conditioning_cell(
        condition_payload=cell, policy_payload=cell
    )
    assert spec.future_event_constraints[0].constraint["regimen_matches"] == [
        "THP",
        "trastuzumab/pertuzumab/docetaxel",
    ]


def test_lower_plus_one_l_explicit_aliases_override_wins_over_drugs() -> None:
    """When ``first_treatment.aliases`` is set but no ``regimen_sequence``
    line-1 row exists, the explicit override wins over the raw label+drugs.
    """
    cell = {
        "baseline_seed": {},
        "first_treatment": {
            "regimen_label": "trastuzumab_pertuzumab_docetaxel",
            "aliases": ["THP", "trastuzumab/pertuzumab/docetaxel"],
            "drugs": ["trastuzumab", "pertuzumab", "docetaxel"],
        },
        "text": "",
    }
    spec = lower_conditioning_cell(
        condition_payload=cell, policy_payload=cell
    )
    assert spec.future_event_constraints[0].constraint["regimen_matches"] == [
        "THP",
        "trastuzumab/pertuzumab/docetaxel",
    ]


def test_lower_plus_one_l_regimen_sequence_without_line1_falls_through() -> None:
    """A ``regimen_sequence`` without a ``line: 1`` row does not block the
    fallback to raw label+drugs.
    """
    cell = {
        "baseline_seed": {},
        "first_treatment": {
            "regimen_label": "THP",
            "drugs": ["trastuzumab"],
        },
        "regimen_sequence": [
            {"line": 2, "aliases": ["T-DM1"]},
        ],
        "text": "",
    }
    spec = lower_conditioning_cell(
        condition_payload=cell, policy_payload=cell
    )
    # Fall through to (regimen_label, *drugs).
    assert spec.future_event_constraints[0].constraint["regimen_matches"] == [
        "THP",
        "trastuzumab",
    ]


def test_lower_plus_one_l_regimen_sequence_without_line1_loses_explicit_aliases() -> None:
    """Runner-equivalence edge: ``regimen_sequence`` present but no ``line: 1``
    row → ``first_treatment.aliases`` is LOST, not used as override.

    The runner's two-stage logic synthesizes a default line-1 row from
    ``first_treatment.aliases`` ONLY when ``regimen_sequence`` is fully
    empty. When ``regimen_sequence`` is non-empty but lacks a line-1 row,
    the runner's ``_resolve_line1_aliases`` falls through directly to
    ``(regimen_label, *drugs)`` — the explicit ``aliases`` override is
    silently dropped. This test locks that behavior.
    """
    cell = {
        "baseline_seed": {},
        "first_treatment": {
            "regimen_label": "THP",
            "aliases": ["explicit_override_that_runner_loses"],
            "drugs": ["trastuzumab"],
        },
        "regimen_sequence": [
            {"line": 2, "aliases": ["T-DM1"]},
        ],
        "text": "",
    }
    spec = lower_conditioning_cell(
        condition_payload=cell, policy_payload=cell
    )
    # NOT ["explicit_override_that_runner_loses"] — that alias is lost.
    # Falls through to (regimen_label, *drugs).
    assert spec.future_event_constraints[0].constraint["regimen_matches"] == [
        "THP",
        "trastuzumab",
    ]


def test_lower_plus_one_l_no_declaration_emits_no_constraint() -> None:
    """A cell with no ``first_treatment`` or ``fixed_first_line_treatment``
    block emits NO future-constraint, even if ``regimen_sequence`` is set.
    """
    cell = {
        "baseline_seed": {},
        "regimen_sequence": [
            {"line": 1, "aliases": ["THP"]},
        ],
        "text": "",
    }
    spec = lower_conditioning_cell(
        condition_payload=cell, policy_payload=cell
    )
    assert spec.future_event_constraints == ()


# ---------------------------------------------------------------------------
# Fix 2 — non-`observed_prefix` origins are rejected.
# ---------------------------------------------------------------------------


def test_lower_generalized_rejects_unknown_origin() -> None:
    payload = {
        "baseline_seed": {},
        "fixed_events": {
            "origin": "synthetic_prior_run",
            "cutoff_date": "2024-07-15",
            "events": [],
        },
    }
    with pytest.raises(ValueError, match="origin: 'synthetic_prior_run'"):
        lower_conditioning_cell(conditioning_payload=payload)


def test_lower_generalized_origin_defaults_to_observed_prefix() -> None:
    """Operator may omit ``origin`` entirely; default is ``observed_prefix``."""
    payload = {
        "baseline_seed": {},
        "fixed_events": {
            "cutoff_date": "2024-07-15",
            "events": [],
        },
    }
    spec = lower_conditioning_cell(conditioning_payload=payload)
    assert spec.observed_prefix is not None
    assert spec.observed_prefix.origin == "observed_prefix"


# ---------------------------------------------------------------------------
# Generalized regime — the new ``conditioning:`` YAML shape.
# ---------------------------------------------------------------------------


def test_lower_generalized_with_prefix_and_constraint() -> None:
    payload = {
        "baseline_seed": {
            "identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}
        },
        "fixed_events": {
            "origin": "observed_prefix",
            "cutoff_date": "2024-07-15",
            "events": [
                {
                    "date": "2024-04-15",
                    "type": "systemic_treatment_start",
                    "regimen": {
                        "label": "THP",
                        "components": [
                            "trastuzumab",
                            "pertuzumab",
                            "docetaxel",
                        ],
                    },
                },
                {
                    "date": "2024-06-30",
                    "type": "response",
                },
            ],
        },
        "future_event_constraints": [
            {
                "constraint_id": "post_cutoff_switch_to_tdm1",
                "selector": {
                    "event_type": "systemic_treatment_start",
                    "ordinal_after_cutoff": "first",
                },
                "constraint": {"regimen_matches": ["T-DM1"]},
            }
        ],
        "free_text_operator_constraints": "Switch to T-DM1 on first progression.",
    }
    spec = lower_conditioning_cell(conditioning_payload=payload)

    assert spec.pinned_paths == ("identity_card.diagnosis.oncotree_code",)
    assert spec.observed_prefix is not None
    assert spec.observed_prefix.cutoff_date == date(2024, 7, 15)
    assert len(spec.observed_prefix.events) == 2
    assert spec.observed_prefix.events[0].type == "systemic_treatment_start"
    assert spec.observed_prefix.events[1].type == "response"

    assert len(spec.future_event_constraints) == 1
    c = spec.future_event_constraints[0]
    assert c.constraint_id == "post_cutoff_switch_to_tdm1"
    assert c.selector.event_type == "systemic_treatment_start"
    assert c.selector.ordinal_after_cutoff == "first"
    assert c.constraint["regimen_matches"] == ["T-DM1"]

    assert spec.free_text_operator_constraints.startswith("Switch to T-DM1")
    # All four hashes are populated (not sentinel) — this spec exercises every axis.
    assert compute_observed_prefix_hash(spec) != EMPTY_PREFIX_SENTINEL
    assert compute_future_constraint_hash(spec) != EMPTY_CONSTRAINTS_SENTINEL


def test_lower_generalized_with_only_baseline() -> None:
    """A generalized payload with no prefix and no constraints lowers cleanly."""
    payload = {
        "baseline_seed": {
            "identity_card": {"diagnosis": {"oncotree_code": "LUAD"}}
        }
    }
    spec = lower_conditioning_cell(conditioning_payload=payload)
    assert spec.pinned_paths == ("identity_card.diagnosis.oncotree_code",)
    assert spec.observed_prefix is None
    assert spec.future_event_constraints == ()


def test_lower_generalized_rejects_fixed_events_without_cutoff() -> None:
    payload = {
        "baseline_seed": {},
        "fixed_events": {"events": [{"date": "2024-01-01", "type": "response"}]},
    }
    with pytest.raises(ValueError, match="cutoff_date is required"):
        lower_conditioning_cell(conditioning_payload=payload)


def test_lower_generalized_rejects_constraint_without_id() -> None:
    payload = {
        "baseline_seed": {},
        "future_event_constraints": [
            {
                "selector": {"event_type": "response"},
                "constraint": {"must_not_occur": True},
            }
        ],
    }
    with pytest.raises(ValueError, match="constraint_id is required"):
        lower_conditioning_cell(conditioning_payload=payload)


def test_lower_generalized_rejects_selector_without_event_type() -> None:
    payload = {
        "baseline_seed": {},
        "future_event_constraints": [
            {
                "constraint_id": "missing_event_type",
                "selector": {},
                "constraint": {"must_not_occur": True},
            }
        ],
    }
    with pytest.raises(ValueError, match="event_type is required"):
        lower_conditioning_cell(conditioning_payload=payload)


def test_lower_generalized_parses_selector_date_range() -> None:
    payload = {
        "baseline_seed": {},
        "future_event_constraints": [
            {
                "constraint_id": "window",
                "selector": {
                    "event_type": "progression",
                    "date_range": ["2024-04-01", "2024-12-31"],
                },
                "constraint": {"must_not_occur": True},
            }
        ],
    }
    spec = lower_conditioning_cell(conditioning_payload=payload)
    sel = spec.future_event_constraints[0].selector
    assert sel.date_range == (date(2024, 4, 1), date(2024, 12, 31))


# ---------------------------------------------------------------------------
# Precedence — generalized wins when both shapes are passed.
# ---------------------------------------------------------------------------


def test_generalized_payload_takes_precedence_over_legacy() -> None:
    """When both shapes are passed, the generalized payload wins outright.

    No silent merging — the legacy condition/policy payloads are ignored so
    the lowering path is unambiguous at the call site.
    """
    legacy_condition = {
        "baseline_seed": {
            "identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}
        }
    }
    legacy_policy = {
        "text": "ignored",
        "fixed_first_line_treatment": {
            "regimen_label": "THP",
            "drugs": ["trastuzumab"],
        },
    }
    generalized = {
        "baseline_seed": {
            "identity_card": {"diagnosis": {"oncotree_code": "LUAD"}}
        },
        "free_text_operator_constraints": "kept",
    }
    spec = lower_conditioning_cell(
        condition_payload=legacy_condition,
        policy_payload=legacy_policy,
        conditioning_payload=generalized,
    )
    # Legacy ignored: baseline came from generalized, not legacy.
    assert spec.pinned_paths == ("identity_card.diagnosis.oncotree_code",)
    seed_oncotree = spec.baseline_seed["identity_card"]["diagnosis"]["oncotree_code"]
    assert seed_oncotree == "LUAD"
    # Legacy 1L constraint NOT applied.
    assert spec.future_event_constraints == ()
    assert spec.free_text_operator_constraints == "kept"


def test_all_none_lowers_to_empty_spec() -> None:
    spec = lower_conditioning_cell()
    assert spec.baseline_seed == {}
    assert spec.pinned_paths == ()
    assert spec.observed_prefix is None
    assert spec.future_event_constraints == ()
    assert spec.free_text_operator_constraints == ""
