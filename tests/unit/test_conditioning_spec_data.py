"""Gate A1 — generalized ``ConditioningSpec`` dataclasses + four-hash composition.

These tests cover the additive surface introduced in Gate A1:

- dataclass-level validation (duplicate constraint_id, mutually-exclusive
  ordinals on a selector);
- canonicalization round-trips through ``_canonicalize_spec_value``;
- deterministic + reorder-stable + sentinel-composable hashing across the
  four helpers (``compute_baseline_hash`` / ``compute_observed_prefix_hash`` /
  ``compute_future_constraint_hash`` / ``compute_conditioning_spec_hash``);
- back-compat lock: the legacy ``compute_conditioning_hash(seed_hash=...,
  canonical_policy_payload=...)`` still works unchanged, so the existing
  runner pipeline is not affected by the A1 additions.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from synthonccases.oncollm.conditioning_audit import (
    EMPTY_CONSTRAINTS_SENTINEL,
    EMPTY_PREFIX_SENTINEL,
    ConditioningSpec,
    FutureEventConstraint,
    FutureEventSelector,
    ObservedPrefix,
    _canonicalize_spec_value,
    canonical_json,
    compute_baseline_hash,
    compute_conditioning_hash,
    compute_conditioning_spec_hash,
    compute_future_constraint_hash,
    compute_observed_prefix_hash,
    compute_pinned_paths,
    compute_seed_hash,
)
from synthonccases.oncollm.replay import LedgerEvent


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "rev5_poc" / "her2_brca_baseline_002.json"
)


# ---------------------------------------------------------------------------
# Fixtures (plain Python builders, not pytest fixtures — keeps the tests
# explicit about what shape they exercise).
# ---------------------------------------------------------------------------


def _empty_spec() -> ConditioningSpec:
    return ConditioningSpec(
        baseline_seed={},
        pinned_paths=(),
        observed_prefix=None,
        future_event_constraints=(),
        free_text_operator_constraints="",
    )


def _tumor_only_spec() -> ConditioningSpec:
    seed = {"identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}}
    return ConditioningSpec(
        baseline_seed=seed,
        pinned_paths=compute_pinned_paths(seed),
        observed_prefix=None,
        future_event_constraints=(),
        free_text_operator_constraints="",
    )


def _spec_with_two_constraints(*, swap: bool = False) -> ConditioningSpec:
    c_a = FutureEventConstraint(
        constraint_id="a_first_treatment",
        selector=FutureEventSelector(
            event_type="systemic_treatment_start", ordinal="first"
        ),
        constraint={"regimen_matches": ["THP"]},
    )
    c_b = FutureEventConstraint(
        constraint_id="b_no_local",
        selector=FutureEventSelector(event_type="local_intervention"),
        constraint={"must_not_occur": True},
    )
    order = (c_b, c_a) if swap else (c_a, c_b)
    return ConditioningSpec(
        baseline_seed={},
        pinned_paths=(),
        observed_prefix=None,
        future_event_constraints=order,
        free_text_operator_constraints="",
    )


def _spec_with_prefix() -> ConditioningSpec:
    prefix_event = LedgerEvent.model_validate(
        {
            "date": "2024-04-15",
            "type": "systemic_treatment_start",
            "regimen": {
                "label": "THP",
                "components": ["trastuzumab", "pertuzumab", "docetaxel"],
            },
        }
    )
    return ConditioningSpec(
        baseline_seed={},
        pinned_paths=(),
        observed_prefix=ObservedPrefix(
            events=(prefix_event,),
            cutoff_date=date(2024, 7, 15),
        ),
        future_event_constraints=(),
        free_text_operator_constraints="",
    )


# ---------------------------------------------------------------------------
# Dataclass-level validation
# ---------------------------------------------------------------------------


def test_selector_rejects_empty_event_type() -> None:
    with pytest.raises(ValueError, match="event_type must be non-empty"):
        FutureEventSelector(event_type="")


def test_selector_rejects_whitespace_only_event_type() -> None:
    with pytest.raises(ValueError, match="event_type must be non-empty"):
        FutureEventSelector(event_type="   ")


def test_selector_rejects_unknown_ordinal() -> None:
    with pytest.raises(ValueError, match="ordinal: 'fourth' is not one of"):
        FutureEventSelector(event_type="response", ordinal="fourth")  # type: ignore[arg-type]


def test_selector_rejects_unknown_ordinal_after_cutoff() -> None:
    with pytest.raises(
        ValueError, match="ordinal_after_cutoff: 'middle' is not one of"
    ):
        FutureEventSelector(
            event_type="response",
            ordinal_after_cutoff="middle",  # type: ignore[arg-type]
        )


def test_selector_rejects_both_ordinals_set() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        FutureEventSelector(
            event_type="systemic_treatment_start",
            ordinal="first",
            ordinal_after_cutoff="first",
        )


def test_selector_accepts_only_ordinal() -> None:
    sel = FutureEventSelector(
        event_type="systemic_treatment_start", ordinal="first"
    )
    assert sel.ordinal == "first"
    assert sel.ordinal_after_cutoff is None


def test_selector_accepts_only_ordinal_after_cutoff() -> None:
    sel = FutureEventSelector(
        event_type="systemic_treatment_start", ordinal_after_cutoff="first"
    )
    assert sel.ordinal is None
    assert sel.ordinal_after_cutoff == "first"


def test_constraint_rejects_empty_constraint_id() -> None:
    selector = FutureEventSelector(event_type="response")
    with pytest.raises(ValueError, match="constraint_id must be non-empty"):
        FutureEventConstraint(
            constraint_id="", selector=selector, constraint={"must_not_occur": True}
        )


def test_spec_rejects_duplicate_constraint_ids() -> None:
    selector = FutureEventSelector(event_type="response")
    c1 = FutureEventConstraint(
        constraint_id="dup", selector=selector, constraint={"must_not_occur": True}
    )
    c2 = FutureEventConstraint(
        constraint_id="dup",
        selector=FutureEventSelector(event_type="progression"),
        constraint={"must_not_occur": True},
    )
    with pytest.raises(ValueError, match="duplicate constraint_id 'dup'"):
        ConditioningSpec(
            baseline_seed={},
            pinned_paths=(),
            observed_prefix=None,
            future_event_constraints=(c1, c2),
            free_text_operator_constraints="",
        )


# ---------------------------------------------------------------------------
# Canonicalization round-trip
# ---------------------------------------------------------------------------


def test_canonicalize_handles_date() -> None:
    assert _canonicalize_spec_value(date(2024, 7, 15)) == "2024-07-15"


def test_canonicalize_handles_ledger_event() -> None:
    event = LedgerEvent.model_validate(
        {"date": "2024-04-15", "type": "response", "disease": None}
    )
    canon = _canonicalize_spec_value(event)
    assert isinstance(canon, dict)
    assert canon["date"] == "2024-04-15"
    assert canon["type"] == "response"
    # exclude_none=True: optional `disease`/`patient`/... must NOT appear.
    assert "disease" not in canon
    assert "patient" not in canon


def test_canonicalize_handles_dataclass() -> None:
    selector = FutureEventSelector(event_type="response", ordinal="first")
    canon = _canonicalize_spec_value(selector)
    assert canon == {
        "event_type": "response",
        "ordinal": "first",
        "ordinal_after_cutoff": None,
        "date_range": None,
    }


def test_canonicalize_sorts_mapping_keys() -> None:
    obj = {"z": 1, "a": {"y": 2, "b": 3}}
    canon = _canonicalize_spec_value(obj)
    # Sorted by key at every depth.
    assert list(canon.keys()) == ["a", "z"]
    assert list(canon["a"].keys()) == ["b", "y"]


def test_canonicalize_preserves_list_order() -> None:
    obj = [3, 1, 2]
    assert _canonicalize_spec_value(obj) == [3, 1, 2]


def test_canonical_json_is_byte_stable() -> None:
    spec = _spec_with_two_constraints()
    canon = _canonicalize_spec_value(spec)
    json_text = canonical_json(canon)
    # Re-canonicalizing and re-serializing yields identical bytes.
    assert canonical_json(_canonicalize_spec_value(spec)) == json_text
    # The bytes are valid JSON.
    assert json.loads(json_text)["free_text_operator_constraints"] == ""


# ---------------------------------------------------------------------------
# Hash determinism
# ---------------------------------------------------------------------------


def test_empty_spec_uses_sentinels() -> None:
    spec = _empty_spec()
    assert compute_observed_prefix_hash(spec) == EMPTY_PREFIX_SENTINEL
    assert compute_future_constraint_hash(spec) == EMPTY_CONSTRAINTS_SENTINEL


def test_baseline_hash_matches_legacy_seed_hash() -> None:
    """The generalized baseline hash equals the legacy seed hash byte-for-byte
    **when (a) ``pinned_paths == compute_pinned_paths(seed)`` AND (b) the
    seed has no empty intermediate dicts**.

    Condition (a) holds for every A1 construction path (``lower_conditioning_cell``
    derives ``pinned_paths`` from the seed). Condition (b) holds for any
    seed an operator would write in practice — empty intermediate dicts are
    structurally meaningless. Together they lock the "no behavior change"
    promise for the existing pipeline.

    The divergence under (b) is documented on ``_project_seed_to_paths``:
    an empty dict carries no pinned leaves, so the projection drops it,
    and the subset-pinning contract holds that the empty dict is not part
    of the pinned state.
    """
    seed = {
        "identity_card": {
            "demographics": {"age_at_diagnosis": 62, "sex": "female"},
            "diagnosis": {
                "oncotree_code": "BRCA",
                "diagnosis_date": "2024-01-15",
            },
        }
    }
    spec = ConditioningSpec(
        baseline_seed=seed,
        pinned_paths=compute_pinned_paths(seed),
        observed_prefix=None,
        future_event_constraints=(),
        free_text_operator_constraints="",
    )
    assert compute_baseline_hash(spec) == compute_seed_hash(seed)


def test_baseline_hash_diverges_from_legacy_on_empty_intermediate_dicts() -> None:
    """Documented divergence: the new hash drops empty intermediate dicts;
    the legacy ``compute_seed_hash`` keeps them.

    This is intentional. The subset-pinning contract says only pinned leaves
    contribute to ``baseline_hash``; an empty dict has no leaves, so it can
    never be part of the pinned state. Two seeds differing ONLY by an empty
    intermediate dict therefore produce the same ``compute_baseline_hash``,
    even though they produce different ``compute_seed_hash`` outputs.
    """
    seed_with_empty = {
        "identity_card": {
            "diagnosis": {"oncotree_code": "BRCA"},
            "staging": {"formal_stage": {"components": {}}},
        }
    }
    seed_without_empty = {
        "identity_card": {
            "diagnosis": {"oncotree_code": "BRCA"},
        }
    }
    spec_with = ConditioningSpec(
        baseline_seed=seed_with_empty,
        pinned_paths=compute_pinned_paths(seed_with_empty),
        observed_prefix=None,
        future_event_constraints=(),
        free_text_operator_constraints="",
    )
    spec_without = ConditioningSpec(
        baseline_seed=seed_without_empty,
        pinned_paths=compute_pinned_paths(seed_without_empty),
        observed_prefix=None,
        future_event_constraints=(),
        free_text_operator_constraints="",
    )
    # New hash: same — both seeds carry the same pinned leaves.
    assert compute_baseline_hash(spec_with) == compute_baseline_hash(spec_without)
    # Legacy hash: different — the empty intermediate dict shows up in the
    # canonical form of the seed.
    assert compute_seed_hash(seed_with_empty) != compute_seed_hash(seed_without_empty)


def test_baseline_hash_keys_off_spec_pinned_paths_explicitly() -> None:
    """Two specs sharing ``baseline_seed`` but pinning different subsets
    hash differently.

    This is the architectural contract: ``compute_baseline_hash`` keys off
    ``spec.pinned_paths`` directly, not a re-derivation from the seed. A
    future UI loading a full reference baseline but pinning only a subset
    must produce a distinct ``baseline_hash`` from a spec that pins the
    whole thing.
    """
    seed = {
        "identity_card": {
            "diagnosis": {
                "oncotree_code": "BRCA",
                "diagnosis_date": "2024-01-15",
            }
        }
    }
    derived_paths = compute_pinned_paths(seed)
    full_spec = ConditioningSpec(
        baseline_seed=seed,
        pinned_paths=derived_paths,
        observed_prefix=None,
        future_event_constraints=(),
        free_text_operator_constraints="",
    )
    subset_paths = ("identity_card.diagnosis.oncotree_code",)
    assert set(subset_paths).issubset(set(derived_paths))
    subset_spec = ConditioningSpec(
        baseline_seed=seed,
        pinned_paths=subset_paths,
        observed_prefix=None,
        future_event_constraints=(),
        free_text_operator_constraints="",
    )
    assert compute_baseline_hash(full_spec) != compute_baseline_hash(subset_spec)


def test_baseline_hash_ignores_unpinned_baseline_values() -> None:
    """Subset-pinning contract: changing UNPINNED fields does NOT change
    ``baseline_hash``.

    A future UI loading a full reference baseline but pinning only
    ``oncotree_code`` produces a hash that is insensitive to changes in
    other (unpinned) leaves of the same seed. This is the projection-down
    semantics — ``compute_baseline_hash`` keys off the projection of
    ``baseline_seed`` to ``pinned_paths``, not the full seed.
    """
    seed_a = {
        "identity_card": {
            "diagnosis": {
                "oncotree_code": "BRCA",
                "diagnosis_date": "2024-01-15",
            }
        }
    }
    seed_b = {
        "identity_card": {
            "diagnosis": {
                "oncotree_code": "BRCA",
                "diagnosis_date": "2025-12-31",  # changed but unpinned
            }
        }
    }
    pinned_paths = ("identity_card.diagnosis.oncotree_code",)
    spec_a = ConditioningSpec(
        baseline_seed=seed_a,
        pinned_paths=pinned_paths,
        observed_prefix=None,
        future_event_constraints=(),
        free_text_operator_constraints="",
    )
    spec_b = ConditioningSpec(
        baseline_seed=seed_b,
        pinned_paths=pinned_paths,
        observed_prefix=None,
        future_event_constraints=(),
        free_text_operator_constraints="",
    )
    assert compute_baseline_hash(spec_a) == compute_baseline_hash(spec_b)


def test_baseline_hash_keys_off_pinned_value_changes() -> None:
    """The flip side: changing a PINNED leaf DOES change the hash."""
    pinned_paths = ("identity_card.diagnosis.oncotree_code",)
    spec_a = ConditioningSpec(
        baseline_seed={
            "identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}
        },
        pinned_paths=pinned_paths,
        observed_prefix=None,
        future_event_constraints=(),
        free_text_operator_constraints="",
    )
    spec_b = ConditioningSpec(
        baseline_seed={
            "identity_card": {"diagnosis": {"oncotree_code": "LUAD"}}
        },
        pinned_paths=pinned_paths,
        observed_prefix=None,
        future_event_constraints=(),
        free_text_operator_constraints="",
    )
    assert compute_baseline_hash(spec_a) != compute_baseline_hash(spec_b)


def test_constraint_hash_is_stable_under_input_reorder() -> None:
    h1 = compute_future_constraint_hash(_spec_with_two_constraints(swap=False))
    h2 = compute_future_constraint_hash(_spec_with_two_constraints(swap=True))
    assert h1 == h2


def test_spec_preserves_input_order_in_storage() -> None:
    spec = _spec_with_two_constraints(swap=True)
    assert [c.constraint_id for c in spec.future_event_constraints] == [
        "b_no_local",
        "a_first_treatment",
    ]


def test_observed_prefix_hash_is_deterministic() -> None:
    h1 = compute_observed_prefix_hash(_spec_with_prefix())
    h2 = compute_observed_prefix_hash(_spec_with_prefix())
    assert h1 == h2
    assert h1 != EMPTY_PREFIX_SENTINEL


def test_conditioning_spec_hash_changes_with_each_axis() -> None:
    base = _empty_spec()
    base_hash = compute_conditioning_spec_hash(base)

    with_prefix = _spec_with_prefix()
    with_prefix_hash = compute_conditioning_spec_hash(with_prefix)
    assert with_prefix_hash != base_hash

    with_constraints = _spec_with_two_constraints()
    with_constraints_hash = compute_conditioning_spec_hash(with_constraints)
    assert with_constraints_hash != base_hash
    assert with_constraints_hash != with_prefix_hash

    with_text = ConditioningSpec(
        baseline_seed={},
        pinned_paths=(),
        observed_prefix=None,
        future_event_constraints=(),
        free_text_operator_constraints="hello",
    )
    assert compute_conditioning_spec_hash(with_text) != base_hash


def test_conditioning_spec_hash_is_stable_under_constraint_reorder() -> None:
    h1 = compute_conditioning_spec_hash(_spec_with_two_constraints(swap=False))
    h2 = compute_conditioning_spec_hash(_spec_with_two_constraints(swap=True))
    assert h1 == h2


# ---------------------------------------------------------------------------
# Back-compat lock — the legacy ``compute_conditioning_hash`` is unchanged.
# ---------------------------------------------------------------------------


def test_legacy_compute_conditioning_hash_signature_unchanged() -> None:
    """Gate A1 is purely additive — the legacy helper still works as before.

    The runner uses ``compute_conditioning_hash(seed_hash=...,
    canonical_policy_payload=...)`` and that surface must stay intact until
    A2/A3/B1 migrate call sites.
    """
    seed = {"identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}}
    seed_hash = compute_seed_hash(seed)
    payload = {
        "text": "Treat per oncologist's clinical judgment.",
        "regimen_sequence": [],
        "unconstrained_after_line": None,
        "fixed_first_line_treatment": None,
    }
    h1 = compute_conditioning_hash(
        seed_hash=seed_hash, canonical_policy_payload=payload
    )
    # Deterministic — second call returns the same hex digest.
    h2 = compute_conditioning_hash(
        seed_hash=seed_hash, canonical_policy_payload=payload
    )
    assert h1 == h2
    # Distinct payload → distinct hash (the same property the runner relies on).
    other = compute_conditioning_hash(
        seed_hash=seed_hash,
        canonical_policy_payload={**payload, "text": "different"},
    )
    assert other != h1
