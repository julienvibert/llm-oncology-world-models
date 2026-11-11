"""Conditioning framework for generation cells.

Implements the three-axis conditioning specification used by the generation
runner: baseline pins, fixed/observed prefix events, and future-event
constraints. Also provides:

- ``validate_baseline_seed``: walk a partial seed against ``BaselineObject``'s
  Pydantic model tree, surfacing typo paths and bad enum/type values BEFORE
  any LLM call.
- ``compute_pinned_paths``: derive sorted dotted leaf paths from the seed.
  List-valued seed entries are pinned as whole canonical values, never
  pseudo-indexed (e.g. ``clone_registry`` not ``clone_registry[0].clone_id``).
- ``compute_required_patch_leaves``: walk ``BaselineObject``'s required-field
  tree; return the leaves the model must fill in its ``baseline_patch``.
- ``assert_no_pin_collision``: enforce that the model's emitted
  ``baseline_patch`` does not write any pinned path. Collision rule is
  equality or ancestor-either-way; no suffix matching.
- ``resolve_and_validate``: deep-merge seed + patch; validate the resolved
  baseline against ``BaselineObject``. Raises ``BaselineResolutionError``.
- ``derive_conditioning_label`` / ``compute_seed_hash`` /
  ``compute_conditioning_hash``: deterministic per-cell label + hash. The
  hash includes the full canonical policy payload, not only the 1L pin.

The generalized-conditioning layer adds:

- ``ConditioningSpec`` + ``ObservedPrefix`` + ``FutureEventConstraint`` +
  ``FutureEventSelector``: the generalized three-axis spec.
- ``compute_baseline_hash`` / ``compute_observed_prefix_hash`` /
  ``compute_future_constraint_hash`` / ``compute_conditioning_spec_hash``:
  four-hash composition for the generalized spec. Distinct names from the
  legacy ``compute_conditioning_hash`` so the existing runner path is
  unchanged in A1.

This module is the only module that knows about the patch contract. The
runner ``classify_response`` calls into here; everything else reads the
threaded ``conditioning_label`` / ``conditioning_hash`` strings.
"""

from __future__ import annotations

import hashlib
import json
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
from typing import Any, Literal, Union

from pydantic import BaseModel, TypeAdapter, ValidationError

from synthonccases.oncollm.replay import (
    BaselineObject,
    LedgerEvent,
    MaterializedLedgerEvent,
)


_GENERATION_CONTRACT_REQUIRED_BASELINE_LEAVES: tuple[str, ...] = (
    "identity_card.diagnosis.histology_bucket",
    "identity_card.diagnosis.primary_site_class",
    "identity_card.molecular_profile.primary_driver_bucket",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BaselineSeedError(ValueError):
    """A partial baseline seed is structurally or type-wise invalid.

    Raised at preflight (before any LLM call). The ``path`` field gives the
    dotted leaf path that failed; ``reason`` gives the underlying explanation
    (unknown path / Pydantic ValidationError message)."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


class BaselineResolutionError(ValueError):
    """The deep-merge of seed + patch did not validate as a ``BaselineObject``.

    Raised inside ``classify_response`` after a model attempt is parsed but
    before replay. Maps to ``failure_kind="baseline_validation"``."""


@dataclass(frozen=True, slots=True)
class PinCollision:
    """One concrete collision between a patch path and a pinned path."""

    patch_path: str
    pinned_path: str
    relation: str  # "equal", "patch_ancestor_of_pinned", "pinned_ancestor_of_patch"


# ---------------------------------------------------------------------------
# Canonical JSON + hashing
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Deterministic JSON for hashing. Sorted keys, no whitespace, lists kept
    in source order (no domain-specific sorting)."""
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pinned-paths + required-patch-leaves
# ---------------------------------------------------------------------------


def compute_pinned_paths(seed: Mapping[str, Any]) -> tuple[str, ...]:
    """Walk a nested partial seed; return sorted dotted leaf paths.

    Leaf semantics: any value that is NOT a dict is a leaf. Lists are leaves
    (pinned as whole canonical JSON values, never per-index). Empty dicts
    contribute no paths.
    """
    paths: list[str] = []
    _collect_leaf_paths(seed, prefix="", out=paths)
    return tuple(sorted(paths))


def _collect_leaf_paths(node: Any, *, prefix: str, out: list[str]) -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                if value:
                    _collect_leaf_paths(value, prefix=child_prefix, out=out)
                # else: empty dict contributes no paths
            else:
                out.append(child_prefix)


def compute_seed_hash(seed: Mapping[str, Any]) -> str:
    """SHA-256 of {pinned_paths (sorted) + pinned_values (canonical seed)}.

    Stable identifier of a condition's seed. Survives YAML key reordering."""
    pinned = compute_pinned_paths(seed)
    payload = {"pinned_paths": list(pinned), "pinned_values": _canonicalize(seed)}
    return _sha256_hex(canonical_json(payload))


def _canonicalize(obj: Any) -> Any:
    """Return obj with dicts recursively sorted by key (canonical form).

    Lists keep source order; non-dict leaves are returned as-is. Used for
    hashing; reading the canonical JSON of this output is byte-stable."""
    if isinstance(obj, Mapping):
        return {k: _canonicalize(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_canonicalize(v) for v in obj]
    return obj


def compute_required_patch_leaves(seed: Mapping[str, Any]) -> tuple[str, ...]:
    """Required leaves of ``BaselineObject`` not present in the seed.

    Used to populate ``{{REQUIRED_PATCH_LEAVES}}`` in the prompt. List-valued
    required fields appear as whole leaves (e.g. ``clone_registry``), not
    pseudo-indexed."""
    required = sorted(
        {
            *_required_leaves(BaselineObject, prefix=""),
            *_GENERATION_CONTRACT_REQUIRED_BASELINE_LEAVES,
        }
    )
    pinned = set(compute_pinned_paths(seed))
    pinned_ancestors: set[str] = set()
    for p in pinned:
        parts = p.split(".")
        for i in range(1, len(parts) + 1):
            pinned_ancestors.add(".".join(parts[:i]))
    return tuple(sorted(p for p in required if p not in pinned_ancestors))


def _required_leaves(model_cls: type[BaseModel], *, prefix: str) -> list[str]:
    """Walk a Pydantic model; return required leaves as dotted paths.

    A field is required if it has no default. A field whose annotation
    resolves to a BaseModel subclass is recursed into; otherwise it is
    emitted as a leaf (including lists, dicts, enums, scalars)."""
    out: list[str] = []
    for name, field in model_cls.model_fields.items():
        if not field.is_required():
            continue
        dotted = f"{prefix}.{name}" if prefix else name
        sub_model = _annotation_model(field.annotation)
        if sub_model is not None:
            out.extend(_required_leaves(sub_model, prefix=dotted))
        else:
            out.append(dotted)
    return out


def _annotation_model(annotation: Any) -> type[BaseModel] | None:
    """If ``annotation`` resolves to a single BaseModel subclass (possibly
    wrapped in Optional / Union), return it; otherwise return None.

    Used by both ``_required_leaves`` (to decide whether to recurse) and
    ``validate_baseline_seed`` (to walk into nested seeds). list[X] /
    dict[X, Y] / tuple[X, ...] annotations explicitly return None — list-
    valued fields are leaves, not recursion points (per the rev3-feedback
    list-semantics decision)."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = typing.get_origin(annotation)
    if origin is None:
        return None
    # Only unwrap Union / Optional. Container origins (list, dict, tuple,
    # set, ...) are leaves.
    if origin is not Union and origin is not types.UnionType:
        return None
    args = [
        a for a in typing.get_args(annotation)
        if a is not type(None)
    ]
    if len(args) != 1:
        return None
    inner = args[0]
    if isinstance(inner, type) and issubclass(inner, BaseModel):
        return inner
    return None


# ---------------------------------------------------------------------------
# Preflight seed validation
# ---------------------------------------------------------------------------


def validate_baseline_seed(seed: Mapping[str, Any]) -> None:
    """Validate a partial baseline seed against ``BaselineObject`` directly.

    Walks ``BaselineObject.model_fields`` to verify every dotted path in the
    seed exists. Validates leaf values with ``TypeAdapter``. Raises
    ``BaselineSeedError`` at the first failure with the dotted path that
    failed.

    No ``GOLD_TEMPLATE`` merge — purely structural. A seed that pins one
    field on demographics validates without inventing a sex / age. A seed
    that pins ``identity_card.diagnosis.oncotree`` (typo, missing ``_code``)
    fails fast with that path.
    """
    if not isinstance(seed, Mapping):
        raise BaselineSeedError(
            path="", reason=f"baseline_seed must be a mapping; got {type(seed).__name__}"
        )
    _validate_node(seed, model_cls=BaselineObject, prefix="")


def _validate_node(node: Any, *, model_cls: type[BaseModel], prefix: str) -> None:
    if not isinstance(node, Mapping):
        # Reached a non-dict at a position where we still expected a model.
        # That's a structural mismatch.
        raise BaselineSeedError(
            path=prefix or "<root>",
            reason=f"expected a sub-mapping for model {model_cls.__name__}; "
                   f"got {type(node).__name__}",
        )
    for key, value in node.items():
        field = model_cls.model_fields.get(key)
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if field is None:
            raise BaselineSeedError(
                path=dotted,
                reason=f"{dotted!r} is not a known path on BaselineObject",
            )
        sub_model = _annotation_model(field.annotation)
        if isinstance(value, Mapping) and sub_model is not None:
            _validate_node(value, model_cls=sub_model, prefix=dotted)
            continue
        # Leaf — validate value against annotation.
        try:
            TypeAdapter(field.annotation).validate_python(value)
        except ValidationError as exc:
            raise BaselineSeedError(path=dotted, reason=str(exc)) from exc


# ---------------------------------------------------------------------------
# Patch-pin collision rule
# ---------------------------------------------------------------------------


def assert_no_pin_collision(
    patch: Mapping[str, Any],
    pinned_paths: tuple[str, ...],
) -> list[PinCollision]:
    """Return every collision between leaf paths of ``patch`` and ``pinned_paths``.

    Collision = equality OR ancestor-either-way. No suffix matching.
    Disjoint sibling paths do not collide.

    Empty list = no collisions. The caller decides what to do (the runner
    converts a non-empty list into ``failure_kind="conditioning_audit"``).
    """
    if not pinned_paths:
        return []
    patch_paths: list[str] = []
    _collect_leaf_paths(patch, prefix="", out=patch_paths)
    collisions: list[PinCollision] = []
    pinned_set = set(pinned_paths)
    for patch_path in patch_paths:
        for pinned in pinned_set:
            relation = _path_relation(patch_path, pinned)
            if relation is not None:
                collisions.append(
                    PinCollision(
                        patch_path=patch_path,
                        pinned_path=pinned,
                        relation=relation,
                    )
                )
    return collisions


def _path_relation(patch_path: str, pinned_path: str) -> str | None:
    """Return the relation between two dotted paths.

    Returns ``"equal"`` / ``"patch_ancestor_of_pinned"`` /
    ``"pinned_ancestor_of_patch"`` when they collide, or ``None`` when
    disjoint. Ancestor check is on dotted-segment prefix, not raw string
    prefix — ``identity_card.diagnosis`` is NOT an ancestor of
    ``identity_card.diagnosis_label``.
    """
    if patch_path == pinned_path:
        return "equal"
    patch_parts = patch_path.split(".")
    pinned_parts = pinned_path.split(".")
    if len(patch_parts) < len(pinned_parts) and pinned_parts[: len(patch_parts)] == patch_parts:
        return "patch_ancestor_of_pinned"
    if len(pinned_parts) < len(patch_parts) and patch_parts[: len(pinned_parts)] == pinned_parts:
        return "pinned_ancestor_of_patch"
    return None


# ---------------------------------------------------------------------------
# Resolve + validate (deep merge → BaselineObject.model_validate)
# ---------------------------------------------------------------------------


def deep_merge(seed: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive merge of seed and patch.

    Both must be dicts. Where both are dicts at the same key, recurse. Where
    only one has the key, that value wins. Where both have a non-dict value
    at the same key, raise ``BaselineResolutionError`` — pin-collision check
    should have already prevented this; the assertion catches programmer
    error (e.g. caller skipped the collision check).
    """
    out: dict[str, Any] = {}
    keys = set(seed) | set(patch)
    for key in keys:
        if key in seed and key in patch:
            seed_val, patch_val = seed[key], patch[key]
            if isinstance(seed_val, Mapping) and isinstance(patch_val, Mapping):
                out[key] = deep_merge(seed_val, patch_val)
            elif isinstance(seed_val, Mapping) or isinstance(patch_val, Mapping):
                raise BaselineResolutionError(
                    f"deep_merge type clash at {key!r}: one side mapping, other not"
                )
            else:
                raise BaselineResolutionError(
                    f"deep_merge leaf collision at {key!r}: "
                    "pin-collision check should have caught this"
                )
        elif key in seed:
            out[key] = seed[key]
        else:
            out[key] = patch[key]
    return out


def resolve_and_validate(
    seed: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> BaselineObject:
    """Deep-merge seed + patch; validate as ``BaselineObject``.

    Raises ``BaselineResolutionError`` on either failure. The merge step
    can raise on programmer-caught leaf collisions; the validate step
    raises on schema violations (missing required leaves, bad enum values,
    etc.).
    """
    try:
        merged = deep_merge(seed, patch)
    except BaselineResolutionError:
        raise
    try:
        return BaselineObject.model_validate(merged)
    except ValidationError as exc:
        raise BaselineResolutionError(
            f"resolved baseline failed BaselineObject validation: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Condition label + condition hash (cell-level)
# ---------------------------------------------------------------------------


_TUMOR_ONLY_PINNED_SHAPES = {
    ("identity_card.diagnosis.oncotree_code",),
    (
        "identity_card.diagnosis.diagnosis_label",
        "identity_card.diagnosis.oncotree_code",
    ),
}


def derive_conditioning_label(
    *,
    baseline_seed: Mapping[str, Any],
    pinned_paths: tuple[str, ...],
    fixed_first_line_regimen_label: str | None,
    label_ignored_pinned_paths: tuple[str, ...] = (),
) -> str:
    """Human-readable label derived from the seed shape + 1L policy.

    Rules:
    - empty seed → ``cond:spontaneous``
    - ``BaselineObject.model_validate(seed)`` succeeds → ``cond:full_baseline``
      (a complete valid baseline; required-leaf counting is too brittle)
    - only ``identity_card.diagnosis.oncotree_code`` pinned, optionally with
      ``identity_card.diagnosis.diagnosis_label`` for human-readable anchor
      clarity, after dropping bookkeeping-only ``label_ignored_pinned_paths`` →
      ``cond:tumor_only``
    - otherwise → ``cond:partial_baseline_<N>`` (N = visible pinned-leaf count)

    When ``fixed_first_line_regimen_label`` is supplied, append
    ``+1L=<regimen_label>``. The hash captures the full policy regardless;
    the label only carries the 1L for legibility.
    """
    ignored = set(label_ignored_pinned_paths)
    visible_pinned_paths = tuple(p for p in pinned_paths if p not in ignored)
    if not baseline_seed or not visible_pinned_paths:
        core = "cond:spontaneous"
    else:
        try:
            BaselineObject.model_validate(baseline_seed)
            core = "cond:full_baseline"
        except ValidationError:
            if visible_pinned_paths in _TUMOR_ONLY_PINNED_SHAPES:
                core = "cond:tumor_only"
            else:
                core = f"cond:partial_baseline_{len(visible_pinned_paths)}"
    if fixed_first_line_regimen_label:
        return f"{core}+1L={fixed_first_line_regimen_label}"
    return core


def compute_conditioning_hash(
    *,
    seed_hash: str,
    canonical_policy_payload: Mapping[str, Any],
) -> str:
    """SHA-256 over the seed-hash plus the full canonical policy payload.

    The caller is responsible for assembling ``canonical_policy_payload`` —
    typically ``{"text": ..., "regimen_sequence": ..., "unconstrained_after_line":
    ..., "fixed_first_line_treatment": ...}``. Including the full policy means
    two policies that share the same 1L but diverge later hash differently.
    """
    payload = {
        "seed_hash": seed_hash,
        "policy": _canonicalize(dict(canonical_policy_payload)),
    }
    return _sha256_hex(canonical_json(payload))


# ---------------------------------------------------------------------------
# Gate A1 — generalized ConditioningSpec (three orthogonal axes)
# ---------------------------------------------------------------------------
#
# The generalized spec replaces the implicit (baseline, +1L) pairing with one
# object varying along three axes:
#
#   - baseline_seed (with derived pinned_paths)
#   - observed_prefix (events up to cutoff_date) — A3 enables population
#   - future_event_constraints (selector + constraint pairs) — A2 generalizes
#       the audit; A1 only stores the data
#
# In A1 the legacy ``ConditionSpec`` + ``PolicySpec`` + ``ConditioningCellSpec``
# in ``poc_runner`` continue to drive the live pipeline. The new dataclasses
# and four hash helpers are additive — no existing function is renamed or
# rewired. A2/A3/B1 migrate call sites.


OrdinalLiteral = Literal["first", "second", "third", "last"]
PrefixOrigin = Literal["observed_prefix"]


EMPTY_PREFIX_SENTINEL = "EMPTY_PREFIX"
EMPTY_CONSTRAINTS_SENTINEL = "EMPTY_CONSTRAINTS"


_ALLOWED_ORDINALS: frozenset[str] = frozenset({"first", "second", "third", "last"})


@dataclass(frozen=True, slots=True)
class FutureEventSelector:
    """Selects events in a materialized ledger for constraint evaluation.

    ``ordinal`` resolves over the entire materialized ledger of events of
    matching ``event_type``. ``ordinal_after_cutoff`` resolves only over
    events strictly after the spec's observed-prefix ``cutoff_date`` (only
    meaningful when an ``ObservedPrefix`` is present); A2 enforces this
    semantic. Setting both is rejected at construction.

    Runtime validation (``__post_init__``) enforces:
    - ``event_type`` is non-empty (PEP 484 ``Literal`` is not runtime-checked).
    - ``ordinal`` and ``ordinal_after_cutoff`` take values from
      ``_ALLOWED_ORDINALS`` only.
    - ``ordinal`` and ``ordinal_after_cutoff`` are mutually exclusive.
    """

    event_type: str
    ordinal: OrdinalLiteral | None = None
    ordinal_after_cutoff: OrdinalLiteral | None = None
    date_range: tuple[date, date] | None = None
    #: Narrows a ``local_intervention`` selection to particular ``intent`` values. Added
    #: because "best supportive care" was expressible only as "no systemic therapy": a
    #: generated arm could still receive tumour-directed surgery or radiotherapy and change
    #: its own outcome, which makes it a poor floor to compare an active treatment against.
    #: Forbidding local intervention wholesale is not the fix either, since palliative
    #: radiotherapy IS supportive care. The intent enum already distinguishes them; this lets
    #: a constraint say so.
    intent_is_one_of: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.event_type or not str(self.event_type).strip():
            raise ValueError("FutureEventSelector.event_type must be non-empty")
        if self.intent_is_one_of is not None and not self.intent_is_one_of:
            raise ValueError(
                "FutureEventSelector.intent_is_one_of must be a non-empty tuple when set; "
                "an empty filter would select nothing and silently satisfy any constraint"
            )
        if self.ordinal is not None and self.ordinal not in _ALLOWED_ORDINALS:
            raise ValueError(
                f"FutureEventSelector.ordinal: {self.ordinal!r} is not one of "
                f"{sorted(_ALLOWED_ORDINALS)}"
            )
        if (
            self.ordinal_after_cutoff is not None
            and self.ordinal_after_cutoff not in _ALLOWED_ORDINALS
        ):
            raise ValueError(
                f"FutureEventSelector.ordinal_after_cutoff: "
                f"{self.ordinal_after_cutoff!r} is not one of "
                f"{sorted(_ALLOWED_ORDINALS)}"
            )
        if self.ordinal is not None and self.ordinal_after_cutoff is not None:
            raise ValueError(
                "FutureEventSelector: ordinal and ordinal_after_cutoff are "
                "mutually exclusive"
            )


@dataclass(frozen=True, slots=True)
class FutureEventConstraint:
    """One constraint on the model-generated future ledger.

    ``constraint`` is a frozen mapping with constraint-vocabulary keys
    (``regimen_matches`` / ``event_type_is_one_of`` / ``must_occur_within_days``
    / ``must_not_occur``). A2's ``future_constraints_audit`` resolves and
    evaluates each constraint against the materialized ledger. A1 only stores
    the data; no evaluation here.
    """

    constraint_id: str
    selector: FutureEventSelector
    constraint: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.constraint_id or not str(self.constraint_id).strip():
            raise ValueError("FutureEventConstraint.constraint_id must be non-empty")


@dataclass(frozen=True, slots=True)
class ObservedPrefix:
    """The fixed-events axis of a ``ConditioningSpec``.

    ``events`` is an ordered tuple of ``LedgerEvent`` (Pydantic) values.
    Chronology + ``baseline.diagnosis_date <= event.date <= cutoff_date``
    invariants are enforced in A3's prefix preflight, not at A1 construction.
    """

    events: tuple[LedgerEvent, ...]
    cutoff_date: date
    origin: PrefixOrigin = "observed_prefix"


@dataclass(frozen=True, slots=True)
class ConditioningSpec:
    """Generalized conditioning along three orthogonal axes.

    Coexists with the legacy ``ConditionSpec`` + ``PolicySpec`` +
    ``ConditioningCellSpec`` in ``poc_runner`` during A1; subsequent commits
    (A2/A3/B1) generalize the audit, replay, runner, and derivation around
    this object.

    Contract — ``baseline_seed`` vs ``pinned_paths``:

    The ``pinned_paths`` tuple is the **authoritative** statement of which
    leaves of ``baseline_seed`` are pinned. In all A1 construction paths
    (``lower_conditioning_cell`` and its helpers), ``pinned_paths`` is
    populated via ``compute_pinned_paths(baseline_seed)``, so the two are
    consistent. A future use case may want to load a full reference baseline
    but only pin a subset of it — in that case ``pinned_paths`` would be a
    strict subset of what ``compute_pinned_paths(baseline_seed)`` returns.
    The hash helpers therefore key off ``spec.pinned_paths`` directly, not
    a re-derivation from the seed.

    Validation here is intentionally minimal — duplicate ``constraint_id``
    raises. Deeper semantic checks (selector/observed-prefix coherence,
    chronology, replay feasibility) live in A2/A3.
    """

    baseline_seed: Mapping[str, Any]
    pinned_paths: tuple[str, ...]
    observed_prefix: ObservedPrefix | None
    future_event_constraints: tuple[FutureEventConstraint, ...]
    free_text_operator_constraints: str

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for c in self.future_event_constraints:
            if c.constraint_id in seen:
                raise ValueError(
                    f"ConditioningSpec: duplicate constraint_id "
                    f"{c.constraint_id!r} in future_event_constraints"
                )
            seen.add(c.constraint_id)


# ---------------------------------------------------------------------------
# Canonicalization helper for the generalized spec.
# ---------------------------------------------------------------------------


def _canonicalize_spec_value(obj: Any) -> Any:
    """Canonical form for hashing ``ConditioningSpec`` components.

    Extends ``_canonicalize`` (which only handles plain dict/list/scalar) to
    cover the spec-specific types:

    - ``date`` → ISO string
    - ``LedgerEvent`` (Pydantic) → ``model_dump(mode="json", exclude_none=True)``
    - dataclasses → stable dict via ``asdict``
    - ``tuple`` → list (JSON has no tuple; source order preserved)
    - mappings → dict with keys sorted

    The output is intended to be passed directly to ``canonical_json`` to get
    byte-stable hashing input. ``default=str`` is NOT used — types we do not
    explicitly handle fall through to ``canonical_json`` which will raise
    rather than silently stringify.
    """
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, LedgerEvent):
        return _canonicalize_spec_value(
            obj.model_dump(mode="json", exclude_none=True)
        )
    if is_dataclass(obj) and not isinstance(obj, type):
        as_dict = asdict(obj)
        # `intent_is_one_of` was added to FutureEventSelector after runs had already been
        # hashed. Emitting it as an explicit null would change the constraint hash of every
        # config that never uses it, and silently break the reproducibility of prior runs.
        # Omitting it when unset keeps those hashes byte-identical; the hash moves only for
        # a config that actually uses the filter, which is correct.
        if isinstance(obj, FutureEventSelector) and as_dict.get("intent_is_one_of") is None:
            as_dict.pop("intent_is_one_of", None)
        return _canonicalize_spec_value(as_dict)
    if isinstance(obj, Mapping):
        return {k: _canonicalize_spec_value(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize_spec_value(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Four hashes for the generalized spec.
# ---------------------------------------------------------------------------


def _project_seed_to_paths(
    seed: Mapping[str, Any], paths: tuple[str, ...]
) -> dict[str, Any]:
    """Project ``seed`` down to the subset of leaves named by ``paths``.

    Each path is a dotted-segment string. The output is a nested dict
    containing only the leaves named in ``paths``; intermediate dicts are
    created as needed. Paths that walk through non-dict values, or land on
    leaves the seed does not actually have, are silently skipped.

    Lists at a pinned leaf are taken as a whole (matching the existing
    ``compute_pinned_paths`` list-as-leaf semantics).

    **Empty intermediate dicts are dropped.** ``compute_pinned_paths`` emits
    no path for an empty dict (it has no leaves), so the projection has no
    way to recreate such a dict. This means the projection equals the seed
    byte-for-byte iff the seed has no empty intermediate dicts. When that
    holds, ``compute_baseline_hash`` produces the same hex digest as the
    legacy ``compute_seed_hash``; when an operator's seed carries an empty
    intermediate dict, the two helpers diverge (and the subset-pinning
    contract says the empty dict is not part of the pinned state anyway).
    """
    out: dict[str, Any] = {}
    for path in paths:
        parts = path.split(".")
        cursor: Any = seed
        for part in parts[:-1]:
            if not isinstance(cursor, Mapping) or part not in cursor:
                cursor = None
                break
            cursor = cursor[part]
        if not isinstance(cursor, Mapping) or parts[-1] not in cursor:
            continue
        leaf = cursor[parts[-1]]
        write_cursor: dict[str, Any] = out
        for part in parts[:-1]:
            existing = write_cursor.get(part)
            if not isinstance(existing, dict):
                existing = {}
                write_cursor[part] = existing
            write_cursor = existing
        write_cursor[parts[-1]] = leaf
    return out


def compute_baseline_hash(spec: ConditioningSpec) -> str:
    """SHA-256 over ``(sorted spec.pinned_paths, pinned-subset of spec.baseline_seed)``.

    Both axes of the hash key off ``spec.pinned_paths`` directly:

    - ``pinned_paths`` is hashed in sorted form for byte-stability under YAML
      key reorder.
    - ``pinned_values`` is the projection of ``spec.baseline_seed`` **down to
      those paths only** — unpinned fields of ``spec.baseline_seed`` do NOT
      contribute. A future UI loading a full reference baseline but pinning
      only ``oncotree_code`` therefore produces a hash insensitive to
      changes in, say, ``diagnosis_date`` on the reference baseline.

    When the legacy invariant ``spec.pinned_paths ==
    compute_pinned_paths(spec.baseline_seed)`` holds — true for every A1
    construction path — the projection equals ``spec.baseline_seed`` itself
    and the bytes match ``compute_seed_hash(spec.baseline_seed)``.
    """
    projected = _project_seed_to_paths(spec.baseline_seed, spec.pinned_paths)
    payload = {
        "pinned_paths": sorted(spec.pinned_paths),
        "pinned_values": _canonicalize_spec_value(projected),
    }
    return _sha256_hex(canonical_json(payload))


def compute_observed_prefix_hash(spec: ConditioningSpec) -> str:
    """SHA-256 over (canonical events, cutoff_date, origin).

    Returns the sentinel ``EMPTY_PREFIX`` when ``spec.observed_prefix is
    None``, so the composite ``conditioning_spec_hash`` is stable when the
    prefix axis is unused.
    """
    if spec.observed_prefix is None:
        return EMPTY_PREFIX_SENTINEL
    payload = {
        "events": _canonicalize_spec_value(spec.observed_prefix.events),
        "cutoff_date": spec.observed_prefix.cutoff_date.isoformat(),
        "origin": spec.observed_prefix.origin,
    }
    return _sha256_hex(canonical_json(payload))


def compute_future_constraint_hash(spec: ConditioningSpec) -> str:
    """SHA-256 over canonical future_event_constraints, sorted by constraint_id.

    Returns the sentinel ``EMPTY_CONSTRAINTS`` when the constraint tuple is
    empty. The stored tuple on the spec preserves *input* order (so the
    runner can present constraints to the model in the operator's intended
    order); the hash sorts by ``constraint_id`` so reordering the same set
    of constraints in YAML does not change the hash.
    """
    if not spec.future_event_constraints:
        return EMPTY_CONSTRAINTS_SENTINEL
    sorted_constraints = sorted(
        spec.future_event_constraints, key=lambda c: c.constraint_id
    )
    payload = [_canonicalize_spec_value(c) for c in sorted_constraints]
    return _sha256_hex(canonical_json(payload))


def compute_conditioning_spec_hash(spec: ConditioningSpec) -> str:
    """Composite SHA-256 over the three sub-hashes + free-text constraints.

    Distinct name from the legacy ``compute_conditioning_hash(seed_hash=...,
    canonical_policy_payload=...)`` which is still wired into today's runner.
    A2/A3/B1 migrate the runner path to ``ConditioningSpec`` + this helper;
    A1 leaves both functions in place so the existing pipeline is unchanged.
    """
    payload = {
        "baseline_hash": compute_baseline_hash(spec),
        "observed_prefix_hash": compute_observed_prefix_hash(spec),
        "future_constraint_hash": compute_future_constraint_hash(spec),
        "free_text_operator_constraints": spec.free_text_operator_constraints,
    }
    return _sha256_hex(canonical_json(payload))


# ---------------------------------------------------------------------------
# Gate B1b — spec ↔ JSON-dict round-trip.
# ---------------------------------------------------------------------------
#
# The runner persists ``conditioning_spec.json`` per attempt (accepted +
# failed). The derivation layer reads that file, reconstructs the spec, and
# recomputes the four hashes plus re-runs ``future_constraints_audit`` to
# recover ``constraint_match_ids`` per event. These two helpers are the
# serialization boundary.
#
# Round-trip invariant: for any spec, every hash helper
# (``compute_baseline_hash`` / ``compute_observed_prefix_hash`` /
# ``compute_future_constraint_hash`` / ``compute_conditioning_spec_hash``)
# produces the same digest on the original and on the reconstructed spec.


def spec_to_json_dict(spec: ConditioningSpec) -> dict[str, Any]:
    """JSON-safe dict form of ``spec`` for round-tripping through disk.

    Pairs with ``spec_from_json_dict``. The output is pure dict / list /
    scalar / string (no ``date``, no Pydantic models, no dataclasses, no
    tuples) and uses the same sorted-key ordering as
    ``_canonicalize_spec_value`` so the file bytes are stable across
    emissions of the same spec.
    """
    canonical = _canonicalize_spec_value(spec)
    if not isinstance(canonical, dict):
        raise TypeError(
            f"spec_to_json_dict: expected dict, got "
            f"{type(canonical).__name__}"
        )
    return canonical


def _selector_from_json_dict(data: Mapping[str, Any]) -> FutureEventSelector:
    date_range_raw = data.get("date_range")
    if date_range_raw is None:
        date_range: tuple[date, date] | None = None
    else:
        date_range = (
            date.fromisoformat(date_range_raw[0]),
            date.fromisoformat(date_range_raw[1]),
        )
    return FutureEventSelector(
        event_type=data["event_type"],
        ordinal=data.get("ordinal"),
        ordinal_after_cutoff=data.get("ordinal_after_cutoff"),
        date_range=date_range,
    )


def _constraint_from_json_dict(
    data: Mapping[str, Any],
) -> FutureEventConstraint:
    # Plain ``dict`` matches the original construction path
    # (``test_conditioning_spec_data.py`` builds constraints with plain dicts).
    # ``MappingProxyType`` is not pickleable, which breaks ``asdict`` during
    # downstream canonicalization.
    return FutureEventConstraint(
        constraint_id=data["constraint_id"],
        selector=_selector_from_json_dict(data["selector"]),
        constraint=dict(data["constraint"]),
    )


def spec_from_json_dict(data: Mapping[str, Any]) -> ConditioningSpec:
    """Reconstruct a ``ConditioningSpec`` from the dict produced by
    ``spec_to_json_dict``.

    Reconstructs ``LedgerEvent`` via ``model_validate``, parses ISO date
    strings to ``date``, rebuilds frozen dataclasses, and restores
    ``tuple`` from JSON-array lists. The reconstructed
    ``FutureEventConstraint.constraint`` is a plain ``dict`` (matches the
    original construction path; ``MappingProxyType`` is not pickleable,
    which would break downstream ``asdict`` canonicalization).
    """
    observed_prefix_data = data.get("observed_prefix")
    if observed_prefix_data is None:
        observed_prefix: ObservedPrefix | None = None
    else:
        events = tuple(
            LedgerEvent.model_validate(ev)
            for ev in observed_prefix_data["events"]
        )
        observed_prefix = ObservedPrefix(
            events=events,
            cutoff_date=date.fromisoformat(observed_prefix_data["cutoff_date"]),
            origin=observed_prefix_data.get("origin", "observed_prefix"),
        )

    constraints = tuple(
        _constraint_from_json_dict(c)
        for c in data.get("future_event_constraints", ())
    )

    return ConditioningSpec(
        baseline_seed=dict(data.get("baseline_seed", {})),
        pinned_paths=tuple(data.get("pinned_paths", ())),
        observed_prefix=observed_prefix,
        future_event_constraints=constraints,
        free_text_operator_constraints=data.get(
            "free_text_operator_constraints", ""
        ),
    )


# ---------------------------------------------------------------------------
# Gate A2 — generalized future-constraints audit (post-replay).
# ---------------------------------------------------------------------------
#
# Operates on the **materialized** ledger (post-replay), not the raw emitted
# ``LedgerEvent`` stream. The materialized form carries normalized regimens
# (label + components), derived ``line``, and ``event_id``, so alias matching
# and ordinal resolution are simpler than against the raw stream.
#
# This function is purely additive — the legacy ``_check_first_line_match`` in
# ``poc_runner`` is unchanged. A3/B1 will rewire the runner to call this gate
# instead, at which point the legacy helper becomes a thin wrapper.


_RECOGNIZED_CONSTRAINT_KEYS: frozenset[str] = frozenset(
    {
        "regimen_matches",
        "event_type_is_one_of",
        "must_occur_within_days",
        "must_not_occur",
    }
)

_ORDINAL_INDEX: dict[str, int] = {"first": 0, "second": 1, "third": 2}


@dataclass(frozen=True, slots=True)
class ConstraintMatch:
    """Result of evaluating one ``FutureEventConstraint``.

    ``matched_event_indices`` and ``matched_event_ids`` are aligned tuples of
    zero-based positions in the *original* ``materialized_ledger`` (not after
    filtering or sorting) and the corresponding ``event_id``s. ``failure_reason``
    is populated iff ``passed is False``.
    """

    constraint_id: str
    passed: bool
    matched_event_indices: tuple[int, ...]
    matched_event_ids: tuple[str, ...]
    failure_reason: str | None


def future_constraints_audit(
    spec: ConditioningSpec,
    materialized_ledger: Sequence[MaterializedLedgerEvent],
) -> list[ConstraintMatch]:
    """Evaluate every ``spec.future_event_constraints`` against the materialized ledger.

    Returns one ``ConstraintMatch`` per constraint, in input order. The caller
    decides what to do with failures (the runner converts ``passed=False`` into
    ``failure_kind="future_constraints_audit"`` in A3/B1).

    Raises ``ValueError`` for incoherent spec/constraint configurations — these
    are programmer/operator errors, not model failures, and propagate rather
    than producing a soft ``passed=False``:

    - ``ordinal_after_cutoff`` selector without an ``observed_prefix``.
    - ``must_occur_within_days.reference='cutoff'`` without ``observed_prefix``.
    - ``must_occur_within_days.reference='prior_event'`` (deferred in A2).
    - Constraint containing ``must_not_occur: true`` plus any other semantic key
      (exclusive-with rule).
    - Unknown keys in the constraint mapping (typo guard).
    """
    return [
        _evaluate_constraint(constraint, spec, materialized_ledger)
        for constraint in spec.future_event_constraints
    ]


def _evaluate_constraint(
    constraint: FutureEventConstraint,
    spec: ConditioningSpec,
    materialized_ledger: Sequence[MaterializedLedgerEvent],
) -> ConstraintMatch:
    mapping = constraint.constraint
    unknown_keys = set(mapping.keys()) - _RECOGNIZED_CONSTRAINT_KEYS
    if unknown_keys:
        raise ValueError(
            f"FutureEventConstraint {constraint.constraint_id!r}: unknown "
            f"constraint key(s) {sorted(unknown_keys)!r}. Recognized keys: "
            f"{sorted(_RECOGNIZED_CONSTRAINT_KEYS)!r}"
        )

    # ``must_not_occur`` is binary by spec: only literal Python ``True`` is
    # accepted. Reject ``False`` / ``None`` / non-bool truthy values like
    # ``"true"`` or ``1`` so the DSL cannot be silently misread as a no-op
    # constraint or accidentally activated by a string. The check must fire
    # before the exclusivity test so an invalid pairing like
    # ``{"must_not_occur": False, "regimen_matches": [...]}`` is rejected at
    # the value-validation step, not as exclusivity.
    is_must_not_occur = False
    if "must_not_occur" in mapping:
        value = mapping["must_not_occur"]
        if value is not True:
            raise ValueError(
                f"FutureEventConstraint {constraint.constraint_id!r}: "
                f"must_not_occur must be exactly True (bool); got "
                f"{value!r}"
            )
        is_must_not_occur = True
        other_keys = set(mapping.keys()) - {"must_not_occur"}
        if other_keys:
            raise ValueError(
                f"FutureEventConstraint {constraint.constraint_id!r}: "
                f"must_not_occur=true is exclusive with other constraint "
                f"keys; also found {sorted(other_keys)!r}"
            )

    matches = _resolve_selector(constraint.selector, spec, materialized_ledger)
    matched_indices = tuple(idx for idx, _ in matches)
    matched_ids = tuple(ev.event_id for _, ev in matches)

    if is_must_not_occur:
        if matches:
            return ConstraintMatch(
                constraint_id=constraint.constraint_id,
                passed=False,
                matched_event_indices=matched_indices,
                matched_event_ids=matched_ids,
                failure_reason=(
                    f"must_not_occur=true but selector resolved to "
                    f"{len(matches)} event(s)"
                ),
            )
        return ConstraintMatch(
            constraint_id=constraint.constraint_id,
            passed=True,
            matched_event_indices=(),
            matched_event_ids=(),
            failure_reason=None,
        )

    if not matches:
        return ConstraintMatch(
            constraint_id=constraint.constraint_id,
            passed=False,
            matched_event_indices=(),
            matched_event_ids=(),
            failure_reason="selector resolved to no events",
        )

    for key, value in mapping.items():
        if key == "must_not_occur":
            continue  # handled exclusively above
        reason = _check_constraint_key(key, value, matches, spec)
        if reason is not None:
            return ConstraintMatch(
                constraint_id=constraint.constraint_id,
                passed=False,
                matched_event_indices=matched_indices,
                matched_event_ids=matched_ids,
                failure_reason=reason,
            )

    return ConstraintMatch(
        constraint_id=constraint.constraint_id,
        passed=True,
        matched_event_indices=matched_indices,
        matched_event_ids=matched_ids,
        failure_reason=None,
    )


def _resolve_selector(
    selector: FutureEventSelector,
    spec: ConditioningSpec,
    materialized_ledger: Sequence[MaterializedLedgerEvent],
) -> list[tuple[int, MaterializedLedgerEvent]]:
    """Apply a ``FutureEventSelector`` to the materialized ledger.

    Returns matched ``(original_index, event)`` tuples sorted chronologically
    by ``(event.date, original_index)`` so same-day events preserve ledger
    order. Out-of-range ordinals return an empty list — the caller's
    constraint-key evaluation then resolves per inversion semantics.
    """
    # Step 1: filter by event_type, preserving original positions.
    filtered: list[tuple[int, MaterializedLedgerEvent]] = [
        (idx, ev)
        for idx, ev in enumerate(materialized_ledger)
        if ev.type == selector.event_type
    ]

    # Step 2: optional date_range filter.
    if selector.date_range is not None:
        start, end = selector.date_range
        filtered = [(idx, ev) for idx, ev in filtered if start <= ev.date <= end]

    # Step 2b: optional intent filter. Only meaningful for event types that carry an intent
    # (local_intervention); an event with intent=None never matches, which is the safe
    # direction — an unlabelled intervention does not quietly pass a tumour-directed ban.
    if selector.intent_is_one_of is not None:
        wanted = set(selector.intent_is_one_of)
        filtered = [(idx, ev) for idx, ev in filtered
                    if getattr(ev, "intent", None) in wanted]

    # Step 3: optional ordinal_after_cutoff filter — requires observed_prefix.
    if selector.ordinal_after_cutoff is not None:
        if spec.observed_prefix is None:
            raise ValueError(
                f"FutureEventSelector uses ordinal_after_cutoff="
                f"{selector.ordinal_after_cutoff!r} but spec has no "
                f"observed_prefix; cutoff_date is undefined"
            )
        cutoff = spec.observed_prefix.cutoff_date
        filtered = [(idx, ev) for idx, ev in filtered if ev.date > cutoff]

    # Deterministic tie-break: chronological, ties broken by original index.
    filtered.sort(key=lambda item: (item[1].date, item[0]))

    ordinal = selector.ordinal or selector.ordinal_after_cutoff
    if ordinal is None:
        return filtered

    if ordinal == "last":
        return [filtered[-1]] if filtered else []
    idx_into = _ORDINAL_INDEX.get(ordinal)
    if idx_into is None:
        # FutureEventSelector.__post_init__ validates this; defensive only.
        raise ValueError(f"unknown ordinal {ordinal!r}")
    if idx_into >= len(filtered):
        return []
    return [filtered[idx_into]]


def _check_constraint_key(
    key: str,
    value: Any,
    matches: list[tuple[int, MaterializedLedgerEvent]],
    spec: ConditioningSpec,
) -> str | None:
    """Return ``None`` on pass, human-readable reason string on fail.

    Raises ``ValueError`` for malformed constraint values (operator errors).
    """
    if key == "regimen_matches":
        aliases = value or []
        if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes)):
            raise ValueError(
                f"regimen_matches must be a sequence of alias strings; got "
                f"{type(value).__name__}"
            )
        if not aliases:
            # Empty alias list = any regimen accepted (mirrors the runner's
            # behavior when ``_resolve_line1_aliases`` returns an empty tuple).
            return None
        for _, ev in matches:
            if ev.regimen is None:
                return (
                    f"regimen_matches: matched event_id={ev.event_id!r} "
                    f"(type={ev.type!r}, date={ev.date.isoformat()}) has no "
                    f"regimen to compare against"
                )
            observed_label = ev.regimen.label or ""
            observed_components = list(ev.regimen.components or [])
            ok = any(
                _regimen_matches_alias(observed_label, observed_components, alias)
                for alias in aliases
            )
            if not ok:
                return (
                    f"regimen_matches: matched event_id={ev.event_id!r} "
                    f"(label={observed_label!r}, components={observed_components!r}) "
                    f"does not match any alias in {list(aliases)!r}"
                )
        return None

    if key == "event_type_is_one_of":
        allowed = value or []
        if not isinstance(allowed, Sequence) or isinstance(allowed, (str, bytes)):
            raise ValueError(
                f"event_type_is_one_of must be a sequence of event-type "
                f"strings; got {type(value).__name__}"
            )
        allowed_set = {str(t) for t in allowed}
        for _, ev in matches:
            if ev.type not in allowed_set:
                return (
                    f"event_type_is_one_of: matched event_id={ev.event_id!r} "
                    f"has type {ev.type!r}, not in {sorted(allowed_set)!r}"
                )
        return None

    if key == "must_occur_within_days":
        if not isinstance(value, Mapping):
            raise ValueError(
                "must_occur_within_days must be a mapping with `reference` "
                "and `days_max`"
            )
        reference = value.get("reference")
        days_max = value.get("days_max")
        if not isinstance(days_max, int) or isinstance(days_max, bool) or days_max < 0:
            raise ValueError(
                "must_occur_within_days.days_max must be a non-negative integer"
            )
        if reference == "cutoff":
            if spec.observed_prefix is None:
                raise ValueError(
                    "must_occur_within_days.reference='cutoff' but spec has "
                    "no observed_prefix; cutoff_date is undefined"
                )
            cutoff = spec.observed_prefix.cutoff_date
            for _, ev in matches:
                delta_days = (ev.date - cutoff).days
                if not (0 <= delta_days <= days_max):
                    return (
                        f"must_occur_within_days(reference=cutoff, "
                        f"days_max={days_max}): event_id={ev.event_id!r} "
                        f"date={ev.date.isoformat()} is {delta_days} day(s) "
                        f"after cutoff={cutoff.isoformat()} "
                        f"(allowed window: 0 to {days_max})"
                    )
            return None
        if reference == "prior_event":
            raise ValueError(
                "must_occur_within_days.reference='prior_event' is not "
                "implemented in A2"
            )
        raise ValueError(
            f"must_occur_within_days.reference must be 'cutoff' or "
            f"'prior_event'; got {reference!r}"
        )

    # must_not_occur is handled in _evaluate_constraint before this dispatch.
    raise AssertionError(f"unreachable: unhandled constraint key {key!r}")


# ---------------------------------------------------------------------------
# Regimen alias matching — copied verbatim from ``poc_runner`` to avoid an
# import cycle (poc_runner imports conditioning_audit). The parity test in
# ``tests/unit/test_future_constraints_audit.py`` locks bit-for-bit equivalence
# with ``poc_runner._regimen_matches_alias`` over a representative input set.
# ---------------------------------------------------------------------------


def _normalize_regimen_str(text: str) -> str:
    """Lowercase, normalize punctuation, collapse whitespace. Mirrors
    ``poc_runner._normalize_regimen_str`` exactly."""
    out = text.strip().lower()
    for ch in ("-", "/", "+", ","):
        out = out.replace(ch, " ")
    return " ".join(out.split())


def _regimen_matches_alias(
    observed_label: str, observed_components: list[str], alias: str,
) -> bool:
    """Mirrors ``poc_runner._regimen_matches_alias`` exactly.

    Returns True when the alias matches via any of:
    - normalized label equality;
    - alias-token-subset of observed-label tokens;
    - alias equals a normalized component;
    - alias-token-subset of the union of component tokens.
    """
    a = _normalize_regimen_str(alias)
    if not a:
        return False
    obs_label = _normalize_regimen_str(observed_label)
    if obs_label == a:
        return True
    obs_label_tokens = set(obs_label.split()) if obs_label else set()
    a_tokens = set(a.split())
    if a_tokens and a_tokens.issubset(obs_label_tokens):
        return True
    obs_comp_tokens = {_normalize_regimen_str(c) for c in observed_components}
    obs_comp_tokens.discard("")
    if a in obs_comp_tokens:
        return True
    if a_tokens.issubset({t for c in obs_comp_tokens for t in c.split()}):
        return True
    return False


# ---------------------------------------------------------------------------
# Gate A3 — fixed-prefix preflight + collision detection.
# ---------------------------------------------------------------------------
#
# Purely additive: the legacy runner does not currently call these helpers.
# A3.5/B1 wires them in alongside the prompt-template change that introduces
# the ``## Fixed Prefix`` section and the ``{{OBSERVED_PREFIX}}``
# placeholder. The internal data model keeps the historical
# ``ObservedPrefix`` name; prompt-facing language is "fixed prefix" because
# these events can be observed history, operator-imposed counterfactual
# interventions, or both.


class ObservedPrefixError(ValueError):
    """An observed prefix is structurally or chronologically invalid.

    Raised at preflight (before any LLM call). ``path`` is a dotted-style
    locator (``observed_prefix.events[i].date`` etc.); ``reason`` gives the
    human-readable explanation. Mirrors the ``BaselineSeedError`` shape.
    """

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


@dataclass(frozen=True, slots=True)
class PrefixCollision:
    """One concrete prefix-collision between the emitted ledger and a prefix.

    - ``reason``: short machine-readable label (``"emitted_event_before_or_at_cutoff"``
      or ``"emitted_event_duplicates_prefix"``).
    - ``emitted_index``: zero-based position of the offending event in the
      emitted ledger.
    - ``prefix_index``: zero-based position of the duplicated prefix event
      when ``reason == "emitted_event_duplicates_prefix"``; ``None`` otherwise.
    - ``detail``: human-readable detail string for failure-message assembly.
    """

    reason: str
    emitted_index: int
    prefix_index: int | None
    detail: str


def validate_observed_prefix(
    spec: ConditioningSpec,
    *,
    baseline_diagnosis_date: date | None = None,
) -> None:
    """Preflight validation of ``spec.observed_prefix``.

    No-op when ``spec.observed_prefix`` is ``None`` or carries no events.

    Checks (raises ``ObservedPrefixError`` on first failure):

    - Each event's date is ``<= cutoff_date``.
    - When ``baseline_diagnosis_date`` is given, each event's date is
      ``>= baseline_diagnosis_date``. Omitted when ``None`` (the caller may
      not have a single canonical diagnosis date yet — e.g., during fixture
      construction before ``StudyReference`` is applied).
    - Event dates are non-decreasing (chronological order, ties allowed).

    Does NOT check ``event_id`` uniqueness: ``LedgerEvent`` is the
    model-emitted shape and has no ``event_id`` field (it lives on
    ``MaterializedLedgerEvent``, assigned by replay). If a future
    ID-bearing prefix is needed, it requires a deliberate schema change.
    """
    if spec.observed_prefix is None:
        return
    prefix = spec.observed_prefix
    cutoff = prefix.cutoff_date
    previous: date | None = None
    for i, event in enumerate(prefix.events):
        loc = f"observed_prefix.events[{i}]"
        if event.date > cutoff:
            raise ObservedPrefixError(
                path=f"{loc}.date",
                reason=(
                    f"event date {event.date.isoformat()} is after "
                    f"cutoff_date {cutoff.isoformat()}"
                ),
            )
        if (
            baseline_diagnosis_date is not None
            and event.date < baseline_diagnosis_date
        ):
            raise ObservedPrefixError(
                path=f"{loc}.date",
                reason=(
                    f"event date {event.date.isoformat()} is before "
                    f"baseline diagnosis_date "
                    f"{baseline_diagnosis_date.isoformat()}"
                ),
            )
        if previous is not None and event.date < previous:
            raise ObservedPrefixError(
                path=f"{loc}.date",
                reason=(
                    f"event date {event.date.isoformat()} is before the "
                    f"previous event's date {previous.isoformat()}; "
                    f"observed_prefix events must be chronologically "
                    f"non-decreasing"
                ),
            )
        previous = event.date


def _event_semantic_key(event: LedgerEvent) -> str:
    """Canonical content-based key for prefix-collision comparison.

    Uses the Pydantic ``model_dump(mode="json", exclude_none=True)`` so two
    events with identical observable content produce the same key, while
    events that differ in any populated field produce different keys.
    Irrelevant ``None`` fields are dropped via ``exclude_none``. Treatment
    events naturally include ``regimen.label`` and ``regimen.components``
    because those nested objects survive the dump.

    The key is the byte-stable ``canonical_json`` of the dump — same
    serialization used for the four spec-level hashes.
    """
    return canonical_json(
        _canonicalize_spec_value(event.model_dump(mode="json", exclude_none=True))
    )


def assert_no_prefix_collision(
    spec: ConditioningSpec,
    emitted_ledger: Sequence[LedgerEvent],
) -> list[PrefixCollision]:
    """Return collisions between the emitted ledger and ``spec.observed_prefix``.

    No-op (returns ``[]``) when ``spec.observed_prefix`` is ``None``.

    Two collision rules:

    - **Date floor**: every emitted event must have ``date > cutoff_date``.
      An emitted event with ``date == cutoff_date`` is also a collision —
      the contract is strictly after.
    - **No restating prefix events**: an emitted event whose semantic key
      (per :func:`_event_semantic_key`) matches any prefix event's
      semantic key is a collision. Same-day same-type events that differ
      in other populated fields are NOT collisions.

    The caller decides what to do with the result (B1 will convert a
    non-empty list into ``failure_kind="observed_prefix_audit"``).
    """
    if spec.observed_prefix is None:
        return []
    prefix = spec.observed_prefix
    cutoff = prefix.cutoff_date
    prefix_keys: dict[str, int] = {
        _event_semantic_key(ev): i for i, ev in enumerate(prefix.events)
    }
    collisions: list[PrefixCollision] = []
    for emitted_index, event in enumerate(emitted_ledger):
        if event.date <= cutoff:
            collisions.append(
                PrefixCollision(
                    reason="emitted_event_before_or_at_cutoff",
                    emitted_index=emitted_index,
                    prefix_index=None,
                    detail=(
                        f"emitted event #{emitted_index} (type={event.type!r}, "
                        f"date={event.date.isoformat()}) is not strictly after "
                        f"cutoff_date {cutoff.isoformat()}"
                    ),
                )
            )
            continue
        key = _event_semantic_key(event)
        prefix_index = prefix_keys.get(key)
        if prefix_index is not None:
            collisions.append(
                PrefixCollision(
                    reason="emitted_event_duplicates_prefix",
                    emitted_index=emitted_index,
                    prefix_index=prefix_index,
                    detail=(
                        f"emitted event #{emitted_index} (type={event.type!r}, "
                        f"date={event.date.isoformat()}) is semantically "
                        f"identical to prefix event #{prefix_index}; "
                        f"emitted ledger must not restate prefix events"
                    ),
                )
            )
    return collisions
