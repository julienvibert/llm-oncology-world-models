"""Generalized conditioning lowering.

Gate A1 of the conditioning-core refactor. Lowers raw per-cell YAML
payloads into a :class:`ConditioningSpec`. Two shapes are accepted:

- **Legacy** — the existing ``cells:`` entries in ``configs/paper_v1_*.yaml``
  (``baseline_seed`` / ``baseline_seed_path`` for the condition side,
  ``first_treatment`` / ``policy_id`` / ``regimen_sequence`` for the policy
  side). Lowered into a ``ConditioningSpec`` with no ``observed_prefix`` and
  at most one ``FutureEventConstraint`` (the implicit "first systemic
  treatment must match the 1L policy").
- **Generalized** — a single ``conditioning: { ... }`` block carrying all
  three axes directly. Built without any legacy interpretation.

This module is the **only** place that knows about the legacy YAML shape.
Pure function; no file I/O, no side effects. Wiring into the runner happens
in subsequent commits (A2/A3/B1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from synthonccases.oncollm.conditioning_audit import (
    ConditioningSpec,
    FutureEventConstraint,
    FutureEventSelector,
    ObservedPrefix,
    compute_pinned_paths,
)
from synthonccases.oncollm.replay import LedgerEvent


LEGACY_FIRST_TREATMENT_CONSTRAINT_ID = "legacy_first_systemic_treatment"


def _as_str_sequence(value: Any) -> list[str] | None:
    """Coerce a string-sequence payload into a ``list[str]``; reject strings.

    Returns ``None`` for non-sequence input or empty sequences, so callers
    can fall through to the next alias source in the priority chain.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    out = [str(item) for item in value if item is not None]
    return out or None


def lower_conditioning_cell(
    *,
    condition_payload: Mapping[str, Any] | None = None,
    policy_payload: Mapping[str, Any] | None = None,
    conditioning_payload: Mapping[str, Any] | None = None,
) -> ConditioningSpec:
    """Lower one cell's raw YAML payloads into a :class:`ConditioningSpec`.

    Precedence: when ``conditioning_payload`` is non-None it is used
    directly (the generalized shape). Otherwise ``condition_payload`` +
    ``policy_payload`` are lowered from the legacy shape. At least one of
    the two paths must be exercised — if all three arguments are None or
    empty the result is a fully-empty spec (``cond:spontaneous`` regime).

    Pure function. No I/O, no side effects. The only place that knows
    about the legacy YAML shape.
    """
    if conditioning_payload is not None:
        return _build_from_generalized(conditioning_payload)
    return _build_from_legacy(
        condition_payload=condition_payload or {},
        policy_payload=policy_payload or {},
    )


# ---------------------------------------------------------------------------
# Legacy lowering (the only place that knows the old YAML shape).
# ---------------------------------------------------------------------------


def _build_from_legacy(
    *,
    condition_payload: Mapping[str, Any],
    policy_payload: Mapping[str, Any],
) -> ConditioningSpec:
    baseline_seed = dict(condition_payload.get("baseline_seed") or {})
    pinned_paths = compute_pinned_paths(baseline_seed)

    # Cells flatten everything into one dict; the legacy split shape spreads
    # condition and policy across two dicts. Merge both so the +1L scan can
    # find ``first_treatment`` / ``fixed_first_line_treatment`` /
    # ``regimen_sequence`` regardless of where the operator put them. Policy
    # keys win on collision because that matches the runner's read order.
    combined: dict[str, Any] = {**condition_payload, **policy_payload}

    future_constraints: list[FutureEventConstraint] = []
    if _has_first_line_treatment_declaration(combined):
        aliases = _resolve_line1_aliases_from_payload(combined)
        future_constraints.append(_legacy_first_line_constraint(aliases))

    free_text = (
        policy_payload.get("text")
        or condition_payload.get("text")
        or policy_payload.get("operator_constraints")
        or condition_payload.get("operator_constraints")
        or ""
    )

    return ConditioningSpec(
        baseline_seed=baseline_seed,
        pinned_paths=pinned_paths,
        observed_prefix=None,
        future_event_constraints=tuple(future_constraints),
        free_text_operator_constraints=str(free_text),
    )


def _has_first_line_treatment_declaration(payload: Mapping[str, Any]) -> bool:
    """True when the cell/policy declares any kind of +1L pin.

    Either the cell-shape ``first_treatment`` block or the legacy split-shape
    ``fixed_first_line_treatment`` block triggers the constraint. Both must
    carry at least a ``regimen_label`` to be considered declared.
    """
    for key in ("first_treatment", "fixed_first_line_treatment"):
        block = payload.get(key)
        if isinstance(block, Mapping) and block.get("regimen_label"):
            return True
    return False


def _resolve_line1_aliases_from_payload(payload: Mapping[str, Any]) -> list[str]:
    """Mirror ``poc_runner._resolve_line1_aliases`` over the raw YAML payload.

    Mirrors the runner's two-stage logic exactly:

    - The runner first computes a default alias set from
      ``first_treatment.aliases`` OR ``(regimen_label, *drugs)`` and uses it
      ONLY to synthesize a default ``line: 1`` row when ``regimen_sequence``
      is empty.
    - If ``regimen_sequence`` is non-empty in the YAML, the runner uses it
      AS-IS — the explicit ``first_treatment.aliases`` override is silently
      dropped and ``_resolve_line1_aliases`` either returns the matching
      line-1 row OR falls back to ``(regimen_label, *drugs)``.

    Priority chain:

    1. ``regimen_sequence`` block carrying a ``line: 1`` row → its ``aliases``.
    2. else, if ``regimen_sequence`` is empty (no rows at all),
       ``first_treatment.aliases`` (explicit operator override).
    3. else ``(regimen_label, *drugs)`` from whichever of ``first_treatment``
       or ``fixed_first_line_treatment`` is present, de-duplicated in order.

    Returns ``[]`` if no source produces aliases. Empty alias lists are
    legal — the audit gate (A2) treats them as "any first systemic_treatment
    accepted", same as the runner today.
    """
    regimen_sequence = payload.get("regimen_sequence")
    has_regimen_sequence_rows = False
    if isinstance(regimen_sequence, Sequence) and not isinstance(
        regimen_sequence, (str, bytes)
    ):
        for row in regimen_sequence:
            if not isinstance(row, Mapping):
                continue
            has_regimen_sequence_rows = True
            if row.get("line") != 1:
                continue
            row_aliases = _as_str_sequence(row.get("aliases"))
            if row_aliases:
                return row_aliases

    block = payload.get("first_treatment") or payload.get(
        "fixed_first_line_treatment"
    )
    if not isinstance(block, Mapping):
        return []

    # The override is only honored when the operator did NOT supply a
    # regimen_sequence — that is the only YAML shape under which the
    # runner's synthesized default line-1 row is materialized.
    if not has_regimen_sequence_rows:
        override = _as_str_sequence(block.get("aliases"))
        if override:
            return override

    label = str(block.get("regimen_label") or "").strip()
    drugs = _as_str_sequence(block.get("drugs")) or []
    aliases: list[str] = []
    for candidate in [label, *drugs]:
        if candidate and candidate not in aliases:
            aliases.append(candidate)
    return aliases


def _legacy_first_line_constraint(aliases: list[str]) -> FutureEventConstraint:
    return FutureEventConstraint(
        constraint_id=LEGACY_FIRST_TREATMENT_CONSTRAINT_ID,
        selector=FutureEventSelector(
            event_type="systemic_treatment_start",
            ordinal="first",
        ),
        constraint={"regimen_matches": list(aliases)},
    )


# ---------------------------------------------------------------------------
# Generalized lowering (the new YAML shape; nothing legacy here).
# ---------------------------------------------------------------------------


def _build_from_generalized(
    conditioning_payload: Mapping[str, Any],
) -> ConditioningSpec:
    baseline_seed = dict(conditioning_payload.get("baseline_seed") or {})
    pinned_paths = compute_pinned_paths(baseline_seed)
    observed_prefix = _build_observed_prefix(conditioning_payload.get("fixed_events"))
    future_constraints = _build_future_constraints(
        conditioning_payload.get("future_event_constraints") or ()
    )
    return ConditioningSpec(
        baseline_seed=baseline_seed,
        pinned_paths=pinned_paths,
        observed_prefix=observed_prefix,
        future_event_constraints=tuple(future_constraints),
        free_text_operator_constraints=str(
            conditioning_payload.get("free_text_operator_constraints") or ""
        ),
    )


def _build_observed_prefix(
    fixed_events_payload: Any,
) -> ObservedPrefix | None:
    if not fixed_events_payload:
        return None
    if not isinstance(fixed_events_payload, Mapping):
        raise ValueError(
            "conditioning.fixed_events must be a mapping with `cutoff_date` "
            "and `events`"
        )
    origin = fixed_events_payload.get("origin", "observed_prefix")
    if origin != "observed_prefix":
        raise ValueError(
            f"conditioning.fixed_events.origin: {origin!r} is not supported. "
            "A1 supports 'observed_prefix' only; other origins are deferred."
        )
    cutoff_raw = fixed_events_payload.get("cutoff_date")
    if not cutoff_raw:
        raise ValueError(
            "conditioning.fixed_events.cutoff_date is required when "
            "fixed_events is present"
        )
    cutoff_date = (
        cutoff_raw
        if isinstance(cutoff_raw, date)
        else date.fromisoformat(str(cutoff_raw))
    )
    raw_events = fixed_events_payload.get("events") or ()
    events = tuple(
        ev if isinstance(ev, LedgerEvent) else LedgerEvent.model_validate(ev)
        for ev in raw_events
    )
    return ObservedPrefix(events=events, cutoff_date=cutoff_date)


def _build_future_constraints(payload: Any) -> list[FutureEventConstraint]:
    out: list[FutureEventConstraint] = []
    for raw in payload:
        if not isinstance(raw, Mapping):
            raise ValueError(
                "conditioning.future_event_constraints[]: each item must be "
                f"a mapping; got {type(raw).__name__}"
            )
        constraint_id = raw.get("constraint_id")
        if not constraint_id:
            raise ValueError(
                "conditioning.future_event_constraints[]: constraint_id is required"
            )
        selector_raw = raw.get("selector") or {}
        if not isinstance(selector_raw, Mapping):
            raise ValueError(
                "conditioning.future_event_constraints[].selector must be a mapping"
            )
        event_type = selector_raw.get("event_type")
        if not event_type:
            raise ValueError(
                "conditioning.future_event_constraints[].selector.event_type "
                "is required"
            )
        selector = FutureEventSelector(
            event_type=str(event_type),
            ordinal=selector_raw.get("ordinal"),
            ordinal_after_cutoff=selector_raw.get("ordinal_after_cutoff"),
            date_range=_parse_date_range(selector_raw.get("date_range")),
            intent_is_one_of=_parse_intents(selector_raw.get("intent_is_one_of")),
        )
        constraint = raw.get("constraint") or {}
        if not isinstance(constraint, Mapping):
            raise ValueError(
                "conditioning.future_event_constraints[].constraint must be a "
                "mapping"
            )
        out.append(
            FutureEventConstraint(
                constraint_id=str(constraint_id),
                selector=selector,
                constraint=dict(constraint),
            )
        )
    return out


def _parse_intents(value: Any) -> tuple[str, ...] | None:
    """Lower ``selector.intent_is_one_of`` from YAML. A bare string is accepted as one value."""
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            "selector.intent_is_one_of must be a non-empty string or sequence of strings"
        )
    return tuple(str(v) for v in value)


def _parse_date_range(value: Any) -> tuple[date, date] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            "selector.date_range must be a 2-element [start, end] sequence"
        )
    start_raw, end_raw = value
    start = (
        start_raw
        if isinstance(start_raw, date)
        else date.fromisoformat(str(start_raw))
    )
    end = (
        end_raw if isinstance(end_raw, date) else date.fromisoformat(str(end_raw))
    )
    return (start, end)
