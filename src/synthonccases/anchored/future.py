"""Acceptance for the continuation contract: is this a legal future for this patient?

Full lean-ledger replay reconstructs a deterministic state trajectory, which needs every
event to carry its disease and host deltas. That is the right bar for a synthetic patient
generated from nothing. It is the wrong bar here, and it was actively costing us: to satisfy
replay we had to drop the query patient's observed imaging progressions from the prefix
while still showing `last_scan_progressing` for all ten anchors, so the model was told less
about the index patient than about the neighbours.

For the primary PFS endpoint we need much less, and it can be checked exactly. The events
must be a legal continuation -- strictly after the cutoff, chronological, one death and it
last -- and they must not rewrite what was already observed. Everything replay adds beyond
that is state we do not score.

Rejection here is deliberately HARD and deliberately typed. Every failure carries a kind, so
acceptance can be compared across arms and across attempts: if difficult trajectories fail
more often than easy ones, then retrying until acceptance is rejection sampling on the very
distribution being scored, and the failure kinds are how that shows up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

#: The event vocabulary, read from the frozen schema rather than restated, so a schema
#: change surfaces here instead of silently admitting an event replay would refuse.
def _allowed_types() -> frozenset[str]:
    from synthonccases.oncollm.poc_runner import LedgerEvent
    schema = LedgerEvent.model_json_schema()
    return frozenset(schema["properties"]["type"]["enum"])


REGIMEN_REQUIRED_TYPES = frozenset({"systemic_treatment_start", "systemic_treatment_switch"})
TREATMENT_TYPES = frozenset({
    "systemic_treatment_start",
    "systemic_treatment_switch",
    "systemic_treatment_stop",
    "systemic_treatment_modification",
})
TERMINAL_TYPES = frozenset({"death"})


@dataclass(frozen=True, slots=True)
class FutureLedger:
    """A validated continuation, or a typed refusal. Never a repaired one."""

    events: tuple[dict, ...] = ()
    failure_kind: str = ""
    detail: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def accepted(self) -> bool:
        return not self.failure_kind

    @property
    def n_events(self) -> int:
        return len(self.events)


def _reject(kind: str, detail: str) -> FutureLedger:
    return FutureLedger(failure_kind=kind, detail=detail)


def _as_date(v: Any) -> date | None:
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, str):
        try:
            return date.fromisoformat(v.strip()[:10])
        except ValueError:
            return None
    return None


def _prefix_key(ev: Mapping[str, Any]) -> tuple:
    reg = ev.get("regimen") or {}
    return (str(ev.get("date"))[:10], str(ev.get("type")), str(reg.get("label", "")))


def validate_future_ledger(payload: Any, *, cutoff: str | date,
                           prefix: Sequence[Mapping[str, Any]] = (),
                           require_terminal: bool = False) -> FutureLedger:
    """Accept a continuation as STRUCTURALLY legal, or refuse it with a countable kind.

    Structure only: after the cutoff, in order, known event types, no restating the prefix,
    one death at most and nothing after it. Whether the course answers a given question is
    a separate judgement -- see :func:`endpoints` -- because the two are different failures.
    A ledger that reaches progression and is then cut off by the token limit is a complete
    PFS observation and a truncated trajectory, and rejecting it would discard a valid
    datum for a reason correlated with the datum's own value.

    ``require_terminal`` is for callers who only want complete courses. It is NOT the
    default: the generation target is still the whole future to death, but the acceptance
    bar belongs to the endpoint being scored.
    """
    cut = _as_date(cutoff)
    if cut is None:
        raise ValueError(f"cutoff is not a date: {cutoff!r}")

    if not isinstance(payload, Mapping):
        return _reject("not_an_object", f"expected a JSON object, got {type(payload).__name__}")
    # The contract forbids a baseline. A model that writes one has answered a different
    # question, and merging it would put invented ECOG and organ function under a real
    # patient's name.
    if "baseline_patch" in payload:
        return _reject("wrote_baseline",
                       "emitted baseline_patch; this contract fixes the patient")
    extra = sorted(set(payload) - {"ledger", "future_ledger"})
    if extra:
        return _reject("unexpected_keys", f"object carries {extra} besides the ledger")
    raw = payload.get("ledger", payload.get("future_ledger"))
    if raw is None:
        return _reject("no_ledger", "object has neither `ledger` nor `future_ledger`")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return _reject("ledger_not_a_list", f"ledger is {type(raw).__name__}")
    if not raw:
        return _reject("empty_ledger", "a continuation with no events says nothing")

    allowed = _allowed_types()
    seen_prefix = {_prefix_key(e) for e in prefix}
    events: list[dict] = []
    previous: date | None = None
    warnings: list[str] = []
    for i, ev in enumerate(raw):
        at = f"ledger[{i}]"
        if not isinstance(ev, Mapping):
            return _reject("event_not_an_object", f"{at} is {type(ev).__name__}")
        etype = str(ev.get("type", ""))
        if etype not in allowed:
            return _reject("unknown_event_type", f"{at}.type={etype!r} is not in the schema")
        when = _as_date(ev.get("date"))
        if when is None:
            return _reject("unparseable_date", f"{at}.date={ev.get('date')!r}")
        if when <= cut:
            # the single most important check. An event at or before the cutoff overwrites
            # observed history, and the endpoint is measured from that cutoff.
            return _reject("event_at_or_before_cutoff",
                           f"{at} dated {when.isoformat()} is not after {cut.isoformat()}")
        if previous is not None and when < previous:
            return _reject("dates_out_of_order",
                           f"{at} dated {when.isoformat()} precedes {previous.isoformat()}")
        if _prefix_key(ev) in seen_prefix:
            return _reject("restated_prefix", f"{at} repeats a fixed prefix event")
        if etype in REGIMEN_REQUIRED_TYPES:
            reg = ev.get("regimen") or {}
            if not reg.get("label") or not reg.get("components"):
                return _reject("treatment_without_regimen",
                               f"{at} is a treatment event with no regimen label/components")
            if not reg.get("component_classes"):
                # a self-annotation, not a fact about the patient: recorded and allowed
                warnings.append(f"{at} has no regimen.component_classes")
        elif etype == "systemic_treatment_modification" and not ev.get("modification_type"):
            warnings.append(f"{at} has no modification_type")
        previous = when
        events.append(dict(ev))

    deaths = [i for i, e in enumerate(events) if str(e["type"]) in TERMINAL_TYPES]
    if len(deaths) > 1:
        return _reject("multiple_deaths", f"{len(deaths)} death events")
    if deaths and deaths[0] != len(events) - 1:
        return _reject("events_after_death",
                       f"{len(events) - 1 - deaths[0]} event(s) follow the death")
    if require_terminal and not deaths:
        return _reject("no_terminal_event",
                       "the contract asks for a complete course ending in one death")
    return FutureLedger(events=tuple(events), warnings=tuple(warnings))


@dataclass(frozen=True, slots=True)
class Endpoints:
    """What one generated course actually resolved, endpoint by endpoint.

    The generator is asked for a complete clinical trajectory. Not every response is one,
    and the ones that fall short do not all fall short of everything: a course that reaches
    progression at six months and stops has resolved PFS and not OS. Scoring each trajectory
    only for the endpoints it resolved is what keeps the PFS benchmark unbiased, because the
    alternative -- one acceptance bar for all endpoints -- discards long courses more often
    than short ones.
    """

    pfs_months: float | None = None
    pfs_event: str = ""
    """Which event ended progression-free survival: progression, recurrence or death."""
    os_months: float | None = None
    death_reached: bool = False
    last_event_month: float | None = None
    n_events: int = 0
    reason: str = ""
    """Why an endpoint is unresolved, when it is."""

    @property
    def pfs_valid(self) -> bool:
        return self.pfs_months is not None

    @property
    def full_course_valid(self) -> bool:
        """Usable for OS, late toxicity, and anything about the end of life."""
        return self.death_reached

    def to_dict(self) -> dict:
        return {"pfs_valid": self.pfs_valid, "pfs_months": self.pfs_months,
                "pfs_event": self.pfs_event, "full_course_valid": self.full_course_valid,
                "death_reached": self.death_reached, "os_months": self.os_months,
                "last_event_month": self.last_event_month, "n_events": self.n_events,
                "reason": self.reason}


def endpoints(ledger: FutureLedger, *, cutoff: str | date) -> Endpoints:
    """Everything this course resolves, and what it does not."""
    from .endpoint import DEATH_EVENTS, MONTH_DAYS, PROGRESSION_EVENTS

    if not ledger.accepted:
        return Endpoints(reason=ledger.failure_kind)
    cut = _as_date(cutoff)
    months = [(_as_date(e["date"]) - cut).days / MONTH_DAYS for e in ledger.events]
    pfs = pfs_event = None
    death = None
    for ev, m in zip(ledger.events, months, strict=True):
        t = str(ev["type"])
        if pfs is None and t in (*PROGRESSION_EVENTS, *DEATH_EVENTS):
            pfs, pfs_event = m, t
        if t in DEATH_EVENTS:
            death = m
    return Endpoints(
        pfs_months=pfs, pfs_event=pfs_event or "", os_months=death,
        death_reached=death is not None, last_event_month=months[-1] if months else None,
        n_events=len(ledger.events),
        reason="" if pfs is not None else "no_progression_or_death")


def pfs_from_future(ledger: FutureLedger, *, cutoff: str | date) -> tuple[float | None, str]:
    """Months from the landmark to the first progression, recurrence or death.

    Returns ``(months, reason)``; months is None when the ledger is not accepted or carries
    no qualifying event, and the reason says which. The extraction is deliberately the same
    rule the real GENIE endpoint uses, so the two are comparable by construction.
    """
    e = endpoints(ledger, cutoff=cutoff)
    return e.pfs_months, (e.pfs_event or e.reason)
