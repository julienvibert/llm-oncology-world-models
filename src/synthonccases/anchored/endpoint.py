"""Derive PFS from a generated record, unambiguously or not at all.

The model simulates a course; the endpoint is extracted afterwards. That is the whole point
of asking for a trajectory rather than a number -- but it only works if the extraction is
deterministic and refuses when the record does not actually say what happened.

PFS from a landmark is the first of:

    * a ``progression`` event dated after the landmark,
    * a ``recurrence`` event dated after the landmark,
    * the ``progression_date`` recorded on the treatment line itself,
    * death.

If none of those exist the record is CENSORED at last contact, not scored as
progression-free forever. If the record is unparseable, or dates run backwards, or death
precedes the landmark, extraction returns a failure with a reason. A failed extraction is
reported and excluded; it is never quietly turned into a number, because a parse failure that
becomes a data point is indistinguishable from a prediction.

Dates are handled in months from the landmark. The generator writes absolute dates, so a
landmark date is required and every event is expressed relative to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

MONTH_DAYS = 30.44

#: Timeline event types that end progression-free survival. Taken from the record schema's
#: own vocabulary rather than invented here, so a schema change surfaces as a KeyError in
#: tests rather than as a silently missed event.
PROGRESSION_EVENTS = ("progression", "recurrence")
DEATH_EVENTS = ("death",)


@dataclass(frozen=True, slots=True)
class Endpoint:
    """One extracted PFS observation, or a stated failure."""

    months: float | None
    event: int | None
    """1 progressed or died, 0 censored at last contact, None when extraction failed."""
    reason: str = ""
    source: str = ""
    """Which rule fired: progression_event, line_progression_date, death, or last_contact."""

    @property
    def ok(self) -> bool:
        return self.months is not None and self.event is not None


def _as_date(v: Any) -> date | None:
    if isinstance(v, date):
        return v
    if not isinstance(v, str) or not v.strip():
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(v.strip()[:10], fmt).date()
        except ValueError:
            continue
    return None


def _months_between(a: date, b: date) -> float:
    return (b - a).days / MONTH_DAYS


def extract_pfs(record: dict, landmark: str | date) -> Endpoint:
    """PFS in months from `landmark`, from one generated record."""
    lm = _as_date(landmark)
    if lm is None:
        return Endpoint(None, None, reason="landmark date is unparseable")
    if not isinstance(record, dict):
        return Endpoint(None, None, reason="record is not an object")

    traj = record.get("trajectory")
    if not isinstance(traj, dict):
        return Endpoint(None, None, reason="record has no trajectory block")
    timeline = traj.get("timeline") or []
    if not isinstance(timeline, list):
        return Endpoint(None, None, reason="timeline is not a list")

    candidates: list[tuple[float, int, str]] = []
    for ev in timeline:
        if not isinstance(ev, dict):
            continue
        et = str(ev.get("event_type", "")).strip().lower()
        d = _as_date(ev.get("date"))
        if d is not None and d >= lm:
            if et in PROGRESSION_EVENTS:
                candidates.append((_months_between(lm, d), 1, "progression_event"))
            elif et in DEATH_EVENTS:
                candidates.append((_months_between(lm, d), 1, "death"))
        # a treatment line may record its own progression date rather than emit an event
        pd_ = _as_date(ev.get("progression_date"))
        if pd_ is not None and pd_ >= lm:
            candidates.append((_months_between(lm, pd_), 1, "line_progression_date"))

    outcome = traj.get("outcome") or {}
    death = _as_date(outcome.get("date_of_death")) if isinstance(outcome, dict) else None
    if death is not None and death >= lm:
        candidates.append((_months_between(lm, death), 1, "death"))
    if death is not None and death < lm:
        return Endpoint(None, None, reason="death precedes the landmark")

    if candidates:
        months, ev_flag, src = min(candidates, key=lambda c: c[0])
        if months < 0:
            return Endpoint(None, None, reason="event precedes the landmark")
        return Endpoint(round(months, 3), ev_flag, source=src)

    last = _as_date(outcome.get("last_contact_date")) if isinstance(outcome, dict) else None
    if last is not None and last >= lm:
        # no progression and no death recorded: censored, never "progression-free forever"
        return Endpoint(round(_months_between(lm, last), 3), 0, source="last_contact")
    return Endpoint(None, None,
                    reason="no progression, death, or last-contact date after the landmark")


def extract_many(records: list[dict], landmark: str | date) -> tuple[list[float], dict]:
    """PFS samples from repeated generations, plus an honest audit of what was dropped.

    Only PROGRESSED observations become samples: the scorer builds an empirical survival
    curve from simulated futures, and a censored simulated future is a simulation that was
    stopped early rather than a patient who was lost. Those are counted and reported so the
    share is visible, not folded into the distribution as if they were events.
    """
    samples, censored, failed = [], 0, []
    for r in records:
        ep = extract_pfs(r, landmark)
        if not ep.ok:
            failed.append(ep.reason)
            continue
        if ep.event == 1:
            samples.append(ep.months)
        else:
            censored += 1
    audit = {"n_records": len(records), "n_samples": len(samples),
             "n_censored": censored, "n_failed": len(failed),
             "failure_reasons": sorted(set(failed))}
    return samples, audit
