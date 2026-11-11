#!/usr/bin/env python3
"""Gate-3 eligibility for the joint 17-coordinate score: the observation adapter and the
resolution rules, and nothing that scores.

Specification: `manuscript/SCORING_SPEC_2026-09-28.md` (sections 4 and 7) as amended in
`manuscript/review/2026-09-30_codex_gate3_amendment.md`. This module decides WHETHER each
core coordinate is resolved over [0, H]; it never computes a coordinate value, a distance,
a score, an arm effect or any agreement between a forecast and an outcome.

THREE KINDS OF COURSE, ONE READER

    real query     its own GENIE record, projected by `canonical_future_v1.project`
    donor member   its own GENIE record, on its OWN landmark clock, same projector
    LLM repeat     the accepted ledger, projected by `generated_future_v1.project` in
                   truth mode (the asserted course; it delegates to the same projector)

Each is then read through `observe()`, the single observation adapter, on the QUERY's
scoring calendar: time zero, the saved post-landmark assessment times (checked against the
source presence/status records by the caller) and the query's timed curated new-site
documentation. At each calendar time the forecast's latest supported state is read, and it
holds until the next calendar time or until the forecast's own source stops supporting a
state, whichever comes first. A read never renews the source's evidence clock and never
creates an earlier state. Death is exact and absorbing; it is not calendar-sampled.

THE SEVENTEEN CORE COORDINATES, by block, and what resolves them at H

    survival (2)    a death by H, or follow-up through H
    disease (4)     survival resolved, and a known observed state at every living instant
                    of [0, min(death, H)) under the selected convention
    events (2)      derived from the same observed path, so resolved exactly when disease is
    treatment (9)   survival resolved, every regimen active in [0, min(death, H)) maps to a
                    known modality, and no agent record that could be active in that window
                    lacks a stop date

A death resolves only the time after it. Nothing here imputes, shortens the window, drops a
coordinate per draw, or reads unknown as healthy.

GOVERNANCE. GENIE-BPC under DUA. This module holds no data; callers write patient-level
rows only under the gitignored `outputs/four_arm_preprint_20260928/`.
"""

from __future__ import annotations

import contextlib
import math
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
import canonical_future_v1 as CF  # noqa: E402
import generated_future_v1 as GF  # noqa: E402
import genie_operator_registry_v2 as GV2  # noqa: E402

ARMS = ("patient_only", "setting", "generic_cohort", "patients_like_me")
HORIZONS = (3.0, 6.0, 12.0, 24.0)
#: (record rule, persistence) in canonical_future_v1. Primary: chronological forward
#: persistence. Sensitivity: conservative conflict handling WITH four-month expiry; both
#: switches together, as the specification requires.
CONVENTIONS = {"primary": ("chronological", "under_followup"),
               "sensitivity": ("conservative", "assessment_validity")}
MIN_ACCEPTED = 7
CARD_SIZE = 20
PAIRED_FLOOR = 20
#: card name -> (query field holding the members, the LLM arm that received that card)
CARDS = {"generic": ("generic_donors", "generic_cohort"),
         "matched": ("relevant_donors", "patients_like_me")}
BLOCKS = ("survival", "disease", "events", "treatment")
N_COORDINATES = {"survival": 2, "disease": 4, "events": 2, "treatment": 9}
EPS = 1e-9
UNKNOWN = "unknown"

#: Reason codes that make a block unresolved. A missing agent stop date is split by where
#: the agent starts (inside the window, or before the landmark and so possibly still active
#: at it) and by the registry's discontinuation flag (`DRUGS_DC_YN`): `ongoing` = "No",
#: `undated` = "Yes" without a date, `undocumented` = unknown. All six make the treatment
#: block unresolved; the split exists so the effect of each can be reported, not relaxed.
STOP_IN_WINDOW = "treatment_stop_date_missing_in_window"
STOP_BEFORE_LANDMARK = "treatment_stop_date_missing_before_landmark"
STOP_KINDS = ("ongoing", "undated", "undocumented")
UNMAPPED = "treatment_modality_unmapped"
DISEASE_AT_LANDMARK = "disease_unknown_from_landmark"
DISEASE_AFTER_LANDMARK = "disease_unknown_after_landmark"

Path_ = Sequence[tuple[float, float, str]]


@contextlib.contextmanager
def convention(name: str) -> Iterator[None]:
    """Switch the shared projector to one reading of the record, and always restore it."""
    rule, persistence = CONVENTIONS[name]
    old = (CF.RECORD_RULE, CF.DEFAULT_PERSISTENCE)
    CF.RECORD_RULE, CF.DEFAULT_PERSISTENCE = rule, persistence
    try:
        yield
    finally:
        CF.RECORD_RULE, CF.DEFAULT_PERSISTENCE = old


# --------------------------------------------------------------------------------------
# The observation adapter
# --------------------------------------------------------------------------------------

def _segment_at(path: Path_, t: float) -> tuple[float, float, str] | None:
    """The half-open segment [a, b) containing t, or None when t is not covered."""
    for a, b, st in path:
        if a <= t + EPS and t < b - EPS:
            return (a, b, st)
    return None


def state_at(path: Path_, t: float) -> str:
    seg = _segment_at(path, t)
    return seg[2] if seg else UNKNOWN


def supported_until(path: Path_, start: float, stop: float) -> float:
    """First time in [start, stop) at which the source no longer supports a known state."""
    cur = start
    while cur < stop - EPS:
        seg = _segment_at(path, cur)
        if seg is None or seg[2] == UNKNOWN:
            return cur
        if seg[1] <= cur + EPS:        # defensive: a degenerate segment cannot advance
            return cur
        cur = seg[1]
    return stop


def _append(out: list[tuple[float, float, str]], a: float, b: float, st: str) -> None:
    if b - a <= 1e-6:
        return
    if out and out[-1][2] == st and abs(out[-1][1] - a) <= 1e-6:
        out[-1] = (out[-1][0], b, st)
    else:
        out.append((a, b, st))


def observe(source: Path_, calendar: Sequence[float], end: float) -> list[tuple[float, float, str]]:
    """THE observation adapter: a course read on the query's scoring calendar over [0, end).

    `source` is the course's own path on its own landmark clock, as the canonical projector
    built it under the selected convention (real or donor: from the record, with that
    convention's persistence; generated: the asserted course). `calendar` is the query's
    scoring calendar; time zero is always included. `end` is min(own death, H).

    At each calendar time c_k the source's state at c_k is read and held until c_{k+1}, or
    until the source stops supporting a known state (its own evidence lapses, a conflict,
    its own follow-up ends), whichever is first; then the observed state is unknown until
    the next read. So a read never renews a source's evidence clock, never fills time
    before the source's first supported state, and a change inside (c_k, c_{k+1}) is seen
    only at c_{k+1}.
    """
    out: list[tuple[float, float, str]] = []
    if end <= EPS:
        return out
    points = sorted({0.0} | {float(c) for c in calendar if EPS < c < end - EPS})
    for k, c in enumerate(points):
        hi = points[k + 1] if k + 1 < len(points) else end
        st = state_at(source, c)
        if st == UNKNOWN:
            _append(out, c, hi, UNKNOWN)
            continue
        lapse = supported_until(source, c, hi)
        _append(out, c, lapse, st)
        if lapse < hi - EPS:
            _append(out, lapse, hi, UNKNOWN)
    return out


def first_unresolved(path: Path_, end: float) -> float | None:
    """Earliest time in [0, end) with no known state (gap or unknown); None if fully known."""
    cur = 0.0
    for a, b, st in sorted(path):
        if b <= cur + EPS:
            continue
        if a > cur + 1e-6 or st == UNKNOWN:
            return cur if a > cur + 1e-6 else max(cur, a)
        cur = b
        if cur >= end - 1e-6:
            return None
    return None if cur >= end - 1e-6 else cur


# --------------------------------------------------------------------------------------
# Courses
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Course:
    """One course on its own landmark clock (months), before observation."""

    path: tuple[tuple[float, float, str], ...]
    death: float                       # months to death, inf when none by follow-up end
    follow_end: float                  # months to the end of observation
    regimens: tuple[tuple[float, float, str], ...]   # (start, stop, label)
    stop_unknown: tuple[tuple[float, str], ...] = ()  # (agent start, kind), no stop date
    terminal: str = "registry"         # registry / death / censored / truncated


def real_course(p, tau: float, horizon: float,
                stop_unknown: Sequence[tuple[float, str]] = ()) -> Course:
    """A GENIE patient (query or donor) from its own saved landmark `tau`."""
    fut = CF.project(p, tau, horizon)
    death = p.death - tau if math.isfinite(p.death) else math.inf
    return Course(path=tuple(fut.path), death=death, follow_end=p.last_contact - tau,
                  regimens=tuple((ln["start"] - tau, ln["stop"] - tau, str(ln["label"]))
                                 for ln in p.lines),
                  stop_unknown=tuple((t - tau, kind) for t, kind in stop_unknown),
                  terminal="registry")


def generated_course(ledger: list[dict], horizon: float, **kw) -> Course:
    """An accepted ledger in truth mode. Raises GF.NotScoreable when it has no path."""
    fut = GF.project(ledger, horizon=horizon, **kw)
    p = GF.as_patient(ledger, horizon=horizon, **kw)
    death = float(p.death) if math.isfinite(p.death) else math.inf
    follow = math.inf if math.isfinite(death) else float(p.last_contact)
    return Course(path=tuple(fut.path), death=death, follow_end=follow,
                  regimens=tuple((float(ln["start"]), float(ln["stop"]), str(ln["label"]))
                                 for ln in p.lines),
                  terminal=p.terminal_kind)


# --------------------------------------------------------------------------------------
# Resolution of the 17 coordinates
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Resolution:
    blocks: tuple[bool, bool, bool, bool]   # in BLOCKS order
    reasons: tuple[str, ...]

    @property
    def full_core(self) -> bool:
        return all(self.blocks)

    def full_core_ignoring(self, ignorable: frozenset[str]) -> bool:
        """Diagnostic only: would the core resolve if these TREATMENT reasons were ignored?"""
        surv, dis, ev, tx = self.blocks
        if tx:
            return surv and dis and ev
        tx_reasons = {r for r in self.reasons if r.startswith("treatment_")}
        return surv and dis and ev and tx_reasons <= ignorable


def unresolved(reason: str) -> Resolution:
    return Resolution((False, False, False, False), (reason,))


def treatment_reasons(course: Course, end: float) -> list[str]:
    out: list[str] = []
    for start, stop, label in course.regimens:
        active = start < end - EPS and (start >= -EPS or stop > EPS)
        if active and GV2.modalities(label) == frozenset({"other"}):
            out.append(UNMAPPED)
            break
    for t, kind in course.stop_unknown:
        if t < end - EPS:
            code = f"{STOP_IN_WINDOW if t >= -EPS else STOP_BEFORE_LANDMARK}_{kind}"
            if code not in out:
                out.append(code)
    return out


def resolve(course: Course, calendar: Sequence[float], horizon: float) -> Resolution:
    """Which of the four blocks (17 coordinates) this course resolves over [0, horizon]."""
    died = course.death <= horizon + EPS
    if not (died or course.follow_end >= horizon - EPS):
        return unresolved(f"survival_unresolved_{course.terminal}")
    end = min(course.death, horizon) if died else horizon
    reasons: list[str] = []
    gap = first_unresolved(observe(course.path, calendar, end), end)
    disease = gap is None
    if not disease:
        reasons.append(DISEASE_AT_LANDMARK if gap <= 1e-6 else DISEASE_AFTER_LANDMARK)
    tx = treatment_reasons(course, end)
    reasons += tx
    return Resolution((True, disease, disease, not tx), tuple(reasons))


# --------------------------------------------------------------------------------------
# Cells, cards and intersections
# --------------------------------------------------------------------------------------

def llm_cell(n_accepted: int, repeats: Sequence[Resolution]) -> tuple[bool, str]:
    """An LLM (query, arm) cell at one horizon and convention.

    At least seven ORIGINALLY accepted repeats, and EVERY accepted repeat fully resolved.
    An unresolved repeat makes the cell unavailable; it is never discarded to save the cell.
    """
    if len(repeats) != n_accepted:
        raise ValueError("every accepted repeat must have a resolution")
    if n_accepted < MIN_ACCEPTED:
        return False, "fewer_than_seven_accepted"
    if not all(r.full_core for r in repeats):
        return False, "accepted_repeat_unresolved"
    return True, ""


def donor_card(members: Sequence[Resolution]) -> tuple[bool, str]:
    """A donor card: all twenty saved members resolved, decided before any subsampling."""
    if len(members) != CARD_SIZE:
        return False, "card_size_not_twenty"
    if not all(m.full_core for m in members):
        return False, "card_member_unresolved"
    return True, ""


def emit_cells(expected: Sequence[tuple[str, str]],
               accepted: dict[tuple[str, str], list[Resolution]]) -> list[dict]:
    """One row per REQUESTED (query, arm) cell, including cells with no accepted response."""
    rows = []
    for key in expected:
        reps = accepted.get(key, [])
        ok, why = llm_cell(len(reps), reps)
        rows.append({"query_id": key[0], "arm": key[1], "n_accepted": len(reps),
                     "n_resolved": sum(r.full_core for r in reps),
                     "eligible": ok, "reason": why})
    return rows


def calendar_for(p, tau: float, saved: Sequence[float],
                 max_horizon: float = max(HORIZONS)) -> list[float]:
    """The query's scoring calendar, checked against its source records.

    The saved `assessment_times` are status and presence times in (0, 24], rounded to four
    decimals when the prompts were built. They are recomputed here and must agree; the
    UNROUNDED times are used, so the calendar coincides exactly with the record times the
    real path changes at. Timed curated new-site documentation is added, as the
    specification requires it to be represented in both the real path and the calendar.
    """
    rec = sorted({w - tau for w, _v in (list(p.status) + list(p.presence))
                  if tau < w <= tau + max_horizon})
    if sorted({round(t, 4) for t in rec}) != sorted(saved):
        raise ValueError("saved assessment calendar disagrees with source records")
    mets = p.context.get("mets")
    sites = {m - tau for m, _s in (mets.sites if mets else []) if tau < m <= tau + max_horizon}
    return sorted({0.0} | set(rec) | sites)
