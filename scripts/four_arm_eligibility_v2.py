#!/usr/bin/env python3
"""Gate-3 eligibility, version 2: the same observation adapter, corrected readers.

Specification: `manuscript/SCORING_SPEC_2026-09-28.md` as amended by
`manuscript/SCORING_SPEC_amendment_2026-10-12.md` (post-hoc freeze, not preregistration).
Version 1 (`four_arm_eligibility_v1.py`) is kept unchanged as the record of the first
precheck. This version differs in what it reads, not in how it observes:

    disease, real      unchanged: `canonical_future_v1.project` under each convention
    disease, generated R1: `four_arm_readers_v2.generated_course` (cumulative truth, no
                       restatement); identical under both conventions
    treatment, both    R2-R5: one `Treatment` object per course (`four_arm_readers_v2`),
                       used for resolution AND for the coordinate values

The observation adapter (`observe`) and the cell/card rules are imported from version 1, so
there is still exactly one adapter. This module also builds the 17 coordinate VALUES for
traced examples and the section-6 common windows; it computes no distance, score, arm
effect or outcome agreement.

GOVERNANCE. GENIE-BPC under DUA. No data held here.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
sys.path.insert(0, str(ROOT / "src"))
import canonical_future_v1 as CF  # noqa: E402
import four_arm_eligibility_v1 as E1  # noqa: E402
import four_arm_readers_v2 as R  # noqa: E402

# One adapter and one set of cell/card rules, shared with version 1.
ARMS, HORIZONS, CONVENTIONS = E1.ARMS, E1.HORIZONS, E1.CONVENTIONS
MIN_ACCEPTED, CARD_SIZE, PAIRED_FLOOR, CARDS = (E1.MIN_ACCEPTED, E1.CARD_SIZE,
                                                E1.PAIRED_FLOOR, E1.CARDS)
BLOCKS, UNKNOWN, EPS = E1.BLOCKS, E1.UNKNOWN, E1.EPS
DISEASE_AT_LANDMARK, DISEASE_AFTER_LANDMARK = E1.DISEASE_AT_LANDMARK, E1.DISEASE_AFTER_LANDMARK
convention, observe, state_at, first_unresolved = (E1.convention, E1.observe, E1.state_at,
                                                   E1.first_unresolved)
calendar_for, llm_cell, donor_card, emit_cells = (E1.calendar_for, E1.llm_cell,
                                                  E1.donor_card, E1.emit_cells)
Resolution, unresolved = E1.Resolution, E1.unresolved
#: Section 6: minimum comparable time per donor, and usable donors per patient.
MIN_WINDOW_MONTHS = 1.0
MIN_USABLE_DONORS = 7
LIVING = ("no_active_disease", "responding", "stable", "progressing")


@dataclass(frozen=True)
class Course:
    """One course on its own landmark clock (months), before observation."""

    path: tuple[tuple[float, float, str], ...]
    death: float
    follow_end: float
    treatment: R.Treatment
    terminal: str = "registry"


def real_course(p, tau: float, horizon: float, agents: Sequence[R.RealAgent]) -> Course:
    """A GENIE patient from its own landmark: disease as before, treatment from R2."""
    fut = CF.project(p, tau, horizon)
    death = p.death - tau if math.isfinite(p.death) else math.inf
    return Course(path=tuple(fut.path), death=death, follow_end=p.last_contact - tau,
                  treatment=R.real_treatment(agents, tau), terminal="registry")


def generated_course(ledger: list[dict], horizon: float, **kw) -> Course:
    """An accepted ledger (R1 disease truth, R4 treatment). Convention-independent."""
    g = R.generated_course(ledger, horizon=horizon, **kw)
    follow = math.inf if math.isfinite(g.death) else g.last_contact
    return Course(path=g.path, death=g.death, follow_end=follow, treatment=g.treatment,
                  terminal=g.terminal)


def _survival_end(course: Course, horizon: float) -> tuple[bool, float]:
    died = course.death <= horizon + EPS
    ok = died or course.follow_end >= horizon - EPS
    return ok, (min(course.death, horizon) if died else horizon)


def resolve(course: Course, calendar: Sequence[float], horizon: float) -> Resolution:
    """Which of the four blocks this course resolves over [0, H] (same rules as v1 for
    survival, disease and events; treatment from the coordinate builder itself)."""
    ok, end = _survival_end(course, horizon)
    if not ok:
        return unresolved(f"survival_unresolved_{course.terminal}")
    reasons: list[str] = []
    gap = first_unresolved(observe(course.path, calendar, end), end)
    disease = gap is None
    if not disease:
        reasons.append(DISEASE_AT_LANDMARK if gap <= 1e-6 else DISEASE_AFTER_LANDMARK)
    values, tx_reasons = R.treatment_coordinates(course.treatment, course.death, horizon)
    tx_ok = all(v is not None for v in values.values())
    reasons += tx_reasons if not tx_ok else []
    return Resolution((True, disease, disease, tx_ok), tuple(reasons))


# --------------------------------------------------------------------------------------
# The 17 coordinate values (for traced examples and gate 4; no distance here)
# --------------------------------------------------------------------------------------

COORDINATES = (("survival", ("alive_at_h", "months_alive")),
               ("disease", ("months_no_active_disease", "months_responding",
                            "months_stable", "months_progressing")),
               ("events", ("entries_progressing", "entries_responding")),
               ("treatment", R.TREATMENT_COORDS))


def _transform(name: str, value: float, horizon: float) -> float:
    if name == "alive_at_h" or name == "first_change":
        return value
    if name.startswith(("months_", "time_to_")):
        return value / horizon
    return value / (1.0 + value)              # counts: n / (1 + n)


def coordinates(course: Course, calendar: Sequence[float], horizon: float) -> dict:
    """Raw values (clinical units), transformed values and block-weighted vector.

    Any unresolved coordinate is None, and so is the weighted vector. Block b of
    dimension d_b is multiplied by sqrt((1/4)/d_b), as specified; no distance is computed.
    """
    ok, end = _survival_end(course, horizon)
    raw: dict[str, float | None] = {}
    died = course.death <= horizon + EPS
    raw["alive_at_h"] = None if not ok else (0.0 if died else 1.0)
    raw["months_alive"] = None if not ok else float(end)
    seen = observe(course.path, calendar, end) if ok else []
    disease_ok = ok and first_unresolved(seen, end) is None
    for st in LIVING:
        raw[f"months_{st}"] = (float(sum(b - a for a, b, s in seen if s == st))
                               if disease_ok else None)
    for st, key in (("progressing", "entries_progressing"),
                    ("responding", "entries_responding")):
        raw[key] = (float(sum(1 for i, seg in enumerate(seen)
                              if i > 0 and seg[2] == st and seen[i - 1][2] != st))
                    if disease_ok else None)
    tx, reasons = (R.treatment_coordinates(course.treatment, course.death, horizon) if ok
                   else ({k: None for k in R.TREATMENT_COORDS}, []))
    raw |= tx
    transformed = {k: (None if v is None else _transform(k, v, horizon))
                   for k, v in raw.items()}
    vector = None
    if all(v is not None for v in transformed.values()):
        vector = []
        for _block, names in COORDINATES:
            w = math.sqrt(0.25 / len(names))
            vector += [w * transformed[n] for n in names]
    return {"raw": raw, "transformed": transformed, "weighted_vector": vector,
            "observed_path": seen, "treatment_reasons": reasons}


# --------------------------------------------------------------------------------------
# Section 6: supported time and common windows (fixed before any LLM future is seen)
# --------------------------------------------------------------------------------------

def supported(course: Course, calendar: Sequence[float], horizon: float
              ) -> list[tuple[float, float]]:
    """Relative times in [0, H] on which this course is supported for the disease
    diagnostic: observed living time with a known state, plus known post-death time."""
    died = course.death <= horizon + EPS
    end = min(course.death, horizon) if died else horizon
    spans = [(a, b) for a, b, s in observe(course.path, calendar, end) if s != UNKNOWN]
    if died:
        spans.append((course.death, horizon))
    return R.merge(spans)


def intersect(x: Sequence[tuple[float, float]], y: Sequence[tuple[float, float]]
              ) -> list[tuple[float, float]]:
    out = []
    for a, b in x:
        for c, d in y:
            lo, hi = max(a, c), min(b, d)
            if hi > lo + EPS:
                out.append((lo, hi))
    return R.merge(out)


def measure(spans: Sequence[tuple[float, float]]) -> float:
    return sum(b - a for a, b in spans)


def donor_window(query: Sequence[tuple[float, float]], donor: Course,
                 calendar: Sequence[float], horizon: float) -> list[tuple[float, float]]:
    """W for one (query, donor) pair: the donor read on the QUERY's calendar."""
    return intersect(query, supported(donor, calendar, horizon))


def fallback_patient(windows: Sequence[Sequence[tuple[float, float]]]) -> dict:
    """Section 6 at the patient level: usable donors (|W| >= one month), scored time."""
    sizes = [measure(w) for w in windows]
    usable = [s for s in sizes if s >= MIN_WINDOW_MONTHS - EPS]
    return {"n_donors": len(sizes), "n_usable": len(usable),
            "eligible": len(usable) >= MIN_USABLE_DONORS,
            "all_twenty_usable": len(sizes) == CARD_SIZE and len(usable) == CARD_SIZE,
            "scored_months": sum(usable)}
