#!/usr/bin/env python3
"""ONE projection of what happened after the landmark, read by cards, inspection and scoring.

Three consumers were each deriving the future separately and each derived a different one:

    the card       raw response-status transitions only. A localized donor who stayed
                   free of disease rendered as "no change recorded", and a distant relapse
                   was invisible, because neither is a response-status row.
    the scorer     responding and progressing occupancy only -- no stable, and no
                   no_active_disease -- contradicting the registry's claimed five-state
                   outcome.
    the inspection a third listing, so what a reviewer read was not what either of the
                   others used.

That is the same failure as the three disease-state vocabularies, one layer along. This is
the single source: every consumer calls `project()` and renders or scores its result.

WHAT A FUTURE CONTAINS, in the canonical vocabulary:

    states        no_active_disease / responding / stable / progressing / unknown, with
                  the months spent in each over the DEFINED window
    transitions   entries INTO each state, which occupancy cannot express -- twelve months
                  of stability and a year of alternating response and progression can
                  occupy identical state fractions
    new sites     first documented NEW distant metastasis, timed, from the curated block
                  and independent of whether anyone curated a response. Not "relapse":
                  for a patient already metastatic this is spread, not recurrence
    treatment     regimens started, and months on treatment
    terminal      death, or the censoring that ended observation

CENSORING IS EXPLICIT. A patient lost to follow-up at month 18 of a 24-month horizon is
NOT alive at 24 months; they are unobserved. Testing only death returned alive=1 and
months_alive=24 for exactly that patient.

GOVERNANCE. GENIE-BPC under DUA. Aggregate output only, gitignored.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import genie_operator_registry_v2 as GV2  # noqa: E402
from synthonccases import disease_state as DS  # noqa: E402


@dataclass(frozen=True)
class Future:
    """What happened after one landmark, in the canonical vocabulary."""

    window_end: float                       # months from the landmark
    window_reason: str                      # complete / closed by death / censored
    alive_at_horizon: float                 # 1, 0, or NaN when censored before it
    months_alive: float                     # NaN when censored before the horizon
    months_in: dict[str, float] = field(default_factory=dict)
    entries_into: dict[str, int] = field(default_factory=dict)
    #: Whether the source registry collects per-site metastasis detail for this cohort at
    #: all. BRCA does not. Without this flag, "no site collected" and "no new site" both
    #: rendered as an empty list, and every consumer scored the real BRCA coordinate as
    #: zero new metastases against generated named sites -- a structural absence coded as
    #: a clinical fact. Consumers must score the site channel only when this is True.
    #: Defaults to True so a patient-shaped view built from a generated ledger (which can
    #: always name sites) needs no change; the REAL side decides comparability.
    sites_collected: bool = True
    #: The first NEW distant site after the landmark. Not "relapse": for a patient
    #: already metastatic at the landmark this is anatomical spread or progression, and
    #: calling it relapse would assert a disease-free interval that never existed.
    first_new_distant_site: float | None = None
    new_distant_sites: list[tuple[float, str]] = field(default_factory=list)
    path: list[tuple[float, float, str]] = field(default_factory=list)
    regimens: list[tuple[float, str, float]] = field(default_factory=list)
    subsequent_regimens: list[tuple[float, str, float]] = field(default_factory=list)
    months_on_treatment: float = 0.0
    #: TREATMENT EVOLUTION, not just its total. Counting months on treatment and the
    #: number of subsequent regimens says nothing about WHAT was given or in what order,
    #: so a model that answered "three more lines" with three arbitrary drugs scored
    #: identically to one that reproduced the actual sequence.
    modality_sequence: list[tuple[float, tuple[str, ...]]] = field(default_factory=list)
    months_on_modality: dict[str, float] = field(default_factory=dict)
    months_to_first_switch: float = float("nan")
    terminal: str = ""


#: How long a documented state stands. Two conventions, and which is PRIMARY is a clinical
#: judgement, not a detail:
#:
#:   under_followup        The last documented state persists until a recorded event
#:                         changes it, for as long as follow-up continues. The detection
#:                         date is the change point. This is how a chart is read: stable at
#:                         month 0 and progressing at month 9 means stable until
#:                         progression was documented, not five unknowable months.
#:                         Follow-up ending is already handled -- `defined_window` closes
#:                         the window at death or last contact -- so nothing else is
#:                         needed, and there is no arbitrary constant here at all.
#:   assessment_validity   Every assessment expires after STATUS_VALIDITY. Conservative,
#:                         and it manufactures unknown time in a continuously followed
#:                         patient, so it is the SENSITIVITY analysis.
#:
#: An earlier attempt put a twelve-month limit here and cancelled it whenever any later
#: contact fell inside the window. That was wrong twice over: the constant was arbitrary,
#: and one contact cancelled expiry PERMANENTLY -- a stable assessment at month 0 with a
#: single treatment contact at month 1 held "stable" all the way to month 24. Part of the
#: 0%-unknown result came from that bug rather than from the convention.
PERSISTENCE = {"under_followup": float("inf"),
               "assessment_validity": GV2.STATUS_VALIDITY}
PRIMARY_PERSISTENCE = "under_followup"
#: Resolved at CALL time, so a sensitivity run can switch it. A default bound in the
#: function signature was fixed at import, and the "conservative" run silently kept
#: unlimited persistence while its docstring promised four-month expiry.
DEFAULT_PERSISTENCE = PRIMARY_PERSISTENCE

#: HOW THE RECORD IS READ. Approved 2026-09-08, corrected after review the same day.
#:
#:   chronological   Start from the documented state AT THE LANDMARK -- the same
#:                   classification the prompt was given (`genie_state`), so the course is
#:                   scored from what the model was told, not from a stale record the
#:                   prompt had already called unknown. Carry it forward until a later
#:                   disease record updates it. The most recent record decides, in both
#:                   directions and strictly later; records with the same timestamp that
#:                   contradict each other are a conflict. Repeated confirmations of an
#:                   unchanged state are not events. Observation stops at death or last
#:                   follow-up. NOTHING IS READ BACKWARD: a first record at month 5 does
#:                   not tell us months 0-5. That was tried and withdrawn -- it erased
#:                   the very transitions we score, and made the first months of a course
#:                   depend on the horizon requested.
#:                   `unknown` remains where the data cannot say: no established starting
#:                   state, or a genuine contradiction. Those intervals are NOT SCORABLE for
#:                   disease state -- missing documentation, not a fifth kind of cancer --
#:                   and the comparison is made over the time scorable on both sides.
#:   conservative    The earlier behaviour, kept as the sensitivity analysis: conflicts
#:                   named and left unknown, four-month validity, no chronological update.
#:
#: The rule applies to SCORING on both sides and never touches the prompts. Real records
#: are DETECTION times; generated events are asserted onsets; all disease-state
#: comparisons are therefore made on the real assessment calendar.
RECORD_RULES = ("chronological", "conservative")
RECORD_RULE = "chronological"
#: Same recorded time, to the day.
SAME_TIME = 1e-6


def state_path(p, tau: float, end: float,
               persistence: str | None = None) -> list[tuple[float, float, str]]:
    """The ORDERED path of state intervals: (start, stop, state), in time order.

    STALENESS, NOT DECAY. Marks were once added only when a new assessment arrived, never
    when the record stopped supporting one, so a single stable assessment at month 1 held
    stability for 23 months. The correction was to expire every assessment at
    STATUS_VALIDITY -- which overshot: in a closely followed academic cohort, a state
    documented at month 0 and unchanged at month 9, on unaltered treatment, was not unknown
    in between. It was stable until progression was DETECTED. See `PERSISTENCE`: the
    primary convention keeps a state alive while the patient is still being seen, and the
    four-month rule is retained as the conservative sensitivity analysis.

    WHAT COUNTS AS BEING SEEN. Any contact -- a response assessment, a presence note, or a
    treatment record -- because a patient receiving therapy is under observation whether or
    not anyone curated a response that month. Judging staleness on assessments alone was
    what put unknown months inside active treatment.

    ORDER. Occupancy alone cannot say whether a patient responded then progressed, or
    progressed then stabilised -- and the whole design rests on individual courses
    carrying joint temporal structure. The path is the primary object; occupancy and
    transition counts are DERIVED from it, and the card renders it in order.
    """
    if persistence is None:
        persistence = DEFAULT_PERSISTENCE
    if persistence not in PERSISTENCE:
        raise ValueError(f"unknown persistence convention {persistence!r}")
    validity = PERSISTENCE[persistence]
    mets = p.context.get("mets")
    assessments = [w for w, _v in p.presence] + [w for w, _v in p.status]
    # A STATE EXPIRES ONLY UNDER THE CONSERVATIVE CONVENTION. Under the primary one it is
    # simply carried until the next recorded event changes it, and the window's own end --
    # death, or last contact -- is what stops it. No contact-renewal rule, so there is no
    # way for a single early contact to license a state indefinitely.
    # Under the sensitivity convention, SITE EVENTS EXPIRE TOO. Expiry marks were placed
    # for assessments only, so a site-only progression never lapsed under the four-month
    # rule while every assessment did -- a regression introduced with the timestamp fix.
    site_times = [m for m, _s in (mets.sites if mets else [])]
    stale = ({w + validity + 1e-6 for w in assessments + site_times
              if tau < w + validity + 1e-6 < end}
             if np.isfinite(validity) else set())
    # THE DOCUMENTED STATE AT THE LANDMARK, as the prompt classified it, and the time of
    # the first record after it. Records at or before tau are summarised by `genie_state`
    # (four-month validity, conflicts named); everything after is read chronologically.
    landmark = GV2.genie_state(p, tau, mets)
    post = [w for w in assessments if w > tau + 1e-9] + \
        [m for m, _s in (mets.sites if mets else []) if m > tau + 1e-9]
    first_record = min(post) if post else float("inf")
    # INFORMATION DISCARDED AT THE LANDMARK STAYS DISCARDED. `genie_state` accepts records
    # within STATUS_VALIDITY of the landmark and no older; after the landmark, a later
    # note updates only the aspect it carries. Without this, a month-2 note saying only
    # "cancer present" revived a stale "progressing" assessment from seven months before
    # treatment and scored the course as progressing from month 2 on. Under the
    # chronological rule every lookup below sees only carried-in and post-landmark records.
    if RECORD_RULE == "chronological":
        carry_from = tau - GV2.STATUS_VALIDITY - 1e-9
        presence_rec = [(w, v) for w, v in p.presence if w >= carry_from]
        status_rec = [(w, v) for w, v in p.status if w >= carry_from]
        site_rec = [(m, s_) for m, s_ in (mets.sites if mets else []) if m >= carry_from]
    else:
        presence_rec, status_rec = list(p.presence), list(p.status)
        site_rec = list(mets.sites) if mets else []
    marks = sorted({tau}
                   | {w for w in assessments if tau < w < end}
                   # JUST AFTER the limit, not at it. `recent()` accepts an assessment
                   # exactly that old, so a mark placed at w + validity re-evaluated to
                   # the same state and the transition never took effect -- a no-evidence
                   # assessment at month 9.1 held remission to month 14.9.
                   | set(stale)
                   | {m for m, _s in (mets.sites if mets else []) if tau < m < end})
    path: list[tuple[float, float, str]] = []
    for i, when in enumerate(marks):
        stop = marks[i + 1] if i + 1 < len(marks) else end
        if stop <= when:
            continue
        # RESOLVED BY TIMESTAMP, mirroring the prefix reader `genie_state`. A window
        # boolean let a site documented at month 2 force "progressing" through month 12
        # while an assessment at month 3 said responding -- older evidence overriding
        # newer. A new site only speaks louder if it is NEWER than the last status look.
        # Its reach is the convention's validity: unbounded under the primary rule (an
        # event persists until a newer record changes it), four months under the
        # sensitivity rule -- which is exactly the prefix reader's window.
        last_look = max([w for w, _v in status_rec if w <= when + 1e-9] + [-np.inf])
        newest_site = max([m for m, _s in site_rec
                           if when - validity <= m <= when + 1e-9] + [-np.inf])
        new_site = bool(np.isfinite(newest_site) and newest_site >= last_look)
        # THE LOOKUP WINDOW MUST MATCH THE MARK RULE, or the two disagree: marks placed
        # under a twelve-month convention would be re-evaluated by a four-month `recent()`
        # and every one of them would read `unknown` anyway.
        presence_now = GV2.recent(presence_rec, when, validity)
        status_now = GV2.recent(status_rec, when, validity)
        # ... and under the finite-validity sensitivity convention the landmark state
        # itself lapses after `validity`, so the seed applies only while it is current.
        if RECORD_RULE == "chronological" and when < first_record - 1e-9 and \
                (not np.isfinite(validity) or when < tau + validity):
            # BEFORE THE FIRST POST-LANDMARK RECORD the state is the landmark
            # classification the prompt used -- not a stale pre-landmark assessment
            # revived by infinite persistence.
            path_state = landmark.state
            if path and path[-1][2] == path_state:
                path[-1] = (path[-1][0], stop - tau, path_state)
            else:
                path.append((when - tau, stop - tau, path_state))
            continue
        if RECORD_RULE == "chronological" and presence_now == "no evidence":
            t_ned = max([w for w, _v in presence_rec
                         if when - validity <= w <= when + 1e-9] + [-np.inf])
            active = status_now in ("responding", "stable", "progressing", "mixed")
            t_other = max(last_look if active else -np.inf, newest_site)
            if t_ned > t_other + SAME_TIME:
                # the NED note is strictly the most recent record: it supersedes the
                # older progression / site records
                status_now, new_site = None, False
            elif t_other > t_ned + SAME_TIME:
                # a strictly later status or new site supersedes the older NED note:
                # disease is present again, in the state that record says
                presence_now = "evidence"
            # same recorded time: left to `from_genie`, which names the conflict
        st = DS.from_genie(presence=presence_now, curated_status=status_now,
                           new_distant_site=new_site)
        # Merge on construction: consecutive identical states are one interval, and
        # zero-length marks (an assessment and a site on the same day) are dropped. A path
        # with duplicate adjacent states would inflate the transition counts derived from
        # it.
        if path and path[-1][2] == st.state:
            path[-1] = (path[-1][0], stop - tau, st.state)
        else:
            path.append((when - tau, stop - tau, st.state))
    path = [(a, b, st) for a, b, st in path if b - a > 1e-6]
    return path


def project(p, tau: float, horizon: float) -> Future:
    """The one canonical future. Every consumer reads this."""
    _a, end, reason = DS.defined_window(tau, horizon, p.death, p.last_contact)
    horizon_end = tau + horizon
    died = p.death <= horizon_end

    # CENSORING. Testing only death scored a patient lost at month 18 as alive at 24 with
    # 24 months of survival. Survival is unknown when observation stopped first.
    censored_early = (not died) and p.last_contact < horizon_end
    alive = np.nan if censored_early else (1.0 if not died else 0.0)
    months_alive = (np.nan if censored_early
                    else float(min(p.death, horizon_end) - tau))

    # THE PATH IS PRIMARY. Occupancy and transitions are derived from it, so they cannot
    # disagree with what the card renders.
    path = state_path(p, tau, end)
    months_in = dict.fromkeys(DS.State.__args__, 0.0)
    entries: dict[str, int] = dict.fromkeys(DS.State.__args__, 0)
    # THE STATE AT THE LANDMARK IS NOT A NEW EVENT. Starting from None counted the first
    # segment as an entry, so a patient already progressing when treatment began scored
    # n_new_progressions = 1 with nothing having happened.
    previous = path[0][2] if path else None
    for start, stop, st in path:
        months_in[st] = months_in.get(st, 0.0) + (stop - start)
        if st != previous:
            entries[st] = entries.get(st, 0) + 1
        previous = st

    # ALL FUTURE WINDOWS END AT THE CENSORING-DEFINED END, not the nominal horizon. A
    # patient censored at month 8 cannot accrue a metastasis at month 20.
    mets = p.context.get("mets")
    new_sites = [(m - tau, s) for m, s in (mets.sites if mets else [])
                 if tau < m <= end]
    # INCLUDING the index regimen. How long the patient stayed on the treatment being
    # asked about is a core outcome, and taking only regimens starting strictly after tau
    # dropped it -- the card then read "no further systemic treatment recorded" for a
    # patient who was on the index drug for eleven months.
    regimens = [(ln["start"] - tau, ln["label"], ln["stop"] - ln["start"])
                for ln in p.lines if tau - 1e-9 <= ln["start"] <= end]
    # SUBSEQUENT means after the index regimen, which starts AT tau. Counting the index
    # one made every patient look as though they had switched at least once.
    subsequent = [r for r in regimens if r[0] > 1e-9]
    # UNION of intervals, not their sum. Overlapping regimens -- which is most combination
    # therapy -- made a twelve-month and a six-month course report eighteen months on
    # treatment.
    spans = [(max(ln["start"], tau), min(ln["stop"], end)) for ln in p.lines
             if ln["stop"] > tau and ln["start"] < end]
    on_tx = GV2.M.overlap(GV2.M.merge(spans), tau, end) if spans else 0.0

    # The modality sequence and per-modality occupancy. Overlapping by design: twelve
    # months of chemotherapy WITH a biologic is twelve months in both channels, and the
    # channels must never be summed into a total.
    seq: list[tuple[float, tuple[str, ...]]] = []
    by_mod: dict[str, list[tuple[float, float]]] = {}
    for start, label, dur in regimens:
        mods = tuple(sorted(GV2.modalities(label)))
        if not seq or seq[-1][1] != mods:
            seq.append((start, mods))
        for m in mods:
            by_mod.setdefault(m, []).append((start, min(start + dur, end - tau)))
    # UNION within each modality, not a sum. Two overlapping chemotherapy regimens gave
    # eighteen chemotherapy-months in a twelve-month window. Channels still OVERLAP ACROSS
    # modalities by design -- a chemo-plus-biologic month counts in both -- and must never
    # be summed into a total.
    on_mod = {m: GV2.M.overlap(GV2.M.merge(v), 0.0, end - tau)
              for m, v in by_mod.items()}
    first_switch = seq[1][0] if len(seq) > 1 else float("nan")

    terminal = (f"death at month {p.death - tau:.0f}" if died
                else f"censored at month {p.last_contact - tau:.0f}" if censored_early
                else f"alive at month {horizon:.0f}")
    return Future(window_end=end - tau, window_reason=reason, alive_at_horizon=alive,
                  months_alive=months_alive, months_in=months_in, entries_into=entries,
                  sites_collected=bool(mets is not None
                                       and getattr(mets, "site_detail_available", True)),
                  first_new_distant_site=(new_sites[0][0] if new_sites else None),
                  new_distant_sites=new_sites, path=path, regimens=regimens,
                  subsequent_regimens=subsequent,
                  months_on_treatment=float(max(0.0, on_tx)),
                  modality_sequence=seq, months_on_modality=on_mod,
                  months_to_first_switch=first_switch, terminal=terminal)


def render(fut: Future, canonical) -> list[str]:
    """The future as the card shows it. Same object the scorer reads, so a reviewer and
    the metric can never be looking at different things."""
    out = []
    if fut.regimens:
        out.append("Treatment course: " + "; ".join(
            f"{canonical(lbl)} months {s:.0f}-{min(s + d, fut.window_end):.0f}"
            for s, lbl, d in fut.regimens))
    else:
        out.append("Treatment course: no systemic treatment recorded")
    # THE ORDERED PATH, not a ranking by total time. "Responding 12 months; progressing 8"
    # cannot say whether the patient responded then progressed or the reverse, and the
    # temporal structure of the individual course is the whole point of showing it.
    runs = []
    for start, stop, st in fut.path:
        if runs and runs[-1][2] == st:
            runs[-1][1] = stop
        else:
            runs.append([start, stop, st])
    # One decimal, because integer months rendered genuine sub-month intervals as
    # "months 7-7". And a cap, because a fortnightly-scanned patient can produce thirteen
    # segments and the tail is churn rather than trajectory -- the count is kept so
    # nothing is silently dropped.
    shown = [f"{st.replace('_', ' ')} {a:.1f}-{b:.1f}" for a, b, st in runs[:8]]
    tail = (f", then {len(runs) - 8} further change{'' if len(runs) - 8 == 1 else 's'}"
            if len(runs) > 8 else "")
    label = ("Disease course (compact sketch; months from start)" if len(runs) > 8
             else "Disease course (months from start)")
    out.append(f"{label}: " + (" -> ".join(shown) + tail if runs
                               else "no documented disease state in this window"))
    # NOT "relapse". For a patient already metastatic at the landmark this is spread, and
    # the card renders both populations.
    out.append("New distant sites: " + ("; ".join(
        f"{s} at month {m:.0f}" for m, s in fut.new_distant_sites)
        if fut.new_distant_sites else "none documented"))
    out.append(f"Outcome: {fut.terminal}")
    return out
