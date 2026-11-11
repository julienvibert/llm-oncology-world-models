#!/usr/bin/env python3
"""A GENERATED ledger, projected onto the same observable ruler as a real patient.

Without this the four-arm run produces text nobody can score. "36 calls" would mean 36
billed attempts, not 36 comparable futures, and the difference only becomes visible after
the money is spent.

THE RULER MUST BE ONE RULER. The real side is measured by `canonical_future_v1.project()`:
a five-state path with STATUS_VALIDITY expiry, occupancy derived from the path, entries
derived from the path, a window closed by death and left open by censoring. A generated
trajectory scored by any second implementation of "the same" projection would be compared
against the real one across a definition gap, and every disagreement would be
uninterpretable -- is the model wrong, or are the two rulers?

So this file does not reimplement the projection. It builds a patient-SHAPED object out of
the generated events and hands it to the very same `project()`. The two sides then share
one operator by construction, not by review.

    progression event   -> a disease assessment reading "progressing" on that date
    response event      -> a disease assessment reading "responding" on that date
    death event         -> closes the window
    last_followup       -> censors it
    treatment events    -> the regimen channel, not the disease channel

THE GENERATED SIDE HAS NO UNKNOWNS, AND MUST NOT BE GIVEN ANY. The contract asks the model
to narrate what HAPPENS, to death. An emitted event is a fact of that world, not a scan
result, and silence between events means "nothing changed" -- not "nobody looked". An
earlier version of this file ran the generated events through the real side's
STATUS_VALIDITY expiry, so a model that said nothing for five months was scored as five
months of `unknown`. That invented an observation gap out of the model's brevity, and made
generated `unknown` time a property of how chatty the model is rather than of the disease.

THE ASYMMETRY IS THE OTHER WAY, and it is the real reason M matters here:

    generated   a step function, known at every instant, because it was asserted
    real        observed at scan times, stale after STATUS_VALIDITY, genuinely unknown
                in between

Compare those directly and the real patient shows `unknown` months the generated one never
can. That difference is measurement, not biology, and it would be read as a model error.

So there are two projections, and both are needed for different questions:

    TRUTH (default, `observation_times=None`). The trajectory as the model asserted it:
    carry-forward, no expiry. This is the model's claim, and it is what a clinician reads.

    AS OBSERVED (`observation_times=[...]`). The same trajectory sampled at the REAL
    patient's assessment times, and only then allowed to go stale -- so both sides pass
    through one measurement process and their occupancy is comparable. This is the
    projection the arm-vs-arm comparison must use. It spends the real follow-up schedule at
    scoring time; that is never shown to the model, so it is not a leak.

    A caveat that belongs in the caption rather than in code: a real patient's assessment
    times stop when they die, so masking scores the generated course only over the window
    the real one was followed. Survival is compared separately and directly, precisely
    because it must not inherit that window.

Raw responses are saved, so either projection can be recomputed without generating again.

GOVERNANCE. GENIE-BPC under DUA. This module reads generated text and a landmark state; it
holds no patient-level real data of its own.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import canonical_future_v1 as CF  # noqa: E402
import genie_operator_registry_v2 as GV2  # noqa: E402
from synthonccases import disease_state as DS  # noqa: E402

#: Months per day, matching the registry convention used throughout the real side.
DAYS = 30.4375

#: Ledger event types that assert something about the DISEASE, and what they assert.
DISEASE_EVENTS = {"progression": "progressing", "response": "responding"}


class NotScoreable(ValueError):
    """The ledger validated but cannot be placed on the observable ruler."""


@dataclass
class _Mets:
    """The curated-metastasis shape `project()` reads new distant sites from."""

    sites: list


@dataclass
class _AsPatient:
    """A patient-shaped view of a generated trajectory, for `project()` to consume.

    Exactly the surface `state_path`/`project` touch -- presence, status, death,
    last_contact, context["mets"], lines -- because a channel left empty here is a column
    the comparison silently scores as zero rather than as missing.
    """

    pid: str
    status: list
    presence: list
    death: float
    last_contact: float
    context: dict
    lines: list
    terminal_kind: str = "truncated"


def _months(when: str | date, origin: date) -> float:
    d = when if isinstance(when, date) else date.fromisoformat(str(when)[:10])
    return (d - origin).days / DAYS


class _Burden:
    """The patient's disease across ALL sites, updated event by event.

    THE PROMPT ASKS FOR CHANGED SITES ONLY, so one event is not a snapshot. Reading each
    event on its own turned "the liver lesion cleared" into a complete response while the
    lung was still involved and simply not mentioned -- 13.5 months scored disease-free for
    a patient with active disease, which is the error that most flatters the model.

    So the map is CUMULATIVE: seeded with what was active at the landmark, then each event
    overwrites only the sites it names. Complete response is reached when every site the
    map knows about has cleared.

    WHEN THE BASELINE IS UNKNOWN, remission is not concluded. If we never learned what was
    active at the landmark, an all-clear event tells us those sites cleared -- not that
    nothing else remains. Presence is then reported only when it can be asserted safely: an
    active band means disease is present regardless of what else is unknown, while
    "no evidence" is withheld.
    """

    #: Presence readings that establish a baseline at all. `equivocal` does NOT: it says
    #: the record could not tell, so clearing one mentioned lesion afterwards licensed a
    #: complete response out of an unresolved starting point.
    DEFINITE = ("evidence", "no evidence")

    def __init__(self, presence: str | None, sites: list[str] | None,
                 extent: str | None = None) -> None:
        self.by_site: dict[tuple[str, str], str] = {}
        self.known = presence in self.DEFINITE
        if presence == "evidence":
            if sites:
                # SEEDED FROM WHAT THE PREFIX ACTUALLY SAYS, which is also what the model
                # was told -- holding it to a map it never saw would be scoring a standard
                # it could not meet.
                for s in sites:
                    self.by_site[("_baseline", s)] = "moderate"
            elif extent == "no":
                # Local disease with nothing distant on record: the disease IS the primary,
                # and `primary_site` is both the clinical reading and a site the model can
                # clear. An opaque sentinel here would leave one query in five unable to
                # reach complete response no matter what the model emitted.
                self.by_site[("_baseline", "primary_site")] = "moderate"
            else:
                # Metastatic (or unknown extent) with no site named -- stage IV can be
                # assigned on criteria the curated site block does not enumerate. We know
                # disease is present and NOT where, so completeness cannot be checked and
                # remission must not be concluded.
                self.known = False

    def apply(self, changes: list) -> tuple[str | None, list[str]]:
        """Fold one event's disease block in. Returns (presence, newly appearing sites)."""
        new_sites, touched = [], False
        for c in changes if isinstance(changes, list) else []:
            if not isinstance(c, dict):
                continue
            band = str(c.get("burden") or "")
            site = str(c.get("site_class") or c.get("site") or "").strip() or "_unnamed"
            kinetic = str(c.get("kinetic") or "")
            if kinetic == "cleared":
                band = "absent"
            # KEYED ON (CLONE, SITE), because the contract emits changes per clone and per
            # site. Keying on site alone let one clone's entry overwrite another's IN THE
            # SAME EVENT: C0 growing in liver followed by C1 cleared in liver read as no
            # evidence, and the same two entries in the other order read as evidence. A
            # clinical state that depends on JSON ordering is not a measurement.
            clone = str(c.get("clone") or "_uncloned")
            self.by_site[(clone, site)] = band
            # A NAMED CLONE SUPERSEDES THE UNNAMED BASELINE at that site: the seed stands
            # in for "something active here", and a report about that site is a report
            # about that something.
            self.by_site.pop(("_baseline", site), None)
            touched = True
            if kinetic == "appearing" and site not in ("primary_site", "unknown",
                                                       "_unnamed"):
                new_sites.append(site)
        if not touched:
            return None, new_sites
        if any(b in DS.ACTIVE_BANDS for b in self.by_site.values()):
            return "evidence", new_sites
        # everything we know of has cleared -- but that is only remission if we knew the
        # whole picture to begin with
        return ("no evidence" if self.known else None), new_sites


def as_patient(events: list[dict], *, cutoff: str | date, landmark_state: str,
               horizon: float, observation_times: list[float] | None = None,
               index_regimen: str | None = None,
               landmark_presence: str | None = None,
               landmark_sites: list[str] | None = None,
               landmark_extent: str | None = None) -> _AsPatient:
    """Turn a validated future ledger into something `project()` can measure.

    EVERY channel the real side scores, or the comparison is against an empty column.
    Mapping only progression/response left presence unset (so a patient starting in
    no_active_disease projected as unknown), complete response unreachable, new metastatic
    sites absent, the index regimen missing, and every generated line zero-length -- so a
    response could pass "scoreable" with several outcome blocks empty or wrong.

    `landmark_state` is the state the patient is IN at the cutoff, taken from the prefix.
    Without it the path would start from whatever the model first mentions.
    """
    origin = cutoff if isinstance(cutoff, date) else date.fromisoformat(str(cutoff)[:10])
    status: list[tuple[float, str]] = []
    presence: list[tuple[float, str]] = []
    sites: list[tuple[float, str]] = []
    death = np.inf
    censored_at: float | None = None
    last_event = 0.0
    lines: list[dict] = []

    # THE LANDMARK STATE HOLDS FROM TIME ZERO until the model says otherwise. On the real
    # side it would expire; here it is the state the patient is IN when the narration
    # starts, and the narration is what changes it.
    if landmark_state in DS.State.__args__ and landmark_state != "unknown":
        status.append((0.0, landmark_state))
    # PRESENCE AT THE LANDMARK, likewise. `no_active_disease` is derived from presence, so
    # leaving presence empty made it unreachable no matter what the model emitted.
    seed_presence = landmark_presence
    if not seed_presence:
        # The derived state implies presence when it was not recorded directly.
        if landmark_state == "no_active_disease":
            seed_presence = "no evidence"
        elif landmark_state in ("responding", "stable", "progressing"):
            seed_presence = "evidence"
    if seed_presence:
        presence.append((0.0, seed_presence))
    # THE RUNNING PICTURE, seeded from the landmark. Each event names only what CHANGED,
    # so a per-event reading cannot tell "everything cleared" from "the one thing
    # mentioned cleared".
    burden = _Burden(seed_presence, landmark_sites, landmark_extent)

    # THE INDEX REGIMEN IS ALREADY RUNNING at the cutoff -- that is what the landmark IS.
    # Omitting it made months-on-treatment start from the first generated switch, and a
    # patient who stayed on the index drug for eleven months scored zero.
    if index_regimen:
        lines.append({"start": 0.0, "stop": float(horizon), "label": index_regimen,
                      "_open": True})

    for ev in events:
        try:
            when = _months(ev["date"], origin)
        except (KeyError, ValueError) as exc:
            raise NotScoreable(f"event with an unreadable date: {ev!r}") from exc
        last_event = max(last_event, when)
        kind = ev.get("type")
        reading, new_sites = burden.apply(ev.get("disease") or [])
        if reading:
            presence.append((when, reading))
        for s in new_sites:
            sites.append((when, s))
        if kind in DISEASE_EVENTS:
            # A CLEARED BURDEN OUTRANKS THE EVENT TYPE. "response" with everything absent
            # is a complete response, and calling it `responding` would put a disease-free
            # patient in a state that asserts residual disease.
            status.append((when, "no_active_disease" if reading == "no evidence"
                           else DISEASE_EVENTS[kind]))
        elif kind == "death":
            death = when
        elif kind == "last_followup":
            censored_at = when
        elif kind in ("systemic_treatment_start", "systemic_treatment_switch"):
            # A SWITCH CLOSES THE LINE IT SWITCHES FROM. Leaving the previous line open
            # double-counted treatment time; leaving new lines at zero duration reported
            # every subsequent regimen as instantaneous.
            for ln in lines:
                if ln.get("_open"):
                    ln["stop"] = when
                    ln["_open"] = False
            lines.append({"start": when, "stop": float(horizon),
                          "label": ((ev.get("regimen") or {}).get("label") or ""),
                          "_open": True})
        elif kind == "systemic_treatment_stop":
            for ln in lines:
                if ln.get("_open"):
                    ln["stop"] = when
                    ln["_open"] = False

    # THE TERMINAL RULE. `last_contact` used to default to the requested horizon, so a
    # ledger that simply stopped at month 5 -- no death, no censoring -- was scored as
    # OBSERVED AND ALIVE at month 24. That silently converts the model running out of
    # output into nineteen months of survival it never claimed, on the endpoint most likely
    # to be reported.
    #
    #   death           the course is complete; the window closes there
    #   last_followup   explicitly censored; usable only where outcomes are observable
    #   neither         TRUNCATED. Observation ends at the last event, and every horizon
    #                   beyond it is unresolved rather than favourable.
    if np.isfinite(death):
        terminal_kind, last_contact = "death", death
    elif censored_at is not None:
        terminal_kind, last_contact = "censored", censored_at
    else:
        terminal_kind, last_contact = "truncated", last_event

    for ln in lines:
        ln["stop"] = min(ln["stop"], last_contact if np.isfinite(last_contact) else horizon)
        ln.pop("_open", None)

    status.sort(key=lambda x: x[0])
    presence.sort(key=lambda x: x[0])
    if observation_times is None:
        # TRUTH MODE. An asserted state persists as FACT until the model changes it, so it
        # must not decay. `state_path` is the real side's operator and applies expiry by
        # construction -- correctly, for a record of scans. Rather than fork it and end up
        # with two definitions of the ruler, the assertion is simply restated often enough
        # that staleness never bites. Same operator, no invented unknowns.
        status = _restate(status, horizon, min(death, last_contact))
        presence = _restate(presence, horizon, min(death, last_contact))
    else:
        # AS OBSERVED. Sampled where the real patient was actually assessed; expiry then
        # applies to those readings exactly as it does on the real side, so both pass
        # through one measurement process.
        status = _resample(status, sorted(observation_times))
        presence = _resample(presence, sorted(observation_times))
    status.sort(key=lambda x: x[0])
    presence.sort(key=lambda x: x[0])
    if np.isfinite(death):
        last_contact = min(last_contact, death)
    p = _AsPatient(pid="generated", status=status, presence=presence, death=death,
                   last_contact=float(last_contact), context={}, lines=lines)
    # NEW DISTANT SITES, in the shape `project()` reads them from -- the same
    # `context["mets"]` slot the real side fills from the curated block.
    p.context["mets"] = _Mets(sites=sorted(sites))
    p.terminal_kind = terminal_kind
    return p


def _restate(status: list[tuple[float, str]], horizon: float,
             end: float) -> list[tuple[float, str]]:
    """Repeat each asserted state often enough that the expiry rule never fires.

    The generated trajectory is a step function the model asserted; it does not go stale.
    Restating at half STATUS_VALIDITY keeps `state_path` -- which is built for scans and
    expires them -- reporting that step function unchanged, without a second copy of the
    path logic existing anywhere.
    """
    if not status:
        return []
    step = GV2.STATUS_VALIDITY / 2.0
    stop = min(horizon, end if np.isfinite(end) else horizon)
    out: list[tuple[float, str]] = []
    for i, (when, state) in enumerate(status):
        nxt = status[i + 1][0] if i + 1 < len(status) else stop
        t = when
        while t < min(nxt, stop) or not out or out[-1][0] < when:
            out.append((t, state))
            t += step
            if t >= min(nxt, stop):
                break
    return out


def _resample(status: list[tuple[float, str]], when: list[float]) -> list[tuple[float, str]]:
    """The generated state, read at someone else's observation times.

    Carry-forward with no expiry HERE on purpose: expiry is applied once, downstream, by
    `state_path`. Applying it twice would penalise the generated side for a gap in the real
    patient's follow-up as well as for its own silence.
    """
    out = []
    for t in when:
        prior = [s for w, s in status if w <= t + 1e-9]
        if prior:
            out.append((t, prior[-1]))
    return out


def project(events: list[dict], *, cutoff: str | date, landmark_state: str,
            horizon: float, observation_times: list[float] | None = None,
            index_regimen: str | None = None,
            landmark_presence: str | None = None,
            landmark_sites: list[str] | None = None,
            landmark_extent: str | None = None) -> CF.Future:
    """The generated future, on the real side's ruler. Raises NotScoreable if it cannot be."""
    p = as_patient(events, cutoff=cutoff, landmark_state=landmark_state, horizon=horizon,
                   observation_times=observation_times, index_regimen=index_regimen,
                   landmark_presence=landmark_presence,
                   landmark_sites=landmark_sites,
                   landmark_extent=landmark_extent)
    fut = CF.project(p, 0.0, horizon)
    # A PATH OF LENGTH ZERO IS NOT A FUTURE. It means nothing observable was asserted and
    # nothing was carried in, so there is no trajectory to compare -- better refused here
    # than averaged in as a run of zeros.
    if not fut.path:
        raise NotScoreable("the ledger yields no observable state interval")
    return fut


def scoreability(events: list[dict], *, cutoff: str | date, landmark_state: str,
                 horizons: tuple[float, ...], index_regimen: str | None = None,
                 landmark_presence: str | None = None,
                 landmark_sites: list[str] | None = None,
                 landmark_extent: str | None = None) -> dict:
    """What this trajectory is scoreable FOR, per horizon -- the runner's acceptance test.

    Reported per horizon rather than as one verdict for the same reason the PFS runner
    records `pfs_valid` separately from `full_course_valid`: a course that resolves 3 and 6
    months and is cut off before 24 is a usable datum at two horizons, and discarding it
    whole would drop short courses and long ones at different rates.
    """
    out: dict = {"scoreable_at": {}, "failure": None}
    try:
        for h in horizons:
            fut = project(events, cutoff=cutoff, landmark_state=landmark_state, horizon=h,
                          index_regimen=index_regimen,
                          landmark_presence=landmark_presence,
                          landmark_sites=landmark_sites,
                          landmark_extent=landmark_extent)
            out["scoreable_at"][h] = {
                "window_end": fut.window_end, "window_reason": fut.window_reason,
                "alive_at_horizon": (None if not np.isfinite(fut.alive_at_horizon)
                                     else float(fut.alive_at_horizon)),
                "n_state_intervals": len(fut.path),
                "entries_into_progressing": fut.entries_into.get("progressing", 0),
                "months_unknown": round(fut.months_in.get("unknown", 0.0), 3),
                "months_no_active_disease": round(
                    fut.months_in.get("no_active_disease", 0.0), 3),
                "months_on_treatment": round(fut.months_on_treatment, 3),
                "n_regimens": len(fut.regimens),
                "n_new_distant_sites": len(fut.new_distant_sites),
            }
        # THE TERMINAL KIND, recorded once for the whole trajectory: a course that simply
        # stopped is not the same datum as one that reached death or explicit censoring,
        # and averaging the three together is how a truncated response becomes survival.
        out["terminal_kind"] = as_patient(
            events, cutoff=cutoff, landmark_state=landmark_state,
            horizon=max(horizons), index_regimen=index_regimen,
            landmark_presence=landmark_presence,
            landmark_sites=landmark_sites,
            landmark_extent=landmark_extent).terminal_kind
    except NotScoreable as exc:
        out["failure"] = str(exc)
    return out
