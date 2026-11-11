#!/usr/bin/env python3
"""Versioned readers for the four-arm scoring entry point (corrections R1-R5, 12 Oct 2026).

Used ONLY by `scripts/four_arm_eligibility_v2.py` and `scripts/score_four_arm_final_v1.py`.
The shared exploratory modules (`generated_future_v1`, `canonical_future_v1`,
`genie_operator_registry_v2`) are read, never edited: earlier outputs depend on them.
Definitions are in `manuscript/SCORING_SPEC_amendment_2026-10-12.md`.

WHAT THIS MODULE REPLACES, AND WHY

R1  Generated disease. `generated_future_v1.as_patient()` restated the status and the
    presence channels separately, every two months, so `state_path` would not expire them.
    A superseded value then received fresh timestamps and beat the newer one: a lesion
    cleared at month 3 read NED, progressing, NED, progressing ... `generated_disease()`
    applies each event to ONE cumulative truth (presence, status) in event order. Nothing
    is restated, so nothing old is revived; generated assertions do not expire.

R2  Real treatment. Agent rows come from `regimen_cancer_level_dataset.csv` (all three
    cohorts), reconciled with `data_timeline_treatment.txt` by patient, cancer, regimen,
    agent and start. A documented STOP and a LAST ADMINISTRATION are kept distinct: for
    discontinuation No/Unknown, exposure after the last administration is unresolved; a
    Yes without a stop is unresolved from its start; investigational durations are
    redacted (analytic data guides: NSCLC v2.0 pp. 70-72, BRCA v1.0 pp. 93-96, CRC v2.0
    pp. 84-87). Nothing is carried to last follow-up.

R3  Drug classes. One versioned table, `scripts/four_arm_drug_classes_v1.csv`, exact
    alias matching after a fixed normalisation, for real agents, donors and generated
    components alike. A combination with any unrecognised component has an unresolved
    modality set; an adjunct is recognised but carries no modality.

R4  Per-agent exposure. Each agent (real) or component (generated) has its own intervals;
    `systemic_treatment_modification` discontinues components and applies holds. Modality
    time is a union over agents and includes exposure already running at the landmark.

R5  Exposure versus events. Positive-duration exposure inside [0, end) is separate from the
    regimen-start events used for counts and modality-set changes, which include a start
    exactly at H.

GOVERNANCE. GENIE-BPC under DUA. This module holds no data.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
import disease_extent_v1 as DE  # noqa: E402

from synthonccases import disease_state as DS  # noqa: E402

DRUG_TABLE = ROOT / "scripts" / "four_arm_drug_classes_v1.csv"
MODALITIES = ("chemotherapy", "immunotherapy", "targeted", "endocrine")
ADJUNCT = "adjunct"
UNRECOGNISED = "unrecognised"
#: Registry clock (the one `genie_operator_registry_v2.load` uses, so saved landmarks agree).
REGISTRY_DAYS = 30.4
#: Generated clock (as in `generated_future_v1`).
GENERATED_DAYS = 30.4375
#: Regimen starts within half a day are one modality-set group (GV2.SAME_DAY).
SAME_DAY = 0.5 / REGISTRY_DAYS
#: GENIE ends a regimen at a treatment break of eight weeks or more; a generated hold that
#: long interrupts exposure (but is not a new regimen). Shorter holds do not.
HOLD_BREAK_MONTHS = 56.0 / GENERATED_DAYS
EPS = 1e-9
ACTIVE_STATUS = ("responding", "stable", "progressing")
SEVERITY = {"responding": 0, "stable": 1, "progressing": 2}


# ======================================================================================
# R3. Drug classification
# ======================================================================================

_PAREN = re.compile(r"\([^)]*\)")


def normalise(name: object) -> str:
    """Lower case, parenthetical synonyms dropped (an unclosed '(' drops the rest, as in
    GENIE's truncated synonym lists), underscores as spaces, spaces collapsed."""
    text = _PAREN.sub(" ", str(name or "")).split("(")[0].lower().replace("_", " ")
    return " ".join(text.split())


@dataclass(frozen=True)
class DrugClass:
    drug: str                           # canonical name, or the normalised input if unknown
    kind: str                           # "modality" / "adjunct" / "unrecognised"
    modalities: frozenset[str]          # empty for adjunct and unrecognised


def load_drug_table(path: Path = DRUG_TABLE) -> dict[str, DrugClass]:
    """alias -> class. Raises if an alias maps to two drugs or a class is not allowed."""
    out: dict[str, DrugClass] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            cls = row["class"].strip()
            if cls in (ADJUNCT, UNRECOGNISED):
                dc = DrugClass(row["drug"], cls, frozenset())
            else:
                mods = frozenset(c.strip() for c in cls.split(";"))
                if not mods or not mods <= set(MODALITIES):
                    raise ValueError(f"drug table: bad class {cls!r} for {row['drug']!r}")
                dc = DrugClass(row["drug"], "modality", mods)
            for alias in [row["drug"], *row["aliases"].split("|")]:
                key = normalise(alias)
                if not key:
                    continue
                if key in out and out[key] != dc:
                    raise ValueError(f"drug table: alias {key!r} maps to two drugs")
                out[key] = dc
    return out


_TABLE: dict[str, DrugClass] | None = None


def classify(name: object) -> DrugClass:
    """The class of one agent name. Exact alias match only; never a substring guess."""
    global _TABLE
    if _TABLE is None:
        _TABLE = load_drug_table()
    key = normalise(name)
    return _TABLE.get(key, DrugClass(key, UNRECOGNISED, frozenset()))


def table_sha256() -> str:
    import hashlib
    return hashlib.sha256(DRUG_TABLE.read_bytes()).hexdigest()


# ======================================================================================
# The common treatment object (R2-R5): real and generated courses are both reduced to it
# ======================================================================================

@dataclass(frozen=True)
class Agent:
    """One agent's exposure on a landmark clock (months).

    `segments` are documented exposure; `unresolved` are intervals on which exposure is
    unknown (b may be inf). `directed` says whether the source establishes the agent as
    cancer-directed even when its class is unrecognised (true for every GENIE agent: the
    curated regimen table holds cancer-directed drugs only).
    """

    name: str
    cls: DrugClass
    segments: tuple[tuple[float, float], ...]
    unresolved: tuple[tuple[float, float], ...] = ()
    directed: bool = False
    regimen: str = ""
    unresolved_kind: str = ""            # why `unresolved` exists (for reason codes)


@dataclass(frozen=True)
class RegimenStart:
    """A recorded regimen start (one row of a modality-set group)."""

    time: float
    components: tuple[DrugClass, ...]
    regimen: str = ""
    directed_by_source: bool = False

    @property
    def cancer_directed(self) -> bool | None:
        """True / False, or None when that cannot be established (unrecognised only)."""
        if any(c.kind == "modality" for c in self.components):
            return True
        if all(c.kind == ADJUNCT for c in self.components):
            return False
        return True if self.directed_by_source else None

    @property
    def modality_set(self) -> frozenset[str] | None:
        """The modality set, or None when any component is unrecognised (R3)."""
        if any(c.kind == UNRECOGNISED for c in self.components):
            return None
        return frozenset().union(*(c.modalities for c in self.components))


@dataclass(frozen=True)
class Treatment:
    agents: tuple[Agent, ...]
    starts: tuple[RegimenStart, ...]
    problems: tuple[str, ...] = ()       # reasons the whole object is unreliable
    active_start_conversions: int = 0    # starts read as switches (treatment already active)


# ---- interval helpers -------------------------------------------------------------------

def merge(spans: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if out and a <= out[-1][1] + EPS:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def overlap(spans: Iterable[tuple[float, float]], lo: float, hi: float) -> float:
    return float(sum(max(0.0, min(b, hi) - max(a, lo)) for a, b in merge(spans)))


def _positive(spans: Iterable[tuple[float, float]], lo: float, hi: float) -> bool:
    """R5: positive-duration overlap with [lo, hi)."""
    return overlap(spans, lo, hi) > EPS


# ======================================================================================
# Treatment coordinates (values and resolution), R4/R5
# ======================================================================================

TREATMENT_COORDS = ("months_any", "months_chemotherapy", "months_immunotherapy",
                    "months_targeted", "months_endocrine", "n_subsequent_regimens",
                    "n_modality_changes", "first_change", "time_to_first_change")


def start_groups(starts: Sequence[RegimenStart], lo: float,
                 hi_inclusive: float, before: float) -> list[tuple[float, list[RegimenStart]]]:
    """Co-dated cancer-directed (or undecidable) starts, as groups, time-ordered.

    A start counts when lo <= t <= hi_inclusive and t < before (death). Adjunct-only starts
    are not cancer-directed regimens and are skipped.
    """
    keep = sorted((s for s in starts if s.cancer_directed is not False
                   and lo - EPS <= s.time <= hi_inclusive + EPS and s.time < before - EPS),
                  key=lambda s: s.time)
    groups: list[tuple[float, list[RegimenStart]]] = []
    for s in keep:
        if groups and s.time - groups[-1][0] < SAME_DAY:
            groups[-1][1].append(s)
        else:
            groups.append((s.time, [s]))
    return groups


def _group_set(group: list[RegimenStart]) -> frozenset[str] | None:
    sets = [s.modality_set for s in group]
    if any(x is None for x in sets):
        return None
    return frozenset().union(*sets)


def treatment_coordinates(tx: Treatment, death: float, horizon: float
                          ) -> tuple[dict[str, float | None], list[str]]:
    """The nine treatment coordinates in raw units over [0, H], or None when unresolved.

    Exposure: union of positive-duration documented intervals inside [0, min(death, H)),
    including agents started before the landmark. Events: regimen-start groups in (0, H]
    before death (a start exactly at H counts); the time-zero group is the reference and
    the index regimen is never subsequent. Survival must be resolved by the caller.
    """
    end = min(death, horizon)
    reasons: list[str] = []
    out: dict[str, float | None] = {}
    if tx.problems:
        reasons += list(tx.problems)

    # ---- exposure ----------------------------------------------------------------------
    def agents_for(mod: str | None) -> tuple[list[Agent], list[Agent]]:
        """(agents that contribute to this channel, agents whose contribution is unknown)."""
        known, unknown = [], []
        for ag in tx.agents:
            if ag.cls.kind == ADJUNCT:
                continue
            if ag.cls.kind == UNRECOGNISED:
                if mod is None and ag.directed:
                    known.append(ag)        # any-therapy: a GENIE agent is cancer-directed
                elif _positive(ag.segments + ag.unresolved, 0.0, end):
                    unknown.append(ag)
                continue
            if mod is None or mod in ag.cls.modalities:
                known.append(ag)
        return known, unknown

    for mod, key in [(None, "months_any")] + [(m, f"months_{m}") for m in MODALITIES]:
        known, unknown = agents_for(mod)
        tail = [ag for ag in known if _positive(ag.unresolved, 0.0, end)]
        if unknown:
            reasons.append("treatment_modality_unrecognised_agent_exposed")
        for ag in tail:
            pre = "_pre_landmark" if min(a for a, _b in ag.unresolved) < -EPS else ""
            reasons.append(f"treatment_exposure_unresolved_{ag.unresolved_kind}{pre}")
        out[key] = (None if unknown or tail or tx.problems else
                    overlap([s for ag in known for s in ag.segments], 0.0, end))

    # ---- regimen events (R5: inclusive of H) ---------------------------------------------
    groups = start_groups(tx.starts, 0.0, horizon, death)
    base = [g for g in groups if g[0] <= SAME_DAY]
    later = [g for g in groups if g[0] > SAME_DAY]
    undecidable = [g for g in later if any(s.cancer_directed is None for s in g[1])]
    if undecidable:
        reasons.append("treatment_regimen_cancer_directed_unknown")
    out["n_subsequent_regimens"] = None if undecidable or tx.problems else float(len(later))

    prev = _group_set(base[0][1]) if base else frozenset()
    changes: list[bool | None] = []
    for _t, g in later:
        cur = _group_set(g)
        changes.append(None if (prev is None or cur is None) else cur != prev)
        prev = cur
    first_idx = next((i for i, c in enumerate(changes) if c is not False), None)
    if any(c is None for c in changes):
        reasons.append("treatment_modality_set_unresolved")
    if tx.problems:
        out["n_modality_changes"] = out["first_change"] = out["time_to_first_change"] = None
    else:
        out["n_modality_changes"] = (None if any(c is None for c in changes)
                                     else float(sum(bool(c) for c in changes)))
        if first_idx is None:
            out["first_change"], out["time_to_first_change"] = 0.0, float(horizon)
        elif changes[first_idx] is None:
            out["first_change"] = out["time_to_first_change"] = None
        else:
            out["first_change"] = 1.0
            out["time_to_first_change"] = float(min(later[first_idx][0], horizon))
    return out, sorted(set(reasons))


# ======================================================================================
# R2. Real treatment from the curated agent fields
# ======================================================================================

@dataclass(frozen=True)
class RealAgent:
    """One curated agent record on the DIAGNOSIS clock (months)."""

    regimen: str                 # "ca_seq:regimen_number"
    name: str
    start: float
    stop: float                  # documented stop, nan if none
    last_admin: float            # last known administration when not discontinued, nan if none
    discontinued: str            # "Yes" / "No" / "Unknown"
    investigational: bool
    source: str                  # matched / regimen_only / timeline_other_ca_seq


def _dc(value: object) -> str:
    text = str(value)
    if text == "Yes":
        return "Yes"
    if text == "No":
        return "No"
    return "Unknown"


def load_real_agents(genie: Path) -> tuple[dict[str, list[RealAgent]], dict[str, int]]:
    """Index-cancer agents per patient, reconciled between the two GENIE exports.

    Primary source: the five agent slots of `regimen_cancer_level_dataset.csv`, restricted
    to the index cancer (`DE.index_cancer`). Each timeline row of the index cancer is
    matched to a slot by (patient, regimen number, agent, start day). Unmatched timeline
    rows whose slot is filed under another ca_seq of the same patient are added with that
    slot's fields (the dates agree); any other unmatched timeline row is an error.
    Returns per-patient agents and aggregate reconciliation counts.
    """
    keep = DE.index_cancer(genie)
    reg = pd.read_csv(genie / "regimen_cancer_level_dataset.csv", low_memory=False)
    slots = []
    for k in range(1, 6):
        sub = reg[["record_id", "ca_seq", "regimen_number", "drugs_dc_ynu",
                   f"drugs_drug_{k}", f"dx_drug_start_int_{k}", f"dx_drug_end_int_{k}",
                   f"dx_drug_end_or_lastadm_int_{k}"]].copy()
        sub.columns = ["pid", "seq", "reg", "dc", "drug", "s", "e", "el"]
        slots.append(sub)
    st = pd.concat(slots, ignore_index=True)
    st = st[st["drug"].notna()]
    st["pid"] = st["pid"].astype(str)
    st["seq"] = st["seq"].astype(int)
    st["reg"] = st["reg"].astype(int)
    st["key"] = list(zip(st["pid"], st["reg"], st["drug"].map(normalise), st["s"].round(),
                         strict=True))
    tx = pd.read_csv(genie / "data_timeline_treatment.txt", sep="\t", low_memory=False)
    tx = tx[[keep.get(str(a)) == b for a, b in zip(tx["PATIENT_ID"], tx["CA_SEQ"],
                                                    strict=True)]]
    tkeys = Counter(zip(tx["PATIENT_ID"].astype(str), tx["REGIMEN_NUMBER"].astype(int),
                        tx["AGENT"].map(normalise),
                        pd.to_numeric(tx["START_DATE"], errors="coerce").round(),
                        strict=True))
    counts: Counter = Counter()
    out: dict[str, list[RealAgent]] = defaultdict(list)

    def add(r, source: str) -> None:
        s = float(r.s)
        if not math.isfinite(s):
            counts["slot_without_start_date"] += 1
            return
        name = str(r.drug)
        out[r.pid].append(RealAgent(
            regimen=f"{r.seq}:{r.reg}", name=name, start=s / REGISTRY_DAYS,
            stop=float(r.e) / REGISTRY_DAYS if pd.notna(r.e) else math.nan,
            last_admin=float(r.el) / REGISTRY_DAYS if pd.notna(r.el) else math.nan,
            discontinued=_dc(r.dc),
            investigational="investigational" in normalise(name), source=source))

    index_slots = st[[keep.get(p) == q for p, q in zip(st["pid"], st["seq"], strict=True)]]
    remaining = Counter(tkeys)
    for r in index_slots.itertuples():
        if remaining.get(r.key, 0) > 0:
            remaining[r.key] -= 1
            add(r, "matched")
            counts["matched"] += 1
        else:
            add(r, "regimen_only")
            counts["regimen_table_only"] += 1
    other = st[[keep.get(p) is not None and keep.get(p) != q
                for p, q in zip(st["pid"], st["seq"], strict=True)]]
    by_key = defaultdict(list)
    for r in other.itertuples():
        by_key[r.key].append(r)
    for key, n in remaining.items():
        for _ in range(n):
            cands = by_key.get(key) or []
            if not cands:
                raise ValueError("a timeline agent row has no curated agent record")
            add(cands.pop(0), "timeline_other_ca_seq")
            counts["timeline_row_filed_under_other_ca_seq"] += 1
    for agents in out.values():
        for a in agents:
            counts[f"dc_{a.discontinued}"] += 1
            if a.investigational:
                counts["investigational"] += 1
            if not math.isfinite(a.stop):
                counts["no_documented_stop"] += 1
                if math.isfinite(a.last_admin):
                    counts["no_stop_but_last_administration"] += 1
    return dict(out), dict(counts)


def real_agent_exposure(a: RealAgent, tau: float
                        ) -> tuple[tuple[tuple[float, float], ...],
                                   tuple[tuple[float, float], ...], str]:
    """(documented, unresolved) exposure of one real agent on the landmark clock.

    Investigational: duration redacted -> unresolved from its start.
    Discontinued with a stop: documented [start, stop].
    Discontinued without a stop: unresolved from its start.
    Not discontinued / unknown, with a last administration: documented [start, last
    administration], unresolved afterwards (no carry to last follow-up).
    Not discontinued / unknown, without one: unresolved from its start.
    """
    s = a.start - tau
    if a.investigational:
        return (), ((s, math.inf),), "investigational_redacted"
    if a.discontinued == "Yes":
        if math.isfinite(a.stop):
            return ((s, max(s, a.stop - tau)),), (), ""
        return (), ((s, math.inf),), "discontinued_without_stop"
    kind = "not_discontinued" if a.discontinued == "No" else "discontinuation_unknown"
    if math.isfinite(a.last_admin):
        e = max(s, a.last_admin - tau)
        return ((s, e),), ((e, math.inf),), f"{kind}_after_last_administration"
    return (), ((s, math.inf),), f"{kind}_no_dates"


def real_treatment(agents: Sequence[RealAgent], tau: float) -> Treatment:
    """The treatment object of one real patient (query or donor) from its landmark `tau`."""
    out_agents = []
    by_regimen: dict[str, list[RealAgent]] = defaultdict(list)
    for a in agents:
        seg, unres, why = real_agent_exposure(a, tau)
        out_agents.append(Agent(name=a.name, cls=classify(a.name), segments=seg,
                                unresolved=unres, directed=True, regimen=a.regimen,
                                unresolved_kind=why))
        by_regimen[a.regimen].append(a)
    # The regimen-start EVENT is its earliest agent start on or after diagnosis, as in
    # `genie_operator_registry_v2.load`, which built the saved landmarks: an agent dated
    # before diagnosis still contributes exposure but does not move the regimen's start.
    starts = tuple(RegimenStart(time=min([x.start for x in ags if x.start >= 0]
                                         or [x.start for x in ags]) - tau,
                                components=tuple(classify(x.name) for x in ags),
                                regimen=reg, directed_by_source=True)
                   for reg, ags in by_regimen.items())
    if not any(abs(s.time) < SAME_DAY for s in starts):
        raise ValueError("no curated regimen starts at the saved landmark")
    return Treatment(agents=tuple(out_agents), starts=starts)


# ======================================================================================
# R1 + R4. Generated course: disease truth and treatment from the ledger
# ======================================================================================

class NotScoreable(ValueError):
    """The ledger validated but cannot be placed on the observable ruler."""


def _months(when: object, origin: date) -> float:
    d = when if isinstance(when, date) else date.fromisoformat(str(when)[:10])
    return (d - origin).days / GENERATED_DAYS


class Burden:
    """The cumulative clone/site picture (copied unchanged from `generated_future_v1._Burden`
    so this versioned reader does not move if the exploratory module is edited)."""

    DEFINITE = ("evidence", "no evidence")

    def __init__(self, presence: str | None, sites: list[str] | None,
                 extent: str | None = None) -> None:
        self.by_site: dict[tuple[str, str], str] = {}
        self.known = presence in self.DEFINITE
        if presence == "evidence":
            if sites:
                for s in sites:
                    self.by_site[("_baseline", s)] = "moderate"
            elif extent == "no":
                self.by_site[("_baseline", "primary_site")] = "moderate"
            else:
                self.known = False

    def apply(self, changes: object) -> tuple[str | None, list[str]]:
        new_sites, touched = [], False
        for c in changes if isinstance(changes, list) else []:
            if not isinstance(c, dict):
                continue
            band = str(c.get("burden") or "")
            site = str(c.get("site_class") or c.get("site") or "").strip() or "_unnamed"
            kinetic = str(c.get("kinetic") or "")
            if kinetic == "cleared":
                band = "absent"
            clone = str(c.get("clone") or "_uncloned")
            self.by_site[(clone, site)] = band
            self.by_site.pop(("_baseline", site), None)
            touched = True
            if kinetic == "appearing" and site not in ("primary_site", "unknown", "_unnamed"):
                new_sites.append(site)
        if not touched:
            return None, new_sites
        if any(b in DS.ACTIVE_BANDS for b in self.by_site.values()):
            return "evidence", new_sites
        return ("no evidence" if self.known else None), new_sites


@dataclass(frozen=True)
class GeneratedCourse:
    path: tuple[tuple[float, float, str], ...]
    death: float
    last_contact: float
    terminal: str
    new_sites: tuple[tuple[float, str], ...]
    conflicts: int                         # genuine same-time contradictions asserted
    treatment: Treatment


def _seed(landmark_state: str, landmark_presence: str | None) -> tuple[str | None,
                                                                       str | None]:
    presence = landmark_presence
    if not presence:
        if landmark_state == "no_active_disease":
            presence = "no evidence"
        elif landmark_state in ACTIVE_STATUS:
            presence = "evidence"
    status = landmark_state if landmark_state in ACTIVE_STATUS else None
    return presence, status


def generated_disease(events: Sequence[dict], *, cutoff: str | date, landmark_state: str,
                      landmark_presence: str | None = None,
                      landmark_sites: list[str] | None = None,
                      landmark_extent: str | None = None
                      ) -> tuple[list[tuple[float, str]], float, float, str,
                                 list[tuple[float, str]], int]:
    """R1: the asserted disease course as cumulative truth, updated in event order.

    State variables: presence P and curated-style status S, seeded from the landmark the
    prompt gave. Events sharing a date are one update: the cumulative burden after all of
    them gives the presence reading, their disease event types give the status (worst
    wins), a newly appearing distant site asserts progression. Supersession is strictly by
    time: a later NED reading clears an older status; a later active status or new site
    makes an older NED reading evidence of disease. NED and active disease asserted at the
    same time are a GENUINE contradiction: unknown until a later event updates either
    channel. A response whose cumulative burden has all cleared is a complete response.
    Returns (change points [(t, state)], death, last contact, terminal kind, new sites,
    number of genuine contradictions).
    """
    origin = cutoff if isinstance(cutoff, date) else date.fromisoformat(str(cutoff)[:10])
    p_, s_ = _seed(landmark_state, landmark_presence)
    burden = Burden(p_, landmark_sites, landmark_extent)
    points = [(0.0, DS.from_genie(presence=p_, curated_status=s_).state)]
    death, censored, last_event = math.inf, None, 0.0
    sites: list[tuple[float, str]] = []
    conflicts = 0
    timed = []
    for i, ev in enumerate(events):
        try:
            when = max(0.0, _months(ev["date"], origin))
        except (KeyError, ValueError) as exc:
            raise NotScoreable("event with an unreadable date") from exc
        timed.append((when, i, ev))
    timed.sort(key=lambda x: (x[0], x[1]))
    k = 0
    while k < len(timed):
        when = timed[k][0]
        group = []
        while k < len(timed) and timed[k][0] == when:
            group.append(timed[k][2])
            k += 1
        last_event = max(last_event, when)
        reading, statuses, new_site = None, [], False
        for ev in group:
            dis = ev.get("disease") or []
            r, new = burden.apply(dis)
            if isinstance(dis, list) and any(isinstance(c, dict) for c in dis):
                reading = r          # the cumulative picture after this event (may be None)
            for s in new:
                sites.append((when, s))
                new_site = True
            kind = ev.get("type")
            if kind == "progression":
                statuses.append("progressing")
            elif kind == "response":
                statuses.append("responding")
            elif kind == "death":
                death = min(death, when)
            elif kind == "last_followup":
                censored = when
        status = max(statuses, key=SEVERITY.get) if statuses else None
        if status == "responding" and reading == "no evidence" and not new_site:
            status = None                       # complete response
        if new_site:
            status = "progressing"
        if reading == "no evidence" and status is not None:
            p_, s_ = "no evidence", status      # genuine same-time contradiction
            conflicts += 1
        elif reading == "no evidence":
            p_, s_ = "no evidence", None
        elif status is not None:
            s_ = status
            p_ = reading if reading is not None else ("evidence" if p_ == "no evidence"
                                                      else p_)
        elif reading is not None:
            p_ = reading
        st = DS.from_genie(presence=p_, curated_status=s_).state
        if st != points[-1][1]:
            points.append((when, st))
    if math.isfinite(death):
        terminal, last_contact = "death", death
    elif censored is not None:
        terminal, last_contact = "censored", censored
    else:
        terminal, last_contact = "truncated", last_event
    return points, death, last_contact, terminal, sorted(sites), conflicts


def _path(points: Sequence[tuple[float, str]], end: float) -> tuple[tuple[float, float, str],
                                                                    ...]:
    out: list[tuple[float, float, str]] = []
    for i, (t, st) in enumerate(points):
        b = points[i + 1][0] if i + 1 < len(points) else end
        a, b = min(t, end), min(b, end)
        if b - a <= 1e-9:
            continue
        if out and out[-1][2] == st:
            out[-1] = (out[-1][0], b, st)
        else:
            out.append((a, b, st))
    return tuple(out)


def split_index(label: str | None) -> list[str]:
    """Components of a saved index-regimen label ('a + b + c')."""
    return [x.strip() for x in str(label or "").split(" + ") if x.strip()]


def _components(regimen: object) -> list[str]:
    if isinstance(regimen, str):
        return [regimen] if regimen.strip() else [""]
    if isinstance(regimen, dict):
        comps = regimen.get("components")
        if isinstance(comps, list) and comps:
            return [str(c) for c in comps]
        return [str(regimen.get("label") or "")]
    return [""]


def generated_treatment(events: Sequence[dict], *, cutoff: str | date,
                        index_regimen: str | None, death: float, last_contact: float
                        ) -> Treatment:
    """R3/R4: per-component exposure from start/switch/stop/modification events.

    The index regimen is running at time zero. A switch or stop closes every active
    component; a start opens new ones. A start recorded while treatment is still active is
    read as a switch: it closes the active components before opening its own. This follows
    the frozen generation contract (a start only for the first regimen or after a stop) and
    frozen replay, which promotes such a start to a switch
    (SOFT_LIFECYCLE_AUTO_PROMOTED_START_TO_SWITCH); it is never read as an added concurrent
    regimen. These conversions are counted in `active_start_conversions`. `discontinuation_of_component` closes the named
    components at that date. `dose_hold` on named components (all, if none named)
    interrupts exposure only when it lasts eight weeks or more, until `dose_resume`, or
    until the component ends if never resumed (GENIE's own regimen-break rule). Dose
    reductions do not change exposure. A modification naming a component that is not
    active is ambiguous: a discontinuation makes every active component of that regimen
    unresolved from that date; a hold does so for the hold, if it lasts eight weeks or
    more. Exposure ends at death or last contact. Regimen counts follow start/switch events
    only; an interrupted exposure is not a new regimen.
    """
    origin = cutoff if isinstance(cutoff, date) else date.fromisoformat(str(cutoff)[:10])
    stop_all = min(death, last_contact)
    agents: list[dict] = []
    starts: list[RegimenStart] = []
    reg_no = 0
    conversions = 0

    def ident(name: str) -> str:
        c = classify(name)
        return c.drug if c.kind != UNRECOGNISED else normalise(name)

    def open_regimen(t: float, comps: list[str]) -> None:
        nonlocal reg_no
        reg_no += 1
        rid = f"g{reg_no}"
        classes = [classify(c) for c in comps]
        starts.append(RegimenStart(time=t, components=tuple(classes), regimen=rid))
        for name, cls in zip(comps, classes, strict=True):
            agents.append({"name": name, "cls": cls, "id": ident(name), "start": t,
                           "stop": None, "regimen": rid, "gaps": [], "held": None,
                           "amb_held": None, "unres": [], "unres_from": None})

    def active(t: float) -> list[dict]:
        return [a for a in agents if a["stop"] is None and a["start"] <= t + EPS]

    def end_hold(a: dict, t: float) -> None:
        if a["held"] is not None and t - a["held"] >= HOLD_BREAK_MONTHS - EPS:
            a["gaps"].append((a["held"], t))
        if a["amb_held"] is not None and t - a["amb_held"] >= HOLD_BREAK_MONTHS - EPS:
            a["unres"].append((a["amb_held"], t))
        a["held"] = a["amb_held"] = None

    def close(a: dict, t: float) -> None:
        end_hold(a, t)
        a["stop"] = max(a["start"], t)

    if index_regimen:
        open_regimen(0.0, split_index(index_regimen) or [index_regimen])
    for ev in sorted(events, key=lambda e: str(e.get("date"))):
        kind = ev.get("type")
        if kind not in ("systemic_treatment_start", "systemic_treatment_switch",
                        "systemic_treatment_stop", "systemic_treatment_modification"):
            continue
        t = max(0.0, _months(ev["date"], origin))
        if kind != "systemic_treatment_modification":
            if kind == "systemic_treatment_start" and active(t):
                conversions += 1
            for a in active(t):
                close(a, t)
            if kind != "systemic_treatment_stop":
                open_regimen(t, _components(ev.get("regimen")))
            continue
        mod = ev.get("modification_type")
        named = [x for x in (ev.get("affected_components") or []) if str(x).strip()]
        now = active(t)
        want = {ident(x) for x in named}
        hit = [a for a in now if a["id"] in want] if named else list(now)
        ambiguous = bool(named) and not want <= {a["id"] for a in now}
        if mod == "discontinuation_of_component":
            if not named or ambiguous:
                for a in now:
                    if a["unres_from"] is None:
                        a["unres_from"] = t
            for a in (hit if named else []):
                close(a, t)
        elif mod == "dose_hold":
            for a in hit:
                if a["held"] is None:
                    a["held"] = t
            if ambiguous:
                for a in now:
                    if a["amb_held"] is None:
                        a["amb_held"] = t
        elif mod == "dose_resume":
            for a in now:
                if a["id"] in want or not named or a["amb_held"] is not None:
                    end_hold(a, t)
    out = []
    for a in agents:
        if a["stop"] is None:
            close(a, stop_all)
        stop = min(a["stop"], stop_all)
        spans = [(a["start"], stop)]
        for g0, g1 in a["gaps"]:
            spans = [piece for s0, s1 in spans
                     for piece in ((s0, min(s1, g0)), (max(s0, g1), s1)) if piece[1] > piece[0]]
        unres = [(u0, min(u1, stop)) for u0, u1 in a["unres"] if u0 < stop]
        if a["unres_from"] is not None and a["unres_from"] < stop:
            unres.append((a["unres_from"], stop))
        if unres:
            spans = [piece for s0, s1 in spans for piece in _minus((s0, s1), unres)]
        out.append(Agent(name=a["name"], cls=a["cls"], segments=tuple(spans),
                         unresolved=tuple(unres), regimen=a["regimen"],
                         unresolved_kind="generated_ambiguous_modification" if unres else ""))
    return Treatment(agents=tuple(out), starts=tuple(starts),
                     active_start_conversions=conversions)


def _minus(span: tuple[float, float], cuts: Sequence[tuple[float, float]]
           ) -> list[tuple[float, float]]:
    pieces = [span]
    for c0, c1 in cuts:
        pieces = [p for s0, s1 in pieces
                  for p in ((s0, min(s1, c0)), (max(s0, c1), s1)) if p[1] > p[0]]
    return pieces


def generated_course(events: Sequence[dict], *, cutoff: str | date, landmark_state: str,
                     horizon: float, index_regimen: str | None = None,
                     landmark_presence: str | None = None,
                     landmark_sites: list[str] | None = None,
                     landmark_extent: str | None = None) -> GeneratedCourse:
    """The accepted ledger on the landmark clock: R1 disease path plus R4 treatment."""
    points, death, last_contact, terminal, sites, conflicts = generated_disease(
        events, cutoff=cutoff, landmark_state=landmark_state,
        landmark_presence=landmark_presence, landmark_sites=landmark_sites,
        landmark_extent=landmark_extent)
    end = min(death, last_contact, horizon)
    tx = generated_treatment(events, cutoff=cutoff, index_regimen=index_regimen,
                             death=death, last_contact=last_contact)
    return GeneratedCourse(path=_path(points, end), death=death, last_contact=last_contact,
                           terminal=terminal, new_sites=tuple(sites), conflicts=conflicts,
                           treatment=tx)


def finite(x: float) -> bool:
    return bool(np.isfinite(x))
