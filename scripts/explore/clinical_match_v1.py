#!/usr/bin/env python3
"""Patients like me, by criteria a clinician can check by eye.

The previous matcher was a Euclidean distance over 38 standardised columns. It was not
wrong so much as unreadable, and its weighting was an accident: 25 of the 38 columns were
treatment history, several near-duplicates of each other, so the history block carried
65-73% of the distance while age, sex, stage and histology together carried 12-17%. The
consequence was measurable -- "patients like me" differed from the query by nearly a decade
of age, barely closer than a randomly drawn donor.

This replaces it with an explicit count of clinically meaningful agreements. Every criterion
counts once, so nothing is diluted by how many correlated columns happen to describe it,
and a clinician can audit any single match by reading the two patients side by side.

HARD REQUIREMENTS define comparability rather than similarity: same cancer, same treatment
modality set, landmark year within the era window. Donors failing these are not distant
patients, they are patients being asked a different question.

WHAT EACH COHORT ACTUALLY RECORDS differs, and the matcher uses only what is there:
metastatic sites and histology exist in NSCLC and CRC and are absent from BRCA, so breast
matches on six criteria rather than eight. Scoring a missing field as agreement would let
two patients look alike because neither was measured.

UNKNOWN IS NEVER AGREEMENT. Two patients with no molecular result share an absence of
information, not a genotype. Unknown scores zero on both sides -- neither a match nor a
mismatch -- the same rule the retrieval offset already used.

GOVERNANCE. GENIE-BPC under DUA. Aggregate output only, gitignored.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disease_extent_v1 as DE  # noqa: E402
import genie_operator_registry_v2 as GV2  # noqa: E402

AGE_BAND = 10.0          # years; within this counts as an age match
#: Time from diagnosis to the landmark is compared as a RATIO, not a difference, because
#: it is heavily skewed: within one regimen-sequence stratum it spans 6 to 323 months.
#: "Within a factor of two" means the same thing at 6 months and at 60; "within 12 months"
#: would be everything at one end and nothing at the other.
TEMPO_FACTOR = 2.0
STATES = GV2.STATES


@dataclass(frozen=True)
class Card:
    """One patient at one landmark, as the matcher sees them. Eight readable facts."""

    presence: str | None            # oncologist: evidence / no evidence / equivocal
    stage_iv: float | None          # 1 stage IV at diagnosis, 0 stage I-III
    mets_documented: str            # extent AT THE LANDMARK: yes / no / unknown
    driver: str                     # molecular driver, or "unknown"
    status: str | None              # disease state at the landmark
    depth: str                      # regimen sequence 1 / 2 / 3+
    female: float | None
    age: float | None               # age AT THE LANDMARK, not at diagnosis
    tempo: float | None             # months from diagnosis to the landmark
    sites: frozenset[str] | None    # documented metastatic sites, None where not collected
    histology: str | None           # None where not collected


def describe(patient, tau: float, driver: str, mets) -> Card:
    f = GV2.prefix_features(patient, tau)
    extent = DE.extent_at_landmark(mets, tau)
    # The CANONICAL state, so the matcher and the outcome scorer speak one vocabulary.
    # Reading raw status here meant a patient in remission had no state to match on, and
    # two such patients could never be recognised as alike.
    ds = GV2.genie_state(patient, tau, mets)
    status = None if ds.state == "unknown" else ds.state
    # PRESENCE as its own criterion, not folded into the state. It is what distinguishes
    # an adjuvant patient with no evidence of disease from one being treated for active
    # cancer -- a difference the dynamics vocabulary cannot express at all.
    pres = GV2.recent(patient.presence, tau)
    histology = ("adeno" if f.get("histology_adeno") else
                 "squamous" if f.get("histology_squamous") else None)
    number = f.get("now_line_number", np.nan)
    return Card(
        presence=pres,
        stage_iv=(f["stage_at_diagnosis_iv"] if not f["stage_at_diagnosis_unknown"]
                  else None),
        # Extent at the LANDMARK, separate from named sites. Breast records timed
        # any-metastasis but no site list, so keeping only sites lost extent entirely for
        # that cohort -- and extent is the more basic clinical fact of the two.
        mets_documented=extent.documented,
        driver=driver,
        status=status,
        depth=({1: "regimen 1", 2: "regimen 2"}.get(int(number), "regimen 3+")
               if np.isfinite(number) and number >= 1 else "unknown"),
        female=f.get("sex_female"),
        # Age AT THE LANDMARK. Patients like me at the moment of THIS decision are not
        # patients who were my age when they were diagnosed years earlier.
        age=(f["age_at_diagnosis"] + tau / 12.0
             if np.isfinite(f.get("age_at_diagnosis", np.nan)) else None),
        # How long it took to reach this treatment. Given the regimen sequence, this is
        # essentially how well the previous lines worked -- reaching third line at six
        # months and at seventy-nine months are opposite clinical situations, and the
        # sequence number alone calls them the same.
        tempo=float(tau) if tau > 0 else None,
        sites=(frozenset(s for _m, s in extent.sites)
               if extent.site_detail_available else None),
        histology=histology)


def informative(cards: list[Card], floor: float = 0.95) -> frozenset[str]:
    """Criteria that actually vary in THIS cohort.

    Sex is 98% female in breast and histology is ~100% adenocarcinoma in colorectal, so
    those criteria are satisfied by almost any pair. Counting them hands every donor a free
    point and dilutes the criteria that discriminate -- the same reason a constant column
    was worthless in the old distance. A criterion is kept only if two randomly drawn
    patients would disagree at least sometimes.
    """
    rng = np.random.default_rng(0)
    tally: dict[str, list[bool]] = {}
    for _ in range(min(2000, max(1, len(cards) * 4))):
        i, j = rng.integers(len(cards), size=2)
        if i == j:
            continue
        _h, _t, detail = agreements(cards[i], cards[j])
        for k, v in detail.items():
            tally.setdefault(k, []).append(v)
    return frozenset(k for k, v in tally.items()
                     if v and float(np.mean(v)) < floor)


def agreements(a: Card, b: Card,
               keep: frozenset[str] | None = None) -> tuple[int, int, dict[str, bool]]:
    """How many criteria agree, out of how many could be compared, and which.

    The denominator is returned because it differs by cohort and by patient. A pair scoring
    5 of 6 is a better match than one scoring 5 of 8, and collapsing both to "5" would
    reward missing data.
    """
    checks: dict[str, bool | None] = {
        "cancer present at the landmark": (
            None if a.presence in (None, "equivocal") or b.presence in (None, "equivocal")
            else a.presence == b.presence),
        "stage at diagnosis": (None if a.stage_iv is None or b.stage_iv is None
                               else a.stage_iv == b.stage_iv),
        "metastatic disease by landmark": (
            None if "unknown" in (a.mets_documented, b.mets_documented)
            else a.mets_documented == b.mets_documented),
        "molecular driver": (None if "unknown" in (a.driver, b.driver)
                             else a.driver == b.driver),
        "disease state at landmark": (None if a.status is None or b.status is None
                                      else a.status == b.status),
        "regimen sequence": (None if "unknown" in (a.depth, b.depth)
                             else a.depth == b.depth),
        "time to this treatment": (
            None if not a.tempo or not b.tempo
            else (1.0 / TEMPO_FACTOR) <= (a.tempo / b.tempo) <= TEMPO_FACTOR),
        "sex": (None if a.female is None or b.female is None else a.female == b.female),
        "age within a decade": (None if a.age is None or b.age is None
                                else abs(a.age - b.age) <= AGE_BAND),
        # Site OVERLAP, not identity: liver-and-bone against liver-and-lung is a partial
        # match, and demanding identical site sets would make every multi-site patient
        # unmatchable.
        "metastatic sites overlap": (None if a.sites is None or b.sites is None
                                     else bool(a.sites & b.sites)
                                     or (not a.sites and not b.sites)),
        "histology": (None if a.histology is None or b.histology is None
                      else a.histology == b.histology),
    }
    usable = {k: v for k, v in checks.items()
              if v is not None and (keep is None or k in keep)}
    return sum(usable.values()), len(usable), usable


def similarity(a: Card, b: Card, keep: frozenset[str] | None = None) -> float:
    """Signed agreement, averaged over the cohort's FIXED criteria.

        same +1      different -1      unknown or unmeasured 0

    Dividing by only the comparable criteria rewarded missing data: a donor matching on
    four criteria with an unknown driver scored 4/4 = 100%, beating a donor matching on the
    same four with a KNOWN different driver at 4/5 = 80%. Unknown molecular status became
    an advantage, which is the very thing the neutral-unknown rule exists to prevent.

    Averaging over the fixed cohort denominator makes unknown genuinely neutral: it neither
    adds nor subtracts, and it cannot shrink the denominator to flatter a donor. The score
    runs from -1 (every criterion disagrees) to +1 (every criterion agrees).
    """
    _hits, _total, detail = agreements(a, b, keep)
    fixed = len(keep) if keep else len(detail)
    if not fixed:
        return float("nan")
    return sum(1.0 if v else -1.0 for v in detail.values()) / fixed


def explain(query: Card, donor: Card, keep: frozenset[str] | None = None) -> str:
    """One line a clinician can check. This is the point of the whole exercise."""
    hits, total, detail = agreements(query, donor, keep)
    agree = [k for k, v in detail.items() if v]
    differ = [k for k, v in detail.items() if not v]
    out = f"{hits}/{total} criteria agree"
    if agree:
        out += f" -- same {', '.join(agree)}"
    if differ:
        out += f"; differs in {', '.join(differ)}"
    return out
