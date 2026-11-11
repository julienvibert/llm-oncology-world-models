#!/usr/bin/env python3
"""The query universe: who gets asked, who may be shown, and who is held back.

This is the irreversible part of the experiment. Everything downstream -- prompts, scores,
analyses -- can be rebuilt from generated data, but which patients were asked and which
were held out cannot be changed after generation without invalidating the confirmation.

WHAT IT FIXES

  landmark      one treatment-line start per patient, stratified across line 1, line 2 and
                line 3+ so the history-depth axis is populated by design.
  confirmation  patients held back before anything else happens. They are never queries in
                development and NEVER appear in any donor pool, so their futures cannot
                reach the model through somebody else's card.
  donors        drawn only from development-eligible patients, never the query patient.
  candidates    for the RELEVANT card only: same cancer, the same SET of treatment
                modalities, and a comparable era. The generic card holds cancer and era
                but NOT treatment -- an earlier version of this header said both cards
                shared treatment modality, which described a different experiment from the
                one the code runs. The modality class is the union -- "chemotherapy +
                targeted", not "targeted" -- because a first-match priority made FOLFOX
                plus bevacizumab share a class with cetuximab monotherapy while excluding
                FOLFOX itself. Holding treatment
                fixed is what makes the contrast about the PATIENT: if the wrong card's
                donors were on another drug, the model should discount them, and a
                positive result would partly mean "it noticed the treatment differs".
  correct card  the k NEAREST candidates in the standardised prefix space (C, X, H, with
                line depth inside H). Retrieval uses PREFIX ONLY; no donor is chosen
                because of what happened to them.
  generic card  k patients drawn at RANDOM from the same cancer and era, with no
                constraint on treatment, stage, driver, state or regimen sequence. It is a
                generic same-cancer reference cohort -- informative, not absurd -- and it
                answers the question a clinician actually has: must I build a retrieval
                system, or would any detailed cohort card do? Its accidental treatment and
                clinical concordance are MEASURED, never constrained.
  support       with k fixed, support is HOW FAR we had to travel, not how many exist. A
                short neighbour radius means a dense region. The achieved correct-to-wrong
                separation is the second support coordinate, and the primary effect is
                read against it.

GOVERNANCE. GENIE-BPC under DUA. Patient-level records are written only to the gitignored
outputs tree, never to the repository, and the manifest carries counts and hashes only.

Run:  uv run python scripts/explore/build_query_universe_v1.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clinical_match_v1 as CM  # noqa: E402
import disease_extent_v1 as DE  # noqa: E402
import genie_operator_registry_v2 as GV2  # noqa: E402
import molecular_prefix_v1 as MP  # noqa: E402
import render_reference_card_v1 as RC  # noqa: E402
import predictor_role_audit_v1 as RA  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GENIE_ROOT = Path.home() / "GenieBPC"
COHORTS = {"NSCLC": "NSCLC_Genie", "BRCA": "BRCA_GENIE", "CRC": "CRC_GENIE"}
OUT = ROOT / "outputs" / "exploratory_query_universe"
#: THREE DIFFERENT QUANTITIES, previously conflated under one name "horizon".
#: Generation is NOT bounded: the model writes the course to death or to the end of
#: documented follow-up, as in every previous run. Nothing here caps it.
MIN_FOLLOWUP = 12.0       # ELIGIBILITY floor: admit a patient only if their outcome at
                          # 12 months is knowable -- followed that long, or died sooner.
                          # This is why scoreability is exactly 100% at 12 months.
SCORING_HORIZONS = (3.0, 6.0, 12.0, 24.0)   # where the fan is compared to the observed
                          # future. 3/6/12 are complete by construction; 24 carries 5-9%
                          # censoring, handled by the scoring rule, not by exclusion.
K = 20
SEED = 20260824
N_CONFIRM = 100
N_DEV_PER_DEPTH = 50
#: REGIMEN sequence, not inferred line of therapy. GENIE records regimen changes; whether
#: two consecutive regimens are one clinical line depends on progression, toxicity and
#: intent, which the source does not reliably encode at the boundary. Naming these
#: "regimen 1/2/3+" says exactly what the data support and invents nothing.
DEPTHS = ("regimen 1", "regimen 2", "regimen 3+")
ERA_WINDOW = 3.0          # years; both donor sets must sit within this of the query
MOLECULAR_WEIGHT = 1.0    # standardised units; one feature's worth, deliberately untuned

#: The HARD constraints, named here so the operator registry checks what the code
#: enforces rather than a second declaration that could drift.
#: SCOPE MATTERS: cancer and era bind BOTH cards; treatment modality binds only the
#: relevant one. Listing modality as global implied the generic control was
#: treatment-matched, which it is not.
HARD_CONSTRAINTS = ("cancer", "treatment_modalities", "landmark_era")
HARD_CONSTRAINT_SCOPE = {"cancer": "both cards", "landmark_era": "both cards",
                         "treatment_modalities": "relevant card only"}


def landmark_years(genie: Path) -> dict[str, float]:
    """Calendar year of each patient's diagnosis, from birth year plus age at diagnosis.

    Era is part of C and must be shared by every arm. Without it the unaided arm defaults
    to today's standard of care while the anchored arm learns the era for free from the
    donors' drug names -- so part of anchoring's apparent benefit would be era
    calibration rather than patient relevance.
    """
    people = pd.read_csv(genie / "patient_level_dataset.csv", low_memory=False)
    cancers = pd.read_csv(genie / "cancer_level_dataset_index.csv", low_memory=False)
    merged = people[["record_id", "birth_year"]].merge(
        cancers[["record_id", "dob_ca_dx_yrs"]], on="record_id", how="inner")
    merged["dx_year"] = merged["birth_year"] + merged["dob_ca_dx_yrs"]
    return dict(zip(merged["record_id"].astype(str),
                    merged["dx_year"].astype(float), strict=True))


def depth3(number: float) -> str:
    if not np.isfinite(number) or number < 1:
        return "unknown"
    return {1: "regimen 1", 2: "regimen 2"}.get(int(number), "regimen 3+")


def eligible(patient, tau: float) -> bool:
    """Two requirements only: alive at the landmark, and enough future to score.

    TWO ASSESSMENT GATES HAVE BEEN REMOVED, and both changed the study population rather
    than merely its size.

    The first required a disease assessment BEFORE the landmark. It was mine, it duplicated
    work the three-way state encoding already did correctly, and it removed two thirds of
    first-line patients -- because a response assessment before any treatment has started
    barely applies to a newly diagnosed patient.

    The second required an assessment INSIDE the outcome window. That one selected for
    metastatic disease: among first-line landmarks it kept 41% stage IV and dropped
    patients who were only 4% stage IV, because an adjuvant patient with no measurable
    disease has no response to curate.

    Unknown state is now a legitimate INPUT, not an exclusion. It is matched neutrally, and
    the anchoring effect is reported separately for fully observed, partially observed and
    unknown current state.
    """
    if patient.death <= tau:
        return False
    return (patient.last_contact >= tau + MIN_FOLLOWUP
            or patient.death <= tau + MIN_FOLLOWUP)


def candidate_landmarks(patient) -> list[dict]:
    """EVERY regimen start, each judged on its own merits.

    Validity is a property of the landmark, not of the patient. A patient whose second
    regimen is an unnamed trial drug but whose first is carboplatin/pemetrexed has one
    unusable landmark and one perfectly good one, and must be kept on the strength of the
    good one. Judging the patient by a landmark drawn at random before testing it threw
    such patients away for a reason that had nothing to do with them.
    """
    out = []
    for ln in patient.lines:
        if not np.isfinite(ln.get("number", np.nan)) or ln["start"] <= 0.0:
            continue
        tau = float(ln["start"])
        why = []
        if not eligible(patient, tau):
            why.append("insufficient follow-up for scoring")
        if RC.regimen_name(ln.get("label", ""))[1]:
            why.append("unnamed index regimen: the treatment-conditioned question is "
                       "undefined")
        # Simultaneous records are unified into one action upstream, so no landmark can
        # carry a co-dated distinct regimen. Asserted rather than filtered: if this ever
        # fires, the merge stopped working and the action would be under-reported.
        assert not any(other is not ln and abs(other["start"] - tau) < SAME_DAY
                       for other in patient.lines), (
            f"{patient.pid} still has co-dated regimens at {tau}")
        out.append({"tau": tau, "line": float(ln["number"]),
                    "depth": depth3(ln["number"]), "bucket": ln["bucket"],
                    "valid": not why, "reasons": why})
    return out


def assign(patients: list, seed: int) -> list[dict]:
    """One VALID landmark per patient, chosen to fill the scarce depth strata first.

    A uniform draw over each patient's own line starts starves line 1 in breast cancer,
    where patients are followed for years and most of their qualifying starts are late.
    Patients whose only option is a given depth are assigned first, then the rest fill
    whichever stratum is furthest from target.
    """
    rng = np.random.default_rng(seed)
    options = {}
    for p in sorted(patients, key=lambda q: q.pid):
        got = [g for g in candidate_landmarks(p) if g["valid"]]
        if got:
            options[p.pid] = (p, got)
    filled = dict.fromkeys(DEPTHS, 0)
    out, deferred = [], []
    for pid, (p, got) in options.items():
        depths = {g["depth"] for g in got}
        if len(depths) == 1:
            pick = got[int(rng.integers(len(got)))]
            out.append({"patient": p, **pick})
            filled[pick["depth"]] += 1
        else:
            deferred.append((pid, p, got))
    rng.shuffle(deferred)
    for _pid, p, got in deferred:
        scarcest = min(DEPTHS, key=lambda d: filled[d])
        here = [g for g in got if g["depth"] == scarcest] or got
        pick = here[int(rng.integers(len(here)))]
        out.append({"patient": p, **pick})
        filled[pick["depth"]] += 1
    return out


def donor_record(records: list[dict], pids: list[str], i: int) -> dict:
    """Everything the card needs to render this donor exactly as retrieval saw them."""
    r = records[i]
    return {"patient": pids[i], "tau_months": float(r["tau"]),
            "year": r.get("year"), "line": float(r["line"]), "depth": r["depth"],
            "treatment_class": r["bucket"],
            # Persisted, not recomputed: the card must show the stage, extent and
            # molecular status that retrieval actually matched on, evaluated at this
            # donor's own landmark.
            "stage_at_diagnosis": r["stage_at_diagnosis"],
            "metastatic_extent": r["metastatic_extent"],
            "molecular_driver": r["driver"],
            "action": r["action"],
            # The eligibility window this donor had to satisfy -- NOT a generation
            # or scoring bound.
            "min_followup_window": [float(r["tau"]), float(r["tau"]) + MIN_FOLLOWUP]}


#: Only an IDENTICAL recorded start date is unresolved. GENIE records relative dates in
#: DAYS, so regimens beginning four and twenty-six days apart are ordered and the later one
#: is plainly a subsequent event. Treating anything inside a month as co-dated discarded
#: 31 breast landmarks where only 16 were genuinely ambiguous.
SAME_DAY = GV2.SAME_DAY


def action_at_landmark(patient, tau: float, line_number: float) -> dict:
    """A_tau: the exact treatment being started, not merely its class.

    The experiment conditions on (C, X, H, A). Persisting only `treatment_class` leaves A
    as "chemotherapy" -- which is a bucket the retrieval matched on, not the action the
    patient actually received. The model must be told the drugs, because carboplatin plus
    pemetrexed and cisplatin plus etoposide are the same class and different decisions.

    Four fields, each for a distinct reason:

      regimen             the canonical name, salt words stripped, for the prompt.
      regimen_raw         the untouched source string, so canonicalisation is auditable
                          and a rendering bug cannot silently rewrite the exposure.
      concurrent          every other regimen sharing the IDENTICAL start date. Those
                          cannot be ordered against the index regimen and must never be
                          rendered as a sequence. Landmarks with any such regimen do not
                          enter the study at all, so this should be empty; it is kept as
                          an audit field.
      index_treatment     a fixed marker naming this as THE treatment under study, so the
                          prompt cannot leave the model to infer which of several
                          same-month regimens the question is about.
    """
    # Matched on start AND line number. Several regimens can share an identical start
    # month, and taking the first match let the persisted action name a different drug --
    # and a different treatment class -- from the one retrieval selected the patient on.
    landmark = next((ln for ln in patient.lines
                     if abs(ln["start"] - tau) < 1e-9
                     and float(ln.get("number", np.nan)) == line_number), None)
    if landmark is None:
        raise ValueError(f"{patient.pid} has no line {line_number} starting at its own "
                         f"landmark {tau}")

    name, unnamed = RC.regimen_name(landmark.get("label", ""))
    concurrent = []
    for ln in patient.lines:
        if ln is landmark or abs(ln["start"] - tau) >= SAME_DAY:
            continue
        other, other_unnamed = RC.regimen_name(ln.get("label", ""))
        concurrent.append({
            "regimen": other, "regimen_raw": str(ln.get("label", "")),
            "regimen_unnamed": bool(other_unnamed),
            "treatment_class": ln["bucket"],
            "line_number": (float(ln["number"]) if np.isfinite(
                ln.get("number", np.nan)) else None)})

    return {
        "index_treatment": True,
        "regimen": name,
        "regimen_raw": str(landmark.get("label", "")),
        "regimen_unnamed": bool(unnamed),
        "treatment_class": landmark["bucket"],
        "line_number": (float(landmark["number"]) if np.isfinite(
            landmark.get("number", np.nan)) else None),
        "start_month": float(tau),
        "concurrent_same_date": concurrent,
        "n_concurrent_same_date": len(concurrent),
    }


def molecular_of(records: list[dict], cancer: str, genie: Path) -> list[str]:
    """Each record's driver bucket, as it was knowable AT that record's own landmark."""
    reports = MP.load_reports(genie)
    panels = MP.panel_genes(genie)
    matrix = dict(pd.read_csv(genie / "data_gene_matrix.txt", sep="\t",
                              low_memory=False)[["SAMPLE_ID", "mutations"]].dropna().values)
    variants: dict[str, list] = {}
    muts = pd.read_csv(genie / "data_mutations_extended.txt", sep="\t", low_memory=False,
                       usecols=["Hugo_Symbol", "HGVSp_Short", "Variant_Classification",
                                "Tumor_Sample_Barcode"])
    for m in muts.itertuples():
        variants.setdefault(m.Tumor_Sample_Barcode, []).append(
            (m.Hugo_Symbol, str(m.HGVSp_Short or "").lstrip("p."),
             m.Variant_Classification))
    fus = genie / "data_fusions.txt"
    if fus.exists():
        for m in pd.read_csv(fus, sep="\t", low_memory=False).itertuples():
            variants.setdefault(m.Tumor_Sample_Barcode, []).append(
                (m.Hugo_Symbol, "fusion", "Fusion"))
    return [MP.molecular_at_landmark(r["patient"].pid, r["tau"], reports, variants,
                                     matrix, panels, cancer).driver_bucket
            for r in records]


def molecular_offset(query: str, donors: np.ndarray) -> np.ndarray:
    """Concordant drivers pull closer, discordant push apart, UNKNOWN does neither.

    Unknown is not a third genotype that donors can agree on. Two patients with no
    sequencing report before their landmark share an absence of information, not a shared
    tumour biology, and one-hot encoding would have made them nearest neighbours on the
    strength of it. Molecular status is unknown for 40-60% of patients here, so that
    mistake would have driven a large part of retrieval.

    The shift is ONE standardised unit -- the same weight as any single feature. It is a
    simple, deliberately untuned choice; the sensitivity is reported alongside so its
    influence is visible rather than assumed.
    """
    known = (donors != "unknown") & (query != "unknown")
    return np.where(known, np.where(donors == query, -MOLECULAR_WEIGHT,
                                    MOLECULAR_WEIGHT), 0.0)


def comparable(query: str, donors: np.ndarray) -> int:
    """Donor pairs where BOTH sides have a driver call. The honest denominator."""
    return int(np.sum((donors != "unknown") & (query != "unknown")))


def known_agreement(query: str, donors: np.ndarray) -> float:
    """Agreement among comparable pairs only; NaN when there are none.

    Unknown is not a genotype. Two patients with no sequencing report before their
    landmark do not agree about anything, and scoring them as concordant would report
    shared missingness as shared biology.
    """
    donors = np.asarray(donors)
    n = comparable(query, donors)
    if n == 0:
        return float("nan")
    known = (donors != "unknown") & (query != "unknown")
    return float(np.sum(known & (donors == query)) / n)


def prefix_matrix(records: list[dict]) -> tuple[np.ndarray, list[str]]:
    frame = RA.clean_frame(pd.DataFrame(
        [GV2.prefix_features(r["patient"], r["tau"]) for r in records]))
    roles = RA.roles_of(list(frame.columns))
    cols = roles["C"] + roles["X"] + roles["H"]          # situation, not measurement
    raw = frame[cols].to_numpy(float)
    med = np.nanmedian(np.where(np.isfinite(raw), raw, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isfinite(raw), raw, med)
    sd = np.where(filled.std(axis=0) > 1e-9, filled.std(axis=0), 1.0)
    return (filled - filled.mean(axis=0)) / sd, cols


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # Separate streams: changing retrieval must not perturb who was selected or held
    # back. A single shared generator would have let a donor-rule change alter the next
    # cohort's confirmation carve.
    manifest: dict = {"k": K, "seed": SEED,
                      "n_confirmation_per_cancer": N_CONFIRM,
                      "n_development_per_depth": N_DEV_PER_DEPTH,
                      "generic_card_rule": "k patients drawn at RANDOM from the same "
                                           "cancer and era, with NO constraint on "
                                           "treatment, stage, driver, disease state or "
                                           "regimen sequence. It is a generic same-cancer "
                                           "reference cohort, not a mis-anchored or "
                                           "wrong card, and it is informative rather than "
                                           "absurd. Accidental treatment and clinical "
                                           "concordance are measured, never constrained.",
                      "generation_horizon": "unbounded: to death or to the end of "
                                            "documented follow-up",
                      "min_followup_months": MIN_FOLLOWUP,
                      "min_followup_rule": "eligibility floor, NOT a generation or "
                                           "scoring horizon: a patient is admitted only "
                                           "if their 12-month outcome is knowable",
                      "scoring_horizons_months": list(SCORING_HORIZONS),
                      "card_horizon_months": 24.0,
                      "landmark_unit": "REGIMEN start, not an inferred line of therapy. "
                                       "Depth strata are regimen sequence 1/2/3+ from "
                                       "regimen_number_within_cancer, restricted to a "
                                       "single index cancer. No regimens are merged into "
                                       "inferred lines: the intent and progression "
                                       "information that would justify a merge is not "
                                       "reliably encoded at the boundary.",
                      "arms": ["patient only", "patient + setting",
                               "patient + generic cohort",
                               "patient + patients-like-me"],
                      "contrasts": {
                          "2 - 1": "does describing the care setting help?",
                          "3 - 2": "does generic empirical cohort information help?",
                          "4 - 3": "does clinically targeted retrieval improve on an "
                                   "equally detailed but unselected same-cancer cohort?",
                          "4 - 1": "does the complete method improve on ordinary "
                                   "generation?"},
                      "futures_per_query_arm": 10,
                      "arm_rule": "every arm receives the identical CHAX prefix, the exact "
                                  "index regimen, and the same date anchors. Arms 2-4 add "
                                  "one setting sentence; arms 3-4 add a reference card.",
                      "control_scope": "contrast 4-3 tests the WHOLE retrieval strategy, "
                                       "treatment relevance included. It does not isolate "
                                       "whether the benefit comes from treatment matching "
                                       "or from state and history matching. The claim it "
                                       "supports is: query-relevant patients-like-me "
                                       "retrieval improves generation beyond an equally "
                                       "detailed but unselected same-cancer reference "
                                       "cohort.",
                      "support_rule": "neighbour radius at fixed k, not donor count. TWO "
                                      "radii are reported: the base C/X/H Euclidean "
                                      "radius is the primary support measure and is "
                                      "comparable across queries; the molecular-adjusted "
                                      "radius describes the retrieval that actually ran. "
                                      "They differ because the molecular term is an "
                                      "additive +/-1 adjustment, not one of the "
                                      "Euclidean coordinates, so the adjusted radius is "
                                      "not comparable between molecular-known and "
                                      "molecular-unknown queries",
                      "molecular_rule": "driver concordance shifts distance by "
                                        f"{MOLECULAR_WEIGHT} standardised unit; unknown "
                                        "is neutral and never matches another unknown",
                      "stage_rule": "stage at diagnosis (stage_dx_iv) and metastatic "
                                    "extent documented by the landmark (curated dmets "
                                    "block) are separate variables in both the retrieval "
                                    "space and the rendered prefix",
                      "era_window_years": ERA_WINDOW,
                      "era_rule": "donors era-matched on LANDMARK year, since that is "
                                  "when their prediction window opens; differences in "
                                  "diagnosis year are course-tempo differences and belong "
                                  "to X/H, not to era. Every arm's prefix renders two "
                                  "anchors: diagnosis year and landmark year.",
                      "cohorts": {}}
    all_queries: list[dict] = []

    for cohort, folder in COHORTS.items():
        genie = GENIE_ROOT / folder
        if not genie.exists():
            continue
        patients = GV2.load(genie)
        dx_year = landmark_years(genie)

        # LANDMARK-level accounting, before any patient is judged. Validity belongs to the
        # landmark; a patient is lost only when EVERY one of their regimen starts fails.
        audit: dict = {"candidate_landmarks": 0, "valid_landmarks": 0,
                       "patients_with_any_landmark": 0,
                       "patients_with_a_valid_landmark": 0,
                       "patients_with_no_valid_landmark": 0, "reasons": {}}
        for q in patients:
            got = candidate_landmarks(q)
            if not got:
                continue
            audit["patients_with_any_landmark"] += 1
            audit["candidate_landmarks"] += len(got)
            good = [g for g in got if g["valid"]]
            audit["valid_landmarks"] += len(good)
            audit["patients_with_a_valid_landmark"] += bool(good)
            audit["patients_with_no_valid_landmark"] += (not good)
            for g in got:
                for why in g["reasons"]:
                    audit["reasons"][why] = audit["reasons"].get(why, 0) + 1

        records = assign(patients, SEED)
        # TWO time anchors. A patient diagnosed in 2012 who starts third line in 2016
        # lived their history under 2012-2015 standards and is being treated under 2016
        # standards; one year alone makes the model either deny them a drug that had
        # arrived or wonder why their first line looks outdated. There is deliberately no
        # third "end year" anchor, because the course is generated to death rather than to
        # a fixed horizon. Era violations past the landmark are caught by the anachronism
        # audit instead of prevented by a stated cutoff.
        for r in records:
            base = dx_year.get(r["patient"].pid)
            r["diagnosis_year"] = round(base) if base is not None else None
            r["year"] = (round(base + r["tau"] / 12.0) if base is not None else None)
            # No horizon_end_year. Generation runs to death, so there is no end year to
            # state; asserting one would tell the model to stop at twelve months.
        pids = [r["patient"].pid for r in records]
        matrix, cols = prefix_matrix(records)
        depth = np.array([r["depth"] for r in records])
        bucket = np.array([r["bucket"] for r in records])
        # A landmark regimen that is ENTIRELY unnamed identifies no treatment. Such a
        # donor cannot serve as evidence about a therapy, so they are skipped and the next
        # eligible candidate is taken; such a query is flagged rather than dropped.
        unnamed = np.array([
            RC.regimen_name(next((ln.get("label", "") for ln in r["patient"].lines
                                  if abs(ln["start"] - r["tau"]) < 1e-9), ""))[1]
            for r in records])
        year = np.array([r["year"] if r["year"] is not None else np.nan
                         for r in records], float)
        driver = np.array(molecular_of(records, cohort, genie))
        mets = DE.load_metastasis(genie)
        for r, d in zip(records, driver, strict=True):
            e = DE.extent_at_landmark(mets.get(r["patient"].pid), r["tau"])
            r["stage_at_diagnosis"] = e.stage_at_diagnosis
            r["metastatic_extent"] = DE.render_extent(e)
            r["driver"] = str(d)
            r["action"] = action_at_landmark(r["patient"], r["tau"], r["line"])
            r["card"] = CM.describe(r["patient"], r["tau"], str(d),
                                    mets.get(r["patient"].pid))
            assert r["action"]["treatment_class"] == r["bucket"], (
                "the persisted action names a different treatment class than the one "
                "retrieval matched on")
        # Criteria that carry information in THIS cohort. Sex drops out in breast and
        # histology in colorectal, because there they are satisfied by almost any pair.
        keep = CM.informative([r["card"] for r in records])

        # Support: how many era- and class-matched candidates a landmark can draw on.
        # Checked BEFORE selection, so a query is never admitted whose four-arm contrast
        # is undefined. Evaluated against the whole assigned set; the confirmation carve
        # removes at most N_CONFIRM of them and the 2K floor leaves ample margin.
        def support(i: int, among: list[int]) -> int:
            return sum(1 for j in among
                       if j != i and bucket[j] == bucket[i]
                       and np.isfinite(year[j]) and np.isfinite(year[i])
                       and abs(year[j] - year[i]) <= ERA_WINDOW)

        # 1. Confirmation carve, FIRST, before anything else touches these patients, and
        #    drawn only from landmarks that could themselves support all four arms.
        rng = np.random.default_rng(SEED)
        everyone = list(range(len(records)))
        order = list(rng.permutation(len(records)))
        supported = [i for i in order if support(i, everyone) >= 2 * K]
        confirm = {pids[i] for i in supported[:min(N_CONFIRM, len(supported))]}
        pool = [i for i, pid in enumerate(pids) if pid not in confirm]

        # 2. Development queries, stratified by depth. Walk the seeded order and take the
        #    first N per stratum that clear the support floor. Selection never looks at
        #    the observed future, at outcome magnitude, or at how SMALL the radius is --
        #    only at whether the pre-specified floor is crossed at all.
        queries: list[int] = []
        unsupported = 0
        for d in DEPTHS:
            here = [i for i in pool if depth[i] == d]
            rng.shuffle(here)
            taken = []
            for i in here:
                if len(taken) >= N_DEV_PER_DEPTH:
                    break
                if support(i, pool) >= 2 * K:
                    taken.append(i)
                else:
                    unsupported += 1
            queries.extend(taken)

        # 3. Retrieval. Candidates share cancer AND treatment class; the two cards are
        #    the nearest k and the furthest k, disjoint by construction.
        rows, radii, separations, dropped = [], [], [], 0
        skipped_patients: set[str] = set()
        for qi in queries:
            # Era-matched as well as class-matched, for BOTH cards. The same treatment
            # class spans many years and exact drug names betray the era, so an
            # era-blind control would let the correct card carry historical context the
            # wrong card lacks.
            candidates = [i for i in pool
                          if i != qi and bucket[i] == bucket[qi]
                          and np.isfinite(year[i]) and np.isfinite(year[qi])
                          and abs(year[i] - year[qi]) <= ERA_WINDOW]
            if len(candidates) < 2 * K:
                dropped += 1
                contrast_ok = False
            else:
                contrast_ok = True
            if not candidates:
                continue
            # RELEVANT CARD: the k patients agreeing on the most clinical criteria,
            # among those already comparable on cancer, treatment and era. Ties are broken
            # by a seeded shuffle, never by outcome and never by how the future looks.
            score = np.array([CM.similarity(records[qi]["card"], records[j]["card"], keep)
                              for j in candidates])
            jitter = np.random.default_rng(SEED + qi).random(len(candidates)) * 1e-6
            near = list(np.argsort(-(np.nan_to_num(score, nan=-1.0) + jitter)))
            contrast_ok = contrast_ok and len(near) >= K
            correct = [candidates[j] for j in near[:K]]

            # CONTROL CARD: drawn at RANDOM from the same cancer and era, with no
            # constraint on treatment, stage, driver, state or line. Previously it was the
            # k FURTHEST patients inside the hard-matched stratum, which made it thin
            # wherever the stratum was small -- pools near the 2k floor left only three
            # patients between the two cards, and achieved separation collapsed with pool
            # size (r = +0.48 to +0.74). A random draw removes that failure mode and is a
            # genuinely not-like-me control. Era is kept because otherwise the control
            # would show obsolete drugs, and part of any advantage would be calendar
            # calibration rather than patient relevance.
            wrong = []
            if contrast_ok:
                same_era = [j for j in pool
                            if j != qi and pids[j] not in {pids[c] for c in correct}
                            and np.isfinite(year[j]) and np.isfinite(year[qi])
                            and abs(year[j] - year[qi]) <= ERA_WINDOW]
                if len(same_era) >= K:
                    wrong = [same_era[j] for j in np.random.default_rng(
                        SEED * 7 + qi).permutation(len(same_era))[:K]]
                else:
                    contrast_ok = False
            assert not (set(correct) & set(wrong)), "donor sets overlap"
            # TWO radii. The molecular term is an additive adjustment, not one of the
            # Euclidean coordinates, so the adjusted radius is not comparable between
            # molecular-known and molecular-unknown queries. The BASE radius is the
            # primary support measure; the adjusted radius describes what retrieval did.
            # SUPPORT is the WEAKEST donor admitted to the card -- the score of the 20th
            # -- not the average. The mean says how good the card is on the whole; the
            # boundary says how far we had to reach to fill it, which is the quantity that
            # tells us whether this patient had support at all. A card averaging 0.7 whose
            # 20th donor scores -0.2 is not a well-supported card.
            radius = float(score[near[K - 1]]) if len(near) >= K else float("nan")
            radius_adjusted = float(np.nanmean(score[near[:K]]))   # card quality
            far_mean = (float(np.nanmean(
                [CM.similarity(records[qi]["card"], records[j]["card"], keep)
                 for j in wrong]))
                if contrast_ok and wrong else float("nan"))
            radii.append(radius)
            if contrast_ok:
                separations.append(radius_adjusted - far_mean)
            rows.append({
                "query_id": f"{cohort}:{pids[qi]}",
                "cancer": cohort, "patient": pids[qi], "tau_months": records[qi]["tau"],
                "diagnosis_year": records[qi].get("diagnosis_year"),
                "landmark_year": records[qi].get("year"),

                "history_span_years": (
                    None if records[qi].get("diagnosis_year") is None
                    else [records[qi]["diagnosis_year"], records[qi]["year"]]),
                "line": records[qi]["line"], "depth": records[qi]["depth"],
                "treatment_class": records[qi]["bucket"],
                # A_tau, persisted. Identical in all three arms: the arms differ only in
                # the reference card, never in the patient or the action.
                "action": records[qi]["action"],
                "n_candidates": len(candidates),
                "landmark_unnamed": bool(unnamed[qi]),
                # Entirely unnamed investigational regimens identify no therapy, so they
                # sit outside the primary treatment-conditioned analysis and are reported
                # separately rather than dropped.
                # Out of the primary set when the landmark's own boundary is unresolved:
                # a second DISTINCT regimen recorded in the same month means we cannot say
                # which was the index treatment. This is an exposure-definition exclusion,
                # not an outcome one -- but it does remove patients whose treatment changed
                # within the landmark month, which correlates with early toxicity and
                # progression, so they are retained in the file and analysed as a
                # pre-specified sensitivity stratum rather than deleted.
                "codated_regimen": bool(
                    records[qi]["action"]["n_concurrent_same_date"] > 0),
                "primary_analysis_eligible": bool(
                    not unnamed[qi] and contrast_ok
                    and records[qi]["action"]["n_concurrent_same_date"] == 0),
                "contrast_eligible": contrast_ok,
                "support_boundary": radius,
                "mean_agreement_relevant": radius_adjusted,
                "mean_agreement_generic": far_mean,
                "separation": (radius_adjusted - far_mean if contrast_ok
                               else float("nan")),
                "molecular_driver": str(driver[qi]),
                # KNOWN-KNOWN denominator. Counting unknown == unknown as agreement would
                # report the two arms as concordant on the strength of shared missingness,
                # which is not a shared genotype and not what the offset rewards.
                "correct_driver_agreement": known_agreement(driver[qi], driver[correct]),
                "correct_driver_comparable": int(comparable(driver[qi], driver[correct])),
                "wrong_driver_agreement": (known_agreement(driver[qi], driver[wrong])
                                           if contrast_ok else float("nan")),
                "wrong_driver_comparable": (int(comparable(driver[qi], driver[wrong]))
                                            if contrast_ok else 0),
                "correct_depth_agreement": float(np.mean(
                    [depth[i] == depth[qi] for i in correct])),
                "wrong_depth_agreement": (float(np.mean(
                    [depth[i] == depth[qi] for i in wrong])) if contrast_ok
                    else float("nan")),
                # The donor's LANDMARK is persisted, not just their identifier. It is
                # the landmark used for matching, and the card must render that exact
                # treatment, era and future. Reconstructing it later -- picking some
                # middle line -- would show evidence from a different treatment and a
                # different year than the one retrieval actually matched on.
                "relevant_donors": [donor_record(records, pids, i) for i in correct],
                "generic_donors": [donor_record(records, pids, i) for i in wrong]})

        def donor_ids(row: dict) -> set[str]:
            return {d["patient"] for d in row["relevant_donors"] + row["generic_donors"]}

        leak = {r["query_id"] for r in rows if confirm & donor_ids(r)}
        assert not leak, f"confirmation patients reached {len(leak)} donor pools"
        assert all(r["patient"] not in donor_ids(r) for r in rows), "self-donation"
        missing = [r["query_id"] for r in rows
                   for d in r["relevant_donors"] + r["generic_donors"]
                   if d["year"] is None or d["tau_months"] is None]
        assert not missing, f"{len(missing)} donor landmarks incomplete"

        all_queries.extend(rows)
        counts = pd.Series([r["depth"] for r in rows]).value_counts().to_dict()
        manifest["cohorts"][cohort] = {
            "landmark_audit": audit,
            "queries_lacking_support": unsupported,
            "patients_with_landmark": len(records),
            "assigned_depths": pd.Series(depth).value_counts().to_dict(),
            "confirmation_held": len(confirm),
            "donor_pool": len(pool),
            "development_queries": len(rows),
            "queries_by_depth": counts,
            "median_support_boundary": float(np.median(radii)),
            "median_mean_agreement_relevant": float(np.median(
                [r["mean_agreement_relevant"] for r in rows])),
            "median_mean_agreement_generic": float(np.nanmedian(
                [r["mean_agreement_generic"] for r in rows])),
            "score_scale": "-1 (every criterion disagrees) to +1 (every criterion "
                           "agrees); unknown contributes 0 and never shrinks the "
                           "denominator",
            "median_wrong_minus_correct_distance": float(np.median(separations)),
            "contrast_ineligible": dropped,
            "queries_with_unnamed_landmark": int(sum(1 for r in rows
                                                     if r["landmark_unnamed"])),
            "queries_total": len(rows),
            "donor_pool_patients": len(pool),
            "unnamed_donor_patients_excluded": len(skipped_patients),
            "molecular_known_share": float(np.mean([r["molecular_driver"] != "unknown"
                                                    for r in rows])),
            "median_correct_driver_agreement": float(np.nanmedian(
                [r["correct_driver_agreement"] for r in rows])),
            "median_wrong_driver_agreement": float(np.nanmedian(
                [r["wrong_driver_agreement"] for r in rows])),
            "driver_agreement_denominator": "known-known donor pairs only",
            "median_correct_driver_comparable": float(np.median(
                [r["correct_driver_comparable"] for r in rows])),
            "median_wrong_driver_comparable": float(np.median(
                [r["wrong_driver_comparable"] for r in rows])),
            "median_correct_depth_agreement": float(np.median(
                [r["correct_depth_agreement"] for r in rows])),
            "median_wrong_depth_agreement": float(np.nanmedian(
                [r["wrong_depth_agreement"] for r in rows])),
            "match_criteria": sorted(keep),
            "prefix_features": len(cols)}
        m = manifest["cohorts"][cohort]
        a = audit
        print(f"\n{cohort}: {a['candidate_landmarks']} candidate regimen-start landmarks "
              f"-> {a['valid_landmarks']} valid")
        for why, n in sorted(a["reasons"].items(), key=lambda kv: -kv[1]):
            print(f"      {n:5d} landmarks excluded: {why}")
        print(f"    patients with >=1 valid landmark {a['patients_with_a_valid_landmark']}"
              f"; with none {a['patients_with_no_valid_landmark']}; "
              f"{unsupported} otherwise-valid landmarks passed over for lacking 2k support")
        print(f"{cohort}: {m['patients_with_landmark']} with a landmark -> "
              f"{m['confirmation_held']} held for confirmation, {m['donor_pool']} donors, "
              f"{m['development_queries']} queries {counts}")
        print(f"    agreement: relevant {m['median_mean_agreement_relevant']:+.2f} vs "
              f"generic {m['median_mean_agreement_generic']:+.2f} "
              f"(scale -1 to +1); weakest donor admitted "
              f"{m['median_support_boundary']:+.2f}; advantage "
              f"{m['median_wrong_minus_correct_distance']:+.2f}; depth agreement correct "
              f"{m['median_correct_depth_agreement']:.2f} vs wrong "
              f"{m['median_wrong_depth_agreement']:.2f}; "
              f"{m['contrast_ineligible']} lack 2k candidates; "
              f"{m['queries_with_unnamed_landmark']}/{m['queries_total']} queries have "
              f"an unnamed landmark regimen; "
              f"{m['unnamed_donor_patients_excluded']}/{m['donor_pool_patients']} donor "
              f"patients excluded as unnamed")
        print(f"    molecular known for {m['molecular_known_share']:.0%} of queries; "
              f"driver agreement (known-known only) correct "
              f"{m['median_correct_driver_agreement']:.2f} vs "
              f"wrong {m['median_wrong_driver_agreement']:.2f}; comparable pairs "
              f"{m['median_correct_driver_comparable']:.0f}/{K} vs "
              f"{m['median_wrong_driver_comparable']:.0f}/{K}")

    path = OUT / "queries.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in all_queries))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                                   # noqa: BLE001
        commit = "unknown"
    manifest["git_commit"] = commit
    manifest["queries_total"] = len(all_queries)
    manifest["queries_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{len(all_queries)} queries -> {path}")
    print(f"manifest checksum {manifest['queries_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
