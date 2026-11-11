#!/usr/bin/env python3
"""Do the enriched operators carry information in REAL patients? GENIE-BPC, three cancers.

v2 after review. v1 confounded clinical history with the observation process, and five of
its defects were verified rather than argued:

  * treatment rows are one per AGENT, not per regimen (1.6, 1.6 and 2.6 rows per regimen
    in NSCLC, BRCA and CRC), so every line count in v1 was inflated;
  * 17 / 50 / 28% of patients have no disease-status assessment at all before the
    landmark, and v1 encoded them as an all-zero current state, indistinguishable from a
    patient known to be in none of those states;
  * 41 / 58 / 36% have no assessment inside the outcome window, and v1 carried the last
    status forward indefinitely, so occupancy outcomes were partly imputed persistence;
  * imaging and med-onc statuses were appended without reconciliation, and they disagree
    at the same timepoint for 25-42% of patients;
  * assessment counts before and after the landmark correlate 0.40-0.49, so history can
    predict who keeps being assessed, which mechanically drives event counts.

v2 fixes each of those and, per the same review, treats observation as part of the
baseline rather than as something to ignore:

  CANONICAL REGIMENS   rows collapsed by regimen number; start = first agent start,
                       stop = last agent stop.
  RECONCILED STATUS    imaging takes precedence over med-onc at the same timepoint; ties
                       inside one source resolve to the worse status.
  OBSERVABILITY IN X   months since last assessment, prefix assessment count, an explicit
                       "no status on record yet" indicator, and the current treatment
                       class all sit in the BASELINE, so history is never credited for
                       predicting who gets scanned.
  BOUNDED CARRY        a status is valid for STATUS_VALIDITY months. Beyond that the
                       patient is UNOBSERVED and contributes to no occupancy. Outcomes
                       therefore measure documented time, never imputed persistence, and
                       the documented fraction is reported.
  BLOCK-NORMALISED     the joint primary score weights four outcome blocks equally
                       (survival, disease state, events, treatment) instead of weighting
                       fifteen partly redundant coordinates equally.
  CONDITIONAL NULL     history is residualised on the baseline design and only the
                       residual is permuted, so the test is incremental information given
                       context, current state and observation intensity -- not "real
                       history versus clinically incoherent history".
  REPEATED FOLDS       three fold assignments, averaged.

GOVERNANCE. GENIE-BPC is under a data use agreement. Aggregate results only, written to a
gitignored directory. No patient-level row, identifier or trajectory is serialised.

Run:  uv run python scripts/explore/genie_operator_registry_v2.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disease_extent_v1 as DE  # noqa: E402
sys.path.insert(0, str(ROOT_SRC := Path(__file__).resolve().parents[2] / "src"))
from synthonccases import disease_state as DS  # noqa: E402
import genie_operator_registry_v1 as V1  # noqa: E402
import trajectory_feature_poc_v1 as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GENIE_ROOT = Path.home() / "GenieBPC"
COHORTS = {"NSCLC": "NSCLC_Genie", "BRCA": "BRCA_GENIE", "CRC": "CRC_GENIE"}
OUT = ROOT / "outputs" / "exploratory_genie_operators_v2"
DAYS = 30.4
TAU = 12.0
H = 12.0
TRAIL = 3.0
STATUS_VALIDITY = 4.0
SAME_DAY = 0.5 / DAYS     # regimens recorded on the identical day are ONE action
SEED = 20260824
FOLDS = 5
REPEATS = 3

STATUS = V1.STATUS
STATES = V1.STATES
SEVERITY = {"responding": 0, "stable": 1, "mixed": 2, "progressing": 3}


@dataclass
class Patient:
    pid: str
    context: dict
    status: list[tuple] = field(default_factory=list)
    sites: list[tuple] = field(default_factory=list)
    presence: list[tuple] = field(default_factory=list)   # (months, evidence/no/equivocal)
    lines: list[dict] = field(default_factory=list)
    death: float = np.inf
    last_contact: float = np.inf


def documented(p: Patient, end: float) -> dict[str, list[tuple]]:
    """State windows that were actually DOCUMENTED.

    A status holds for STATUS_VALIDITY months or until the next assessment, whichever
    comes first. After that the patient is unobserved and contributes to nothing. This is
    the difference between measuring occupancy and inventing it.
    """
    out: dict[str, list[tuple]] = {}
    for i, (when, what) in enumerate(p.status):
        stop = p.status[i + 1][0] if i + 1 < len(p.status) else np.inf
        stop = min(stop, when + STATUS_VALIDITY, end)
        if stop > when:
            out.setdefault(what, []).append((when, stop))
    return out


def observed_span(p: Patient, a: float, b: float) -> float:
    windows = [(w, min(w + STATUS_VALIDITY, b)) for w, _ in p.status if w < b]
    return M.overlap(M.merge(windows), a, b) if windows else 0.0


def modalities(label: str) -> frozenset[str]:
    """EVERY modality present, not the first one on a priority list.

    `V1.bucket` returns a single first match, so FOLFOX plus bevacizumab is "targeted" --
    the same class as cetuximab monotherapy -- and the chemotherapy backbone disappears.
    That affects 37% of colorectal actions, 20% of breast and 12% of lung, and it is why
    letrozole plus palbociclib looked like a targeted regimen with no endocrine component.

    The treatment class is the SET. It is rendered as a joined string so that matching
    stays plain string equality while the string finally says what the treatment is.
    """
    text = str(label).lower()
    found = {name for name, drugs in (("immunotherapy", V1.IO_DRUGS),
                                      ("targeted", V1.TARGETED),
                                      ("endocrine", V1.ENDOCRINE),
                                      ("chemotherapy", V1.CHEMO))
             if any(d in text for d in drugs)}
    return frozenset(found) or frozenset({"other"})


def class_label(label: str) -> str:
    return " + ".join(sorted(modalities(label)))


def regimen_sequence(genie: Path) -> dict[tuple, int]:
    """`regimen_number_within_cancer`, the curated sequence for THIS cancer.

    `regimen_number` counts a patient's regimens across all their cancers, so a patient
    treated for an earlier tumour would have their lung cancer's first regimen numbered
    four -- and the depth stratum would be wrong by exactly that offset.
    """
    frame = pd.read_csv(genie / "regimen_cancer_level_dataset.csv", low_memory=False)
    return {(str(r["record_id"]), int(r["ca_seq"]), int(r["regimen_number"])):
            int(r["regimen_number_within_cancer"])
            for _, r in frame.iterrows()
            if pd.notna(r.get("regimen_number_within_cancer"))}


def load(genie: Path) -> list[Patient]:
    dx = pd.read_csv(genie / "data_timeline_cancer_diagnosis.txt", sep="\t",
                     low_memory=False)
    if "INDEX_CANCER" in dx:
        dx = dx[dx["INDEX_CANCER"].astype(str).str.lower().isin({"yes", "true", "1"})]
    context = {}
    for _, r in dx.iterrows():
        context.setdefault(str(r["PATIENT_ID"]), {
            "age": pd.to_numeric(r.get("AGE_DX"), errors="coerce"),
            "histology": str(r.get("CA_HIST_ADENO_SQUAMOUS"))})
    # Stage comes from `stage_dx_iv`, NOT from `CA_TX_PRE_PATH_STAGE` -- the latter is a
    # yes/no flag about treatment given before the pathological specimen, and reading it as
    # a stage group made `stage_iv` constant zero for every patient in every cohort.
    for pid, rec in DE.load_metastasis(genie).items():
        context.setdefault(pid, {})["mets"] = rec
    clinical = pd.read_csv(genie / "data_clinical_patient.txt", sep="\t", comment="#",
                           low_memory=False)
    sex = dict(zip(clinical["PATIENT_ID"].astype(str),
                   clinical.get("SEX", pd.Series(["u"] * len(clinical))).astype(str),
                   strict=False))
    surv = pd.read_csv(genie / "data_clinical_supp_survival.txt", sep="\t",
                       low_memory=False)
    surv.columns = [c.strip() for c in surv.columns]
    pid_col = surv.columns[0]
    loc = surv.columns.get_loc("OS from Diagnosis Status")
    os_status, os_months = surv.columns[loc], surv.columns[loc + 1]

    patients: dict[str, Patient] = {}
    for _, r in surv.iterrows():
        months = pd.to_numeric(r[os_months], errors="coerce")
        if not np.isfinite(months):
            continue
        pid = str(r[pid_col])
        ctx = dict(context.get(pid, {}))
        ctx["sex"] = sex.get(pid, "u")
        p = Patient(pid=pid, context=ctx)
        dead = "1" in str(r[os_status]) or "dead" in str(r[os_status]).lower()
        p.death = float(months) if dead else np.inf
        p.last_contact = float(months)
        patients[pid] = p

    # RECONCILED status: one label per patient per timepoint. Imaging outranks med-onc;
    # ties inside a source resolve to the worse status.
    records: dict[tuple, tuple] = {}
    for name, rank in (("data_timeline_imaging.txt", 0), ("data_timeline_medonc.txt", 1)):
        f = pd.read_csv(genie / name, sep="\t", low_memory=False)
        f["_m"] = pd.to_numeric(f["START_DATE"], errors="coerce") / DAYS
        f = f[f["_m"].notna() & (f["_m"] >= 0)]
        for _, r in f.iterrows():
            pid, when = str(r["PATIENT_ID"]), float(r["_m"])
            if pid not in patients:
                continue
            if isinstance(r.get("SCAN_SITES"), str) and r["SCAN_SITES"].strip():
                patients[pid].sites.append((when, V1.parse_sites(r["SCAN_SITES"])))
            state = STATUS.get(str(r.get("CURATED_CANCER_STATUS")))
            if state is None:
                continue
            key = (pid, round(when, 3))
            best = records.get(key)
            if best is None or (rank, -SEVERITY[state]) < (best[1], -SEVERITY[best[0]]):
                records[key] = (state, rank)
    for (pid, when), (state, _) in records.items():
        patients[pid].status.append((when, state))
    for p in patients.values():
        p.status.sort(key=lambda x: x[0])
        p.sites.sort(key=lambda x: x[0])

    # CANONICAL regimens: one row per regimen, not one per agent row, and restricted to
    # the SINGLE index cancer. Grouping on (patient, regimen number) alone merged regimens
    # belonging to two different index cancers for the handful of patients who have two,
    # and `regimen_number` counts across cancers while the sequence we want is the one
    # within this cancer.
    keep = DE.index_cancer(genie)
    within = regimen_sequence(genie)
    tx = pd.read_csv(genie / "data_timeline_treatment.txt", sep="\t", low_memory=False)
    tx["_start"] = pd.to_numeric(tx["START_DATE"], errors="coerce") / DAYS
    tx["_stop"] = pd.to_numeric(tx["STOP_DATE"], errors="coerce") / DAYS
    tx = tx[tx["_start"].notna() & (tx["_start"] >= 0)]
    tx = tx[[keep.get(str(a)) == b for a, b in
             zip(tx["PATIENT_ID"], tx["CA_SEQ"], strict=True)]]
    for (pid, seq, number), group in tx.groupby(
            ["PATIENT_ID", "CA_SEQ", "REGIMEN_NUMBER"], sort=False):
        p = patients.get(str(pid))
        if p is None:
            continue
        number = within.get((str(pid), int(seq), int(number)), number)
        start = float(group["_start"].min())
        stop = group["_stop"].max()
        stop = float(stop) if np.isfinite(stop) else start
        label = " ".join(sorted({str(v) for v in group["REGIMEN"].dropna()})) or \
            " ".join(sorted({str(v) for v in group["AGENT"].dropna()}))
        p.lines.append({"start": start, "stop": max(stop, start),
                        "bucket": class_label(label),
                        "modalities": modalities(label), "label": label,
                        "number": pd.to_numeric(number, errors="coerce")})
    # SIMULTANEOUS RECORDS ARE ONE ACTION. Regimens sharing an identical start date are
    # not an ambiguous ordering -- there is no ordering to resolve. They are one treatment
    # decision split across drug records, and in these cohorts they are overwhelmingly
    # concurrent BY DESIGN:
    #   breast   leuprolide or goserelin with tamoxifen / letrozole / anastrozole, i.e.
    #            ovarian suppression plus endocrine therapy, standard of care for
    #            premenopausal hormone-receptor-positive disease;
    #   colorectal  floxuridine (hepatic arterial infusion) with systemic FOLFIRI or
    #            FOLFOX, standard for liver-dominant metastatic disease.
    # Excluding them removed those two populations specifically, which is a biased
    # exclusion rather than a data-quality one. Merging is also NOT the line-of-therapy
    # inference we rejected: that needed progression and intent to link SEQUENTIAL
    # regimens, whereas these are simultaneous and the union is simply what was started.
    # PRESENCE: the oncologist's global answer to "is there cancer". A separate question
    # from what the disease is DOING, and the only field in GENIE that can establish no
    # evidence of disease -- the response vocabulary has no complete-response category.
    md = pd.read_csv(genie / "med_onc_note_level_dataset.csv", low_memory=False)
    for r in md.itertuples():
        pt = patients.get(str(r.record_id))
        when = getattr(r, "dx_md_visit_mos", None)
        if pt is None or pd.isna(when):
            continue
        text = str(getattr(r, "md_ca", "")).lower()
        if "no, the impression" in text:
            pt.presence.append((float(when), "no evidence"))
        elif text.startswith("yes,"):
            pt.presence.append((float(when), "evidence"))
        elif "uncertain" in text or "indeterminate" in text or "equivocal" in text:
            pt.presence.append((float(when), "equivocal"))
    for pt in patients.values():
        pt.presence.sort()

    for p in patients.values():
        p.lines.sort(key=lambda x: (x["start"], x.get("number") or 0))
        merged: list[dict] = []
        for ln in p.lines:
            if merged and abs(ln["start"] - merged[-1]["start"]) < SAME_DAY:
                prev = merged[-1]
                prev["label"] = f"{prev['label']}, {ln['label']}"
                prev["stop"] = max(prev["stop"], ln["stop"])
                prev["bucket"] = class_label(prev["label"])
                prev["modalities"] = modalities(prev["label"])
                prev["curated_numbers"].append(ln["number"])
                prev["n_records"] += 1
            else:
                merged.append({**ln, "curated_numbers": [ln["number"]], "n_records": 1})
        # Episode index: distinct treatment START DATES, in order. Once simultaneous
        # records are unified this is the faithful sequence; the curated numbers are kept
        # alongside so the derivation stays auditable.
        for i, ln in enumerate(merged, 1):
            ln["curated_number"] = min(
                (n for n in ln["curated_numbers"] if np.isfinite(n)), default=np.nan)
            ln["number"] = float(i)
        p.lines = merged
    return [p for p in patients.values() if p.status or p.lines]


def recent(events: list[tuple], tau: float, window: float = STATUS_VALIDITY):
    """The most recent event STRICTLY at or before tau, within the validity window.

    Every prefix quantity passes through here, so the temporal boundary is enforced in one
    place rather than restated at each call site.
    """
    got = [(w, v) for w, v in events if tau - window <= w <= tau + 1e-9]
    return got[-1][1] if got else None


def relapse_signal(p: Patient, tau: float, mets, window: float) -> bool:
    """Evidence that disease returned or was never controlled, inside the window.

    A new REGIMEN is not itself a relapse signal: for localized disease the
    chemotherapy-to-endocrine handover is the planned next step of an adjuvant programme,
    and 264 of the localized breast transitions in this cohort are exactly that. Only a
    documented progression, a newly documented distant site, or a regimen started with
    palliative rather than adjuvant intent counts -- and GENIE does not record intent, so
    we use the first two.
    """
    if any(tau < w <= tau + window and what == "progressing" for w, what in p.status):
        return True
    if mets is not None:
        return any(tau < month <= tau + window for month, _site in mets.sites)
    return False


def genie_state(p: Patient, tau: float, mets) -> DS.DiseaseState:
    """The canonical state as GENIE lets us see it, from PRE-LANDMARK data only.

    `adjuvant_sequence` is gone. It claimed to establish a post-operative programme and
    actually returned "three or fewer regimens so far", and the inference it gated read the
    patient's SUBSEQUENT follow-up to decide their state at the landmark. Both are removed
    rather than repaired: GENIE records no surgery, so the fact it was standing in for is
    not establishable here at all.
    """
    # RESOLVED BY TIMESTAMP, not by a window boolean. A site appearing four months ago
    # was forcing "progressing" even when an assessment last week said stable -- the older
    # evidence overriding the newer one. A new site only speaks louder if it is newer.
    last_look = max([w for w, _v in p.status if w <= tau + 1e-9] + [-np.inf])
    newest_site = max([m for m, _s in (mets.sites if mets else [])
                       if tau - STATUS_VALIDITY <= m <= tau + 1e-9] + [-np.inf])
    return DS.from_genie(
        presence=recent(p.presence, tau),
        curated_status=recent(p.status, tau),
        new_distant_site=bool(np.isfinite(newest_site) and newest_site >= last_look))


def prefix_features(p: Patient, tau: float) -> dict[str, float]:
    f: dict[str, float] = {}
    ctx = p.context
    seen = [(w, s) for w, s in p.status if w <= tau]
    windows = documented(p, tau)

    # ---- L0 context -------------------------------------------------------------------
    f["age_at_diagnosis"] = float(pd.to_numeric(ctx.get("age"), errors="coerce"))
    f["sex_female"] = 1.0 if str(ctx.get("sex", "")).lower().startswith("f") else 0.0
    f["histology_adeno"] = 1.0 if "adeno" in str(ctx.get("histology", "")).lower() else 0.0
    f["histology_squamous"] = 1.0 if "squam" in str(ctx.get("histology", "")).lower() \
        else 0.0
    # TWO stage concepts, never one. Diagnosis stage is fixed context; whether metastatic
    # disease has been documented BY THE LANDMARK is timed state, and in these cohorts
    # 91-94% of Stage I-III patients are metastatic by their landmark -- so the diagnosis
    # stage alone describes the majority of them wrongly at the moment being predicted.
    e = DE.extent_at_landmark(ctx.get("mets"), tau)
    f["stage_at_diagnosis_iv"] = 1.0 if e.stage_at_diagnosis == "IV" else 0.0
    f["stage_at_diagnosis_unknown"] = 1.0 if e.stage_at_diagnosis == "unknown" else 0.0
    f["metastatic_documented_by_landmark"] = 1.0 if e.documented == "yes" else 0.0
    f["metastatic_extent_unknown"] = 1.0 if e.documented == "unknown" else 0.0
    # NaN, not 0, where the cohort collects no site detail. A zero would assert that the
    # sites were looked for and none found; BRCA simply does not record them, and the
    # missingness machinery should see that rather than a false negative count.
    f["n_metastatic_sites_by_landmark"] = (float(len(e.sites))
                                           if e.site_detail_available else np.nan)

    # ---- L1 current state, INCLUDING observability and current treatment --------------
    # Three-way encoding, so the baseline keeps WHAT was last recorded and separately
    # knows HOW OLD it is. Marking a stale label simply "unknown" would have thrown the
    # label away, and history features can partly reconstruct it -- which would inflate
    # the history gain by weakening the comparator rather than by carrying information.
    stale = (not seen) or (tau - seen[-1][0]) > STATUS_VALIDITY
    # CANONICAL state, one vocabulary shared with the generated side. The old encoding
    # spread the same fact across fresh_/stale_/never_ indicators in a four-state
    # vocabulary that had no way to say "no active disease", so a patient in remission was
    # recorded as having no state at all.
    ds = genie_state(p, tau, ctx.get("mets"))
    for state in DS.State.__args__:
        f[f"now_state_{state}"] = 1.0 if ds.state == state else 0.0
    f["now_state_inferred"] = 1.0 if ds.evidence == "clinically_inferred" else 0.0
    # Staleness is MEASUREMENT, kept separate from the state itself so that how recently
    # we looked never masquerades as what we saw.
    f["now_status_is_stale"] = 1.0 if (seen and stale) else 0.0
    f["now_never_observed"] = 1.0 if not seen else 0.0
    f["months_since_last_assessment"] = float(tau - seen[-1][0]) if seen else np.nan
    f["n_assessments_prefix"] = float(len(seen))
    f["assessments_per_month"] = float(len(seen)) / tau
    f["months_documented_prefix"] = observed_span(p, 0.0, tau)
    # SCAN_SITES is the region IMAGED -- "Chest", "Full body" -- not the site of disease.
    # Named `now_n_sites` it read as metastatic burden while measuring how broadly the
    # patient was scanned, which is a measurement variable, not a clinical state.
    sites_now = next((s for w, s in reversed(p.sites) if w <= tau), set())
    f["now_n_regions_imaged"] = float(len(sites_now))
    on_now = [ln for ln in p.lines if ln["start"] <= tau < ln["stop"]]
    f["now_on_treatment"] = 1.0 if on_now else 0.0
    for name in ("chemotherapy", "immunotherapy", "targeted", "endocrine"):
        # Membership, not equality: a patient on chemotherapy plus a biologic is on
        # both, and the previous test set exactly one flag and silenced the other.
        f[f"now_tx_{name}"] = 1.0 if on_now and name in on_now[0]["modalities"] else 0.0
    # STRICTLY BEFORE the landmark. At a treatment-start landmark the new line begins at
    # exactly tau, and including it would let history see the very decision it is meant to
    # precede -- the new regimen is A, never H.
    started = [ln for ln in p.lines if ln["start"] < tau - 1e-9]
    # THE REGIMEN STARTING AT THE LANDMARK, when there is one. The `start <= tau < stop`
    # test excludes a regimen recorded as a single day -- stop equals start -- so the
    # fallback counted PRIOR regimens instead and returned 1 for a curated regimen 2. The
    # matcher scored that wrong depth while the query universe and the rendered summary
    # both used the curated number, so a donor could be matched on a sequence position the
    # patient was never at.
    at_landmark = next((ln for ln in p.lines if abs(ln["start"] - tau) < 1e-9
                        and np.isfinite(ln.get("number", np.nan))), None)
    f["now_line_number"] = (float(at_landmark["number"]) if at_landmark
                            else float(on_now[0]["number"]) if on_now
                            and np.isfinite(on_now[0]["number"])
                            else float(len(started)))
    f["now_months_on_current_line"] = float(tau - on_now[0]["start"]) if on_now else 0.0
    f["months_since_last_line_start"] = float(tau - started[-1]["start"]) if started \
        else np.nan

    # ---- L2 treatment history ---------------------------------------------------------
    completed = [ln for ln in started if ln["stop"] <= tau]
    f["n_prior_completed_lines"] = float(len(completed))
    f["months_on_treatment"] = M.overlap(
        M.merge([(ln["start"], min(ln["stop"], tau)) for ln in started]), 0.0, tau)
    f["months_off_treatment"] = max(0.0, min(p.death, tau) - f["months_on_treatment"])
    for name in ("chemotherapy", "immunotherapy", "targeted", "endocrine"):
        f[f"months_on_{name}"] = M.overlap(
            M.merge([(ln["start"], min(ln["stop"], tau)) for ln in started
                     if name in ln["modalities"]]), 0.0, tau)
    durations = [ln["stop"] - ln["start"] for ln in completed]
    f["most_recent_line_duration"] = float(durations[-1]) if durations else np.nan
    f["median_prior_line_duration"] = float(np.median(durations)) if durations else np.nan
    f["n_treatment_buckets_used"] = float(len({ln["bucket"] for ln in started}))

    # ---- L3 disease history -----------------------------------------------------------
    for state in STATES:
        f[f"months_{state}"] = M.overlap(windows.get(state, []), 0.0, tau)
    entries = {state: [] for state in STATES}
    previous = None
    for when, what in seen:
        if what != previous:
            entries[what].append(when)
        previous = what
    f["n_progression_episodes"] = float(len(entries["progressing"]))
    f["n_response_episodes"] = float(len(entries["responding"]))
    f["longest_response_months"] = float(max(
        [b - a for a, b in windows.get("responding", [])] + [0.0]))
    f["longest_progression_months"] = float(max(
        [b - a for a, b in windows.get("progressing", [])] + [0.0]))
    f["months_since_last_progression"] = float(tau - max(entries["progressing"])) \
        if entries["progressing"] else np.nan
    f["months_since_last_response"] = float(tau - max(entries["responding"])) \
        if entries["responding"] else np.nan
    f["recent_progression_3mo"] = 1.0 if any(t > tau - TRAIL
                                             for t in entries["progressing"]) else 0.0
    ever: set[str] = set()
    first = p.sites[0][1] if p.sites else set()
    for when, what in p.sites:
        if when <= tau:
            ever |= what
    f["n_regions_imaged_ever"] = float(len(ever))
    f["n_new_regions_imaged_prefix"] = float(len(ever - first))
    # The genuine metastatic-site history, from the curated block rather than from scan
    # regions: sites that appeared AFTER diagnosis and were documented before the landmark.
    f["n_new_metastatic_sites_prefix"] = float(
        sum(1 for month, _s in e.sites if month > 0.0))
    return f


def future_outcomes(p: Patient, tau: float, horizon: float) -> dict[str, float]:
    """Scored from the ONE canonical projection, not from a second derivation.

    The previous version computed responding and progressing occupancy from raw status
    rows: no stable, no no_active_disease, and nothing about distant relapse -- while the
    registry claimed a canonical five-state outcome and the card rendered a third variant.

    `fraction_defined_window` has been removed from the outcome vector. It is censoring
    metadata, not disease state, and for deaths it duplicates months_alive; it governs
    masking and is reported alongside rather than scored as fidelity.
    """
    import canonical_future_v1 as CF          # local import: CF imports this module

    fut = CF.project(p, tau, horizon)
    known_months = sum(v for k, v in fut.months_in.items() if k != "unknown")
    out = {
        "alive_at_horizon": fut.alive_at_horizon,
        "months_alive": fut.months_alive,
        # MASKED, not zeroed. With nothing documented, four zeros are indistinguishable
        # from a patient observed throughout and never in any of these states -- so an
        # unobserved course read as a definite phenotype. Unknown time now governs the
        # block: when no disease time is documented at all, the block is NaN and drops out
        # of the score rather than contributing a confident-looking zero vector.
        **({k: np.nan for k in ("months_no_active_disease", "months_responding_future",
                                "months_stable_future", "months_progressing_future")}
           if known_months <= 1e-6 else
           {"months_no_active_disease": fut.months_in.get("no_active_disease", 0.0),
            "months_responding_future": fut.months_in.get("responding", 0.0),
            "months_stable_future": fut.months_in.get("stable", 0.0),
            "months_progressing_future": fut.months_in.get("progressing", 0.0)}),
        "n_new_progressions": float(fut.entries_into.get("progressing", 0)),
        "n_new_responses": float(fut.entries_into.get("responding", 0)),
        # NaN, NOT ZERO, where the registry never collected sites (BRCA): a structural
        # absence must not score as "no new metastases".
        "n_new_metastatic_sites_future": (float(len(fut.new_distant_sites))
                                          if fut.sites_collected else float("nan")),
        "months_to_first_new_distant_site": (fut.first_new_distant_site
                                             if fut.first_new_distant_site is not None
                                             else float("nan")),
        "months_on_treatment_future": fut.months_on_treatment,
        # SUBSEQUENT, and REGIMENS not lines. Counting fut.regimens included the index
        # regimen, so every patient appeared to have switched at least once; and these are
        # regimen starts, which is not a line-of-therapy determination.
        "n_subsequent_regimens": float(len(fut.subsequent_regimens)),
        # TREATMENT EVOLUTION. Per-modality months are OVERLAPPING channels -- a
        # chemotherapy-plus-biologic month counts in both -- and must never be summed.
        **{f"months_on_{m}_future": fut.months_on_modality.get(m, 0.0)
           for m in ("chemotherapy", "immunotherapy", "targeted", "endocrine")},
        # NAMED FOR WHAT IT DETECTS. Cisplatin to carboplatin is a regimen switch and
        # not a modality change, so this is invisible to it; exact-regimen concordance
        # stays the separate secondary analysis.
        "months_to_first_modality_change": fut.months_to_first_switch,
        "n_modality_changes": float(max(0, len(fut.modality_sequence) - 1)),
        # REPORTED, NOT SCORED. Unknown is not a disease state, so it is not a fifth
        # coordinate -- but without it a patient nobody assessed and a patient documented
        # as never being in any of the four states produce an identical vector of zeros.
        # It governs masking and weighting; it is not fidelity.
        "_months_state_unknown": fut.months_in.get("unknown", 0.0),
        # THE DENOMINATOR for partial observation. Masking only handles the all-unknown
        # case: one observed stable month among eleven unknown still reports stable=1,
        # which a fully observed patient with one stable month also reports. The scorer
        # must either apply this patient's mask to the generated trajectory as well, or
        # compare occupancy over documented time using this denominator and report
        # coverage separately. It is provided rather than chosen here, because that is a
        # scoring-design decision and not a property of the data.
        "_documented_state_months": known_months,
        "_months_documented_future": observed_span(p, tau, tau + horizon),
        "_defined_window_end": fut.window_end,
    }
    return out


OUTCOME_BLOCKS = {
    "survival": ["alive_at_horizon", "months_alive"],
    # All FOUR occupied states, not two. A patient in remission and one with stable
    # disease were previously both invisible to this block.
    "disease state": ["months_no_active_disease", "months_responding_future",
                      "months_stable_future", "months_progressing_future"],
    "events": ["n_new_progressions", "n_new_responses",
               "n_new_metastatic_sites_future", "months_to_first_new_distant_site"],
    # The most PRACTICE-DEPENDENT block: months on treatment and number of subsequent
    # lines reflect local protocol, tolerance thresholds and trial access. Secondary.
    "treatment": ["months_on_treatment_future", "n_subsequent_regimens",
                  "months_on_chemotherapy_future", "months_on_immunotherapy_future",
                  "months_on_targeted_future", "months_on_endocrine_future",
                  "months_to_first_modality_change", "n_modality_changes"],
}
#: SCORED AT EVERY HORIZON, not only at twelve months. A model can get the one-year
#: picture right and the two-year picture wrong, and reporting a single horizon would
#: never show it. 3/6/12 are complete by construction; 24 carries 5-9% censoring.
#:
#: REPORTED SEPARATELY, NEVER CONCATENATED. The horizons are nested -- the first three
#: months appear in all four -- so stacking twenty coordinates by four horizons into one
#: eighty-dimensional vector would weight early months four times over and call the result
#: a whole-course score.
SCORING_HORIZONS = (3.0, 6.0, 12.0, 24.0)


def future_at_horizons(p: Patient, tau: float,
                       horizons=SCORING_HORIZONS) -> dict[float, dict[str, float]]:
    """The whole outcome vector at each scoring horizon."""
    return {h: future_outcomes(p, tau, h) for h in horizons}
OUTCOMES = [o for v in OUTCOME_BLOCKS.values() for o in v]
BINARY = {"alive_at_horizon"}

LADDER = {
    "L0 fixed context": ["age_at_diagnosis", "sex_female", "histology_adeno",
                         "histology_squamous", "stage_at_diagnosis_iv",
                         "stage_at_diagnosis_unknown"],
    # Observation intensity enters BEFORE clinical state, and on its own rung. Mixing
    # them made "the patient's state matters" indistinguishable from "closely monitored
    # patients are more predictable", which after everything cadence did to the earlier
    # results is not a distinction we can afford to leave blurred.
    "L1 + observation process": ["months_since_last_assessment", "n_assessments_prefix",
                                 "assessments_per_month", "months_documented_prefix",
                                 "now_never_observed", "now_status_is_stale",
                                 "now_n_regions_imaged", "n_regions_imaged_ever",
                                 "n_new_regions_imaged_prefix"],
    "L2 + current clinical state": [f"now_state_{s}" for s in DS.State.__args__] + [
        "now_state_inferred"] + [
        "metastatic_documented_by_landmark", "metastatic_extent_unknown",
        "n_metastatic_sites_by_landmark",
        "now_on_treatment", "now_tx_chemotherapy",
        "now_tx_immunotherapy", "now_tx_targeted", "now_tx_endocrine",
        "now_line_number", "now_months_on_current_line",
        "months_since_last_line_start"],
    "L3 + treatment history": ["n_prior_completed_lines", "months_on_treatment",
                               "months_off_treatment", "months_on_chemotherapy",
                               "months_on_immunotherapy", "months_on_targeted",
                               "months_on_endocrine", "most_recent_line_duration",
                               "median_prior_line_duration", "n_treatment_buckets_used"],
    "L4 + disease history": [f"months_{s}" for s in STATES] + [
        "n_progression_episodes", "n_response_episodes", "longest_response_months",
        "longest_progression_months", "months_since_last_progression",
        "months_since_last_response", "recent_progression_3mo",
        "n_new_metastatic_sites_prefix"],
}


def block_joint_score(raw, ystd, folds, groups, seed, alphas) -> float:
    """Four outcome blocks weighted equally, not fifteen partly redundant coordinates."""
    per_block = []
    index = 0
    for names in OUTCOME_BLOCKS.values():
        sse, tot = 0.0, 0.0
        for _ in names:
            y = ystd[:, index]
            alpha = None if alphas is None else alphas[index]
            pred = M.oof_predict(raw, y, folds, groups, seed, alpha)
            sse += float(((y - pred) ** 2).sum())
            tot += float((y ** 2).sum())
            index += 1
        per_block.append(1.0 - sse / tot if tot > 0 else np.nan)
    return float(np.mean(per_block))


def residualise(hist: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split history into the part the baseline already explains and the part it does not."""
    design = np.column_stack([np.ones(len(base)), np.nan_to_num(base)])
    coef, *_ = np.linalg.lstsq(design, np.nan_to_num(hist), rcond=None)
    fitted = design @ coef
    return fitted, np.nan_to_num(hist) - fitted


def conditional_permutation(raw_base, raw_hist, ystd, groups, strata, seeds,
                            n_perm=199, alpha: float | None = None) -> dict:
    """The primary test, repeated across fold assignments with synchronised permutations.

    Three things the single-assignment version got wrong. The residualisation is now
    CROSS-FITTED inside each training fold. The observed gain is averaged across the same
    three fold assignments used everywhere else. And permutation draw i uses the SAME
    shuffle for every assignment, so the null is averaged over assignments rather than
    being a different null for each.
    """
    observed, bases, null = [], [], np.zeros(n_perm)
    for seed in seeds:
        folds = M.group_folds(groups, FOLDS, seed)
        fitted, residual = M.crossfit_residualise(raw_hist, raw_base, folds)
        full = np.column_stack([raw_base, fitted + residual])
        # A PRESPECIFIED penalty. Selecting alpha by the same out-of-fold predictions
        # that are then scored is selection optimism, and on gains of 0.02-0.04 that is
        # not negligible. Stability across fixed penalties is reported instead.
        n_out = ystd.shape[1]
        alphas_base = ([alpha] * n_out if alpha is not None
                       else M.choose_alphas(raw_base, ystd, folds, groups, seed))
        alphas_full = ([alpha] * n_out if alpha is not None
                       else M.choose_alphas(full, ystd, folds, groups, seed))
        base = block_joint_score(raw_base, ystd, folds, groups, seed, alphas_base)
        bases.append(base)
        observed.append(block_joint_score(full, ystd, folds, groups, seed, alphas_full)
                        - base)
        rng = np.random.default_rng(SEED + 101)          # same shuffles every assignment
        for i in range(n_perm):
            order = M.stratified_permutation(strata, rng)
            shuffled = np.column_stack([raw_base, fitted + residual[order]])
            null[i] += (block_joint_score(shuffled, ystd, folds, groups, seed,
                                          alphas_full) - base) / len(seeds)
    mean_observed = float(np.mean(observed))
    return {"base_joint_score": float(np.mean(bases)),
            "observed": mean_observed,
            "across_assignments_sd": float(np.std(observed)),
            "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, 0.95)),
            "null_max": float(null.max()),
            "p": float((1.0 + (null >= mean_observed).sum()) / (1.0 + len(null))),
            "n_perm": int(len(null)), "n_assignments": len(seeds)}


def analyse(cohort: str, genie: Path, quiet: bool = False, fresh_only: bool = False,
            alpha: float | None = 100.0) -> dict | None:
    patients = load(genie)
    eligible = [p for p in patients
                if p.death > TAU
                and (p.last_contact >= TAU + H or p.death <= TAU + H)
                and any(w <= TAU for w, _ in p.status)]
    alive = sum(1 for p in patients if p.death > TAU)
    documented_future = [p for p in eligible
                         if observed_span(p, TAU, TAU + H) > 0 or p.death <= TAU + H]
    say = (lambda *a, **k: None) if quiet else print
    say(f"\n=== {cohort}: {len(patients)} patients, "
          f"{np.mean([len(p.status) for p in patients]):.1f} reconciled assessments and "
          f"{np.mean([len(p.lines) for p in patients]):.1f} canonical regimens each")
    say(f"    {len(eligible)} with a landmark assessment and full follow-up "
          f"({len(eligible) / max(1, alive):.0%} of those alive at the landmark)")
    eligible = documented_future
    if fresh_only:
        eligible = [p for p in eligible
                    if any(TAU - w <= STATUS_VALIDITY for w, _ in p.status if w <= TAU)]
    if len(eligible) < 200:
        say(f"    only {len(eligible)} with documented future time; skipped")
        return None
    say(f"    {len(eligible)} also have documented time inside the outcome window")

    X = pd.DataFrame([prefix_features(p, TAU) for p in eligible])
    Yfull = pd.DataFrame([future_outcomes(p, TAU, H) for p in eligible])
    coverage = float(Yfull["_months_documented_future"].mean())
    Y = Yfull[OUTCOMES]
    groups = np.array([p.pid for p in eligible])
    say(f"    design {X.shape[0]} x {X.shape[1]}; documented future time "
          f"{coverage:.1f} of {H:.0f} months on average")

    rungs = list(LADDER)
    cum, cumulative = {}, []
    for rung, feats in LADDER.items():
        cumulative = cumulative + [c for c in feats if c in X.columns]
        cum[rung] = list(cumulative)
    # The history baseline is now context + observation process + current state.
    full, base_cols = cum[rungs[-1]], cum[rungs[2]]

    nested = {r: {o: [] for o in OUTCOMES} for r in rungs}
    gains = {o: [] for o in OUTCOMES}
    lobo = {r: {o: [] for o in OUTCOMES} for r in rungs}
    all_preds = []
    for repeat in range(REPEATS):
        seed = SEED + 1000 * repeat
        folds = M.group_folds(groups, FOLDS, seed)
        preds = {}

        def predict(cols, outcome, folds=folds, seed=seed, preds=preds):
            key = (tuple(cols), outcome)
            if key not in preds:
                y = Y[outcome].to_numpy(float)
                ok = np.isfinite(y)
                preds[key] = None if (not cols or ok.sum() < 80) else (
                    y[ok], M.oof_predict(X.loc[ok, list(cols)].to_numpy(float), y[ok],
                                         folds[ok], groups[ok], seed), groups[ok])
            return preds[key]

        for rung in rungs:
            for o in OUTCOMES:
                got = predict(cum[rung], o)
                nested[rung][o].append(np.nan if got is None
                                       else M.metric(got[0], got[1], o in BINARY))
        for o in OUTCOMES:
            gb, gf = predict(base_cols, o), predict(full, o)
            if gb and gf:
                gains[o].append(M.metric(gf[0], gf[1], o in BINARY)
                                - M.metric(gb[0], gb[1], o in BINARY))
        for rung, feats in LADDER.items():
            without = [c for c in full if c not in set(feats)]
            for o in OUTCOMES:
                gb, gf = predict(without, o), predict(full, o)
                if gb and gf:
                    lobo[rung][o].append(M.metric(gf[0], gf[1], o in BINARY)
                                         - M.metric(gb[0], gb[1], o in BINARY))
        all_preds.append(preds)

    # The interval must describe the estimator that is reported. The point estimate is
    # the mean across fold assignments, so the bootstrap runs on out-of-fold predictions
    # AVERAGED across those same assignments, not on the last one.
    history_gain = {}
    for o in OUTCOMES:
        stack_b = [pool[(tuple(base_cols), o)] for pool in all_preds
                   if pool.get((tuple(base_cols), o))]
        stack_f = [pool[(tuple(full), o)] for pool in all_preds
                   if pool.get((tuple(full), o))]
        if not stack_b or not stack_f:
            history_gain[o] = {"gain": float("nan"), "ci": [float("nan")] * 2}
            continue
        y, cards_o = stack_f[0][0], stack_f[0][2]
        mean_b = np.mean([s[1] for s in stack_b], axis=0)
        mean_f = np.mean([s[1] for s in stack_f], axis=0)
        boot = M.paired_gain(y, mean_b, mean_f, o in BINARY, cards_o, SEED)
        boot["across_repeats_sd"] = float(np.std(gains[o])) if gains[o] else float("nan")
        history_gain[o] = boot

    ymat = Y.to_numpy(float)
    keep = np.isfinite(ymat).all(axis=1)
    sd = np.where(ymat[keep].std(axis=0) > 1e-9, ymat[keep].std(axis=0), 1.0)
    ystd = (ymat[keep] - ymat[keep].mean(axis=0)) / sd
    hist_cols = [c for k in rungs[3:] for c in LADDER[k] if c in X.columns]
    strata = X.loc[keep, "stage_iv"].to_numpy(float).astype(str)
    joint = conditional_permutation(
        np.nan_to_num(X.loc[keep, [c for c in base_cols if c in X.columns]]
                      .to_numpy(float)),
        np.nan_to_num(X.loc[keep, hist_cols].to_numpy(float)), ystd,
        groups[keep], strata, [SEED + 1000 * r for r in range(REPEATS)], alpha=alpha)
    say(f"    conditional joint test: base {joint['base_joint_score']:.3f}  gain "
          f"{joint['observed']:+.4f}  null mean {joint['null_mean']:+.4f}  95th "
          f"{joint['null_p95']:+.4f}  max {joint['null_max']:+.4f}  p = {joint['p']:.4f}")
    above = sum(1 for o in OUTCOMES if history_gain[o]["ci"][0] > 0)
    say(f"    {above}/{len(OUTCOMES)} outcomes with intervals above zero")
    for o in OUTCOMES:
        g = history_gain[o]
        say(f"      {o:28s} {np.mean(nested[rungs[1]][o]):6.3f} -> "
              f"{np.mean(nested[rungs[-1]][o]):6.3f}  {g['gain']:+6.3f} "
              f"[{g['ci'][0]:+.3f},{g['ci'][1]:+.3f}]  sd {g['across_repeats_sd']:.3f}")
    return {"cohort": cohort, "n_patients": len(patients), "n_eligible": len(eligible),
            "documented_future_months": coverage,
            "mean_assessments": float(np.mean([len(p.status) for p in patients])),
            "mean_regimens": float(np.mean([len(p.lines) for p in patients])),
            "n_features": int(X.shape[1]),
            "nested": {r: {o: float(np.mean(v)) for o, v in d.items()}
                       for r, d in nested.items()},
            "history_gain": history_gain,
            "lobo": {r: {o: float(np.mean(v)) if v else float("nan")
                         for o, v in d.items()} for r, d in lobo.items()},
            "joint_primary": joint, "n_outcomes_above_zero": int(above)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for cohort, folder in COHORTS.items():
        genie = GENIE_ROOT / folder
        if genie.exists():
            got = analyse(cohort, genie)
            if got:
                results[cohort] = got

    rungs = list(LADDER)
    fig, axes = plt.subplots(2, len(results), figsize=(6.2 * len(results), 10.6),
                             squeeze=False)
    for col, (cohort, r) in enumerate(results.items()):
        ax = axes[0][col]
        ys = np.arange(len(OUTCOMES))
        g = [r["history_gain"][o]["gain"] for o in OUTCOMES]
        lo = [g[i] - r["history_gain"][o]["ci"][0] for i, o in enumerate(OUTCOMES)]
        hi = [r["history_gain"][o]["ci"][1] - g[i] for i, o in enumerate(OUTCOMES)]
        ax.barh(ys, g, 0.62, color="#0072B2", xerr=[lo, hi],
                error_kw={"ecolor": "#333333", "lw": 1.1})
        ax.axvline(0, color="#B00020", lw=1.4)
        ax.set_yticks(ys)
        ax.set_yticklabels([o.replace("_", " ").replace(" future", "")
                            for o in OUTCOMES] if col == 0 else [], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("history gain beyond context, state and observability")
        j = r["joint_primary"]
        ax.set_title(f"{cohort}  n={r['n_eligible']}\njoint gain {j['observed']:+.3f}, "
                     f"null 95th {j['null_p95']:+.3f}, p={j['p']:.3f}",
                     fontsize=10.5, loc="left")

        ax2 = axes[1][col]
        mat = np.array([[r["lobo"][k][o] for o in OUTCOMES] for k in rungs])
        vmax = float(np.nanmax(np.abs(mat))) or 1.0
        im = ax2.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax2.set_yticks(range(len(rungs)))
        ax2.set_yticklabels([k.replace("+ ", "") for k in rungs] if col == 0 else [],
                            fontsize=8.5)
        ax2.set_xticks(range(len(OUTCOMES)))
        ax2.set_xticklabels([o.replace("_", " ").replace(" future", "")
                             for o in OUTCOMES], rotation=45, ha="right", fontsize=7.6)
        for i in range(mat.shape[0]):
            for j2 in range(mat.shape[1]):
                if np.isfinite(mat[i, j2]):
                    ax2.text(j2, i, f"{mat[i, j2]:.02f}".replace("0.", "."),
                             ha="center", va="center", fontsize=7,
                             color="white" if abs(mat[i, j2]) > vmax * .55 else "#222")
        fig.colorbar(im, ax=ax2, fraction=0.035)
        ax2.set_title(f"{cohort}: which block", fontsize=10.5, loc="left")
    fig.suptitle("Does clinical history add beyond context, current state AND the "
                 "observation process?\nGENIE-BPC, canonical regimens, reconciled "
                 "status, documented occupancy only, conditional permutation",
                 fontsize=12.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "genie_history_v2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Sensitivity 1: penalty stability. The primary uses a prespecified alpha of 100;
    # 10 and 1000 are reported so no reader has to trust that choice.
    alpha_stability = {}
    for alpha in (10.0, 1000.0):
        alpha_stability[f"alpha={alpha:.0f}"] = {}
        for cohort, folder in COHORTS.items():
            genie = GENIE_ROOT / folder
            if not genie.exists():
                continue
            got = analyse(cohort, genie, quiet=True, alpha=alpha)
            if got:
                alpha_stability[f"alpha={alpha:.0f}"][cohort] = {
                    "joint_gain": got["joint_primary"]["observed"],
                    "joint_p": got["joint_primary"]["p"]}
    print("\npenalty stability (primary is alpha=100):")
    for label, cohorts in alpha_stability.items():
        row = "  ".join(f"{c} {r['joint_gain']:+.4f} (p={r['joint_p']:.3f})"
                        for c, r in cohorts.items())
        print(f"  {label:>12s}  {row}")

    # Sensitivity 2: restrict to patients whose landmark status is fresh, so no stale
    # label enters the comparator at all.
    fresh = {}
    for cohort, folder in COHORTS.items():
        genie = GENIE_ROOT / folder
        if not genie.exists():
            continue
        got = analyse(cohort, genie, quiet=True, fresh_only=True)
        if got:
            fresh[cohort] = {"joint_gain": got["joint_primary"]["observed"],
                             "joint_p": got["joint_primary"]["p"],
                             "n_eligible": got["n_eligible"],
                             "new_metastatic_sites":
                                 got["history_gain"]["n_new_metastatic_sites_future"]["gain"]}
    print("\nfresh-landmark-only:")
    for cohort, r in fresh.items():
        print(f"  {cohort:6s} n={r['n_eligible']:4d}  joint {r['joint_gain']:+.4f} "
              f"(p={r['joint_p']:.3f})  sites {r['new_metastatic_sites']:+.3f}")

    # Sensitivity 3: the four-month status validity is a choice, so the primary test is
    # repeated at two and six months. The anatomy outcome is the one most exposed to it.
    sensitivity = {}
    global STATUS_VALIDITY
    original = STATUS_VALIDITY
    for window in (2.0, 6.0):
        STATUS_VALIDITY = window
        sensitivity[f"{window:.0f} months"] = {}
        for cohort, folder in COHORTS.items():
            genie = GENIE_ROOT / folder
            if not genie.exists():
                continue
            got = analyse(cohort, genie, quiet=True)
            if got:
                sensitivity[f"{window:.0f} months"][cohort] = {
                    "joint_gain": got["joint_primary"]["observed"],
                    "joint_p": got["joint_primary"]["p"],
                    "n_eligible": got["n_eligible"],
                    "new_metastatic_sites":
                        got["history_gain"]["n_new_metastatic_sites_future"]["gain"]}
    STATUS_VALIDITY = original
    print("\nstatus-validity sensitivity:")
    for window, cohorts in sensitivity.items():
        for cohort, r in cohorts.items():
            print(f"  {window:>10s}  {cohort:6s} n={r['n_eligible']:4d}  joint "
                  f"{r['joint_gain']:+.4f} (p={r['joint_p']:.3f})  sites "
                  f"{r['new_metastatic_sites']:+.3f}")

    (OUT / "readout.json").write_text(json.dumps({
        "status_validity_sensitivity": sensitivity,
        "penalty_stability": alpha_stability,
        "fresh_landmark_only": fresh,
        "primary_alpha": 100.0,
        "source": "GENIE-BPC v2.0-public; aggregate results only, DUA-compliant",
        "landmark_months": TAU, "horizon_months": H,
        "status_validity_months": STATUS_VALIDITY, "repeats": REPEATS,
        "outcome_blocks": OUTCOME_BLOCKS,
        "channels_absent": ["performance status", "weight", "organ function",
                            "symptom burden", "CTCAE toxicity"],
        "cohorts": results}, indent=2) + "\n")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
