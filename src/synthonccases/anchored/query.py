"""The QUERY side of the prompt: the patient a trajectory is being asked for.

Retrieval and rendering answer "who is like this patient". Nothing in them says who the
patient IS -- and a prompt carrying forty comparable patients and no index case asks the
model to predict a stranger. This module builds the two objects the frozen runner already
knows how to render:

* a **baseline seed**, a partial ``BaselineObject`` holding what the source recorded about
  the patient at diagnosis, with the leaves it fills pinned so the model cannot overwrite
  them;
* an **observed prefix**, the treatment history up to and including the landmark, as DATED
  ledger events -- which is what makes this a continuation from a real decision point rather
  than a fresh case.

Only facts the SOURCE dated become prefix events. GENIE-BPC dates regimen starts and imaging
reads; it also carries counts -- how many prior progressions, how many prior scans -- that it
never dated, and those have no entry here. Absolute dates come from the real year of
diagnosis with the day fixed by convention (:data:`ANCHOR_MONTH_DAY`), so every INTERVAL
between events is exact and the calendar position is right to within six months.

Two rules the mapping obeys, because both have cost this project a measurement before:

* **Absent is not zero, and not fillable either.** GENIE-BPC records no ECOG, no laboratory
  values and no weight. Those leaves are left out of the seed rather than seeded "normal",
  which would assert a fact nobody observed. Under the primary `continuation` contract they
  then stay missing: they are UNCERTAINTY the model must carry into a wider distribution of
  futures, not blanks for it to fill. Only the `latent_completion` ablation lets the model
  write them, and that is the question that ablation exists to answer.
* **Nothing after the landmark.** The seed is built from columns evaluated at the treatment
  start. A sequencing panel resulted afterwards is excluded on its REPORT date, and the
  outcome columns are never read at all; that is the thing being predicted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from typing import Any

import numpy as np

#: Cohort fallback when histology is not recorded: the PARENT code, never a child.
#: Mapping every NSCLC row to LUAD asserted adenocarcinoma for the 7% that are squamous --
#: a different disease with a different treatment history and a different prognosis, handed
#: to the model as fact. Where the release records histology we use it; where it does not,
#: the code stays at the level the source actually supports.
ONCOTREE_PARENT = {"NSCLC": "NSCLC", "BRCA": "BRCA", "CRC": "COADREAD"}


def oncotree_code(cancer: str, histology: str | None = None,
                  site: str | None = None) -> str:
    """The most specific code the SOURCE supports, and no more specific than that."""
    c = str(cancer).upper()
    h = str(histology or "").lower()
    s = str(site or "").upper()
    if c == "NSCLC":
        if "small cell" in h:
            return "SCLC"                    # miscoded into the NSCLC cohort; not NSCLC
        if "squamous" in h and "adeno" not in h:
            return "LUSC"
        if "adeno" in h:
            return "LUAD"
        return "NSCLC"                       # carcinoma NOS, mixed, other: not specifiable
    if c == "BRCA":
        if "ductal" in h:
            return "IDC"
        if "lobular" in h:
            return "ILC"
        return "BRCA"
    if c == "CRC":
        # ICD-O topography: C18 colon, C19 rectosigmoid, C20 rectum. Rectal cancer is
        # managed differently enough that calling it COAD would be a clinical error.
        if s.startswith("C18"):
            return "COAD"
        if s.startswith(("C19", "C20")):
            return "READ"
        return "COADREAD"
    return ONCOTREE_PARENT.get(c, c)


def _num(row: Mapping[str, Any], key: str) -> float | None:
    v = row.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def baseline_seed(row: Mapping[str, Any]) -> tuple[dict, tuple[str, ...]]:
    """The patient's recorded baseline, and the dotted paths it pins.

    Only leaves the source actually carries are written. Everything else -- performance
    status, organ function, molecular detail beyond the driver, weight -- stays absent, so
    the model fills it and we can see what it assumes rather than handing it our assumption.
    """
    cancer = str(row.get("cancer", "")).upper()
    seed: dict[str, Any] = {"identity_card": {"diagnosis": {}, "staging": {},
                                              "demographics": {}}}
    pinned: list[str] = []

    code = oncotree_code(cancer, row.get("histology"), row.get("primary_site_code"))
    if code:
        seed["identity_card"]["diagnosis"]["oncotree_code"] = code
        pinned.append("identity_card.diagnosis.oncotree_code")

    # age at DIAGNOSIS, not at this treatment: the schema leaf means the former, and the
    # landmark can be years later. Where months-since-diagnosis is missing the leaf is
    # dropped rather than approximated.
    age_now, since_dx = _num(row, "age_at_start"), _num(row, "months_since_dx")
    if age_now is not None and since_dx is not None:
        seed["identity_card"]["demographics"]["age_at_diagnosis"] = int(
            round(age_now - since_dx / 12.0))
        pinned.append("identity_card.demographics.age_at_diagnosis")
    female = _num(row, "female")
    if female is not None:
        seed["identity_card"]["demographics"]["sex"] = "female" if female else "male"
        pinned.append("identity_card.demographics.sex")

    # stage at diagnosis. `stage_iv` is the recorded stage group; distant metastases seen
    # later are history, not staging, and belong in the prefix.
    stage_iv = _num(row, "stage_iv")
    if stage_iv is not None:
        seed["identity_card"]["staging"]["stage_class"] = (
            "metastatic" if stage_iv else "locoregional")
        pinned.append("identity_card.staging.stage_class")

    # SEQUENCING, AND ONLY WHAT WAS RESULTED BY THE LANDMARK. `panel_reported` is set by
    # the caller from the report date, not the sample date: a panel that came back after
    # this treatment started was not available when the decision was taken, and handing it
    # over would leak the future of the very decision being predicted.
    #
    # Recorded as `biomarkers` (name/value facts) rather than as `alterations`, because an
    # Alteration requires `driver_bucket`, `actionability` and `is_primary_driver` -- three
    # clinical judgements we would be making on the model's behalf and pinning as fact.
    # What the source knows is which gene carried which variant; the interpretation is the
    # model's to make and ours to observe.
    if _num(row, "panel_reported"):
        calls = [c.strip() for c in str(row.get("alterations") or "").split(";")
                 if c.strip()]
        # ONE ROW PER ANALYTE. The first row used to be a SENTENCE wearing a biomarker's
        # costume: name "tumour sequencing panel" is not an analyte, and value "reported
        # before this treatment started; BRAF G466V" fused assay provenance with the
        # result -- then repeated that result verbatim in the structured row beneath it.
        # The schema documents `name` as "e.g., PD-L1 TPS, TMB, MSI"; a free sentence
        # there cannot be parsed, cannot be compared across patients, and models a
        # convention we would reject in generated output.
        #
        # Assay timing belongs in the summary prose, which states it once for the whole
        # panel, not inside a value string.
        mp: dict[str, Any] = {"biomarkers": []}
        for c in calls:
            gene, _, variant = c.partition(" ")
            mp["biomarkers"].append({"name": gene, "value": variant or "altered"})
        # A PANEL THAT RAN AND FOUND NOTHING is a result, and a different fact from no
        # panel at all. It has no home HERE: "targeted sequencing panel" is an assay, not
        # an analyte, so putting it in `name` repeats the same category error the prose row
        # made -- a smaller sentence in the same wrong slot. The schema has no
        # assay-provenance field, and inventing a biomarker row to carry one would be
        # fabricating a measurement that was never taken.
        #
        # So the seed says nothing, and the distinction is carried in the summary prose,
        # which already separates "no relevant alteration recorded on the available panel"
        # from "molecular status unknown". Narrative-only is the honest representation of
        # a fact the structured contract cannot hold.
        if not calls:
            mp.pop("biomarkers")
        driver = row.get("driver_bucket")
        if driver and str(driver) not in ("", "nan", "none found"):
            mp["primary_driver_bucket"] = str(driver)
        # PIN ONLY WHAT IS ACTUALLY SUPPLIED. An empty molecular_profile is dropped by the
        # filter below, so pinning `biomarkers` unconditionally would name a path the seed
        # does not contain -- a pin the validator cannot check and the model cannot honour.
        if mp:
            seed["identity_card"]["molecular_profile"] = mp
            if mp.get("biomarkers"):
                pinned.append("identity_card.molecular_profile.biomarkers")
            if "primary_driver_bucket" in mp:
                pinned.append("identity_card.molecular_profile.primary_driver_bucket")

    seed["identity_card"] = {k: v for k, v in seed["identity_card"].items() if v}
    return seed, tuple(pinned)


# ---------------------------------------------------------------------------- the prefix ---
#: Day of the year the diagnosis is anchored to. GENIE-BPC shifts exact dates and publishes
#: intervals, so the YEAR of diagnosis is real and the day within it is not knowable. Anchoring
#: mid-year makes every interval that follows exact and the absolute date honest to within six
#: months -- which is the resolution the era axis has anyway.
ANCHOR_MONTH_DAY = (7, 1)


def diagnosis_date(dx_year: float | int | None) -> date | None:
    """The artificial calendar anchor. Real year, conventional day; None when unknown."""
    if dx_year is None or (isinstance(dx_year, float) and np.isnan(dx_year)):
        return None
    return date(int(dx_year), *ANCHOR_MONTH_DAY)


def _at(origin: date, months: float) -> date:
    """Elapsed months from diagnosis to a calendar date. The INTERVAL is the real quantity."""
    return origin + timedelta(days=round(float(months) * 30.4375))


#: Same-type observations this close together are one clinical event recognised twice --
#: a CT and an MRI in the same week, or one study entered on several imaging rows. The value
#: and the rule are the hybrid pipeline's, kept identical so the two constructions do not
#: quietly disagree about what counts as a progression.
MERGE_WINDOW_DAYS = 30

#: Event types that ARE the landmark rather than observations leading up to it.
TREATMENT_TYPES = ("systemic_treatment_start", "systemic_treatment_switch",
                   "systemic_treatment_stop", "systemic_treatment_modification")


def _key(ev: Mapping[str, Any]) -> tuple:
    """Identity of an event for de-duplication: when, what, and to which regimen."""
    reg = ev.get("regimen") or {}
    return (ev["date"], ev["type"], str(reg.get("label", "")), str(ev.get("rationale", "")))


def _merges_into_previous(kept: Sequence[Mapping[str, Any]], ev: Mapping[str, Any]) -> bool:
    """True when this observation is the same clinical event as one already recorded.

    Same type, within the merge window, and no treatment event in between -- a treatment
    change between two progression reads means the second one is about a different line and
    is its own event.
    """
    here = date.fromisoformat(ev["date"])
    for prev in reversed(kept):
        if prev["type"] in TREATMENT_TYPES:
            return False
        if prev["type"] != ev["type"]:
            continue
        return (here - date.fromisoformat(prev["date"])).days <= MERGE_WINDOW_DAYS
    return False


def prefix_events(facts: Sequence[Mapping[str, Any]], *, origin: date,
                  landmark_months: float,
                  landmark_action: str | None = None) -> dict[str, Any]:
    """Dated history up to the landmark, as the frozen runner's fixed prefix.

    ``facts`` are ``{"months": float, "type": str, ...}`` records the SOURCE dated. Anything
    the source counted but did not date -- a number of prior progressions, a stage recorded
    without a date -- has no entry here. A prefix is a claim that these things happened on
    these days; padding it with plausible timings would make the model's continuation look
    conditioned on a history that was partly ours.

    Three rules about the boundary, each of which changes what the model reads:

    * **One row per event.** The imaging release carries several rows for one scan, so a
      single progressive study arrived as three identical progression events. Repetition in
      a prompt is emphasis, and it would have made progression look like the patient's
      dominant tempo. Exact duplicates are collapsed, and same-type observations inside
      :data:`MERGE_WINDOW_DAYS` with no treatment change between them are treated as one
      event recognised twice -- the hybrid pipeline's rule, reused rather than reinvented.
    * **Observations must be STRICTLY before the landmark.** A scan dated on the start date
      is usually the restaging that motivated the switch, and its read is downstream of the
      decision. The anchors' own features are computed with the same strict rule, so
      admitting it for the query alone would give the index patient information no anchor
      carries.
    * **The landmark itself is a treatment event at the cutoff.** When ``landmark_action`` is
      given, the treatment event at the cutoff whose regimen matches it is marked as the
      index treatment; any other treatment recorded that same day is labelled concurrent
      rather than dropped, because concurrent regimens are real and silently deleting one
      would misstate what the patient received.
    """
    seen: set[tuple] = set()
    events: list[dict[str, Any]] = []
    dropped = duplicates = at_landmark = merged = 0
    for f in sorted(facts, key=lambda r: float(r["months"])):
        m = float(f["months"])
        if m > landmark_months + 1e-9:
            continue
        if m < 0:
            # dated before diagnosis: the workup scan that led to the diagnosis, not
            # treated history. The frozen validator rejects it outright, and shifting it
            # forward to make it legal would move a real date to suit our calendar.
            dropped += 1
            continue
        is_treatment = str(f["type"]) in TREATMENT_TYPES
        if not is_treatment and m >= landmark_months - 1e-9:
            at_landmark += 1
            continue
        ev: dict[str, Any] = {"date": _at(origin, m).isoformat(), "type": str(f["type"])}
        for k in ("regimen", "rationale", "reason", "detection_method"):
            if f.get(k) is not None:
                ev[k] = f[k]
        k = _key(ev)
        if k in seen:
            duplicates += 1
            continue
        if not is_treatment and _merges_into_previous(events, ev):
            merged += 1
            continue
        seen.add(k)
        events.append(ev)

    cutoff = _at(origin, landmark_months).isoformat()
    # CONCURRENT REGIMENS, LABELLED WHEREVER THEY OCCUR. The release records drugs given
    # together as separate regimens with the same start date. Left unlabelled they read as
    # two switches in one day -- a patient changing treatment twice -- when they are one
    # combination. The cutoff day additionally names which of them the trajectory starts
    # from, because that is the one the endpoint is measured on.
    matched = 0
    n_at_cutoff = 0
    by_date: dict[str, list[dict]] = {}
    for e in events:
        if e["type"] in TREATMENT_TYPES:
            by_date.setdefault(e["date"], []).append(e)
    for day, same_day in by_date.items():
        at_cutoff = day == cutoff
        if at_cutoff:
            n_at_cutoff = len(same_day)
        index: dict[str, Any] | None = None
        if at_cutoff and landmark_action is not None:
            index = next((e for e in same_day
                          if str((e.get("regimen") or {}).get("label", ""))
                          == landmark_action), None)
            if index is not None:
                index["rationale"] = ("the treatment this trajectory starts from; continue "
                                      "from here")
                matched = 1
        if len(same_day) == 1:
            continue
        for e in same_day:
            if e is index or (index is None and e is same_day[0]):
                continue
            e["rationale"] = ("recorded as starting the same day as "
                              + ("the index treatment" if at_cutoff else "the regimen above")
                              + "; concurrent, not sequential")
    return {"origin": "observed_prefix", "cutoff_date": cutoff, "events": events,
            "n_dropped_pre_diagnosis": dropped,
            "n_duplicates_collapsed": duplicates,
            "n_merged_within_window": merged,
            "n_observations_at_landmark_dropped": at_landmark,
            "n_treatment_events_at_cutoff": n_at_cutoff,
            "index_treatment_matched": bool(matched)}


def conditioning_payload(seed: Mapping[str, Any], pinned: Sequence[str],
                         fixed_events: Mapping[str, Any]) -> dict[str, Any]:
    """The payload `lower_conditioning_cell` lowers into a ConditioningSpec.

    Assembled here so every arm of an experiment builds it the same way: the query patient
    and their history are IDENTICAL across arms by construction, and the anchor block is the
    only thing that varies.
    """
    return {"baseline_seed": dict(seed), "pinned_paths": list(pinned),
            "fixed_events": dict(fixed_events)}
