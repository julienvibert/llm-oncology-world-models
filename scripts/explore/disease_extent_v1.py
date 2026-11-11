#!/usr/bin/env python3
"""Two stage concepts, kept apart in data and in language.

    stage_at_diagnosis                        from stage_dx_iv. Belongs to C, fixed.
    metastatic_disease_documented_by_landmark yes / no / unknown. Belongs to X, timed.
    documented_metastatic_sites_by_landmark   which sites, and the month each appeared.

They are different questions and must never collapse into one field. A patient diagnosed
Stage II who is metastatic by line 3 is metastatic, and a seed carrying only diagnosis
stage would describe them as locoregional at the moment being predicted.

WHERE THE SITES COME FROM -- and where they must NOT come from.

`SCAN_SITES` in the imaging timeline is the anatomic region IMAGED: "Chest", "Full body",
"Chest, Abdomen, Pelvis". It is a description of the scan, not of the disease. Read as a
metastatic-site list it says roughly everyone has metastatic disease, because roughly
everyone gets a chest CT. That is the same failure as reading `CA_TX_PRE_PATH_STAGE` as a
stage group: a populated, varying, entirely wrong field.

The correct source is the curated metastasis block in `cancer_level_dataset_index.csv`:
`dmets_<site>` with `dx_to_dmets_<site>_mos`, giving a per-site flag AND the month it first
appeared, already on the same clock as tau. It reproduces the clinical picture -- liver in
63% of colorectal, brain in 33% of lung -- which the scan-region field does not.

BRCA carries no per-site breakdown, only `dmets_post_dx` and `dx_to_dmets_mos`. So breast
gets timed any-metastasis and no site list, and the renderer says so rather than implying
the sites were checked and found absent.

THE DERIVATION.

    yes      Stage IV at diagnosis, or a distant site documented at or before the landmark.
             Stage IV at diagnosis settles it alone -- the question is whether metastatic
             disease has been documented BY the landmark, and a Stage IV diagnosis is that
             documentation. Waiting for a later scan would turn a known fact into unknown.
    no       only where the record explicitly establishes absence, which the curated block
             does: a patient curated as never developing distant metastases had none at any
             earlier landmark. Reachable for non-IV patients ONLY. There is no Stage IV
             exception: stage IV can be assigned on pleural or contralateral intrathoracic
             criteria that the curated site block encodes differently, so an empty
             named-site list there is a gap in the block, never localised disease.
    unknown  everything else: no curated metastasis record, an explicit "Unknown or Not
             mentioned", or a flagged metastasis whose date is missing and so cannot be
             placed before or after the landmark.

NOT "current active metastases". A patient with metastatic disease documented years earlier
may have no active measurable lesion at this landmark. The variable records what has been
documented, and the wording says exactly that.

The rendered sentence lives here, beside the derivation, because a code comment does not
reach the model. If the prefix says only `stage_class: locoregional`, that is what the
model reasons from.

GOVERNANCE. GENIE-BPC under DUA. Aggregate output only, gitignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SITES = ("abdomen", "bone", "brain", "breast", "extremity", "head_neck", "liver",
         "pelvis", "thorax")
SITE_NAMES = {"head_neck": "head and neck"}


@dataclass(frozen=True)
class MetRecord:
    """One patient's curated metastasis history, landmark-independent."""

    stage: str                              # "IV" | "I-III" | "unknown"
    ever: float | None                      # curated: did distant disease ever occur
    first_month: float | None               # months from diagnosis to first distant met
    sites: tuple[tuple[float, str], ...]    # (month, site) for each documented site
    site_detail_available: bool             # False for cohorts without per-site fields
    undated_sites: int                      # flagged but unplaceable in time


@dataclass(frozen=True)
class Extent:
    """One patient's disease extent at one landmark, with the basis for the call."""

    stage_at_diagnosis: str
    documented: str                         # "yes" | "no" | "unknown"
    sites: tuple[tuple[float, str], ...]
    site_detail_available: bool
    basis: str


def _num(value) -> float | None:
    out = pd.to_numeric(value, errors="coerce")
    return float(out) if np.isfinite(out) else None


def index_cancer(genie: Path) -> dict[str, int]:
    """One index cancer per patient: the earliest by sequence number.

    21 NSCLC patients carry two index cancers. Without a single rule applied everywhere,
    the stage could come from one cancer and the treatments from the other -- and the
    regimen clock is per-cancer, so mixing them silently misdates every line.
    """
    frame = pd.read_csv(genie / "cancer_level_dataset_index.csv", low_memory=False)
    out: dict[str, int] = {}
    for _, r in frame.iterrows():
        pid, seq = str(r["record_id"]), int(r["ca_seq"])
        if pid not in out or seq < out[pid]:
            out[pid] = seq
    return out


def load_metastasis(genie: Path) -> dict[str, MetRecord]:
    """The curated metastasis block, per patient. Never the imaging scan-region field."""
    frame = pd.read_csv(genie / "cancer_level_dataset_index.csv", low_memory=False)
    # Restricted to the single index cancer, so stage and metastases describe the same
    # cancer whose treatments and clock the rest of the pipeline uses.
    keep = index_cancer(genie)
    frame = frame[[str(r) == str(r) and keep.get(str(r)) == int(q)
                   for r, q in zip(frame["record_id"], frame["ca_seq"], strict=True)]]
    has_sites = all(f"dmets_{s}" in frame.columns for s in SITES)
    ever_col = ("dmets_stage_i_iii" if "dmets_stage_i_iii" in frame.columns
                else "dmets_post_dx" if "dmets_post_dx" in frame.columns else None)

    out: dict[str, MetRecord] = {}
    for _, r in frame.iterrows():
        pid = str(r["record_id"])
        if pid in out:
            continue
        value = str(r.get("stage_dx_iv") or "").strip()
        stage = ("IV" if value == "Stage IV"
                 else "I-III" if value.startswith("Stage I") else "unknown")

        sites: list[tuple[float, str]] = []
        undated = 0
        if has_sites:
            for s in SITES:
                if _num(r.get(f"dmets_{s}")) != 1.0:
                    continue
                month = _num(r.get(f"dx_to_dmets_{s}_mos"))
                if month is None:
                    undated += 1
                else:
                    sites.append((month, s))

        out[pid] = MetRecord(
            stage=stage,
            ever=_num(r.get(ever_col)) if ever_col else None,
            first_month=_num(r.get("dx_to_dmets_mos")),
            sites=tuple(sorted(sites)),
            site_detail_available=has_sites,
            undated_sites=undated)
    return out


def stage_at_diagnosis(genie: Path) -> dict[str, str]:
    """From `stage_dx_iv`, the only correct source.

    `CA_TX_PRE_PATH_STAGE` is a yes/no flag about treatment given before the pathological
    specimen. It is not a stage group and must never be read as one.
    """
    return {pid: m.stage for pid, m in load_metastasis(genie).items()}


def extent_at_landmark(m: MetRecord | None, tau: float) -> Extent:
    """What had been documented by the landmark, and the basis for saying so."""
    if m is None:
        return Extent("unknown", "unknown", (), False, "no curated cancer record")

    by_now = tuple(s for s in m.sites if s[0] <= tau + 1e-9)

    # Stage IV ALWAYS documents metastatic disease, with no exception for `ca_dmets_yn`
    # recording no distant metastases. Stage IV lung cancer can be assigned on pleural or
    # contralateral intrathoracic criteria that the curated site block encodes differently,
    # so an empty named-site list is a gap in that block, not evidence of localised
    # disease. Flipping such a patient to "no" would render "Stage IV without distant
    # disease" -- a claim the data do not support.
    if m.stage == "IV":
        return Extent(m.stage, "yes", by_now, m.site_detail_available,
                      "stage IV at diagnosis")
    if by_now:
        return Extent(m.stage, "yes", by_now, m.site_detail_available,
                      "distant site documented at or before the landmark")
    if m.first_month is not None and m.first_month <= tau + 1e-9:
        return Extent(m.stage, "yes", (), m.site_detail_available,
                      "distant metastasis dated at or before the landmark, site not "
                      "separately recorded")
    if m.ever == 0.0 and not m.undated_sites:
        return Extent(m.stage, "no", (), m.site_detail_available,
                      "curated as never developing distant metastases")
    if m.ever == 1.0 or m.first_month is not None or m.sites:
        return Extent(m.stage, "no", (), m.site_detail_available,
                      "distant disease documented, but only after the landmark")
    if m.undated_sites:
        return Extent(m.stage, "unknown", (), m.site_detail_available,
                      "distant site flagged but undated, so it cannot be placed relative "
                      "to the landmark")
    return Extent(m.stage, "unknown", (), m.site_detail_available,
                  "no curated metastasis record")


def _sites_phrase(sites: tuple[tuple[float, str], ...]) -> str:
    names = list(dict.fromkeys(SITE_NAMES.get(s, s) for _m, s in sites))
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def render_extent(e: Extent) -> str:
    """The sentence the model actually reads. Both concepts, always, never one alone."""
    head = {"IV": "Stage IV at diagnosis", "I-III": "Stage I-III at diagnosis"}.get(
        e.stage_at_diagnosis, "Stage at diagnosis not recorded")

    # Every branch names the landmark explicitly. "Stage IV at diagnosis" on its own
    # leaves the reader to infer current extent from diagnosis stage, which is the exact
    # collapse these two variables exist to prevent.
    if e.documented == "unknown":
        return (f"{head}; metastatic extent at the current landmark cannot be established "
                f"from available records.")
    if e.documented == "no":
        # Reachable only for non-IV patients, on an explicit curated determination.
        return f"{head}; no distant metastasis documented by the landmark."

    sites = _sites_phrase(e.sites) if e.sites else None
    if e.stage_at_diagnosis == "IV":
        if sites:
            return f"{head}; distant sites documented by the landmark: {sites}."
        if e.site_detail_available:
            # NOT "no distant disease". The curated fields record no NAMED distant-organ
            # site; stage IV assigned on pleural or intrathoracic criteria looks exactly
            # like this, and the sentence says only what the fields actually show.
            return (f"{head}; no named distant-organ site recorded in the curated "
                    f"metastasis fields by the landmark.")
        return (f"{head}; site-level detail is not collected in this cohort, so no site "
                f"list is available at the landmark.")

    if sites:
        return (f"{head}; metastatic disease subsequently documented before the landmark; "
                f"known sites: {sites}.")
    detail = ("specific sites not recorded" if e.site_detail_available
              else "site-level detail is not collected in this cohort")
    return (f"{head}; metastatic disease subsequently documented before the landmark; "
            f"{detail}.")


def render_stage_only(e: Extent) -> str:
    """The diagnosis-stage field, labelled so it can never be read as landmark extent."""
    return {"IV": "stage at diagnosis: IV",
            "I-III": "stage at diagnosis: I-III"}.get(
                e.stage_at_diagnosis, "stage at diagnosis: not recorded")
