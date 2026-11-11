#!/usr/bin/env python3
"""Do the enriched operators work on REAL trajectories? GENIE-BPC NSCLC.

The synthetic POC showed that a clinically engineered cumulative history predicts the
generator's own future course. That is internal coherence. This asks the only question
that makes the registry a clinical contribution rather than a curiosity: does the SAME
history representation carry information in observed patients?

WHAT TRANSPORTS AND WHAT DOES NOT. An audit of GENIE-BPC NSCLC finds no ECOG, no
performance status, no weight, no organ-function grading and no CTCAE toxicity anywhere
in the release. Two of the six clinical channels therefore have no real counterpart, and
the ladder loses its host-and-toxicity rung entirely. What does transport:

    disease state   curated cancer status at ~19 imaging and ~15 med-onc assessments per
                    patient (improving / stable / progressing / mixed)
    anatomy         scan sites at every imaging assessment
    treatment       regimens with start and stop dates, drug names, regimen number
    terminal        death and last contact
    context         age, sex, stage, histology

That loss is not arbitrary: in the synthetic POC the host-and-toxicity block predicted
host and toxicity outcomes and essentially nothing else, and those are exactly the
outcomes GENIE also cannot score. The observable subset is internally coherent.

GOVERNANCE. GENIE-BPC is under a data use agreement. This script writes AGGREGATE
RESULTS ONLY, to a gitignored directory. No patient-level row, identifier or trajectory
is serialised anywhere.

Run:  uv run python scripts/explore/genie_operator_registry_v1.py
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
import trajectory_feature_poc_v1 as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GENIE_ROOT = Path.home() / "GenieBPC"
COHORTS = {"NSCLC": "NSCLC_Genie", "BRCA": "BRCA_GENIE", "CRC": "CRC_GENIE"}
OUT = ROOT / "outputs" / "exploratory_genie_operators"
DAYS = 30.4
TAU = 12.0
H = 12.0
TRAIL = 3.0
SEED = 20260824
FOLDS = 5

STATUS = {
    "Improving/Responding": "responding",
    "Stable/No change": "stable",
    "Progressing/Worsening/Enlarging": "progressing",
    "Mixed": "mixed",
}
STATES = ("responding", "stable", "progressing", "mixed")

IO_DRUGS = ("pembrolizumab", "nivolumab", "atezolizumab", "durvalumab", "ipilimumab",
            "avelumab", "cemiplimab", "tremelimumab")
TARGETED = ("osimertinib", "erlotinib", "gefitinib", "afatinib", "dacomitinib",
            "crizotinib", "alectinib", "ceritinib", "brigatinib", "lorlatinib",
            "dabrafenib", "trametinib", "capmatinib", "tepotinib", "selpercatinib",
            "pralsetinib", "entrectinib", "sotorasib", "adagrasib", "amivantamab",
            "mobocertinib", "bevacizumab", "ramucirumab", "necitumumab",
            "trastuzumab", "pertuzumab", "ado-trastuzumab", "lapatinib", "neratinib",
            "tucatinib", "palbociclib", "ribociclib", "abemaciclib", "olaparib",
            "talazoparib", "cetuximab", "panitumumab", "regorafenib", "ziv-aflibercept",
            "encorafenib", "binimetinib")
ENDOCRINE = ("letrozole", "anastrozole", "exemestane", "tamoxifen", "fulvestrant",
             "goserelin", "leuprolide", "megestrol")
CHEMO = ("carboplatin", "cisplatin", "oxaliplatin", "pemetrexed", "paclitaxel",
         "docetaxel", "gemcitabine", "etoposide", "vinorelbine", "nab-paclitaxel",
         "topotecan", "irinotecan", "cyclophosphamide", "doxorubicin", "epirubicin",
         "methotrexate", "fluorouracil", "capecitabine", "eribulin", "trifluridine",
         "folfox", "folfiri", "folfoxiri")


def bucket(regimen: str) -> str:
    text = str(regimen).lower()
    if any(d in text for d in IO_DRUGS):
        return "immunotherapy"
    if any(d in text for d in TARGETED):
        return "targeted"
    if any(d in text for d in ENDOCRINE):
        return "endocrine"
    if any(d in text for d in CHEMO):
        return "chemotherapy"
    return "other"


@dataclass
class Patient:
    pid: str
    context: dict
    status: list[tuple] = field(default_factory=list)   # (months, state)
    sites: list[tuple] = field(default_factory=list)    # (months, set)
    lines: list[dict] = field(default_factory=list)     # start, stop, bucket, number
    death: float = np.inf
    last_contact: float = np.inf


def parse_sites(text) -> set[str]:
    if not isinstance(text, str) or not text.strip():
        return set()
    return {part.strip().lower() for part in text.replace("|", ",").split(",")
            if part.strip()}


def carry(series: list[tuple], t: float, default=None):
    value = default
    for when, what in series:
        if when > t:
            break
        value = what
    return value


def state_intervals(p: Patient, end: float) -> dict[str, list[tuple]]:
    """The curated status series is a step function: it holds until the next assessment
    that states something. Assessments recorded as indeterminate change nothing."""
    out: dict[str, list[tuple]] = {}
    for i, (when, what) in enumerate(p.status):
        stop = p.status[i + 1][0] if i + 1 < len(p.status) else end
        if min(stop, end) > when:
            out.setdefault(what, []).append((when, min(stop, end)))
    return out


def load(genie: Path) -> list[Patient]:
    dx = pd.read_csv(genie / "data_timeline_cancer_diagnosis.txt", sep="\t",
                     low_memory=False)
    dx = dx[dx["INDEX_CANCER"].astype(str).str.lower().isin({"yes", "true", "1"})] \
        if "INDEX_CANCER" in dx else dx
    context = {}
    for _, r in dx.iterrows():
        context.setdefault(str(r["PATIENT_ID"]), {
            "age": pd.to_numeric(r.get("AGE_DX"), errors="coerce"),
            "histology": str(r.get("CA_HIST_ADENO_SQUAMOUS")),
            "stage": str(r.get("CA_TX_PRE_PATH_STAGE")),
        })
    clinical = pd.read_csv(genie / "data_clinical_patient.txt", sep="\t", comment="#",
                           low_memory=False)
    sex = dict(zip(clinical["PATIENT_ID"].astype(str),
                   clinical.get("SEX", pd.Series(["unknown"] * len(clinical))).astype(str),
                   strict=False))
    surv = pd.read_csv(genie / "data_clinical_supp_survival.txt", sep="\t",
                       low_memory=False)
    surv.columns = [c.strip() for c in surv.columns]
    pid_col = surv.columns[0]
    os_status = surv.columns[surv.columns.get_loc("OS from Diagnosis Status")]
    os_months = surv.columns[surv.columns.get_loc("OS from Diagnosis Status") + 1]

    patients: dict[str, Patient] = {}
    for _, r in surv.iterrows():
        pid = str(r[pid_col])
        months = pd.to_numeric(r[os_months], errors="coerce")
        if not np.isfinite(months):
            continue
        ctx = dict(context.get(pid, {}))
        ctx["sex"] = sex.get(pid, "unknown")
        p = Patient(pid=pid, context=ctx)
        dead = "1" in str(r[os_status]) or "dead" in str(r[os_status]).lower()
        p.death = float(months) if dead else np.inf
        p.last_contact = float(months)
        patients[pid] = p

    frames = []
    for name in ("data_timeline_imaging.txt", "data_timeline_medonc.txt"):
        f = pd.read_csv(genie / name, sep="\t", low_memory=False)
        f["_months"] = pd.to_numeric(f["START_DATE"], errors="coerce") / DAYS
        frames.append(f)
    for f in frames:
        rows = f[f["_months"].notna() & (f["_months"] >= 0)]
        for _, r in rows.sort_values("_months").iterrows():
            p = patients.get(str(r["PATIENT_ID"]))
            if p is None:
                continue
            state = STATUS.get(str(r.get("CURATED_CANCER_STATUS")))
            if state:
                p.status.append((float(r["_months"]), state))
            if "SCAN_SITES" in r and isinstance(r.get("SCAN_SITES"), str):
                p.sites.append((float(r["_months"]), parse_sites(r["SCAN_SITES"])))
    for p in patients.values():
        p.status.sort(key=lambda x: x[0])
        p.sites.sort(key=lambda x: x[0])

    tx = pd.read_csv(genie / "data_timeline_treatment.txt", sep="\t", low_memory=False)
    tx["_start"] = pd.to_numeric(tx["START_DATE"], errors="coerce") / DAYS
    tx["_stop"] = pd.to_numeric(tx["STOP_DATE"], errors="coerce") / DAYS
    for _, r in tx.sort_values(["PATIENT_ID", "_start"]).iterrows():
        p = patients.get(str(r["PATIENT_ID"]))
        if p is None or not np.isfinite(r["_start"]) or r["_start"] < 0:
            continue
        stop = r["_stop"] if np.isfinite(r["_stop"]) else r["_start"]
        p.lines.append({"start": float(r["_start"]),
                        "stop": float(max(stop, r["_start"])),
                        "bucket": bucket(r.get("REGIMEN")),
                        "number": pd.to_numeric(r.get("REGIMEN_NUMBER"),
                                                errors="coerce")})
    return [p for p in patients.values() if p.status]


def prefix_features(p: Patient, tau: float) -> dict[str, float]:
    f: dict[str, float] = {}
    end = min(p.death, tau)
    states = state_intervals(p, tau)
    ctx = p.context

    f["age_at_diagnosis"] = float(pd.to_numeric(ctx.get("age"), errors="coerce"))
    f["sex_female"] = 1.0 if str(ctx.get("sex", "")).lower().startswith("f") else 0.0
    f["histology_adeno"] = 1.0 if "adeno" in str(ctx.get("histology", "")).lower() else 0.0
    f["histology_squamous"] = 1.0 if "squam" in str(ctx.get("histology", "")).lower() \
        else 0.0
    stage = str(ctx.get("stage", "")).lower()
    f["stage_iv"] = 1.0 if ("iv" in stage or "4" in stage) else 0.0

    now = carry(p.status, tau)
    for state in STATES:
        f[f"now_{state}"] = 1.0 if now == state else 0.0
    sites_now = carry(p.sites, tau, set()) or set()
    f["now_n_sites"] = float(len(sites_now))
    on_now = [ln for ln in p.lines if ln["start"] <= tau < ln["stop"]]
    f["now_on_treatment"] = 1.0 if on_now else 0.0
    f["now_line_number"] = float(on_now[0]["number"]) if on_now and np.isfinite(
        on_now[0]["number"]) else float(sum(1 for ln in p.lines if ln["start"] <= tau))
    f["now_months_on_current_line"] = float(tau - on_now[0]["start"]) if on_now else 0.0

    past = [ln for ln in p.lines if ln["start"] < tau]
    completed = [ln for ln in past if ln["stop"] <= tau]
    f["n_prior_completed_lines"] = float(len(completed))
    f["months_on_treatment"] = M.overlap([(ln["start"], min(ln["stop"], tau))
                                          for ln in past], 0.0, tau)
    f["months_off_treatment"] = max(0.0, end - f["months_on_treatment"])
    for name in ("chemotherapy", "immunotherapy", "targeted"):
        f[f"months_on_{name}"] = M.overlap(
            [(ln["start"], min(ln["stop"], tau)) for ln in past
             if ln["bucket"] == name], 0.0, tau)
    durations = [ln["stop"] - ln["start"] for ln in completed]
    f["most_recent_line_duration"] = float(durations[-1]) if durations else np.nan
    f["median_prior_line_duration"] = float(np.median(durations)) if durations else np.nan
    f["n_treatment_buckets_used"] = float(len({ln["bucket"] for ln in past}))

    for state in STATES:
        f[f"months_{state}"] = M.overlap(states.get(state, []), 0.0, tau)
    entries = {state: [] for state in STATES}
    previous = None
    for when, what in p.status:
        if when > tau:
            break
        if what != previous:
            entries[what].append(when)
        previous = what
    f["n_progression_episodes"] = float(len(entries["progressing"]))
    f["n_response_episodes"] = float(len(entries["responding"]))
    f["longest_response_months"] = float(max(
        [b - a for a, b in states.get("responding", [])] + [0.0]))
    f["longest_progression_months"] = float(max(
        [b - a for a, b in states.get("progressing", [])] + [0.0]))
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
    f["n_sites_ever"] = float(len(ever))
    f["n_new_sites_prefix"] = float(len(ever - first))
    return f


def future_outcomes(p: Patient, tau: float, horizon: float) -> dict[str, float]:
    a, b = tau, tau + horizon
    states = state_intervals(p, min(p.death, b))
    before: set[str] = set()
    after: set[str] = set()
    for when, what in p.sites:
        if when <= a:
            before |= what
        elif when <= b:
            after |= what
    entries = {state: [] for state in STATES}
    previous = carry(p.status, a)
    for when, what in p.status:
        if when <= a:
            continue
        if when > b:
            break
        if what != previous:
            entries[what].append(when)
        previous = what
    return {
        "alive_at_horizon": 1.0 if p.death > b else 0.0,
        "months_alive": float(max(0.0, min(p.death, b) - a)),
        "months_responding_future": M.overlap(states.get("responding", []), a, b),
        "months_stable_future": M.overlap(states.get("stable", []), a, b),
        "months_progressing_future": M.overlap(states.get("progressing", []), a, b),
        "n_new_progressions": float(len(entries["progressing"])),
        "n_new_responses": float(len(entries["responding"])),
        "n_new_sites": float(len(after - before)),
        "months_on_treatment_future": M.overlap(
            [(ln["start"], ln["stop"]) for ln in p.lines], a, b),
        "n_subsequent_lines": float(sum(1 for ln in p.lines if a < ln["start"] <= b)),
    }


OUTCOMES = ["alive_at_horizon", "months_alive", "months_responding_future",
            "months_stable_future", "months_progressing_future", "n_new_progressions",
            "n_new_responses", "n_new_sites", "months_on_treatment_future",
            "n_subsequent_lines"]
BINARY = {"alive_at_horizon"}

LADDER = {
    "L0 fixed context": ["age_at_diagnosis", "sex_female", "histology_adeno",
                         "histology_squamous", "stage_iv"],
    "L1 + state today": [f"now_{s}" for s in STATES] + [
        "now_n_sites", "now_on_treatment", "now_line_number",
        "now_months_on_current_line"],
    "L2 + treatment history": ["n_prior_completed_lines", "months_on_treatment",
                               "months_off_treatment", "months_on_chemotherapy",
                               "months_on_immunotherapy", "months_on_targeted",
                               "most_recent_line_duration", "median_prior_line_duration",
                               "n_treatment_buckets_used"],
    "L3 + disease history": [f"months_{s}" for s in STATES] + [
        "n_progression_episodes", "n_response_episodes", "longest_response_months",
        "longest_progression_months", "months_since_last_progression",
        "months_since_last_response", "recent_progression_3mo", "n_sites_ever",
        "n_new_sites_prefix"],
}


def analyse(cohort: str, genie: Path) -> dict | None:
    patients = load(genie)
    if not patients:
        return None
    eligible = [p for p in patients
                if p.death > TAU and (p.last_contact >= TAU + H or p.death <= TAU + H)]
    alive = sum(1 for p in patients if p.death > TAU)
    print(f"\n=== {cohort}: {len(patients)} patients with a status series, "
          f"{np.mean([len(p.status) for p in patients]):.1f} assessments and "
          f"{np.mean([len(p.lines) for p in patients]):.1f} regimens each")
    print(f"    {len(eligible)} eligible ({len(eligible) / max(1, alive):.0%} of those "
          f"alive at the landmark)")
    if len(eligible) < 200:
        print("    too few for a stable fit; skipped")
        return None

    X = pd.DataFrame([prefix_features(p, TAU) for p in eligible])
    Y = pd.DataFrame([future_outcomes(p, TAU, H) for p in eligible])
    groups = np.array([p.pid for p in eligible])
    folds = M.group_folds(groups, FOLDS, SEED)
    cache: dict[tuple, object] = {}

    def predict(cols, outcome):
        key = (tuple(cols), outcome)
        if key not in cache:
            y = Y[outcome].to_numpy(float)
            ok = np.isfinite(y)
            cache[key] = None if (not cols or ok.sum() < 80) else (
                y[ok], M.oof_predict(X.loc[ok, list(cols)].to_numpy(float), y[ok],
                                     folds[ok], groups[ok], SEED), groups[ok])
        return cache[key]

    def score(cols, outcome):
        got = predict(cols, outcome)
        return float("nan") if got is None else M.metric(got[0], got[1],
                                                         outcome in BINARY)

    rungs = list(LADDER)
    nested, cumulative, cum = {}, [], {}
    for rung, feats in LADDER.items():
        cumulative = cumulative + [c for c in feats if c in X.columns]
        cum[rung] = list(cumulative)
        nested[rung] = {o: score(cumulative, o) for o in OUTCOMES}
    full, base_cols = cum[rungs[-1]], cum[rungs[1]]

    history_gain = {}
    for o in OUTCOMES:
        gb, gf = predict(base_cols, o), predict(full, o)
        history_gain[o] = (M.paired_gain(gf[0], gb[1], gf[1], o in BINARY, gf[2], SEED)
                           if gb and gf else {"gain": float("nan"),
                                              "ci": [float("nan")] * 2})
    lobo = {}
    for rung, feats in LADDER.items():
        without = [c for c in full if c not in set(feats)]
        lobo[rung] = {}
        for o in OUTCOMES:
            gb, gf = predict(without, o), predict(full, o)
            lobo[rung][o] = (M.paired_gain(gf[0], gb[1], gf[1], o in BINARY, gf[2],
                                           SEED)["gain"] if gb and gf else float("nan"))

    ymat = Y[OUTCOMES].to_numpy(float)
    keep = np.isfinite(ymat).all(axis=1)
    sd = np.where(ymat[keep].std(axis=0) > 1e-9, ymat[keep].std(axis=0), 1.0)
    ystd = (ymat[keep] - ymat[keep].mean(axis=0)) / sd
    hist_cols = [c for k in rungs[2:] for c in LADDER[k] if c in X.columns]
    stage = X.loc[keep, "stage_iv"].to_numpy(float).astype(str)
    joint = M.joint_permutation(
        X.loc[keep, [c for c in base_cols if c in X.columns]].to_numpy(float),
        X.loc[keep, hist_cols].to_numpy(float), ystd, folds[keep], groups[keep],
        stage, SEED, n_perm=199)
    print(f"    primary joint test: base {joint['base_joint_score']:.3f}  gain "
          f"{joint['observed']:+.4f}  null max {joint['null_max']:+.4f}  "
          f"p = {joint['p']:.4f}")
    significant = sum(1 for o in OUTCOMES if history_gain[o]["ci"][0] > 0)
    print(f"    {significant}/{len(OUTCOMES)} outcomes with intervals above zero")
    for o in OUTCOMES:
        g = history_gain[o]
        print(f"      {o:30s} {nested[rungs[1]][o]:6.3f} -> {nested[rungs[-1]][o]:6.3f}"
              f"  {g['gain']:+6.3f} [{g['ci'][0]:+.3f},{g['ci'][1]:+.3f}]")
    return {"cohort": cohort, "n_patients": len(patients), "n_eligible": len(eligible),
            "mean_assessments": float(np.mean([len(p.status) for p in patients])),
            "mean_regimens": float(np.mean([len(p.lines) for p in patients])),
            "n_features": int(X.shape[1]), "nested": nested,
            "history_gain": history_gain, "lobo": lobo, "joint_primary": joint,
            "n_outcomes_above_zero": int(significant)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for cohort, folder in COHORTS.items():
        genie = GENIE_ROOT / folder
        if not genie.exists():
            print(f"{cohort}: not present locally, skipped")
            continue
        got = analyse(cohort, genie)
        if got:
            results[cohort] = got

    rungs = list(LADDER)
    fig, axes = plt.subplots(2, len(results), figsize=(6.0 * len(results), 10.4),
                             squeeze=False)
    for col, (cohort, r) in enumerate(results.items()):
        ax = axes[0][col]
        ys = np.arange(len(OUTCOMES))
        gains = [r["history_gain"][o]["gain"] for o in OUTCOMES]
        lo = [g - r["history_gain"][o]["ci"][0] for g, o in zip(gains, OUTCOMES,
                                                                strict=True)]
        hi = [r["history_gain"][o]["ci"][1] - g for g, o in zip(gains, OUTCOMES,
                                                                strict=True)]
        ax.barh(ys, gains, 0.62, color="#0072B2", xerr=[lo, hi],
                error_kw={"ecolor": "#333333", "lw": 1.1})
        ax.axvline(0, color="#B00020", lw=1.4)
        ax.set_yticks(ys)
        ax.set_yticklabels([o.replace("_", " ").replace(" future", "") for o in OUTCOMES]
                           if col == 0 else [], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("history gain")
        ax.set_title(f"{cohort}  n={r['n_eligible']}  joint gain "
                     f"{r['joint_primary']['observed']:+.3f}  p="
                     f"{r['joint_primary']['p']:.3f}", fontsize=10.5, loc="left")

        ax2 = axes[1][col]
        mat = np.array([[r["lobo"][k][o] for o in OUTCOMES] for k in rungs])
        vmax = float(np.nanmax(np.abs(mat)))
        im = ax2.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax2.set_yticks(range(len(rungs)))
        ax2.set_yticklabels([k.replace("+ ", "") for k in rungs] if col == 0 else [],
                            fontsize=9)
        ax2.set_xticks(range(len(OUTCOMES)))
        ax2.set_xticklabels([o.replace("_", " ").replace(" future", "")
                             for o in OUTCOMES], rotation=45, ha="right", fontsize=7.6)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax2.text(j, i, f"{mat[i, j]:.02f}".replace("0.", "."), ha="center",
                             va="center", fontsize=7,
                             color="white" if abs(mat[i, j]) > vmax * .55 else "#222222")
        fig.colorbar(im, ax=ax2, fraction=0.035)
        ax2.set_title(f"{cohort}: which block", fontsize=10.5, loc="left")
    fig.suptitle("Do the enriched operators carry information in observed patients?\n"
                 "GENIE-BPC, three cancers, identical registry and identical procedure",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "genie_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (OUT / "readout.json").write_text(json.dumps({
        "source": "GENIE-BPC v2.0-public; aggregate results only, DUA-compliant",
        "landmark_months": TAU, "horizon_months": H, "outcomes": OUTCOMES,
        "channels_absent": ["performance status", "weight", "organ function",
                            "symptom burden", "CTCAE toxicity"],
        "cohorts": results}, indent=2) + "\n")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
