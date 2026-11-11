#!/usr/bin/env python3
"""POC: a clinically engineered prefix representation, and what it buys.

The design under test, in one line: at a landmark tau, summarise the whole past with a
small set of interpretable clinical features Psi(H_tau), and use them to predict a JOINT
set of clinically meaningful future-window outcomes Phi_H -- not survival alone.

THE STATE MATRIX. The generator emits a dense snapshot series per patient: roughly thirty
timepoints from diagnosis, each carrying ECOG, weight, total burden band, active disease
sites, active adverse events, five-organ function, symptom burden, clone counts and vital
status. That series IS the matrix of features across time. A snapshot is written when the
state CHANGES, so the value holds until the next snapshot -- carry-forward is the correct
reading, and there is no missingness in these channels. Events (response, progression,
toxicity with CTCAE grade, treatment start/switch/stop, death) supply the marks.

Every longitudinal variable is passed through the same small set of operators:

    current   value at the landmark
    occupancy cumulative time in a state over a window
    incidence number of new episodes in a window
    recency   time since the most recent event of a kind
    extreme   worst / best value reached
    duration  longest or most recent episode length

The prefix window is [0, tau] with a trailing [tau-3, tau] where recency matters; the
future window is (tau, tau+H]. Because the generator runs every trajectory to 24 months
or death, the future window is FULLY OBSERVED -- no administrative censoring, so window
outcomes are exact.

The ladder asks how much of the future each clinically named part of the history explains
over and above the patient's state today:

    L0  fixed context, including the state at diagnosis
    L1  + the state at the landmark
    L2  + treatment history
    L3  + disease history
    L4  + host and toxicity history

Zero generation, synthetic atlas only, exploratory.

Run:  uv run python scripts/explore/trajectory_feature_poc_v1.py
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_native_atlas_geometry_v1 as A  # noqa: E402

OUT = ROOT / "outputs" / "exploratory_feature_poc"
DAYS = 30.4
HORIZON = 24.0
LANDMARKS = (6.0, 12.0)
H = 12.0
TRAIL = 3.0
SEED = 20260824
FOLDS = 5

BURDEN = {"absent": 0.0, "microscopic": 1.0, "small": 2.0, "moderate": 3.0,
          "bulky": 4.0, "diffuse_or_leukemic": 4.0}
ORGAN_BAD = {"moderate_impairment", "severe_impairment", "failure"}
ORGANS = ("organ_renal", "organ_hepatic", "organ_cardiac", "organ_marrow_reserve",
          "organ_pulmonary")
SYMPTOM = {"none": 0.0, "mild": 1.0, "moderate": 2.0, "severe": 3.0}

CHEMO = {"cytotoxic_chemotherapy", "hypomethylating_agent"}
IMMUNO = {"immune_checkpoint_inhibitor", "other_immunotherapy", "bispecific_antibody",
          "cellular_therapy", "immunomodulatory_imid"}
ENDOCRINE = {"endocrine", "androgen_axis_therapy"}
SUPPORTIVE = {"supportive_or_eol_only"}
ON = {"systemic_treatment_start", "systemic_treatment_switch"}
PROG = {"progression", "recurrence"}


def bucket(classes: list[str]) -> str:
    s = set(classes)
    if s & SUPPORTIVE:
        return "supportive"
    if s & CHEMO:
        return "chemotherapy"
    if s & IMMUNO:
        return "immunotherapy"
    if s & ENDOCRINE:
        return "endocrine"
    return "targeted" if s else "other"


def parse_sites(joined) -> set[str]:
    """'C0:bone marrow(small);C0:peripheral blood(moderate)' -> {'bone marrow', ...}"""
    if not isinstance(joined, str) or not joined.strip():
        return set()
    out = set()
    for part in joined.split(";"):
        part = part.strip()
        if not part:
            continue
        body = part.split(":", 1)[1] if ":" in part else part
        out.add(body.split("(", 1)[0].strip())
    return out


@dataclass
class Traj:
    gid: str
    cancer: str
    baseline: dict
    snap: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    lines: list[dict] = field(default_factory=list)
    tox: list[dict] = field(default_factory=list)
    death: float = np.inf


def build_snapshots(rows: pd.DataFrame, site_map: dict) -> list[dict]:
    out = []
    for _, r in rows.iterrows():
        organ = sum(1 for c in ORGANS if str(r.get(c)) in ORGAN_BAD)
        out.append({
            "t": float(r["_months"]),
            "ecog": float(pd.to_numeric(r.get("ecog_ps"), errors="coerce")),
            "weight": float(pd.to_numeric(r.get("weight_kg"), errors="coerce")),
            "burden": BURDEN.get(str(r.get("total_burden_band")), np.nan),
            "n_sites": float(pd.to_numeric(r.get("num_active_disease_sites"),
                                           errors="coerce")),
            # Canonical SITE CLASSES, not the free-text anatomic labels in the joined
            # string -- those run to more than a thousand distinct strings and would turn
            # a current-state indicator into a thousand-column identifier of the card.
            "sites": site_map.get((str(r["_gid"]),
                                   int(pd.to_numeric(r.get("snapshot_index"),
                                                     errors="coerce") or 0)), set()),
            "n_aes": float(pd.to_numeric(r.get("n_active_aes"), errors="coerce")),
            "organ": float(organ),
            "organ_levels": {c: str(r.get(c)) for c in ORGANS},
            "symptom": SYMPTOM.get(str(r.get("symptom_burden")), np.nan),
            "clones": float(pd.to_numeric(r.get("num_active_clones"), errors="coerce")),
            "acquired": float(pd.to_numeric(r.get("n_acquired_alterations"),
                                            errors="coerce")),
            "line": float(pd.to_numeric(r.get("active_treatment_lines_joined"),
                                        errors="coerce")),
            "on_treatment": 1.0 if isinstance(r.get("active_treatment_regimens_joined"),
                                              str) else 0.0,
            "alive": 0.0 if str(r.get("vital_status")).lower() == "deceased" else 1.0,
        })
    return sorted(out, key=lambda x: x["t"])


def build_events(tj: Traj, rows: pd.DataFrame) -> None:
    open_tox: dict[tuple, dict] = {}
    line_no = 0
    for _, r in rows.iterrows():
        t = float(r["_months"])
        et = str(r["event_type"])
        try:
            pay = json.loads(str(r.get("materialized_payload_json") or "{}"))
        except json.JSONDecodeError:
            pay = {}
        tj.events.append({"t": t, "type": et,
                          "mech": str(r.get("mechanism_bucket") or "unknown")})
        if et in ON:
            classes = (pay.get("regimen") or {}).get("component_classes") or []
            line_no += 1
            if tj.lines:
                tj.lines[-1]["stop"] = min(tj.lines[-1]["stop"], t)
            tj.lines.append({"start": t, "stop": HORIZON, "bucket": bucket(classes),
                             "line_no": line_no})
        elif et == "systemic_treatment_stop" and tj.lines:
            tj.lines[-1]["stop"] = min(tj.lines[-1]["stop"], t)
        elif et == "toxicity":
            key = (str(r.get("ae_category")), str(r.get("ae")))
            trans = str(r.get("toxicity_transition"))
            g = pd.to_numeric(r.get("ctcae_v5_grade"), errors="coerce")
            g = float(g) if pd.notna(g) else np.nan
            if trans == "resolved":
                if key in open_tox:
                    open_tox[key]["stop"] = t
                    tj.tox.append(open_tox.pop(key))
            elif trans == "grade_change" and key in open_tox:
                open_tox[key]["grades"].append((t, g))
            else:
                if key in open_tox:
                    open_tox[key]["stop"] = t
                    tj.tox.append(open_tox.pop(key))
                open_tox[key] = {"start": t, "stop": HORIZON, "category": key[0],
                                 "grades": [(t, g)]}
        elif et == "death":
            tj.death = min(tj.death, t)
    tj.tox.extend(open_tox.values())
    for ln in tj.lines:
        ln["stop"] = min(ln["stop"], tj.death, HORIZON)
    for tx in tj.tox:
        tx["stop"] = min(tx["stop"], tj.death, HORIZON)


# ---------------------------------------------------------------------------------------
# Operators over a window, read off the state matrix.
# ---------------------------------------------------------------------------------------
def overlap(intervals, a: float, b: float) -> float:
    return float(sum(max(0.0, min(y, b) - max(x, a)) for x, y in intervals))


def spans(tj: Traj, end: float) -> list[tuple[float, float, dict]]:
    """Each snapshot holds until the next one; the last holds until death or the horizon."""
    out = []
    for i, row in enumerate(tj.snap):
        a = row["t"]
        b = tj.snap[i + 1]["t"] if i + 1 < len(tj.snap) else end
        if min(b, end) > a:
            out.append((a, min(b, end), row))
    return out


def occupancy(tj: Traj, test, a: float, b: float) -> float:
    end = min(tj.death, HORIZON)
    return float(sum(max(0.0, min(y, b) - max(x, a))
                     for x, y, row in spans(tj, end) if test(row)))


def at(tj: Traj, t: float, key: str, default=np.nan):
    """Carry-forward: the value written at the last snapshot at or before t."""
    value = default
    for row in tj.snap:
        if row["t"] > t:
            break
        value = row[key]
    return value


def upto(tj: Traj, t: float, key: str) -> list[float]:
    return [r[key] for r in tj.snap if r["t"] <= t and np.isfinite(r[key])]


def sites_upto(tj: Traj, a: float, b: float) -> set[str]:
    out: set[str] = set()
    for r in tj.snap:
        if a <= r["t"] <= b:
            out |= r["sites"]
    return out


def tox_grade_at(tx: dict, t: float) -> float:
    seen = [g for s, g in tx["grades"] if s <= t and np.isfinite(g)]
    return float(max(seen)) if seen else np.nan


def merge(intervals: list[tuple]) -> list[tuple]:
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def grade_segments(tx: dict) -> list[tuple[float, float, float]]:
    """An episode is a piecewise-constant GRADE, not one grade for its whole length.

    Grades go down as well as up: the atlas records 2,837 downward grade changes. An
    episode that reaches grade 3 and then settles at grade 1 is not a grade-3 toxicity
    for the rest of its life, and treating it as one inflates every occupancy built on it.
    """
    points = sorted((t, g) for t, g in tx["grades"] if np.isfinite(g))
    out = []
    for i, (t, g) in enumerate(points):
        end = points[i + 1][0] if i + 1 < len(points) else tx["stop"]
        if end > t:
            out.append((t, float(end), float(g)))
    return out


def tox_intervals(tj: Traj, upto_t: float, min_grade: float = 3.0) -> list[tuple]:
    """Windows during which a toxicity was ACTUALLY at grade 3 or worse."""
    out = []
    for x in tj.tox:
        for a, b, g in grade_segments(x):
            if g >= min_grade and min(b, upto_t) > a:
                out.append((a, min(b, upto_t)))
    return out


def g3_crossings(tj: Traj, a: float, b: float, min_grade: float = 3.0) -> float:
    """Count of CROSSINGS into grade 3 or worse inside (a, b].

    A fresh episode that opens at grade 3 counts; so does an existing grade-1 episode
    escalating to grade 3, which counting episode onsets alone would miss.
    """
    total = 0
    for x in tj.tox:
        previous = 0.0
        for start, _, g in grade_segments(x):
            if g >= min_grade > previous and a < start <= b:
                total += 1
            previous = g
    return float(total)


def active_tox(tj: Traj, t: float) -> list[tuple[str, float]]:
    """(category, grade) for every toxicity episode running at time t."""
    out = []
    for x in tj.tox:
        for start, end, g in grade_segments(x):
            if start <= t < end:
                out.append((str(x["category"]), float(g)))
    return out


def disease_states(tj: Traj, upto_t: float) -> dict[str, list[tuple]]:
    """Responding / progressing / unassessed windows, from the marks known by upto_t.

    Between starting therapy and the first assessment the state is UNASSESSED -- calling
    it stable would manufacture biology the generator never asserted.
    """
    marks = sorted((e["t"], e["type"]) for e in tj.events
                   if e["t"] <= upto_t
                   and (e["type"] in ON or e["type"] in PROG or e["type"] == "response"))
    end = min(tj.death, HORIZON, upto_t)
    out: dict[str, list[tuple]] = {}
    state, since = "pre-treatment", 0.0
    for t, kind in marks:
        new = state
        if kind in ON:
            new = "unassessed"
        elif kind == "response":
            new = "responding"
        elif kind in PROG:
            new = "progressing"
        if new != state:
            out.setdefault(state, []).append((since, t))
            state, since = new, t
    out.setdefault(state, []).append((since, max(since, end)))
    return {k: [(a, b) for a, b in v if b > a] for k, v in out.items()}


def prefix_features(tj: Traj, tau: float) -> dict[str, float]:
    f: dict[str, float] = {}
    base = tj.baseline
    first = tj.snap[0]
    states = disease_states(tj, tau)

    # ---- L0: fixed context, including the state at diagnosis -------------------------
    f["age_at_diagnosis"] = float(pd.to_numeric(base.get("age_at_diagnosis"),
                                                errors="coerce"))
    f["sex_female"] = 1.0 if str(base.get("sex")).lower().startswith("f") else 0.0
    f["stage_metastatic"] = 1.0 if any(
        k in str(base.get("stage") or "").lower()
        for k in ("iv", "metasta", "advanced")) else 0.0
    f["n_comorbidities"] = float(pd.to_numeric(base.get("n_comorbidities"),
                                               errors="coerce") or 0.0)
    f["n_baseline_alterations"] = float(pd.to_numeric(
        base.get("n_baseline_alterations"), errors="coerce") or 0.0)
    f["frail"] = 1.0 if str(base.get("frailty_band")).lower() in {"frail", "vulnerable"} \
        else 0.0
    f["baseline_ecog"] = first["ecog"]
    f["baseline_burden"] = first["burden"]
    f["baseline_n_sites"] = first["n_sites"]
    f["baseline_organ_impairments"] = first["organ"]
    f["baseline_symptom_burden"] = first["symptom"]

    # ---- L1: the state at the landmark ----------------------------------------------
    w0 = first["weight"]
    f["now_ecog"] = at(tj, tau, "ecog")
    f["now_weight_ratio"] = at(tj, tau, "weight") / w0 if np.isfinite(w0) and w0 else np.nan
    f["now_burden"] = at(tj, tau, "burden")
    f["now_n_sites"] = at(tj, tau, "n_sites")
    f["now_organ_impairments"] = at(tj, tau, "organ")
    f["now_symptom_burden"] = at(tj, tau, "symptom")
    f["now_active_aes"] = at(tj, tau, "n_aes")
    f["now_clones"] = at(tj, tau, "clones")
    f["now_acquired_alterations"] = at(tj, tau, "acquired")
    f["now_on_treatment"] = at(tj, tau, "on_treatment", 0.0)
    on_now = [ln for ln in tj.lines if ln["start"] <= tau < ln["stop"]]
    f["now_line_number"] = float(on_now[0]["line_no"]) if on_now else float(
        sum(1 for ln in tj.lines if ln["start"] <= tau))
    f["now_months_on_current_line"] = float(tau - on_now[0]["start"]) if on_now else 0.0
    f["now_responding"] = 1.0 if any(a <= tau < b
                                     for a, b in states.get("responding", [])) else 0.0
    f["now_progressing"] = 1.0 if any(a <= tau < b
                                      for a, b in states.get("progressing", [])) else 0.0

    # ---- L1 continued: the COMPLETE current state, not a compact summary -------------
    # Identities, not just counts. Without these, a history feature can win simply by
    # supplying current information the baseline never had, and the comparison stops
    # being "history beyond the state today".
    levels = at(tj, tau, "organ_levels", {}) or {}
    for organ in ORGANS:
        f[f"now_{organ}_impaired"] = 1.0 if str(levels.get(organ)) in ORGAN_BAD else 0.0
    for site in (at(tj, tau, "sites", set()) or set()):
        f[f"now_site::{site}"] = 1.0
    if on_now:
        f[f"now_txbucket::{on_now[0]['bucket']}"] = 1.0
    active = active_tox(tj, tau)
    f["now_max_tox_grade"] = float(max([g for _, g in active], default=0.0))
    f["now_n_active_tox_g3"] = float(sum(1 for _, g in active if g >= 3))
    for category, _ in active:
        f[f"now_aecat::{category}"] = 1.0
    for e in tj.events:
        if e["type"] == "clonal_evolution_event" and e["t"] <= tau:
            f[f"now_mech::{e['mech']}"] = 1.0

    # ---- L2: treatment history -------------------------------------------------------
    past = [ln for ln in tj.lines if ln["start"] < tau]
    completed = [ln for ln in past if ln["stop"] <= tau]
    f["n_prior_completed_lines"] = float(len(completed))
    f["months_on_treatment"] = occupancy(tj, lambda r: r["on_treatment"] > 0.5, 0.0, tau)
    f["months_off_treatment"] = tau - f["months_on_treatment"]
    for name in ("chemotherapy", "immunotherapy", "targeted"):
        f[f"months_on_{name}"] = overlap(
            [(ln["start"], min(ln["stop"], tau)) for ln in past
             if ln["bucket"] == name], 0.0, tau)
    f["n_treatment_stops"] = float(sum(1 for e in tj.events if e["t"] <= tau
                                       and e["type"] == "systemic_treatment_stop"))
    f["n_dose_modifications"] = float(sum(
        1 for e in tj.events if e["t"] <= tau
        and e["type"] == "systemic_treatment_modification"))
    durations = [ln["stop"] - ln["start"] for ln in completed]
    f["most_recent_line_duration"] = float(durations[-1]) if durations else np.nan
    f["median_prior_line_duration"] = float(np.median(durations)) if durations else np.nan
    f["n_treatment_buckets_used"] = float(len({ln["bucket"] for ln in past}))

    # ---- L3: disease history ---------------------------------------------------------
    resp = [e["t"] for e in tj.events if e["type"] == "response" and e["t"] <= tau]
    prog = [e["t"] for e in tj.events if e["type"] in PROG and e["t"] <= tau]
    f["months_responding"] = overlap(states.get("responding", []), 0.0, tau)
    f["months_progressing"] = overlap(states.get("progressing", []), 0.0, tau)
    f["n_responses"] = float(len(resp))
    f["n_progressions"] = float(len(prog))
    f["longest_response_months"] = float(max(
        [b - a for a, b in states.get("responding", [])] + [0.0]))
    f["months_since_last_progression"] = float(tau - max(prog)) if prog else np.nan
    f["months_since_last_response"] = float(tau - max(resp)) if resp else np.nan
    f["recent_progression_3mo"] = 1.0 if any(t > tau - TRAIL for t in prog) else 0.0
    f["n_resistance_events"] = float(sum(1 for e in tj.events
                                         if e["type"] == "clonal_evolution_event"
                                         and e["t"] <= tau))
    f["n_local_interventions"] = float(sum(1 for e in tj.events
                                           if e["type"] == "local_intervention"
                                           and e["t"] <= tau))
    burdens = upto(tj, tau, "burden")
    f["max_burden_reached"] = float(max(burdens)) if burdens else np.nan
    f["burden_auc_per_month"] = float(
        sum((min(b, tau) - a) * row["burden"] for a, b, row in spans(tj, min(tj.death,
                                                                            HORIZON))
            if a < tau and np.isfinite(row["burden"])) / tau)
    f["months_burden_ge_moderate"] = occupancy(
        tj, lambda r: np.isfinite(r["burden"]) and r["burden"] >= 3.0, 0.0, tau)
    f["burden_change_3mo"] = at(tj, tau, "burden") - at(tj, max(0.0, tau - TRAIL),
                                                        "burden")
    f["n_sites_ever"] = float(len(sites_upto(tj, 0.0, tau)))
    f["n_new_sites_prefix"] = float(len(sites_upto(tj, 0.0, tau) - first["sites"]))
    f["max_clones"] = float(max(upto(tj, tau, "clones") or [np.nan]))

    # ---- L4: host and toxicity history -----------------------------------------------
    f["months_ecog_ge2"] = occupancy(
        tj, lambda r: np.isfinite(r["ecog"]) and r["ecog"] >= 2, 0.0, tau)
    ecogs = upto(tj, tau, "ecog")
    f["worst_ecog"] = float(max(ecogs)) if ecogs else np.nan
    f["mean_ecog"] = float(
        sum((min(b, tau) - a) * row["ecog"] for a, b, row in spans(tj, min(tj.death,
                                                                          HORIZON))
            if a < tau and np.isfinite(row["ecog"])) / tau)
    f["ecog_change_3mo"] = at(tj, tau, "ecog") - at(tj, max(0.0, tau - TRAIL), "ecog")
    if np.isfinite(w0) and w0:
        f["months_weight_below_90"] = occupancy(
            tj, lambda r: np.isfinite(r["weight"]) and r["weight"] < 0.9 * w0, 0.0, tau)
        obs = upto(tj, tau, "weight")
        f["nadir_weight_ratio"] = float(min(obs) / w0) if obs else np.nan
        f["weight_change_3mo"] = (at(tj, tau, "weight")
                                  - at(tj, max(0.0, tau - TRAIL), "weight")) / w0
    else:
        f["months_weight_below_90"] = np.nan
        f["nadir_weight_ratio"] = np.nan
        f["weight_change_3mo"] = np.nan
    f["months_any_organ_impaired"] = occupancy(tj, lambda r: r["organ"] >= 1, 0.0, tau)
    f["worst_organ_impairments"] = float(max(upto(tj, tau, "organ") or [np.nan]))
    f["months_with_active_ae"] = occupancy(
        tj, lambda r: np.isfinite(r["n_aes"]) and r["n_aes"] >= 1, 0.0, tau)
    f["mean_symptom_burden"] = float(
        sum((min(b, tau) - a) * row["symptom"] for a, b, row in spans(tj, min(tj.death,
                                                                             HORIZON))
            if a < tau and np.isfinite(row["symptom"])) / tau)
    tox_pre = [x for x in tj.tox if x["start"] <= tau]
    grades = [tox_grade_at(x, tau) for x in tox_pre]
    grades = [g for g in grades if np.isfinite(g)]
    f["n_tox_g3_episodes"] = g3_crossings(tj, 0.0, tau)
    # Two different clinical quantities, both meaningful, previously conflated: how much
    # of the patient's LIFE had a grade-3 toxicity running (merged), and how many
    # episode-months of grade-3 toxicity accumulated (summed, so two concurrent
    # toxicities count twice). Summing overlapping episodes and calling it occupancy
    # produced values larger than the prefix itself.
    f["months_with_active_tox_g3"] = overlap(merge(tox_intervals(tj, tau)), 0.0, tau)
    f["tox_g3_episode_months"] = overlap(tox_intervals(tj, tau), 0.0, tau)
    f["worst_tox_grade"] = float(max(grades)) if grades else 0.0
    f["n_tox_categories"] = float(len({x["category"] for x in tox_pre}))
    return f


def future_outcomes(tj: Traj, tau: float, horizon: float) -> dict[str, float]:
    a, b = tau, tau + horizon
    states = disease_states(tj, HORIZON)
    prog = [e["t"] for e in tj.events if e["type"] in PROG and a < e["t"] <= b]
    resp = [e["t"] for e in tj.events if e["type"] == "response" and a < e["t"] <= b]
    before = sites_upto(tj, 0.0, a)
    ecogs = [r["ecog"] for r in tj.snap if a < r["t"] <= b and np.isfinite(r["ecog"])]
    return {
        "alive_at_horizon": 1.0 if tj.death > b else 0.0,
        "months_alive": float(max(0.0, min(tj.death, b) - a)),
        "months_responding_future": overlap(states.get("responding", []), a, b),
        "months_progressing_future": overlap(states.get("progressing", []), a, b),
        "months_ecog_ge2_future": occupancy(
            tj, lambda r: np.isfinite(r["ecog"]) and r["ecog"] >= 2, a, b),
        "worst_ecog_future": float(max(ecogs)) if ecogs else at(tj, a, "ecog"),
        "months_weight_below_90_future": occupancy(
            tj, lambda r: (np.isfinite(r["weight"]) and np.isfinite(tj.snap[0]["weight"])
                           and r["weight"] < 0.9 * tj.snap[0]["weight"]), a, b),
        "months_organ_impaired_future": occupancy(tj, lambda r: r["organ"] >= 1, a, b),
        "n_new_progressions": float(len(prog)),
        "n_new_responses": float(len(resp)),
        "n_new_sites": float(len(sites_upto(tj, a, b) - before)),
        "n_new_tox_g3": g3_crossings(tj, a, b),
        "months_with_active_tox_g3_future": overlap(merge(tox_intervals(tj, b)), a, b),
        "months_on_treatment_future": occupancy(
            tj, lambda r: r["on_treatment"] > 0.5, a, b),
        "n_subsequent_lines": float(sum(1 for ln in tj.lines if a < ln["start"] <= b)),
    }


PATIENT_OUTCOMES = ["alive_at_horizon", "months_alive", "months_responding_future",
                    "months_progressing_future", "months_ecog_ge2_future",
                    "worst_ecog_future", "months_weight_below_90_future",
                    "months_organ_impaired_future", "n_new_progressions",
                    "n_new_responses", "n_new_sites", "n_new_tox_g3",
                    "months_with_active_tox_g3_future"]
MANAGEMENT_OUTCOMES = ["months_on_treatment_future", "n_subsequent_lines"]
OUTCOME_BLOCKS = {
    "survival": ["alive_at_horizon", "months_alive"],
    "disease state": ["months_responding_future", "months_progressing_future"],
    "host": ["months_ecog_ge2_future", "worst_ecog_future",
             "months_weight_below_90_future", "months_organ_impaired_future"],
    "events": ["n_new_progressions", "n_new_responses", "n_new_sites", "n_new_tox_g3",
               "months_with_active_tox_g3_future"],
    "treatment": ["months_on_treatment_future", "n_subsequent_lines"],
}
BINARY = {"alive_at_horizon"}

LADDER = {
    "L0 fixed context": ["age_at_diagnosis", "sex_female", "stage_metastatic",
                         "n_comorbidities", "n_baseline_alterations", "frail",
                         "baseline_ecog", "baseline_burden", "baseline_n_sites",
                         "baseline_organ_impairments", "baseline_symptom_burden"],
    "L1 + state today": ["now_ecog", "now_weight_ratio", "now_burden", "now_n_sites",
                         "now_organ_impairments", "now_symptom_burden", "now_active_aes",
                         "now_clones", "now_acquired_alterations", "now_on_treatment",
                         "now_line_number", "now_months_on_current_line",
                         "now_responding", "now_progressing", "now_max_tox_grade",
                         "now_n_active_tox_g3",
                         *[f"now_{o}_impaired" for o in ORGANS]],
    "L2 + treatment history": ["n_prior_completed_lines", "months_on_treatment",
                               "months_off_treatment", "months_on_chemotherapy",
                               "months_on_immunotherapy", "months_on_targeted",
                               "n_treatment_stops", "n_dose_modifications",
                               "most_recent_line_duration", "median_prior_line_duration",
                               "n_treatment_buckets_used"],
    "L3 + disease history": ["months_responding", "months_progressing", "n_responses",
                             "n_progressions", "longest_response_months",
                             "months_since_last_progression", "months_since_last_response",
                             "recent_progression_3mo", "n_resistance_events",
                             "n_local_interventions", "max_burden_reached",
                             "burden_auc_per_month", "months_burden_ge_moderate",
                             "burden_change_3mo", "n_sites_ever", "n_new_sites_prefix",
                             "max_clones"],
    "L4 + host and toxicity history": ["months_ecog_ge2", "worst_ecog", "mean_ecog",
                                       "ecog_change_3mo", "months_weight_below_90",
                                       "nadir_weight_ratio", "weight_change_3mo",
                                       "months_any_organ_impaired",
                                       "worst_organ_impairments", "months_with_active_ae",
                                       "mean_symptom_burden", "n_tox_g3_episodes",
                                       "months_with_active_tox_g3", "tox_g3_episode_months",
                                       "worst_tox_grade", "n_tox_categories"],
}


# ---------------------------------------------------------------------------------------
# Ridge regression with out-of-fold evaluation. Pure numpy.
# ---------------------------------------------------------------------------------------
def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    p = x.shape[1]
    xm, ym = x.mean(axis=0), y.mean()
    sd = np.where(x.std(axis=0) > 1e-9, x.std(axis=0), 1.0)
    z = (x - xm) / sd
    beta = np.linalg.solve(z.T @ z + alpha * np.eye(p), z.T @ (y - ym))
    return np.concatenate([[ym - (xm / sd) @ beta], beta / sd])


def ridge_predict(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    return coef[0] + x @ coef[1:]


def auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y > 0.5], p[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def group_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Folds are assigned by ATLAS CARD, never by trajectory.

    Each card contributes about five sibling futures generated from the same prompt and
    the same baseline patient. Splitting by trajectory puts siblings on both sides of the
    fold, so the model is asked to predict another draw from a card it has already seen.
    Splitting by card asks the only question worth asking: a clinical context never seen
    before.
    """
    uniq = np.array(sorted({str(g) for g in groups}))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    lookup = {g: int(order[i] % n_folds) for i, g in enumerate(uniq)}
    return np.array([lookup[str(g)] for g in groups], int)


def supported_columns(block: np.ndarray, min_side: int = 10) -> np.ndarray:
    """Which predictors are estimable from THIS training fold alone.

    Cohort-level cell counts do not catch the failure that matters: a binary feature with
    one exposed patient inside a training fold is unstable even when the cohort is large.
    A binary column needs at least `min_side` observations on BOTH sides; a continuous or
    count column needs at least `min_side` values away from its modal value. Anything else
    is dropped using training data only, and the drop is reported rather than silently
    left estimable.
    """
    keep = []
    for j in range(block.shape[1]):
        column = block[:, j]
        finite = column[np.isfinite(column)]
        if len(finite) < 2 * min_side:
            continue
        levels = np.unique(finite)
        if len(levels) < 2:
            continue
        if len(levels) == 2:
            smaller = min((finite == levels[0]).sum(), (finite == levels[1]).sum())
            if smaller >= min_side:
                keep.append(j)
            continue
        counts = np.bincount(np.searchsorted(levels, finite))
        if len(finite) - counts.max() >= min_side:
            keep.append(j)
    return np.array(keep, int)


def _design(raw: np.ndarray, rows: np.ndarray, med: np.ndarray,
            miss_cols: np.ndarray) -> np.ndarray:
    block = raw[rows]
    miss = ~np.isfinite(block)
    filled = np.where(miss, med, block)
    if miss_cols.size:
        return np.column_stack([filled, miss[:, miss_cols].astype(float)])
    return filled


def oof_predict(raw: np.ndarray, y: np.ndarray, folds: np.ndarray,
                groups: np.ndarray, seed: int,
                alpha_fixed: float | None = None) -> np.ndarray:
    """Out-of-fold predictions. Median imputation and the missingness indicators are
    fitted INSIDE each training fold, never on the whole cohort."""
    pred = np.empty(len(y), float)
    for k in np.unique(folds):
        tr = np.flatnonzero(folds != k)
        te = np.flatnonzero(folds == k)
        block = raw[tr]
        med = np.nanmedian(np.where(np.isfinite(block), block, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        miss_cols = np.flatnonzero((~np.isfinite(block)).any(axis=0))
        xtr, xte, ytr = _design(raw, tr, med, miss_cols), _design(raw, te, med,
                                                                 miss_cols), y[tr]
        if alpha_fixed is not None:
            pred[te] = ridge_predict(ridge_fit(xtr, ytr, alpha_fixed), xte)
            continue
        inner = group_folds(groups[tr], 3, seed + 7 * int(k))
        best, best_score = 10.0, -np.inf
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            errs = []
            for j in np.unique(inner):
                a, b = inner != j, inner == j
                if a.sum() < 20 or b.sum() < 10:
                    continue
                coef = ridge_fit(xtr[a], ytr[a], alpha)
                errs.append(float(((ridge_predict(coef, xtr[b]) - ytr[b]) ** 2).mean()))
            s = -float(np.mean(errs)) if errs else -np.inf
            if s > best_score:
                best, best_score = alpha, s
        pred[te] = ridge_predict(ridge_fit(xtr, ytr, best), xte)
    return pred


def metric(y: np.ndarray, pred: np.ndarray, binary: bool) -> float:
    if binary:
        return auc(y, pred)
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - pred) ** 2).sum()) / ss if ss > 0 else np.nan


def paired_gain(y, base, full, binary, cards, seed, n_boot=399) -> dict:
    """Observed gain plus a CLUSTERED PERCENTILE INTERVAL over cards.

    This resamples around the observed effect, so the fraction of resamples below zero is
    a statement about bootstrap stability, not a null-centred hypothesis test. It is
    reported as an interval and deliberately NOT converted into a p-value, and nothing
    downstream applies a multiplicity correction to it.
    """
    observed = metric(y, full, binary) - metric(y, base, binary)
    uniq = np.array(sorted({str(c) for c in cards}))
    cards = np.array([str(c) for c in cards])
    index = {g: np.flatnonzero(cards == g) for g in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([index[uniq[i]] for i in pick])
        d = (metric(y[rows], full[rows], binary)
             - metric(y[rows], base[rows], binary))
        if np.isfinite(d):
            draws.append(d)
    arr = np.array(draws)
    lo, hi = (np.quantile(arr, [0.025, 0.975]) if arr.size else (np.nan, np.nan))
    share = float((arr <= 0).mean()) if arr.size else np.nan
    return {"gain": float(observed), "ci": [float(lo), float(hi)],
            "share_at_or_below_zero": share, "n_boot": int(arr.size)}


def stratified_group_folds(cards: np.ndarray, cancers: np.ndarray, n_folds: int,
                           seed: int) -> np.ndarray:
    """Sibling groups kept intact, but every cancer appears in training.

    This is the ORDINARY clinical situation: the tumour type is known and represented,
    and only this particular generated card is unseen. It is the complement to the strict
    out-of-cancer split, where the tumour type itself is new.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(len(cards), int)
    for cancer in sorted(set(cancers.tolist())):
        idx = np.flatnonzero(cancers == cancer)
        groups = sorted(set(cards[idx].tolist()))
        perm = rng.permutation(len(groups))
        lookup = {g: int(perm[i] % n_folds) for i, g in enumerate(groups)}
        for i in idx:
            out[i] = lookup[cards[i]]
    return out


def stratified_permutation(strata: np.ndarray, rng) -> np.ndarray:
    """Shuffle only WITHIN a stratum, so cancer, generator and card are all preserved.

    The stratum is one atlas card: a single cancer, a single generator, and the five
    futures it produced. Those five are independent draws from exactly the same
    conditional law, so exchangeability inside a stratum holds by construction rather
    than by assumption. A global shuffle would break history's dependence on cancer,
    generator and current state, and would manufacture clinically impossible
    combinations, making the null far too easy to beat.
    """
    out = np.arange(len(strata))
    for stratum in np.unique(strata):
        where = np.flatnonzero(strata == stratum)
        out[where] = where[rng.permutation(len(where))]
    return out


def joint_score(raw: np.ndarray, ystd: np.ndarray, folds: np.ndarray,
                groups: np.ndarray, seed: int, alphas: list | None) -> float:
    """One prespecified number for the whole future course.

    Every outcome is standardised to unit variance, so months and counts are
    commensurate, and the score is the share of that total variance explained out of
    fold. This is the PRIMARY quantity; the fifteen individual gains are its
    decomposition, reported with intervals rather than as fifteen separate tests.
    """
    sse = 0.0
    for k in range(ystd.shape[1]):
        alpha = None if alphas is None else alphas[k]
        pred = oof_predict(raw, ystd[:, k], folds, groups, seed, alpha)
        sse += float(((ystd[:, k] - pred) ** 2).sum())
    return 1.0 - sse / (ystd.shape[0] * ystd.shape[1])


def choose_alphas(raw: np.ndarray, ystd: np.ndarray, folds: np.ndarray,
                  groups: np.ndarray, seed: int) -> list:
    """Tune once on the observed data, then hold fixed across permutations."""
    out = []
    for k in range(ystd.shape[1]):
        best, best_score = 10.0, -np.inf
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            pred = oof_predict(raw, ystd[:, k], folds, groups, seed, alpha)
            score = -float(((ystd[:, k] - pred) ** 2).mean())
            if score > best_score:
                best, best_score = alpha, score
        out.append(best)
    return out


def block_joint_score(raw: np.ndarray, ystd: np.ndarray, blocks: list,
                      folds: np.ndarray, groups: np.ndarray, seed: int,
                      alphas: list | None) -> float:
    """Outcome BLOCKS weighted equally, not coordinates.

    Weighting fifteen partly redundant coordinates equally lets whichever clinical domain
    happens to have the most coordinates dominate the primary score. Four or five named
    blocks, each contributing equally, is the honest summary of a whole future course.
    """
    per_block = []
    for names in blocks:
        sse = tot = 0.0
        for k in names:
            alpha = None if alphas is None else alphas[k]
            pred = oof_predict(raw, ystd[:, k], folds, groups, seed, alpha)
            sse += float(((ystd[:, k] - pred) ** 2).sum())
            tot += float((ystd[:, k] ** 2).sum())
        per_block.append(1.0 - sse / tot if tot > 0 else np.nan)
    return float(np.nanmean(per_block))


def residualise(hist: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split history into the part the baseline already explains and the part it cannot."""
    design = np.column_stack([np.ones(len(base)), np.nan_to_num(base)])
    coef, *_ = np.linalg.lstsq(design, np.nan_to_num(hist), rcond=None)
    fitted = design @ coef
    return fitted, np.nan_to_num(hist) - fitted


def crossfit_residualise(hist: np.ndarray, base: np.ndarray,
                         folds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The same split, but the baseline-to-history regression is fitted INSIDE each
    training fold and applied to the held-out fold.

    Fitting it once on the whole cohort lets the residual carry information about the
    held-out patients, and residual spread varies with current state and assessment
    cadence, so a single whole-cohort fit is not exchangeable across folds.
    """
    fitted = np.zeros_like(np.nan_to_num(hist))
    clean_h, clean_b = np.nan_to_num(hist), np.nan_to_num(base)
    for k in np.unique(folds):
        tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        dtr = np.column_stack([np.ones(len(tr)), clean_b[tr]])
        dte = np.column_stack([np.ones(len(te)), clean_b[te]])
        coef, *_ = np.linalg.lstsq(dtr, clean_h[tr], rcond=None)
        fitted[te] = dte @ coef
    return fitted, clean_h - fitted


def joint_permutation(raw_base: np.ndarray, raw_hist: np.ndarray, ystd: np.ndarray,
                      folds: np.ndarray, groups: np.ndarray, strata: np.ndarray,
                      seed: int, blocks: list, n_perm: int = 199) -> dict:
    """The single primary test, CONDITIONAL on the baseline.

    Only the part of history the baseline cannot explain is permuted. Shuffling history
    wholesale would break its dependence on context, current state and observation
    intensity, and would then be answering the far easier question of whether real
    history beats clinically incoherent history.
    """
    fitted, residual = residualise(raw_hist, raw_base)
    full = np.column_stack([raw_base, fitted + residual])
    alphas_base = choose_alphas(raw_base, ystd, folds, groups, seed)
    alphas_full = choose_alphas(full, ystd, folds, groups, seed)
    base = block_joint_score(raw_base, ystd, blocks, folds, groups, seed, alphas_base)
    observed = block_joint_score(full, ystd, blocks, folds, groups, seed,
                                 alphas_full) - base
    rng = np.random.default_rng(seed + 101)
    null = []
    for _ in range(n_perm):
        order = stratified_permutation(strata, rng)
        null.append(block_joint_score(np.column_stack([raw_base, fitted + residual[order]]),
                                      ystd, blocks, folds, groups, seed, alphas_full)
                    - base)
    arr = np.array(null)
    return {"base_joint_score": float(base), "observed": float(observed),
            "null_mean": float(arr.mean()), "null_p95": float(np.quantile(arr, 0.95)),
            "null_max": float(arr.max()),
            "p": float((1.0 + (arr >= observed).sum()) / (1.0 + len(arr))),
            "n_perm": int(len(arr))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading the synthetic atlas ...")
    baseline = A.load_table("baseline_rows.csv")
    events = A.load_table("event_rows.csv")
    snaps = A.load_table("snapshot_rows.csv")
    site_rows = A.load_table("disease_site_rows.csv")
    site_rows = site_rows[site_rows["burden_band"].astype(str) != "absent"]
    site_map: dict[tuple[str, int], set[str]] = {}
    for (gid, index), group in site_rows.groupby(["_gid", "snapshot_index"], sort=False):
        site_map[(str(gid), int(index))] = set(group["site_class"].astype(str))
    print(f"  {len(site_map)} snapshot-level site records, "
          f"{site_rows['site_class'].nunique()} distinct site classes")
    for frame in (events, snaps):
        frame["_months"] = pd.to_numeric(frame["days_from_diagnosis"],
                                         errors="coerce") / DAYS
    events = events[events["_months"].notna()
                    & events["_months"].between(0.0, HORIZON, inclusive="both")]
    events = events.sort_values(["_gid", "_months", "event_index", "event_id"])
    snaps = snaps[snaps["_months"].notna()
                  & snaps["_months"].between(0.0, HORIZON, inclusive="both")]
    snaps = snaps.sort_values(["_gid", "_months", "snapshot_index"])
    base_by_gid = baseline.set_index("_gid").to_dict("index")
    events_by_gid = dict(tuple(events.groupby("_gid", sort=False)))

    print("parsing state matrices ...")
    trajectories: list[Traj] = []
    for gid, rows in snaps.groupby("_gid", sort=False):
        base = base_by_gid.get(gid)
        if base is None:
            continue
        tj = Traj(gid=str(gid), cancer=str(base.get("tumor_oncotree_code")),
                  baseline=base, snap=build_snapshots(rows, site_map))
        if not tj.snap:
            continue
        dead = [r["t"] for r in tj.snap if r["alive"] < 0.5]
        if dead:
            tj.death = min(dead)
        ev = events_by_gid.get(gid)
        if ev is not None:
            build_events(tj, ev)
        trajectories.append(tj)
    n_snap = np.mean([len(t.snap) for t in trajectories])
    print(f"  {len(trajectories)} trajectories, {n_snap:.1f} snapshots each on average")

    landmark_data = {}
    for tau in LANDMARKS:
        eligible = [t for t in trajectories if t.death > tau]
        rows_x = [prefix_features(t, tau) for t in eligible]
        rows_y = [future_outcomes(t, tau, H) for t in eligible]
        Xl, Yl = pd.DataFrame(rows_x), pd.DataFrame(rows_y)
        multi = [c for c in Xl.columns if "::" in c]
        Xl[multi] = Xl[multi].fillna(0.0)
        # Frozen, canonical vocabularies. Anything present in under 1% of patients is
        # folded into an "other" indicator for its group, so a rare category can never
        # become a de-facto identifier for one atlas card.
        rare_by_group: dict[str, list[str]] = {}
        for col in multi:
            if float(Xl[col].mean()) < 0.01:
                rare_by_group.setdefault(col.split("::", 1)[0], []).append(col)
        for group, cols in rare_by_group.items():
            Xl[f"{group}::other"] = (Xl[cols].to_numpy().sum(axis=1) > 0).astype(float)
            Xl = Xl.drop(columns=cols)
        cov = {c: float(np.isfinite(Xl[c].astype(float)).mean()) for c in Xl.columns}
        landmark_data[tau] = {"X": Xl, "Y": Yl, "kept": eligible, "coverage": cov,
                              "rows_x": rows_x, "rows_y": rows_y}
        thin = {c: round(v, 2) for c, v in cov.items() if v < 0.999}
        print(f"  landmark {tau:4.0f} mo: {len(eligible)} alive, {Xl.shape[1]} features; "
              f"structurally undefined: {thin}")

    TAU_MAIN = LANDMARKS[-1]
    d = landmark_data[TAU_MAIN]
    X, Y, kept, coverage = d["X"], d["Y"], d["kept"], d["coverage"]
    rows_x, rows_y = d["rows_x"], d["rows_y"]
    print(f"  headline landmark = {TAU_MAIN:.0f} months")

    # Controls. Exact tumour type (not a broad site class) and generator identity, so
    # the history gain cannot be "these features recognise a LUAD prompt" or "these
    # features recognise Gemma's narrative habits". Generator identity is a NUISANCE
    # control here; it is never a clinical feature.
    meta = baseline.set_index("_gid")
    tumour = meta["tumor_oncotree_code"].astype(str)
    model = meta["model_short"].astype(str)
    cards = np.array([str(meta["condition_id"].get(t.gid, t.gid)) for t in kept])
    control_cols: list[str] = []
    dummies = {}
    for name, series in (("tumour", tumour), ("model", model)):
        values = [series.get(t.gid, "unknown") for t in kept]
        for lv in sorted({v for v in values if v != "unknown"}):
            col = f"{name}::{lv}"
            dummies[col] = np.array([1.0 if v == lv else 0.0 for v in values])
            control_cols.append(col)
    X = pd.concat([X, pd.DataFrame(dummies, index=X.index)], axis=1)
    print(f"  controls: {tumour.reindex([t.gid for t in kept]).nunique()} exact tumour "
          f"types + {model.reindex([t.gid for t in kept]).nunique()} generators "
          f"= {len(control_cols)} indicators")
    print(f"  {len(set(cards))} atlas cards, "
          f"{len(kept) / max(1, len(set(cards))):.1f} sibling futures each")

    outcomes = PATIENT_OUTCOMES + MANAGEMENT_OUTCOMES
    sibling = np.array([f"{c}::{meta['model_short'].get(t.gid, '?')}"
                        for c, t in zip(cards, kept, strict=True)])
    # Two splits, named for what they actually test.
    #   held-out cancer            the whole cancer card is unseen: transport to a tumour
    #                              type that never appeared in training at all.
    #   known cancer, cross-generator
    #                              sibling futures stay together and the cancer appears in
    #                              training, but only via OTHER generators -- each
    #                              cancer x generator cell is a single card, so this design
    #                              cannot produce a within-generator unseen-card split.
    #                              It is cross-generator transport, not ordinary
    #                              within-population prediction, and is named accordingly.
    SPLITS = {
        "held-out cancer": group_folds(cards, FOLDS, SEED),
        "known cancer, cross-generator": stratified_group_folds(sibling, cards, FOLDS,
                                                                SEED),
    }
    folds = SPLITS["held-out cancer"]
    cache: dict[tuple, np.ndarray] = {}

    def predict(cols: list[str], outcome: str):
        cols = tuple(c for c in cols if c in X.columns)
        key = (cols, outcome)
        if key in cache:
            return cache[key]
        y = Y[outcome].to_numpy(float)
        ok = np.isfinite(y)
        if not cols or ok.sum() < 100:
            cache[key] = None
            return None
        raw = X.loc[ok, list(cols)].to_numpy(float)
        pred = oof_predict(raw, y[ok], folds[ok], cards[ok], SEED)
        cache[key] = (y[ok], pred, cards[ok], ok)
        return cache[key]

    def score(cols: list[str], outcome: str) -> float:
        got = predict(cols, outcome)
        return float("nan") if got is None else metric(got[0], got[1],
                                                       outcome in BINARY)

    analyses: dict[str, dict] = {}
    passes = [("held-out cancer", "tumour and generator controls", control_cols),
              ("held-out cancer", "no controls", []),
              ("known cancer, cross-generator", "tumour and generator controls",
               control_cols)]
    for split_name, control_name, extra in passes:
        variant = f"{split_name} / {control_name}"
        folds = SPLITS[split_name]
        cache.clear()
        print(f"\nrunning the history ladder, {variant} ...")
        blocks = {k: (list(v) + extra if k.startswith("L0") else list(v))
                  for k, v in LADDER.items()}
        nested, cumulative, cum_cols = {}, [], {}
        for rung, feats in blocks.items():
            cumulative = cumulative + feats
            cum_cols[rung] = list(cumulative)
            nested[rung] = {o: score(cumulative, o) for o in outcomes}
            print(f"  {rung:34s} done")
        full = [c for v in blocks.values() for c in v]
        full_score = {o: score(full, o) for o in outcomes}

        # The headline: everything against context + the complete state today.
        base_cols = cum_cols[list(blocks)[1]]
        history_gain = {}
        for o in outcomes:
            gb, gf = predict(base_cols, o), predict(full, o)
            history_gain[o] = (paired_gain(gf[0], gb[1], gf[1], o in BINARY, gf[2],
                                           SEED) if gb and gf else
                               {"gain": float("nan"), "ci": [float("nan")] * 2})
        lobo = {}
        for rung, feats in blocks.items():
            drop = set(feats)
            without = [c for c in full if c not in drop]
            row = {}
            for o in outcomes:
                gb, gf = predict(without, o), predict(full, o)
                row[o] = (paired_gain(gf[0], gb[1], gf[1], o in BINARY, gf[2], SEED)
                          if gb and gf else {"gain": float("nan")})
            lobo[rung] = row
            print(f"  leave out {rung:24s} done")
        analyses[variant] = {"split": split_name, "controls": control_name,
                             "nested": nested, "lobo": lobo, "full": full_score,
                             "history_gain": history_gain}

    # A real null, on the primary split, for two headline outcomes.
    folds = SPLITS["held-out cancer"]
    base_cols = [c for c in LADDER["L0 fixed context"]] + control_cols + \
                [c for c in X.columns if c.startswith(("now_", "site::"))
                 and c not in control_cols]
    hist_cols = [c for k, v in LADDER.items() if k.startswith(("L2", "L3", "L4"))
                 for c in v if c in X.columns]
    ymat = Y[outcomes].to_numpy(float)
    keep = np.isfinite(ymat).all(axis=1)
    sd = np.where(ymat[keep].std(axis=0) > 1e-9, ymat[keep].std(axis=0), 1.0)
    ystd = (ymat[keep] - ymat[keep].mean(axis=0)) / sd
    raw_base = X.loc[keep, [c for c in base_cols if c in X.columns]].to_numpy(float)
    raw_hist = X.loc[keep, hist_cols].to_numpy(float)
    print(f"\nprimary test: joint future-course score over {len(outcomes)} standardised "
          f"outcomes, {int(keep.sum())} patients")
    blocks = [[outcomes.index(o) for o in names] for names in OUTCOME_BLOCKS.values()]
    permutation = {}
    for split_name in SPLITS:
        permutation[split_name] = joint_permutation(
            raw_base, raw_hist, ystd, SPLITS[split_name][keep], cards[keep],
            sibling[keep], SEED, blocks, n_perm=199)
        r = permutation[split_name]
        print(f"  {split_name:30s} base {r['base_joint_score']:.3f}  gain "
              f"{r['observed']:+.4f}  null mean {r['null_mean']:+.4f}  95th "
              f"{r['null_p95']:+.4f}  max {r['null_max']:+.4f}  p = {r['p']:.4f}")

    rungs = list(LADDER)

    PRIMARY = "held-out cancer / tumour and generator controls"
    shown = [PRIMARY, "known cancer, cross-generator / tumour and generator controls"]
    fig, axes = plt.subplots(1, 2, figsize=(17.4, 7.2))
    for ax, variant in zip(axes, shown, strict=True):
        lobo = analyses[variant]["lobo"]
        mat = np.array([[lobo[r][o]["gain"] for o in outcomes] for r in rungs])
        vmax = float(np.nanmax(np.abs(mat)))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_yticks(range(len(rungs)))
        ax.set_yticklabels([r.replace("+ ", "") for r in rungs], fontsize=9)
        ax.set_xticks(range(len(outcomes)))
        ax.set_xticklabels([o.replace("_", " ").replace(" future", "") for o in outcomes],
                           rotation=45, ha="right", fontsize=7.6)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.02f}".replace("0.", "."), ha="center",
                            va="center", fontsize=7.2,
                            color="white" if abs(mat[i, j]) > vmax * 0.55 else "#222222")
        ax.axvline(len(PATIENT_OUTCOMES) - 0.5, color="#333333", lw=1.4)
        ax.set_title(variant, fontsize=10.5, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.03, label="loss when this block is removed")
    fig.suptitle("Which part of the history predicts which future\n"
                 "Drop one block from the full model; the number is how much prediction "
                 "is lost. Order-free.", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "poc_blocks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14.0, 7.4))
    width = 0.16
    colours = ["#BBBBBB", "#56B4E9", "#0072B2", "#009E73", "#D55E00"]
    xs = np.arange(len(outcomes))
    nested = analyses[PRIMARY]["nested"]
    for i, (rung, colour) in enumerate(zip(rungs, colours, strict=True)):
        ax.bar(xs + (i - 2) * width, [nested[rung][o] for o in outcomes], width,
               color=colour, label=rung)
    ax.axhline(0, color="#333333", lw=1)
    ax.axvline(len(PATIENT_OUTCOMES) - 0.5, color="#999999", ls="--", lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([o.replace("_", " ") for o in outcomes], rotation=42, ha="right",
                       fontsize=8.5)
    ax.set_ylabel("out-of-fold variance explained  (AUC for the binary outcome)")
    ax.set_title("What each clinically named part of the history adds\n"
                 "Nested feature sets, five-fold out-of-sample, landmark "
                 f"{TAU_MAIN:.0f} months, horizon {H:.0f} months",
                 fontsize=12.5, loc="left")
    ax.legend(fontsize=9, frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "poc_ladder.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, (axc, axj) = plt.subplots(1, 2, figsize=(16.0, 6.8),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})
    mat = Y[outcomes].to_numpy(float)
    keep = np.isfinite(mat).all(axis=1)
    corr = np.corrcoef(mat[keep].T)
    im = axc.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    short = [o.replace("_", " ").replace(" future", "") for o in outcomes]
    axc.set_xticks(range(len(outcomes)))
    axc.set_xticklabels(short, rotation=45, ha="right", fontsize=7.2)
    axc.set_yticks(range(len(outcomes)))
    axc.set_yticklabels(short, fontsize=7.2)
    for i in range(len(outcomes)):
        for j in range(len(outcomes)):
            axc.text(j, i, f"{corr[i, j]:.1f}".replace("0.", "."), ha="center",
                     va="center", fontsize=6.0,
                     color="white" if abs(corr[i, j]) > 0.6 else "#333333")
    fig.colorbar(im, ax=axc, fraction=0.04)
    axc.set_title("The outcomes are dependent, not interchangeable", fontsize=10.5,
                  loc="left")
    cancers = pd.Series([t.cancer for t in kept])
    jitter = np.random.default_rng(1)
    for code, colour in zip(cancers.value_counts().index[:3],
                            ("#0072B2", "#D55E00", "#009E73"), strict=False):
        sel = (cancers == code).to_numpy() & keep
        axj.scatter(Y["months_alive"].to_numpy()[sel] + jitter.normal(0, 0.12, sel.sum()),
                    Y["months_responding_future"].to_numpy()[sel], s=34, alpha=0.6,
                    color=colour, edgecolor="white", lw=0.6, label=str(code))
    axj.set_xlabel(f"months alive in the next {H:.0f} months")
    axj.set_ylabel(f"months responding in the next {H:.0f} months")
    axj.set_title("One fan is a joint distribution, not a point", fontsize=10.5, loc="left")
    axj.legend(fontsize=9, frameon=False, title="cancer")
    fig.tight_layout()
    fig.savefig(OUT / "poc_joint.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "landmark_months": TAU_MAIN, "landmarks": list(LANDMARKS), "horizon_months": H,
        "n_trajectories": len(trajectories), "n_eligible": len(kept),
        "mean_snapshots": float(n_snap),
        "n_features": int(len(X.columns) - len(control_cols)),
        "n_cards": int(len(set(cards))),
        "controls": len(control_cols),
        "coverage": coverage,
        "analyses": analyses,
        "permutation": permutation,
        "vocabularies": {
            group: {"levels": sorted(c.split("::", 1)[1] for c in X.columns
                                     if c.startswith(group + "::")),
                    "min_prevalence": float(min(
                        X[c].mean() for c in X.columns if c.startswith(group + "::")))}
            for group in sorted({c.split("::", 1)[0] for c in X.columns if "::" in c})},
        "splits": {k: int(len(set(v.tolist()))) for k, v in SPLITS.items()},
        "ladder_features": {k: list(v) for k, v in LADDER.items()},
        "patient_outcomes": PATIENT_OUTCOMES, "management_outcomes": MANAGEMENT_OUTCOMES,
        "outcome_correlation": {"outcomes": outcomes, "matrix": corr.tolist()},
        "outcome_summary": {c: {"mean": float(np.nanmean(Y[c])),
                                "sd": float(np.nanstd(Y[c]))} for c in Y.columns},
        "examples": [
            {"gid": kept[i].gid, "cancer": kept[i].cancer,
             "n_snapshots": len(kept[i].snap),
             "prefix": {k: (None if not np.isfinite(v) else float(v))
                        for k, v in rows_x[i].items()},
             "future": {k: (None if not np.isfinite(v) else float(v))
                        for k, v in rows_y[i].items()}}
            for i in (0, len(kept) // 3, 2 * len(kept) // 3)],
    }
    (OUT / "poc.json").write_text(json.dumps(payload, indent=2) + "\n")
    X.to_csv(OUT / "prefix_features.csv", index=False)
    Y.to_csv(OUT / "future_outcomes.csv", index=False)

    for variant, res in analyses.items():
        print(f"\n=== nested ladder, out-of-fold, {variant} ===")
        print(f"{'outcome':34s} " + "  ".join(f"{r.split()[0]:>6s}" for r in rungs))
        for o in outcomes:
            print(f"{o:34s} " + "  ".join(f"{res['nested'][r][o]:6.3f}" for r in rungs))
        print(f"--- leave-one-block-out (loss when removed), {variant} ---")
        for o in outcomes:
            print(f"{o:34s} " + "  ".join(f"{res['lobo'][r][o]['gain']:6.3f}"
                                          for r in rungs))
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
