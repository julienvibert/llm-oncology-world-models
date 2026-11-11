#!/usr/bin/env python3
"""Gate 4 step 3: the cohort extraction (called by `score_four_arm_final_v1.py --score`).

Specification: `manuscript/SCORING_SPEC_2026-09-28.md` §3-§9, its 30 September amendment and
`manuscript/SCORING_SPEC_amendment_2026-10-12.md` (including §11, the step-1 definitions and
the step-3 acceptance gates). Scores come from `four_arm_scores_v1`; courses, coordinates and
windows from `four_arm_eligibility_v2`; eligibility from the saved precheck tables, re-derived
here and checked to agree before any score is computed.

Read-only on the frozen run, the saved query universe and prompts, and the local GENIE-BPC
releases. Never calls a provider and never rebuilds a prompt. Patient-level rows go only to
the gitignored `private/` folder; aggregate tables to `outputs/four_arm_preprint_20260928/`.
The console prints counts only.

Primary questions (fixed before extraction):
- joint score: equal-fan-size energy score, four contrasts (2-1, 3-2, 4-3, 4-1) at four
  horizons, primary convention, pooled: one Holm family of 16 slots;
- donor comparison: the section-6 documented-interval fallback (amendment of 12 October 2026),
  LLM arm that received the card minus the card, time-aligned disagreement, two cards at four
  horizons: one Holm family of 8 slots. Occupancy total variation is reported beside it.
Everything else is a sensitivity or a descriptive check and carries no adjusted P value.
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
sys.path.insert(0, str(ROOT / "src"))
import four_arm_eligibility_v2 as E  # noqa: E402
import four_arm_readers_v2 as R  # noqa: E402
import four_arm_scores_v1 as S  # noqa: E402
import genie_operator_registry_v2 as GV2  # noqa: E402

from synthonccases.oncollm.parse_utils import strip_code_fence  # noqa: E402

SUFFIX = "_scores_20261012"
CONTRASTS = (("2-1", 1, 0), ("3-2", 2, 1), ("4-3", 3, 2), ("4-1", 3, 0))
LOSSES = S.LOSSES                       # occupancy_tv, disagreement
PRIMARY_LOSS = "disagreement"
TX_SCALARS = ("months_any", "n_subsequent_regimens")     # §5 treatment scalars, clinical units
RMST_H = 24.0
#: The existing Cox baseline (scripts/explore/four_arm_baselines_v1.py): covariates, ridge,
#: ten folds stratified by cancer and event, fold seed. Re-fitted here with persisted fold IDs.
COX_FRAME = ROOT / "outputs" / "exploratory_four_arm_deep" / "baseline_landmark_frame.csv"
COX_SAVED = ROOT / "outputs" / "exploratory_four_arm_deep" / "baseline_forecasts.csv"
COX_COVARIATES = ("age_landmark", "sex_female", "stage_iv", "mets_documented", "line_number",
                  "months_since_dx", "state_progressing", "state_responding", "state_stable",
                  "state_ned", "recent_progression_3mo", "histology_squamous",
                  "panel_reported", "driver_present")
COX_FOLDS, COX_SEED, COX_PENALTY = 10, 20260908, 0.1


# ======================================================================================
# Small helpers
# ======================================================================================

def _r(x, nd: int | None = None):
    """Full precision for every analysis and machine-readable value (step-3 acceptance gate R2);
    rounding is for manuscript and figure display only. `nd` is accepted and ignored."""
    if x is None:
        return ""
    if isinstance(x, (float, np.floating)):
        return "" if not math.isfinite(float(x)) else float(x)
    return x


def adjust_family(pvalues: dict, slots: list) -> dict:
    """Holm over a FIXED family of slots, on unrounded P values; a slot that is missing or
    None stays unavailable and still counts in the family size."""
    adj = S.holm([pvalues.get(s) for s in slots])
    return dict(zip(slots, adj, strict=True))


def eligibility_mismatches(saved: dict, derived: dict) -> dict:
    """Saved precheck inclusion against re-derived inclusion, in both directions."""
    keys = set(saved) | set(derived)
    return {"saved_only": sorted(k for k in keys if saved.get(k) and not derived.get(k)),
            "derived_only": sorted(k for k in keys if derived.get(k) and not saved.get(k))}


def load_precheck_sets(path: Path, queries: set[str]) -> pd.DataFrame:
    """The saved precheck inclusion grid, validated: every query
    at every horizon under both conventions exactly once, Boolean flags only."""
    sets = pd.read_csv(path)
    grid = sets[sets.convention.isin(list(E.CONVENTIONS))]
    if grid.duplicated(["query_id", "horizon_months", "convention"]).any():
        raise RuntimeError("duplicate rows in the precheck grid")
    if len(grid) != len(queries) * len(E.HORIZONS) * len(E.CONVENTIONS) \
            or set(grid.query_id) != queries:
        raise RuntimeError("incomplete precheck grid")
    col = grid["four_arm_intersection"]
    if col.isna().any() or not col.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise RuntimeError("missing or non-Boolean four-arm inclusion flags")
    return sets.set_index(["query_id", "horizon_months", "convention"])


def assign_joint_holm(rows: list[dict], slots: list) -> int:
    """Holm over the fixed joint family, read back from the contrast rows themselves:
    equal-size, primary, pooled contrast rows; P values unrounded. Returns the number of
    available slots; writes `p_holm_primary_family` into those rows."""
    index = {(r["horizon_months"], r["row"].split(":")[1]): i for i, r in enumerate(rows)
             if r["quantity"] == "equal_size" and r["convention"] == "primary"
             and r["cancer"] == "all" and r["row"].startswith("contrast:")}
    pv = {s: (None if rows[index[s]]["p_wilcoxon"] in ("", None) else rows[index[s]]["p_wilcoxon"])
          for s in slots if s in index}
    for s, a in adjust_family(pv, slots).items():
        if s in index:
            rows[index[s]]["p_holm_primary_family"] = _r(a)
    return sum(1 for s in slots if pv.get(s) is not None)


def paired_matrix(table: dict, n_arms: int = 4) -> tuple[list[str], np.ndarray]:
    """Patients with a value for EVERY arm, in a fixed order; refuses partial pairing."""
    qids = sorted(table)
    for q in qids:
        if len(table[q]) != n_arms or any(v is None for v in table[q]):
            raise ValueError("a patient lacks a paired value for some arm")
    return qids, np.asarray([table[q] for q in qids], float).reshape(len(qids), n_arms)


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path.name}")
    pd.DataFrame(rows).to_csv(path, index=False)


def summarise_paired(diffs: np.ndarray, strata: list[str], *keys) -> dict:
    """Mean paired difference, cohort-bootstrap 95% interval, Wilcoxon P; n with reason."""
    d = np.asarray(diffs, float)
    if len(d) == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "p_wilcoxon": None}
    ci = S.paired_mean_ci(d, S.bootstrap_cohorts(strata, *keys))
    return ci | {"p_wilcoxon": S.wilcoxon_p(d)}


def equal_cancer_mean(values: np.ndarray, strata: list[str], *keys) -> dict:
    """Mean of cancer means (sensitivity when evaluability changes the cancer balance)."""
    v = np.asarray(values, float)
    st = np.asarray(strata)
    cancers = sorted(set(strata))

    def stat(idx: np.ndarray) -> float:
        return float(np.mean([v[idx][st[idx] == c].mean() for c in cancers
                              if (st[idx] == c).any()]))
    boot = [stat(idx) for idx in S.bootstrap_cohorts(strata, "equal_cancer", *keys)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": stat(np.arange(len(v))), "ci_low": float(lo), "ci_high": float(hi)}


# ======================================================================================
# Survival helpers: cohort-level IPCW with weights re-estimated inside each resample
# ======================================================================================

def brier_on(pop: np.ndarray, w: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
    """IPCW Brier over the analysis population (denominator = its patient count)."""
    known = ~np.isnan(y[pop])
    loss = np.where(known, (p[pop] - np.nan_to_num(y[pop])) ** 2, 0.0)
    return float((w[pop] * loss).sum() / len(pop))


def km_rmst(time_: np.ndarray, event: np.ndarray, h: float) -> float | None:
    """Kaplan-Meier restricted mean to h; None without support to h unless S reached 0."""
    t = S.tie_groups(np.asarray(time_, float))
    ev = np.asarray(event).astype(bool)
    s, last, area = 1.0, 0.0, 0.0
    for u in np.unique(t[ev & (t < h)]):
        area += s * (u - last)
        s *= 1.0 - int((ev & (t == u)).sum()) / int((t >= u).sum())
        last = u
    if s > 0 and not (t >= h - S.EPS).any():
        return None
    return float(area + s * (h - last))


def cox_rebuild(horizons: tuple[float, ...]) -> tuple[pd.DataFrame, dict]:
    """The existing cross-validated Cox model, fold IDs persisted, 3-month readout added.

    Same frame, covariates, missingness rule (indicator + training-fold median), ridge and
    fold assignment as `four_arm_baselines_v1.cox_cross_validated`; checked against its saved
    6/12/24-month forecasts. Predictions are joined by label, never by stratum order.
    """
    from lifelines import CoxPHFitter

    df = pd.read_csv(COX_FRAME)
    cols = list(COX_COVARIATES) + [c for c in df.columns if c.endswith("_missing")]
    rng = np.random.default_rng(COX_SEED)
    key = df.cancer + "_" + df.event.astype(str)
    fold = np.zeros(len(df), int)
    for _k, idx in df.groupby(key).indices.items():
        idx = rng.permutation(idx)
        fold[idx] = np.arange(len(idx)) % COX_FOLDS
    preds = []
    for k in range(COX_FOLDS):
        train, test = df[fold != k].copy(), df[fold == k].copy()
        med = train[list(COX_COVARIATES)].median()
        train[list(COX_COVARIATES)] = train[list(COX_COVARIATES)].fillna(med)
        test[list(COX_COVARIATES)] = test[list(COX_COVARIATES)].fillna(med)
        use = [c for c in cols if train[c].std() > 1e-6]
        cph = CoxPHFitter(penalizer=COX_PENALTY)
        cph.fit(train[use + ["time", "event", "cancer"]], duration_col="time",
                event_col="event", strata=["cancer"])
        sf = cph.predict_survival_function(test[use + ["cancer"]], times=list(horizons))
        full = cph.predict_survival_function(test[use + ["cancer"]])
        for label, qid in zip(test.index, test.query_id, strict=True):
            curve = full[label].dropna()
            tt, ss = curve.index.to_numpy(float), curve.to_numpy(float)
            t = np.concatenate([[0.0], tt[tt < RMST_H], [RMST_H]])
            sv = np.concatenate([[1.0], ss[tt < RMST_H]])
            preds.append({"query_id": qid, "fold": int(fold[label]),
                          **{f"p_alive_{h:g}": float(sf.loc[h, label]) for h in horizons},
                          "rmst24": float(np.sum(np.diff(t) * sv))})
    out = pd.DataFrame(preds)
    saved = pd.read_csv(COX_SAVED)
    saved = saved[saved.forecaster == "cox_cv"].set_index("query_id")
    check = out.set_index("query_id").join(saved, rsuffix="_saved", how="inner")
    diffs = {h: float((check[f"p_alive_{h:g}"] - check[f"p_alive_{h:g}_saved"]).abs().max())
             for h in (6.0, 12.0, 24.0)}
    return out, {"n": int(len(out)), "matched_saved_rows": int(len(check)),
                 "max_abs_diff_vs_saved": diffs,
                 "max_abs_diff_rmst_vs_saved": float((check["rmst24"] - check["rmst"]).abs().max())}


# ======================================================================================
# The extraction
# ======================================================================================

def extract(out: Path) -> int:
    import score_four_arm_final_v1 as F

    t0 = time.time()
    private = out / "private"
    queries, quartets, attempts, checks = F.load_and_check()
    print(f"[inputs] section-2 checks passed: {checks['requested_slots']} slots, "
          f"{checks['accepted_responses']} accepted")
    sets = load_precheck_sets(private / f"query_sets{F.SUFFIX}.csv", set(queries))
    H = E.HORIZONS
    CONV = tuple(E.CONVENTIONS)
    cancer_of = {q: queries[q]["cancer"] for q in queries}

    # ---- real side: courses, coordinates, windows, donors ---------------------------------
    real: dict[tuple, dict] = {}          # (qid, h, conv) -> query objects
    donors: dict[tuple, list] = {}        # (qid, card, h, conv) -> [(labelled, window, course)]
    surv_real: dict[str, tuple[float, int]] = {}
    donor_surv: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    calendars: dict[str, list[float]] = {}
    donor_tx: dict[tuple, list] = {}       # (qid, card, h) -> per member raw treatment totals
    for cancer in F.CANCERS:
        genie = GV2.GENIE_ROOT / GV2.COHORTS[cancer]
        patients = {p.pid: p for p in GV2.load(genie)}
        agents, _ = R.load_real_agents(genie)
        cache: dict[tuple, E.Course] = {}

        def course(pid: str, tau: float, h: float, conv: str, patients=patients,
                   agents=agents, cache=cache) -> E.Course:
            k = (pid, tau, h, conv)
            if k not in cache:
                with E.convention(conv):
                    cache[k] = E.real_course(patients[pid], tau, h, agents.get(pid, []))
            return cache[k]

        for q in (x for x in queries.values() if x["cancer"] == cancer):
            qid, tau = q["query_id"], float(q["tau_months"])
            p = patients[q["patient"]]
            cal = E.calendar_for(p, tau, quartets[qid]["assessment_times"])
            calendars[qid] = cal
            died = math.isfinite(p.death)
            surv_real[qid] = ((p.death if died else p.last_contact) - tau, int(died))
            for card, (field, _arm) in E.CARDS.items():
                tt, ee = [], []
                for d in q[field]:
                    dp = patients[d["patient"]]
                    dd = math.isfinite(dp.death)
                    tt.append((dp.death if dd else dp.last_contact) - float(d["tau_months"]))
                    ee.append(int(dd))
                donor_surv[(qid, card)] = (np.asarray(tt), np.asarray(ee))
            for h in H:
                for conv in CONV:
                    c = course(q["patient"], tau, h, conv)
                    lab = S.labelled(E.observe(c.path, cal, min(c.death, h)), c.death, h)
                    sup = E.supported(c, cal, h)
                    real[(qid, h, conv)] = {"coords": E.coordinates(c, cal, h), "labelled": lab,
                                            "support": sup, "course": c}
                    for card, (field, _arm) in E.CARDS.items():
                        mem, tx = [], []
                        for d in q[field]:
                            dc = course(d["patient"], float(d["tau_months"]), h, conv)
                            mem.append((S.labelled(E.observe(dc.path, cal, min(dc.death, h)),
                                                   dc.death, h),
                                        E.donor_window(sup, dc, cal, h)))
                            if conv == "primary":
                                raw = E.coordinates(dc, cal, h)["raw"]
                                tx.append({k: raw[k] for k in TX_SCALARS})
                        donors[(qid, card, h, conv)] = mem
                        if conv == "primary":
                            donor_tx[(qid, card, h)] = tx
        print(f"[real] {cancer} done ({time.time() - t0:.0f}s)")

    # ---- generated side --------------------------------------------------------------------
    gen: dict[tuple, list[dict]] = defaultdict(list)     # (qid, arm) -> repeats
    first_status: Counter = Counter()
    for a in attempts:
        first_status[(a["cancer"], a["arm"], f"first_{a['first_status']}")] += 1
        first_status[(a["cancer"], a["arm"], "requested")] += 1
        if not a["accepted"]:
            continue
        first_status[(a["cancer"], a["arm"], "accepted")] += 1
        qid, arm = a["query_id"], a["arm"]
        qt = quartets[qid]
        kw = {"cutoff": qt["arms"][arm]["conditioning_payload"]["fixed_events"]["cutoff_date"],
              "landmark_state": qt["landmark_state"],
              "landmark_presence": qt["landmark_presence"],
              "landmark_sites": qt["landmark_sites"],
              "landmark_extent": qt["landmark_extent"], "index_regimen": qt["index_regimen"]}
        rep = {"repeat": a["repeat"], "first_valid": a["first_valid"], "h": {}}
        try:
            ledger = json.loads(strip_code_fence(a["ledger_text"]))["ledger"]
        except Exception:                                               # noqa: BLE001
            ledger = None
        a["ledger_text"] = None
        for h in H:
            if ledger is None:
                rep["h"][h] = None
                continue
            try:
                c = E.generated_course(ledger, h, **kw)
            except R.NotScoreable:
                rep["h"][h] = None
                continue
            co = E.coordinates(c, calendars[qid], h)
            rep["h"][h] = {
                "vector": co["weighted_vector"], "alive": co["raw"]["alive_at_h"],
                **{k: co["raw"][k] for k in TX_SCALARS},
                "labelled": S.labelled(E.observe(c.path, calendars[qid], min(c.death, h)),
                                       c.death, h),
                "death": c.death, "follow_end": c.follow_end,
                "timing": S.progression_to_regimen(
                    co["observed_path"], [st.time for st in c.treatment.starts if st.time > 0],
                    min(c.death, h))}
        gen[(qid, arm)].append(rep)
    print(f"[generated] {sum(len(v) for v in gen.values())} accepted ledgers projected "
          f"({time.time() - t0:.0f}s)")

    def fan(qid: str, arm: str, h: float, first_only: bool = False) -> list[dict | None]:
        return [r["h"][h] for r in gen[(qid, arm)] if r["first_valid"] or not first_only]

    # ---- consistency with the frozen precheck ----------------------------------------------
    saved_incl, derived_incl = {}, {}
    for (qid, h, conv), row in sets.iterrows():
        if conv not in CONV:
            continue
        saved_incl[(qid, h, conv)] = bool(row["four_arm_intersection"])
        derived_incl[(qid, h, conv)] = real[(qid, h, conv)]["coords"]["weighted_vector"] \
            is not None and all(
                len(f) >= E.MIN_ACCEPTED and all(x is not None and x["vector"] is not None
                                                 for x in f)
                for f in (fan(qid, arm, h) for arm in E.ARMS))
    mm = eligibility_mismatches(saved_incl, derived_incl)
    if mm["saved_only"] or mm["derived_only"]:
        raise RuntimeError(f"eligibility differs from the precheck: {len(mm['saved_only'])} "
                           f"saved only, {len(mm['derived_only'])} re-derived only")
    print(f"[check] four-arm eligibility agrees with the precheck in both directions "
          f"({len(saved_incl)} rows)")

    rows_joint, rows_joint_private, rows_dep = [], [], []
    rows_course, rows_course_private = [], []
    rows_donor, rows_donor_private = [], []
    rows_sens, rows_timing, rows_comp = [], [], []

    def composition(name: str, qids: list[str], h: float) -> None:
        def stats(qs: list[str]) -> dict:
            t = np.array([surv_real[q][0] for q in qs])
            e = np.array([surv_real[q][1] for q in qs])
            y = S.outcome_at(t, e, h)
            y6 = S.outcome_at(t, e, 6.0)
            return {"n": len(qs), **{f"n_{c}": sum(cancer_of[q] == c for q in qs)
                                     for c in F.CANCERS},
                    "deaths_by_h": int((y == 0).sum()), "alive_through_h": int((y == 1).sum()),
                    "unresolved_at_h": int(np.isnan(y).sum()),
                    "deaths_by_6": int((y6 == 0).sum()),
                    "death_fraction_6": _r((y6 == 0).mean() if len(qs) else None, 4)}
        rows_comp.append({"population": name, "horizon_months": h, **stats(qids),
                          **{f"cohort_{k}": v for k, v in stats(list(queries)).items()}})

    # ---- §7 joint score -----------------------------------------------------------------------
    per_patient: dict[tuple, dict] = {}            # (h, conv, quantity) -> {qid: [4 arm scores]}
    for h in H:
        for conv in CONV:
            pop = sorted(q for q in queries if sets.loc[(q, h, conv), "four_arm_intersection"])
            composition(f"joint:{conv}", pop, h)
            for qid in pop:
                y = np.asarray(real[(qid, h, conv)]["coords"]["weighted_vector"], float)
                fans = {arm: np.asarray([x["vector"] for x in fan(qid, arm, h)], float)
                        for arm in E.ARMS}
                m_star = S.four_arm_m_star([len(fans[a]) for a in E.ARMS])
                for j, arm in enumerate(E.ARMS):
                    js = S.joint_scores(fans[arm], y, m_star, "four_arm", qid, arm, h)
                    vals = {"equal_size": js.equal_size, "as_delivered": js.as_delivered,
                            "fair": js.fair,
                            **{f"block_{b}": v for b, v in js.blocks_equal_size.items()}}
                    for k, v in vals.items():
                        per_patient.setdefault((h, conv, k), {}).setdefault(
                            qid, [None] * 4)[j] = v
                    rows_joint_private.append({"query_id": qid, "cancer": cancer_of[qid],
                                               "horizon_months": h, "convention": conv,
                                               "arm": arm, "m": js.m, "m_star": m_star,
                                               **{k: _r(v) for k, v in vals.items()}})
                    if conv == "primary":
                        d = S.dependence_check(fans[arm], y, qid, arm, h)
                        rows_dep.append({"query_id": qid, "cancer": cancer_of[qid],
                                         "horizon_months": h, "arm": arm, "m": d["m"],
                                         "unstratified": d["unstratified_mean_shift"],
                                         "alive_stratified": d["alive_stratified_mean_shift"],
                                         "stratified_possible":
                                             d["stratified_permutation_possible"]})
            print(f"[joint] H={h:g} {conv}: {len(pop)} patients ({time.time() - t0:.0f}s)")

    def contrast_rows(table: dict, family: str, h: float, conv: str, quantity: str,
                      extra: dict | None = None) -> list[dict]:
        out_rows = []
        qids, _ = paired_matrix(table)
        for cancer in ("all", *F.CANCERS):
            qs = [q for q in qids if cancer == "all" or cancer_of[q] == cancer]
            strata = [cancer_of[q] for q in qs]
            mat = np.asarray([table[q] for q in qs], float).reshape(len(qs), 4)
            for j, arm in enumerate(E.ARMS):
                if len(qs):
                    ci = S.paired_mean_ci(mat[:, j], S.bootstrap_cohorts(
                        strata, family, "arm", h, conv, quantity, cancer, arm))
                else:
                    ci = {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
                out_rows.append({"family": family, "horizon_months": h, "convention": conv,
                                 "quantity": quantity, "cancer": cancer, "row": f"arm:{arm}",
                                 **{k: _r(v) for k, v in ci.items()}, "p_wilcoxon": "",
                                 "below_floor": len(qs) < E.PAIRED_FLOOR, **(extra or {})})
            for name, later, earlier in CONTRASTS:
                d = mat[:, later] - mat[:, earlier] if len(qs) else np.array([])
                sm = summarise_paired(d, strata, family, h, conv, quantity, cancer, name)
                row = {"family": family, "horizon_months": h, "convention": conv,
                       "quantity": quantity, "cancer": cancer, "row": f"contrast:{name}",
                       **{k: _r(v) for k, v in sm.items()},
                       "below_floor": len(qs) < E.PAIRED_FLOOR, **(extra or {})}
                if cancer == "all" and len(qs):
                    ec = equal_cancer_mean(d, strata, family, h, conv, quantity, name)
                    row |= {f"equal_cancer_{k}": _r(v) for k, v in ec.items()}
                out_rows.append(row)
        return out_rows

    for (h, conv, quantity), table in sorted(per_patient.items()):
        if quantity == "fair":
            table = {q: v for q, v in table.items() if all(x is not None for x in v)}
        rows_joint += contrast_rows(table, "joint", h, conv, quantity)
    # Holm: equal-size, primary, pooled, four contrasts x four horizons (16 slots)
    slots = [(h, name) for h in H for name, _, _ in CONTRASTS]
    family_report = {"joint_slots": len(slots),
                     "joint_available": assign_joint_holm(rows_joint, slots)}
    # convention comparison on the common set
    for h in H:
        common = sorted(set(per_patient[(h, "primary", "equal_size")])
                        & set(per_patient[(h, "sensitivity", "equal_size")]))
        for conv in CONV:
            tab = {q: per_patient[(h, conv, "equal_size")][q] for q in common}
            if tab:
                rows_sens += [r | {"analysis": "convention_common_set"}
                              for r in contrast_rows(tab, "joint_common", h, conv,
                                                     "equal_size")
                              if r["cancer"] == "all"]
        composition("joint:both_conventions", common, h)

    # dependence: patient-level mean over arms' fans, summarised per arm and horizon
    dep = pd.DataFrame(rows_dep)
    for (h, arm), g in dep.groupby(["horizon_months", "arm"]):
        for kind in ("unstratified", "alive_stratified"):
            ci = S.paired_mean_ci(g[kind].to_numpy(), S.bootstrap_cohorts(
                g.cancer.tolist(), "dependence", h, arm, kind))
            rows_sens.append({"analysis": f"dependence_{kind}", "horizon_months": h,
                              "convention": "primary", "row": f"arm:{arm}",
                              "n_fans_stratified_possible": int(g.stratified_possible.sum()),
                              **{k: _r(v) for k, v in ci.items()}})

    # ---- first-attempt sensitivity (§9) --------------------------------------------------------
    for h in H:
        pop = sorted(q for q in queries if sets.loc[(q, h, "primary"), "four_arm_intersection"])
        fa_tab, prim_tab = {}, {}
        for qid in pop:
            fans = {arm: [x for x in fan(qid, arm, h, first_only=True)] for arm in E.ARMS}
            if any(len(f) < E.MIN_ACCEPTED or any(x is None or x["vector"] is None for x in f)
                   for f in fans.values()):
                continue
            y = np.asarray(real[(qid, h, "primary")]["coords"]["weighted_vector"], float)
            z = {a: np.asarray([x["vector"] for x in fans[a]], float) for a in E.ARMS}
            m_star = S.four_arm_m_star([len(z[a]) for a in E.ARMS])
            fa_tab[qid] = [S.joint_scores(z[a], y, m_star, "first_attempt", qid, a,
                                          h).equal_size for a in E.ARMS]
            prim_tab[qid] = per_patient[(h, "primary", "equal_size")][qid]
        composition("joint:first_attempt", sorted(fa_tab), h)
        for label, tab in (("first_attempt", fa_tab), ("full_pipeline_same_patients", prim_tab)):
            if tab:
                rows_sens += [r | {"analysis": label} for r in
                              contrast_rows(tab, label, h, "primary", "equal_size")
                              if r["cancer"] == "all"]
    print(f"[sensitivity] first-attempt and convention sets done ({time.time() - t0:.0f}s)")

    # ---- §6 four-arm documented-time diagnostic ---------------------------------------------------
    for h in H:
        for conv in CONV:
            tabs: dict[str, dict] = {k: {} for k in (*LOSSES, "unresolved_fraction")}
            for qid in sorted(queries):
                rr = real[(qid, h, conv)]
                if E.measure(rr["support"]) < E.MIN_WINDOW_MONTHS - S.EPS:
                    continue
                fans = {arm: fan(qid, arm, h) for arm in E.ARMS}
                if any(len(f) < E.MIN_ACCEPTED or any(x is None for x in f)
                       for f in fans.values()):
                    continue
                res = [S.arm_diagnostic(rr["labelled"], [x["labelled"] for x in fans[a]],
                                        rr["support"]) for a in E.ARMS]
                for k in tabs:
                    tabs[k][qid] = [r[k] for r in res]
                rows_course_private.append({"query_id": qid, "cancer": cancer_of[qid],
                                            "horizon_months": h, "convention": conv,
                                            "window_months": _r(res[0]["window_months"]),
                                            **{f"{k}_{a}": _r(r[k]) for a, r in
                                               zip(E.ARMS, res, strict=True) for k in tabs}})
            composition(f"course_four_arm:{conv}", sorted(tabs[PRIMARY_LOSS]), h)
            for k, tab in tabs.items():
                rows_course += contrast_rows(tab, "course_four_arm", h, conv, k)
    print(f"[course] four-arm diagnostic done ({time.time() - t0:.0f}s)")

    # ---- §6 donor fallback (the donor result) -----------------------------------------------------
    donor_p: dict[tuple, float | None] = {}
    for card, (_field, arm) in E.CARDS.items():
        for h in H:
            for conv in CONV:
                res: dict[str, dict] = {}
                n_fallback = 0
                for qid in sorted(queries):
                    rr = real[(qid, h, conv)]
                    f = fan(qid, arm, h)
                    if len(f) < E.MIN_ACCEPTED or any(x is None for x in f):
                        continue
                    mem = donors[(qid, card, h, conv)]
                    if not E.fallback_patient([w for _, w in mem])["eligible"]:
                        continue
                    n_fallback += 1
                    res[qid] = S.donor_diagnostic(rr["labelled"], mem,
                                                  [x["labelled"] for x in f])
                    rows_donor_private.append({"query_id": qid, "cancer": cancer_of[qid],
                                               "card": card, "arm": arm, "horizon_months": h,
                                               "convention": conv,
                                               **{k: _r(v) for k, v in res[qid].items()}})
                qs = sorted(res)
                composition(f"donor_{card}:{conv}", qs, h)
                for loss in LOSSES:
                    for variant in ("", "_secondary"):
                        key = f"{loss}{variant}_llm_minus_donor"
                        keep = [q for q in qs if res[q].get(key) is not None]
                        d = np.asarray([res[q][key] for q in keep], float)
                        sm = summarise_paired(d, [cancer_of[q] for q in keep], "donor", card,
                                              h, conv, key)
                        # component means of the SAME comparison (step-3 gate R1): primary
                        # windows for the primary row, retained secondary pairs for the secondary
                        rows_donor.append({
                            "card": card, "arm": arm, "horizon_months": h, "convention": conv,
                            "quantity": key, "cancer": "all", **{k: _r(v) for k, v in sm.items()},
                            "llm_mean": _r(np.mean([res[q][f"{loss}{variant}_llm"] for q in keep])
                                           if keep else None),
                            "donor_mean": _r(np.mean([res[q][f"{loss}{variant}_donor"]
                                                      for q in keep]) if keep else None),
                            "secondary_pairs_kept": (sum(res[q]["secondary_pairs_kept"]
                                                         for q in keep) if variant else ""),
                            "secondary_pairs": (sum(res[q]["secondary_pairs"] for q in keep)
                                                if variant else ""),
                            "secondary_months_total": (_r(sum(res[q]["secondary_months"]
                                                              for q in keep)) if variant else ""),
                            "n_all_twenty_usable": sum(res[q]["all_twenty_usable"] for q in keep),
                            "scored_months_total": _r(sum(res[q]["scored_months"] for q in keep)),
                            "llm_unresolved_fraction_mean": _r(np.mean(
                                [res[q]["llm_unresolved_fraction"] for q in keep])
                                if keep else None),
                            "below_floor": len(keep) < E.PAIRED_FLOOR})
                        if variant == "" and loss == PRIMARY_LOSS and conv == "primary":
                            donor_p[(card, h)] = sm["p_wilcoxon"]
                    for cancer in F.CANCERS:
                        keep = [q for q in qs if cancer_of[q] == cancer]
                        key = f"{loss}_llm_minus_donor"
                        d = np.asarray([res[q][key] for q in keep], float)
                        sm = summarise_paired(d, [cancer] * len(keep), "donor", card, h, conv,
                                              key, cancer)
                        rows_donor.append({"card": card, "arm": arm, "horizon_months": h,
                                           "convention": conv, "quantity": key,
                                           "cancer": cancer,
                                           **{k: _r(v) for k, v in sm.items()},
                                           "below_floor": len(keep) < E.PAIRED_FLOOR})
    dslots = [(c, h) for c in E.CARDS for h in H]
    family_report |= {"donor_slots": len(dslots),
                      "donor_available": sum(1 for x in dslots if donor_p.get(x) is not None)}
    for (card, h), a in adjust_family(donor_p, dslots).items():
        for r in rows_donor:
            if (r["card"], r["horizon_months"], r["convention"], r["cancer"], r["quantity"]) == \
                    (card, h, "primary", "all", f"{PRIMARY_LOSS}_llm_minus_donor"):
                r["p_holm_donor_family"] = _r(a)
    print(f"[donor] section-6 fallback done ({time.time() - t0:.0f}s)")

    # ---- §8 timing diagnostic ------------------------------------------------------------------
    for h in H:
        pop = sorted(q for q in queries if sets.loc[(q, h, "primary"), "four_arm_intersection"])
        for who in ("real", *E.ARMS):
            items = []
            for qid in pop:
                if who == "real":
                    rr = real[(qid, h, "primary")]
                    c = rr["course"]
                    items.append([S.progression_to_regimen(
                        rr["coords"]["observed_path"],
                        [st.time for st in c.treatment.starts if st.time > 0],
                        min(c.death, h))])
                else:
                    items.append([x["timing"] for x in fan(qid, who, h)])
            frac = lambda k: float(np.mean([np.mean([t[k] for t in it]) for it in items]))  # noqa: E731
            times = [t["months_progression_to_regimen"] for it in items for t in it
                     if t["months_progression_to_regimen"] is not None]
            rows_timing.append({"horizon_months": h, "who": who, "n_patients": len(items),
                                "n_courses": sum(len(it) for it in items),
                                "fraction_progression": _r(frac("progression"), 4),
                                "fraction_both": _r(frac("both"), 4),
                                "fraction_regimen_precedes_any_progression":
                                    _r(frac("regimen_precedes_any_progression"), 4),
                                "n_paired_times": len(times),
                                "median_months_progression_to_regimen":
                                    _r(float(np.median(times)) if times else None, 3)})

    # ---- §5 survival ---------------------------------------------------------------------------------
    cox, cox_check = cox_rebuild(H)
    cox = cox.set_index("query_id")
    qall = sorted(queries)
    t_all = np.array([surv_real[q][0] for q in qall])
    e_all = np.array([surv_real[q][1] for q in qall])
    st_all = np.array([cancer_of[q] for q in qall])
    pos = {q: i for i, q in enumerate(qall)}
    if not np.allclose(cox.loc[qall, "fold"].notna(), True):
        raise RuntimeError("Cox frame does not cover the 450 queries")
    frame = pd.read_csv(COX_FRAME).set_index("query_id").loc[qall]
    if not (np.allclose(frame.time.to_numpy(), t_all) and np.array_equal(frame.event, e_all)):
        raise RuntimeError("saved Cox frame disagrees with the survival read here")
    FORECASTERS = (*E.ARMS, "donor_generic", "donor_matched", "cox_cv")
    rows_surv, rows_surv_private = [], []
    for h in H:
        pred = {f: np.full(len(qall), np.nan) for f in FORECASTERS}
        rmst = {f: np.full(len(qall), np.nan) for f in FORECASTERS}
        msize = np.full(len(qall), np.nan)
        for q in qall:
            i = pos[q]
            for arm in E.ARMS:
                f = fan(q, arm, h)
                f24 = fan(q, arm, 24.0)
                if len(f) >= E.MIN_ACCEPTED and all(x is not None and x["alive"] is not None
                                                    for x in f):
                    pred[arm][i] = float(np.mean([x["alive"] for x in f]))
                    if arm == "patients_like_me":
                        msize[i] = len(f)
                if len(f24) >= E.MIN_ACCEPTED and all(x is not None and x["alive"] is not None
                                                      for x in f24):
                    rmst[arm][i] = float(np.mean([min(x["death"], RMST_H) for x in f24]))
            for card in E.CARDS:
                tt, ee = donor_surv[(q, card)]
                v = S.km_alive_at(tt, ee, h)
                pred[f"donor_{card}"][i] = np.nan if v is None else v
                r24 = km_rmst(tt, ee, RMST_H)
                rmst[f"donor_{card}"][i] = np.nan if r24 is None else r24
            pred["cox_cv"][i] = cox.loc[q, f"p_alive_{h:g}"]
            rmst["cox_cv"][i] = cox.loc[q, "rmst24"]
        fan_m = {arm: np.array([len(fan(q, arm, h)) for q in qall], float) for arm in E.ARMS}
        pop = np.flatnonzero(np.all([~np.isnan(pred[f]) for f in FORECASTERS], axis=0))
        composition("survival_all_forecasters", [qall[i] for i in pop], h)
        w, y, low = S.ipcw_weights(t_all, e_all, st_all, h)
        unsupported = bool(low and (w[pop] > 1 / S.G_FLOOR).any())
        in_pop = np.zeros(len(qall), bool)
        in_pop[pop] = True
        # censoring weights re-estimated inside every resampled cohort, kept for all uses
        resampled = []
        n_boot_unsupported = 0
        for idx in S.bootstrap_cohorts(st_all.tolist(), "survival", h):
            wb, yb, _ = S.ipcw_weights(t_all[idx], e_all[idx], st_all[idx], h)
            n_boot_unsupported += bool((wb[in_pop[idx]] > 1 / S.G_FLOOR).any())
            resampled.append((idx, wb, yb))
        unavailable = {f: int(np.isnan(pred[f]).sum()) for f in FORECASTERS}

        def val(f: str, idx: np.ndarray, wv: np.ndarray, yv: np.ndarray, mask: np.ndarray,
                corrected: bool = False) -> float:
            p = pred[f][idx]
            loss = (p - np.nan_to_num(yv)) ** 2
            if corrected:
                loss = loss - p * (1 - p) / (fan_m[f][idx] - 1)
            loss = np.where(np.isnan(yv), 0.0, loss)
            sel = mask[idx]
            return float((wv[sel] * loss[sel]).sum() / max(sel.sum(), 1))

        def boot_brier(fa: str, fb: str | None, mask: np.ndarray, corrected: bool = False) -> tuple:
            full = np.arange(len(qall))
            est = val(fa, full, w, y, mask, corrected) - (val(fb, full, w, y, mask) if fb else 0.0)
            draws = [val(fa, idx, wb, yb, mask, corrected)
                     - (val(fb, idx, wb, yb, mask) if fb else 0.0) for idx, wb, yb in resampled]
            lo, hi = np.percentile(draws, [2.5, 97.5])
            return est, float(lo), float(hi)

        pairs = [(f"contrast:{n}", E.ARMS[a], E.ARMS[b]) for n, a, b in CONTRASTS]
        pairs += [("vs:generic_cohort-donor_generic", "generic_cohort", "donor_generic"),
                  ("vs:patients_like_me-donor_matched", "patients_like_me", "donor_matched")]
        pairs += [(f"vs:{a}-cox_cv", a, "cox_cv") for a in E.ARMS]
        for cancer in ("all", *F.CANCERS):
            mask = in_pop & ((st_all == cancer) if cancer != "all" else True)
            sub = np.flatnonzero(mask)
            for f in FORECASTERS:
                est, lo, hi = boot_brier(f, None, mask)
                known = sub[~np.isnan(y[sub])]
                e_ = S.ece(pred[f][sub], y[sub], w[sub])
                row = {"horizon_months": h, "cancer": cancer, "row": f"forecaster:{f}",
                       "n": len(sub), "n_evaluable": len(known),
                       "n_unavailable_of_450": unavailable[f], "brier_ipcw": _r(est),
                       "ci_low": _r(lo), "ci_high": _r(hi),
                       "brier_evaluable_raw": _r(float(np.mean((pred[f][known] - y[known]) ** 2))),
                       "mean_predicted_alive": _r(float(pred[f][sub].mean())),
                       "observed_alive_ipcw": _r(float((w[sub] * np.nan_to_num(y[sub])).sum()
                                                       / max(w[sub].sum(), 1e-12))),
                       "ece": _r(e_["ece"]), "ece_n_evaluable": e_["n_evaluable"],
                       "ece_bins": json.dumps(e_["bins"]),
                       "unsupported_weights": unsupported,
                       "bootstrap_cohorts_with_unsupported_weights": n_boot_unsupported,
                       "max_weight": _r(float(w[sub].max())),
                       "ess": _r(float(w[sub].sum() ** 2 / (w[sub] ** 2).sum())),
                       "below_floor": len(sub) < E.PAIRED_FLOOR}
                if f in E.ARMS:
                    est_c, lo_c, hi_c = boot_brier(f, None, mask, corrected=True)
                    row |= {"brier_ipcw_fan_corrected": _r(est_c),
                            "fan_corrected_ci_low": _r(lo_c), "fan_corrected_ci_high": _r(hi_c)}
                rows_surv.append(row)
            for name, fa, fb in pairs:
                est, lo, hi = boot_brier(fa, fb, mask)
                d = np.where(np.isnan(y[sub]), np.nan,
                             (pred[fa][sub] - np.nan_to_num(y[sub])) ** 2
                             - (pred[fb][sub] - np.nan_to_num(y[sub])) ** 2)
                rows_surv.append({"horizon_months": h, "cancer": cancer, "row": name,
                                  "n": len(sub), "brier_ipcw_difference": _r(est),
                                  "ci_low": _r(lo), "ci_high": _r(hi),
                                  "p_wilcoxon_evaluable": _r(S.wilcoxon_p(d[~np.isnan(d)])),
                                  "below_floor": len(sub) < E.PAIRED_FLOOR})
        for i in pop:
            rows_surv_private.append({"query_id": qall[i], "cancer": st_all[i],
                                      "horizon_months": h, "time": _r(t_all[i]),
                                      "event": int(e_all[i]), "weight": _r(w[i]),
                                      **{f"p_{f}": _r(pred[f][i]) for f in FORECASTERS}})
        if h == RMST_H:
            ok = np.all([~np.isnan(rmst[f]) for f in FORECASTERS], axis=0)
            idx_ok = np.flatnonzero(ok)
            cboots = S.bootstrap_cohorts([st_all[i] for i in idx_ok], "cindex")
            cvals = {f: S.c_index(t_all[idx_ok], e_all[idx_ok], rmst[f][idx_ok])
                     for f in FORECASTERS}
            for f in FORECASTERS:
                rows_surv.append({"horizon_months": RMST_H, "cancer": "all", "row": f"cindex:{f}",
                                  "n": int(len(idx_ok)), "c_index": _r(cvals[f])})
            for name, fa, fb in pairs:
                draws = []
                for b in cboots:
                    ii = idx_ok[b]
                    draws.append(S.c_index(t_all[ii], e_all[ii], rmst[fa][ii])
                                 - S.c_index(t_all[ii], e_all[ii], rmst[fb][ii]))
                lo, hi = np.percentile(draws, [2.5, 97.5])
                rows_surv.append({"horizon_months": RMST_H, "cancer": "all",
                                  "row": f"cindex_{name}",
                                  "n": int(len(idx_ok)),
                                  "c_index_difference": _r(cvals[fa] - cvals[fb]),
                                  "ci_low": _r(float(lo)), "ci_high": _r(float(hi))})
        print(f"[survival] H={h:g}: {len(pop)} patients with all forecasters "
              f"({time.time() - t0:.0f}s)")

    # ---- §5 treatment scalars in clinical units -----------------------------------------------------
    rows_tx = []
    for h in H:
        pop, vals = [], {}
        for q in qall:
            rr = real[(q, h, "primary")]["coords"]["raw"]
            if any(rr[k] is None for k in TX_SCALARS):
                continue
            ok, v = True, {"observed": {k: rr[k] for k in TX_SCALARS}}
            for arm in E.ARMS:
                f = fan(q, arm, h)
                if len(f) < E.MIN_ACCEPTED or any(x is None or any(x[k] is None for k in TX_SCALARS)
                                                  for x in f):
                    ok = False
                    break
                v[arm] = {k: float(np.mean([x[k] for x in f])) for k in TX_SCALARS}
            for card in E.CARDS:
                complete = [m for m in donor_tx[(q, card, h)]
                            if all(m[k] is not None for k in TX_SCALARS)]
                if len(complete) < E.MIN_USABLE_DONORS:       # a censored donor gives no total
                    ok = False
                    break
                v[f"donor_{card}"] = {k: float(np.mean([m[k] for m in complete]))
                                      for k in TX_SCALARS}
            if ok:
                pop.append(q)
                vals[q] = v
        composition("treatment_scalars", pop, h)
        strata = [cancer_of[q] for q in pop]
        fc = (*E.ARMS, "donor_generic", "donor_matched")
        for k in TX_SCALARS:
            obs = np.array([vals[q]["observed"][k] for q in pop])
            err = {f: np.array([vals[q][f][k] for q in pop]) - obs for f in fc}
            for f in fc:
                ci = (S.paired_mean_ci(np.abs(err[f]), S.bootstrap_cohorts(strata, "tx", h, k, f))
                      if pop else {"n": 0, "mean": None, "ci_low": None, "ci_high": None})
                rows_tx.append({"horizon_months": h, "quantity": k, "row": f"forecaster:{f}",
                                "n": len(pop), "mean_forecast": _r(float(np.mean(obs + err[f]))
                                                                   if pop else None),
                                "mean_observed": _r(float(obs.mean()) if pop else None),
                                "mean_signed_error": _r(float(err[f].mean()) if pop else None),
                                "mean_abs_error": _r(ci["mean"]), "ci_low": _r(ci["ci_low"]),
                                "ci_high": _r(ci["ci_high"])})
            cmp_ = [(f"contrast:{n}", E.ARMS[a], E.ARMS[b]) for n, a, b in CONTRASTS]
            cmp_ += [("vs:generic_cohort-donor_generic", "generic_cohort", "donor_generic"),
                     ("vs:patients_like_me-donor_matched", "patients_like_me", "donor_matched")]
            for name, fa, fb in cmp_:
                d = np.abs(err[fa]) - np.abs(err[fb]) if pop else np.array([])
                sm = summarise_paired(d, strata, "tx", h, k, name)
                rows_tx.append({"horizon_months": h, "quantity": k, "row": name, "n": len(pop),
                                "abs_error_difference": _r(sm["mean"]), "ci_low": _r(sm["ci_low"]),
                                "ci_high": _r(sm["ci_high"]), "p_wilcoxon": _r(sm["p_wilcoxon"])})
    print(f"[treatment] scalar comparisons done ({time.time() - t0:.0f}s)")

    # ---- coverage -------------------------------------------------------------------------------
    rows_cov = []
    for cancer in F.CANCERS:
        for arm in E.ARMS:
            g = {k[2]: v for k, v in first_status.items() if k[:2] == (cancer, arm)}
            rows_cov.append({"cancer": cancer, "arm": arm, **{k: g.get(k, 0) for k in (
                "requested", "accepted", "first_valid", "first_invalid", "first_unavailable")}})
    for h in H:
        for conv in CONV:
            sub = sets.xs((h, conv), level=("horizon_months", "convention"))
            for cancer in ("all", *F.CANCERS):
                s2 = sub if cancer == "all" else sub[sub.cancer == cancer]
                rows_cov.append({"horizon_months": h, "convention": conv, "cancer": cancer,
                                 "real_target": int(s2.real_target.sum()),
                                 "llm_cells_all_four_arms": int(s2.llm_cells_all_four_arms.sum()),
                                 "four_arm_intersection": int(s2.four_arm_intersection.sum())})

    # ---- write --------------------------------------------------------------------------------
    _write(out / f"joint_scores{SUFFIX}.csv", rows_joint)
    _write(out / f"course_diagnostics{SUFFIX}.csv", rows_course)
    _write(out / f"donor_diagnostics{SUFFIX}.csv", rows_donor)
    _write(out / f"survival{SUFFIX}.csv", rows_surv)
    _write(out / f"sensitivity{SUFFIX}.csv", rows_sens)
    _write(out / f"timing{SUFFIX}.csv", rows_timing)
    _write(out / f"treatment_scalars{SUFFIX}.csv", rows_tx)
    _write(out / f"coverage{SUFFIX}.csv", rows_cov)
    _write(out / f"population_composition{SUFFIX}.csv", rows_comp)
    _write(private / f"joint_patient_rows{SUFFIX}.csv", rows_joint_private)
    _write(private / f"dependence_patient_rows{SUFFIX}.csv", rows_dep)
    _write(private / f"course_patient_rows{SUFFIX}.csv", rows_course_private)
    _write(private / f"donor_patient_rows{SUFFIX}.csv", rows_donor_private)
    _write(private / f"survival_patient_rows{SUFFIX}.csv", rows_surv_private)
    cox.reset_index().to_csv(private / f"cox_cv_forecasts_with_folds{SUFFIX}.csv", index=False)
    # provenance: every project module actually loaded, every frozen input, kept apart from
    # the generation commit (step-3 gate R4)
    loaded = sorted({Path(m.__file__).resolve() for m in list(sys.modules.values())
                     if getattr(m, "__file__", None)
                     and str(Path(m.__file__).resolve()).startswith(str(ROOT))
                     and ".venv" not in str(m.__file__)})
    code_files = loaded + [ROOT / "scripts" / "four_arm_drug_classes_v1.csv"]
    inputs = [F.RUN / "attempts.jsonl", F.RUN / "manifest.json", F.UNIVERSE / "queries.jsonl",
              F.UNIVERSE / "manifest.json", F.PROMPTS / "quartets.jsonl",
              F.PROMPTS / "provenance.json", private / f"query_sets{F.SUFFIX}.csv",
              out / f"manifest{F.SUFFIX}.json", COX_FRAME, COX_SAVED]
    genie_inputs = {f"{c}/{f}": F.sha256_file(GV2.GENIE_ROOT / GV2.COHORTS[c] / f)
                    for c in F.CANCERS for f in F.GENIE_FILES}
    dirty = F.git("status", "--porcelain", "--", *[str(x.relative_to(ROOT)) for x in code_files])
    manifest = {
        "what": "four-arm gate-4 cohort extraction (step 3, corrected after Codex review)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_commit": F.git("rev-parse", "HEAD"),
        "analysis_code_uncommitted": bool(dirty),
        "generation_commit": F.GENERATION_COMMIT,
        "specification": {"spec": F.sha256_file(F.SPEC), "amendment_v2": F.sha256_file(F.AMENDMENT_V2)},
        "code_sha256": {str(x.relative_to(ROOT)): F.sha256_file(x) for x in code_files},
        "input_sha256": {str(x.relative_to(ROOT)): F.sha256_file(x) for x in inputs},
        "genie_input_sha256": genie_inputs,
        "input_checks": checks,
        "eligibility_check": {"rows_compared": len(saved_incl), "saved_only": 0,
                              "derived_only": 0},
        "adjustment_families": family_report,
        "primary_questions": {"joint": "equal_size, primary convention, pooled; Holm over 4 "
                                       "contrasts x 4 horizons",
                              "donor": f"{PRIMARY_LOSS}_llm_minus_donor, primary convention, "
                                       "pooled; Holm over 2 cards x 4 horizons"},
        "parameters": {"seed": S.SEED, "subsets": S.N_SUBSETS, "permutations": S.N_PERMUTATIONS,
                       "bootstrap_cohorts": S.N_BOOTSTRAP, "g_floor": S.G_FLOOR,
                       "min_accepted": E.MIN_ACCEPTED, "paired_floor": E.PAIRED_FLOOR,
                       "min_usable_donors": E.MIN_USABLE_DONORS,
                       "min_window_months": E.MIN_WINDOW_MONTHS},
        "precision": "full floating-point precision in every table; rounding only for display",
        "cox_rebuild_check": cox_check,
        "outputs": sorted(str(x.relative_to(ROOT)) for x in out.glob(f"*{SUFFIX}.csv")),
        "runtime_seconds": round(time.time() - t0, 1),
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__}
    (out / f"manifest{SUFFIX}.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    write_reconciliation(out, rows_surv)
    print(f"[done] aggregate tables and manifest in {out.relative_to(ROOT)} "
          f"({time.time() - t0:.0f}s)")
    return 0


def write_reconciliation(out: Path, rows_surv: list[dict]) -> None:
    """RECONCILIATION note (aggregate only): what this extraction supersedes and why, with
    the pooled survival numbers beside the earlier exploratory ones."""
    old = json.loads((ROOT / "outputs" / "exploratory_four_arm_deep" / "baselines.json").read_text())
    lines = [
        "# Four-arm results: reconciliation with earlier numbers",
        "",
        f"Written by `four_arm_extract_v1.py` ({SUFFIX[1:]}). Aggregate values only.",
        "",
        "## What is superseded",
        "",
        "- `outputs/exploratory_four_arm_deep/` (deep extract, baselines, distance battery,",
        "  chronology sensitivity, floors/reliability): exploratory, built with the generated",
        "  reader that restated status and presence (amendment 12 October, R1) and with treatment",
        "  from regimen lines rather than agents (R2-R5). Its disease-course and treatment numbers",
        "  are superseded by `course_diagnostics`, `donor_diagnostics`, `joint_scores` and",
        "  `treatment_scalars` here. Do not carry old-reader course or treatment values forward.",
        "- Survival probabilities do not depend on the disease reader. The Cox model is re-fitted",
        "  with the same folds and reproduces the saved forecasts (manifest, `cox_rebuild_check`).",
        "  Differences below come from the comparison population (all seven forecasters",
        "  available) and from cancer-stratified censoring weights with left limits.",
        "",
        "## Pooled survival Brier score, old (exploratory) against new",
        "",
        "| H | Forecaster | Old IPCW Brier | New IPCW Brier | New n |",
        "|---|---|---|---|---|",
    ]
    for h in ("6", "12", "24"):
        block = old["brier_at_horizon"].get(h, {})
        for f, v in block.items():
            if not isinstance(v, dict) or "brier_ipcw_all" not in v:
                continue
            new = [r for r in rows_surv if r.get("cancer") == "all" and r["horizon_months"] == float(h)
                   and r["row"] == f"forecaster:{f}"]
            nv = f"{new[0]['brier_ipcw']:.4f}" if new and new[0]["brier_ipcw"] != "" else "n/a"
            nn = new[0]["n"] if new else ""
            lines.append(f"| {h} | {f} | {v['brier_ipcw_all'][0]:.4f} | {nv} | {nn} |")
    lines += ["", "Values for the manuscript come only from the new tables, after independent review."]
    (out / f"RECONCILIATION{SUFFIX}.md").write_text("\n".join(lines) + "\n")
