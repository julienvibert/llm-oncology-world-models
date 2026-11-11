#!/usr/bin/env python3
"""Is there room for an anchored predictor between "matching works" and "no data"?

The anchored-LLM proposal retrieves real neighbours around a query patient and asks the
model to transport their observed outcome contrast onto that patient. Before spending
anything on generation, two facts decide whether the method can exist at all:

  1. THE OPERATING WINDOW.  Where close neighbours exist, plain matching already answers the
     question. Where the space is genuinely empty, no anchor can be retrieved and the LLM has
     nothing to be anchored to. The method can only earn its keep in the band between --
     neighbours present but far enough that matching degrades. This measures whether that
     band exists, how wide it is, and how many real decisions fall inside it.

  2. ANCHOR AVAILABILITY.  The clean construction needs TWO real retrievals near the query --
     neighbours who received A1 and neighbours who received A2 -- so their observed
     difference is a measured contrast rather than an assumed one. This counts how often such
     a paired local cohort actually exists in GENIE.

VOCABULARY. In GENIE, treatment is not randomised, so the difference between two local
treatment groups is a LOCAL OUTCOME CONTRAST, not a local average treatment effect. It is
confounded by indication and everything else that drove the choice. The word ATE is reserved
for the randomised setting (PRIME). Nothing here estimates a causal quantity; this measures
what is retrievable, which is a question about support, not about effects.

UNIT. One row per treatment-line start: the decision point at which (context, state,
history) are known and an action is chosen. Endpoint is time to next treatment or death
(TTNT), the standard real-world readout for a line decision, binarised at a fixed horizon so
that censoring is handled by exclusion rather than by an assumption -- a patient whose status
at the horizon is unknown is dropped from both numerator and denominator.

WHAT THIS IS NOT. No causal claim, no LLM, no generation, and no prediction method being
advocated. k-nearest-neighbour here is a probe for how much local information exists at a
given radius, not a proposed clinical predictor.

GOVERNANCE: GENIE-BPC is under a data use agreement. Aggregate curves and counts only,
written to a gitignored directory. No patient-level rows are written anywhere.

Run:  uv run python scripts/explore/measure_anchor_operating_window.py --self-test
      uv run python scripts/explore/measure_anchor_operating_window.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

GENIE_ROOT = Path.home() / "GenieBPC"
COHORTS = ("NSCLC_Genie", "BRCA_GENIE", "CRC_GENIE")
GENIE = GENIE_ROOT / "NSCLC_Genie"          # kept: the biomarker loader is NSCLC-specific
OUT = ROOT / "outputs" / "exploratory_anchor_operating_window"

HORIZON_MONTHS = 6.0      # binarisation point for TTNT
MIN_ANCHORS = 5           # neighbours needed on an arm before a contrast is computable
K_NEIGHBOURS = 10
SEED = 20260810

# Action classes. Deliberately coarse and shared across cancers, because the question is
# whether a comparable group exists, and splitting by agent would answer "no" for arithmetic
# reasons rather than clinical ones. Ordered so the most specific rule wins.
DRUG_CLASSES = (
    ("investigational", ("investigational",)),
    ("targeted TKI", ("tinib", "ciclib", "parib", "rafenib", "afenib")),
    ("anti-HER2", ("trastuzumab", "pertuzumab", "ado-trastuzumab", "lapatinib", "neratinib")),
    ("anti-EGFR mAb", ("cetuximab", "panitumumab")),
    ("anti-VEGF", ("bevacizumab", "ramucirumab", "ziv-aflibercept", "regorafenib")),
    ("immunotherapy", ("nivolumab", "pembrolizumab", "atezolizumab", "durvalumab",
                       "ipilimumab", "avelumab")),
    ("endocrine", ("tamoxifen", "anastrozole", "letrozole", "exemestane", "fulvestrant",
                   "leuprolide", "goserelin")),
    ("platinum", ("carboplatin", "cisplatin", "oxaliplatin")),
    ("anthracycline", ("doxorubicin", "epirubicin")),
    ("taxane", ("paclitaxel", "docetaxel", "cabazitaxel")),
    ("antimetabolite", ("fluorouracil", "capecitabine", "gemcitabine", "pemetrexed",
                        "methotrexate")),
    ("topoisomerase", ("irinotecan", "etoposide", "topotecan")))


def action_class(drugs: str) -> str:
    """First matching class wins; a combination is named by its most disease-defining part."""
    d = str(drugs).lower()
    for name, keys in DRUG_CLASSES:
        if any(k in d for k in keys):
            return name
    return "other systemic"


# cancer joins CONTEXT: pooling the cohorts is the only way C carries disease at all
CONTEXT = ["era", "age_at_start", "female", "institution", "cancer"]
# thin X: what a registry row alone gives you. rich X adds everything the release records
# about the patient's actual condition at the decision — the difference between the two is
# the point, because a weak representation would make any predictor look bad and any
# anchoring story look good for the wrong reason.
STATE_THIN = ["stage_iv", "dmets", "nonsquamous"]
# `imaging_observed` and `md_note_observed` are MISSINGNESS INDICATORS, and they are part of
# the state rather than bookkeeping. Without them "no scan was ever done" and "a scan was
# done and showed no progression" are the same row of zeros, which turns absent data into a
# reassuring clinical fact -- in the distance metric, in the anchor table, and in every
# baseline fitted on this matrix. 3.5% of landmarks have no prior imaging at all and were
# being rendered as `last_scan_progressing = 0`.
STATE_RICH = [*STATE_THIN, "n_disease_sites", "brain_met", "liver_met", "bone_met",
              "last_scan_progressing", "last_scan_responding", "months_since_last_scan",
              "md_status_progressing", "md_status_responding",
              "imaging_observed", "md_note_observed"]

#: Which channels are only meaningful when their indicator is 1. Consumed by the anchored
#: renderer ("not assessed" rather than a number) and by the corpus metric.
OBSERVATION_FLAGS: dict[str, tuple[str, ...]] = {
    "imaging_observed": ("n_disease_sites", "brain_met", "liver_met", "bone_met",
                         "last_scan_progressing", "last_scan_responding",
                         "months_since_last_scan", "n_prior_scans",
                         "n_prior_progressions"),
    "md_note_observed": ("md_status_progressing", "md_status_responding",
                         "n_prior_visits", "visit_rate"),
}
HISTORY_THIN = ["line_number", "months_since_dx", "prior_lines"]
HISTORY_RICH = [*HISTORY_THIN, "n_prior_scans", "n_prior_progressions", "n_prior_visits",
                "visit_rate", "months_on_prior_line"]
# GENIE-BPC v2.0 public records no ECOG and no laboratory values, so performance status and
# organ function are absent from BOTH representations. Physician-assessed disease status and
# radiological assessment are the closest available stand-ins and are what "rich" adds.
ABSENT_FROM_RELEASE = ("ECOG performance status", "laboratory values", "weight / weight loss")


def _site_flags(sites: pd.Series) -> pd.Series:
    s = sites.astype(str).str.lower()
    return pd.Series({"brain": s.str.contains("brain").any(),
                      "liver": s.str.contains("liver").any(),
                      "bone": s.str.contains("vertebr|bone").any()})


def prior_state(genie: Path, prefix: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Imaging and med-onc notes, tidied to (record_id, months, ...) for as-of joins.

    `prefix` namespaces the ids to match the decision table. record_id is unique only WITHIN
    a cohort, so the decision table prefixes it with the cancer; if these tables are not
    prefixed the same way every lookup misses and every feature silently defaults to zero,
    which looks exactly like a cohort with no imaging rather than like a broken join.
    """
    im = pd.read_csv(genie / "imaging_level_dataset.csv", low_memory=False)
    im = im.rename(columns={"dx_scan_mos": "months"}).dropna(subset=["months"])
    im["record_id"] = prefix + im["record_id"].astype(str)
    sitecols = [c for c in im.columns if c.startswith("image_casite")]
    im["n_sites"] = im[sitecols].notna().sum(axis=1)
    im["sites_txt"] = im[sitecols].astype(str).agg(" ".join, axis=1)
    ov = im["image_overall"].astype(str)
    im["progressing"] = ov.str.startswith("Progressing").astype(float)
    im["responding"] = ov.str.startswith("Improving").astype(float)

    mo = pd.read_csv(genie / "med_onc_note_level_dataset.csv", low_memory=False)
    mo = mo.rename(columns={"dx_md_visit_mos": "months"}).dropna(subset=["months"])
    mo["record_id"] = prefix + mo["record_id"].astype(str)
    st = mo["md_ca_status"].astype(str)
    mo["md_progressing"] = st.str.startswith("Progressing").astype(float)
    mo["md_responding"] = st.str.startswith("Improving").astype(float)
    return (im[["record_id", "months", "n_sites", "sites_txt", "progressing", "responding"]]
            .sort_values(["record_id", "months"]),
            mo[["record_id", "months", "md_progressing", "md_responding"]]
            .sort_values(["record_id", "months"]))


def attach_rich_state(d: pd.DataFrame, genie: Path, prefix: str = "") -> pd.DataFrame:
    """Everything known about the patient's condition STRICTLY BEFORE the line starts.

    Every feature is computed from records dated earlier than the decision. A scan performed
    on the start date is excluded: in practice it is often the restaging that motivated the
    switch, and its result is downstream of information the predictor is not entitled to.
    """
    im, mo = prior_state(genie, prefix)
    if not set(d["record_id"]) & set(im["record_id"]):
        raise ValueError(
            "no decision id matches any imaging id — the namespacing of record_id and of "
            "the imaging tables has diverged, and every rich-state feature would silently "
            "default to zero")
    im_by = {k: v for k, v in im.groupby("record_id")}
    mo_by = {k: v for k, v in mo.groupby("record_id")}
    rows = []
    for rec, t0 in zip(d["record_id"], d["months_since_dx"], strict=True):
        g = im_by.get(rec)
        g = g[g["months"] < t0] if g is not None else None
        if g is None or g.empty:
            # the values stay 0 so the matrix keeps its shape; `imaging_observed` is what
            # makes them readable as "nothing was assessed" rather than "nothing was found"
            r = {"n_disease_sites": 0.0, "brain_met": 0.0, "liver_met": 0.0, "bone_met": 0.0,
                 "last_scan_progressing": 0.0, "last_scan_responding": 0.0,
                 "months_since_last_scan": float(t0), "n_prior_scans": 0.0,
                 "n_prior_progressions": 0.0, "imaging_observed": 0.0}
        else:
            last = g.iloc[-1]
            flags = _site_flags(g["sites_txt"])
            r = {"n_disease_sites": float(last["n_sites"]),
                 "brain_met": float(flags["brain"]), "liver_met": float(flags["liver"]),
                 "bone_met": float(flags["bone"]),
                 "last_scan_progressing": float(last["progressing"]),
                 "last_scan_responding": float(last["responding"]),
                 "months_since_last_scan": float(t0 - last["months"]),
                 "n_prior_scans": float(len(g)),
                 "n_prior_progressions": float(g["progressing"].sum()),
                 "imaging_observed": 1.0}
        m = mo_by.get(rec)
        m = m[m["months"] < t0] if m is not None else None
        if m is None or m.empty:
            r |= {"md_status_progressing": 0.0, "md_status_responding": 0.0,
                  "n_prior_visits": 0.0, "visit_rate": 0.0, "md_note_observed": 0.0}
        else:
            r |= {"md_status_progressing": float(m.iloc[-1]["md_progressing"]),
                  "md_status_responding": float(m.iloc[-1]["md_responding"]),
                  "n_prior_visits": float(len(m)),
                  "visit_rate": float(len(m) / max(t0, 1.0)), "md_note_observed": 1.0}
        rows.append(r)
    return pd.concat([d.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def load_decisions(cohorts: tuple[str, ...] = COHORTS) -> pd.DataFrame:
    """One row per line start, pooled over cohorts, with the endpoint attached."""
    return pd.concat([_load_cohort(c) for c in cohorts], ignore_index=True)


def _load_cohort(cohort: str) -> pd.DataFrame:
    genie = GENIE_ROOT / cohort
    reg = pd.read_csv(genie / "regimen_cancer_level_dataset.csv", low_memory=False)
    idx = pd.read_csv(genie / "cancer_level_dataset_index.csv", low_memory=False)
    pat = pd.read_csv(genie / "patient_level_dataset.csv", low_memory=False)

    d = reg.merge(idx, on=["cohort", "record_id", "ca_seq"], how="left", suffixes=("", "_idx"))
    d = d.merge(pat[["record_id", "naaccr_sex_code", "birth_year"]], on="record_id", how="left")

    d["months_since_dx"] = pd.to_numeric(d["dx_reg_start_int_mos"], errors="coerce")
    d["line_number"] = pd.to_numeric(d["regimen_number_within_cancer"], errors="coerce")
    d["prior_lines"] = (d["line_number"] - 1).clip(lower=0)
    d["age_at_start"] = (pd.to_numeric(d["age_dx"], errors="coerce")
                         + d["months_since_dx"] / 12.0)
    # era from year of diagnosis plus elapsed months -- GENIE shifts exact dates, so this is
    # a coarse calendar position, which is all the context axis needs
    d["era"] = (pd.to_numeric(d["birth_year"], errors="coerce")
                + pd.to_numeric(d["dob_ca_dx_yrs"], errors="coerce")
                + d["months_since_dx"] / 12.0)
    # the release stores a LABEL here, not the NAACCR numeric code; matching on "2"
    # silently produced an all-zero column that was invisible until Cox refused to invert
    d["female"] = (d["naaccr_sex_code"].astype(str).str.strip().str.lower()
                   .eq("female").astype(float))
    d["stage_iv"] = d["stage_dx"].astype(str).str.contains("IV", na=False).astype(float)
    d["dmets"] = d["ca_dmets_yn"].astype(str).str.lower().str.startswith("yes").astype(float)
    d["nonsquamous"] = (~d["ca_hist_adeno_squamous"].astype(str).str.lower()
                        .str.contains("squamous", na=False)).astype(float)
    d["action"] = d["regimen_drugs"].map(action_class)
    d["cancer"] = cohort.split("_")[0]
    # record_id is only unique WITHIN a cohort; patient-grouped folds would leak
    # across diseases without this
    d["record_id"] = d["cancer"] + "::" + d["record_id"].astype(str)

    # Genomic drivers are cancer-specific (EGFR/ALK in lung, RAS/BRAF in colon, HER2 in
    # breast) and only the lung loader exists, so they are NOT part of the shared state.
    # Adding them for one cohort and zeroing them for the others would encode "this is lung"
    # three times over and call it biology.

    # endpoint: reached next treatment or death within the horizon. Unknown status at the
    # horizon (censored earlier) is dropped -- never imputed as "no event".
    t = pd.to_numeric(d["ttnt_ca_seq_mos"], errors="coerce")
    ev = pd.to_numeric(d["ttnt_ca_seq_status"], errors="coerce")
    d["y"] = np.where((ev == 1) & (t <= HORIZON_MONTHS), 1.0,
                      np.where(t >= HORIZON_MONTHS, 0.0, np.nan))
    d["ttnt_mos"], d["ttnt_event"] = t, ev
    # PFS from this line start, carried through for the distributional benchmark
    # imaging-confirmed PFS, from this regimen's start. The "i_or_m" composite also
    # counts a med-onc note as progression, which is more sensitive and much noisier;
    # imaging-only is the definition a trial would recognise.
    d["pfs_mos"] = pd.to_numeric(d["tt_pfs_i_g_mos"], errors="coerce")
    d["pfs_event"] = pd.to_numeric(d["pfs_i_g_status"], errors="coerce")
    d = d.sort_values(["record_id", "months_since_dx"])
    d["months_on_prior_line"] = (d.groupby("record_id")["months_since_dx"].diff()
                                 .fillna(0.0).clip(lower=0.0))
    keep = ["record_id", "cancer", "action", "y", "ttnt_mos", "ttnt_event", "pfs_mos",
            "pfs_event", *CONTEXT,
            *STATE_THIN, *HISTORY_THIN, "months_on_prior_line"]
    d = d[list(dict.fromkeys(keep))].dropna().reset_index(drop=True)
    d["institution"] = d["institution"].astype(str)
    d["cancer"] = d["cancer"].astype(str)
    d = attach_rich_state(d, genie, prefix=d["cancer"].iloc[0] + "::")
    return d


def feature_matrix(d: pd.DataFrame, cols: list[str] | None = None, *,
                   drop_first: bool = False) -> np.ndarray:
    """Standardised (C, X, H) — the space distances are measured in. Action is NOT in it.

    Action is the thing being varied, so putting it in the metric would make patients on
    different arms far apart by construction and destroy the paired retrieval the whole
    method rests on.
    """
    cols = cols or [*CONTEXT, *STATE_RICH, *HISTORY_RICH]
    num = d[[c for c in cols if c not in ("institution", "cancer")]].astype(float)
    # a constant column carries no distance and makes a regression's information matrix
    # singular, so it is dropped rather than divided by a substituted 1.0
    num = num.loc[:, num.std(ddof=0) > 1e-12]
    num = (num - num.mean()) / num.std(ddof=0)
    # complete one-hot for the DISTANCE metric (every institution must be equally far
    # from every other); drop_first for MODEL matrices, where the redundant column is
    # perfectly collinear and makes Cox's information matrix singular
    cats = [c for c in ("institution", "cancer") if c in cols]
    inst = (pd.get_dummies(d[cats], prefix=cats, drop_first=drop_first)
            .astype(float) if cats else pd.DataFrame(index=d.index))
    # one-hot columns are scaled so a whole institution switch costs about one SD of a
    # numeric channel, rather than dominating or vanishing
    inst = inst / np.sqrt(2.0)
    return np.hstack([num.to_numpy(float), inst.to_numpy(float)])


def knn_curve(x: np.ndarray, y: np.ndarray, action: np.ndarray, patient: np.ndarray,
              k: int) -> pd.DataFrame:
    """Leave-one-patient-out k-NN within the same action, keyed by neighbour distance.

    Excluding the query patient's OTHER lines matters: the same person appears several times,
    and letting their own later line vote would make matching look far better than it is.
    """
    rows = []
    for i in range(len(x)):
        same = (action == action[i]) & (patient != patient[i])
        if same.sum() < k:
            continue
        dist = np.linalg.norm(x[same] - x[i], axis=1)
        order = np.argsort(dist)[:k]
        rows.append({"d_k": float(dist[order].max()), "d_1": float(dist[order].min()),
                     "pred": float(y[same][order].mean()), "y": float(y[i]),
                     "action": action[i]})
    return pd.DataFrame(rows)


def anchor_availability(x: np.ndarray, action: np.ndarray, patient: np.ndarray,
                        radii: list[float], min_anchors: int) -> pd.DataFrame:
    """How often a query has >=min_anchors real neighbours on its own arm AND on another."""
    rows = []
    for r in radii:
        own, paired, best_alt = 0, 0, []
        for i in range(len(x)):
            other = patient != patient[i]
            near = other & (np.linalg.norm(x - x[i], axis=1) <= r)
            n_own = int((near & (action == action[i])).sum())
            alts = pd.Series(action[near & (action != action[i])]).value_counts()
            n_alt = int(alts.iloc[0]) if len(alts) else 0
            own += n_own >= min_anchors
            paired += (n_own >= min_anchors) and (n_alt >= min_anchors)
            best_alt.append(n_alt)
        rows.append({"radius": r, "own_arm_ok": own / len(x), "paired_ok": paired / len(x),
                     "median_best_alt_arm": float(np.median(best_alt))})
    return pd.DataFrame(rows)


def baseline_ladder(d: pd.DataFrame, x_model: np.ndarray, y: np.ndarray,
                    patient: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """The bar an anchored predictor has to clear, cheapest first.

    Grouped by patient, never by row: several lines belong to one person, and a random split
    would leak that person across the fold boundary and flatter every model. Skill is versus
    always predicting the base rate, so a model at 0 has learned nothing the marginal event
    rate does not already say.

    Cox is fitted on the real censored TTNT and then read out as 6-month risk, so it competes
    on exactly the same binary target as the rest rather than on a metric of its own.
    """
    from lifelines import CoxPHFitter
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold

    a = pd.get_dummies(d["action"], prefix="act",
                       drop_first=True).astype(float).to_numpy()
    hist = feature_matrix(d, HISTORY_RICH, drop_first=True)
    xa = np.hstack([x_model, a])
    null = float(((y.mean() - y) ** 2).mean())
    folds = list(GroupKFold(n_splits=5).split(xa, y, groups=patient))

    def run(make, feats):
        p = np.zeros(len(y))
        for tr, te in folds:
            p[te] = make().fit(feats[tr], y[tr]).predict_proba(feats[te])[:, 1]
        return p

    def run_cox():
        """6-month risk from a Cox fit on the censored endpoint."""
        cols = [f"f{i}" for i in range(xa.shape[1])]
        frame = pd.DataFrame(xa, columns=cols)
        frame["T"] = np.clip(d["ttnt_mos"].to_numpy(float), 1e-3, None)
        frame["E"] = d["ttnt_event"].to_numpy(float)
        p = np.zeros(len(y))
        for tr, te in folds:
            cph = CoxPHFitter(penalizer=0.5).fit(frame.iloc[tr], "T", "E")
            sf = cph.predict_survival_function(frame.iloc[te][cols],
                                               times=[HORIZON_MONTHS])
            p[te] = 1.0 - sf.to_numpy().ravel()
        return p

    specs = [("action only", lambda: run(lambda: LogisticRegression(max_iter=2000), a)),
             ("history only", lambda: run(lambda: LogisticRegression(max_iter=2000), hist)),
             ("logistic regression", lambda: run(lambda: LogisticRegression(max_iter=4000),
                                                 xa)),
             ("ridge logistic (CV)",
              lambda: run(lambda: LogisticRegressionCV(Cs=8, max_iter=4000, penalty="l2",
                                                       scoring="neg_brier_score"), xa)),
             ("lasso logistic (CV)",
              lambda: run(lambda: LogisticRegressionCV(Cs=8, max_iter=4000, penalty="l1",
                                                       solver="saga",
                                                       scoring="neg_brier_score"), xa)),
             ("random forest",
              lambda: run(lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=8,
                                                         random_state=SEED, n_jobs=-1), xa)),
             ("gradient boosting",
              lambda: run(lambda: HistGradientBoostingClassifier(random_state=SEED,
                                                                 max_iter=200,
                                                                 learning_rate=0.05,
                                                                 min_samples_leaf=30), xa)),
             ("Cox -> 6-month risk", run_cox)]
    rows, preds = [], {}
    for name, fn in specs:
        p = fn()
        preds[name] = p
        brier = float(brier_score_loss(y, p))
        rows.append({"model": name, "auc": float(roc_auc_score(y, p)), "brier": brier,
                     "skill": 1 - brier / null})
    return pd.DataFrame(rows), preds


def skill_by_support(x: np.ndarray, y: np.ndarray, action: np.ndarray, patient: np.ndarray,
                     pred: np.ndarray, radius: float) -> pd.DataFrame:
    """Does the fitted model ALSO degrade where the neighbourhood is empty?

    This is the question anchoring lives or dies on. If a conventional learner is equally
    good in sparse regions, retrieval has nothing to add there. If it falls away exactly
    where neighbours run out, that region is the target -- though it is also, by definition,
    where anchors are hardest to retrieve.
    """
    n_near = np.array([int(((patient != patient[i])
                            & (action == action[i])
                            & (np.linalg.norm(x - x[i], axis=1) <= radius)).sum())
                       for i in range(len(x))])
    bins = [(0, 0), (1, 4), (5, 19), (20, 10**9)]
    base = float(y.mean())
    rows = []
    for lo, hi in bins:
        m = (n_near >= lo) & (n_near <= hi)
        if m.sum() < 30:
            continue
        null = float(((base - y[m]) ** 2).mean())
        brier = float(((pred[m] - y[m]) ** 2).mean())
        rows.append({"support": f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9
                                                          else f"{lo}+"),
                     "n": int(m.sum()), "event_rate": float(y[m].mean()),
                     "brier": brier, "skill": 1 - brier / null if null > 0 else float("nan")})
    return pd.DataFrame(rows)


def self_test() -> int:
    # THE TAXONOMY CONTRACT, frozen 2026-08-20: twelve classes plus the "other systemic"
    # fallback, first match wins in DRUG_CLASSES order, so a combination is named by its
    # most disease-defining component. These assertions previously named three categories
    # ("platinum doublet", "IO alone", "chemo-IO") from a superseded taxonomy and FAILED --
    # while the entire corpus, the anchors, and now the class-typed trajectory marks were
    # built on the current one. A taxonomy whose own test asserts a different taxonomy is
    # not frozen. Ordering consequences are asserted explicitly: chemo-IO combinations
    # classify as immunotherapy (IO precedes platinum), platinum doublets as platinum
    # (platinum precedes antimetabolite).
    assert action_class("Carboplatin, Pemetrexed Disodium") == "platinum"
    assert action_class("Nivolumab") == "immunotherapy"
    assert action_class("Carboplatin, Pembrolizumab") == "immunotherapy"
    assert action_class("Osimertinib") == "targeted TKI"
    assert action_class("Investigational Drug") == "investigational"
    assert action_class("Fluorouracil, Oxaliplatin, Leucovorin") == "platinum"   # FOLFOX
    # FOLFIRI classifies as antimetabolite, NOT topoisomerase: fluorouracil matches the
    # antimetabolite rule, which precedes topoisomerase in DRUG_CLASSES order. FOLFOX and
    # FOLFIRI therefore land in DIFFERENT classes (platinum vs antimetabolite), which is
    # what lets the class-typed trajectory marks separate them.
    assert action_class("Fluorouracil, Irinotecan, Leucovorin") == "antimetabolite"
    assert action_class("Trastuzumab, Paclitaxel") == "anti-HER2"
    assert action_class("Letrozole") == "endocrine"
    assert action_class("Unknown Agent XYZ") == "other systemic"

    # k-NN never lets a patient's own other lines vote for them
    x = np.array([[0.0], [0.01], [5.0], [5.01]])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    act = np.array(["a"] * 4)
    pid = np.array(["p", "p", "q", "q"])          # each patient contributes two lines
    out = knn_curve(x, y, act, pid, k=1)
    assert len(out) == 4
    # the nearest ALLOWED neighbour of p's rows is a q row, so the prediction must be 0
    assert set(out.loc[out["y"] == 1.0, "pred"]) == {0.0}

    # availability: with a tiny radius nothing qualifies; with a huge one everything does
    av = anchor_availability(np.array([[0.0], [0.1], [0.2], [0.3]]),
                             np.array(["a", "a", "b", "b"]),
                             np.array(["p", "q", "r", "s"]),
                             radii=[0.001, 10.0], min_anchors=1)
    assert av.iloc[0]["paired_ok"] == 0.0 and av.iloc[1]["paired_ok"] == 1.0
    print("self-test OK — action classes, leave-one-patient-out k-NN, availability bounds")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=("rich", "thin"), default="rich",
                    help="rich adds imaging burden and physician-assessed status")
    ap.add_argument("--k", type=int, default=K_NEIGHBOURS)
    ap.add_argument("--min-anchors", type=int, default=MIN_ANCHORS)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    d = load_decisions()
    fcols = ([*CONTEXT, *STATE_RICH, *HISTORY_RICH] if args.features == "rich"
             else [*CONTEXT, *STATE_THIN, *HISTORY_THIN])
    x = feature_matrix(d, fcols)                      # distances
    x_model = feature_matrix(d, fcols, drop_first=True)   # regressions
    y = d["y"].to_numpy(float)
    act = d["action"].to_numpy()
    pid = d["record_id"].to_numpy()
    base = float(y.mean())
    print(f"GENIE-BPC NSCLC: {len(d)} line-start decisions from {d['record_id'].nunique()} "
          f"patients, {x.shape[1]} covariates ({args.features} X)")
    print(f"endpoint: next treatment or death within {HORIZON_MONTHS:.0f} months — "
          f"{base:.1%} event rate\n")
    print("actions:", d["action"].value_counts().to_dict(), "\n")

    # ---- 1. the operating window ---------------------------------------------------------
    cur = knn_curve(x, y, act, pid, args.k)
    q = np.quantile(cur["d_k"], np.linspace(0, 1, 6))
    cur["bin"] = pd.cut(cur["d_k"], bins=q, include_lowest=True, labels=False)
    print(f"1. OPERATING WINDOW — {args.k}-NN within the same action, leave-one-patient-out")
    print(f"   Brier skill is versus always predicting the base rate {base:.1%}; "
          "0 means matching adds nothing here.\n")
    print(f"   {'quintile of neighbour distance':32s} {'n':>6} {'d_k':>10} {'Brier':>8} "
          f"{'skill':>8}")
    print("   " + "-" * 68)
    win = []
    b0 = float(((base - cur["y"]) ** 2).mean())
    for b in range(5):
        g = cur[cur["bin"] == b]
        if g.empty:
            continue
        brier = float(((g["pred"] - g["y"]) ** 2).mean())
        null = float(((base - g["y"]) ** 2).mean())
        skill = 1 - brier / null if null > 0 else float("nan")
        win.append({"quintile": b + 1, "n": len(g), "d_k_mid": float(g["d_k"].median()),
                    "brier": brier, "skill": skill})
        print(f"   {'Q' + str(b + 1) + ' (closest)' if b == 0 else 'Q' + str(b + 1):32s} "
              f"{len(g):6d} {g['d_k'].median():10.2f} {brier:8.3f} {skill:8.3f}")
    print(f"\n   overall Brier {float(((cur['pred'] - cur['y']) ** 2).mean()):.3f} "
          f"vs base-rate {b0:.3f}")

    # ---- 2. anchor availability ----------------------------------------------------------
    radii = [float(np.quantile(cur["d_k"], p)) for p in (0.1, 0.25, 0.5, 0.75)]
    av = anchor_availability(x, act, pid, radii, args.min_anchors)
    print(f"\n2. ANCHOR AVAILABILITY — share of decisions with >={args.min_anchors} real "
          "neighbours inside a radius")
    print(f"   {'radius':>8} {'own arm':>10} {'own + another arm':>19} "
          f"{'median n on best other arm':>28}")
    print("   " + "-" * 70)
    for _, r in av.iterrows():
        print(f"   {r['radius']:8.2f} {r['own_arm_ok']:9.0%} {r['paired_ok']:18.0%} "
              f"{r['median_best_alt_arm']:28.0f}")

    # ---- 3. the ladder an anchored predictor has to climb ---------------------------------
    lad, preds = baseline_ladder(d, x_model, y, pid)
    knn_brier = float(((cur["pred"] - cur["y"]) ** 2).mean())
    knn_skill = 1 - knn_brier / b0
    print(f"\n3. BASELINE LADDER ({args.features} X) — patient-grouped 5-fold, "
          "skill versus the base rate")
    print(f"   {'model':30s} {'AUC':>7} {'Brier':>8} {'skill':>8}")
    print("   " + "-" * 58)
    print(f"   {f'{args.k}-NN, same action':30s} {'—':>7} {knn_brier:8.3f} {knn_skill:8.3f}")
    for _, r in lad.iterrows():
        print(f"   {r['model']:30s} {r['auc']:7.3f} {r['brier']:8.3f} {r['skill']:8.3f}")
    best = lad.loc[lad["skill"].idxmax()]
    print(f"\n   strongest non-LLM baseline: {best['model']} — AUC {best['auc']:.3f}, "
          f"skill {best['skill']:.3f}")
    print("   not measurable in this release: " + ", ".join(ABSENT_FROM_RELEASE))

    # ---- 4. does the strongest baseline degrade where neighbours run out? -----------------
    med_r = float(np.median(cur["d_k"]))
    sup = skill_by_support(x, y, act, pid, preds[best["model"]], med_r)
    print(f"\n4. WHERE THE BASELINE STANDS VS SUPPORT — same-action neighbours within "
          f"r={med_r:.2f}")
    print(f"   {'neighbours':>12} {'n':>7} {'event rate':>11} {'Brier':>8} {'skill':>8}")
    print("   " + "-" * 52)
    for _, r in sup.iterrows():
        print(f"   {r['support']:>12} {r['n']:7d} {r['event_rate']:10.0%} "
              f"{r['brier']:8.3f} {r['skill']:8.3f}")
    print("\n   anchoring can only help where this skill is LOW and anchors still exist.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "operating_window.json").write_text(json.dumps(
        {"n_decisions": int(len(d)), "n_patients": int(d["record_id"].nunique()),
         "horizon_months": HORIZON_MONTHS, "base_rate": base, "k": args.k,
         "window": win, "availability": av.to_dict("records"),
         "features": args.features, "absent_from_release": list(ABSENT_FROM_RELEASE),
         "knn": {"brier": knn_brier, "skill": knn_skill},
         "ladder": lad.to_dict("records"),
         "skill_by_support": sup.to_dict("records")}, indent=2) + "\n")
    print(f"\nwrote {args.out_dir} (aggregate only — no patient-level rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
