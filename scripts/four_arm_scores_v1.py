#!/usr/bin/env python3
"""Four-arm scoring layer: pure functions, no data access.

Specification: `manuscript/SCORING_SPEC_2026-09-28.md` §5-§9, as amended on 30 September and
by `manuscript/SCORING_SPEC_amendment_2026-10-12.md`. This module turns already-built
objects (17-coordinate vectors from `four_arm_eligibility_v2.coordinates`, observed
interval paths, survival times) into scores. It reads no file, calls no provider and
writes nothing; the cohort extraction belongs to `score_four_arm_final_v1.py`.

Primary and sensitivity questions are kept apart (12 October 2026):

- joint score: equal-fan-size energy score is PRIMARY; as-delivered and the LLM-only fair
  estimator are sensitivities that answer different questions;
- dependence shuffles test relations between blocks, not chronology; event order is
  checked separately by time-aligned disagreement and the progression-to-regimen timing;
- bootstrap: 2,000 resampled patient cohorts, stratified by cancer; each sampled patient
  brings all arms, repeats and donor cards together.

Definitions this module fixes where §6 is silent (to be recorded in the amendment):
occupancy total variation and time-aligned disagreement are fractions of the window
length |W| (0 = identical, 1 = completely different), so each donor window has the same
weight; |W| is returned beside every loss so that months can be recovered.
"""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import four_arm_eligibility_v2 as E  # noqa: E402

SEED = 20260928
N_SUBSETS = 200
N_PERMUTATIONS = 200
N_BOOTSTRAP = 2000
G_FLOOR = 0.05                    # §5: below this, an IPCW estimate is flagged unsupported
ECE_MAX_BINS = 5
EPS = E.EPS

# block layout of the weighted 17-vector, in `E.COORDINATES` order
_starts = np.cumsum([0] + [len(n) for _, n in E.COORDINATES])
BLOCKS: dict[str, slice] = {b: slice(int(_starts[i]), int(_starts[i + 1]))
                            for i, (b, _) in enumerate(E.COORDINATES)}
N_COORDS = int(_starts[-1])
SURVIVAL, DISEASE, EVENTS, TREATMENT = (BLOCKS[k] for k in
                                        ("survival", "disease", "events", "treatment"))

Span = tuple[float, float]
Segment = tuple[float, float, str]
DEAD, UNRESOLVED = "dead", "unresolved"


# ======================================================================================
# Deterministic seeds
# ======================================================================================

def stable_seed(*keys: Hashable) -> int:
    """A seed from SEED and the keys, independent of the Python process (§7)."""
    text = "|".join([str(SEED)] + [str(k) for k in keys])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


# ======================================================================================
# §7 Energy score
# ======================================================================================

def _pairwise(z: np.ndarray) -> np.ndarray:
    return np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))


def energy_score(z: np.ndarray, y: np.ndarray) -> float:
    """Equal-weight empirical energy score of forecast rows z against vector y.

    ES = mean_i ||z_i - y|| - 0.5 mean_i mean_j ||z_i - z_j||, Euclidean, not squared.
    """
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] != y.shape[0]:
        raise ValueError("z must be (m, d) and y (d,)")
    if not (np.isfinite(z).all() and np.isfinite(y).all()):
        raise ValueError("energy score needs finite, fully resolved vectors")
    return float(np.linalg.norm(z - y, axis=1).mean() - 0.5 * _pairwise(z).mean())


def fair_energy_score(z: np.ndarray, y: np.ndarray) -> float:
    """LLM-only fair estimator: the spread term over m(m-1) ordered pairs, no self-pairs.

    Estimates the generator's score under independent, identically distributed accepted
    draws. Never applied to a donor card (§7).
    """
    z = np.asarray(z, float)
    m = z.shape[0]
    if m < 2:
        raise ValueError("the fair estimator needs at least two draws")
    first = float(np.linalg.norm(z - np.asarray(y, float), axis=1).mean())
    return first - 0.5 * float(_pairwise(z).sum()) / (m * (m - 1))


def equal_size_subsets(m: int, m_star: int, *keys: Hashable,
                       n: int = N_SUBSETS) -> np.ndarray | None:
    """n subsets of size m_star from range(m), each without replacement; None if m == m_star.

    Repeated subsets across draws are permitted. Keys: comparison family, query,
    forecaster, horizon (§7). Draw once; reuse for the joint and the per-block scores.
    """
    if m_star < 1 or m_star > m:
        raise ValueError("need 1 <= m_star <= m")
    if m == m_star:
        return None
    rng = np.random.default_rng(stable_seed(*keys))
    return np.stack([rng.choice(m, size=m_star, replace=False) for _ in range(n)])


def equal_size_score(z: np.ndarray, y: np.ndarray, subsets: np.ndarray | None,
                     cols: slice | None = None) -> float:
    """Mean ordinary energy score over the fixed subsets (complete score if None)."""
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    if cols is not None:
        z, y = z[:, cols], y[cols]
    if subsets is None:
        return energy_score(z, y)
    return float(np.mean([energy_score(z[idx], y) for idx in subsets]))


def block_scores(z: np.ndarray, y: np.ndarray, subsets: np.ndarray | None) -> dict:
    """Per-block energy scores on the weighted block coordinates, same subsets.

    For interpretation only: they are never summed into a joint score (§7).
    """
    return {b: equal_size_score(z, y, subsets, cols) for b, cols in BLOCKS.items()}


def four_arm_m_star(fan_sizes: Sequence[int]) -> int:
    """One m* per patient: the smallest accepted fan over the four arms (§7)."""
    if len(fan_sizes) != len(E.ARMS) or min(fan_sizes) < 1:
        raise ValueError("need four positive fan sizes")
    return int(min(fan_sizes))


@dataclass(frozen=True)
class JointScores:
    equal_size: float              # primary
    as_delivered: float            # sensitivity
    fair: float | None             # LLM-only sensitivity (None for a donor card)
    blocks_equal_size: dict
    m: int
    m_star: int


def joint_scores(z: np.ndarray, y: np.ndarray, m_star: int, *keys: Hashable,
                 llm: bool = True) -> JointScores:
    """All joint quantities for one forecaster on one patient and horizon."""
    z = np.asarray(z, float)
    if not np.isfinite(z).all():
        raise ValueError("an unresolved member makes the forecast unavailable; no rescue")
    subsets = equal_size_subsets(z.shape[0], m_star, *keys)
    return JointScores(equal_size=equal_size_score(z, y, subsets),
                       as_delivered=energy_score(z, y),
                       fair=fair_energy_score(z, y) if llm and z.shape[0] >= 2 else None,
                       blocks_equal_size=block_scores(z, y, subsets),
                       m=int(z.shape[0]), m_star=int(m_star))


# ======================================================================================
# §6 Documented-time course diagnostics
# ======================================================================================

def labelled(observed: Sequence[Segment], death: float, horizon: float) -> list[Segment]:
    """Observed living segments (state or E.UNKNOWN) plus the absorbing post-death exit.

    Unknown generated time is labelled `unresolved`; it is a safeguard mass, not a state.
    Living segments are clipped to [0, min(death, H)), so no living time accrues after
    death and death is absorbing whatever the input.
    """
    stop = min(death, horizon)
    out = []
    for a, b, s in observed:
        a, b = max(0.0, a), min(b, stop)
        if b > a + EPS:
            out.append((a, b, UNRESOLVED if s == E.UNKNOWN else s))
    if death < horizon - EPS:
        out.append((max(0.0, death), horizon, DEAD))
    return out


def _label_at(segs: Sequence[Segment], t: float) -> str | None:
    for a, b, s in segs:
        if a - EPS <= t < b - EPS:
            return s
    return None


def _pieces(window: Sequence[Span], *paths: Sequence[Segment]) -> list[Span]:
    cuts = {x for a, b in window for x in (a, b)}
    for segs in paths:
        cuts |= {x for a, b, _ in segs for x in (a, b)}
    cuts = sorted(cuts)
    out = []
    for lo, hi in zip(cuts, cuts[1:], strict=False):
        if hi > lo + EPS and any(a - EPS <= lo and hi <= b + EPS for a, b in window):
            out.append((lo, hi))
    return out


def occupancy(segs: Sequence[Segment], window: Sequence[Span]) -> dict[str, float]:
    """Months in each label inside the window (labels absent from the path are omitted)."""
    occ: dict[str, float] = {}
    for lo, hi in _pieces(window, segs):
        s = _label_at(segs, (lo + hi) / 2) or UNRESOLVED
        occ[s] = occ.get(s, 0.0) + hi - lo
    return occ


def course_losses(query: Sequence[Segment], forecast: Sequence[Segment],
                  window: Sequence[Span]) -> dict:
    """Occupancy total variation and time-aligned disagreement on one window, as fractions
    of |W|, plus the forecast's unresolved fraction (charged, reported separately)."""
    size = E.measure(window)
    if size <= EPS:
        raise ValueError("empty window")
    qo, fo = occupancy(query, window), occupancy(forecast, window)
    if UNRESOLVED in qo:
        raise ValueError("the window must lie inside the query's supported time")
    labels = set(qo) | set(fo)
    tv = 0.5 * sum(abs(qo.get(k, 0.0) - fo.get(k, 0.0)) for k in labels) / size
    differ = sum(hi - lo for lo, hi in _pieces(window, query, forecast)
                 if _label_at(query, (lo + hi) / 2) != (_label_at(forecast, (lo + hi) / 2)
                                                         or UNRESOLVED))
    return {"occupancy_tv": tv, "disagreement": differ / size,
            "unresolved_fraction": fo.get(UNRESOLVED, 0.0) / size, "window_months": size}


def resolved_time(segs: Sequence[Segment]) -> list[Span]:
    return E.R.merge((a, b) for a, b, s in segs if s != UNRESOLVED)


LOSSES = ("occupancy_tv", "disagreement")


def donor_diagnostic(query: Sequence[Segment], donors: Sequence[tuple[Sequence[Segment],
                                                                       Sequence[Span]]],
                     repeats: Sequence[Sequence[Segment]]) -> dict:
    """§6 for one patient, one card and its LLM arm.

    donors: (donor path read on the query calendar, its common window W) for every card
    member. Primary: on each usable W (>= one month), the mean loss over LLM repeats minus
    the donor's loss; averaged over usable donors. Unresolved generated time is charged.
    Secondary: per repeat, W reduced to the repeat's resolved time (same one-month floor),
    repeat and donor scored on that reduced interval; retained time and exclusions kept.
    """
    usable = [(d, w) for d, w in donors if E.measure(w) >= E.MIN_WINDOW_MONTHS - EPS]
    out: dict = {"n_donors": len(donors), "n_usable": len(usable),
                 "eligible": len(usable) >= E.MIN_USABLE_DONORS and len(repeats) > 0,
                 "all_twenty_usable": len(donors) == E.CARD_SIZE
                 and len(usable) == E.CARD_SIZE,
                 "scored_months": sum(E.measure(w) for _, w in usable)}
    if not out["eligible"]:
        return out
    prim = {k: [] for k in LOSSES}
    llm_mean = {k: [] for k in LOSSES}
    don = {k: [] for k in LOSSES}
    unres = []
    sec = {k: [] for k in LOSSES}
    sec_llm = {k: [] for k in LOSSES}
    sec_don = {k: [] for k in LOSSES}
    sec_kept = sec_pairs = 0
    sec_months = 0.0
    for dpath, w in usable:
        dl = course_losses(query, dpath, w)
        rl = [course_losses(query, r, w) for r in repeats]
        unres.append(float(np.mean([x["unresolved_fraction"] for x in rl])))
        for k in LOSSES:
            m = float(np.mean([x[k] for x in rl]))
            llm_mean[k].append(m)
            don[k].append(dl[k])
            prim[k].append(m - dl[k])
        per_d = {k: [] for k in LOSSES}
        per_r = {k: [] for k in LOSSES}
        per_dd = {k: [] for k in LOSSES}
        for r in repeats:
            sec_pairs += 1
            w2 = E.intersect(w, resolved_time(r))
            if E.measure(w2) < E.MIN_WINDOW_MONTHS - EPS:
                continue
            sec_kept += 1
            sec_months += E.measure(w2)
            r2, d2 = course_losses(query, r, w2), course_losses(query, dpath, w2)
            for k in LOSSES:
                per_d[k].append(r2[k] - d2[k])
                per_r[k].append(r2[k])
                per_dd[k].append(d2[k])
        for k in LOSSES:
            if per_d[k]:
                sec[k].append(float(np.mean(per_d[k])))
                sec_llm[k].append(float(np.mean(per_r[k])))
                sec_don[k].append(float(np.mean(per_dd[k])))
    for k in LOSSES:
        out[f"{k}_llm_minus_donor"] = float(np.mean(prim[k]))
        out[f"{k}_llm"] = float(np.mean(llm_mean[k]))
        out[f"{k}_donor"] = float(np.mean(don[k]))
        out[f"{k}_secondary_llm_minus_donor"] = float(np.mean(sec[k])) if sec[k] else None
        # component means on exactly the retained secondary pairs and reduced windows
        out[f"{k}_secondary_llm"] = float(np.mean(sec_llm[k])) if sec_llm[k] else None
        out[f"{k}_secondary_donor"] = float(np.mean(sec_don[k])) if sec_don[k] else None
    out["llm_unresolved_fraction"] = float(np.mean(unres))
    out["secondary_pairs_kept"], out["secondary_pairs"] = sec_kept, sec_pairs
    out["secondary_months"] = sec_months
    return out


def arm_diagnostic(query: Sequence[Segment], repeats: Sequence[Sequence[Segment]],
                   window: Sequence[Span]) -> dict | None:
    """§6 four-arm analogue: the query's own supported time (>= one month), identical
    across arms and repeats; mean loss over repeats. None when the window is too short."""
    if E.measure(window) < E.MIN_WINDOW_MONTHS - EPS or not repeats:
        return None
    rl = [course_losses(query, r, window) for r in repeats]
    return {k: float(np.mean([x[k] for x in rl])) for k in
            (*LOSSES, "unresolved_fraction")} | {"window_months": E.measure(window)}


# ======================================================================================
# §8 Dependence (not chronology) and the progression-to-regimen timing check
# ======================================================================================

_MOVABLE = (slice(DISEASE.start, EVENTS.stop), TREATMENT)   # disease+events together


def _shuffle(z: np.ndarray, rng: np.random.Generator, groups: Sequence[np.ndarray]
             ) -> np.ndarray:
    out = z.copy()
    for cols in _MOVABLE:
        for g in groups:
            if len(g) > 1:
                out[g, cols] = z[rng.permutation(g), cols]
    return out


def dependence_check(z: np.ndarray, y: np.ndarray, *keys: Hashable,
                     n: int = N_PERMUTATIONS) -> dict:
    """Shuffled minus original joint energy score on a complete fan (§8).

    Disease+events and treatment blocks are permuted independently across repeats; the
    survival block stays fixed. Unstratified, and within alive-at-H strata (column 0).
    Marginal block distributions are unchanged by construction (checked). This tests the
    relations between blocks; it says nothing about the order of events.
    """
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    base = energy_score(z, y)
    m = z.shape[0]
    alive = z[:, SURVIVAL.start] > 0
    strata = [np.flatnonzero(alive), np.flatnonzero(~alive)]
    out = {"m": m, "original": base}
    for name, groups in (("unstratified", [np.arange(m)]), ("alive_stratified", strata)):
        rng = np.random.default_rng(stable_seed("dependence", name, *keys))
        diffs = []
        for _ in range(n):
            s = _shuffle(z, rng, groups)
            for cols in _MOVABLE:
                assert np.array_equal(np.sort(s[:, cols], 0), np.sort(z[:, cols], 0))
            diffs.append(energy_score(s, y) - base)
        out[f"{name}_mean_shift"] = float(np.mean(diffs))
    out["stratum_sizes"] = (int(alive.sum()), int((~alive).sum()))
    out["stratified_permutation_possible"] = any(len(g) > 1 for g in strata)
    return out


def progression_to_regimen(observed: Sequence[Segment], starts: Sequence[float],
                           end: float) -> dict:
    """First post-landmark entry into progression, and the first regimen start after it.

    `starts`: subsequent regimen start times (index regimen excluded). Paired time only
    when both are observed in [0, end]; a regimen change is never read as progression.
    """
    prog = next((a for i, (a, b, s) in enumerate(observed)
                 if i > 0 and s == "progressing" and observed[i - 1][2] != "progressing"
                 and a <= end + EPS), None)
    later = sorted(t for t in starts if 0 < t <= end + EPS)
    after = next((t for t in later if prog is not None and t >= prog - EPS), None)
    return {"progression": prog is not None, "regimen_after": after is not None,
            "both": prog is not None and after is not None,
            "months_progression_to_regimen": (after - prog) if after is not None else None,
            "regimen_precedes_any_progression": bool(later)
            and (prog is None or later[0] < prog - EPS)}


# ======================================================================================
# §5 Survival: stratified reverse Kaplan-Meier IPCW Brier, ECE, C-index
# ======================================================================================

def tie_groups(time: np.ndarray) -> np.ndarray:
    """Times within EPS of their predecessor (chained, sorted) share the group's first time.

    One tie rule for event groups and risk sets in both Kaplan-Meier helpers; after this
    every comparison uses exact equality.
    """
    time = np.asarray(time, float)
    order = np.argsort(time, kind="stable")
    out = time.copy()
    rep = None
    prev = None
    for i in order:
        t = time[i]
        if prev is None or t - prev > EPS:
            rep = t
        out[i] = rep
        prev = t
    return out


def reverse_km_left(time: np.ndarray, event: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Censoring survival G(t-) = P(C >= t) at each `at`, product over censoring times s < t.

    Kaplan-Meier with censoring as the event. Risk set at s = patients with time >= s,
    deaths at s included (a death tied with a censoring is treated as occurring first in
    the survival analysis, so it is still under observation for censoring at s).
    """
    time = tie_groups(np.asarray(time, float))
    cens = np.asarray(event) == 0
    cts = np.unique(time[cens])
    g_after = []
    g = 1.0
    for s in cts:
        n_risk = int((time >= s).sum())
        c = int((cens & (time == s)).sum())
        g *= 1.0 - c / n_risk
        g_after.append(g)
    g_after = np.asarray(g_after)
    out = np.ones(len(np.atleast_1d(at)))
    for i, t in enumerate(np.atleast_1d(at)):
        k = int((cts < t - EPS).sum())                        # censoring groups strictly < t
        out[i] = g_after[k - 1] if k > 0 else 1.0
    return out


def outcome_at(time: np.ndarray, event: np.ndarray, horizon: float) -> np.ndarray:
    """1 alive through H, 0 dead by H (death at H is a death), nan unknown.

    A recorded last contact exactly at H counts as observed through H.
    """
    time = np.asarray(time, float)
    ev = np.asarray(event).astype(bool)
    y = np.full(len(time), np.nan)
    y[ev & (time <= horizon + EPS)] = 0.0
    y[(time > horizon + EPS) | (~ev & (time >= horizon - EPS))] = 1.0
    return y


def ipcw_weights(time: np.ndarray, event: np.ndarray, strata: np.ndarray,
                 horizon: float) -> tuple[np.ndarray, np.ndarray, bool]:
    """Weights 1/G(T-) for deaths by H and 1/G(H-) for survivors through H; 0 otherwise.

    G is estimated within each cancer stratum on the fixed query cohort. Returns weights,
    outcomes and whether any required G is below G_FLOOR (then: unsupported, no capping).
    """
    time = np.asarray(time, float)
    event = np.asarray(event)
    strata = np.asarray(strata)
    y = outcome_at(time, event, horizon)
    w = np.zeros(len(time))
    low = False
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        tt, ee = time[idx], event[idx]
        died = np.flatnonzero(y[idx] == 0)
        alive = np.flatnonzero(y[idx] == 1)
        if len(died):
            g = reverse_km_left(tt, ee, tt[died])
            low |= bool((g < G_FLOOR).any())
            w[idx[died]] = 1.0 / g
        if len(alive):
            g = reverse_km_left(tt, ee, np.array([horizon]))[0]
            low |= bool(g < G_FLOOR)
            w[idx[alive]] = 1.0 / g
    return w, y, low


def ipcw_brier(time: np.ndarray, event: np.ndarray, strata: np.ndarray, p_alive: np.ndarray,
               horizon: float, m: np.ndarray | None = None) -> dict:
    """Censoring-adjusted Brier score for P(alive at H) over the WHOLE cohort (denominator
    = cohort count), the evaluable-subset Brier, weight support and effective sample size.

    With m (LLM fan sizes) the finite-fan sensitivity (p-y)^2 - p(1-p)/(m-1) is also
    returned, weighted the same way; negative corrected values are kept.
    """
    p = np.asarray(p_alive, float)
    w, y, low = ipcw_weights(time, event, strata, horizon)
    known = ~np.isnan(y)
    loss = np.where(known, (p - np.nan_to_num(y)) ** 2, 0.0)
    out = {"n": int(len(p)), "n_evaluable": int(known.sum()),
           "brier_ipcw": float((w * loss).sum() / len(p)),
           "brier_evaluable": float(loss[known].mean()) if known.any() else None,
           "unsupported": low,
           "max_weight": float(w.max()) if len(w) else None,
           "ess": float(w.sum() ** 2 / (w ** 2).sum()) if (w > 0).any() else 0.0}
    if m is not None:
        mm = np.asarray(m, float)
        if (mm < 2).any():
            raise ValueError("finite-fan correction needs m >= 2")
        corr = np.where(known, loss - p * (1 - p) / (mm - 1), 0.0)
        out["brier_ipcw_fan_corrected"] = float((w * corr).sum() / len(p))
    return out


def tie_preserving_bins(p: np.ndarray, max_bins: int = ECE_MAX_BINS) -> np.ndarray:
    """Bin index per prediction: at most max_bins bins of roughly equal count, never
    splitting identical predicted probabilities. Rule: walk the sorted distinct values and
    close a bin once its cumulative count reaches the next k*n/max_bins."""
    p = np.asarray(p, float)
    vals, counts = np.unique(p, return_counts=True)
    n, b, cum = len(p), 0, 0
    which = {}
    for v, c in zip(vals, counts, strict=True):
        which[v] = b
        cum += c
        if cum >= (b + 1) * n / max_bins - EPS and b < max_bins - 1:
            b += 1
    return np.array([which[v] for v in p])


def ece(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> dict:
    """Descriptive tie-preserving ECE on patients with a known outcome; optional weights
    (e.g. IPCW). Not a proper score; bin counts and the rule are returned."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    w = np.ones(len(p)) if w is None else np.asarray(w, float)
    keep = ~np.isnan(y) & (w > 0)
    p, y, w = p[keep], y[keep], w[keep]
    rule = f"tie-preserving, at most {ECE_MAX_BINS} bins of roughly equal count"
    if len(p) == 0:                       # missing information is not perfect calibration
        return {"ece": None, "n_evaluable": 0, "bins": [], "rule": rule}
    bins = tie_preserving_bins(p)
    rows, total = [], 0.0
    for k in np.unique(bins):
        sel = bins == k
        mp = float(np.average(p[sel], weights=w[sel]))
        my = float(np.average(y[sel], weights=w[sel]))
        total += w[sel].sum() / w.sum() * abs(mp - my)
        rows.append({"bin": int(k), "n": int(sel.sum()), "mean_predicted": mp,
                     "mean_observed": my})
    return {"ece": float(total), "n_evaluable": int(len(p)), "bins": rows, "rule": rule}


def km_alive_at(time: np.ndarray, event: np.ndarray, horizon: float) -> float | None:
    """Kaplan-Meier P(alive through H) for a donor card (its own residual follow-up).

    None when the curve has no support to H and has not reached zero: the last positive
    estimate is never carried forward (§5). A death exactly at H counts as a death.
    """
    time = tie_groups(np.asarray(time, float))
    ev = np.asarray(event).astype(bool)
    s = 1.0
    for t in np.unique(time[ev & (time <= horizon + EPS)]):
        n_risk = int((time >= t).sum())
        d = int((ev & (time == t)).sum())
        s *= 1.0 - d / n_risk
    if s > 0 and not (time >= horizon - EPS).any():
        return None
    return float(s)


def c_index(time: np.ndarray, event: np.ndarray, predicted_rmst: np.ndarray) -> float:
    """Harrell's C for a predicted restricted mean survival (higher = longer survival)."""
    from lifelines.utils import concordance_index
    return float(concordance_index(time, predicted_rmst, event))


# ======================================================================================
# §9 Inference: patient-cohort bootstrap, Wilcoxon, Holm; first-attempt status
# ======================================================================================

def bootstrap_cohorts(strata: Sequence[str], *keys: Hashable,
                      n: int = N_BOOTSTRAP) -> list[np.ndarray]:
    """n resampled patient cohorts (row indices), stratified by cancer, seed SEED.

    Each sampled patient brings all its arms, repeats and donor cards together; callers
    index their per-patient rows with these arrays.
    """
    strata = np.asarray(strata)
    rng = np.random.default_rng(stable_seed("bootstrap", *keys))
    groups = [np.flatnonzero(strata == s) for s in sorted(set(strata.tolist()))]
    return [np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
            for _ in range(n)]


def paired_mean_ci(diffs: np.ndarray, cohorts: Sequence[np.ndarray]) -> dict:
    """Mean paired difference and percentile 95% interval over the resampled cohorts."""
    d = np.asarray(diffs, float)
    boot = np.array([d[idx].mean() for idx in cohorts])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": int(len(d)), "mean": float(d.mean()), "ci_low": float(lo),
            "ci_high": float(hi)}


def wilcoxon_p(diffs: np.ndarray) -> float | None:
    """Two-sided paired Wilcoxon signed-rank P (secondary; tests a different feature from
    the mean). None when every difference is exactly zero or fewer than two pairs; no
    numerical threshold is applied."""
    from scipy.stats import wilcoxon
    d = np.asarray(diffs, float)
    if len(d) < 2 or not np.any(d != 0):
        return None
    return float(wilcoxon(d, alternative="two-sided").pvalue)


def holm(pvalues: Sequence[float | None]) -> list[float | None]:
    """Holm-adjusted P over a fixed family; unavailable slots (None) still count in m."""
    m = len(pvalues)
    avail = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    out: list[float | None] = [None] * m
    running = 0.0
    for rank, (p, i) in enumerate(avail):
        running = max(running, min(1.0, (m - rank) * p))
        out[i] = running
    return out


def first_attempt_status(record: dict) -> str:
    """'valid', 'invalid' or 'unavailable' for the FIRST attempt of one requested slot.

    Read from attempts_detail[0] (acceptance flag and raw response), never from
    attempts_used alone; a successful retry never makes the first attempt valid.
    Unavailable means not saved: no detail, no acceptance flag, or a raw response that is
    absent or null. A saved empty response is an observed output and keeps its flag (an
    unaccepted empty output is invalid). Transport failures have no separate status here.
    """
    detail = record.get("attempts_detail") or []
    if not detail:
        return "unavailable"
    first = detail[0]
    if first.get("accepted") is None or first.get("raw_response") is None:
        return "unavailable"
    return "valid" if first["accepted"] else "invalid"


def finite_or_none(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)
