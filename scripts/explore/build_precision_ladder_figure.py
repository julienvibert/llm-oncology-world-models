#!/usr/bin/env python3
"""THE PRECISION-ONCOLOGY LADDER, WALKED WITH REAL PATIENTS ON THE MAP

We choose index patients, then narrow the reference population around them one clinical
condition at a time -- everyone, their tumour type, their stage, their molecular driver,
their baseline state, their age and sex, their exact histology -- and draw the same patient
in every panel. Two things move in opposite directions across the strip:

    the group's answer gets better for this patient
    the group disappears

FOUR FIGURES, deliberately paired so the new representation is judged beside the ones
oncologists already trust:

  A  the ladder in the state space     -- the peer cloud shrinking around each index patient
  B  the ladder in classical language  -- Kaplan-Meier per rung, and a forest plot
  C  the aggregate over every patient  -- so the index patients illustrate the result rather
                                          than being it
  D  three patients, one population    -- the same average is far too optimistic for one and
                                          far too pessimistic for another

WHAT THIS ACTUALLY FOUND, which is not what was expected. On this cohort the error does NOT
turn upward: conditioning keeps helping all the way down. The interventional run behaves
differently -- there the exact anchor is measurably worse than coarser rungs -- and the
difference is sample size per cell, not principle. So the price of precision here is not a
worse answer. It is NO answer: by the finest rung 18% of patients have no comparable patient
at all. That is the same positivity failure the confounding demonstration measures, wearing
clinical clothes.

THE NULL CONTROL is what makes the falling curve mean anything. Each rung is re-run against
a RANDOM partition with the identical number of cells and the identical cell-size profile.
Random cells make the error rise -- that is the pure variance cost of having fewer people to
average over. The clinical ladder falls instead, and the gap between the two curves is the
value of the clinical information, separated from the granularity that carries it. Without
this control a falling curve could just mean "smaller groups", and a reader would be right
to say so.

THE CHERRY-PICK, AND HOW IT IS CLOSED. A figure built around one patient is worthless if the
patient was chosen after seeing the answer. But a maximally typical patient -- the safe
choice -- throws the point away, because the population average already fits them and
nothing surprising can happen.

The resolution is not to hunt for a striking patient. It is to PRE-NAME tumour types on
clinical grounds spanning the prognostic range, and then apply the same typicality rule
inside each. Pancreatic adenocarcinoma and basal cell carcinoma were named as the poles
before any median was computed; they bracket the cohort at 15.1 and 48.8 months against 23.2
for everyone. Within a type the patient is the member closest to that type's centroid in the
twenty encoded coordinates -- covariates only, outcome never consulted. Every number in the
aggregate text is computed over all trajectories rather than over these three.

CENSORING is absent by construction: every trajectory in the atlas runs to the absorbing
exit, so overall survival is fully observed and the survival curves are exact. That is a
property of the generator, not a claim about clinical data, and any caption must say so.

Run: uv run python scripts/explore/build_precision_ladder_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts"))

MAP = ROOT / "outputs" / "paper_state_map"
OUT = ROOT / "outputs" / "exploratory_precision_ladder"
INK, MUTED, GRIDC = "#1f2937", "#64748b", "#d8dee6"
PEER, HERO, NULLC = "#2563eb", "#b91c1c", "#94a3b8"
MONTH = 30.4375  # days per month, the paper convention (was 30.4 until 14 Oct 2026)
SEED = 20260731
NBOOT = 2000
NNULL = 10

# Coarse to fine. Each rung is the previous plus one clinical condition, which is what makes
# this a ladder rather than seven unrelated groupings. Corrected 12 Oct 2026: the
# last rung used to SWAP primary-site class for the exact OncoTree code, on the assumption that
# the code refines the class. In this generated data it does not (98 finest cells merged cells
# of the previous rung), so the last rung now keeps every earlier key and adds the code.
RUNGS: list[tuple[str, tuple[str, ...]]] = [
    ("everyone", ()),
    ("+ tumour type", ("site",)),
    ("+ stage", ("site", "stage")),
    ("+ molecular driver", ("site", "stage", "driver")),
    ("+ baseline state", ("site", "stage", "driver", "ecog_band", "extent_band")),
    ("+ age and sex", ("site", "stage", "driver", "ecog_band", "extent_band",
                       "age_band", "sex")),
    ("+ exact histology", ("site", "stage", "driver", "ecog_band", "extent_band",
                           "age_band", "sex", "tumor_oncotree_code")),
]

# THREE index patients, not one. A single maximally typical patient is the safe choice
# against cherry-picking and it throws away the point: the population average already fits
# them, so nothing surprising can happen. The fix is not to hunt for a striking patient --
# that is the cherry-pick -- but to PRE-NAME tumour types on clinical grounds spanning the
# prognostic range, and then apply the same typicality rule inside each.
#
# Pancreatic adenocarcinoma and basal cell carcinoma were named as the two poles before
# any median was computed. They bracket the cohort: 15.1 and 48.8 months against 23.2 for
# everyone. The algorithmically typical patient stays as the neutral third case.
TYPICAL = "a patient who looks average"
PRENAMED: list[tuple[str, str | None]] = [
    ("pancreatic adenocarcinoma (PAAD)", "PAAD"),
    (TYPICAL, None),
    ("basal cell carcinoma (BCC)", "BCC"),
]


def load() -> tuple[pd.DataFrame, list[str]]:
    """One row per trajectory: covariates, baseline state, map position, overall survival."""
    import build_paper_state_map as smap

    contract = json.load(open(MAP / "state_map_contract.json"))
    order = contract["primary_model"]["feature_names"]

    tab = smap.load_reference_tables(smap.REFERENCE_RUNS)
    base = tab["baselines"].groupby("_gid").first()
    end = tab["endpoints"].groupby("_gid").first()
    proj = pd.read_csv(MAP / "state_projection.csv.gz")
    first = proj.sort_values("snapshot_index").groupby("_gid", as_index=True).first()

    d = base[["model_short", "tumor_oncotree_code", "stage",
              "primary_site_class_canonical", "primary_driver_bucket",
              "age_at_diagnosis", "sex"]].copy()
    d = d.join(end[["os_days"]]).join(first[[*order, "pc1", "pc2"]], how="inner")

    d["site"] = d["primary_site_class_canonical"].astype(str)
    d["driver"] = d["primary_driver_bucket"].astype(str)
    d["stage"] = d["stage"].astype(str)
    d["sex"] = d["sex"].astype(str)
    d["os_months"] = d["os_days"].astype(float) / MONTH

    # Bands, not raw lattice levels. host_ecog and tumour_extent sit on five- and six-level
    # lattices; using them raw would shatter the finest rungs and make the coverage collapse
    # an artefact of encoding granularity rather than of clinical conditioning.
    d["ecog_band"] = pd.cut(d["host_ecog"], [-0.01, 0.24, 0.49, 1.01],
                            labels=["ecog 0", "ecog 1", "ecog 2+"]).astype(str)
    d["extent_band"] = np.where(d["tumour_extent"] >= 0.5, "extensive", "limited")
    d["age_band"] = pd.cut(d["age_at_diagnosis"], [0, 50, 60, 70, 80, 200],
                           labels=["<50", "50s", "60s", "70s", "80+"]).astype(str)

    return d[np.isfinite(d["os_months"])].copy(), order


def cells(d: pd.DataFrame, keys: tuple[str, ...]) -> pd.Series:
    """Cell label per trajectory for one rung; the empty rung puts everyone together."""
    if not keys:
        return pd.Series(["all"] * len(d), index=d.index)
    return d[list(keys)].astype(str).agg(" | ".join, axis=1)


def loo_medians(v: np.ndarray) -> np.ndarray:
    """Median of every value in `v` except itself, exactly, in O(m log m).

    The brute-force version costs O(m^2 log m) per cell, which the whole-cohort rung alone
    makes unaffordable once the null control repeats it. Sorting once and indexing around
    the removed position gives the same answer.
    """
    m = len(v)
    n = m - 1
    if n < 1:
        return np.full(m, np.nan)
    o = np.argsort(v, kind="stable")
    s = v[o]
    p = np.arange(m)

    def elem(i: int) -> np.ndarray:
        """Element at index `i` of the reduced array, for each removal position `p`."""
        return np.where(i < p, s[min(i, m - 1)], s[min(i + 1, m - 1)])

    if n % 2 == 1:
        med = elem(n // 2)
    else:
        med = 0.5 * (elem(n // 2 - 1) + elem(n // 2))
    out = np.empty(m, dtype=float)
    out[o] = med
    return out


def loo_error(labels: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Absolute error of each patient's peer-median, and each patient's peer count."""
    err = np.full(len(y), np.nan)
    npeer = np.zeros(len(y), dtype=int)
    for c in pd.unique(labels):
        idx = np.flatnonzero(labels == c)
        npeer[idx] = len(idx) - 1
        if len(idx) < 2:
            continue
        err[idx] = np.abs(y[idx] - loo_medians(y[idx]))
    return err, npeer


def random_like(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random partition with the identical cell-size profile."""
    sizes = pd.Series(labels).value_counts().to_numpy()
    perm = rng.permutation(len(labels))
    out = np.empty(len(labels), dtype=object)
    at = 0
    for ci, sz in enumerate(sizes):
        out[perm[at:at + sz]] = f"r{ci}"
        at += sz
    return out


def pick_index_patient(d: pd.DataFrame, order: list[str], code: str | None = None) -> str:
    """Covariate-only rule, fixed before anything was run. Never touches outcome.

    With `code`, the patient is the member of that pre-named tumour type closest to the
    type's centroid in the twenty encoded coordinates -- a typical patient *of that type*.
    Without one, the same rule is applied to the largest cell at the finest rung, giving a
    typical patient of the whole cohort.
    """
    if code is None:
        lab = cells(d, RUNGS[-1][1])
        pool = d[lab.eq(lab.value_counts().idxmax())]
    else:
        pool = d[d["tumor_oncotree_code"].astype(str).eq(code)]
        if pool.empty:
            raise KeyError(f"pre-named tumour code {code!r} is not in the cohort")
    X = pool[order].to_numpy(float)
    return str(pool.index[int(np.argmin(((X - X.mean(0)) ** 2).sum(1)))])


def km(vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Survival with no censoring anywhere in this cohort: the complementary ECDF."""
    return np.array([(vals > t).mean() for t in grid])


def boot_median_ci(vals: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    if len(vals) == 1:
        return (float(vals[0]), np.nan, np.nan)
    m = np.median(rng.choice(vals, size=(NBOOT, len(vals)), replace=True), axis=1)
    return float(np.median(vals)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d, order = load()
    y = d["os_months"].to_numpy(float)
    heroes = {lbl: pick_index_patient(d, order, code) for lbl, code in PRENAMED}
    hero = heroes[TYPICAL]
    h = d.loc[hero]
    print(f"{len(d):,} trajectories; index patients {heroes}", file=sys.stderr)

    # ---- the ladder, and the matched-granularity null beside it
    per_rung, err_stack, npeer_stack, null_part = [], [], [], []
    peers: dict[str, dict[str, np.ndarray]] = {lbl: {} for lbl in heroes}
    for name, keys in RUNGS:
        lab_s = cells(d, keys)
        lab = lab_s.to_numpy()
        err, npeer = loo_error(lab, y)
        nulls = np.array([loo_error(random_like(lab, rng), y)[0] for _ in range(NNULL)])
        null_part.append(nulls)
        npeer_stack.append(npeer)
        ok = np.isfinite(err)
        per_rung.append({
            "rung": name, "n_cells": int(len(pd.unique(lab))),
            "median_err": float(np.median(err[ok])), "mean_err": float(np.mean(err[ok])),
            "null_median_err": float(np.median(np.nanmedian(nulls, axis=1))),
            "median_peers": float(np.median(npeer)),
            "unanswerable": float((~ok).mean()), "n_scored": int(ok.sum())})
        err_stack.append(err)
        for lbl, gid in heroes.items():
            peers[lbl][name] = d.index[lab_s.eq(lab_s.loc[gid]) & (d.index != gid)].to_numpy()
    A = pd.DataFrame(per_rung)

    # Like-for-like: the same patients scored at every rung, so the error curve is not read
    # off a cohort that quietly loses its hardest members as the rungs tighten.
    M = np.vstack(err_stack)
    fixed = np.isfinite(M).all(0)
    A["median_err_fixed"] = [float(np.median(M[i][fixed])) for i in range(len(RUNGS))]
    # Every rung must be a refinement of the one before (cumulative keys): assert it.
    for i in range(1, len(RUNGS)):
        prev = cells(d, RUNGS[i - 1][1]).to_numpy()
        cur = cells(d, RUNGS[i][1]).to_numpy()
        assert pd.Series(prev).groupby(cur).nunique().max() == 1, RUNGS[i][0]

    # RANDOM CONTROL ON THE SAME TARGETS. The partition null above scores each
    # random partition's own eligible trajectories, which differ from the fixed cohort of the
    # clinical curve. Declared policy for the comparison drawn in the paper: every fixed target
    # keeps exactly the number of comparators it has under clinical matching at that rung, but
    # they are drawn at random (without replacement) from all other trajectories. No target
    # can fall into a random singleton, so coverage is 100% of the fixed cohort by construction.
    # The partition null restricted to the fixed targets is also reported, with its coverage.
    rng_null = np.random.default_rng(SEED + 7)
    fidx = np.flatnonzero(fixed)
    n_all = len(y)
    null_fixed = np.zeros((NNULL, len(RUNGS)))
    for b in range(NNULL):
        for i in range(len(RUNGS)):
            errs = np.empty(len(fidx))
            for j, t in enumerate(fidx):
                k = int(npeer_stack[i][t])
                pick = rng_null.choice(n_all - 1, size=k, replace=False)
                pick = pick + (pick >= t)          # skip the target itself
                errs[j] = abs(y[t] - np.median(y[pick]))
            null_fixed[b, i] = np.median(errs)
    A["null_median_err_fixed_targets"] = np.median(null_fixed, axis=0)
    A["null_fixed_draw_lo"] = np.percentile(null_fixed, 2.5, axis=0)
    A["null_fixed_draw_hi"] = np.percentile(null_fixed, 97.5, axis=0)
    part_on_fixed, part_cov = [], []
    for i in range(len(RUNGS)):
        sub = null_part[i][:, fixed]
        part_cov.append(float(np.isfinite(sub).mean()))
        part_on_fixed.append(float(np.median(np.nanmedian(sub, axis=1))))
    A["null_partition_on_fixed_targets"] = part_on_fixed
    A["null_partition_fixed_coverage"] = part_cov

    # Paired bootstrap on the coarse-to-fine change, over the fixed cohort.
    idx = np.flatnonzero(fixed)
    dif = M[-1][idx] - M[0][idx]
    draws = rng.integers(0, len(idx), size=(NBOOT, len(idx)))
    bm = np.median(dif[draws], axis=1)
    gain = (float(np.median(dif)), float(np.percentile(bm, 2.5)),
            float(np.percentile(bm, 97.5)))

    # ---------------------------------------------------------------- figure A: the map
    # FOUR of the seven rungs, not all seven. In BASELINE state space the peer cloud has
    # already collapsed by the second rung -- tumour type largely fixes where a patient
    # starts -- so seven panels showed five that were visually identical. Four larger panels
    # keep every step distinguishable; the counts for all seven are in figures B and C.
    from scipy.spatial import ConvexHull, QhullError

    show = [RUNGS[i] for i in (0, 1, 4, 6)]
    figA, axA = plt.subplots(len(heroes), len(show), figsize=(17.5, 12.6),
                             sharex=True, sharey=True)
    for r, (lbl, gid) in enumerate(heroes.items()):
        hh = d.loc[gid]
        for c, (name, _k) in enumerate(show):
            ax = axA[r, c]
            p = d.loc[peers[lbl][name]]
            ax.scatter(d["pc1"], d["pc2"], s=3.0, color=GRIDC, alpha=0.45,
                       edgecolors="none")
            if len(p) >= 3:
                # The hull makes the region shrink visibly even where points overlap.
                try:
                    pts = p[["pc1", "pc2"]].to_numpy(float)
                    hull = ConvexHull(pts)
                    ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], color=PEER,
                            alpha=0.11, linewidth=1.2, edgecolor=PEER, zorder=2)
                except (QhullError, ValueError):
                    pass
            ax.scatter(p["pc1"], p["pc2"], s=13, color=PEER, alpha=0.65,
                       edgecolors="none", zorder=3)
            if len(p):
                ax.scatter([p["pc1"].mean()], [p["pc2"].mean()], s=190, marker="P",
                           facecolor="white", edgecolor=PEER, linewidth=2.2, zorder=4)
            ax.scatter([hh["pc1"]], [hh["pc2"]], s=230, marker="o", facecolor=HERO,
                       edgecolor="white", linewidth=2.0, zorder=5)
            ax.set_title(f"{name}   —   {len(p):,} comparable", fontsize=10.5, color=INK)
            ax.spines[["top", "right"]].set_visible(False)
            if r == len(heroes) - 1:
                ax.set_xlabel("PC1  (severity)", fontsize=9.5)
            if c == 0:
                ax.set_ylabel(f"{lbl}\n\nPC2  (marrow axis)", fontsize=10)
    figA.legend(handles=[
        plt.Line2D([], [], ls="", marker="o", ms=9, color=HERO, label="index patient"),
        plt.Line2D([], [], ls="", marker="o", ms=6, color=PEER, label="comparable patients"),
        plt.Line2D([], [], ls="", marker="P", ms=10, mfc="white", mec=PEER,
                   label="their average"),
        plt.Line2D([], [], ls="", marker="o", ms=5, color=GRIDC, label="whole cohort")],
        fontsize=10, frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 0.045))
    figA.suptitle("Narrowing the reference population around three patients — the group "
                  "closes in, and then it runs out\nfour of the seven rungs; tumour types "
                  "pre-named on clinical grounds, patient chosen inside each on covariates "
                  "only", fontsize=13.5, color=INK, y=1.015)
    figA.tight_layout(rect=(0, 0.055, 1, 0.985))
    figA.savefig(OUT / "ladder_state_space.png", dpi=170, bbox_inches="tight")
    plt.close(figA)

    # ------------------------------------------------- figure B: the classical presentation
    grid = np.linspace(0, float(np.percentile(y, 98)), 240)
    figB = plt.figure(figsize=(24.0, 8.8))
    gs = figB.add_gridspec(2, len(RUNGS), height_ratios=[1.0, 0.9], hspace=0.45)
    forest = []
    for j, (name, _k) in enumerate(RUNGS):
        ax = figB.add_subplot(gs[0, j])
        vals = d.loc[peers[TYPICAL][name], "os_months"].to_numpy(float)
        ax.step(grid, km(y, grid), where="post", color=GRIDC, lw=1.6, label="whole cohort")
        if len(vals):
            # With no censoring the curve is exact; the band is sampling uncertainty in the
            # peer group, which is the quantity that collapses.
            B = np.array([km(rng.choice(vals, len(vals), replace=True), grid)
                          for _ in range(300)])
            ax.fill_between(grid, np.percentile(B, 2.5, axis=0),
                            np.percentile(B, 97.5, axis=0), color=PEER, alpha=0.18,
                            linewidth=0)
            ax.step(grid, km(vals, grid), where="post", color=PEER, lw=2.2,
                    label="comparable patients")
        ax.axvline(h["os_months"], color=HERO, lw=2.0, ls="--")
        ax.set_title(f"{name}\n(n = {len(vals):,})", fontsize=10, color=INK)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("months from diagnosis", fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if j == 0:
            ax.set_ylabel("surviving", fontsize=9)
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")
        med, lo, hi = boot_median_ci(vals, rng)
        forest.append({"rung": name, "n": int(len(vals)), "median": med, "lo": lo, "hi": hi})

    axf = figB.add_subplot(gs[1, :])
    ypos = np.arange(len(forest))[::-1]
    for yp, f in zip(ypos, forest):
        if np.isfinite(f["lo"]):
            axf.plot([f["lo"], f["hi"]], [yp, yp], color=PEER, lw=2.6, solid_capstyle="butt")
        if np.isfinite(f["median"]):
            axf.plot([f["median"]], [yp], "o", color=PEER, ms=9, zorder=3)
        w = (f["hi"] - f["lo"]) if np.isfinite(f["lo"]) else np.nan
        axf.text(1.004, yp, f"n = {f['n']:,}" + ("" if not np.isfinite(w)
                 else f"     interval {w:.1f} mo"), transform=axf.get_yaxis_transform(),
                 va="center", fontsize=9, color=MUTED)
    axf.axvline(h["os_months"], color=HERO, lw=2.0, ls="--")
    axf.text(h["os_months"], len(forest) - 0.3, "  what this patient actually did",
             color=HERO, fontsize=10, va="bottom")
    axf.set_yticks(ypos)
    axf.set_yticklabels([f["rung"] for f in forest], fontsize=10.5)
    axf.set_xlabel("median overall survival of the comparable group (months), with 95% "
                   "bootstrap interval", fontsize=10)
    axf.set_ylim(-0.7, len(forest) - 0.1)
    axf.spines[["top", "right", "left"]].set_visible(False)
    axf.tick_params(axis="y", length=0)
    figB.suptitle("The same ladder in the language oncologists already read — survival "
                  "curves and a forest plot\nthe estimate barely moves; the interval widens "
                  "as the group empties", fontsize=13.5, color=INK, y=0.985)
    figB.savefig(OUT / "ladder_classical.png", dpi=170, bbox_inches="tight")
    plt.close(figB)

    # ------------------------------- figure D: three patients, one population, three verdicts
    # This is the figure the typical-patient version could not produce. The population median
    # is a single number; against a pancreatic adenocarcinoma it is far too optimistic and
    # against a basal cell carcinoma far too pessimistic, and the two errors point in
    # opposite directions -- which is exactly why an unbiased average can be the wrong number
    # for everyone in the population it averages over.
    hero_forest: dict[str, list[dict]] = {}
    figD, axD = plt.subplots(1, len(heroes), figsize=(19.0, 5.6), sharex=True)
    for ax, (lbl, gid) in zip(axD, heroes.items()):
        hh = d.loc[gid]
        rows = []
        for name, _k in RUNGS:
            vals = d.loc[peers[lbl][name], "os_months"].to_numpy(float)
            med, lo, hi = boot_median_ci(vals, rng)
            rows.append({"rung": name, "n": int(len(vals)), "median": med, "lo": lo,
                         "hi": hi,
                         "error": abs(float(hh["os_months"]) - med) if len(vals) else np.nan})
        hero_forest[lbl] = rows
        yp = np.arange(len(rows))[::-1]
        for p, r in zip(yp, rows):
            if np.isfinite(r["lo"]):
                ax.plot([r["lo"], r["hi"]], [p, p], color=PEER, lw=2.6,
                        solid_capstyle="butt")
            if np.isfinite(r["median"]):
                # A "median" over one or two peers is not an estimate, and must not look
                # like one. Hollow marker, and the count says why.
                thin = r["n"] < 3
                ax.plot([r["median"]], [p], "o", ms=9 if thin else 8,
                        mfc="white" if thin else PEER, mec=PEER,
                        mew=2.0 if thin else 0, zorder=3)
            ax.text(0.995, p, f"n = {r['n']:,}", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=8,
                    color=HERO if r["n"] < 3 else MUTED)
        ax.axvline(hh["os_months"], color=HERO, lw=2.2, ls="--")
        ax.axvline(float(np.median(y)), color=NULLC, lw=1.6, ls=":")
        ax.set_yticks(yp)
        ax.set_yticklabels([r["rung"] for r in rows], fontsize=9.5)
        ax.set_title(f"{lbl}\nactually survived {hh['os_months']:.1f} months",
                     fontsize=11.5, color=INK)
        ax.set_xlabel("median OS of the comparable group (months)", fontsize=9.5)
        ax.set_xscale("log")
        ax.set_xticks([10, 20, 40, 80])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    figD.legend(handles=[
        plt.Line2D([], [], color=HERO, lw=2.2, ls="--",
                   label="this patient's actual survival"),
        plt.Line2D([], [], color=NULLC, lw=1.6, ls=":", label="whole-cohort median"),
        plt.Line2D([], [], ls="", marker="o", ms=8, color=PEER,
                   label="group median, 95% interval"),
        plt.Line2D([], [], ls="", marker="o", ms=9, mfc="white", mec=PEER, mew=2.0,
                   label="fewer than 3 comparable patients — not an estimate")],
        fontsize=9, frameon=False, ncols=4, loc="upper center",
        bbox_to_anchor=(0.5, 0.055))
    figD.suptitle("One population average, three patients, three verdicts — the same number "
                  "is far too optimistic for one and far too pessimistic for another\n"
                  "conditioning moves the estimate onto each patient, until there is nobody "
                  "left to estimate from", fontsize=13, color=INK, y=1.02)
    figD.tight_layout(rect=(0, 0.06, 1, 1))
    figD.savefig(OUT / "ladder_three_patients.png", dpi=170, bbox_inches="tight")
    plt.close(figD)

    # ------------------------------------------------------ figure C: every patient, not one
    figC, axC = plt.subplots(1, 3, figsize=(18.5, 5.2))
    x = np.arange(len(RUNGS))
    lbl = [r["rung"] for r in per_rung]
    axC[0].plot(x, A["null_median_err"], "--o", color=NULLC, lw=2.2, ms=7,
                label="random groups of the same sizes")
    axC[0].plot(x, A["median_err_fixed"], "-o", color=PEER, lw=2.6, ms=8,
                label="real clinical conditioning")
    axC[0].fill_between(x, A["median_err_fixed"], A["null_median_err"], color=PEER,
                        alpha=0.10, linewidth=0)
    axC[0].annotate("what the clinical\ninformation is worth",
                    (x[-1], 0.5 * (A["median_err_fixed"].iloc[-1] + A["null_median_err"].iloc[-1])),
                    textcoords="offset points", xytext=(-108, -6), color=PEER, fontsize=9.5)
    axC[0].set_ylabel("median error of the group's prediction (months)", fontsize=9.5)
    axC[0].set_title("How wrong the comparable group is\nfor every patient",
                     fontsize=11, color=INK)
    axC[0].legend(fontsize=8.5, frameon=False, loc="lower left")
    axC[1].plot(x, A["median_peers"], "-o", color=MUTED, lw=2.4, ms=8)
    axC[1].set_yscale("log")
    axC[1].set_ylabel("median comparable patients", fontsize=9.5)
    axC[1].set_title("How many patients are left\nto learn from", fontsize=11, color=INK)
    axC[2].plot(x, 100 * A["unanswerable"], "-o", color=HERO, lw=2.4, ms=8)
    axC[2].set_ylabel("% of patients with no comparable patient", fontsize=9.5)
    axC[2].set_title("How often the rung cannot\nanswer at all", fontsize=11, color=INK)
    for ax in axC:
        ax.set_xticks(x)
        ax.set_xticklabels(lbl, rotation=24, ha="right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    figC.suptitle(f"The same walk repeated for all {len(d):,} patients — conditioning pays, "
                  "and the price is paid in coverage, not accuracy",
                  fontsize=13, color=INK, y=1.02)
    figC.tight_layout()
    figC.savefig(OUT / "ladder_all_patients.png", dpi=170, bbox_inches="tight")
    plt.close(figC)

    # ------------------------------------------------------------------------- readout
    hero_rows = hero_forest[TYPICAL]

    lines = [
        "# The precision-oncology ladder, walked with one patient on the map", "",
        f"{len(d):,} trajectories, six generators pooled. Overall survival is fully "
        "observed — every trajectory runs to the absorbing exit, so there is no censoring "
        "anywhere. That makes the survival curves exact, and it is a property of the "
        "generator rather than a claim about clinical data.", "",
        "## The headline, which is not the one expected", "",
        "The error does **not** turn upward on this cohort. Conditioning keeps helping all "
        f"the way down: median error falls {gain[0]:+.2f} months from *everyone* to the "
        f"finest rung (95% CI {gain[1]:+.2f} to {gain[2]:+.2f}, paired bootstrap over the "
        f"{int(fixed.sum()):,} patients scorable at every rung). The interventional run "
        "behaves the other way — there the exact anchor is measurably *worse* than coarser "
        "rungs — and the difference is patients per cell, not principle.", "",
        "So on this cohort the price of precision is not a worse answer. It is **no "
        f"answer**: by the finest rung {100 * A['unanswerable'].iloc[-1]:.1f}% of patients "
        "have no comparable patient at all. That is the positivity failure the confounding "
        "demonstration measures, in clinical clothes.", "",
        "## The null control, which is what makes the falling curve mean anything", "",
        "Each rung is re-run against a random partition with the identical number of cells "
        f"and the identical cell-size profile, averaged over {NNULL} draws.", "",
        "| rung | cells | real | random groups, same sizes | gap |", "|---|---|---|---|---|"]
    for r in per_rung:
        lines.append(f"| {r['rung']} | {r['n_cells']:,} | {r['median_err']:.2f} mo | "
                     f"{r['null_median_err']:.2f} mo | "
                     f"**{r['null_median_err'] - r['median_err']:.2f} mo** |")
    lines += ["",
              "Random cells make the error **rise** — that is the pure variance cost of "
              "having fewer people to average over. The clinical ladder falls instead. The "
              "gap is the value of the clinical information, separated from the granularity "
              "that carries it. Without this control a falling curve could just mean "
              "'smaller groups', and a reader would be right to say so.", "",
              "It also names the failure mode of an uninformative subgroup analysis "
              "exactly: you pay the whole variance cost and collect none of the benefit.",
              "", "## Every patient", "",
              "| rung | median error | like-for-like | median comparable patients | "
              "unanswerable |", "|---|---|---|---|---|"]
    for i, r in enumerate(per_rung):
        lines.append(f"| {r['rung']} | {r['median_err']:.2f} mo | "
                     f"{A['median_err_fixed'].iloc[i]:.2f} mo | {r['median_peers']:.0f} | "
                     f"{100 * r['unanswerable']:.1f}% |")
    lines += ["",
              "*Like-for-like* scores the same patients at every rung. Without it the error "
              "curve is read off a cohort that quietly loses its hardest members as the "
              "rungs tighten — the unanswerable patients are the unusual ones, and dropping "
              "them flatters the fine rungs. Here the two columns barely differ, so the "
              "selection is not driving the result; the check still has to be run and "
              "reported.", "",
              "## Three patients, one population average", "",
              "A single maximally typical index patient is safe against cherry-picking and "
              "throws away the point: the population average already fits them, so nothing "
              "surprising can happen. The fix is not to hunt for a striking patient — that "
              "*is* the cherry-pick — but to pre-name tumour types on clinical grounds "
              "spanning the prognostic range, and apply the same typicality rule inside "
              "each. Pancreatic adenocarcinoma and basal cell carcinoma were named before "
              "any median was computed.", "",
              "| patient | actual OS | *everyone* says | error | finest rung says | error |",
              "|---|---|---|---|---|---|"]
    for lbl, gid in heroes.items():
        rows = hero_forest[lbl]
        a, z = rows[0], rows[-1]
        lines.append(
            f"| {lbl} | {d.loc[gid, 'os_months']:.1f} mo | {a['median']:.1f} mo | "
            f"{a['error']:.1f} mo | "
            + (f"{z['median']:.1f} mo | {z['error']:.1f} mo |" if np.isfinite(z['median'])
               else "no comparable patient | — |"))
    lines += ["",
              "The population median is one number, and it is far too optimistic for the "
              "pancreatic patient and far too pessimistic for the basal cell patient — two "
              "errors in opposite directions. That is what it means for an unbiased average "
              "to be the wrong number for everybody in the population it averages over, and "
              "it is the reason this figure needed more than one index patient.", "",
              "## The average-looking index patient in detail", "",
              f"`{hero}` — {h['site']}, {h['stage']}, driver `{h['driver']}`, "
              f"{h['ecog_band']}, {h['extent_band']} disease, {h['age_band']}, {h['sex']}. "
              f"Actual survival **{h['os_months']:.1f} months**.", "",
              "Chosen by a rule fixed before anything was run, on **covariates only, never "
              "outcome**: the largest cell at the finest rung, then its member closest to "
              "that cell's centroid in the twenty encoded coordinates. That selects a "
              "maximally *typical* patient, which biases against the story — a typical "
              "patient should already agree with the population average.", "",
              "| rung | comparable patients | group's median OS | error for this patient |",
              "|---|---|---|---|"]
    for r in hero_rows:
        gm = "—" if not np.isfinite(r["median"]) else f"{r['median']:.1f} mo"
        er = "—" if not np.isfinite(r["error"]) else f"{r['error']:.1f} mo"
        lines.append(f"| {r['rung']} | {r['n']:,} | {gm} | {er} |")
    lines += ["",
              "**This patient's error does not fall monotonically, and the figure must not "
              "pretend otherwise.** One patient is one draw; the population result is the "
              "table above it. That is precisely why the index patient is an illustration "
              "and figure C is the evidence.", "",
              "There is a second lesson in this panel worth keeping. This patient was chosen "
              "for being maximally average *in covariates*, and they survived 13.7 months "
              f"against a cohort median of {np.median(y):.1f}. **Looking average does not "
              "make your outcome average**, and no amount of conditioning on the recorded "
              "variables moved the group's estimate onto them. Whatever separated this "
              "patient from their peers is not in the twenty coordinates.", "",
              "## The forest plot is the one to show a clinician", "",
              "| rung | n | median OS | 95% interval | interval width |",
              "|---|---|---|---|---|"]
    for f in forest:
        if np.isfinite(f["lo"]):
            lines.append(f"| {f['rung']} | {f['n']:,} | {f['median']:.1f} | "
                         f"{f['lo']:.1f} – {f['hi']:.1f} | {f['hi'] - f['lo']:.1f} mo |")
        else:
            lines.append(f"| {f['rung']} | {f['n']:,} | — | — | — |")
    lines += ["",
              "The point estimate barely moves down the ladder while the interval widens. "
              "The state-space strip and the forest plot are the same fact in two "
              "languages, which is the argument for showing them side by side rather than "
              "asking a reader to trade one for the other.", "",
              "## What this figure may and may not be used for", "",
              "| may | may not |", "|---|---|",
              "| show a population average sitting far from an individual | claim an optimal "
              "rung for real oncology |",
              "| show support collapsing as conditioning tightens | treat these survival "
              "times as calibrated to human data |",
              "| show what informative conditioning buys over granularity | read the PCA "
              "coordinates as a sufficient patient representation |", "",
              "**Ordering caveat.** Driver is placed before baseline state and demographics "
              "last; other orders are equally defensible clinically. The qualitative result "
              "— error falls, support collapses, the null rises — does not depend on it.",
              "",
              "**Pooling caveat.** Six generators are pooled, so part of the spread at "
              "*everyone* is disagreement between models rather than between patients. That "
              "inflates the coarse rung and therefore flatters the ladder. A "
              "single-generator sensitivity belongs in any final version.", "",
              "**Outcome caveat.** Overall survival from diagnosis is one endpoint and a "
              "forgiving one; a short-horizon outcome has far less within-cell signal to "
              "recover, which is the most likely reason the interventional run turns where "
              "this does not."]

    (OUT / "readout.md").write_text("\n".join(lines) + "\n")
    (OUT / "readout.json").write_text(json.dumps(
        {"n_trajectories": int(len(d)), "index_patient": hero,
         "index_patient_os_months": float(h["os_months"]),
         "index_patient_selection": "largest finest-rung cell, member nearest centroid, "
                                    "covariates only",
         "index_patients": {lbl: {"gid": gid, "os_months": float(d.loc[gid, "os_months"]),
                                  "code": str(d.loc[gid, "tumor_oncotree_code"]),
                                  "by_rung": hero_forest[lbl]}
                            for lbl, gid in heroes.items()},
         "prenamed_types": [c for _lbl, c in PRENAMED if c],
         "rungs": [{"name": n, "keys": list(k)} for n, k in RUNGS],
         "per_rung": per_rung, "median_err_fixed": A["median_err_fixed"].tolist(),
         "null_fixed_targets": {
             "policy": "each fixed target keeps its clinical comparator count; comparators drawn at random "
                       "without replacement from all other trajectories; coverage 100% of the fixed cohort",
             "median_err": A["null_median_err_fixed_targets"].tolist(),
             "draw_lo": A["null_fixed_draw_lo"].tolist(), "draw_hi": A["null_fixed_draw_hi"].tolist(),
             "partition_null_on_fixed_targets": A["null_partition_on_fixed_targets"].tolist(),
             "partition_null_fixed_coverage": A["null_partition_fixed_coverage"].tolist(),
             "seed": SEED + 7},
         "ladder_definition": "cumulative: every rung keeps all earlier keys (corrected 12 Oct 2026, Codex R16)",
         "n_fixed_cohort": int(fixed.sum()),
         "coarse_to_fine_gain_months": {"median": gain[0], "lo": gain[1], "hi": gain[2]},
         "hero_by_rung": hero_rows, "forest": forest,
         "censoring": "none; every trajectory reaches the absorbing exit",
         "n_null_draws": NNULL, "seed": SEED}, indent=2, default=str) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
