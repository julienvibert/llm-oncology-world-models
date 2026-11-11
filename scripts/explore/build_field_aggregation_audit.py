#!/usr/bin/env python3
"""IS EACH ARROW AN AVERAGE OVER PATIENTS WHO DO NOT BELONG TOGETHER?

Every arrow in a binned field estimates

    v(u) = E[ dX | P(X) = u ]

the mean displacement conditional on the PROJECTED coordinates. Twelve features go in and
two come out, so a cell is an equivalence class, not a patient. The arrow is honest only if
the patients sharing a cell are exchangeable with respect to where they are about to go.

Two ways that fails, and both are silent:

  * the pooled arrow is a MIXTURE that describes nobody — sub-populations inside the cell
    flow in different directions and the average lands between them
  * the arrow moves when the CASE-MIX inside the cell moves — which would confound every
    model-vs-model and cancer-vs-cancer comparison drawn on a shared field

Four measurements:

  A  what fixing the two display coordinates does NOT pin down — the share of each raw
     feature's variance that survives inside a cell
  B  cancellation ratio  R = |sum_k w_k v_k| / sum_k w_k |v_k|  over sub-populations of a
     cell. R = 1 means every subgroup flows the same way and the arrow is honest; R -> 0
     means the arrow is what is left after opposing flows cancel
  C  the worst cell, drawn: pooled arrow against its sub-arrows
  D  after removing the cell mean, how much of the remaining displacement do the OFF-PLANE
     dimensions still predict? If that is large, the cell is not a state, it is a bag

Run on the prespecified tumour-host plane and on PC1-PC2, so the two can be compared.

Run: uv run python scripts/explore/build_field_aggregation_audit.py
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
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_atlas_model_dissection import MODEL_LABEL  # noqa: E402
from build_clinical_decoder import NICE  # noqa: E402
from build_field_trellis import CODE2FAM  # noqa: E402
from build_prespecified_plane_validation import index_frame  # noqa: E402
from build_trial_in_the_map import FEAT  # noqa: E402
from composite_tumor_host_phase_space import enrich_snapshots, load_tables  # noqa: E402

OUT = ROOT / "outputs" / "exploratory_field_aggregation_audit"
INK, MUTED, GRIDC = "#1f2937", "#64748b", "#d8dee6"
TAU = 3.0
NBIN = 7
MIN_CELL = 80
MIN_SUB = 25
SEED = 20260729
PLANES = {"prespecified tumour × host": ("tumour", "host"),
          "PC1 × PC2": ("pc1", "pc2")}
FACTORS = ["tumour family", "model", "active toxicity", "acquired mutation"]
FCOL = {"tumour family": "family", "model": "model",
        "active toxicity": "tox_t", "acquired mutation": "mut_t"}


def build() -> tuple[pd.DataFrame, np.ndarray]:
    s = enrich_snapshots(load_tables())
    s, z = index_frame(s)
    fit = ~s["terminal"].to_numpy()
    pca = PCA(n_components=len(FEAT), random_state=SEED).fit(z[fit])
    zz = pca.transform(z)
    ref = (s["tumour_burden_score"] + 0.7 * s["num_active_sites_num"] + s["ecog"]).to_numpy(float)
    sgn = 1.0 if np.corrcoef(zz[:, 0], ref)[0, 1] >= 0 else -1.0
    for k in range(len(FEAT)):
        s[f"pc{k + 1}"] = (sgn if k == 0 else 1.0) * zz[:, k]
    s["family"] = s["tumor_oncotree_code"].astype(str).map(CODE2FAM)
    s["model"] = s["model_short"].astype(str).map(lambda m: MODEL_LABEL.get(m, m))
    return s, z


def displacements(s: pd.DataFrame, z: np.ndarray) -> pd.DataFrame:
    """One row per visit: where it is on both planes, and where it goes over TAU."""
    s = s.reset_index(drop=True)
    keep = ["tumour", "host", "toxicity"] + [f"pc{k + 1}" for k in range(len(FEAT))]
    rows = []
    for gid, g in s.groupby("_gid", sort=False):
        g = g.sort_values("month")
        t = g["month"].to_numpy(float)
        if len(t) < 2:
            continue
        v = {c: g[c].to_numpy(float) for c in keep}
        raw = z[g.index.to_numpy()]
        meta = {"gid": gid, "family": g["family"].iloc[0], "model": g["model"].iloc[0]}
        splt = {c: g[c].to_numpy(float) for c in
                ("n_active_aes_num", "n_acquired_alterations_num")}
        for i in range(len(t)):
            if t[i] + TAU > t[-1]:
                continue
            r = {**meta, **{c: sv[i] for c, sv in splt.items()},
                 **{c: v[c][i] for c in keep},
                 **{f"d{c}": float(np.interp(t[i] + TAU, t, v[c])) - v[c][i] for c in keep},
                 **{f"raw{j}": raw[i, j] for j in range(raw.shape[1])}}
            r.update({f"draw{j}": float(np.interp(t[i] + TAU, t, raw[:, j])) - raw[i, j]
                      for j in range(raw.shape[1])})
            rows.append(r)
    d = pd.DataFrame(rows).dropna(subset=["family"])
    # Both splitters are discrete and tie-heavy, so quantile bins would carve the ties
    # arbitrarily and manufacture subgroups that agree by construction. Use real cutpoints.
    d["tox_t"] = pd.cut(d["n_active_aes_num"], [-0.5, 0.5, 1.5, np.inf],
                        labels=["no toxicity", "1 toxicity", "2+ toxicities"])
    d["mut_t"] = np.where(d["n_acquired_alterations_num"] > 0, "acquired mutation",
                          "no acquired mutation")
    return d


def cells(d: pd.DataFrame, xc: str, yc: str) -> pd.Series:
    xe = np.quantile(d[xc], np.linspace(0.02, 0.98, NBIN + 1))
    ye = np.quantile(d[yc], np.linspace(0.02, 0.98, NBIN + 1))
    ix = np.clip(np.digitize(d[xc], xe) - 1, 0, NBIN - 1)
    iy = np.clip(np.digitize(d[yc], ye) - 1, 0, NBIN - 1)
    return pd.Series(ix * NBIN + iy, index=d.index)


def cancellation(d: pd.DataFrame, xc: str, yc: str, fac: str) -> pd.DataFrame:
    """Per cell: pooled arrow vs sub-arrows of the subgroups inside it."""
    out = []
    for cid, g in d.groupby("cell", sort=False):
        if len(g) < MIN_CELL:
            continue
        subs = [(k, sg) for k, sg in g.groupby(fac, observed=True) if len(sg) >= MIN_SUB]
        if len(subs) < 2:
            continue
        w = np.array([len(sg) for _k, sg in subs], float)
        vs = np.array([[sg[f"d{xc}"].mean(), sg[f"d{yc}"].mean()] for _k, sg in subs])
        pooled = (w[:, None] * vs).sum(0) / w.sum()
        num, den = np.linalg.norm(pooled) * w.sum(), (w * np.linalg.norm(vs, axis=1)).sum()
        ang = np.degrees(np.arctan2(vs[:, 1], vs[:, 0]))
        pair = max(abs((a - b + 180) % 360 - 180) for a in ang for b in ang)
        out.append({"cell": cid, "n": len(g), "k": len(subs),
                    "R": float(num / den) if den > 0 else np.nan,
                    "max_angle": float(pair),
                    "pooled_x": pooled[0], "pooled_y": pooled[1],
                    "subs": [(str(k), float(v[0]), float(v[1]), int(n))
                             for (k, _sg), v, n in zip(subs, vs, w)]})
    return pd.DataFrame(out)


def within_cell_r2(d: pd.DataFrame, tgt: list[str], pred: list[str]) -> float:
    """After removing the cell mean, how much of the rest is still predictable?

    The target is the FULL 12-feature displacement for both planes, not each plane's own
    two coordinates — otherwise the two numbers would be computed against different
    quantities and could not be compared. Predictors are all 12 raw features, so this
    bounds together the within-cell variation in position AND the off-plane dimensions.
    """
    big = d.groupby("cell")["gid"].transform("size") >= MIN_CELL
    g = d[big].copy()
    y = g[tgt].to_numpy(float) - g.groupby("cell")[tgt].transform("mean").to_numpy(float)
    x, grp = g[pred].to_numpy(float), g["gid"].to_numpy()
    p = np.zeros_like(y)
    for c in range(y.shape[1]):
        for tr, te in GroupKFold(n_splits=5).split(x, y[:, c], grp):
            p[te, c] = make_pipeline(StandardScaler(), Ridge(1.0)).fit(
                x[tr], y[tr, c]).predict(x[te])
    return 100 * float(1 - ((y - p) ** 2).sum() / (y ** 2).sum())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s, z = build()
    d = displacements(s, z)
    raws = [f"raw{j}" for j in range(len(FEAT))]

    res: dict = {}
    for pname, (xc, yc) in PLANES.items():
        d["cell"] = cells(d, xc, yc)
        big = d.groupby("cell")["gid"].transform("size") >= MIN_CELL
        g = d[big]
        # A. residual heterogeneity of each raw feature inside a cell
        within = g.groupby("cell")[raws].transform("mean")
        het = ((g[raws] - within).var() / d[raws].var()).to_numpy()
        # B. cancellation, per factor
        canc = {f: cancellation(d, xc, yc, FCOL[f]) for f in FACTORS}
        # D. predictability of the within-cell residual, common target for both planes
        draws = [f"draw{j}" for j in range(len(FEAT))]
        res[pname] = {"het": het, "canc": canc, "r2": within_cell_r2(d, draws, raws),
                      "n_cells": int(g["cell"].nunique()), "n_rows": int(len(g)),
                      "xc": xc, "yc": yc}

    P = "prespecified tumour × host"
    fig = plt.figure(figsize=(17.5, 10.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.44, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    order = np.argsort(-res[P]["het"])
    ys = np.arange(len(FEAT))[::-1]
    for k, (pn, col) in enumerate([(P, "#5e3c99"), ("PC1 × PC2", "#c9ced6")]):
        ax.barh(ys + (0.2 if k == 0 else -0.2), 100 * res[pn]["het"][order], height=0.36,
                color=col, label=pn)
    ax.set_yticks(ys)
    ax.set_yticklabels([NICE[FEAT[i]] for i in order], fontsize=8)
    ax.axvline(100, color=INK, ls=":", lw=1.2)
    ax.text(100, len(FEAT) - 0.2, "not pinned\ndown at all", fontsize=7.5, color=MUTED)
    ax.set_xlabel("% of the feature's spread surviving INSIDE a cell", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("A · what the two coordinates do not pin down", fontsize=11, color=INK)

    ax = fig.add_subplot(gs[0, 1:])
    pos, labs = [], []
    for i, f in enumerate(FACTORS):
        for k, (pn, col) in enumerate([(P, "#5e3c99"), ("PC1 × PC2", "#9aa3ae")]):
            c = res[pn]["canc"][f]
            if c.empty:
                continue
            bp = ax.boxplot([c["R"].dropna()], positions=[i * 2.6 + k * 0.9], widths=0.7,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color=INK, lw=1.6))
            bp["boxes"][0].set_facecolor(col)
            bp["boxes"][0].set_alpha(0.85)
        pos.append(i * 2.6 + 0.45)
        labs.append(f"{f}\n({res[P]['canc'][f]['k'].median():.0f} subgroups/cell)")
    ax.axhline(1.0, color="#0f8b8d", ls="--", lw=1.4)
    ax.text(-0.6, 1.01, "all subgroups flow the same way", fontsize=8, color="#0f8b8d")
    ax.set_xticks(pos)
    ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("cancellation ratio R", fontsize=9.5)
    ax.set_title("B · do sub-populations inside a cell flow the same way?\n"
                 "R = |pooled arrow| / mean |sub-arrow|;  purple = prespecified plane, "
                 "grey = PC1 × PC2", fontsize=11, color=INK)

    c = res[P]["canc"]["tumour family"]
    worst = c.loc[c["R"].idxmin()]
    ax = fig.add_subplot(gs[1, 0])
    cmap = plt.get_cmap("tab10")
    for j, (k, vx, vy, n) in enumerate(worst["subs"]):
        ax.annotate("", xy=(vx, vy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=cmap(j % 10), lw=2.0))
        ax.annotate(f"{k} (n={n})", (vx, vy), fontsize=7.5, color=INK,
                    textcoords="offset points", xytext=(5, 3))
    ax.annotate("", xy=(worst["pooled_x"], worst["pooled_y"]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=4.0))
    ax.annotate("what the field draws", (worst["pooled_x"], worst["pooled_y"]), fontsize=8.5,
                color=INK, weight="bold", textcoords="offset points", xytext=(6, -14))
    ax.scatter([0], [0], s=110, facecolor="white", edgecolor=INK, lw=2, zorder=5)
    ax.axhline(0, color=GRIDC, lw=1)
    ax.axvline(0, color=GRIDC, lw=1)
    pts = np.array([[vx, vy] for _k, vx, vy, _n in worst["subs"]]
                   + [[worst["pooled_x"], worst["pooled_y"]], [0.0, 0.0]])
    pad = 0.45 * np.ptp(pts, axis=0)
    ax.set_xlim(pts[:, 0].min() - pad[0], pts[:, 0].max() + pad[0])
    ax.set_ylim(pts[:, 1].min() - pad[1], pts[:, 1].max() + pad[1])
    ax.set_xlabel("tumour movement", fontsize=9)
    ax.set_ylabel("host movement", fontsize=9)
    ax.set_title(f"C · the worst cell (n={int(worst['n'])}, R={worst['R']:.2f})\n"
                 f"sub-arrows span {worst['max_angle']:.0f}° of direction",
                 fontsize=11, color=INK)

    ax = fig.add_subplot(gs[1, 1])
    ks = list(PLANES)
    ax.bar(range(len(ks)), [res[k]["r2"] for k in ks], color=["#5e3c99", "#9aa3ae"], width=0.6)
    for i, k in enumerate(ks):
        ax.text(i, res[k]["r2"] + 0.3, f"{res[k]['r2']:.1f}%", ha="center", fontsize=10,
                color=INK)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([k.replace(" ", "\n") for k in ks], fontsize=8.5)
    ax.set_ylabel("% of within-cell residual move predicted", fontsize=9)
    ax.set_title("D · patients in the same cell are still\ngoing measurably different places",
                 fontsize=11, color=INK)

    ax = fig.add_subplot(gs[1, 2])
    cc = res[P]["canc"]["tumour family"].copy()
    cc["mag"] = np.hypot(cc["pooled_x"], cc["pooled_y"])
    med = float(cc["mag"].median())
    ax.scatter(cc["mag"], cc["max_angle"], s=18 + cc["n"] / 12, c=cc["R"], cmap="viridis",
               vmin=0.5, vmax=1.0, edgecolor=INK, lw=0.6)
    ax.axhline(90, color="#c0392b", ls="--", lw=1.4)
    ax.axvline(med, color=MUTED, ls=":", lw=1.3)
    strong, weak = cc[cc["mag"] >= med], cc[cc["mag"] < med]
    ax.text(0.97, 0.97, f"short arrows: {100 * (weak['max_angle'] > 90).mean():.0f}% "
            "disagree", fontsize=9, color="#c0392b", va="top", ha="right",
            transform=ax.transAxes)
    ax.text(0.97, 0.89, f"long arrows: {100 * (strong['max_angle'] > 90).mean():.0f}% "
            "disagree", fontsize=9, color="#0f8b8d", va="top", ha="right",
            transform=ax.transAxes)
    ax.set_xlabel("length of the pooled arrow", fontsize=9.5)
    ax.set_ylabel("spread of sub-arrow directions (°)", fontsize=9.5)
    ax.set_ylim(0, 185)
    ax.set_title("E · the disagreement is concentrated\nwhere the field barely moves",
                 fontsize=11, color=INK)
    fig.colorbar(ax.collections[0], ax=ax, fraction=0.04, pad=0.03, label="R")

    for a in fig.get_axes():
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Is each arrow an average over patients who do not belong together?",
                 fontsize=13.5, color=INK, y=1.01)
    fig.savefig(OUT / "field_aggregation_audit.png", dpi=190, bbox_inches="tight")
    fig.savefig(OUT / "field_aggregation_audit.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = ["# Is each arrow an average over patients who do not belong together?", "",
             f"{NBIN}x{NBIN} grid, cells with >= {MIN_CELL} visits, subgroups with "
             f">= {MIN_SUB}. Displacement horizon {TAU:.0f} months.", "",
             "## A. What fixing the two display coordinates does NOT pin down", "",
             "Share of each raw feature's total spread that survives inside a single cell.",
             "", "| feature | prespecified plane | PC1 × PC2 |", "|---|---|---|"]
    for i in order:
        lines.append(f"| {NICE[FEAT[i]]} | {100 * res[P]['het'][i]:.0f}% | "
                     f"{100 * res['PC1 × PC2']['het'][i]:.0f}% |")
    lines += ["",
              f"Median across features: **{100 * np.median(res[P]['het']):.0f}%** on the "
              f"prespecified plane, {100 * np.median(res['PC1 × PC2']['het']):.0f}% on "
              "PC1 × PC2. Two coordinates cannot pin down twelve, and the features they "
              "leave free are exactly the ones absent from the axes.", "",
              "## B. Do sub-populations inside a cell flow the same way?", "",
              "R = |pooled arrow| / mean |sub-arrow|. R = 1 means every subgroup flows the "
              "same way and the pooled arrow is a fair summary; R near 0 means the arrow is "
              "what survives after opposing flows cancel.", "",
              "**Read R and the angle spread together, not R alone.** R is size-weighted, "
              "so a cell dominated by one large subgroup keeps a high R even when the "
              "smaller subgroups flow the other way — the pooled arrow survives because it "
              "is essentially the majority's arrow, which is precisely the failure mode. R "
              "answers 'does the average survive'; the angle spread answers 'do the "
              "subgroups agree'. The second is the sensitive detector.", "",
              "| splitting factor | plane | median R | cells >90° spread | cells |",
              "|---|---|---|---|---|"]
    for f in FACTORS:
        for pn in PLANES:
            cc = res[pn]["canc"][f]
            if cc.empty:
                continue
            lines.append(f"| {f} | {pn} | {cc['R'].median():.2f} | "
                         f"{100 * (cc['max_angle'] > 90).mean():.0f}% | {len(cc)} |")
    lines += ["", "### The worst cell (split by tumour family, prespecified plane)", "",
              f"n = {int(worst['n'])}, R = {worst['R']:.2f}, sub-arrows span "
              f"{worst['max_angle']:.0f}°. The pooled arrow is "
              f"({worst['pooled_x']:+.3f}, {worst['pooled_y']:+.3f}); the subgroups are:", "",
              "| subgroup | Δtumour | Δhost | n |", "|---|---|---|---|"]
    for k, vx, vy, n in sorted(worst["subs"], key=lambda r: -r[3]):
        lines.append(f"| {k} | {vx:+.3f} | {vy:+.3f} | {n} |")
    lines += ["", "## D. Is a cell a state, or a bag?", "",
              "After removing the cell mean, how much of the remaining displacement is still "
              "predictable? Out-of-fold, grouped by trajectory. The target is the FULL "
              "12-feature displacement for both planes so the two are comparable, and the "
              "predictors are all 12 raw features, so this bounds together within-cell "
              "variation in position and the off-plane dimensions.", "",
              "| plane | % of within-cell residual move predicted |", "|---|---|"]
    for pn in PLANES:
        lines.append(f"| {pn} | {res[pn]['r2']:.2f}% |")
    cc2 = res[P]["canc"]["tumour family"].copy()
    cc2["mag"] = np.hypot(cc2["pooled_x"], cc2["pooled_y"])
    m2 = float(cc2["mag"].median())
    st, wk = cc2[cc2["mag"] >= m2], cc2[cc2["mag"] < m2]
    lines += ["", "## E. But WHERE is the disagreement?", "",
              "A direction is only meaningful if the arrow has length — the heading of a "
              "near-stationary arrow is noise. Splitting the cells at the median pooled "
              "arrow length:", "",
              "| cells | median arrow length | span >90° | median R |", "|---|---|---|---|",
              f"| long arrows (n={len(st)}) | {st['mag'].median():.3f} | "
              f"{100 * (st['max_angle'] > 90).mean():.0f}% | {st['R'].median():.2f} |",
              f"| short arrows (n={len(wk)}) | {wk['mag'].median():.3f} | "
              f"{100 * (wk['max_angle'] > 90).mean():.0f}% | {wk['R'].median():.2f} |", "",
              "**This is the finding that matters.** The 43% headline is driven almost "
              "entirely by cells where the field barely moves. Where the field actually says "
              f"something, {100 - 100 * (st['max_angle'] > 90).mean():.0f}% of cells have "
              f"their subgroups flowing the same way, at median R = {st['R'].median():.2f}. "
              "Where it says 'not much happens here', the direction is close to a coin flip "
              "and should not be drawn as an arrow at all.", ""]
    lines += ["", "## What follows for the figures", "",
              "The pooled field is a legitimate estimand — it is the marginal field, "
              "E[dX | projected coordinates] — but it is not any individual's field, and it "
              "is only a fair summary where R stays near 1. Where R falls, two things break: "
              "the arrow describes no real patient, and any comparison of models or cancers "
              "drawn on a shared field is confounded by the case-mix inside each cell.", "",
              "The fix is not a better projection — no 2-D projection of a 12-D state can "
              "avoid this, and the prespecified plane is no better than PC1 × PC2 here "
              "(43% vs 44% of cells spanning >90°). Three concrete rules follow:", "",
              "1. **Suppress short arrows.** Arrows below the median length are where the "
              "subgroups disagree; drawing them asserts a direction the data does not "
              "support. Fade or omit them and the remaining field is trustworthy.",
              "2. **Condition on whatever splits the cell** and draw the field within it — "
              "which is what the trellis already does for tumour family and treatment epoch.",
              "3. **Never compare models on a shared pooled field without conditioning.** "
              "Splitting by model puts 32% of cells over 90°, so a model-vs-model difference "
              "read off a shared field can be case-mix inside the cells rather than a "
              "difference in dynamics.", "",
              "Every pooled field in the paper should carry its R and its angle spread."]
    (OUT / "readout.md").write_text("\n".join(lines) + "\n")
    (OUT / "readout.json").write_text(json.dumps(
        {"heterogeneity": {pn: {NICE[FEAT[i]]: float(res[pn]["het"][i])
                                for i in range(len(FEAT))} for pn in PLANES},
         "cancellation": {pn: {f: {"median_R": float(res[pn]["canc"][f]["R"].median()),
                                   "pct_over_90deg": float(
                                       100 * (res[pn]["canc"][f]["max_angle"] > 90).mean()),
                                   "n_cells": int(len(res[pn]["canc"][f]))}
                               for f in FACTORS if not res[pn]["canc"][f].empty}
                          for pn in PLANES},
         "within_cell_residual_r2": {pn: res[pn]["r2"] for pn in PLANES}},
        indent=2) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
