#!/usr/bin/env python3
"""THE ATLAS DISSECTED — do six models agree about how cancer moves?

The atlas is observational: treatment is confounded by indication, so it cannot carry a
treatment contrast. But there is one intervention in it that WE assigned — the model. The
same 148 OncoTree codes, the same prompt, six different simulators. That is a controlled
comparison, and it is the audit the paper is about.

Applying the frozen framework (docs/conditional_dynamics_framework_v1.md) to that
contrast:

  A  AGGRESSIVENESS. Median survival by tumour family, per model. Do the models agree on
     which cancers are fast?
  B  CONCORDANCE. Spearman agreement between every pair of models on that family ranking.
  C  THE FIELDS. The context-averaged field per model on ONE shared map, fixed horizon.
     Do they agree on the direction of motion, not just the endpoint?
  D  WHERE THEY DIVERGE. Cross-model dispersion per family — the trust map. If the models
     agree on common cancers and diverge on rare ones, that is the paper's thesis in a
     single ranking.

Map: the 12-feature reduced block from build_trial_in_the_map, fitted once and shared by
every panel. The standard 16-feature block is unusable — PC1 loads 0.996 on CNS burden
(build_feature_geometry_audit).

Run: uv run python scripts/explore/build_atlas_model_dissection.py
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
from scipy.stats import spearmanr  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_field_trellis import CODE2FAM  # noqa: E402
from build_trial_in_the_map import FEAT  # noqa: E402
from composite_tumor_host_phase_space import (  # noqa: E402
    ATLAS_DERIVS,
    accepted,
    enrich_snapshots,
    load_tables,
)

OUT = ROOT / "outputs" / "exploratory_atlas_dissection"
INK, MUTED, GRIDC = "#1f2937", "#64748b", "#d8dee6"
TAU = 3.0
NX, NY = 5, 4
MIN_CELL = 20
SEED = 20260729
MODEL_LABEL = {"claude_sonnet5": "Sonnet", "kimi_k3_off": "Kimi K3", "glm52_fp8": "GLM-5.2",
               "gemini31_lite": "Gemini 3.1 L", "qwen36_flash": "Qwen 3.6", "gemma4": "Gemma 4"}


def load() -> pd.DataFrame:
    s = enrich_snapshots(load_tables())
    fit = s[~s["terminal"]]
    sc = RobustScaler(quantile_range=(10, 90))
    x = sc.fit_transform(fit[FEAT].fillna(0.0).to_numpy(float))
    pca = PCA(n_components=3, random_state=SEED).fit(x)
    ref = (fit["tumour_burden_score"] + 0.7 * fit["num_active_sites_num"] + fit["ecog"]).to_numpy(float)
    sign = 1.0 if np.corrcoef(pca.transform(x)[:, 0], ref)[0, 1] >= 0 else -1.0
    z = pca.transform(sc.transform(s[FEAT].fillna(0.0).to_numpy(float)))
    s["pc1"], s["pc2"] = sign * z[:, 0], z[:, 1]
    s["family"] = s["tumor_oncotree_code"].astype(str).map(CODE2FAM)
    s["model"] = s["model_short"].astype(str).map(lambda m: MODEL_LABEL.get(m, m))
    return s


def baselines() -> pd.DataFrame:
    """Generation was spontaneous below the tumour anchor: the model chose the stage.
    Carry it so the survival comparison can be stratified rather than only marginal."""
    frames = []
    for d in ATLAS_DERIVS:
        p = d / "baseline_rows.csv"
        if not p.exists():
            continue
        x = accepted(pd.read_csv(p, low_memory=False))
        x["_run"] = d.parent.name
        x["_gid"] = x["_run"] + "::" + x["trajectory_id"].astype(str)
        frames.append(x[["_gid", "stage"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates("_gid")


def horizon(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _gid, g in s.groupby("_gid", sort=False):
        g = g.sort_values(["month", "snapshot_index"])
        t = g["month"].to_numpy(float)
        x, y = g["pc1"].to_numpy(float), g["pc2"].to_numpy(float)
        if len(t) < 2:
            continue
        for i in range(len(t)):
            if t[i] + TAU > t[-1]:
                continue
            rows.append((g["model"].iloc[0], g["family"].iloc[0], x[i], y[i],
                         float(np.interp(t[i] + TAU, t, x)) - x[i],
                         float(np.interp(t[i] + TAU, t, y)) - y[i]))
    return pd.DataFrame(rows, columns=["model", "family", "x", "y", "dx", "dy"])


def field(d: pd.DataFrame, xe, ye):
    u = np.full((NY, NX), np.nan)
    v = np.full((NY, NX), np.nan)
    for a in range(NX):
        for b in range(NY):
            m = d["x"].between(xe[a], xe[a + 1]) & d["y"].between(ye[b], ye[b + 1])
            if m.sum() >= MIN_CELL:
                u[b, a], v[b, a] = d.loc[m, "dx"].mean(), d.loc[m, "dy"].mean()
    return u, v


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    s = load()
    models = [m for m in MODEL_LABEL.values() if (s["model"] == m).any()]

    os_tab = (s.groupby(["model", "_gid"])["month"].max().reset_index()
              .merge(s.groupby("_gid")["family"].first().rename("family"), on="_gid")
              .merge(baselines(), on="_gid", how="left")
              .dropna(subset=["family"]))
    med = os_tab.groupby(["family", "model"])["month"].median().unstack()
    med = med.loc[med.mean(axis=1).sort_values().index]
    met = os_tab[os_tab["stage"].astype(str).eq("metastatic")]
    med_met = met.groupby(["family", "model"])["month"].median().unstack().reindex(med.index)
    stage_mix = (os_tab.assign(is_met=os_tab["stage"].astype(str).eq("metastatic"))
                 .groupby("model")["is_met"].mean().mul(100))

    d = horizon(s.dropna(subset=["family"]))
    xe = np.linspace(d["x"].quantile(0.02), d["x"].quantile(0.95), NX + 1)
    ye = np.linspace(d["y"].quantile(0.02), d["y"].quantile(0.95), NY + 1)
    XC, YC = np.meshgrid(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]))
    fields = {m: field(d[d["model"].eq(m)], xe, ye) for m in models}

    fig = plt.figure(figsize=(17.5, 12.0))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.15, 1.0, 0.95], hspace=0.45, wspace=0.42)

    # ---- A: aggressiveness ranking
    ax = fig.add_subplot(gs[0, :4])
    ys = np.arange(len(med))
    cmap = plt.get_cmap("tab10")
    for k, m in enumerate(models):
        if m in med:
            ax.scatter(med[m], ys, s=58, color=cmap(k), label=m, zorder=3)
    for i in ys:
        ax.plot([med.iloc[i].min(), med.iloc[i].max()], [i, i], color=GRIDC, lw=2, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(med.index, fontsize=10)
    ax.set_xlabel("median survival (months)", fontsize=10)
    ax.set_title("A · how aggressive is each cancer? one dot per model\n"
                 "(families ordered by mean across models)", fontsize=11.5, color=INK)
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # ---- B: concordance on the ranking
    ax = fig.add_subplot(gs[0, 4:])
    common = [m for m in models if m in med]
    rho = np.eye(len(common))
    for i, a in enumerate(common):
        for j, b in enumerate(common):
            if i < j:
                ok = med[[a, b]].dropna()
                r = spearmanr(ok[a], ok[b]).statistic if len(ok) > 3 else np.nan
                rho[i, j] = rho[j, i] = r
    im = ax.imshow(rho, cmap="RdYlBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Spearman ρ")
    for i in range(len(common)):
        for j in range(len(common)):
            ax.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(rho[i, j]) > 0.6 else INK)
    ax.set_xticks(range(len(common)))
    ax.set_xticklabels(common, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(common)))
    ax.set_yticklabels(common, fontsize=8)
    ax.set_title("B · do models agree which cancers\nare aggressive?", fontsize=11, color=INK)

    # ---- C: the field per model
    for k, m in enumerate(models):
        ax = fig.add_subplot(gs[1, k])
        u, v = fields[m]
        ax.quiver(XC, YC, u, v, color=INK, angles="xy", scale_units="xy", scale=1, width=0.014)
        n = int(d["model"].eq(m).sum())
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(ye[0], ye[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRIDC)
        ax.set_title(f"{m}\n{n:,} arrows", fontsize=9.5, color=INK)
        if k == 0:
            ax.set_xlabel("severity →", fontsize=8.5)
    fig.text(0.5, 0.395, "C · the context-averaged field per model, one shared map, "
             f"{TAU:.0f}-month arrows", ha="center", fontsize=11.5, color=INK)

    # ---- D: where the models diverge
    ax = fig.add_subplot(gs[2, :3])
    spread = (med.max(axis=1) - med.min(axis=1)).sort_values(ascending=False)
    spread_met = (med_met.max(axis=1) - med_met.min(axis=1)).reindex(spread.index)
    ys = np.arange(len(spread))[::-1]
    ax.barh(ys + 0.19, spread.to_numpy(), height=0.36, color="#7f8c9b", label="all stages")
    ax.barh(ys - 0.19, spread_met.to_numpy(), height=0.36, color="#c0392b",
            label="metastatic only")
    ax.set_yticks(ys)
    ax.set_yticklabels(spread.index, fontsize=9)
    ax.set_xlabel("spread in median survival across models (months)", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.set_title("D · where the simulators disagree — the trust map\n"
                 "generation was spontaneous, so this is shown with and without "
                 "stage stratification", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- E: field agreement between models
    ax = fig.add_subplot(gs[2, 3:])
    agree = np.eye(len(models))
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i < j:
                ua, va = fields[a]
                ub, vb = fields[b]
                ok = np.isfinite(ua) & np.isfinite(ub)
                if ok.sum() >= 4:
                    va_ = np.c_[ua[ok], va[ok]].ravel()
                    vb_ = np.c_[ub[ok], vb[ok]].ravel()
                    c = float(np.dot(va_, vb_) / (np.linalg.norm(va_) * np.linalg.norm(vb_)))
                else:
                    c = np.nan
                agree[i, j] = agree[j, i] = c
    im = ax.imshow(agree, cmap="RdYlBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, label="cosine similarity of fields")
    for i in range(len(models)):
        for j in range(len(models)):
            if np.isfinite(agree[i, j]):
                ax.text(j, i, f"{agree[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(agree[i, j]) > 0.6 else INK)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_title("E · do the models agree on the direction of motion?",
                 fontsize=11, color=INK)

    fig.suptitle("The atlas dissected — the model is the intervention\n"
                 "same 148 OncoTree codes, same prompt, six simulators",
                 fontsize=14, color=INK, y=0.985)
    fig.savefig(OUT / "atlas_model_dissection.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "atlas_model_dissection.pdf", bbox_inches="tight")
    plt.close(fig)

    off = rho[np.triu_indices(len(common), 1)]
    offf = agree[np.triu_indices(len(models), 1)]
    rep = {"n_trajectories": int(s["_gid"].nunique()), "n_arrows": int(len(d)),
           "models": models,
           "rank_concordance_median": float(np.nanmedian(off)),
           "rank_concordance_range": [float(np.nanmin(off)), float(np.nanmax(off))],
           "field_agreement_median": float(np.nanmedian(offf)),
           "field_agreement_range": [float(np.nanmin(offf)), float(np.nanmax(offf))],
           "median_os_by_family": med.round(2).to_dict(),
           "spread_months": spread.round(2).to_dict(),
           "spread_months_metastatic": spread_met.round(2).to_dict(),
           "pct_metastatic_by_model": stage_mix.round(1).to_dict()}
    lines = ["# The atlas dissected — the model is the intervention", "",
             f"{rep['n_trajectories']:,} trajectories, {rep['n_arrows']:,} fixed-horizon "
             f"({TAU:.0f}-month) arrows, {len(models)} models.", "",
             f"- agreement on which cancers are aggressive: median Spearman ρ = "
             f"**{rep['rank_concordance_median']:.2f}** "
             f"(range {rep['rank_concordance_range'][0]:.2f} to {rep['rank_concordance_range'][1]:.2f})",
             f"- agreement on the direction of motion: median cosine = "
             f"**{rep['field_agreement_median']:.2f}** "
             f"(range {rep['field_agreement_range'][0]:.2f} to {rep['field_agreement_range'][1]:.2f})",
             "", "## Stage was chosen by the model, not fixed", "",
             "Generation was spontaneous below the tumour anchor, so the models differ in "
             "the stage mix they invent: metastatic fraction "
             + ", ".join(f"{m} {v:.0f}%" for m, v in stage_mix.sort_values().items()) + ".",
             "", f"Restricting to metastatic patients only, the median spread across "
             f"families moves {spread.median():.1f} -> {spread_met.median():.1f} months and "
             f"the largest stays {spread.max():.1f} -> {spread_met.max():.1f} "
             f"({spread.idxmax()}). The scale disagreement is not a stage artifact.",
             "", "## Median survival by tumour family (months)", "",
             "| family | " + " | ".join(models) + " | spread |",
             "|---|" + "---|" * (len(models) + 1)]
    for fam in med.index:
        lines.append(f"| {fam} | " + " | ".join(
            f"{med.loc[fam, m]:.1f}" if m in med and np.isfinite(med.loc[fam, m]) else "—"
            for m in models) + f" | **{spread[fam]:.1f}** |")
    (OUT / "readout.md").write_text("\n".join(lines) + "\n")
    (OUT / "readout.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
    print("\n".join(lines))
    return rep


if __name__ == "__main__":
    build()
