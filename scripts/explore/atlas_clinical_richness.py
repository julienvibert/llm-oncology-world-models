#!/usr/bin/env python3
"""Clinical-richness pack — auditing the fine detail of LLM-imagined oncology trajectories.

Four under-used seams the atlas already contains, none previously characterized:
  1. RESISTANCE-EVOLUTION map  — mechanism_bucket x primary driver + top resistance genes
  2. TOXICITY fingerprint       — CTCAE grade 1-5 + AE categories, by model
  3. TREATMENT-JOURNEY flow     — 1L->2L->3L regimen-class transitions
  4. ENDPOINT atlas             — imagined OS by tumour x model; PFS by line (realism signature)

Synthetic atlas -> git-ok. Descriptive characterization (structure), directional not calibrated.
Run: uv run python scripts/explore/atlas_clinical_richness.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs" / "exploratory_atlas_clinical_richness"
DERIVS = [ROOT / "runs" / r / "derivation" for r in (
    "pillar1_anchored_atlas_tranche1_v4_n5_live01", "pillar1_anchored_atlas_gemma4_n5_live01",
    "sonnet5_mini_atlas_n100_2026_07_08", "sonnet5_full_atlas_topup_remaining128_n5_2026_07_09")]
MLAB = {"gemini31_lite": "Gemini", "qwen36_flash": "Qwen", "gemma4": "Gemma", "claude_sonnet5": "Sonnet"}
MORDER = ["gemini31_lite", "qwen36_flash", "gemma4", "claude_sonnet5"]


def load(table: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(d / f"{table}.csv", low_memory=False) for d in DERIVS
                      if (d / f"{table}.csv").exists()], ignore_index=True)


def tx_class(joined: str) -> str:
    s = str(joined).lower()
    io = "immune_checkpoint" in s
    chemo = "cytotoxic" in s
    targ = any(k in s for k in ("tki", "targeted", "monoclonal", "parp", "cdk", "her2"))
    if io and chemo:
        return "chemo-IO"
    if io:
        return "IO"
    if targ:
        return "targeted"
    if chemo:
        return "chemo"
    if "endocrine" in s:
        return "endocrine"
    return "other"


def fig_resistance():
    ev = load("event_rows")
    bl = load("baseline_rows")[["trajectory_id", "primary_driver_bucket"]].drop_duplicates("trajectory_id")
    c = ev[ev.event_type == "clonal_evolution_event"].merge(bl, on="trajectory_id", how="left")
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle("LLM-imagined resistance evolution (2,320 clonal events) — realistic mechanism grammar",
                 fontsize=12, fontweight="bold")
    mech = c.mechanism_bucket.value_counts().head(7)
    ax[0].barh(range(len(mech))[::-1], mech.to_numpy(), color="#4C6EB5")
    ax[0].set_yticks(range(len(mech))[::-1])
    ax[0].set_yticklabels([m.replace("_", " ")[:26] for m in mech.index], fontsize=8)
    ax[0].set_title("A. Resistance mechanisms", fontsize=10, loc="left")
    ax[0].set_xlabel("events")
    drivers = c.primary_driver_bucket.value_counts().head(8).index
    mechs = mech.index
    grid = np.zeros((len(mechs), len(drivers)))
    for i, m in enumerate(mechs):
        for j, dr in enumerate(drivers):
            sub = c[c.primary_driver_bucket == dr]
            grid[i, j] = (sub.mechanism_bucket == m).mean() if len(sub) else np.nan
    im = ax[1].imshow(grid, cmap="viridis", aspect="auto")
    ax[1].set_xticks(range(len(drivers)))
    ax[1].set_xticklabels([d.replace("_", " ")[:10] for d in drivers], rotation=40, ha="right", fontsize=7)
    ax[1].set_yticks(range(len(mechs)))
    ax[1].set_yticklabels([m.replace("_", " ")[:22] for m in mechs], fontsize=7)
    ax[1].set_title("B. Mechanism × primary driver (fraction)", fontsize=10, loc="left")
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    genes = c.gene.dropna().value_counts().head(12)
    ax[2].barh(range(len(genes))[::-1], genes.to_numpy(), color="#B23A48")
    ax[2].set_yticks(range(len(genes))[::-1])
    ax[2].set_yticklabels(genes.index, fontsize=8)
    ax[2].set_title("C. Genes driving imagined resistance", fontsize=10, loc="left")
    ax[2].set_xlabel("events")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_resistance_evolution.png", dpi=150)
    plt.close(fig)
    return len(c)


def fig_toxicity():
    ev = load("event_rows")
    t = ev[ev.event_type == "toxicity"].copy()
    t["model"] = t["model_short"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle("Toxicity fingerprint — CTCAE-graded adverse events by model (12,519 events)",
                 fontsize=12, fontweight="bold")
    models = [m for m in MORDER if (t.model == m).any()]
    grades = [1, 2, 3, 4, 5]
    bottom = np.zeros(len(models))
    cols = plt.get_cmap("YlOrRd")(np.linspace(0.2, 0.95, len(grades)))
    for gi, gr in enumerate(grades):
        vals = [((t.model == m) & (t.ctcae_v5_grade == gr)).sum() for m in models]
        ax[0].bar(range(len(models)), vals, bottom=bottom, color=cols[gi], label=f"G{gr}")
        bottom += vals
    ax[0].set_xticks(range(len(models)))
    ax[0].set_xticklabels([MLAB[m] for m in models])
    ax[0].set_ylabel("toxicity events")
    ax[0].set_title("A. AE count by CTCAE grade", fontsize=10, loc="left")
    ax[0].legend(fontsize=8, ncol=5)
    # grade-3+ fraction per model
    frac = [((t.model == m) & (t.ctcae_v5_grade >= 3)).sum() / max((t.model == m).sum(), 1) for m in models]
    ax[1].bar(range(len(models)), frac, color="#8B1a1a")
    for i, v in enumerate(frac):
        ax[1].text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
    ax[1].set_xticks(range(len(models)))
    ax[1].set_xticklabels([MLAB[m] for m in models])
    ax[1].set_ylabel("fraction grade ≥3")
    ax[1].set_title("B. Severe-toxicity fraction", fontsize=10, loc="left")
    cat = t.ae_category.dropna().value_counts().head(10)
    ax[2].barh(range(len(cat))[::-1], cat.to_numpy(), color="#E08214")
    ax[2].set_yticks(range(len(cat))[::-1])
    ax[2].set_yticklabels([c.replace("_", " ")[:24] for c in cat.index], fontsize=7.5)
    ax[2].set_title("C. Top adverse-event categories", fontsize=10, loc="left")
    ax[2].set_xlabel("events")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_toxicity_fingerprint.png", dpi=150)
    plt.close(fig)
    return len(t)


def fig_treatment_journey():
    tr = load("treatment_rows")
    tr["cls"] = tr["component_classes_joined"].map(tx_class)
    # per trajectory, ordered lines
    lines = {}
    for tid, g in tr.sort_values("line_number").groupby("trajectory_id"):
        seq = g.drop_duplicates("line_number")["cls"].tolist()
        lines[tid] = seq
    classes = ["targeted", "chemo-IO", "IO", "chemo", "endocrine", "other"]
    cidx = {c: i for i, c in enumerate(classes)}
    # 1L->2L and 2L->3L transition matrices
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Treatment-journey grammar — imagined line-of-therapy transitions (regimen class)",
                 fontsize=12, fontweight="bold")
    for pi, (a, b, title) in enumerate([(0, 1, "A. 1L → 2L"), (1, 2, "B. 2L → 3L")]):
        M = np.zeros((len(classes), len(classes)))
        for seq in lines.values():
            if len(seq) > b:
                M[cidx.get(seq[a], 5), cidx.get(seq[b], 5)] += 1
        Mn = M / M.sum(1, keepdims=True).clip(min=1)
        im = ax[pi].imshow(Mn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax[pi].set_xticks(range(len(classes)))
        ax[pi].set_xticklabels(classes, rotation=35, ha="right", fontsize=8)
        ax[pi].set_yticks(range(len(classes)))
        ax[pi].set_yticklabels(classes, fontsize=8)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if M[i, j] > 0:
                    ax[pi].text(j, i, f"{Mn[i, j]:.0%}", ha="center", va="center", fontsize=7,
                                color="white" if Mn[i, j] > 0.5 else "#333")
        ax[pi].set_ylabel("from (earlier line)")
        ax[pi].set_xlabel("to (next line)")
        ax[pi].set_title(title, fontsize=10, loc="left")
        plt.colorbar(im, ax=ax[pi], fraction=0.046, label="row fraction")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_treatment_journey.png", dpi=150)
    plt.close(fig)
    return sum(len(s) > 1 for s in lines.values())


def fig_endpoint_atlas():
    e = load("endpoint_rows")
    e = e[e.accepted.astype(bool)] if "accepted" in e else e
    for c in ["os_days", "pfs_line_1_days", "pfs_line_2_days", "pfs_line_3_days"]:
        e[c] = pd.to_numeric(e[c], errors="coerce")
    fig, ax = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [2.2, 1]})
    fig.suptitle("Endpoint atlas — imagined outcomes across cancers and models (directional, not calibrated)",
                 fontsize=12, fontweight="bold")
    # OS by tumour (top-20 by n) x model
    top = e.tumor_oncotree_code.value_counts().head(22).index
    piv = (e[e.tumor_oncotree_code.isin(top)].assign(os_mo=e.os_days / 30.4)
           .pivot_table(index="tumor_oncotree_code", columns="model_short", values="os_mo", aggfunc="median"))
    piv = piv.reindex(columns=[m for m in MORDER if m in piv.columns])
    piv = piv.loc[piv.median(1).sort_values().index]
    im = ax[0].imshow(piv.to_numpy(float), cmap="RdYlGn", aspect="auto",
                      vmin=0, vmax=np.nanpercentile(piv.to_numpy(float), 95))
    ax[0].set_xticks(range(piv.shape[1]))
    ax[0].set_xticklabels([MLAB.get(m, m) for m in piv.columns])
    ax[0].set_yticks(range(piv.shape[0]))
    ax[0].set_yticklabels(piv.index, fontsize=8)
    ax[0].set_title("A. Median imagined OS (months) by tumour × model", fontsize=10, loc="left")
    plt.colorbar(im, ax=ax[0], fraction=0.046, label="median OS (months)")
    # PFS by line (realism: does PFS shrink across lines?)
    models = [m for m in MORDER if (e.model_short == m).any()]
    lines = ["pfs_line_1_days", "pfs_line_2_days", "pfs_line_3_days"]
    for m in models:
        med = [e.loc[e.model_short == m, c].median() / 30.4 for c in lines]
        ax[1].plot([1, 2, 3], med, "-o", label=MLAB[m])
    ax[1].set_xticks([1, 2, 3])
    ax[1].set_xlabel("line of therapy")
    ax[1].set_ylabel("median PFS (months)")
    ax[1].set_title("B. PFS by line — does duration shrink\nwith later lines? (realism signature)", fontsize=10, loc="left")
    ax[1].legend(fontsize=8)
    ax[1].grid(color="#eee")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_endpoint_atlas.png", dpi=150)
    plt.close(fig)
    return len(e)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[richness] resistance events: {fig_resistance()}")
    print(f"[richness] toxicity events:   {fig_toxicity()}")
    print(f"[richness] multi-line courses:{fig_treatment_journey()}")
    print(f"[richness] endpoint rows:     {fig_endpoint_atlas()}")
    print(f"[done] wrote 4 figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
