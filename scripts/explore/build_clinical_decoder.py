#!/usr/bin/env python3
"""THE CLINICAL DECODER — translating movement on the map back into clinical variables.

The map is a linear projection, so nothing is lost in translation: a displacement in PC
space decomposes EXACTLY into changes in the underlying clinical features. This makes the
abstract arrows readable without the map.

Three questions, all answered exactly rather than approximately:

  A  WHAT IS ONE UNIT OF PC1?  Moving one unit along the severity axis corresponds to a
     specific bundle of clinical changes — so many sites, so much burden, so many ECOG
     points. This is the axis decoder, and it is the answer to "what does +0.3 mean".

  B  WHAT ACTUALLY CHANGES, BY CANCER?  Mean 3-month change in each clinical variable, per
     tumour family. Two cancers can push equally hard on the map and be doing completely
     different things clinically — one adding metastatic sites, another eroding
     performance status.

  C  WHAT DOES EACH TREATMENT CHANGE?  The same decomposition per arm in EGFR-mutant
     disease, from the interventional run, where arms are assigned.

Atlas for the cancers (structural facets); interventional run for the treatments.

Run: uv run python scripts/explore/build_clinical_decoder.py
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
from build_atlas_model_dissection import load  # noqa: E402
from build_field_trellis import CODE2FAM  # noqa: E402
from build_trial_in_the_map import ARM_LABEL, FEAT, fit_map, load_run  # noqa: E402
from composite_tumor_host_phase_space import enrich_snapshots, load_tables  # noqa: E402

OUT = ROOT / "outputs" / "exploratory_clinical_decoder"
INK, MUTED, GRIDC = "#1f2937", "#64748b", "#d8dee6"
TAU = 3.0
SEED = 20260729
ARMS = ["osimertinib", "chemo", "pembrolizumab", "supportive"]

NICE = {
    "tumour_burden_score": "total tumour burden",
    "num_active_sites_num": "number of metastatic sites",
    "high_risk_burden": "burden in high-risk sites",
    "max_site_burden": "largest single deposit",
    "liver_burden": "liver involvement",
    "visceral_burden": "visceral involvement",
    "num_active_clones_num": "active tumour clones",
    "n_acquired_alterations_num": "acquired mutations",
    "ecog": "performance status (ECOG)",
    "symptom_ord": "symptom burden",
    "n_active_aes_num": "active toxicities",
    "organ_max": "worst organ impairment",
}
UNIT = {
    "tumour_burden_score": "burden pts", "num_active_sites_num": "sites",
    "high_risk_burden": "burden pts", "max_site_burden": "burden pts",
    "liver_burden": "burden pts", "visceral_burden": "burden pts",
    "num_active_clones_num": "clones", "n_acquired_alterations_num": "mutations",
    "ecog": "ECOG pts", "symptom_ord": "grades", "n_active_aes_num": "events",
    "organ_max": "grades",
}


def deltas(s: pd.DataFrame, keys: list[str], t0: dict | None = None) -> pd.DataFrame:
    """Mean change in every RAW clinical feature over tau months, per stratum."""
    rows = []
    for gid, g in s.groupby("_gid", sort=False):
        g = g.sort_values("month")
        t = g["month"].to_numpy(float)
        if len(t) < 2:
            continue
        anchor = t0.get(gid, -np.inf) if t0 else -np.inf
        vals = {f: g[f].to_numpy(float) for f in FEAT}
        meta = {k: g[k].iloc[0] for k in keys if k in g}
        for i in range(len(t)):
            if t[i] < anchor or t[i] + TAU > t[-1]:
                continue
            r = dict(meta)
            for f in FEAT:
                r[f] = float(np.interp(t[i] + TAU, t, vals[f])) - vals[f][i]
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- A: what one unit of PC1 means, in clinical units
    pca, sc, sign, load_tab = fit_map()
    w = sign * pca.components_[0]
    per_unit = w * sc.scale_          # feature change per +1 of raw PC1
    raw = enrich_snapshots(load_tables())
    alive = raw[~raw["terminal"]]
    z = sc.transform(alive[FEAT].fillna(0.0).to_numpy(float))
    pc1 = z @ w
    iqr = float(np.subtract(*np.percentile(pc1, [75, 25]))) or 1.0
    dec = pd.DataFrame({"feature": FEAT, "per_unit_pc1": per_unit,
                        "per_iqr": per_unit * iqr}).sort_values("per_unit_pc1",
                                                                ascending=False)

    # ---- B: what changes, by cancer
    s = load()
    s["family"] = s["tumor_oncotree_code"].astype(str).map(CODE2FAM)
    fam_d = deltas(s.dropna(subset=["family"]), ["family"])
    fam_mean = fam_d.groupby("family")[FEAT].mean()
    fam_mean = fam_mean.loc[fam_mean["tumour_burden_score"].sort_values(ascending=False).index]

    # ---- C: what each treatment changes
    si, t0, _prog, _death = load_run(pca, sc, sign)
    si = si[si["context"].eq("EGFR_1L")]
    arm_d = deltas(si, ["arm"], t0=t0)
    arm_mean = arm_d.groupby("arm")[FEAT].mean().reindex(ARMS)

    fig = plt.figure(figsize=(18.0, 10.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.0, 1.55],
                          hspace=0.34, wspace=0.26)

    ax = fig.add_subplot(gs[0, 0])
    ys = np.arange(len(dec))[::-1]
    ax.barh(ys, dec["per_unit_pc1"], color="#5e3c99")
    for i, (f, v) in enumerate(zip(dec["feature"], dec["per_unit_pc1"])):
        ax.text(v + 0.02, ys[i], f"{v:+.2f} {UNIT[f]}", va="center", fontsize=8, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([NICE[f] for f in dec["feature"]], fontsize=8.5)
    ax.set_xlabel("change per +1 unit of the severity axis", fontsize=9.5)
    ax.set_title("A · what one unit of PC1 actually means\n"
                 "moving right by 1 = this bundle of clinical changes",
                 fontsize=11.5, color=INK)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    m = fam_mean[FEAT].to_numpy()
    lim = float(np.abs(m).max())
    im = ax.pcolormesh(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=ax, fraction=0.03, label=f"mean change over {TAU:.0f} months")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if abs(m[i, j]) > 0.05 * lim:
                ax.text(j + 0.5, i + 0.5, f"{m[i, j]:+.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(m[i, j]) > 0.55 * lim else INK)
    ax.set_xticks(np.arange(len(FEAT)) + 0.5)
    ax.set_xticklabels([NICE[f] for f in FEAT], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(fam_mean)) + 0.5)
    ax.set_yticklabels(fam_mean.index, fontsize=9)
    ax.set_title("B · what actually changes, by cancer — mean 3-month change in each "
                 "clinical variable", fontsize=11.5, color=INK)

    ax = fig.add_subplot(gs[1, :])
    m = arm_mean[FEAT].to_numpy()
    lim = float(np.nanmax(np.abs(m)))
    im = ax.pcolormesh(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=ax, fraction=0.02, label=f"mean change over {TAU:.0f} months")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(m[i, j]):
                ax.text(j + 0.5, i + 0.5, f"{m[i, j]:+.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(m[i, j]) > 0.55 * lim else INK)
    ax.set_xticks(np.arange(len(FEAT)) + 0.5)
    ax.set_xticklabels([f"{NICE[f]}\n({UNIT[f]})" for f in FEAT], rotation=35,
                       ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(arm_mean)) + 0.5)
    ax.set_yticklabels([ARM_LABEL[a] for a in arm_mean.index], fontsize=10)
    ax.set_title("C · what each treatment actually changes — EGFR-mutant, from the "
                 "treatment decision onward\nblue = improves, red = worsens",
                 fontsize=11.5, color=INK)

    fig.suptitle("The clinical decoder — movement on the map, read back in clinical "
                 "variables\nthe projection is linear, so the translation is exact",
                 fontsize=13.5, color=INK, y=1.0)
    fig.savefig(OUT / "clinical_decoder.png", dpi=185, bbox_inches="tight")
    fig.savefig(OUT / "clinical_decoder.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = ["# The clinical decoder", "",
             "## A · what one unit of the severity axis means", "",
             "| clinical variable | change per +1 PC1 |", "|---|---|"]
    for _i, r in dec.iterrows():
        lines.append(f"| {NICE[r['feature']]} | {r['per_unit_pc1']:+.2f} "
                     f"{UNIT[r['feature']]} |")
    lines += ["", "## B · mean 3-month change by cancer", "",
              "| family | " + " | ".join(NICE[f] for f in FEAT) + " |",
              "|---|" + "---|" * len(FEAT)]
    for f, r in fam_mean.iterrows():
        lines.append(f"| {f} | " + " | ".join(f"{r[c]:+.2f}" for c in FEAT) + " |")
    lines += ["", "## C · mean 3-month change by treatment (EGFR-mutant)", "",
              "| arm | " + " | ".join(NICE[f] for f in FEAT) + " |",
              "|---|" + "---|" * len(FEAT)]
    for a, r in arm_mean.iterrows():
        lines.append(f"| {ARM_LABEL[a]} | " + " | ".join(f"{r[c]:+.2f}" for c in FEAT) + " |")
    (OUT / "readout.md").write_text("\n".join(lines) + "\n")
    (OUT / "readout.json").write_text(json.dumps(
        {"per_unit_pc1": dec.to_dict("records"),
         "by_family": fam_mean.reset_index().to_dict("records"),
         "by_arm": arm_mean.reset_index().to_dict("records")}, indent=2, default=str) + "\n")
    print("\n".join(lines[:20]))
    print("\n... full tables in", OUT / "readout.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
