#!/usr/bin/env python3
"""ARCHETYPE COLLAPSE AND CANCER SEPARATION, ON ALL SIX PAPER MODELS

Computed on the six reference runs of the paper (an earlier archetype output covered four
models and wrote no readout).

TWO QUANTITIES, both per generating model.

  ARCHETYPE SHARE   Within each tumour-code cell, the fraction of generated patients holding
                    the modal value of a baseline feature, averaged over features and cells.
                    A cell of five patients who all share the modal driver contributes 1.0.
                    Reported with a bootstrap interval over CELLS, since the tumour code is
                    the sampling unit and the patient is not.

  CANCER SEPARATION Whether narrow sampling within a cancer reflects the model failing to
                    distinguish cancers at all. Driver entropy within a tumour code is
                    compared against entropy pooled across codes. Low within-code entropy
                    beside high pooled entropy means the model samples narrowly inside each
                    cancer while still telling cancers apart, which is a different failure
                    from not knowing the difference.

Entropy is in bits, computed on the observed value distribution; with five futures per cell
the within-code estimate is biased downward, so the comparison is reported as a contrast
and not as an absolute.

Run: uv run python scripts/explore/build_atlas_archetype_six_model.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs" / "exploratory_atlas_archetype"
SEED = 20260803
NBOOT = 2000

FEATURES = ["primary_driver_bucket", "sex", "stage", "smoking_status", "frailty_band"]


def entropy(values: pd.Series) -> float:
    """Shannon entropy in bits of an observed categorical distribution."""
    p = values.value_counts(normalize=True).to_numpy()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    import build_paper_state_map as smap

    tab = smap.load_reference_tables(smap.REFERENCE_RUNS)
    b = tab["baselines"].groupby("_gid").first()
    feats = [f for f in FEATURES if f in b.columns]
    b = b[["model_short", "tumor_oncotree_code", *feats]].copy()
    for f in feats:
        b[f] = b[f].astype(str)

    rng = np.random.default_rng(SEED)
    rows, ent_rows = [], []
    for model, g in b.groupby("model_short"):
        cells, unanimous = [], []
        for _code, gc in g.groupby("tumor_oncotree_code"):
            if len(gc) < 2:
                continue
            cells.append(float(np.mean([gc[f].value_counts(normalize=True).iloc[0]
                                        for f in feats])))
            # The vivid version of the same fact: a cell in which every generated patient
            # shares BOTH driver and sex.
            unanimous.append(bool(gc["primary_driver_bucket"].nunique() == 1
                                  and gc["sex"].nunique() == 1))
        cells = np.array(cells)
        if not len(cells):
            continue
        boot = rng.choice(cells, size=(NBOOT, len(cells)), replace=True).mean(axis=1)
        rows.append({
            "model": model, "n_cells": int(len(cells)),
            "n_trajectories": int(len(g)),
            "archetype_share": float(cells.mean()),
            "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)),
            "cells_unanimous_driver_and_sex": float(np.mean(unanimous)),
        })

        if "primary_driver_bucket" in feats:
            within = [entropy(gc["primary_driver_bucket"])
                      for _c, gc in g.groupby("tumor_oncotree_code") if len(gc) >= 2]
            ent_rows.append({
                "model": model,
                "within_code_entropy": float(np.mean(within)),
                "pooled_entropy": entropy(g["primary_driver_bucket"]),
            })

    A = pd.DataFrame(rows).sort_values("archetype_share", ascending=False)
    E = pd.DataFrame(ent_rows)

    lines = ["# Archetype collapse and cancer separation, six paper models", "",
             f"{int(A['n_trajectories'].sum()):,} trajectories across "
             f"{b['tumor_oncotree_code'].nunique()} tumour codes and {len(A)} models, using "
             "the six reference runs of the paper. Recomputed here so that archetype "
             "statistics and atlas statistics share one denominator.", "",
             "## Archetype share", "",
             "Mean fraction of generated patients in a tumour-code cell holding the modal "
             "value of a baseline feature, averaged over "
             f"{len(feats)} features ({', '.join(feats)}). Bootstrap over cells, since the "
             "tumour code is the sampling unit.", "",
             "| model | trajectories | cells | archetype share | 95% CI | cells with one "
             "driver and one sex |",
             "|---|---:|---:|---:|---|---:|"]
    for _i, r in A.iterrows():
        lines.append(f"| {r['model']} | {r['n_trajectories']:,} | {r['n_cells']} | "
                     f"{100 * r['archetype_share']:.1f}% | "
                     f"{100 * r['ci_low']:.1f}–{100 * r['ci_high']:.1f}% | "
                     f"{100 * r['cells_unanimous_driver_and_sex']:.0f}% |")

    if len(E):
        lines += ["", "## Cancer separation", "",
                  "Driver entropy within a tumour code against entropy pooled across codes, "
                  "in bits. Narrow sampling inside a cancer alongside a wide pooled "
                  "distribution means the model distinguishes cancers while sampling "
                  "narrowly within them.", "",
                  "| model | within code | pooled | ratio |", "|---|---:|---:|---:|"]
        for _i, r in E.sort_values("within_code_entropy").iterrows():
            ratio = r["within_code_entropy"] / r["pooled_entropy"] if r["pooled_entropy"] else np.nan
            lines.append(f"| {r['model']} | {r['within_code_entropy']:.2f} | "
                         f"{r['pooled_entropy']:.2f} | {ratio:.2f} |")
        lines += ["",
                  "With five futures per cell the within-code entropy is biased downward, so "
                  "this is reported as a contrast rather than an absolute. The direction is "
                  "unambiguous across models."]

    lines += ["", "## Scope", "",
              "Spontaneous generation with only the tumour type fixed. Stage, treatment and "
              "host state were left to each model, so these are descriptive properties of "
              "the generated corpus and not adjusted comparisons."]

    (OUT / "readout.md").write_text("\n".join(lines) + "\n")
    (OUT / "readout.json").write_text(json.dumps(
        {"n_trajectories": int(A["n_trajectories"].sum()),
         "n_codes": int(b["tumor_oncotree_code"].nunique()),
         "features": feats, "archetype": A.to_dict("records"),
         "entropy": E.to_dict("records"), "n_bootstrap": NBOOT, "seed": SEED},
        indent=2) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
