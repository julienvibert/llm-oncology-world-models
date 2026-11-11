#!/usr/bin/env python3
"""The sentinel treatment-reasoning trust map (forced-line PFS; Extended Data Fig. 4a,b).

Data-driven: discovers the completed sentinel_usecase_* runs, scores every present
context PFS-primary (via score_sentinel_benchmark), and renders one figure that tells
the static-vs-dynamic story. Fills in automatically as more contexts finish — run it
again any time.

Panels:
  A  trust-map heatmap: contexts (rows, banded STATIC=biomarker vs DYNAMIC=history/host)
     x models (cols, cheap->frontier). Cell = fraction of the context's core primitive
     the model got right on FORCED-LINE PFS. Glyphs: check/cross + guardrail dot.
  B  the dissociation: per-model mean score in STATIC vs DYNAMIC contexts.

Real-derived runs -> aggregate only; synthetic seeds so the figure is git-ok.
  uv run python scripts/explore/plot_sentinel_trust_map.py            # auto-discover
  uv run python scripts/explore/plot_sentinel_trust_map.py --runs <dir> ... --manifest <json>
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_sentinel_benchmark import (  # noqa: E402
    PREFIX_CONTEXTS,
    score_hgsoc_delta,
    score_pancreas_host,
    score_tiers,
    _meds,
)

SPEC = Path(__file__).resolve().parent / "sentinel_benchmark_spec_v1.yaml"
OUT = ROOT / "outputs" / "exploratory_sentinel_benchmark"
DEFAULT_MANIFEST = Path("/private/tmp/sentinel_reasoning_battery_v5_remaining_source/arm_manifest.json")
MODEL_ORDER = ["gemini31_lite", "qwen36_flash", "gemma4", "claude_sonnet5"]
MODEL_LABEL = {"gemini31_lite": "Gemini 3.1\nFlash-Lite", "qwen36_flash": "Qwen 3.6\nFlash",
               "gemma4": "Gemma 4", "claude_sonnet5": "Claude\nSonnet-5"}
HGSOC_DELTA_MIN = 1.5  # months of platinum-PFS gain (sensitive-resistant) to credit history use


def combined_table(run_dirs: list[Path], manifest: dict) -> pd.DataFrame:
    """Concatenate endpoint_rows across runs -> median forced-line PFS + OS per cell."""
    frames = [pd.read_csv(r / "derivation" / "endpoint_rows.csv")
              for r in run_dirs if (r / "derivation" / "endpoint_rows.csv").exists()]
    e = pd.concat(frames, ignore_index=True)
    if "accepted" in e:
        e = e[e["accepted"].astype(bool)]
    pol2 = {m["policy_id"]: (m["context"], m["arm_key"]) for m in manifest.values()}
    e = e[e["policy_id"].isin(pol2)].copy()
    e["context"] = e["policy_id"].map(lambda p: pol2[p][0])
    e["arm_key"] = e["policy_id"].map(lambda p: pol2[p][1])
    e["model"] = e["model_short"]
    l1 = pd.to_numeric(e["pfs_line_1_days"], errors="coerce")
    l2 = pd.to_numeric(e["pfs_line_2_days"], errors="coerce")
    e["pfs_mo"] = np.where(e["context"].isin(PREFIX_CONTEXTS), l2, l1) / 30.4
    e["os_mo"] = pd.to_numeric(e["os_days"], errors="coerce") / 30.4
    return e.groupby(["context", "condition_id", "arm_key", "model"], as_index=False)[["pfs_mo", "os_mo"]].median()


def correctness(ctx_id: str, ctx: dict, df: pd.DataFrame, model: str) -> tuple[float, float | None]:
    """(core-primitive score in [0,1], guardrail in [0,1] or None) on forced-line PFS."""
    cs = ctx.get("core_signal")
    if cs == "history_delta" and ctx_id == "HGSOC_relapse":
        hg = score_hgsoc_delta(df, model)
        if not hg:
            return np.nan, None
        return float(np.mean([hg["platinum_delta"] > HGSOC_DELTA_MIN, hg["platinum_not_top_resistant"]])), None
    if cs == "host_intensity":
        pan = score_pancreas_host(df, model)
        if not pan:
            return np.nan, None
        return float(np.mean([pan["folfirinox_top_fit"], pan["folfirinox_not_top_frail"]])), None
    # direction / biomarker_flip / post-osi: pairwise signs (+ guardrail) on PFS
    t = score_tiers(ctx, _meds(df, ctx_id, model, "pfs_mo"), _meds(df, ctx_id, model, "os_mo"))
    p = t["primary"]
    return (np.nan if p is None else float(p)), t["guardrail"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", type=Path,
                    default=[Path(p) for p in sorted(glob.glob(str(ROOT / "runs" / "sentinel_usecase_*")))])
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    spec = yaml.safe_load(SPEC.read_text())["contexts"]
    manifest = json.loads(args.manifest.read_text())
    runs = [r for r in args.runs if (r / "derivation" / "endpoint_rows.csv").exists()]
    # EGFR_post_progression: prefer the EXPLICIT-resistance run (the FAIR test — resistance
    # made explicit) over the implicit one, so the two don't average into the same context.
    if any("post_osi_explicit" in r.name for r in runs):
        runs = [r for r in runs if not (r.name.startswith("sentinel_usecase_egfr_post_osi_v"))]
    df = combined_table(runs, manifest)
    present = [c for c in spec if (df["context"] == c).any()]
    dynamic = {c for c in present if spec[c].get("core_signal") in ("history_delta", "host_intensity")}
    # order: static block then dynamic block; within, spec order
    rows = [c for c in present if c not in dynamic] + [c for c in present if c in dynamic]
    models = [m for m in MODEL_ORDER if (df["model"] == m).any()]

    score = np.full((len(rows), len(models)), np.nan)
    guard = np.full((len(rows), len(models)), np.nan)
    for i, c in enumerate(rows):
        for j, m in enumerate(models):
            s, g = correctness(c, spec[c], df, m)
            score[i, j] = s
            guard[i, j] = np.nan if g is None else g

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(2.2 + 1.5 * len(models), 1.6 + 0.62 * len(rows)))
    gs = fig.add_gridspec(2, 1, height_ratios=[len(rows), 2.4], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    cmap = plt.get_cmap("RdYlGn")
    ax.imshow(score, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    n_static = sum(c not in dynamic for c in rows)
    for i, c in enumerate(rows):
        for j in range(len(models)):
            v = score[i, j]
            if np.isnan(v):
                ax.text(j, i, "·", ha="center", va="center", color="#999", fontsize=12)
                continue
            gl = "✓" if v >= 0.99 else ("✗" if v <= 0.01 else "◐")
            txt = "white" if (v >= 0.75 or v <= 0.25) else "#222"   # contrast on saturated cells
            ax.text(j, i - 0.16, gl, ha="center", va="center", fontsize=13, color=txt)
            ax.text(j, i + 0.26, f"{v:.0%}", ha="center", va="center", fontsize=7.5, color=txt)
            if guard[i, j] == 0:   # wrong-target ranked FIRST -> red guardrail flag
                ax.text(j + 0.34, i - 0.30, "!", ha="center", va="center", fontsize=9,
                        color="white", bbox=dict(boxstyle="circle,pad=0.1", fc="#b3261e", ec="none"))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in models], fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([spec[c]["label"] for c in rows], fontsize=8)
    ax.set_xlim(-0.5, len(models) - 0.5)
    # static/dynamic band separator + labels
    if 0 < n_static < len(rows):
        ax.axhline(n_static - 0.5, color="black", lw=1.6)
    ax.add_patch(Rectangle((-0.5, -0.5), len(models), n_static, fill=False, ec="#1b5e20", lw=1.4))
    if n_static < len(rows):
        ax.add_patch(Rectangle((-0.5, n_static - 0.5), len(models), len(rows) - n_static,
                               fill=False, ec="#7b1fa2", lw=1.4))
    ax.text(len(models) - 0.42, (n_static - 1) / 2, "STATIC\nbiomarker\nuse", va="center", ha="left",
            fontsize=7.5, color="#1b5e20", fontweight="bold")
    if n_static < len(rows):
        ax.text(len(models) - 0.42, n_static + (len(rows) - n_static - 1) / 2, "DYNAMIC\nhistory /\nhost", va="center",
                ha="left", fontsize=7.5, color="#7b1fa2", fontweight="bold")
    ax.set_title("Sentinel treatment-reasoning trust map — core primitive on forced-line PFS\n"
                 "✓ correct · ◐ partial · ✗ wrong · ! = wrong-target ranked first (guardrail breach)",
                 fontsize=9.5, loc="left")

    # panel B: the dissociation
    axb = fig.add_subplot(gs[1])
    stat_idx = [i for i, c in enumerate(rows) if c not in dynamic]
    dyn_idx = [i for i, c in enumerate(rows) if c in dynamic]
    x = np.arange(len(models))
    sm = [np.nanmean(score[stat_idx, j]) if stat_idx else np.nan for j in range(len(models))]
    dm = [np.nanmean(score[dyn_idx, j]) if dyn_idx else np.nan for j in range(len(models))]
    axb.bar(x - 0.2, sm, 0.38, color="#2e7d32", label="STATIC (biomarker)")
    axb.bar(x + 0.2, dm, 0.38, color="#7b1fa2", label="DYNAMIC (history/host)")
    for j in range(len(models)):
        for off, v in ((-0.2, sm[j]), (0.2, dm[j])):
            if not np.isnan(v):
                axb.text(x[j] + off, v + 0.02, f"{v:.0%}", ha="center", fontsize=7.5)
    axb.set_xticks(x)
    axb.set_xticklabels([MODEL_LABEL[m] for m in models], fontsize=8)
    axb.set_ylim(0, 1.12)
    axb.set_ylabel("mean primitive correct")
    axb.set_title("The dissociation — static biomarker reasoning works; dynamic/history reasoning fails",
                  fontsize=9, loc="left")
    axb.legend(fontsize=7.5, ncol=2, loc="upper left")
    axb.grid(axis="y", color="#eee")
    axb.set_axisbelow(True)

    out = OUT / "fig_sentinel_trust_map.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[trustmap] runs={len(runs)} contexts={len(rows)} models={len(models)}")
    print("[trustmap] static mean: " + ", ".join(f"{m} {v:.0%}" for m, v in zip(models, sm)))
    print("[trustmap] dynamic mean: " + ", ".join(f"{m} {v:.0%}" for m, v in zip(models, dm) if not np.isnan(v)))
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
