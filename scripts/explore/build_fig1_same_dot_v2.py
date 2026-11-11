#!/usr/bin/env python3
"""Same state, different history: two patients who look identical today are not the same patient.

Changes from build_fig1_same_dot_draft.py:
  1. the atlas was pooled from 4 runs, not 6 — Kimi K3 and GLM 5.2 were missing, so the
     per-code cohort was 20 trajectories instead of 30 (fixed in ATLAS_DERIVS);
  2. the exemplar tumour was auto-chosen to maximise patients-through-a-cell, which
     selects for tumours whose trajectories are tightly bunched — i.e. exactly the ones
     with no visible 2-D structure. IBC moves 84% along PC1. Sarcomas move ~70% in PC2.
     The tumour is now chosen for movement, and it is reported.

Two displays of the same patients, side by side:
  A  STATE SPACE — the dot's path through PC1 x PC2; trails in, futures out.
  B  SWIMMER — the same patients with time on the x-axis, aligned at the shared visit.
     Everyone is at the same severity at t=0 by construction; the pasts and futures fan.
  C  the pooled quantification: matched state, matched time, ONE visit per patient.
  D  the real-data anchor from GENIE-BPC NSCLC (aggregate readout, not recomputed).

Panel D is real-patient derived and aggregate-only. Outputs gitignored.
Run: uv run python scripts/explore/build_fig1_same_dot_v2.py [ONCOTREE_CODE]
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
from build_fig1_same_dot_draft import (  # noqa: E402
    BAND,
    DEEP_WINDOW,
    MIN_TRAIL,
    collect,
    find_matched_cell,
    pooled_matched_contrast,
    progression_times,
)
from build_phase_plane_grammar_draft import INK, fit_combined_axis  # noqa: E402
from composite_tumor_host_phase_space import enrich_snapshots, load_tables  # noqa: E402

OUT = ROOT / "outputs" / "exploratory_fig1_same_dot"
GENIE_READOUT = ROOT / "outputs" / "exploratory_memory_kernel_genie" / "readout.json"

DIED, ALIVE, TRAIL = "#c0392b", "#2a7ab0", "#6b7480"
DEFAULT_TUMOUR = "LMS"        # leiomyosarcoma: 70% of movement in PC2, well-spread directions
TRAIL_MONTHS, FUTURE_MONTHS = 9.0, 12.0
SEED = 20260728


def movement_profile(s: pd.DataFrame) -> pd.DataFrame:
    """Per tumour: how much of the motion is in PC2, and how spread the directions are."""
    rows = []
    for code, g in s[~s["terminal"]].groupby("tumor_oncotree_code"):
        dx, dy = [], []
        for _gid, t in g.sort_values(["_gid", "month"]).groupby("_gid"):
            dx += list(np.diff(t["sev"].to_numpy(float)))
            dy += list(np.diff(t["pc2"].to_numpy(float)))
        dx, dy = np.array(dx), np.array(dy)
        if len(dx) < 50:
            continue
        ang = np.arctan2(dy, dx)
        r = float(np.hypot(np.mean(np.cos(ang)), np.mean(np.sin(ang))))
        share = float(np.mean(np.abs(dy)) / (np.mean(np.abs(dx)) + np.mean(np.abs(dy))))
        rows.append({"code": code, "pc2_share": share, "directionality": r,
                     "fan_score": share * (1 - r), "n_traj": int(g["_gid"].nunique())})
    return pd.DataFrame(rows).sort_values("fan_score", ascending=False)


def per_model_contrast(s: pd.DataFrame, prog: dict[str, list[float]]) -> pd.DataFrame:
    """The same-snapshot contrast computed WITHIN each model, then compared across models.

    Pooling raw across models is Simpson's paradox here: Kimi K3's 12-month death base rate
    is ~27% where the others sit near 80%, and most of its patients fall in the earlier-
    progression arm, so a raw pool drags that arm down and erases the effect.
    """
    rows = []
    for (mdl, code), sub in s.groupby(["model_short", "tumor_oncotree_code"], sort=False):
        elig = sub[sub["month"].ge(MIN_TRAIL) & ~sub["terminal"]]
        if elig["_gid"].nunique() < 3:
            continue
        xlo, xhi = elig["sev"].quantile(BAND)
        ylo, yhi = elig["pc2"].quantile(BAND)
        cell = elig[elig["sev"].between(xlo, xhi) & elig["pc2"].between(ylo, yhi)]
        if cell.empty:
            continue
        mlo, mhi = cell["month"].quantile([0.10, 0.90])
        cell = cell[cell["month"].between(mlo, mhi)]
        for gid, g in cell.groupby("_gid", sort=False):
            r = g.sort_values("month").iloc[0]
            if not np.isfinite(r["future_death_12m"]):
                continue
            rows.append({"model": mdl, "died": bool(r["future_death_12m"]),
                         "deep": bool(any(p <= r["month"] - DEEP_WINDOW
                                          for p in prog.get(gid, [])))})
    d = pd.DataFrame(rows)
    out = []
    for mdl, g in d.groupby("model"):
        a, b = g[~g["deep"]], g[g["deep"]]
        if a.empty or b.empty:
            continue
        out.append({"model": mdl, "no_prior": 100 * a["died"].mean(),
                    "prior": 100 * b["died"].mean(),
                    "delta": 100 * (b["died"].mean() - a["died"].mean()),
                    "n_no": len(a), "n_prior": len(b)})
    return pd.DataFrame(out).sort_values("delta", ascending=False)


def build(tumour: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = load_tables()
    s = enrich_snapshots(tables)
    s, _p, _sc, _sg = fit_combined_axis(s)
    prog = progression_times(tables.events)
    profile = movement_profile(s)
    pooled = pooled_matched_contrast(s, prog)
    bymodel = per_model_contrast(s, prog)

    sub = s[s["tumor_oncotree_code"].astype(str).eq(tumour)].copy()
    box = find_matched_cell(sub)
    entries = collect(sub, box, prog)
    if len(entries) < 8:
        raise SystemExit(f"{tumour}: only {len(entries)} patients through the matched cell")

    fig = plt.figure(figsize=(18.5, 5.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.35, 0.62, 0.62], wspace=0.34)

    # ---------------- A: state space
    ax = fig.add_subplot(gs[0, 0])
    pts = pd.concat([e["trail"] for e in entries] + [e["future"] for e in entries])
    xlim = (pts["sev"].quantile(0.02) - 0.2, pts["sev"].quantile(0.98) + 0.2)
    ylim = (pts["pc2"].quantile(0.02) - 0.2, pts["pc2"].quantile(0.98) + 0.2)
    for e in entries:
        col = DIED if e["died"] else ALIVE
        ax.plot(e["trail"]["sev"], e["trail"]["pc2"], color=TRAIL, lw=1.1, alpha=0.55,
                ls=(0, (4, 2)), zorder=3)
        ax.plot(e["future"]["sev"], e["future"]["pc2"], color=col, lw=1.5, alpha=0.75, zorder=4)
        ax.scatter(e["trail"]["sev"].iloc[0], e["trail"]["pc2"].iloc[0], s=16,
                   facecolor="white", edgecolor=TRAIL, lw=1.0, zorder=5)
    cx, cy = 0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])
    ax.scatter([cx], [cy], s=300, facecolor="white", edgecolor=INK, lw=2.4, zorder=7)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("clinical state — PC1 (severity)", fontsize=9.5)
    ax.set_ylabel("clinical state — PC2", fontsize=9.5)
    ax.set_title(f"A · state space\n{tumour}, {len(entries)} patients", fontsize=10.5, color=INK)

    # ---------------- B: swimmer, aligned on the shared visit
    ax = fig.add_subplot(gs[0, 1])
    for e in entries:
        t0 = e["t"]
        col = DIED if e["died"] else ALIVE
        ax.plot(e["trail"]["month"] - t0, e["trail"]["sev"], color=TRAIL, lw=1.2,
                alpha=0.6, ls=(0, (4, 2)), zorder=3)
        ax.plot(e["future"]["month"] - t0, e["future"]["sev"], color=col, lw=1.7,
                alpha=0.85, zorder=4)
        ax.scatter([e["future"]["month"].iloc[-1] - t0], [e["future"]["sev"].iloc[-1]],
                   s=22, color=col, zorder=5)
    ax.axvline(0, color=INK, lw=1.6, zorder=6)
    ax.text(0, ax.get_ylim()[1], "  the same chart today", fontsize=10.5, color=INK,
            weight="bold", va="top", ha="left")
    ax.set_xlabel("months relative to that visit", fontsize=9.5)
    ax.set_ylabel("clinical severity (PC1)", fontsize=9.5)
    ax.set_title("B · the same patients, on time\nidentical today; pasts and futures differ",
                 fontsize=10.5, color=INK)
    ax.plot([], [], color=TRAIL, lw=1.2, ls=(0, (4, 2)), label="how they got here")
    ax.plot([], [], color=DIED, lw=1.7, label="died within 12 months")
    ax.plot([], [], color=ALIVE, lw=1.7, label="alive at 12 months")
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")

    # ---------------- C / D: the numbers
    grp = {k: v for k, v in pooled.items() if not k.startswith("_")}
    real = json.loads(GENIE_READOUT.read_text())["recursive_test"]["+ time-since-dx matched"]

    ax = fig.add_subplot(gs[0, 2])
    for i, r in enumerate(bymodel.itertuples(index=False)):
        ax.plot([0, 1], [r.no_prior, r.prior], color="#9aa4b1", lw=1.4, zorder=2)
        ax.scatter([0, 1], [r.no_prior, r.prior], s=42, color=[ALIVE, DIED], zorder=3)
        ax.annotate(f"{r.model}  {r.delta:+.1f}", xy=(1.04, r.prior), fontsize=7.5,
                    color=INK, va="center")
    ax.set_xlim(-0.25, 1.95)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no earlier\nprogression", "earlier\nprogression"], fontsize=8.5)
    ax.set_ylabel("died within 12 months (%)", fontsize=9.5)
    ax.set_title(f"C · simulated, within each model\nhigher in {len(bymodel)}/{len(bymodel)} models",
                 fontsize=10.5, color=INK)

    ax = fig.add_subplot(gs[0, 3])
    vals = [real["no prior progression"]["death_12m"] * 100,
            real["prior progression"]["death_12m"] * 100]
    notes = [f"{real['no prior progression']['n_patients']} patients",
             f"{real['prior progression']['n_patients']} patients"]
    ax.bar(range(2), vals, color=[ALIVE, DIED], width=0.58)
    for i, (v, n) in enumerate(zip(vals, notes)):
        ax.text(i, v + max(vals) * 0.03, f"{v:.0f}%\n{n}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(2))
    ax.set_xticklabels(["no earlier\nprogression", "earlier\nprogression"], fontsize=8.5)
    ax.set_ylabel("died within 12 months (%)", fontsize=9.5)
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title("D · real patients\nGENIE-BPC NSCLC", fontsize=10.5, color=INK)

    for a in fig.get_axes():
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Figure 1 (draft v2) — two patients who look identical today are not the same patient",
                 fontsize=13.5, color=INK, y=1.03)
    fig.savefig(OUT / "fig1_same_dot_v2.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig1_same_dot_v2.pdf", bbox_inches="tight")
    plt.close(fig)

    prof = profile.set_index("code")
    return {
        "tumour": tumour, "n_patients_in_cell": len(entries),
        "n_trajectories_total": int(s["_gid"].nunique()),
        "n_atlas_runs": 6,
        "tumour_movement": {k: float(prof.loc[tumour, k]) for k in
                            ("pc2_share", "directionality", "fan_score")},
        "top_fan_tumours": profile.head(10)[["code", "pc2_share", "directionality", "fan_score"]]
        .to_dict("records"),
        "pooled_raw_SIMPSON_ARTIFACT": grp,
        "per_model": bymodel.to_dict("records"),
        "real_genie": {"no prior progression": real["no prior progression"]["death_12m"] * 100,
                       "prior progression": real["prior progression"]["death_12m"] * 100},
    }


def main() -> int:
    tumour = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TUMOUR
    rep = build(tumour)
    lines = [
        "# Figure 1 v2 — same dot, different trail", "",
        f"Atlas: {rep['n_trajectories_total']:,} trajectories from {rep['n_atlas_runs']} runs "
        "(Kimi K3 and GLM 5.2 now included).",
        f"Exemplar tumour: **{rep['tumour']}** — {rep['n_patients_in_cell']} patients through the "
        f"matched region; PC2 share of movement {rep['tumour_movement']['pc2_share']:.2f}, "
        f"directionality {rep['tumour_movement']['directionality']:.2f}.", "",
        "## Panel C — simulated, WITHIN each model (matched state + time, one visit/patient)",
        "", "| model | no earlier prog. | earlier prog. | Δ | n |", "|---|---|---|---|---|",
    ]
    for r in rep["per_model"]:
        lines.append(f"| {r['model']} | {r['no_prior']:.1f}% | {r['prior']:.1f}% | "
                     f"{r['delta']:+.1f} | {r['n_no']}/{r['n_prior']} |")
    lines += ["", "Raw pooling across models is a Simpson's paradox here and is reported only "
              "as an artifact: "
              + ", ".join(f"{k.replace(chr(10), ' ')} {v['death_12m_pct']:.1f}%"
                          for k, v in rep["pooled_raw_SIMPSON_ARTIFACT"].items()) + "."]
    lines += ["", "## Panel D — real GENIE-BPC NSCLC",
              f"- no prior progression: {rep['real_genie']['no prior progression']:.1f}%",
              f"- prior progression: {rep['real_genie']['prior progression']:.1f}%", "",
              "## Tumours with the most 2-D trajectory structure", "",
              "| code | PC2 share | directionality | fan score |", "|---|---|---|---|"]
    for r in rep["top_fan_tumours"]:
        lines.append(f"| {r['code']} | {r['pc2_share']:.3f} | {r['directionality']:.3f} "
                     f"| {r['fan_score']:.3f} |")
    (OUT / "readout_v2.md").write_text("\n".join(lines) + "\n")
    (OUT / "readout_v2.json").write_text(json.dumps(rep, indent=2) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
