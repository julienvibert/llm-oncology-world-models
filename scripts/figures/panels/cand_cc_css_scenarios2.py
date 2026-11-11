"""Page cand_cc_css_scenarios: four scored scenario comparisons of the benchmark, from the treatment decision.

Message: four scored comparisons of the benchmark, each drawn from the treatment decision: does the course change as it should, and do models agree?
Data: runs/sentinel_usecase_*/derivation (read only), the runs and pairs of the scorer (scripts/explore/sentinel_benchmark_spec_v1.yaml, explicit-resistance run for EGFR after
osimertinib). Time 0 is the decision: the end of the supplied past, or diagnosis for first-line scenarios; the supplied past is not shown. Alive counts use the death endpoint. The scored endpoint is forced-line PFS (as in scripts/explore/score_sentinel_benchmark.py:
line-1 PFS for first-line scenarios, line-2 PFS where the supplied past consumes line 1), shown per model and pooled. Clouds pool the models and are illustrative; the benchmark itself stays in Extended Data Fig. 6. Model-generated.
Run: uv run python scripts/figures/panels/cand_cc_css_scenarios2.py
"""
import glob
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.text import Annotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _css_axes as A  # noqa: E402
import _css_grammar as G  # noqa: E402
from common import GREY, INK, MODEL_COLOURS, ROOT, save_panel  # noqa: E402
from _f3util import axes_mm, new_fig  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
import build_paper_precision_ladder_paired_views as paired  # noqa: E402
import plot_sentinel_trust_map as P  # noqa: E402

M = 30.4375
FR = [6, 12, 24]
LEFT, RIGHT = "#1F6F66", "#D55E00"
MODELS = [("claude_sonnet5", "Sonnet"), ("gemini31_lite", "Gemini 3.1 L"), ("gemma4", "Gemma 4"), ("qwen36_flash", "Qwen 3.6")]
TICK = {"claude_sonnet5": "Sonnet 5", "gemini31_lite": "Gemini 3.1", "gemma4": "Gemma 4", "qwen36_flash": "Qwen 3.6"}
MODEL_LINE = "Sonnet 5, Gemini 3.1 Flash-Lite, Gemma 4, Qwen 3.6 Flash; 3 futures each"
SCORED = {  # the scored comparison of each row, as in the scorer (forced line in brackets)
    "EGFR_WT_1L": "chemotherapy vs osimertinib, line-1 PFS",
    "HER2_pos_breast": "HER2 therapy + chemotherapy vs chemotherapy, line-1 PFS",
    "EGFR_post_progression": "chemotherapy vs continued osimertinib, line-2 PFS",
    "HGSOC_relapse": "platinum, sensitive vs resistant relapse, line-2 PFS",
}
ROWS = [  # title, run, context, (label, arm, condition) expected longer PFS, (label, arm, condition)
    ("EGFR wild-type lung", "sentinel_usecase_egfr_flip_v5_2026_07_13", "EGFR_WT_1L", ("Chemotherapy", "chemo", None), ("Osimertinib", "osimertinib", None)),
    ("HER2-positive breast", "sentinel_usecase_her2_v5_2026_07_13", "HER2_pos_breast", ("HER2 therapy + chemo", "her2_chemo", None), ("Chemotherapy", "chemo", None)),
    ("EGFR after osimertinib", "sentinel_usecase_egfr_post_osi_explicit_resistance_v5_v2_2026_07_13", "EGFR_post_progression", ("Chemotherapy", "chemo", None), ("Continue osimertinib", "continue_osi", None)),
    ("Platinum, ovarian relapse", "sentinel_usecase_hgsoc_v5_2026_07_13", "HGSOC_relapse", ("Platinum-sensitive", "platinum", "hgsoc_platinum_sensitive"), ("Platinum-resistant", "platinum", "hgsoc_platinum_resistant")),
]
man = json.loads((ROOT / "scripts/explore/sentinel_battery_manifest_reconstructed_2026-10-14.json").read_text())
man = {k: {a: b for a, b in v.items() if not a.startswith("_")} for k, v in man.items() if not k.startswith("_")}
runs = [Path(p) for p in sorted(glob.glob(str(ROOT / "runs" / "sentinel_usecase_*"))) if (Path(p) / "derivation/endpoint_rows.csv").exists()]
runs = [r for r in runs if not r.name.startswith("sentinel_usecase_egfr_post_osi_v")]
med = P.combined_table(runs, man)
pol = {(m["context"], m["arm_key"]): m["policy_id"] for m in man.values()}


def pfs(run: str, ctx: str, policy: str, cond: str | None) -> np.ndarray:
    """Forced-line PFS (months) of accepted futures, defined as in the scorer (P.combined_table):
    days from the forced line's start to the first progression or death; line 2 where the supplied past consumes line 1."""
    e = pd.read_csv(ROOT / "runs" / run / "derivation/endpoint_rows.csv")
    e = e[e.accepted.astype(bool) & e.policy_id.eq(policy)]
    if cond:
        e = e[e.condition_id.eq(cond)]
    assert not e.censored.astype(bool).any(), "censored futures: a Kaplan-Meier estimator would be needed"
    col = "pfs_line_2_days" if ctx in P.PREFIX_CONTEXTS else "pfs_line_1_days"
    v = pd.to_numeric(e[col], errors="coerce")
    assert v.notna().all(), f"missing {col} in {policy}"
    return v.to_numpy(float) / M  # 30.4375 days per month


def cutoff(run: str, tid: str) -> float:
    """Days from diagnosis to the decision: the end of the supplied past, or diagnosis for first-line scenarios, which have none."""
    spec = json.loads(sorted((ROOT / "runs" / run / "trajectories" / tid).glob("attempt_*/conditioning_spec.json"))[0].read_text())
    pre = spec.get("observed_prefix")
    if not pre:
        return 0.0
    d0 = date.fromisoformat(spec["baseline_seed"]["identity_card"]["diagnosis"]["diagnosis_date"])
    return float((date.fromisoformat(pre["cutoff_date"]) - d0).days)


def load(run: str) -> tuple[pd.DataFrame, pd.Series]:
    t = paired.state_map.load_reference_tables((run,))
    active, _ = paired.ledger.load_active_condition_features((run,))
    st, _ = paired.state_map.build_state_matrix(t, active)
    cut = {t_: cutoff(run, t_) for t_ in st.trajectory_id.astype(str).unique()}
    st["cut"] = st.trajectory_id.astype(str).map(cut).astype(float)
    st["x"], st["y"] = A.project(st)
    st["t"] = (st.days_from_diagnosis - st.cut) / M
    en = t["endpoints"].set_index("_gid")
    os_ = (pd.to_numeric(en.os_days, errors="coerce") - st.groupby("_gid").cut.first()) / M
    return st.sort_values(["_gid", "t", "snapshot_index"]), os_.dropna()


data = {r[1]: load(r[1]) for r in ROWS}
allp = pd.concat([d[d.t.ge(0)][["x", "y"]] for d, _ in data.values()])
XL = (float(allp.x.min()) - 0.1, float(allp.x.max()) + 0.1)
YL = (float(allp.y.min()) - 0.1, float(allp.y.max()) + 0.1)
W, H = 180.0, 156.0
fig = new_fig(W, H)
FW, GAP, FH = 18.0, 0.6, 26.0
XA, XB = 4.0, 4.0 + 3 * (FW + GAP) + 3.0
fig.text((XA + 27) / W, (H - 2.2) / H, "Expected longer forced-line PFS", fontsize=7, ha="center", va="center", fontweight="bold")
fig.text((XB + 27) / W, (H - 2.2) / H, "Expected shorter", fontsize=7, ha="center", va="center", fontweight="bold")
out = {}
for r, (title, run, ctx, a, b) in enumerate(ROWS):
    y0 = H - 13 - (r + 1) * FH - r * 10.5
    d, os_ = data[run]
    tt = fig.text(XA / W, (y0 + FH + 1.0) / H, title, fontsize=6.5, fontweight="bold", va="bottom")
    fig.add_artist(Annotation(f"Scored: {SCORED[ctx]}  |  {MODEL_LINE}", xy=(1, 0), xycoords=tt, xytext=(6, 0),
                              textcoords="offset points", fontsize=6, va="bottom"))
    curves = []
    for (name, arm, cond), x0, col in ((a, XA, LEFT), (b, XB, RIGHT)):
        q = d[d.policy_id.eq(pol[(ctx, arm)])]
        if cond:
            q = q[q.condition_id.eq(cond)]
        ids = q._gid.unique()
        start = np.array([q[q._gid.eq(g) & q.t.le(0)].iloc[-1][["x", "y"]].to_numpy(float) for g in ids if len(q[q._gid.eq(g) & q.t.le(0)])], float).mean(axis=0)
        curves.append((pfs(run, ctx, pol[(ctx, arm)], cond), col))
        for j, t in enumerate(FR):
            ax = axes_mm(fig, x0 + j * (FW + GAP), y0, FW, FH)
            cur = q[q.t.le(t + 1e-9)].groupby("_gid").tail(1)
            cur = cur[cur._gid.map(os_) > t]
            pts = cur[["x", "y"]].to_numpy(float)
            G.cloud(ax, pts, col, XL, YL, sigma=4.0, min_n=8)
            ax.scatter(pts[:, 0], pts[:, 1], s=1.6, color=INK, lw=0, zorder=4)
            if len(pts) >= 3:
                G.fan(ax, start, pts, col, lines=False)
            ax.scatter(*start, marker="*", s=12, color="white", edgecolor=INK, lw=0.4, zorder=8)
            G.bare(ax, XL, YL)
            ax.text(0.97, 0.04, f"alive {len(pts)}/{len(ids)}", transform=ax.transAxes, fontsize=6, ha="right", va="bottom", zorder=9)
            if j == 0:
                ax.text(0.03, 0.97, name.replace(" + chemo", "\n+ chemo").replace("Continue ", "Continue\n"), transform=ax.transAxes, fontsize=6, va="top", zorder=9)
            if r == 0:
                fig.text((x0 + j * (FW + GAP) + FW / 2) / W, (H - 6.6) / H, f"{t} months", fontsize=6.5, ha="center", va="center")
    # the scored comparison, per model
    ax = axes_mm(fig, 133, y0, 20, FH)
    vals = []
    for i, (m, lab) in enumerate(MODELS):
        e = med[med.context.eq(ctx) & med.model.eq(m)]
        pa = e[e.arm_key.eq(a[1]) & (e.condition_id.eq(a[2]) if a[2] else True)].pfs_mo.median()
        pb = e[e.arm_key.eq(b[1]) & (e.condition_id.eq(b[2]) if b[2] else True)].pfs_mo.median()
        vals.append(round(float(pa - pb), 1))
        ax.plot([0, pa - pb], [-i, -i], color=MODEL_COLOURS[lab], lw=1.0)
        ax.plot(pa - pb, -i, "o", ms=3.4, color=MODEL_COLOURS[lab], mec=INK, mew=0.25)
    ax.plot([0, 0], [-3.6, 0.6], color=GREY, lw=0.6)
    ax.set_ylim(-3.6, 0.6)
    ax.set_yticks([-i for i in range(4)])
    ax.set_yticklabels([TICK[m[0]] for m in MODELS], fontsize=6)
    ax.tick_params(axis="y", length=0, pad=1)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("PFS difference (mo)", labelpad=1, fontsize=6)
    kx = axes_mm(fig, 161, y0, 17, FH)
    grid = np.linspace(0, 36, 361)
    for v, col in curves:  # no censoring (asserted), so the empirical curve is the Kaplan-Meier estimate
        kx.step(grid, [(v > g).mean() for g in grid], where="post", color=col, lw=1)
    kx.set_xlim(0, 36)
    kx.set_ylim(0, 1.02)
    kx.set_xticks([0, 12, 24, 36])
    kx.set_yticks([0, 1])
    kx.set_yticklabels(["0", "100"])
    kx.tick_params(length=1.5, pad=1)
    kx.set_xlabel("Months from line start", labelpad=1, fontsize=6)
    kx.set_ylabel("PFS, pooled (%)", labelpad=0, fontsize=6)
    out[title] = (vals, [len(c[0]) for c in curves], [round(float(np.median(c[0])), 1) for c in curves])
save_panel(fig, "cand_cc_css_scenarios", legend=(
    "Four scored comparisons of the scenario benchmark, drawn from the treatment decision. Each row names its scored comparison and the four models that generated it "
    "(Sonnet 5, Gemini 3.1 Flash-Lite, Gemma 4 and Qwen 3.6 Flash; 3 futures per model and situation, n = 12 per situation). Left and right, the two situations of the comparison "
    "on the disease and host axes, four models pooled (shading and contours, smoothed density when at least 8 futures are alive; points, futures; star, mean state at the decision; "
    "arrow, mean move; alive, futures alive over futures generated). Centre right, median forced-line progression-free survival (PFS) of the expected-longer situation minus the other, "
    "for each model; this is the scored endpoint, before the threshold used for ovarian relapse. Right, forced-line PFS of the same two situations, four models pooled, from the start of the "
    "forced line to the first progression or death; line 1 for first-line scenarios, line 2 where the supplied past consumes line 1. No future was censored. "
    "Runs and pairs as in the scorer; the full 12-scenario benchmark is in Extended Data Fig. 6. Model-generated futures."),
    sources=["runs/sentinel_usecase_*/derivation (read only)", "scripts/explore/sentinel_benchmark_spec_v1.yaml"], confirm=["No P value on the pooled PFS curves: futures are clustered within model and seed, and the scored comparison is per model; a test would need that structure."],
    message="Scored comparisons from the decision: does the course change as it should, and do models agree?", strict=False)
print(out)
