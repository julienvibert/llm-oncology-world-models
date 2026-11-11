#!/usr/bin/env python3
"""WHAT CHANGES WHEN EACH INPUT IS VARIED -- THE PERTURBATION-RESPONSE MATRIX

The interventional run is a factorial design that was never analysed as one: context
(EGFR-mutant vs wild-type) x baseline state (S1-S5) x prior history (H0-H2) x assigned
treatment (chemo, osimertinib, pembrolizumab, supportive), 120 cells, 360 futures. That
means each of the four inputs can be varied with the other three held fixed, which is
exactly the experiment the framework describes and which no observational cohort can run.

For every (input, readout) pair we ask one question: how much does changing this input move
this readout, once everything else is held constant? Cell means are computed first, so a
cell with more futures cannot dominate; the effect is then the spread across that input's
levels, averaged over all combinations of the other three.

TWO THINGS MAKE THE NUMBERS READABLE.

  STANDARDISED  Raw spreads are in different units per readout -- months, encoded state,
                toxicity grade -- and cannot be compared across a row. Each spread is
                divided by the within-cell standard deviation of the same readout, giving a
                unitless "how large against the noise between identical patients".

  NULL-CONTROLLED  With three futures per cell, a spread appears even when the input does
                nothing. Level labels are permuted within each held-fixed stratum, which
                destroys the input's effect while preserving the design and the sample
                sizes; the reported value is the observed effect minus the permutation
                mean. A cell at or below zero means the input did not move that readout
                beyond what shuffling produces.

This is not a Jacobian: every input is discrete, so no derivative exists. It is also not a
correlation table -- each column is a controlled perturbation of the stored model law at a
fixed anchor, not an association observed in data.

Run: uv run python scripts/explore/build_perturbation_response_matrix.py
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
sys.path.insert(0, str(ROOT / "scripts"))

MAP = ROOT / "outputs" / "paper_state_map"
OUT = ROOT / "outputs" / "exploratory_perturbation_matrix"
MONTH = 30.4
TAU = 3.0
NPERM = 400
SEED = 20260803

# Nature final-size contract. The figure is authored at 183 mm, so these are the
# actual printed point sizes rather than values that will be scaled down later.
INK = "#202124"
MUTED = "#60656f"
GRIDC = "#c7cbd1"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#8b929b"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 6,
    "axes.titlesize": 7,
    "axes.labelsize": 6,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "legend.fontsize": 5.5,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "lines.linewidth": 0.8,
    "lines.markersize": 3.5,
    "patch.linewidth": 0.6,
    "legend.frameon": False,
})

INPUTS = [("C", "EGFR context\n(mutant vs wild-type)"),
          ("X", "baseline state\n(S1–S5)"),
          ("H", "prior history\n(H0–H2)"),
          ("A", "assigned treatment\n(4 arms)")]

READOUTS = [
    ("d_severity", "severity\nat 3 months"),
    ("d_burden", "tumour burden\nat 3 months"),
    ("d_host", "performance status\nat 3 months"),
    ("tox", "max toxicity\nby 3 months"),
    ("pfs1", "duration of\nfirst-line control"),
    ("exit12", "death by\n12 months"),
    ("os", "overall\nsurvival"),
]


def load() -> pd.DataFrame:
    """One row per generated future: its four design inputs and seven readouts.

    The twenty state coordinates are derived (anatomical group maxima, disease extent,
    encoded weight loss) and do not exist in the raw snapshot table, so the state is built
    through the paper builder's own functions rather than reimplemented here. Design labels
    come from the builder's phase-1 record table, which already carries context, arm, state
    and history per future.
    """
    import build_paper_state_feature_ledger as ledger
    import build_paper_state_map as smap

    tables = smap.load_reference_tables([smap.PHASE1_RUN])
    active, _meta = ledger.load_active_condition_features([smap.PHASE1_RUN])
    state, _sites = smap.build_state_matrix(tables, active)

    c = json.load(open(MAP / "state_map_contract.json"))["primary_model"]
    order, mult = c["feature_names"], np.array(c["feature_multipliers"])
    mu = np.array(c["mean_after_feature_multipliers"])
    V = np.array(c["all_components_after_feature_multipliers"]).T
    Z = (state[order].to_numpy(float) * mult - mu) @ V
    state = state.assign(pc1=Z[:, 0],
                         month=pd.to_numeric(state["days_from_diagnosis"],
                                             errors="coerce") / MONTH)

    lab = pd.read_csv(MAP / "phase1_p0_records.csv").groupby("_gid").first()[
        ["context", "arm", "state", "history"]]

    # anchor state and state at TAU months, LOCF
    rows = {}
    for gid, g in state.sort_values("month").groupby("_gid"):
        g = g.dropna(subset=["month"])
        if g.empty:
            continue
        a = float(g["month"].iloc[0])
        w = g[g["month"] <= a + TAU]
        last, f0 = (w.iloc[-1] if len(w) else g.iloc[0]), g.iloc[0]
        rows[gid] = {
            "d_severity": float(last["pc1"] - f0["pc1"]),
            "d_burden": float(last["tumour_extent"] - f0["tumour_extent"]),
            "d_host": float(last["host_ecog"] - f0["host_ecog"]),
            "tox": float(w["host_active_ae_max_grade"].max()) if len(w)
            else float(f0["host_active_ae_max_grade"]),
        }
    d = pd.DataFrame(rows).T.join(lab, how="inner")

    end = tables["endpoints"].groupby("_gid").first()
    d["os"] = end["os_days"].astype(float) / MONTH
    d["pfs1"] = end["pfs_line_1_days"].astype(float) / MONTH
    d["exit12"] = (end["os_days"].astype(float) <= 365).astype(float)

    d = d.rename(columns={"context": "C", "state": "X", "history": "H", "arm": "A"})
    for k in ("C", "X", "H", "A"):
        d[k] = d[k].astype(str)
    return d.dropna(subset=["C", "X", "H", "A"]).copy()


def effect(d: pd.DataFrame, inp: str, col: str,
           rng: np.random.Generator) -> tuple[float, float, int, np.ndarray]:
    """Null-corrected standardised spread of `col` across levels of `inp`.

    Everything else is held fixed by stratifying on the other three inputs. Within a
    stratum the cell means are taken first, so unequal cell sizes cannot tilt the result.
    """
    others = [k for k in ("C", "X", "H", "A") if k != inp]
    sub = d.dropna(subset=[col])
    if sub[inp].nunique() < 2:
        return (np.nan, np.nan, 0, np.array([]))

    within = sub.groupby(["C", "X", "H", "A"])[col].std(ddof=1)
    scale = float(np.nanmedian(within)) or float(sub[col].std(ddof=1)) or 1.0

    def spread(frame: pd.DataFrame, label: pd.Series) -> float:
        vals = []
        for _, g in frame.groupby(others, observed=True):
            m = g.assign(_L=label.loc[g.index]).groupby("_L")[col].mean()
            if m.notna().sum() >= 2:
                vals.append(m.max() - m.min())
        return float(np.mean(vals)) if vals else np.nan

    obs = spread(sub, sub[inp])
    # Permute the level label WITHIN each held-fixed stratum: design and n preserved,
    # the input's effect destroyed.
    null = []
    for _ in range(NPERM):
        lab = sub.groupby(others, observed=True)[inp].transform(
            lambda s: rng.permutation(s.to_numpy()))
        null.append(spread(sub, lab))
    nm = float(np.nanmean(null))
    return ((obs - nm) / scale, obs / scale, int(sub.groupby(others).ngroups),
            np.asarray(null, dtype=float) / scale)


def panel_label(ax: plt.Axes, label: str) -> None:
    """Place a Nature-style panel label outside the plotting area."""
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=8,
            fontweight="bold", fontstyle="normal", ha="left", va="bottom",
            color=INK, clip_on=False)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = load()
    print(f"{len(d)} futures; design "
          f"{d.C.nunique()}x{d.X.nunique()}x{d.H.nunique()}x{d.A.nunique()}", file=sys.stderr)

    M = np.full((len(INPUTS), len(READOUTS)), np.nan)
    raw = np.full_like(M, np.nan)
    history_null = np.array([])
    for i, (inp, _l) in enumerate(INPUTS):
        for j, (col, _r) in enumerate(READOUTS):
            if col not in d.columns:
                continue
            M[i, j], raw[i, j], _n, null = effect(d, inp, col, rng)
            if inp == "H" and col == "d_severity":
                history_null = null

    fig = plt.figure(figsize=(7.2047, 3.55))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.52, 0.72, 0.76), wspace=0.52)
    ax = fig.add_subplot(gs[0, 0])
    v = np.nanmax(np.abs(M)) or 1.0
    im = ax.imshow(M, cmap="PuOr_r", vmin=-v, vmax=v, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.2,
                        color="white" if abs(M[i, j]) > 0.6 * v else INK)
    ax.set_xticks(range(len(READOUTS)))
    ax.set_xticklabels([r[1] for r in READOUTS], rotation=38, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(range(len(INPUTS)))
    ax.set_yticklabels([i[1] for i in INPUTS])
    ax.set_title("Controlled perturbation-response matrix", loc="left", pad=8,
                 fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.03)
    cb.set_label("Null-corrected effect (within-cell SD)")
    cb.outline.set_linewidth(0.5)
    panel_label(ax, "a")

    # Open the H × early-trajectory cell. All generated futures remain visible; the
    # horizontal segment is only their mean. The factorial design is balanced, so this
    # pooled view does not reweight the fixed C/X/A cells.
    axb = fig.add_subplot(gs[0, 1])
    histories = ["H0", "H1", "H2"]
    shapes = ["o", "s", "^"]
    colours = [BLUE, ORANGE, GREY]
    hist_summary: dict[str, dict[str, float | int]] = {}
    jitter_rng = np.random.default_rng(SEED + 1)
    for k, (h, marker, colour) in enumerate(zip(histories, shapes, colours, strict=True)):
        vals = d.loc[d["H"] == h, "d_severity"].dropna().to_numpy(float)
        jitter = jitter_rng.uniform(-0.18, 0.18, size=len(vals))
        axb.scatter(np.full(len(vals), k) + jitter, vals, s=5, marker=marker,
                    facecolors="none", edgecolors=colour, linewidths=0.5, alpha=0.38,
                    rasterized=True)
        mean = float(np.mean(vals))
        axb.plot([k - 0.24, k + 0.24], [mean, mean], color=INK, lw=1.4,
                 solid_capstyle="round")
        hist_summary[h] = {"n_futures": int(len(vals)), "mean": mean}
    axb.axhline(0, color=GRIDC, lw=0.6, zorder=0)
    axb.set_xticks(range(3), histories)
    axb.set_ylabel("PC1 displacement (score, 3 months)")
    axb.set_title("Prior history opens into futures", loc="left", pad=8,
                  fontweight="bold")
    axb.text(0.02, 0.98, "2.23 SD\nnull-corrected", transform=axb.transAxes,
             ha="left", va="top", fontsize=5.5, color=INK)
    panel_label(axb, "b")

    # Show the exact permutation correction used by the highlighted matrix cell.
    axc = fig.add_subplot(gs[0, 2])
    axc.hist(history_null, bins=22, color="#b7bcc3", edgecolor="white", linewidth=0.5)
    observed = float(raw[2, 0])
    axc.axvline(observed, color=ORANGE, lw=1.5)
    axc.text(observed, axc.get_ylim()[1] * 0.96, "observed\n5.15 SD",
             ha="right", va="top", fontsize=5.5, color=ORANGE)
    axc.set_xlabel("Permuted raw effect (within-cell SD)")
    axc.set_ylabel("Permutations (count)")
    axc.set_title("Finite-sample null", loc="left", pad=8, fontweight="bold")
    axc.text(0.02, 0.98, "400 within-stratum\nlabel permutations",
             transform=axc.transAxes, ha="left", va="top", fontsize=5.5, color=MUTED)
    panel_label(axc, "c")

    fig.subplots_adjust(left=0.12, right=0.985, top=0.89, bottom=0.28)
    fig.savefig(OUT / "perturbation_response_matrix.png", dpi=400)
    fig.savefig(OUT / "perturbation_response_matrix.pdf")
    fig.savefig(OUT / "perturbation_response_matrix.svg")
    plt.close(fig)

    lines = ["# What changes when each input is varied", "",
             f"{len(d)} generated futures in a "
             f"{d.C.nunique()} × {d.X.nunique()} × {d.H.nunique()} × {d.A.nunique()} "
             "factorial over tumour context, baseline state, prior history and assigned "
             "treatment. Each input is varied with the other three held fixed — the "
             "experiment the framework describes, and one no observational cohort can run.",
             "",
             "Values are null-corrected standardised spreads: the change in a readout "
             "across levels of an input, in units of the variation between identical "
             "patients, minus what label-shuffling produces at the same sample size. "
             "**Zero means the input did not move that readout beyond chance.**", "",
             "| input | " + " | ".join(r[1].replace("\n", " ") for r in READOUTS) + " |",
             "|---" * (len(READOUTS) + 1) + "|"]
    for i, (_inp, lab) in enumerate(INPUTS):
        cells = ["—" if not np.isfinite(x) else f"{x:.2f}" for x in M[i]]
        lines.append(f"| {lab.replace(chr(10), ' ')} | " + " | ".join(cells) + " |")
    lines += ["",
              "## Reading it", "",
              "Rows are perturbations, columns are readouts. A row that is large in one "
              "column and near zero in others identifies an input with a specific effect "
              "rather than a general one — which is the claim the framework makes and the "
              "reason treatment and history are examined separately below.", "",
              "## Scope", "",
              "One interventional run, one tumour setting, three futures per cell. The "
              "matrix shows which perturbations produce structured responses in this "
              "model; it does not establish human effects, and the permutation null "
              "controls for sample size but not for the design being model-implied "
              "throughout."]
    (OUT / "readout.md").write_text("\n".join(lines) + "\n")
    (OUT / "readout.json").write_text(json.dumps(
        {"n_futures": int(len(d)),
         "design": {k: int(d[k].nunique()) for k in ("C", "X", "H", "A")},
         "inputs": [i[0] for i in INPUTS], "readouts": [r[0] for r in READOUTS],
         "effect_null_corrected": M.tolist(), "effect_raw": raw.tolist(),
         "history_open_cell": hist_summary,
         "history_permutation_null_sd": history_null.tolist(),
         "n_permutations": NPERM, "seed": SEED}, indent=2) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
