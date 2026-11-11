#!/usr/bin/env python3
"""Build the paper-facing treatment-reasoning trust map.

This consolidates two already generated assay families:

* the broad sentinel treatment-reasoning battery (four models, 12 contexts);
* the exact-anchor phase-1 EGFR specificity gate (three models, five checks).

The assays remain visually and analytically separate. Their scores are not
pooled because the model rosters, prompts and estimands differ.

Run:
  uv run python scripts/build_paper_interventional_trust_map.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "outputs" / "paper_interventional_trust_map"
DEFAULT_SENTINEL = (
    ROOT
    / "outputs"
    / "exploratory_sentinel_benchmark"
    / "sentinel_trust_map_table.csv"
)
DEFAULT_PHASE1 = (
    ROOT / "outputs" / "exploratory_interventional_phase1" / "phase1_gate_scores.csv",
    ROOT
    / "outputs"
    / "exploratory_interventional_phase1_kimi_k3_combined"
    / "phase1_gate_scores.csv",
    ROOT
    / "outputs"
    / "exploratory_interventional_phase1_glm52_full"
    / "phase1_gate_scores.csv",
)

PAPER = "#ffffff"
INK = "#172033"
MUTED = "#65748b"
GRID = "#dce3ec"
PASS = "#0072B2"
FAIL = "#D55E00"
STATIC = "#277da1"
DYNAMIC = "#8f5aa9"

SENTINEL_MODELS = (
    "gemini31_lite",
    "qwen36_flash",
    "gemma4",
    "claude_sonnet5",
)
SENTINEL_LABEL = {
    "gemini31_lite": "Gemini 3.1\nLite",
    "qwen36_flash": "Qwen 3.6\nFlash",
    "gemma4": "Gemma 4",
    "claude_sonnet5": "Claude\nSonnet 5",
}
PHASE1_ORDER = ("claude_sonnet5", "kimi_k3_off", "glm52_fp8")
PHASE1_LABEL = {
    "claude_sonnet5": "Claude Sonnet 5",
    "kimi_k3_off": "Kimi K3",
    "glm52_fp8": "GLM-5.2",
}
PHASE1_COLOR = {
    "claude_sonnet5": "#5e3c99",
    "kimi_k3_off": "#277da1",
    "glm52_fp8": "#D55E00",
}
PHASE1_CHECKS = (
    ("egfr_mut_osi_gt_chemo", "EGFR-mutant:\nosimertinib > chemo"),
    ("egfr_mut_osi_gt_bsc_ttp", "EGFR-mutant:\nosimertinib > supportive care"),
    ("egfr_wt_osi_not_top", "EGFR wild-type:\nosimertinib not top"),
    ("bsc_control", "supportive-care\ncontrol"),
    ("no_impossible_reversal", "no impossible\nreversal"),
)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=8,
            fontweight="bold", fontstyle="normal", ha="left", va="bottom",
            color=INK, clip_on=False)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sentinel(path: Path) -> pd.DataFrame:
    """Load and validate the broad sentinel score table."""
    frame = pd.read_csv(path)
    score_columns = [f"{model}_primary_pct" for model in SENTINEL_MODELS]
    required = {"context_id", "label", "axis", "core_signal", *score_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Sentinel table is missing columns: {sorted(missing)}")
    if not set(frame["axis"]).issubset({"static", "dynamic"}):
        raise ValueError("Sentinel axis must be static or dynamic")
    values = frame[score_columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not values.stack().between(0, 100).all():
        raise ValueError("Sentinel primary scores must be complete percentages")
    return frame


def load_phase1(paths: tuple[Path, ...]) -> pd.DataFrame:
    """Load one locked phase-1 EGFR gate row per model."""
    frames = [pd.read_csv(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    required = {
        "model",
        "phase1_pass",
        "checks_passed",
        "checks_total",
        "egfr_mut_osi_mo",
        "egfr_wt_osi_mo",
        *[key for key, _ in PHASE1_CHECKS],
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Phase-1 table is missing columns: {sorted(missing)}")
    if frame["model"].duplicated().any():
        raise ValueError("Phase-1 table must contain one row per model")
    absent = set(PHASE1_ORDER) - set(frame["model"])
    if absent:
        raise ValueError(f"Phase-1 table is missing models: {sorted(absent)}")
    return frame.set_index("model").loc[list(PHASE1_ORDER)].reset_index()


def axis_summary(sentinel: pd.DataFrame) -> pd.DataFrame:
    """Return static and dynamic mean correctness for each sentinel model."""
    rows: list[dict[str, object]] = []
    for model in SENTINEL_MODELS:
        column = f"{model}_primary_pct"
        for axis in ("static", "dynamic"):
            rows.append(
                {
                    "model": model,
                    "axis": axis,
                    "mean_primary_pct": float(
                        sentinel.loc[sentinel["axis"].eq(axis), column].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _score_glyph(value: float) -> str:
    if value >= 99:
        return "OK"
    if value <= 1:
        return "×"
    return "~"


def draw_sentinel_matrix(ax: plt.Axes, sentinel: pd.DataFrame) -> None:
    score_columns = [f"{model}_primary_pct" for model in SENTINEL_MODELS]
    matrix = sentinel[score_columns].to_numpy(float)
    cmap = LinearSegmentedColormap.from_list(
        "trust",
        ["#D55E00", "#f7f7f7", "#0072B2"],
    )
    ax.imshow(matrix, aspect="auto", vmin=0, vmax=100, cmap=cmap)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
                row - 0.10,
                _score_glyph(value),
                ha="center",
                va="center",
                fontsize=7,
                weight="bold",
                color="white" if value in {0, 100} else INK,
            )
            ax.text(
                column,
                row + 0.25,
                f"{value:.0f}%",
                ha="center",
                va="center",
                fontsize=5.8,
                color="white" if value in {0, 100} else INK,
            )
    n_static = int(sentinel["axis"].eq("static").sum())
    if 0 < n_static < len(sentinel):
        ax.axhline(n_static - 0.5, color=INK, lw=1.6)
    ax.set_xticks(
        range(len(SENTINEL_MODELS)),
        [SENTINEL_LABEL[model] for model in SENTINEL_MODELS],
    )
    display_labels = (
        "EGFR-mut LUAD, 1L", "PD-L1-high LUAD, 1L", "EGFR-WT LUAD, 1L",
        "MSI-H mCRC", "RAS-mut mCRC", "BRAF-V600E melanoma", "HER2+ breast",
        "HER2− breast", "clear-cell RCC", "HGSOC relapse", "pancreatic: fit vs frail",
        "EGFR-mut LUAD after progression",
    )
    ax.set_yticks(range(len(sentinel)), display_labels)
    ax.tick_params(axis="x", labelsize=5.3, length=0, pad=4)
    ax.tick_params(axis="y", labelsize=5.0, length=0)
    for index, label in enumerate(ax.get_yticklabels()):
        label.set_color(STATIC if index < n_static else DYNAMIC)
    ax.spines[:].set_visible(False)
    ax.set_title(
        "Broad sentinel battery · core treatment primitive on forced-line PFS\n"
        "blue row labels = static · purple = history/host · OK correct · ~ partial · × wrong",
        loc="left",
        fontsize=7.0,
        weight="bold",
        color=INK,
    )
    panel_label(ax, "a")


def draw_axis_summary(
    ax: plt.Axes,
    summary: pd.DataFrame,
    sentinel: pd.DataFrame,
) -> None:
    x = np.arange(len(SENTINEL_MODELS))
    static = (
        summary[summary["axis"].eq("static")]
        .set_index("model")
        .loc[list(SENTINEL_MODELS), "mean_primary_pct"]
        .to_numpy(float)
    )
    dynamic = (
        summary[summary["axis"].eq("dynamic")]
        .set_index("model")
        .loc[list(SENTINEL_MODELS), "mean_primary_pct"]
        .to_numpy(float)
    )
    ax.bar(x - 0.19, static, width=0.36, color=STATIC, label="static")
    ax.bar(x + 0.19, dynamic, width=0.36, color=DYNAMIC, label="dynamic")
    jitter = np.linspace(-0.07, 0.07, 9)
    for index, model in enumerate(SENTINEL_MODELS):
        for offset, axis_name, colour, marker in (
            (-0.19, "static", STATIC, "o"),
            (0.19, "dynamic", DYNAMIC, "s"),
        ):
            values = sentinel.loc[
                sentinel["axis"].eq(axis_name), f"{model}_primary_pct"
            ].to_numpy(float)
            offsets = jitter[:len(values)] if len(values) <= len(jitter) else np.zeros(len(values))
            ax.scatter(index + offset + offsets, values, s=5, marker=marker,
                       facecolor="white", edgecolor=colour, linewidth=0.5, zorder=3)
    for index, values in enumerate(zip(static, dynamic, strict=True)):
        for offset, value in zip((-0.19, 0.19), values, strict=True):
            ax.text(
                index + offset,
                value + 2,
                f"{value:.0f}%",
                ha="center",
                va="bottom",
                fontsize=6.4,
                color=INK,
            )
    ax.set_xticks(
        x,
        [SENTINEL_LABEL[model] for model in SENTINEL_MODELS],
    )
    ax.set_ylim(0, 112)
    ax.set_ylabel("Primitive correct (%)")
    ax.tick_params(labelsize=5.2)
    ax.legend(ncol=2, loc="upper center")
    ax.set_title(
        "History and host reasoning are harder\n"
        "bars = means; points = individual contexts",
        loc="left",
        fontsize=7.0,
        weight="bold",
        color=INK,
    )
    panel_label(ax, "b")


def draw_phase1_gate(ax: plt.Axes, phase1: pd.DataFrame) -> None:
    matrix = phase1[[key for key, _ in PHASE1_CHECKS]].astype(bool).to_numpy().T
    cmap = LinearSegmentedColormap.from_list("binary_gate", ["#fde0cf", "#d8edf7"])
    ax.imshow(matrix.astype(int), aspect="auto", vmin=0, vmax=1, cmap=cmap)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            passed = bool(matrix[row, column])
            ax.text(
                column,
                row,
                "OK" if passed else "×",
                ha="center",
                va="center",
                fontsize=7,
                weight="bold",
                color=PASS if passed else FAIL,
            )
    ax.set_xticks(
        range(len(PHASE1_ORDER)),
        [PHASE1_LABEL[model] for model in PHASE1_ORDER],
    )
    ax.set_yticks(
        range(len(PHASE1_CHECKS)),
        [label for _, label in PHASE1_CHECKS],
    )
    ax.tick_params(labelsize=5.2, length=0)
    ax.spines[:].set_visible(False)
    for column, row in enumerate(phase1.itertuples(index=False)):
        passed = bool(row.phase1_pass)
        ax.text(
            column,
            -0.72,
            f"{'PASS' if passed else 'FAIL'} · {int(row.checks_passed)}/5",
            ha="center",
            va="center",
            fontsize=5.5,
            weight="bold",
            color=PASS if passed else FAIL,
        )
    ax.set_ylim(len(PHASE1_CHECKS) - 0.5, -1.05)
    ax.set_title(
        "Exact-anchor EGFR specificity gate · n=3 per designed cell\n"
        "separate locked assay; five preregistered directional checks",
        loc="left",
        fontsize=7.0,
        weight="bold",
        color=INK,
    )
    panel_label(ax, "c")


def draw_specificity_magnitude(ax: plt.Axes, phase1: pd.DataFrame) -> None:
    for row in phase1.itertuples(index=False):
        model = str(row.model)
        values = [float(row.egfr_mut_osi_mo), float(row.egfr_wt_osi_mo)]
        ax.plot(
            [0, 1],
            values,
            "-o",
            color=PHASE1_COLOR[model],
            lw=2.2,
            ms=6,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=PHASE1_LABEL[model],
        )
        delta = values[0] - values[1]
        ax.text(
            1.04,
            values[1],
            f"{PHASE1_LABEL[model]}  Δ{delta:+.1f} mo",
            ha="left",
            va="center",
            fontsize=5.2,
            color=PHASE1_COLOR[model],
        )
    ax.set_xlim(-0.12, 1.82)
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 1], ["EGFR-mutant", "EGFR wild-type"])
    ax.set_ylabel("Osimertinib median time to progression (months)")
    ax.tick_params(labelsize=5.2)
    ax.set_title(
        "A rank pass is not magnitude calibration\n"
        "biomarker-specific magnitudes diverge",
        loc="left",
        fontsize=7.0,
        weight="bold",
        color=INK,
    )
    panel_label(ax, "d")


def make_figure(
    sentinel: pd.DataFrame,
    summary: pd.DataFrame,
    phase1: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
) -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 6.0,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.0,
        "xtick.labelsize": 5.2,
        "ytick.labelsize": 5.2,
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure = plt.figure(figsize=(7.2047, 5.8), facecolor=PAPER)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.45, 1.0),
        height_ratios=(1.15, 0.85),
        left=0.17,
        right=0.975,
        top=0.84,
        bottom=0.10,
        wspace=0.36,
        hspace=0.44,
    )
    draw_sentinel_matrix(figure.add_subplot(grid[0, 0]), sentinel)
    draw_axis_summary(figure.add_subplot(grid[0, 1]), summary, sentinel)
    draw_phase1_gate(figure.add_subplot(grid[1, 0]), phase1)
    draw_specificity_magnitude(figure.add_subplot(grid[1, 1]), phase1)

    figure.suptitle(
        "Trust is conditional: treatment dynamics must be gated, not assumed",
        x=0.06,
        y=0.965,
        ha="left",
        fontsize=7.0,
        weight="bold",
        color=INK,
    )
    figure.text(
        0.06,
        0.923,
        (
            "Static biomarker ordering is often reproduced; history- and host-dependent "
            "reasoning is harder, and binary correctness does not calibrate magnitude."
        ),
        ha="left",
        fontsize=5.5,
        color=MUTED,
    )
    figure.text(
        0.965,
        0.03,
        (
            "Model-implied endpoints; no observed patient treatment effects. "
            "Every pass and failure is retained."
        ),
        ha="right",
        fontsize=5.2,
        color=MUTED,
    )
    figure.savefig(out_png, dpi=300, facecolor=figure.get_facecolor())
    figure.savefig(out_pdf, facecolor=figure.get_facecolor())
    figure.savefig(out_pdf.with_suffix(".svg"), facecolor=figure.get_facecolor())
    plt.close(figure)


def write_readout(
    out_dir: Path,
    sentinel: pd.DataFrame,
    summary: pd.DataFrame,
    phase1: pd.DataFrame,
    sentinel_path: Path,
    phase1_paths: tuple[Path, ...],
) -> None:
    axis_payload: dict[str, dict[str, float]] = {}
    for model in SENTINEL_MODELS:
        rows = summary[summary["model"].eq(model)].set_index("axis")
        axis_payload[model] = {
            "static_mean_pct": float(rows.loc["static", "mean_primary_pct"]),
            "dynamic_mean_pct": float(rows.loc["dynamic", "mean_primary_pct"]),
        }
    phase_payload: dict[str, dict[str, object]] = {}
    for row in phase1.itertuples(index=False):
        model = str(row.model)
        phase_payload[model] = {
            "pass": bool(row.phase1_pass),
            "checks_passed": int(row.checks_passed),
            "checks_total": int(row.checks_total),
            "egfr_mut_osi_mo": float(row.egfr_mut_osi_mo),
            "egfr_wt_osi_mo": float(row.egfr_wt_osi_mo),
            "biomarker_delta_mo": float(row.egfr_mut_osi_mo - row.egfr_wt_osi_mo),
            "checks": {
                key: bool(getattr(row, key))
                for key, _ in PHASE1_CHECKS
            },
        }
    payload = {
        "interpretation": {
            "primary": (
                "treatment-reasoning validity is conditional on model and context; "
                "static ordering is more reliable than dynamic history/host reasoning"
            ),
            "calibration_limit": (
                "binary direction/rank gates do not calibrate effect magnitude"
            ),
            "causal_status": (
                "all endpoints are model-implied; no panel estimates a human causal effect"
            ),
            "assay_separation": (
                "the broad sentinel and exact-anchor phase-1 assays are reported "
                "separately and never pooled"
            ),
        },
        "sentinel": {
            "n_contexts": int(len(sentinel)),
            "n_static": int(sentinel["axis"].eq("static").sum()),
            "n_dynamic": int(sentinel["axis"].eq("dynamic").sum()),
            "axis_means": axis_payload,
        },
        "phase1_egfr": phase_payload,
        "provenance": {
            "git_commit_at_build": _git_commit(),
            "sentinel_source": str(sentinel_path.relative_to(ROOT)),
            "sentinel_source_sha256": _sha256(sentinel_path),
            "phase1_sources": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": _sha256(path),
                }
                for path in phase1_paths
            ],
        },
    }
    (out_dir / "readout.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Interventional trust map — readout",
        "",
        "## Broad sentinel battery",
        "",
        "| model | static mean | dynamic history/host mean |",
        "|---|---:|---:|",
    ]
    for model in SENTINEL_MODELS:
        values = axis_payload[model]
        lines.append(
            f"| {SENTINEL_LABEL[model].replace(chr(10), ' ')} | "
            f"{values['static_mean_pct']:.1f}% | "
            f"{values['dynamic_mean_pct']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Exact-anchor EGFR phase-1 gate",
            "",
            "| model | verdict | checks | osimertinib TTP, EGFR-mut | EGFR-WT | biomarker difference |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for model in PHASE1_ORDER:
        values = phase_payload[model]
        lines.append(
            f"| {PHASE1_LABEL[model]} | "
            f"{'PASS' if values['pass'] else 'FAIL'} | "
            f"{values['checks_passed']}/{values['checks_total']} | "
            f"{values['egfr_mut_osi_mo']:.2f} mo | "
            f"{values['egfr_wt_osi_mo']:.2f} mo | "
            f"{values['biomarker_delta_mo']:+.2f} mo |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            (
                "- These are direction/rank gates on model-generated endpoints. "
                "They are not human outcome calibration or causal treatment effects."
            ),
            (
                "- The broad sentinel and exact-anchor phase-1 assays have different "
                "designs and model rosters; they are deliberately not pooled."
            ),
            (
                "- A pass is necessary evidence of treatment specificity, not proof "
                "that its magnitude is biologically calibrated."
            ),
            "",
            "## Files",
            "",
            "- `fig_interventional_trust_map.png` / `.pdf`: paper-facing figure.",
            "- `sentinel_scores_used.csv`: broad sentinel source rows.",
            "- `egfr_gate_scores_used.csv`: exact-anchor phase-1 source rows.",
            "- `axis_summary.csv`: static/dynamic means shown in Panel B.",
            "",
        ]
    )
    (out_dir / "readout.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sentinel", type=Path, default=DEFAULT_SENTINEL)
    parser.add_argument(
        "--phase1",
        type=Path,
        nargs=3,
        default=DEFAULT_PHASE1,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase1_paths = tuple(args.phase1)
    for path in (args.sentinel, *phase1_paths):
        if not path.exists():
            raise FileNotFoundError(path)
    sentinel = load_sentinel(args.sentinel)
    phase1 = load_phase1(phase1_paths)
    summary = axis_summary(sentinel)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel.to_csv(out_dir / "sentinel_scores_used.csv", index=False)
    phase1.to_csv(out_dir / "egfr_gate_scores_used.csv", index=False)
    summary.to_csv(out_dir / "axis_summary.csv", index=False)
    make_figure(
        sentinel,
        summary,
        phase1,
        out_dir / "fig_interventional_trust_map.png",
        out_dir / "fig_interventional_trust_map.pdf",
    )
    write_readout(
        out_dir,
        sentinel,
        summary,
        phase1,
        args.sentinel,
        phase1_paths,
    )
    print(f"Wrote {out_dir / 'fig_interventional_trust_map.png'}")
    print(f"Wrote {out_dir / 'readout.md'}")


if __name__ == "__main__":
    main()
