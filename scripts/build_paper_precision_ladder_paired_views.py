#!/usr/bin/env python3
"""Pair a state-space precision ladder with familiar oncology summaries.

One fixed, held-out generated EGFR-mutant future is kept visible while the
reference population narrows in the canonical state-first order:

    assigned action -> current state -> biomarker context -> ordered history.

For every rung, the top panel shows the same PCA anchor with the distributions
of three-month displacements under forced osimertinib and chemotherapy. The
bottom panel shows Kaplan-Meier curves for first-line PFS from exactly the same
reference futures. A companion forest plot reports the corresponding 24-month
restricted-mean PFS contrast, support and held-out movement error.

All paths are stored model outputs. Forced arms are interventions on Q_theta,
not human causal effects; the held-out path is not a real patient prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_paper_state_feature_ledger as ledger  # noqa: E402
from build_paper_state_embedding_bakeoff import load_serialized_full_pca  # noqa: E402
import build_paper_state_map as state_map  # noqa: E402


PHASE1_RUN = "interventional_phase1_sonnet_full_v1c"
STATE_DIR = ROOT / "outputs" / "paper_state_map"
DEFAULT_OUT = ROOT / "outputs" / "paper_precision_ladder_paired_views"
MONTH_DAYS = 30.4375
MOVEMENT_HORIZON_MONTHS = 3.0
RMST_HORIZON_MONTHS = 24.0
SEED = 20260731

EXEMPLAR = {
    "context": "EGFR_1L",
    "state": "S2",
    "history": "H1",
    "future_index": 1,
    "arm": "osimertinib",
}
RUNG_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Population", "assigned action only", ()),
    ("Current state", "+ standard metastatic state", ("state",)),
    ("Biomarker", "+ EGFR-mutant context", ("state", "context")),
    ("Ordered history", "+ intermediate prior course", ("state", "context", "history")),
)
ARMS = ("osimertinib", "chemo")
ARM_COLOR = {"osimertinib": "#0f8b83", "chemo": "#d07a26"}
ARM_LABEL = {"osimertinib": "osimertinib", "chemo": "chemotherapy"}
INK = "#172033"
MUTED = "#667085"
GRID = "#d9dee7"
PALE = "#eef1f5"
EXACT_BG = "#f3eff9"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def locf_row(course: pd.DataFrame, target_day: float) -> pd.Series:
    """Return the last state observed no later than ``target_day``."""
    ordered = course.sort_values(["days_from_diagnosis", "snapshot_index"])
    eligible = ordered[ordered["days_from_diagnosis"].le(target_day + 1e-9)]
    return ordered.iloc[0] if eligible.empty else eligible.iloc[-1]


def load_phase1() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Project Phase-1 states through the frozen twenty-feature PCA."""
    tables = state_map.load_reference_tables((PHASE1_RUN,))
    active, _active_manifest = ledger.load_active_condition_features((PHASE1_RUN,))
    states, _state_manifest = state_map.build_state_matrix(tables, active)
    pca = load_serialized_full_pca(STATE_DIR / "state_map_contract.json")
    coordinates = pca.transform(states)
    states = states.copy()
    states["pc1"] = coordinates[:, 0]
    states["pc2"] = coordinates[:, 1]

    labels = pd.read_csv(STATE_DIR / "phase1_p0_records.csv")
    label_columns = [
        "_gid",
        "context",
        "state",
        "history",
        "arm",
        "cutoff_days",
    ]
    states = states.merge(
        labels[label_columns],
        on="_gid",
        how="inner",
        validate="many_to_one",
    )
    return states, tables["endpoints"].copy(), labels


def trajectory_outcomes(
    states: pd.DataFrame,
    endpoints: pd.DataFrame,
    *,
    horizon_months: float = MOVEMENT_HORIZON_MONTHS,
) -> tuple[pd.DataFrame, int]:
    """Calculate equal-horizon LOCF movement and raw first-line PFS."""
    endpoint_map = endpoints.set_index("_gid")["pfs_line_1_days"].to_dict()
    rows: list[dict[str, Any]] = []
    dropped = 0
    for gid, course in states.groupby("_gid", sort=False):
        course = course.sort_values(["days_from_diagnosis", "snapshot_index"])
        first = course.iloc[0]
        if str(first["arm"]) not in ARMS:
            continue
        cutoff = float(first["cutoff_days"])
        target = cutoff + horizon_months * MONTH_DAYS
        if float(course["days_from_diagnosis"].max()) < target - 1e-9:
            dropped += 1
            continue
        pfs_days = pd.to_numeric(pd.Series([endpoint_map.get(gid)]), errors="coerce").iloc[0]
        if not np.isfinite(pfs_days):
            dropped += 1
            continue
        anchor = locf_row(course, cutoff)
        future = locf_row(course, target)
        trajectory_id = str(first["trajectory_id"])
        match = pd.Series([trajectory_id]).str.extract(r"__f(\d+)$", expand=False).iloc[0]
        rows.append(
            {
                "_gid": gid,
                "trajectory_id": trajectory_id,
                "arm": str(first["arm"]),
                "context": str(first["context"]),
                "state": str(first["state"]),
                "history": str(first["history"]),
                "future_index": int(match) if pd.notna(match) else -1,
                "x0": float(anchor["pc1"]),
                "y0": float(anchor["pc2"]),
                "dx": float(future["pc1"] - anchor["pc1"]),
                "dy": float(future["pc2"] - anchor["pc2"]),
                "pfs_months": float(pfs_days) / MONTH_DAYS,
            }
        )
    return pd.DataFrame(rows), dropped


def select_exemplar(outcomes: pd.DataFrame) -> pd.Series:
    """Select the declared grammar exemplar without looking at its outcome."""
    mask = np.ones(len(outcomes), dtype=bool)
    for key, value in EXEMPLAR.items():
        mask &= outcomes[key].eq(value).to_numpy()
    selected = outcomes.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"Expected one declared exemplar; found {len(selected)}")
    return selected.iloc[0]


def peer_mask(
    outcomes: pd.DataFrame,
    exemplar: pd.Series,
    keys: Sequence[str],
    *,
    arm: str,
) -> pd.Series:
    """Return one balanced reference arm, excluding the held-out exact pair."""
    mask = outcomes["arm"].eq(arm)
    for key in keys:
        mask &= outcomes[key].eq(exemplar[key])
    heldout_pair = (
        outcomes["context"].eq(exemplar["context"])
        & outcomes["state"].eq(exemplar["state"])
        & outcomes["history"].eq(exemplar["history"])
        & outcomes["future_index"].eq(exemplar["future_index"])
    )
    return mask & ~heldout_pair


def rmst(values: np.ndarray, horizon_months: float = RMST_HORIZON_MONTHS) -> float:
    """Restricted mean for complete, uncensored generated PFS values."""
    data = np.asarray(values, dtype=float)
    if len(data) == 0:
        return float("nan")
    return float(np.minimum(data, horizon_months).mean())


def bootstrap_rmst_contrast(
    treatment: np.ndarray,
    control: np.ndarray,
    *,
    horizon_months: float = RMST_HORIZON_MONTHS,
    n_bootstrap: int = 2_000,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Bootstrap the model-future RMST contrast between two forced arms."""
    treatment = np.asarray(treatment, dtype=float)
    control = np.asarray(control, dtype=float)
    estimate = rmst(treatment, horizon_months) - rmst(control, horizon_months)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled_treatment = treatment[rng.integers(0, len(treatment), len(treatment))]
        sampled_control = control[rng.integers(0, len(control), len(control))]
        draws[index] = rmst(sampled_treatment, horizon_months) - rmst(
            sampled_control,
            horizon_months,
        )
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(estimate), float(low), float(high)


def leave_one_out_movement(
    outcomes: pd.DataFrame,
    keys: Sequence[str],
) -> tuple[float, float, int]:
    """Score peer-mean two-dimensional movement without self-prediction."""
    errors: list[float] = []
    peers: list[int] = []
    for index, row in outcomes.iterrows():
        mask = outcomes["arm"].eq(row["arm"]) & outcomes.index.to_series().ne(index)
        for key in keys:
            mask &= outcomes[key].eq(row[key])
        reference = outcomes.loc[mask, ["dx", "dy"]]
        if reference.empty:
            continue
        residual = reference.mean().to_numpy(float) - row[["dx", "dy"]].to_numpy(float)
        errors.append(float(residual @ residual))
        peers.append(len(reference))
    return float(np.sqrt(np.mean(errors))), float(np.median(peers)), len(errors)


def build_rung_metrics(
    outcomes: pd.DataFrame,
    exemplar: pd.Series,
) -> pd.DataFrame:
    """Measure support, movement agreement and conventional RMST at each rung."""
    rows: list[dict[str, Any]] = []
    actual = exemplar[["dx", "dy"]].to_numpy(float)
    for rung_index, (rung, description, keys) in enumerate(RUNG_SPECS):
        arm_frames = {
            arm: outcomes.loc[peer_mask(outcomes, exemplar, keys, arm=arm)].copy()
            for arm in ARMS
        }
        if any(frame.empty for frame in arm_frames.values()):
            raise ValueError(f"Unsupported arm at rung {rung}")
        osi_mean = arm_frames["osimertinib"][["dx", "dy"]].mean().to_numpy(float)
        estimate, low, high = bootstrap_rmst_contrast(
            arm_frames["osimertinib"]["pfs_months"].to_numpy(float),
            arm_frames["chemo"]["pfs_months"].to_numpy(float),
            seed=SEED + rung_index,
        )
        loo_rmse, median_peers, n_scored = leave_one_out_movement(outcomes, keys)
        rows.append(
            {
                "rung_index": rung_index,
                "rung": rung,
                "description": description,
                "conditioning_keys": "+".join(keys) if keys else "action only",
                "n_osimertinib": len(arm_frames["osimertinib"]),
                "n_chemo": len(arm_frames["chemo"]),
                "osimertinib_mean_dx": osi_mean[0],
                "osimertinib_mean_dy": osi_mean[1],
                "exemplar_movement_error": float(np.linalg.norm(osi_mean - actual)),
                "loo_movement_rmse": loo_rmse,
                "median_loo_peers": median_peers,
                "n_loo_scored": n_scored,
                "median_pfs_osimertinib": float(
                    arm_frames["osimertinib"]["pfs_months"].median()
                ),
                "median_pfs_chemo": float(arm_frames["chemo"]["pfs_months"].median()),
                "delta_rmst_24_months": estimate,
                "delta_rmst_low": low,
                "delta_rmst_high": high,
            }
        )
    return pd.DataFrame(rows)


def _add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    delta: tuple[float, float],
    *,
    color: str,
    linewidth: float,
    alpha: float,
    mutation_scale: float,
    zorder: int,
) -> None:
    end = (start[0] + delta[0], start[1] + delta[1])
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            zorder=zorder,
        )
    )


def _display_sample(frame: pd.DataFrame, maximum: int = 30) -> pd.DataFrame:
    """Choose a deterministic, evenly spaced visual subset; statistics use all rows."""
    ordered = frame.sort_values("_gid")
    if len(ordered) <= maximum:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, maximum).round().astype(int)
    return ordered.iloc[np.unique(indices)]


def draw_paired_ladder(
    outcomes: pd.DataFrame,
    exemplar: pd.Series,
    atlas: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Draw state-space fans directly above conventional survival curves."""
    figure, axes = plt.subplots(2, 4, figsize=(18.6, 9.6))
    phase1_x = np.r_[outcomes["x0"], outcomes["x0"] + outcomes["dx"]]
    phase1_y = np.r_[outcomes["y0"], outcomes["y0"] + outcomes["dy"]]
    x_low, x_high = np.quantile(phase1_x, [0.005, 0.995])
    y_low, y_high = np.quantile(phase1_y, [0.005, 0.995])
    x_pad = 0.04 * (x_high - x_low)
    y_pad = 0.04 * (y_high - y_low)
    limits = (x_low - x_pad, x_high + x_pad, y_low - y_pad, y_high + y_pad)
    anchor = (float(exemplar["x0"]), float(exemplar["y0"]))
    actual = (float(exemplar["dx"]), float(exemplar["dy"]))

    for column, (rung, description, keys) in enumerate(RUNG_SPECS):
        map_axis = axes[0, column]
        km_axis = axes[1, column]
        if column == len(RUNG_SPECS) - 1:
            map_axis.set_facecolor(EXACT_BG)
            km_axis.set_facecolor(EXACT_BG)
        map_axis.scatter(
            atlas.iloc[::32]["pc1"],
            atlas.iloc[::32]["pc2"],
            s=2,
            color="#dfe3e9",
            alpha=0.25,
            linewidth=0,
            rasterized=True,
        )
        arm_frames: dict[str, pd.DataFrame] = {}
        for arm in ARMS:
            peers = outcomes.loc[peer_mask(outcomes, exemplar, keys, arm=arm)].copy()
            arm_frames[arm] = peers
            map_axis.scatter(
                peers["x0"],
                peers["y0"],
                s=20,
                color=ARM_COLOR[arm],
                alpha=0.30,
                edgecolors="none",
                zorder=2,
            )
            for peer in _display_sample(peers).itertuples(index=False):
                _add_arrow(
                    map_axis,
                    anchor,
                    (float(peer.dx), float(peer.dy)),
                    color=ARM_COLOR[arm],
                    linewidth=0.78,
                    alpha=0.22,
                    mutation_scale=7,
                    zorder=3,
                )
            mean = peers[["dx", "dy"]].mean().to_numpy(float)
            _add_arrow(
                map_axis,
                anchor,
                (float(mean[0]), float(mean[1])),
                color=ARM_COLOR[arm],
                linewidth=2.5,
                alpha=0.95,
                mutation_scale=13,
                zorder=6,
            )
        _add_arrow(
            map_axis,
            anchor,
            actual,
            color=INK,
            linewidth=2.2,
            alpha=1.0,
            mutation_scale=13,
            zorder=8,
        )
        map_axis.scatter(
            [anchor[0]],
            [anchor[1]],
            marker="*",
            s=165,
            color=INK,
            edgecolor="white",
            linewidth=0.8,
            zorder=9,
        )
        if np.hypot(*actual) <= 1e-10:
            map_axis.scatter(
                [anchor[0]],
                [anchor[1]],
                s=210,
                facecolors="none",
                edgecolors=INK,
                linewidth=1.3,
                zorder=10,
            )
        n_arm = len(arm_frames["osimertinib"])
        map_axis.set_title(
            f"{column + 1}  {rung}\n{description} · {n_arm} peers/arm",
            fontsize=10.5,
            fontweight="bold",
            color=INK,
        )
        map_axis.set_xlim(limits[:2])
        map_axis.set_ylim(limits[2:])
        map_axis.grid(color=GRID, lw=0.55, alpha=0.6)
        map_axis.spines[["top", "right"]].set_visible(False)
        map_axis.set_xlabel("PC1")
        if column == 0:
            map_axis.set_ylabel("State-space view\nPC2")
        else:
            map_axis.set_yticklabels([])

        for arm in ARMS:
            peers = arm_frames[arm]
            fitter = KaplanMeierFitter(label=ARM_LABEL[arm])
            fitter.fit(
                peers["pfs_months"],
                event_observed=np.ones(len(peers), dtype=int),
            )
            fitter.plot_survival_function(
                ax=km_axis,
                color=ARM_COLOR[arm],
                linewidth=2.0,
                ci_show=True,
                ci_alpha=0.10,
            )
        km_axis.axvline(
            exemplar["pfs_months"],
            color=INK,
            linestyle=":",
            linewidth=1.5,
            label="held-out factual PFS" if column == 0 else None,
        )
        km_axis.set_xlim(0, 36)
        km_axis.set_ylim(0, 1.03)
        km_axis.grid(color=GRID, lw=0.55, alpha=0.65)
        km_axis.spines[["top", "right"]].set_visible(False)
        km_axis.set_xlabel("months from treatment anchor")
        if column == 0:
            km_axis.set_ylabel("Classical oncology view\nprogression-free fraction")
        else:
            km_axis.set_ylabel("")
            km_axis.set_yticklabels([])
        if column == 0:
            km_axis.legend(frameon=False, fontsize=8, loc="upper right")
        else:
            legend = km_axis.get_legend()
            if legend is not None:
                legend.remove()

    state_legend = [
        Line2D([0], [0], color=ARM_COLOR[arm], lw=2.6, label=f"peer mean: {ARM_LABEL[arm]}")
        for arm in ARMS
    ] + [Line2D([0], [0], color=INK, lw=2.2, label="held-out factual movement")]
    axes[0, 0].legend(handles=state_legend, frameon=False, fontsize=7.8, loc="upper right")
    figure.suptitle(
        "The same generated patient, progressively localized — state-space futures above, classical survival below",
        fontsize=15,
        fontweight="bold",
        color=INK,
        y=0.995,
    )
    figure.text(
        0.5,
        0.012,
        "Black star/path: one held-out generated osimertinib future. Thin coloured arrows: peer displacements translated to its anchor; thick arrows: peer means. Exact-anchor replication is model-implied, not human evidence.",
        ha="center",
        color=MUTED,
        fontsize=9.1,
    )
    figure.tight_layout(rect=(0.02, 0.045, 0.995, 0.95))
    figure.savefig(out_dir / "fig_precision_ladder_state_vs_survival.png", dpi=210, bbox_inches="tight")
    figure.savefig(out_dir / "fig_precision_ladder_state_vs_survival.pdf", bbox_inches="tight")
    plt.close(figure)


def draw_forest_summary(metrics: pd.DataFrame, out_dir: Path) -> None:
    """Draw the conventional effect forest beside support and prediction error."""
    figure, axes = plt.subplots(1, 3, figsize=(17.4, 5.6))
    y = np.arange(len(metrics))[::-1]
    axis = axes[0]
    for position, row in zip(y, metrics.itertuples(index=False)):
        axis.plot(
            [row.delta_rmst_low, row.delta_rmst_high],
            [position, position],
            color=INK,
            linewidth=1.6,
        )
        axis.scatter(
            [row.delta_rmst_24_months],
            [position],
            s=68,
            color="#5f4b9b",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    axis.axvline(0, color=MUTED, linestyle="--", linewidth=1.1)
    axis.set_yticks(y, metrics["rung"])
    axis.set_xlabel("Δ 24-month RMST, months\nosimertinib − chemotherapy")
    axis.set_title("A  Conventional forest\none scalar effect per rung", loc="left", fontweight="bold")

    axis = axes[1]
    axis.plot(
        metrics["rung_index"],
        metrics["n_osimertinib"],
        "-o",
        color="#b4233d",
        linewidth=2.4,
        markersize=8,
    )
    for row in metrics.itertuples(index=False):
        axis.annotate(
            f"{row.n_osimertinib}/arm",
            (row.rung_index, row.n_osimertinib),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=INK,
        )
    axis.set_yscale("log")
    axis.set_xticks(metrics["rung_index"], metrics["rung"], rotation=18, ha="right")
    axis.set_ylabel("comparable model futures (log)")
    axis.set_title("B  Precision spends support", loc="left", fontweight="bold")

    axis = axes[2]
    axis.plot(
        metrics["rung_index"],
        metrics["loo_movement_rmse"],
        "-o",
        color="#2f7f89",
        linewidth=2.4,
        markersize=8,
        label="all futures: leave-one-out RMSE",
    )
    axis.plot(
        metrics["rung_index"],
        metrics["exemplar_movement_error"],
        "--s",
        color=INK,
        linewidth=1.8,
        markersize=6,
        label="displayed held-out future",
    )
    axis.set_xticks(metrics["rung_index"], metrics["rung"], rotation=18, ha="right")
    axis.set_ylabel("3-month PC1/PC2 movement error")
    axis.set_title("C  Finer is not automatically better", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8.3)

    for axis in axes:
        axis.grid(color=GRID, lw=0.6, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "The scalar effect estimate changes as support and individual agreement change",
        fontsize=14.5,
        fontweight="bold",
        color=INK,
        y=0.995,
    )
    figure.text(
        0.5,
        0.012,
        "Forest intervals bootstrap generated futures, not trial participants. The curve is a model-assay demonstration of the precision–support tradeoff.",
        ha="center",
        fontsize=9.2,
        color=MUTED,
    )
    figure.tight_layout(rect=(0.02, 0.05, 0.99, 0.94))
    figure.savefig(out_dir / "fig_precision_ladder_forest_summary.png", dpi=210, bbox_inches="tight")
    figure.savefig(out_dir / "fig_precision_ladder_forest_summary.pdf", bbox_inches="tight")
    plt.close(figure)


def write_readout(
    out_dir: Path,
    outcomes: pd.DataFrame,
    exemplar: pd.Series,
    metrics: pd.DataFrame,
    dropped: int,
) -> None:
    """Write a clinician-facing interpretation with causal and support guards."""
    lines = [
        "# Precision ladder paired views",
        "",
        "## Fixed exemplar",
        "",
        "The displayed future was fixed by design as EGFR-mutant first-line, "
        "standard metastatic state S2, intermediate history H1, osimertinib, future 1. "
        "It was not selected by its movement or PFS. Its exact anchor/future index was "
        "held out from both arms at every rung.",
        "",
        f"Held-out factual 3-month movement: ΔPC1 {exemplar['dx']:+.4f}, "
        f"ΔPC2 {exemplar['dy']:+.4f}; model-implied first-line PFS "
        f"{exemplar['pfs_months']:.1f} months.",
        "",
        "## Same subsets, two visual languages",
        "",
        "| rung | peers/arm | held-out movement error | all-future LOO RMSE | "
        "median PFS osi / chemo | ΔRMST24 (95% bootstrap interval) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.rung} | {row.n_osimertinib} | {row.exemplar_movement_error:.4f} | "
            f"{row.loo_movement_rmse:.4f} | {row.median_pfs_osimertinib:.1f} / "
            f"{row.median_pfs_chemo:.1f} | {row.delta_rmst_24_months:+.2f} "
            f"[{row.delta_rmst_low:+.2f}, {row.delta_rmst_high:+.2f}] |"
        )
    lines += [
        "",
        f"{len(outcomes)} generated futures had at least {MOVEMENT_HORIZON_MONTHS:g} "
        f"months of state follow-up; {dropped} were excluded before the analysis.",
        "",
        "## What the comparison says",
        "",
        "- Kaplan-Meier and forest displays answer familiar scalar questions: how long "
        "control persists and the average restricted-mean contrast.",
        "- The state-space view shows what those summaries discard: starting-state "
        "support, direction, spread and the held-out future's concordance with its peers.",
        "- The coloured peer vectors are translated to the held-out anchor solely for "
        "visual comparison. At coarse rungs they are not a local individual field.",
        "- More conditioning does not guarantee monotonic improvement. The empirical "
        "question sharpens while the number of comparable futures necessarily collapses.",
        "- The exact-anchor column is the point where repeated model simulation supplies "
        "replicates that an observational cohort cannot. It increases query specificity, "
        "not causal evidence.",
        "",
        "## Evidence boundary",
        "",
        "Forced treatment arms are interventions on the stored model-implied law Q_theta. "
        "They permit an internally controlled model contrast, not a human treatment effect. "
        "The survival curves contain complete generated endpoints and no censoring; their "
        "confidence bands and forest intervals describe Monte Carlo futures, not biological "
        "or trial uncertainty.",
        "",
    ]
    (out_dir / "readout.md").write_text("\n".join(lines))


def build(out_dir: Path = DEFAULT_OUT) -> Mapping[str, Any]:
    """Build both paired-view exercises and their auditable tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    states, endpoints, labels = load_phase1()
    outcomes, dropped = trajectory_outcomes(states, endpoints)
    exemplar = select_exemplar(outcomes)
    metrics = build_rung_metrics(outcomes, exemplar)
    atlas = pd.read_csv(STATE_DIR / "state_projection.csv.gz", usecols=["pc1", "pc2"])

    outcomes.to_csv(out_dir / "phase1_trajectory_outcomes.csv.gz", index=False)
    metrics.to_csv(out_dir / "rung_metrics.csv", index=False)
    pd.DataFrame([exemplar]).to_csv(out_dir / "heldout_exemplar.csv", index=False)
    draw_paired_ladder(outcomes, exemplar, atlas, out_dir)
    draw_forest_summary(metrics, out_dir)
    write_readout(out_dir, outcomes, exemplar, metrics, dropped)

    derivation = ROOT / "runs" / PHASE1_RUN / "derivation"
    source_hashes = {
        filename: _sha256(derivation / filename)
        for filename in (
            "baseline_rows.csv",
            "snapshot_rows.csv",
            "disease_site_rows.csv",
            "event_rows.csv",
            "endpoint_rows.csv",
        )
    }
    source_hashes.update(
        {
            "state_map_contract.json": _sha256(STATE_DIR / "state_map_contract.json"),
            "phase1_p0_records.csv": _sha256(STATE_DIR / "phase1_p0_records.csv"),
        }
    )
    manifest = {
        "artifact": "paper_precision_ladder_paired_views_v1",
        "git_commit": _git_commit(),
        "script_sha256": _sha256(Path(__file__)),
        "source_sha256": source_hashes,
        "phase1_run": PHASE1_RUN,
        "n_futures": int(len(outcomes)),
        "dropped_short_followup_or_missing_pfs": int(dropped),
        "movement_horizon_months": MOVEMENT_HORIZON_MONTHS,
        "rmst_horizon_months": RMST_HORIZON_MONTHS,
        "conditioning_order": [item[0] for item in RUNG_SPECS],
        "exemplar": {key: str(exemplar[key]) for key in EXEMPLAR},
        "heldout_pair_excluded_from_both_arms": True,
        "state_map_role": "linear display of movement",
        "survival_role": "classical scalar duration summary",
        "causal_grade": "model_interventional_not_human_causal",
        "exact_anchor_is_model_implied": True,
        "human_counterfactual_claim": False,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build(args.out_dir)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
