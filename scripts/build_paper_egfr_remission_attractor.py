#!/usr/bin/env python3
"""Build the paper-facing finite-horizon remission-attractor figure.

The figure operationalises a clinical attractor without assuming a physical
potential:

1. pull/capture: model-implied futures enter a clinically defined remission
   phenotype after a shared treatment decision;
2. persistence: futures remain in that phenotype without first-line exit over
   a finite horizon.

The remission definition is expressed in raw clinical variables and is
treatment-neutral. The PCA is used only to display the corresponding paths.
All results are from the locked Sonnet Phase-1 model assay; they are not human
causal-effect estimates or validated individual predictions.

Run:
  uv run python scripts/build_paper_egfr_remission_attractor.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from build_paper_egfr_anchored_fans import (  # noqa: E402
    ARM_COLOR,
    ARM_SHORT,
    ARMS,
    CONTEXTS,
    CONTEXT_LABEL,
    DEFAULT_ENDPOINTS,
    DEFAULT_RUN,
    EXIT,
    GRID,
    INK,
    MAP_ID,
    MONTH_DAYS,
    MUTED,
    PAPER,
    ROOT,
    anchor_position,
    cutoff_days_from_specs,
    fit_display_map,
    git_commit,
    load_atlas_snapshots,
    load_phase1,
    path_until,
    position_at_day,
    sha256,
)


DEFAULT_OUT = ROOT / "outputs" / "paper_egfr_remission_attractor"
LANDMARKS = (0.0, 3.0, 6.0, 9.0, 12.0, 18.0)
CURVE_STEP_MONTHS = 0.25
EXEMPLAR_STATE = "S2"
EXEMPLAR_HISTORY = "H1"
EXEMPLAR_LABEL = "standard metastatic state · rapid pre-treatment growth"
ACTIVE_ARMS = ("chemo", "osimertinib")
REMISSION_COLOR = "#2a9d6f"
SEED = 20260729


def clinical_remission(
    *,
    first_line_active: bool,
    tumour_burden: float,
    ecog: float,
    symptoms: float,
) -> bool:
    """Treatment-neutral raw-feature definition of clinical remission."""
    return bool(
        first_line_active
        and tumour_burden <= 1.0
        and ecog <= 1.0
        and symptoms <= 1.0
    )


def clinical_control(
    *,
    first_line_active: bool,
    tumour_burden: float,
    ecog: float,
    symptoms: float,
) -> bool:
    """Broader disease-control definition used only as a sensitivity check."""
    return bool(
        first_line_active
        and tumour_burden <= 2.0
        and ecog <= 2.0
        and symptoms <= 2.0
    )


def state_at_day(course: pd.DataFrame, day: float) -> pd.Series:
    """Carry the last observed clinical state forward without future leakage."""
    ordered = course.sort_values(
        ["days_from_diagnosis", "snapshot_index"]
    ).reset_index(drop=True)
    observed = ordered[ordered["days_from_diagnosis"] <= day]
    if observed.empty:
        return ordered.iloc[0]
    return observed.iloc[-1]


def build_attractor_table(
    snapshots: pd.DataFrame,
    endpoints: pd.DataFrame,
    run_root: Path,
    horizons: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one state/occupancy row per trajectory and finite horizon."""
    meta = snapshots[
        ["trajectory_id", "context", "state", "history", "arm"]
    ].drop_duplicates("trajectory_id")
    meta["cutoff_days"] = cutoff_days_from_specs(run_root, meta["trajectory_id"])
    meta = meta.merge(
        endpoints[
            [
                "trajectory_id",
                "endpoint_kind",
                "endpoint_mo",
                "used_death_fallback",
            ]
        ],
        on="trajectory_id",
        how="inner",
        validate="one_to_one",
    )

    rows: list[dict[str, object]] = []
    for item in meta.itertuples(index=False):
        course = snapshots[snapshots["trajectory_id"].eq(item.trajectory_id)]
        for horizon in horizons:
            active = float(item.endpoint_mo) > float(horizon)
            elapsed = min(float(item.endpoint_mo), float(horizon))
            day = float(item.cutoff_days) + elapsed * MONTH_DAYS
            if np.isclose(horizon, 0.0):
                x, y = anchor_position(course, float(item.cutoff_days))
            else:
                x, y = position_at_day(course, day)
            state = state_at_day(course, day)
            burden = float(state["tumour_burden_band"])
            ecog = float(state["ecog_ps"])
            symptoms = float(state["symptom_burden_ord"])
            rows.append(
                {
                    "trajectory_id": item.trajectory_id,
                    "context": item.context,
                    "state": item.state,
                    "history": item.history,
                    "arm": item.arm,
                    "cutoff_days": float(item.cutoff_days),
                    "endpoint_kind": item.endpoint_kind,
                    "endpoint_mo": float(item.endpoint_mo),
                    "used_death_fallback": bool(item.used_death_fallback),
                    "horizon_months": float(horizon),
                    "first_line_active": active,
                    "exit_by_horizon": not active,
                    "tumour_burden_score": burden,
                    "ecog": ecog,
                    "symptom_score": symptoms,
                    "clinical_remission": clinical_remission(
                        first_line_active=active,
                        tumour_burden=burden,
                        ecog=ecog,
                        symptoms=symptoms,
                    ),
                    "clinical_control": clinical_control(
                        first_line_active=active,
                        tumour_burden=burden,
                        ecog=ecog,
                        symptoms=symptoms,
                    ),
                    "tumour_only_remission": bool(active and burden <= 1.0),
                    "map_x": x,
                    "map_y": y,
                }
            )
    table = pd.DataFrame(rows)
    anchor_audit = (
        table[np.isclose(table["horizon_months"], 0.0)]
        .groupby(["context", "state", "history"], as_index=False)
        .agg(
            n_trajectories=("trajectory_id", "nunique"),
            n_arms=("arm", "nunique"),
            x_range=("map_x", lambda values: float(values.max() - values.min())),
            y_range=("map_y", lambda values: float(values.max() - values.min())),
        )
    )
    return table, anchor_audit


def occupancy_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Summarise unconditional remission occupancy and first-line residence."""
    return (
        table.groupby(["context", "arm", "horizon_months"], as_index=False)
        .agg(
            n=("trajectory_id", "nunique"),
            remission_fraction=("clinical_remission", "mean"),
            control_fraction=("clinical_control", "mean"),
            tumour_only_fraction=("tumour_only_remission", "mean"),
            first_line_residence=("first_line_active", "mean"),
            exit_fraction=("exit_by_horizon", "mean"),
        )
        .sort_values(["context", "arm", "horizon_months"])
        .reset_index(drop=True)
    )


def dwell_summary(
    table: pd.DataFrame,
    *,
    start_months: float = 3.0,
    end_months: float = 6.0,
) -> pd.DataFrame:
    """Estimate continuous finite-horizon dwell among futures in remission.

    The state is sampled every ``CURVE_STEP_MONTHS`` and carried forward between
    generated clinical events. This is an approximation to continuous dwell,
    not a re-branching experiment from the remission state.
    """
    at_start = table[np.isclose(table["horizon_months"], start_months)]
    captured = at_start[at_start["clinical_remission"]][
        ["trajectory_id", "context", "arm"]
    ]
    interval = table[
        table["horizon_months"].between(start_months, end_months)
    ]
    persistent = (
        interval.groupby("trajectory_id")["clinical_remission"]
        .all()
        .rename("continuous_dwell")
    )
    joined = captured.merge(
        persistent,
        on="trajectory_id",
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for context in CONTEXTS:
        for arm in ARMS:
            group = joined[joined["context"].eq(context) & joined["arm"].eq(arm)]
            rows.append(
                {
                    "context": context,
                    "arm": arm,
                    "start_months": start_months,
                    "end_months": end_months,
                    "n_in_remission_at_start": int(len(group)),
                    "n_continuously_in_remission": int(
                        group["continuous_dwell"].sum()
                    ),
                    "continuous_dwell_fraction": (
                        float(group["continuous_dwell"].mean())
                        if len(group)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def capture_summary(
    table: pd.DataFrame,
    *,
    horizon_months: float = 3.0,
) -> pd.DataFrame:
    """Measure entry into remission among non-remission decision anchors."""
    anchor = table[np.isclose(table["horizon_months"], 0.0)][
        ["trajectory_id", "context", "arm", "clinical_remission"]
    ].rename(columns={"clinical_remission": "remission_at_anchor"})
    horizon = table[np.isclose(table["horizon_months"], horizon_months)][
        ["trajectory_id", "clinical_remission", "first_line_active"]
    ].rename(columns={"clinical_remission": "remission_at_horizon"})
    joined = anchor.merge(horizon, on="trajectory_id", validate="one_to_one")
    eligible = joined[~joined["remission_at_anchor"]]
    return (
        eligible.groupby(["context", "arm"], as_index=False)
        .agg(
            n_nonremission_anchors=("trajectory_id", "nunique"),
            n_captured=(
                "remission_at_horizon",
                lambda values: int(values.sum()),
            ),
            capture_fraction=("remission_at_horizon", "mean"),
            active_fraction=("first_line_active", "mean"),
        )
    )


def sensitivity_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Expose dependence on the operational remission definition."""
    definitions = {
        "strict_clinical_remission": "clinical_remission",
        "tumour_only_small_or_less": "tumour_only_remission",
        "broad_clinical_control": "clinical_control",
        "first_line_residence_only": "first_line_active",
    }
    selected = table[
        table["horizon_months"].apply(
            lambda value: any(np.isclose(value, item) for item in (3.0, 6.0, 12.0))
        )
    ]
    rows: list[dict[str, object]] = []
    for (context, arm, horizon), group in selected.groupby(
        ["context", "arm", "horizon_months"]
    ):
        for label, column in definitions.items():
            rows.append(
                {
                    "context": context,
                    "arm": arm,
                    "horizon_months": horizon,
                    "definition": label,
                    "fraction": float(group[column].mean()),
                    "n": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def fit_local_basin_envelope(table: pd.DataFrame) -> dict[str, object]:
    """Fit a visual ellipse to the local S2/H1 remission projection.

    The clinical definition, not the ellipse, determines every numerical
    result. The ellipse is only a visual envelope of the corresponding points
    after lossy 2-D projection.
    """
    selected = table[
        table["context"].eq("EGFR_1L")
        & table["state"].eq(EXEMPLAR_STATE)
        & table["history"].eq(EXEMPLAR_HISTORY)
        & table["arm"].isin(ACTIVE_ARMS)
        & table["horizon_months"].apply(
            lambda value: any(np.isclose(value, item) for item in (3.0, 6.0, 9.0, 12.0))
        )
        & table["clinical_remission"]
    ]
    points = selected[["map_x", "map_y"]].drop_duplicates().to_numpy(float)
    if len(points) < 5:
        raise RuntimeError("Too few projected remission points for visual envelope")
    center = points.mean(axis=0)
    covariance = np.cov(points, rowvar=False) + np.eye(2) * 1e-6
    inverse = np.linalg.inv(covariance)
    delta = points - center
    squared_distance = np.einsum("ij,jk,ik->i", delta, inverse, delta)
    quantile = float(np.quantile(squared_distance, 0.90))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    semi_axes = np.sqrt(eigenvalues * quantile)
    angle = float(
        np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    )
    return {
        "state": EXEMPLAR_STATE,
        "history": EXEMPLAR_HISTORY,
        "definition": (
            "visual 90% covariance envelope of distinct projected raw-feature "
            "clinical-remission states at 3/6/9/12 months, active arms pooled"
        ),
        "n_points": int(len(points)),
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "width": float(2 * semi_axes[0]),
        "height": float(2 * semi_axes[1]),
        "angle_degrees": angle,
    }


def _add_basin(ax: plt.Axes, basin: dict[str, object]) -> None:
    # Nested envelopes make the visual metaphor a shallow-to-deep basin while
    # keeping the raw-feature rule—not geometry—as the actual classifier.
    for scale, alpha, linewidth in (
        (1.0, 0.045, 1.4),
        (0.68, 0.060, 1.0),
        (0.36, 0.085, 0.8),
    ):
        ax.add_patch(
            Ellipse(
                (basin["center_x"], basin["center_y"]),
                width=scale * basin["width"],
                height=scale * basin["height"],
                angle=basin["angle_degrees"],
                facecolor=REMISSION_COLOR,
                edgecolor=REMISSION_COLOR,
                lw=linewidth,
                alpha=alpha,
                zorder=1,
            )
        )
    ax.text(
        basin["center_x"],
        basin["center_y"] + 0.52 * basin["height"],
        "local projection of the\nclinical-remission basin",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=REMISSION_COLOR,
        weight="bold",
        zorder=10,
    )


def _map_limits(
    table: pd.DataFrame,
    basin: dict[str, object],
) -> tuple[float, float, float, float]:
    selected = table[
        table["context"].eq("EGFR_1L")
        & table["state"].eq(EXEMPLAR_STATE)
        & table["history"].eq(EXEMPLAR_HISTORY)
        & table["arm"].isin(ACTIVE_ARMS)
        & table["horizon_months"].le(12.0)
    ]
    x_values = selected["map_x"].astype(float).tolist()
    y_values = selected["map_y"].astype(float).tolist()
    x_values += [
        float(basin["center_x"]) - 0.55 * float(basin["width"]),
        float(basin["center_x"]) + 0.55 * float(basin["width"]),
    ]
    y_values += [
        float(basin["center_y"]) - 0.60 * float(basin["height"]),
        float(basin["center_y"]) + 0.60 * float(basin["height"]),
    ]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_pad = max(0.12 * (x_max - x_min), 0.08)
    y_pad = max(0.15 * (y_max - y_min), 0.06)
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def _style_map(
    ax: plt.Axes,
    limits: tuple[float, float, float, float],
    *,
    show_y: bool,
) -> None:
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.grid(color=GRID, lw=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("combined clinical state PC1\n(provisional display)", fontsize=7.2)
    if show_y:
        ax.set_ylabel("combined clinical state PC2", fontsize=7.2)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(labelsize=6.5)


def draw_fan_panel(
    ax: plt.Axes,
    *,
    panel_label: str,
    arm: str,
    horizon_months: float,
    snapshots: pd.DataFrame,
    table: pd.DataFrame,
    basin: dict[str, object],
    limits: tuple[float, float, float, float],
    show_y: bool,
) -> None:
    """Draw all stored futures for one exact decision anchor."""
    _add_basin(ax, basin)
    selected = table[
        table["context"].eq("EGFR_1L")
        & table["state"].eq(EXEMPLAR_STATE)
        & table["history"].eq(EXEMPLAR_HISTORY)
        & table["arm"].eq(arm)
        & np.isclose(table["horizon_months"], horizon_months)
    ]
    anchor_rows = table[
        table["context"].eq("EGFR_1L")
        & table["state"].eq(EXEMPLAR_STATE)
        & table["history"].eq(EXEMPLAR_HISTORY)
        & table["arm"].eq(arm)
        & np.isclose(table["horizon_months"], 0.0)
    ]
    for item in selected.itertuples(index=False):
        course = snapshots[snapshots["trajectory_id"].eq(item.trajectory_id)]
        elapsed = min(float(item.endpoint_mo), horizon_months)
        end_day = float(item.cutoff_days) + elapsed * MONTH_DAYS
        path = path_until(course, float(item.cutoff_days), end_day)
        path_color = EXIT if item.exit_by_horizon else ARM_COLOR[arm]
        ax.plot(
            path["map_x"],
            path["map_y"],
            color=path_color,
            lw=1.65,
            alpha=0.58,
            zorder=4,
        )
        if item.exit_by_horizon:
            ax.scatter(
                item.map_x,
                item.map_y,
                marker="X",
                s=58,
                color=EXIT,
                edgecolor="white",
                lw=0.6,
                zorder=8,
            )
        else:
            ax.scatter(
                item.map_x,
                item.map_y,
                s=47,
                facecolor=(
                    REMISSION_COLOR if item.clinical_remission else "white"
                ),
                edgecolor=(
                    ARM_COLOR[arm] if item.clinical_remission else INK
                ),
                lw=1.2,
                zorder=8,
            )
    anchor = anchor_rows[["map_x", "map_y"]].mean()
    ax.scatter(
        anchor["map_x"],
        anchor["map_y"],
        s=120,
        facecolor="white",
        edgecolor=INK,
        lw=2.0,
        zorder=10,
    )
    ax.text(
        anchor["map_x"] + 0.025,
        anchor["map_y"] - 0.025,
        "shared\nanchor",
        fontsize=6.2,
        color=INK,
        weight="bold",
        va="top",
        zorder=11,
    )
    remission_n = int(selected["clinical_remission"].sum())
    exit_n = int(selected["exit_by_horizon"].sum())
    ax.text(
        0.98,
        0.97,
        (
            f"clinical remission {remission_n}/{len(selected)}\n"
            f"first-line exit {exit_n}/{len(selected)}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": GRID,
            "alpha": 0.94,
        },
        zorder=20,
    )
    _style_map(ax, limits, show_y=show_y)
    phase = "early pull" if np.isclose(horizon_months, 3.0) else "persistence"
    ax.set_title(
        (
            f"{panel_label}  {ARM_SHORT[arm]} · {phase} at {horizon_months:g} months\n"
            f"one exact anchor ({EXEMPLAR_LABEL}) · 3 stored futures"
        ),
        loc="left",
        fontsize=8.7,
        weight="bold",
        color=INK,
    )


def _summary_row(
    summary: pd.DataFrame,
    context: str,
    arm: str,
    horizon: float,
) -> pd.Series:
    return summary[
        summary["context"].eq(context)
        & summary["arm"].eq(arm)
        & np.isclose(summary["horizon_months"], horizon)
    ].iloc[0]


def draw_occupancy_curve(
    ax: plt.Axes,
    *,
    panel_label: str,
    context: str,
    summary: pd.DataFrame,
    dwell: pd.DataFrame,
) -> None:
    for arm in ARMS:
        data = summary[
            summary["context"].eq(context) & summary["arm"].eq(arm)
        ].sort_values("horizon_months")
        ax.plot(
            data["horizon_months"],
            data["remission_fraction"],
            color=ARM_COLOR[arm],
            lw=2.45,
            label=ARM_SHORT[arm],
        )
        for horizon in (3.0, 6.0, 12.0):
            row = _summary_row(summary, context, arm, horizon)
            ax.scatter(
                horizon,
                row["remission_fraction"],
                s=25,
                color=ARM_COLOR[arm],
                edgecolor="white",
                lw=0.45,
                zorder=5,
            )
    for horizon in (3.0, 6.0, 12.0):
        ax.axvline(horizon, color="#a8b4c3", ls="--", lw=0.7, zorder=0)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("months from treatment decision", fontsize=7.5)
    ax.set_ylabel("fraction in clinical remission\nand still in first line", fontsize=7.5)
    ax.grid(color=GRID, lw=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.6)
    ax.legend(frameon=False, fontsize=6.6, loc="upper right")

    chemo_3 = _summary_row(summary, context, "chemo", 3.0)
    osi_3 = _summary_row(summary, context, "osimertinib", 3.0)
    chemo_12 = _summary_row(summary, context, "chemo", 12.0)
    osi_12 = _summary_row(summary, context, "osimertinib", 12.0)
    text = (
        f"3 mo: chemo {100 * chemo_3['remission_fraction']:.0f}% · "
        f"osi {100 * osi_3['remission_fraction']:.0f}%\n"
        f"12 mo: chemo {100 * chemo_12['remission_fraction']:.0f}% · "
        f"osi {100 * osi_12['remission_fraction']:.0f}%"
    )
    if context == "EGFR_1L":
        dwell_6 = dwell[
            dwell["context"].eq(context)
            & dwell["arm"].isin(ACTIVE_ARMS)
            & np.isclose(dwell["end_months"], 6.0)
        ].set_index("arm")
        dwell_12 = dwell[
            dwell["context"].eq(context)
            & dwell["arm"].isin(ACTIVE_ARMS)
            & np.isclose(dwell["end_months"], 12.0)
        ].set_index("arm")
        text += (
            "\ncontinuous dwell among captured:\n"
            f"3→6 mo {100 * dwell_6.loc['chemo', 'continuous_dwell_fraction']:.0f}%/"
            f"{100 * dwell_6.loc['osimertinib', 'continuous_dwell_fraction']:.0f}% · "
            f"3→12 mo {100 * dwell_12.loc['chemo', 'continuous_dwell_fraction']:.0f}%/"
            f"{100 * dwell_12.loc['osimertinib', 'continuous_dwell_fraction']:.0f}%"
        )
    else:
        text += "\nlong-tail osi futures retained as an assay warning"
    ax.text(
        0.98,
        0.58,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": GRID,
            "alpha": 0.94,
        },
    )
    title = (
        "Attractor = entry plus finite-horizon dwell"
        if context == "EGFR_1L"
        else "Wrong-target control: the osimertinib basin collapses"
    )
    ax.set_title(
        f"{panel_label}  {title}\n{CONTEXT_LABEL[context]} · all 15 matched anchors",
        loc="left",
        fontsize=8.7,
        weight="bold",
        color=INK,
    )


def build_figure(
    *,
    snapshots: pd.DataFrame,
    table: pd.DataFrame,
    summary: pd.DataFrame,
    dwell: pd.DataFrame,
    basin: dict[str, object],
    out_dir: Path,
) -> tuple[Path, Path]:
    limits = _map_limits(table, basin)
    figure = plt.figure(figsize=(15.8, 10.2), facecolor=PAPER)
    grid = figure.add_gridspec(
        2,
        3,
        left=0.055,
        right=0.982,
        top=0.82,
        bottom=0.105,
        wspace=0.24,
        hspace=0.42,
    )
    axes = np.array(
        [
            [figure.add_subplot(grid[0, index]) for index in range(3)],
            [figure.add_subplot(grid[1, index]) for index in range(3)],
        ]
    )
    draw_fan_panel(
        axes[0, 0],
        panel_label="A",
        arm="chemo",
        horizon_months=3.0,
        snapshots=snapshots,
        table=table,
        basin=basin,
        limits=limits,
        show_y=True,
    )
    draw_fan_panel(
        axes[0, 1],
        panel_label="B",
        arm="osimertinib",
        horizon_months=3.0,
        snapshots=snapshots,
        table=table,
        basin=basin,
        limits=limits,
        show_y=False,
    )
    draw_occupancy_curve(
        axes[0, 2],
        panel_label="C",
        context="EGFR_1L",
        summary=summary,
        dwell=dwell,
    )
    draw_fan_panel(
        axes[1, 0],
        panel_label="D",
        arm="chemo",
        horizon_months=12.0,
        snapshots=snapshots,
        table=table,
        basin=basin,
        limits=limits,
        show_y=True,
    )
    draw_fan_panel(
        axes[1, 1],
        panel_label="E",
        arm="osimertinib",
        horizon_months=12.0,
        snapshots=snapshots,
        table=table,
        basin=basin,
        limits=limits,
        show_y=False,
    )
    draw_occupancy_curve(
        axes[1, 2],
        panel_label="F",
        context="EGFR_WT_1L",
        summary=summary,
        dwell=dwell,
    )

    figure.suptitle(
        "A remission attractor is defined by pull into a clinical state—and persistence within it",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=16.5,
        weight="bold",
        color=INK,
    )
    figure.text(
        0.055,
        0.925,
        (
            "Treatment-neutral clinical remission: tumour burden ≤ small, ECOG 0–1, "
            "symptoms none/mild, and no first-line exit"
        ),
        ha="left",
        fontsize=9.1,
        color=MUTED,
    )
    figure.text(
        0.055,
        0.892,
        (
            "Panels A–B: both therapies pull matched futures toward remission. "
            "Panels D–E: osimertinib keeps them there. Panel F: this persistence is biomarker-specific."
        ),
        ha="left",
        fontsize=7.9,
        color=INK,
    )
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=INK,
            markersize=6,
            label="shared treatment-decision anchor",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=REMISSION_COLOR,
            markeredgecolor=REMISSION_COLOR,
            markersize=6,
            label="clinical remission at horizon",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=INK,
            markersize=6,
            label="active but outside remission",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor=EXIT,
            markeredgecolor="white",
            markersize=7,
            label="first-line exit before horizon",
        ),
    ]
    figure.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.055, 0.025),
        frameon=False,
        ncol=4,
        fontsize=7.0,
        columnspacing=1.6,
        handletextpad=0.5,
    )
    figure.text(
        0.982,
        0.035,
        (
            "Locked Sonnet model assay; not observed human trajectories or an "
            f"individual predictor. Display map: {MAP_ID}."
        ),
        ha="right",
        fontsize=6.9,
        color=MUTED,
    )
    png = out_dir / "fig_egfr_remission_attractor.png"
    pdf = out_dir / "fig_egfr_remission_attractor.pdf"
    figure.savefig(png, dpi=300, facecolor=figure.get_facecolor())
    figure.savefig(pdf, facecolor=figure.get_facecolor())
    plt.close(figure)
    return png, pdf


def _landmark_table(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        summary["horizon_months"].apply(
            lambda value: any(np.isclose(value, item) for item in LANDMARKS)
        )
    ].reset_index(drop=True)


def write_readout(
    *,
    out_dir: Path,
    run_root: Path,
    endpoint_path: Path,
    table: pd.DataFrame,
    summary: pd.DataFrame,
    capture: pd.DataFrame,
    dwell: pd.DataFrame,
    sensitivity: pd.DataFrame,
    anchor_audit: pd.DataFrame,
    basin: dict[str, object],
    map_explained: tuple[float, float],
    figures: tuple[Path, Path],
) -> None:
    landmarks = _landmark_table(summary)
    landmarks.to_csv(out_dir / "attractor_occupancy_landmarks.csv", index=False)
    capture.to_csv(out_dir / "attractor_capture.csv", index=False)
    dwell.to_csv(out_dir / "attractor_dwell.csv", index=False)
    sensitivity.to_csv(out_dir / "attractor_definition_sensitivity.csv", index=False)
    anchor_audit.to_csv(out_dir / "anchor_invariance_audit.csv", index=False)
    table.to_csv(out_dir / "trajectory_horizon_attractor.csv", index=False)
    (out_dir / "local_basin_projection.json").write_text(
        json.dumps(basin, indent=2) + "\n"
    )

    key_results: dict[str, dict[str, dict[str, float]]] = {}
    for context in CONTEXTS:
        key_results[context] = {}
        for arm in ARMS:
            rows = landmarks[
                landmarks["context"].eq(context) & landmarks["arm"].eq(arm)
            ].set_index("horizon_months")
            key_results[context][arm] = {
                "remission_at_3mo": float(rows.loc[3.0, "remission_fraction"]),
                "remission_at_6mo": float(rows.loc[6.0, "remission_fraction"]),
                "remission_at_12mo": float(rows.loc[12.0, "remission_fraction"]),
                "first_line_at_12mo": float(rows.loc[12.0, "first_line_residence"]),
            }
    report = {
        "status": "model_assay_not_human_causal_effect",
        "operational_object": "conditional finite-horizon clinical remission attractor",
        "definition": {
            "clinical_remission": (
                "first-line active AND tumour burden <= small AND ECOG <= 1 "
                "AND symptom burden <= mild"
            ),
            "pull": "entry from a non-remission decision anchor by 3 months",
            "persistence": (
                "continued remission occupancy; continuous 3-to-6-month dwell "
                "approximated on a 0.25-month carried-forward state grid"
            ),
            "ellipse_status": (
                "visual projection only; all results use raw clinical features"
            ),
        },
        "key_results": key_results,
        "capture": capture.to_dict(orient="records"),
        "dwell": dwell.to_dict(orient="records"),
        "support": {
            "n_trajectories": int(table["trajectory_id"].nunique()),
            "n_exact_anchors": int(
                table[["context", "state", "history"]].drop_duplicates().shape[0]
            ),
            "futures_per_arm_anchor": 3,
            "max_anchor_x_range": float(anchor_audit["x_range"].max()),
            "max_anchor_y_range": float(anchor_audit["y_range"].max()),
        },
        "display_map": {
            "map_id": MAP_ID,
            "explained_variance": list(map_explained),
            "provisional": True,
        },
        "provenance": {
            "git_commit_at_build": git_commit(),
            "run_root": str(run_root.relative_to(ROOT)),
            "input_sha256": {
                "snapshot_rows": sha256(
                    run_root / "derivation" / "snapshot_rows.csv"
                ),
                "endpoint_long": sha256(endpoint_path),
            },
            "figures": [str(path.relative_to(ROOT)) for path in figures],
        },
        "limits": [
            "The basin definition and horizon were not pre-registered; this is an exploratory operationalisation.",
            "No re-branching was performed from remission states, so continuous dwell is reconstructed from existing paths.",
            "The 2-D ellipse is a lossy visual projection, not the attractor estimator.",
            "Per-cell n=3 supports a qualitative model-assay result, not calibrated individual prediction.",
        ],
    }
    (out_dir / "readout.json").write_text(json.dumps(report, indent=2) + "\n")

    mutant = landmarks[landmarks["context"].eq("EGFR_1L")]
    wildtype = landmarks[landmarks["context"].eq("EGFR_WT_1L")]

    def value(frame: pd.DataFrame, arm: str, horizon: float) -> float:
        return float(
            frame[
                frame["arm"].eq(arm)
                & np.isclose(frame["horizon_months"], horizon)
            ]["remission_fraction"].iloc[0]
        )

    lines = [
        "# EGFR finite-horizon remission attractor",
        "",
        "## Operational definition",
        "",
        (
            "Clinical remission is defined in raw variables as: first-line active, "
            "tumour burden no greater than `small`, ECOG 0–1, and symptoms "
            "`none`/`mild`. The PCA ellipse is only a visual projection of this "
            "phenotype; it is not used to classify remission."
        ),
        "",
        "An attractor has two measured components here:",
        "",
        "- **capture/pull:** entry into remission after the shared treatment decision;",
        "- **persistence:** continued occupation without first-line exit over a finite horizon.",
        "",
        "## Main result",
        "",
        "| context | arm | remission 3 mo | remission 6 mo | remission 12 mo |",
        "|---|---|---:|---:|---:|",
    ]
    for context, frame in (("EGFR-mutant", mutant), ("EGFR-wild-type", wildtype)):
        for arm in ARMS:
            lines.append(
                f"| {context} | {ARM_SHORT[arm]} | "
                f"{100 * value(frame, arm, 3.0):.1f}% | "
                f"{100 * value(frame, arm, 6.0):.1f}% | "
                f"{100 * value(frame, arm, 12.0):.1f}% |"
            )
    mutant_dwell_6 = dwell[
        dwell["context"].eq("EGFR_1L")
        & dwell["arm"].isin(ACTIVE_ARMS)
        & np.isclose(dwell["end_months"], 6.0)
    ].set_index("arm")
    mutant_dwell_12 = dwell[
        dwell["context"].eq("EGFR_1L")
        & dwell["arm"].isin(ACTIVE_ARMS)
        & np.isclose(dwell["end_months"], 12.0)
    ].set_index("arm")
    lines.extend(
        [
            "",
            (
                "In EGFR-mutant disease, three-month remission occupancy is nearly "
                f"identical (chemotherapy {100 * value(mutant, 'chemo', 3.0):.1f}% "
                f"versus osimertinib {100 * value(mutant, 'osimertinib', 3.0):.1f}%), "
                "but by 12 months it separates to "
                f"{100 * value(mutant, 'chemo', 12.0):.1f}% versus "
                f"{100 * value(mutant, 'osimertinib', 12.0):.1f}%."
            ),
            (
                "Among futures already in remission at three months, continuous "
                "three-to-six-month dwell is "
                f"{100 * mutant_dwell_6.loc['chemo', 'continuous_dwell_fraction']:.1f}% "
                "with chemotherapy and "
                f"{100 * mutant_dwell_6.loc['osimertinib', 'continuous_dwell_fraction']:.1f}% "
                "with osimertinib."
            ),
            (
                "For continuous three-to-twelve-month dwell, the same comparison "
                f"is {100 * mutant_dwell_12.loc['chemo', 'continuous_dwell_fraction']:.1f}% "
                "versus "
                f"{100 * mutant_dwell_12.loc['osimertinib', 'continuous_dwell_fraction']:.1f}%."
            ),
            (
                "In EGFR-wild-type disease, osimertinib remission occupancy is "
                f"{100 * value(wildtype, 'osimertinib', 3.0):.1f}% at three months "
                f"and {100 * value(wildtype, 'osimertinib', 12.0):.1f}% at 12 months, "
                "providing the wrong-target control."
            ),
            "",
            "## Interpretation",
            "",
            (
                "The result supports the language of a **conditional finite-horizon "
                "remission attractor**: chemotherapy and osimertinib initially pull "
                "EGFR-mutant futures toward the same clinically defined phenotype, "
                "but osimertinib produces substantially greater persistence."
            ),
            "",
            "It does **not** establish a global mathematical attractor. No new futures "
            "were generated from every point inside and outside the region, and the "
            "PCA is a lossy display layer.",
            "",
            "## Files",
            "",
            "- `fig_egfr_remission_attractor.png` / `.pdf`: paper-facing figure.",
            "- `trajectory_horizon_attractor.csv`: raw-feature phenotype and map position by trajectory/horizon.",
            "- `attractor_occupancy_landmarks.csv`: main landmark readout.",
            "- `attractor_capture.csv`: capture among non-remission anchors.",
            "- `attractor_dwell.csv`: continuous finite-horizon dwell approximation.",
            "- `attractor_definition_sensitivity.csv`: strict/broad/state-only sensitivity.",
            "- `local_basin_projection.json`: display-only ellipse parameters.",
            "",
        ]
    )
    (out_dir / "readout.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--endpoint-long", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    display_map, _ = fit_display_map(load_atlas_snapshots())
    snapshots, _ = load_phase1(args.run_root, display_map)
    endpoints = pd.read_csv(args.endpoint_long)
    endpoints = endpoints[
        endpoints["context"].isin(CONTEXTS)
        & endpoints["arm_key"].isin(ARMS)
    ].copy()
    horizons = np.round(
        np.arange(0.0, 18.0 + CURVE_STEP_MONTHS, CURVE_STEP_MONTHS),
        8,
    )
    table, anchor_audit = build_attractor_table(
        snapshots,
        endpoints,
        args.run_root,
        horizons,
    )
    summary = occupancy_summary(table)
    capture = capture_summary(table)
    dwell = pd.concat(
        [
            dwell_summary(table, start_months=3.0, end_months=6.0),
            dwell_summary(table, start_months=3.0, end_months=12.0),
        ],
        ignore_index=True,
    )
    sensitivity = sensitivity_summary(table)
    basin = fit_local_basin_envelope(table)
    figures = build_figure(
        snapshots=snapshots,
        table=table,
        summary=summary,
        dwell=dwell,
        basin=basin,
        out_dir=args.out_dir,
    )
    write_readout(
        out_dir=args.out_dir,
        run_root=args.run_root,
        endpoint_path=args.endpoint_long,
        table=table,
        summary=summary,
        capture=capture,
        dwell=dwell,
        sensitivity=sensitivity,
        anchor_audit=anchor_audit,
        basin=basin,
        map_explained=display_map.explained,
        figures=figures,
    )
    print(f"Wrote {figures[0]}")
    print(_landmark_table(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
