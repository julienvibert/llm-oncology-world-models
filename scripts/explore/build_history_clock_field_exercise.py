#!/usr/bin/env python3
"""History as an explicit coordinate or a sequence of state-map slices.

It builds:

1. a disclosed conceptual comparison of 3D augmentation, PC1 × history, and
   the recommended PC1–PC2 filmstrip;
2. a data-grounded treatment-age movie using the locked Sonnet Phase-1 paired
   generated futures;
3. a trajectory-grouped held-out benchmark of candidate clinical clocks;
4. real generated trajectories shown in a learned one-dimensional predictive
   history coordinate.

The display PCA is fitted once from 16 contemporaneous tumour + host features.
No clock, history variable, cancer label, model label, treatment, or outcome is
used to define the two display axes.

Run:
  uv run python scripts/explore/build_history_clock_field_exercise.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_faceted_clinical_field_atlas import (  # noqa: E402
    HORIZON_MONTHS,
    INK,
    MONTH_DAYS,
    MUTED,
    PAPER,
    PHASE1_ROOT,
    add_phase_labels,
    collapse_same_day,
    fixed_horizon_pairs,
    interpolate_map_position,
)
from build_one_map_many_futures_exercise import (  # noqa: E402
    DisplayMap,
    fit_display_map,
    load_derivation,
    map_limits,
    transform_to_map,
)
from composite_tumor_host_phase_space import (  # noqa: E402
    enrich_snapshots,
    load_tables,
)

OUT = ROOT / "outputs" / "exploratory_history_clock_fields"

BLUE = "#3474b8"
TEAL = "#138f86"
GREEN = "#208a53"
ORANGE = "#df7900"
RED = "#c63e3e"
PURPLE = "#6846a5"
GOLD = "#c59a24"
GRID = "#dce4ec"
SEED = 20260728

MOVIE_AGES = (0, 3, 6, 12)
FORECAST_MONTHS = 3.0
FORECAST_DAYS = FORECAST_MONTHS * MONTH_DAYS

STATE_KEY = (
    "S1 fit/low burden, ECOG 0 · S2 fit/liver micrometastasis, ECOG 1 · "
    "S3 brain + liver micrometastases, ECOG 1 · S4 liver + bone disease, "
    "ECOG 1 · S5 frail/higher burden, ECOG 2"
)

HISTORY_NUMERIC = (
    "start_month",
    "line_number",
    "time_since_current_treatment",
    "months_since_first_treatment",
    "months_since_last_progression",
    "n_prior_visits",
    "n_response",
    "n_progression",
    "n_toxicity",
    "n_treatment_switch",
    "recent_pc1_velocity",
    "recent_pc2_velocity",
    "history_pc1_mean",
    "history_pc2_mean",
    "history_pc1_sd",
    "history_pc2_sd",
    "history_pc1_range",
    "history_pc2_range",
    "cumulative_path_length",
)


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(pdf, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return {"png": str(png.relative_to(ROOT)), "pdf": str(pdf.relative_to(ROOT))}


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    linewidth: float = 1.8,
    alpha: float = 0.95,
    mutation_scale: float = 12,
    zorder: int = 7,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            color=color,
            lw=linewidth,
            alpha=alpha,
            zorder=zorder,
            shrinkA=0,
            shrinkB=0,
        )
    )


def style_2d_map(
    ax: plt.Axes,
    limits: tuple[float, float, float, float],
    *,
    labels: bool = True,
) -> None:
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.grid(color=GRID, lw=0.65, alpha=0.82)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    if labels:
        ax.set_xlabel(
            "Clinical state PC1\nless severe  \u2190     \u2192  more severe",
            fontsize=8,
        )
        ax.set_ylabel("Clinical state PC2", fontsize=8)


def mock_vector(x: float, y: float, history: float) -> tuple[float, float]:
    """A disclosed mock field whose destination drifts and weakens with history."""
    attractor_x = -1.35 + 0.60 * history
    attractor_y = -0.45 + 0.35 * np.sin(np.pi * history / 2)
    strength = 0.42 - 0.08 * history
    resistance = max(history - 1.1, 0) * 0.45
    return (
        strength * (attractor_x - x) + resistance,
        strength * (attractor_y - y) + 0.08 * history,
    )


def conceptual_history_figure() -> dict[str, str]:
    """Compare three ways to draw a history-augmented state space."""
    fig = plt.figure(figsize=(18.2, 6.1), facecolor=PAPER)
    grid = fig.add_gridspec(1, 7, width_ratios=[1.3, 1.3, 0.9, 1, 1, 1, 1])

    # A — literal 3D augmentation.
    ax3d = fig.add_subplot(grid[0, :2], projection="3d")
    histories = np.array([0.0, 1.0, 2.0])
    trajectory = np.array(
        [
            [1.8, 1.15, 0.0],
            [0.65, 0.45, 0.5],
            [-0.75, -0.25, 1.0],
            [-0.90, -0.35, 1.5],
            [-0.15, 0.05, 2.0],
        ]
    )
    xx, yy = np.meshgrid(np.linspace(-2.0, 2.0, 6), np.linspace(-1.5, 1.5, 5))
    for history in histories:
        zz = np.full_like(xx, history)
        ax3d.plot_surface(
            xx,
            yy,
            zz,
            color=BLUE,
            alpha=0.045,
            edgecolor="#b9c9d8",
            linewidth=0.3,
        )
        attractor = (
            -1.35 + 0.60 * history,
            -0.45 + 0.35 * np.sin(np.pi * history / 2),
        )
        ax3d.scatter(
            *attractor,
            history,
            marker="*",
            s=115,
            color=GOLD,
            edgecolor=INK,
            zorder=10,
        )
        starts = np.array([[-0.2, -0.8], [0.7, 0.0], [1.4, 0.9]])
        for start in starts:
            dx, dy = mock_vector(start[0], start[1], history)
            ax3d.quiver(
                start[0],
                start[1],
                history,
                dx,
                dy,
                0,
                color=PURPLE,
                linewidth=1.2,
                arrow_length_ratio=0.18,
            )
    ax3d.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        "-o",
        color=RED,
        lw=2.4,
        ms=4,
    )
    ax3d.set_xlabel("PC1", labelpad=3)
    ax3d.set_ylabel("PC2", labelpad=3)
    ax3d.set_zlabel("history h", labelpad=4)
    ax3d.view_init(elev=23, azim=-58)
    ax3d.set_title(
        "A · Literal 3D augmentation\nrigorous, but visually demanding",
        loc="left",
        fontsize=10.5,
        weight="bold",
        color=INK,
    )

    # B — collapse the state map to one component.
    ax = fig.add_subplot(grid[0, 2])
    ax.plot(
        trajectory[:, 2],
        trajectory[:, 0],
        "-o",
        color=RED,
        lw=2.4,
        ms=5,
    )
    attractor_line = -1.35 + 0.60 * np.linspace(0, 2, 80)
    ax.plot(np.linspace(0, 2, 80), attractor_line, "--", color=GOLD, lw=2)
    ax.fill_between(
        np.linspace(0, 2, 80),
        attractor_line - 0.25,
        attractor_line + 0.25,
        color=GOLD,
        alpha=0.12,
    )
    ax.set_xlabel("history h", fontsize=8.5)
    ax.set_ylabel("clinical severity PC1", fontsize=8.5)
    ax.grid(color=GRID, lw=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "B · PC1 × history\nclear, but PC2 is lost",
        loc="left",
        fontsize=10.5,
        weight="bold",
        color=INK,
    )

    # C — fixed map, sequential slices.
    film_grid = grid[0, 3:].subgridspec(1, 3, wspace=0.20)
    limits = (-2.1, 2.1, -1.65, 1.65)
    for index, history in enumerate(histories):
        ax = fig.add_subplot(film_grid[0, index])
        starts = np.array(
            [
                [-1.3, -0.8],
                [-0.3, -0.8],
                [0.5, -0.1],
                [1.3, 0.8],
                [0.2, 0.9],
            ]
        )
        for start in starts:
            dx, dy = mock_vector(start[0], start[1], history)
            ax.scatter(
                start[0],
                start[1],
                s=25,
                facecolor="white",
                edgecolor=PURPLE,
                lw=1,
                zorder=5,
            )
            add_arrow(
                ax,
                (float(start[0]), float(start[1])),
                (float(start[0] + dx), float(start[1] + dy)),
                PURPLE,
                linewidth=1.7,
            )
        attractor = (
            -1.35 + 0.60 * history,
            -0.45 + 0.35 * np.sin(np.pi * history / 2),
        )
        ax.scatter(
            *attractor,
            marker="*",
            s=130,
            color=GOLD,
            edgecolor=INK,
            zorder=10,
        )
        style_2d_map(ax, limits, labels=index == 0)
        if index:
            ax.set_yticklabels([])
            ax.set_xlabel("same PC1", fontsize=8)
        ax.set_title(
            (f"C · Recommended filmstrip\nh = {int(history)}" if index == 0 else f"\nh = {int(history)}"),
            fontsize=9.5,
            weight="bold",
            color=TEAL,
        )
    fig.suptitle(
        "History can be a coordinate—but it need not become a visible third axis",
        x=0.025,
        y=1.02,
        ha="left",
        fontsize=19,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.025,
        0.945,
        (
            "Disclosed conceptual geometry. Gold stars are history-conditioned "
            "destination basins; purple arrows are short-horizon fields."
        ),
        fontsize=9.2,
        color=MUTED,
    )
    fig.text(
        0.025,
        0.015,
        (
            "In full 3D, ordinary time never stops; therefore a remission "
            "attractor is a moving or history-conditioned destination in the "
            "clinical PC1–PC2 plane, not a fixed point in space–time."
        ),
        fontsize=8.3,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.035, right=0.99, top=0.86, bottom=0.12, wspace=0.46)
    return save_figure(fig, "fig01_concept_three_history_views")


def phase1_treatment_age_vectors(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Exact three-month forecasts at several ages of the assigned treatment."""
    eligible = snapshots[
        snapshots["condition_id"].astype(str).str.contains("EGFR_1L")
        & snapshots["condition_id"].astype(str).str.endswith("__H1")
        & snapshots["policy_id"].astype(str).str.endswith(("__osimertinib", "__chemo"))
    ]
    exit_types = {
        "progression",
        "systemic_treatment_switch",
        "systemic_treatment_stop",
        "death",
    }
    rows: list[dict[str, Any]] = []
    for gid, course in eligible.groupby("_gid", sort=False):
        meta = course.iloc[0]
        event_course = events[events["_gid"].eq(gid)]
        starts = event_course[event_course["event_type"].astype(str).eq("systemic_treatment_start")]
        if starts.empty:
            continue
        t0 = float(pd.to_numeric(starts["days_from_diagnosis"], errors="coerce").min())
        future_exits = event_course[
            pd.to_numeric(event_course["days_from_diagnosis"], errors="coerce").gt(t0)
            & event_course["event_type"].astype(str).isin(exit_types)
        ]
        exit_day = (
            float(pd.to_numeric(future_exits["days_from_diagnosis"], errors="coerce").min())
            if not future_exits.empty
            else np.inf
        )
        parts = str(meta["condition_id"]).split("__")
        state = parts[-2]
        arm = str(meta["policy_id"]).split("__")[-1]
        maximum_day = float(pd.to_numeric(course["days_from_diagnosis"], errors="coerce").max())
        for age in MOVIE_AGES:
            now_day = t0 + age * MONTH_DAYS
            future_day = now_day + FORECAST_DAYS
            if maximum_day < future_day:
                continue
            x0, y0 = interpolate_map_position(course, now_day)
            x1, y1 = interpolate_map_position(course, future_day)
            rows.append(
                {
                    "_gid": gid,
                    "state": state,
                    "arm": arm,
                    "age_months": age,
                    "on_assigned_treatment": bool(now_day < exit_day),
                    "exit_within_forecast": bool(now_day < exit_day <= future_day),
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "dx": x1 - x0,
                    "dy": y1 - y0,
                }
            )
    return pd.DataFrame(rows)


def background_points(
    ax: plt.Axes,
    atlas: pd.DataFrame,
    *,
    n: int = 5000,
) -> None:
    pool = atlas[~atlas["terminal"]]
    if len(pool) > n:
        pool = pool.sample(n=n, random_state=SEED)
    ax.scatter(
        pool["map_x"],
        pool["map_y"],
        s=3,
        color="#b8c3ce",
        alpha=0.08,
        linewidth=0,
        rasterized=True,
        zorder=0,
    )


def draw_endpoint_density(
    ax: plt.Axes,
    endpoints: np.ndarray,
    color: str,
) -> tuple[float, float] | None:
    if len(endpoints) < 5:
        return None
    try:
        kde = gaussian_kde(endpoints.T)
    except np.linalg.LinAlgError:
        return None
    xpad = max(float(np.ptp(endpoints[:, 0])), 0.5)
    ypad = max(float(np.ptp(endpoints[:, 1])), 0.5)
    xx, yy = np.meshgrid(
        np.linspace(endpoints[:, 0].min() - 0.45 * xpad, endpoints[:, 0].max() + 0.45 * xpad, 80),
        np.linspace(endpoints[:, 1].min() - 0.45 * ypad, endpoints[:, 1].max() + 0.45 * ypad, 80),
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    positive = zz[zz > 0]
    levels = np.unique(np.quantile(positive, [0.66, 0.82, 0.93]))
    if len(levels) >= 2:
        ax.contour(xx, yy, zz, levels=levels, colors=color, linewidths=0.8, alpha=0.45)
    peak_index = np.unravel_index(np.argmax(zz), zz.shape)
    peak = float(xx[peak_index]), float(yy[peak_index])
    ax.scatter(
        *peak,
        marker="*",
        s=125,
        facecolor=GOLD,
        edgecolor=INK,
        lw=0.8,
        zorder=12,
    )
    return peak


def treatment_age_movie(
    vectors: pd.DataFrame,
    atlas: pd.DataFrame,
    limits: tuple[float, float, float, float],
) -> tuple[dict[str, str], dict[str, Any]]:
    fig, axes = plt.subplots(2, 4, figsize=(17.2, 8.3), sharex=True, sharey=True)
    fig.patch.set_facecolor(PAPER)
    report: dict[str, Any] = {}
    arm_properties = (
        ("osimertinib", TEAL, "Osimertinib"),
        ("chemo", ORANGE, "Chemotherapy"),
    )
    for row, (arm, color, label) in enumerate(arm_properties):
        for col, age in enumerate(MOVIE_AGES):
            ax = axes[row, col]
            selected = vectors[vectors["arm"].eq(arm) & vectors["age_months"].eq(age)]
            ongoing = selected[selected["on_assigned_treatment"]]
            exited = selected[~selected["on_assigned_treatment"]]
            background_points(ax, atlas)
            if not exited.empty:
                ax.scatter(
                    exited["x0"],
                    exited["y0"],
                    marker="x",
                    s=25,
                    color="#8b95a3",
                    lw=0.9,
                    alpha=0.55,
                    label="already exited assigned treatment",
                    zorder=3,
                )
            for item in ongoing.itertuples(index=False):
                add_arrow(
                    ax,
                    (item.x0, item.y0),
                    (item.x1, item.y1),
                    color,
                    linewidth=0.65,
                    alpha=0.18,
                    mutation_scale=5,
                    zorder=2,
                )
            summary = (
                ongoing.groupby("state", as_index=False).agg(
                    x0=("x0", "median"),
                    y0=("y0", "median"),
                    dx=("dx", "mean"),
                    dy=("dy", "mean"),
                    n=("dx", "size"),
                )
                if not ongoing.empty
                else pd.DataFrame()
            )
            for item in summary.itertuples(index=False):
                ax.scatter(
                    item.x0,
                    item.y0,
                    s=44 + 6 * item.n,
                    facecolor="white",
                    edgecolor=color,
                    lw=1.2,
                    zorder=7,
                )
                add_arrow(
                    ax,
                    (item.x0, item.y0),
                    (item.x0 + item.dx, item.y0 + item.dy),
                    color,
                    linewidth=2.3,
                    alpha=0.96,
                    mutation_scale=13,
                    zorder=8,
                )
                ax.text(
                    item.x0 + 0.05,
                    item.y0 + 0.05,
                    item.state,
                    fontsize=6.8,
                    weight="bold",
                    color=INK,
                    zorder=9,
                )
            peak = (
                draw_endpoint_density(
                    ax,
                    ongoing[["x1", "y1"]].to_numpy(float),
                    color,
                )
                if not ongoing.empty
                else None
            )
            style_2d_map(ax, limits, labels=False)
            if row == 1:
                ax.set_xlabel(
                    "Clinical state PC1\nless severe  \u2190     \u2192  more severe",
                    fontsize=7.5,
                )
            if col == 0:
                ax.set_ylabel(f"{label}\nClinical state PC2", fontsize=8.5)
            ax.set_title(
                (f"{age} months after treatment start\n{len(ongoing)}/{len(selected)} remain on assigned treatment"),
                loc="left",
                fontsize=9,
                weight="bold",
                color=color if len(ongoing) else MUTED,
            )
            if ongoing.empty:
                ax.text(
                    0.5,
                    0.50,
                    "No on-treatment\nfield remains",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    weight="bold",
                    color=MUTED,
                )
            report[f"{arm}::{age}mo"] = {
                "n_total_generated_futures": int(len(selected)),
                "n_on_assigned_treatment": int(len(ongoing)),
                "n_exited_assigned_treatment": int(len(exited)),
                "n_exiting_within_next_3_months": int(ongoing["exit_within_forecast"].sum()),
                "destination_peak": list(peak) if peak is not None else None,
                "mean_forecast_dx": (float(ongoing["dx"].mean()) if not ongoing.empty else None),
                "mean_forecast_dy": (float(ongoing["dy"].mean()) if not ongoing.empty else None),
            }
    fig.suptitle(
        "A treatment-age movie: the same map, a changing three-month field",
        x=0.03,
        y=1.01,
        ha="left",
        fontsize=18,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.03,
        0.958,
        (
            "EGFR-mutant first-line NSCLC · H1 matched history · paired Sonnet "
            "Phase-1 generated futures. Gold star = next-three-month destination "
            "peak among patients still on the assigned treatment."
        ),
        fontsize=9,
        color=MUTED,
    )
    fig.text(0.03, 0.045, f"Start-state key: {STATE_KEY}", fontsize=7.4, color=INK)
    fig.text(
        0.03,
        0.012,
        (
            "Crosses show generated paths that have already progressed, stopped, "
            "switched, or died. Later on-treatment fields are survivor-conditional "
            "and must not be read as an unadjusted causal comparison."
        ),
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.085, 0.995, 0.925), h_pad=2.0, w_pad=0.8)
    return save_figure(fig, "fig02_data_treatment_age_movie"), report


def history_feature_table(atlas: pd.DataFrame) -> pd.DataFrame:
    """Compute leakage-free summaries using only each snapshot's preceding path."""
    chunks: list[pd.DataFrame] = []
    treatment_events = {
        "systemic_treatment_start",
        "systemic_treatment_switch",
    }
    for gid, raw_course in atlas.groupby("_gid", sort=False):
        raw_days = pd.to_numeric(raw_course["days_from_diagnosis"], errors="coerce")
        raw_events = raw_course["anchor_event_type"].fillna("").astype(str)
        daily_events = pd.DataFrame({"day": raw_days, "event": raw_events}).groupby("day")["event"].agg(list).to_dict()
        course = collapse_same_day(raw_course)
        days = pd.to_numeric(course["days_from_diagnosis"], errors="coerce").to_numpy(float)
        months = days / MONTH_DAYS
        events_at_day = [daily_events.get(day, []) for day in days]
        x = course["map_x"].to_numpy(float)
        y = course["map_y"].to_numpy(float)
        dt = np.diff(months, prepend=np.nan)
        recent_x = np.divide(
            np.diff(x, prepend=np.nan),
            dt,
            out=np.zeros_like(x),
            where=np.isfinite(dt) & (dt > 0),
        )
        recent_y = np.divide(
            np.diff(y, prepend=np.nan),
            dt,
            out=np.zeros_like(y),
            where=np.isfinite(dt) & (dt > 0),
        )
        step = np.sqrt(np.diff(x, prepend=x[0]) ** 2 + np.diff(y, prepend=y[0]) ** 2)
        treatment_mask = np.array([bool(set(events) & treatment_events) for events in events_at_day])
        treatment_days = np.where(treatment_mask, days, np.nan)
        last_treatment = pd.Series(treatment_days).ffill().to_numpy(float)
        first_treatment = float(np.nanmin(treatment_days)) if np.isfinite(treatment_days).any() else np.nan
        progression_mask = np.array(["progression" in events for events in events_at_day])
        progression_days = np.where(progression_mask, days, np.nan)
        last_progression = pd.Series(progression_days).ffill().to_numpy(float)
        count_by_day = {
            event_type: np.array(
                [events.count(event_type) for events in events_at_day],
                dtype=int,
            )
            for event_type in (
                "response",
                "progression",
                "toxicity",
                "systemic_treatment_switch",
            )
        }
        work = pd.DataFrame(
            {
                "_gid": gid,
                "start_day": days,
                "start_month": months,
                "line_number": np.cumsum(treatment_mask),
                "time_since_current_treatment": (days - last_treatment) / MONTH_DAYS,
                "months_since_first_treatment": (days - first_treatment) / MONTH_DAYS,
                "months_since_last_progression": (days - last_progression) / MONTH_DAYS,
                "n_prior_visits": np.arange(len(course)),
                "n_response": np.cumsum(count_by_day["response"]),
                "n_progression": np.cumsum(count_by_day["progression"]),
                "n_toxicity": np.cumsum(count_by_day["toxicity"]),
                "n_treatment_switch": np.cumsum(count_by_day["systemic_treatment_switch"]),
                "recent_pc1_velocity": np.clip(recent_x, -4, 4),
                "recent_pc2_velocity": np.clip(recent_y, -4, 4),
                "history_pc1_mean": pd.Series(x).expanding().mean().to_numpy(),
                "history_pc2_mean": pd.Series(y).expanding().mean().to_numpy(),
                "history_pc1_sd": pd.Series(x).expanding().std().fillna(0).to_numpy(),
                "history_pc2_sd": pd.Series(y).expanding().std().fillna(0).to_numpy(),
                "history_pc1_range": (pd.Series(x).expanding().max() - pd.Series(x).expanding().min()).to_numpy(),
                "history_pc2_range": (pd.Series(y).expanding().max() - pd.Series(y).expanding().min()).to_numpy(),
                "cumulative_path_length": np.cumsum(step),
            }
        )
        chunks.append(work)
    return pd.concat(chunks, ignore_index=True)


def sample_benchmark_pairs(
    pairs: pd.DataFrame,
    *,
    maximum_per_trajectory: int = 4,
) -> pd.DataFrame:
    """Prevent long generated courses from dominating the held-out benchmark."""
    chunks: list[pd.DataFrame] = []
    for _, course in pairs.sort_values("start_day").groupby("_gid", sort=False):
        if len(course) <= maximum_per_trajectory:
            chunks.append(course)
            continue
        indices = np.linspace(0, len(course) - 1, maximum_per_trajectory)
        chunks.append(course.iloc[np.unique(np.round(indices).astype(int))])
    return pd.concat(chunks, ignore_index=True)


def prepare_benchmark(
    pairs: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    selected = sample_benchmark_pairs(pairs)
    selected = selected.merge(
        history,
        on=["_gid", "start_day"],
        how="left",
        validate="many_to_one",
    )
    selected["phase"] = selected["phase"].fillna("unknown")
    return selected.replace([np.inf, -np.inf], np.nan)


def encoded_matrix(
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    values = frame[numeric + categorical].copy()
    for column in numeric:
        values[column] = pd.to_numeric(values[column], errors="coerce").fillna(-1.0)
    return pd.get_dummies(
        values,
        columns=categorical,
        prefix=categorical,
        dtype=float,
    )


def make_regressor(seed_offset: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=70,
        max_depth=14,
        min_samples_leaf=15,
        max_features=1.0,
        n_jobs=-1,
        random_state=SEED + seed_offset,
    )


def benchmark_clocks(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], tuple[StandardScaler, PLSRegression, float]]:
    """Group-held-out prediction of six-month state displacement."""
    base_numeric = ["x0", "y0"]
    base_categorical = ["cancer", "model"]
    specifications: dict[str, tuple[list[str], list[str]]] = {
        "State only": ([], []),
        "+ diagnosis time": (["start_month"], []),
        "+ treatment line": (["line_number"], []),
        "+ time since treatment": (["time_since_current_treatment"], []),
        "+ clinical phase": ([], ["phase"]),
        "+ hybrid clinical clock": (
            ["line_number", "time_since_current_treatment"],
            ["phase"],
        ),
        "+ full history": (list(HISTORY_NUMERIC), ["phase"]),
    }
    matrices = {
        label: encoded_matrix(
            data,
            base_numeric + extra_numeric,
            base_categorical + extra_categorical,
        )
        for label, (extra_numeric, extra_categorical) in specifications.items()
    }
    target = data[["dx", "dy"]].to_numpy(float)
    groups = data["_gid"].to_numpy()
    splitter = GroupKFold(n_splits=4)
    fold_rows: list[dict[str, Any]] = []
    learned_coordinates = np.full(len(data), np.nan)
    for fold, (train, test) in enumerate(splitter.split(data, groups=groups)):
        for index, (label, matrix) in enumerate(matrices.items()):
            model = make_regressor(100 * fold + index)
            model.fit(matrix.iloc[train], target[train])
            prediction = model.predict(matrix.iloc[test])
            errors = np.linalg.norm(target[test] - prediction, axis=1)
            fold_rows.append(
                {
                    "model": label,
                    "fold": fold,
                    "mean_euclidean_error": float(errors.mean()),
                    "r2": float(
                        r2_score(
                            target[test],
                            prediction,
                            multioutput="variance_weighted",
                        )
                    ),
                    "n_test": int(len(test)),
                }
            )

        history_train = data.iloc[train][list(HISTORY_NUMERIC)].fillna(-1.0)
        history_test = data.iloc[test][list(HISTORY_NUMERIC)].fillna(-1.0)
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(history_train)
        scaled_test = scaler.transform(history_test)
        pls = PLSRegression(n_components=1, scale=False, max_iter=500)
        pls.fit(scaled_train, target[train])
        h_train = pls.transform(scaled_train).ravel()
        h_test = pls.transform(scaled_test).ravel()
        learned_coordinates[test] = h_test
        base = matrices["State only"]
        train_matrix = base.iloc[train].copy()
        test_matrix = base.iloc[test].copy()
        train_matrix["learned_history_h1"] = h_train
        test_matrix["learned_history_h1"] = h_test
        learned_model = make_regressor(100 * fold + 99)
        learned_model.fit(train_matrix, target[train])
        prediction = learned_model.predict(test_matrix)
        errors = np.linalg.norm(target[test] - prediction, axis=1)
        fold_rows.append(
            {
                "model": "+ learned history coordinate",
                "fold": fold,
                "mean_euclidean_error": float(errors.mean()),
                "r2": float(
                    r2_score(
                        target[test],
                        prediction,
                        multioutput="variance_weighted",
                    )
                ),
                "n_test": int(len(test)),
            }
        )
    folds = pd.DataFrame(fold_rows)
    summary = folds.groupby("model", as_index=False).agg(
        mean_error=("mean_euclidean_error", "mean"),
        sd_error=("mean_euclidean_error", "std"),
        mean_r2=("r2", "mean"),
        sd_r2=("r2", "std"),
    )
    state_error = float(summary.loc[summary["model"].eq("State only"), "mean_error"].iloc[0])
    full_error = float(summary.loc[summary["model"].eq("+ full history"), "mean_error"].iloc[0])
    gain = state_error - full_error
    summary["error_reduction_vs_state_percent"] = 100 * (state_error - summary["mean_error"]) / state_error
    summary["fraction_full_history_gain"] = (state_error - summary["mean_error"]) / gain if gain > 0 else np.nan

    # Fit one display-only coordinate on all benchmark rows. This is not used
    # for held-out scores above; its role is solely to draw the learned axis.
    full_history = data[list(HISTORY_NUMERIC)].fillna(-1.0)
    display_scaler = StandardScaler()
    scaled = display_scaler.fit_transform(full_history)
    display_pls = PLSRegression(n_components=1, scale=False, max_iter=500)
    display_pls.fit(scaled, target)
    display_h = display_pls.transform(scaled).ravel()
    orientation_reference = (
        data["line_number"].fillna(0).to_numpy(float) + data["start_month"].fillna(0).to_numpy(float) / 24
    )
    sign = 1.0 if np.corrcoef(display_h, orientation_reference)[0, 1] >= 0 else -1.0
    report = {
        "n_pairs": int(len(data)),
        "n_trajectories": int(data["_gid"].nunique()),
        "n_group_folds": 4,
        "target": "six-month PC1-PC2 displacement",
        "estimator": (
            "ExtraTreesRegressor; trajectory-grouped GroupKFold; one-dimensional fold-specific PLS history coordinate"
        ),
        "fold_metrics": fold_rows,
        "out_of_fold_learned_h1_quantiles": (
            pd.Series(learned_coordinates).quantile([0.01, 0.25, 0.5, 0.75, 0.99]).to_dict()
        ),
    }
    return summary, report, (display_scaler, display_pls, sign)


def benchmark_figure(
    summary: pd.DataFrame,
    report: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    order = [
        "State only",
        "+ diagnosis time",
        "+ treatment line",
        "+ time since treatment",
        "+ clinical phase",
        "+ hybrid clinical clock",
        "+ learned history coordinate",
        "+ full history",
    ]
    display = summary.set_index("model").loc[order].reset_index()
    colors = [
        "#7b8797",
        BLUE,
        TEAL,
        ORANGE,
        PURPLE,
        GOLD,
        RED,
        INK,
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.6))
    fig.patch.set_facecolor(PAPER)

    ax = axes[0]
    y = np.arange(len(display))
    ax.barh(
        y,
        display["mean_error"],
        xerr=display["sd_error"],
        color=colors,
        alpha=0.88,
        error_kw={"ecolor": INK, "elinewidth": 0.8, "capsize": 2},
    )
    ax.set_yticks(y, display["model"], fontsize=8.4)
    ax.invert_yaxis()
    ax.set_xlabel("Held-out mean 6-month displacement error\n(lower is better)")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(display["mean_error"]):
        ax.text(value + 0.004, index, f"{value:.3f}", va="center", fontsize=7.4)
    ax.set_title(
        "A · Which clock predicts the future?",
        loc="left",
        fontsize=11,
        weight="bold",
        color=INK,
    )

    ax = axes[1]
    candidates = display[~display["model"].isin(["State only", "+ full history"])].copy()
    recovery = 100 * candidates["fraction_full_history_gain"]
    bar_colors = [colors[order.index(item)] for item in candidates["model"]]
    ax.barh(np.arange(len(candidates)), recovery, color=bar_colors, alpha=0.88)
    ax.axvline(100, color=INK, lw=1.1, linestyle="--")
    ax.axvline(0, color="#9aa5b3", lw=0.8)
    ax.set_yticks(
        np.arange(len(candidates)),
        candidates["model"].str.replace("+ ", "", regex=False),
        fontsize=8.4,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Share of the full-history improvement recovered (%)")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(recovery):
        ax.text(
            value + (2 if value >= 0 else -2),
            index,
            f"{value:.0f}%",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=7.5,
        )
    ax.set_title(
        "B · How much history does one scalar recover?",
        loc="left",
        fontsize=11,
        weight="bold",
        color=INK,
    )
    fig.suptitle(
        "There is no need to guess the “right” clock—it can be tested",
        x=0.035,
        y=1.01,
        ha="left",
        fontsize=18,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.035,
        0.945,
        (
            f"{report['n_pairs']:,} approximately six-month transitions from "
            f"{report['n_trajectories']:,} generated trajectories · held out by "
            "entire trajectory · current PC1–PC2, cancer and model are always included"
        ),
        fontsize=9,
        color=MUTED,
    )
    fig.text(
        0.035,
        0.015,
        (
            "The learned coordinate is fitted inside each training fold from "
            "preceding-path summaries only. This exploratory benchmark tests "
            "compression of generated histories; it is not yet a clinical validation."
        ),
        fontsize=8.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.995, 0.91), w_pad=2.0)
    return save_figure(fig, "fig03_data_clock_benchmark"), {
        **report,
        "summary": summary.to_dict(orient="records"),
    }


def apply_history_coordinate(
    history: pd.DataFrame,
    scaler: StandardScaler,
    pls: PLSRegression,
    sign: float,
) -> pd.DataFrame:
    work = history.copy()
    values = work[list(HISTORY_NUMERIC)].fillna(-1.0)
    work["history_h1"] = sign * pls.transform(scaler.transform(values)).ravel()
    median = float(work["history_h1"].median())
    scale = float(work["history_h1"].quantile(0.75) - work["history_h1"].quantile(0.25))
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    work["history_h1"] = (work["history_h1"] - median) / scale
    return work


def select_display_trajectories(
    atlas: pd.DataFrame,
    history: pd.DataFrame,
    *,
    n: int = 6,
) -> pd.DataFrame:
    merged = atlas.merge(
        history[["_gid", "start_day", "history_h1"]],
        left_on=["_gid", "days_from_diagnosis"],
        right_on=["_gid", "start_day"],
        how="inner",
        validate="many_to_one",
    )
    counts = (
        merged[merged["tumor_oncotree_code"].eq("LUAD") & merged["model_short"].eq("claude_sonnet5")]
        .groupby("_gid")
        .agg(n=("snapshot_index", "size"), span=("days_from_diagnosis", "max"))
        .query("n >= 12")
        .sort_values(["span", "n"], ascending=False)
    )
    chosen = counts.head(n).index
    return merged[merged["_gid"].isin(chosen)].copy()


def learned_coordinate_figure(
    atlas: pd.DataFrame,
    history: pd.DataFrame,
    limits: tuple[float, float, float, float],
) -> tuple[dict[str, str], dict[str, Any]]:
    selected = select_display_trajectories(atlas, history)
    colors = plt.cm.tab10(np.linspace(0, 1, selected["_gid"].nunique()))
    color_by_gid = dict(zip(sorted(selected["_gid"].unique()), colors, strict=True))
    fig = plt.figure(figsize=(16.0, 5.5), facecolor=PAPER)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])

    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    for gid, course in selected.groupby("_gid"):
        course = course.sort_values(["days_from_diagnosis", "snapshot_index"])
        color = color_by_gid[gid]
        ax3d.plot(
            course["map_x"],
            course["map_y"],
            course["history_h1"],
            "-o",
            color=color,
            lw=1.5,
            ms=2.5,
            alpha=0.90,
        )
    ax3d.set_xlabel("PC1", fontsize=8)
    ax3d.set_ylabel("PC2", fontsize=8)
    ax3d.set_zlabel("learned history h₁", fontsize=8)
    ax3d.view_init(elev=22, azim=-55)
    ax3d.set_title(
        "A · Literal augmented space",
        loc="left",
        fontsize=10.5,
        weight="bold",
        color=INK,
    )

    ax = fig.add_subplot(grid[0, 1])
    for gid, course in selected.groupby("_gid"):
        course = course.sort_values(["days_from_diagnosis", "snapshot_index"])
        color = color_by_gid[gid]
        ax.plot(
            course["history_h1"],
            course["map_x"],
            "-o",
            color=color,
            lw=1.7,
            ms=3,
            alpha=0.90,
        )
    ax.axhline(0, color=GRID, lw=0.8)
    ax.grid(color=GRID, lw=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("learned predictive history coordinate h₁")
    ax.set_ylabel("clinical state PC1\nless severe  \u2190     \u2192  more severe")
    ax.set_title(
        "B · PC1 × learned history",
        loc="left",
        fontsize=10.5,
        weight="bold",
        color=INK,
    )

    ax = fig.add_subplot(grid[0, 2])
    background_points(ax, atlas)
    norm = plt.Normalize(
        selected["history_h1"].quantile(0.02),
        selected["history_h1"].quantile(0.98),
    )
    for gid, course in selected.groupby("_gid"):
        course = course.sort_values(["days_from_diagnosis", "snapshot_index"])
        ax.plot(
            course["map_x"],
            course["map_y"],
            color="#9aa5b3",
            lw=0.7,
            alpha=0.55,
            zorder=3,
        )
        ax.scatter(
            course["map_x"],
            course["map_y"],
            c=course["history_h1"],
            cmap="plasma",
            norm=norm,
            s=19,
            edgecolor="white",
            lw=0.25,
            zorder=5,
        )
    style_2d_map(ax, limits)
    ax.set_title(
        "C · Keep the map; encode history by colour",
        loc="left",
        fontsize=10.5,
        weight="bold",
        color=INK,
    )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="plasma"),
        ax=ax,
        fraction=0.048,
        pad=0.03,
    )
    colorbar.set_label("learned history h₁", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        "What does a one-dimensional history embedding look like?",
        x=0.035,
        y=1.02,
        ha="left",
        fontsize=18,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.035,
        0.945,
        (
            f"{selected['_gid'].nunique()} stored Sonnet-generated LUAD "
            "trajectories. h₁ is a supervised "
            "one-component compression of preceding-path summaries for six-month "
            "movement; its sign and scale are arbitrary."
        ),
        fontsize=9,
        color=MUTED,
    )
    fig.text(
        0.035,
        0.012,
        (
            "The geometry is real generated data, but this display-only h₁ was "
            "refitted on the full exploratory set. Held-out performance is reported "
            "separately in Fig. 3."
        ),
        fontsize=8.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.995, 0.90), w_pad=1.5)
    report = {
        "n_display_trajectories": int(selected["_gid"].nunique()),
        "n_display_snapshots": int(len(selected)),
        "cancer": "LUAD",
        "model": "claude_sonnet5",
        "history_h1_quantiles": (selected["history_h1"].quantile([0.01, 0.25, 0.5, 0.75, 0.99]).to_dict()),
    }
    return save_figure(fig, "fig04_data_learned_history_coordinate"), report


def display_map_report(display: DisplayMap) -> dict[str, Any]:
    return {
        "features": list(display.features),
        "explained_variance": display.explained_variance.tolist(),
        "centers": display.centers.tolist(),
        "scales": display.scales.tolist(),
    }


def write_readme(report: dict[str, Any]) -> None:
    summary = pd.DataFrame(report["clock_benchmark"]["summary"]).set_index("model")
    best_simple = summary.drop(index=["State only", "+ full history"]).sort_values("mean_error").iloc[0]
    lines = [
        "# History-clock field exercise",
        "",
        "## Central decision",
        "",
        "The clinical PC1–PC2 geography remains fixed. History can be added "
        "mathematically as a coordinate, but the main visual recommendation is "
        "to show sequential slices through the unchanged map.",
        "",
        "## Sources",
        "",
        "- Figure 1 is explicitly conceptual mock geometry.",
        "- Figures 2–4 use stored model-generated project trajectories, not human patients.",
        "- Figure 2 uses paired Sonnet Phase-1 generated counterfactual futures.",
        "",
        "## Clock benchmark",
        "",
        f"The benchmark contains {report['clock_benchmark']['n_pairs']:,} "
        "approximately six-month transitions from "
        f"{report['clock_benchmark']['n_trajectories']:,} trajectories, with "
        "entire trajectories held out together.",
        "",
        f"The best one-dimensional/simple augmentation in this run was "
        f"**{best_simple.name}** with mean displacement error "
        f"{best_simple['mean_error']:.3f}. Full history achieved "
        f"{summary.loc['+ full history', 'mean_error']:.3f}, versus "
        f"{summary.loc['State only', 'mean_error']:.3f} for state only.",
        "",
        "## Interpretation boundaries",
        "",
        "- Ordinary time is monotone, so a remission attractor in augmented "
        "space is a moving/history-conditioned destination, not a fixed "
        "space–time point.",
        "- Time since treatment and line number are clinical clocks; they do "
        "not encode treatment identity, response sequence, comorbidity changes, "
        "or the entire preceding path.",
        "- Later panels in the treatment-age movie are survivor-conditional.",
        "- The learned h1 coordinate is supervised and exploratory. Its sign "
        "and scale are arbitrary, and it requires external validation before "
        "clinical interpretation.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    figure_1 = conceptual_history_figure()

    atlas_tables = load_tables()
    atlas_raw = enrich_snapshots(atlas_tables)
    display, atlas = fit_display_map(atlas_raw)
    atlas = add_phase_labels(atlas)

    phase1_tables = load_derivation(PHASE1_ROOT)
    phase1 = transform_to_map(enrich_snapshots(phase1_tables), display)
    limits = map_limits(atlas, phase1)
    movie_vectors = phase1_treatment_age_vectors(phase1, phase1_tables.events)
    figure_2, movie_report = treatment_age_movie(movie_vectors, atlas, limits)

    pairs = fixed_horizon_pairs(atlas)
    history = history_feature_table(atlas)
    benchmark = prepare_benchmark(pairs, history)
    benchmark_summary, benchmark_report, coordinate_model = benchmark_clocks(benchmark)
    figure_3, clock_report = benchmark_figure(benchmark_summary, benchmark_report)

    scaler, pls, sign = coordinate_model
    history_with_h1 = apply_history_coordinate(history, scaler, pls, sign)
    figure_4, coordinate_report = learned_coordinate_figure(
        atlas,
        history_with_h1,
        limits,
    )

    report: dict[str, Any] = {
        "contract": {
            "state_space": "one frozen 16-feature atlas PCA",
            "future_horizon_months": HORIZON_MONTHS,
            "movie_forecast_months": FORECAST_MONTHS,
            "movie_ages_months": list(MOVIE_AGES),
            "benchmark_split": "GroupKFold by complete trajectory",
            "history_features_use_future_information": False,
        },
        "display_map": display_map_report(display),
        "figures": {
            "concept_history_views": figure_1,
            "treatment_age_movie": figure_2,
            "clock_benchmark": figure_3,
            "learned_history_coordinate": figure_4,
        },
        "treatment_age_movie": movie_report,
        "clock_benchmark": clock_report,
        "learned_history_coordinate": coordinate_report,
        "counts": {
            "atlas_snapshots": int(len(atlas)),
            "atlas_trajectories": int(atlas["_gid"].nunique()),
            "eligible_six_month_pairs": int(len(pairs)),
            "benchmark_pairs": int(len(benchmark)),
            "movie_vectors": int(len(movie_vectors)),
        },
    }
    (OUT / "readout.json").write_text(json.dumps(report, indent=2) + "\n")
    write_readme(report)
    for figure in report["figures"].values():
        print(f"[done] {figure['png']}")
    print(f"[done] {OUT.relative_to(ROOT) / 'README.md'}")
    print(f"[done] {OUT.relative_to(ROOT) / 'readout.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
