#!/usr/bin/env python3
"""Faceted finite-horizon clinical fields on one frozen state map.

It separates:

* one global PCA state map fitted once from 16 contemporaneous tumour + host
  features;
* context, which changes the six-month displacement field but never the axes;
* finite-horizon destination basins, which are empirical convergence regions
  rather than claims about asymptotic dynamical-system attractors;
* paired generated counterfactual treatment futures, which are the appropriate
  source for treatment-specific differences.

Run:
  uv run python scripts/explore/build_faceted_clinical_field_atlas.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_one_map_many_futures_exercise import (  # noqa: E402
    PHASE1_ROOT,
    DisplayMap,
    fit_display_map,
    load_derivation,
    map_limits,
    transform_to_map,
)
from composite_tumor_host_phase_space import (  # noqa: E402
    MONTH_DAYS,
    enrich_snapshots,
    load_tables,
)

OUT = ROOT / "outputs" / "exploratory_faceted_clinical_field_atlas"

INK = "#182334"
MUTED = "#66758c"
GRID = "#dce4ec"
PAPER = "#fbfcfe"
BLUE = "#3474b8"
TEAL = "#138f86"
GREEN = "#208a53"
ORANGE = "#df7900"
RED = "#c63e3e"
PURPLE = "#6846a5"
GOLD = "#c59a24"
LIGHT_BLUE = "#dcecf5"
SEED = 20260728

HORIZON_MONTHS = 6.0
HORIZON_DAYS = HORIZON_MONTHS * MONTH_DAYS
TOLERANCE_DAYS = 2.5 * MONTH_DAYS

CANCERS = ("LUAD", "HGSOC", "PAAD")
CANCER_LABEL = {
    "LUAD": "Lung adenocarcinoma",
    "HGSOC": "High-grade serous ovarian cancer",
    "PAAD": "Pancreatic adenocarcinoma",
}
PHASES = ("diagnosis", "first_line", "post_progression")
PHASE_LABEL = {
    "diagnosis": "Before first systemic therapy",
    "first_line": "First line, before progression",
    "post_progression": "After first progression",
}
MODEL_LABEL = {
    "claude_sonnet5": "Claude Sonnet 5",
    "gemini31_lite": "Gemini 3.1 Lite",
    "gemma4": "Gemma 4",
    "glm52_fp8": "GLM-5.2",
    "kimi_k3_off": "Kimi K3",
    "qwen36_flash": "Qwen 3.6 Flash",
}
PHASE_COLOR = {
    "diagnosis": BLUE,
    "first_line": TEAL,
    "post_progression": RED,
}


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    """Save a publication-readable raster and vector copy."""
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
    linewidth: float,
    alpha: float,
    mutation_scale: float,
    zorder: int,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            lw=linewidth,
            color=color,
            alpha=alpha,
            zorder=zorder,
            shrinkA=0,
            shrinkB=0,
        )
    )


def style_map(
    ax: plt.Axes,
    limits: tuple[float, float, float, float],
    *,
    row: int,
    col: int,
) -> None:
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.grid(color=GRID, lw=0.65, alpha=0.82)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    if row >= 0:
        ax.set_ylabel(
            "Clinical state PC2\n(residual state pattern)" if col == 0 else "",
            fontsize=8,
        )
    if col >= 0:
        ax.set_xlabel(
            "Clinical state PC1\nless severe  \u2190     \u2192  more severe",
            fontsize=8,
        )


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
        alpha=0.09,
        linewidth=0,
        rasterized=True,
        zorder=0,
    )


def add_phase_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign three clinically legible phases from observed event landmarks."""
    chunks: list[pd.DataFrame] = []
    for _, course in frame.groupby("_gid", sort=False):
        course = course.sort_values(["days_from_diagnosis", "snapshot_index"]).copy()
        days = pd.to_numeric(course["days_from_diagnosis"], errors="coerce")
        event = course["anchor_event_type"].fillna("").astype(str)
        starts = days[event.eq("systemic_treatment_start")]
        first_start = float(starts.min()) if starts.notna().any() else np.inf
        progressions = days[event.eq("progression") & days.ge(first_start)]
        first_progression = float(progressions.min()) if progressions.notna().any() else np.inf
        phase = np.full(len(course), "diagnosis", dtype=object)
        phase[days.to_numpy(float) >= first_start] = "first_line"
        phase[days.to_numpy(float) >= first_progression] = "post_progression"
        course["clinical_phase"] = phase
        chunks.append(course)
    return pd.concat(chunks, ignore_index=True)


def collapse_same_day(course: pd.DataFrame) -> pd.DataFrame:
    """Keep the last post-event state when several events share a day."""
    return (
        course.sort_values(["days_from_diagnosis", "snapshot_index"])
        .drop_duplicates("days_from_diagnosis", keep="last")
        .reset_index(drop=True)
    )


def fixed_horizon_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Match each state to the nearest observable state at six months."""
    rows: list[dict[str, object]] = []
    for gid, raw_course in frame.groupby("_gid", sort=False):
        course = collapse_same_day(raw_course)
        course = course[~course["terminal"]].reset_index(drop=True)
        if len(course) < 2:
            continue
        days = pd.to_numeric(course["days_from_diagnosis"], errors="coerce").to_numpy(float)
        for index, start in course.iterrows():
            target = days[index] + HORIZON_DAYS
            future_indices = np.flatnonzero(days > days[index])
            if not len(future_indices):
                continue
            nearest_index = int(future_indices[np.argmin(np.abs(days[future_indices] - target))])
            if abs(days[nearest_index] - target) > TOLERANCE_DAYS:
                continue
            end = course.iloc[nearest_index]
            rows.append(
                {
                    "_gid": gid,
                    "trajectory_id": start["trajectory_id"],
                    "model": start["model_short"],
                    "cancer": start["tumor_oncotree_code"],
                    "phase": start["clinical_phase"],
                    "start_day": float(days[index]),
                    "end_day": float(days[nearest_index]),
                    "dt_months": float((days[nearest_index] - days[index]) / MONTH_DAYS),
                    "x0": float(start["map_x"]),
                    "y0": float(start["map_y"]),
                    "x1": float(end["map_x"]),
                    "y1": float(end["map_y"]),
                    "dx": float(end["map_x"] - start["map_x"]),
                    "dy": float(end["map_y"] - start["map_y"]),
                }
            )
    return pd.DataFrame(rows)


def first_landmark_per_phase(pairs: pd.DataFrame) -> pd.DataFrame:
    """Give each trajectory equal weight within a clinical-phase facet."""
    return (
        pairs.sort_values(["_gid", "phase", "start_day"])
        .groupby(["_gid", "phase"], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def local_field(vectors: pd.DataFrame, max_centres: int = 6) -> pd.DataFrame:
    """Estimate local displacement arrows only where starts were observed."""
    if vectors.empty:
        return pd.DataFrame()
    n_centres = min(max_centres, max(1, len(vectors) // 5))
    points = vectors[["x0", "y0"]].to_numpy(float)
    if n_centres == 1:
        labels = np.zeros(len(points), dtype=int)
    else:
        labels = KMeans(
            n_clusters=n_centres,
            random_state=SEED,
            n_init=20,
        ).fit_predict(points)
    work = vectors.copy()
    work["_cluster"] = labels
    return (
        work.groupby("_cluster", as_index=False)
        .agg(
            x0=("x0", "median"),
            y0=("y0", "median"),
            dx=("dx", "mean"),
            dy=("dy", "mean"),
            n=("dx", "size"),
        )
        .sort_values("_cluster")
    )


def draw_displacement_field(
    ax: plt.Axes,
    vectors: pd.DataFrame,
    atlas: pd.DataFrame,
    limits: tuple[float, float, float, float],
    color: str,
    *,
    row: int,
    col: int,
    max_centres: int = 6,
) -> pd.DataFrame:
    """Draw individual evidence and its local equal-horizon average."""
    background_points(ax, atlas)
    for item in vectors.itertuples(index=False):
        add_arrow(
            ax,
            (float(item.x0), float(item.y0)),
            (float(item.x1), float(item.y1)),
            color,
            linewidth=0.65,
            alpha=0.16,
            mutation_scale=5,
            zorder=2,
        )
    field = local_field(vectors, max_centres=max_centres)
    for item in field.itertuples(index=False):
        ax.scatter(
            item.x0,
            item.y0,
            s=16 + 7 * item.n,
            facecolor="white",
            edgecolor=color,
            lw=1.2,
            zorder=7,
        )
        add_arrow(
            ax,
            (float(item.x0), float(item.y0)),
            (float(item.x0 + item.dx), float(item.y0 + item.dy)),
            color,
            linewidth=2.2,
            alpha=0.96,
            mutation_scale=13,
            zorder=8,
        )
    style_map(ax, limits, row=row, col=col)
    return field


def cancer_phase_figure(
    landmarks: pd.DataFrame,
    atlas: pd.DataFrame,
    limits: tuple[float, float, float, float],
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    fig, axes = plt.subplots(3, 3, figsize=(14.8, 12.2), sharex=True, sharey=True)
    fig.patch.set_facecolor(PAPER)
    report: dict[str, dict[str, float]] = {}
    for row, cancer in enumerate(CANCERS):
        for col, phase in enumerate(PHASES):
            ax = axes[row, col]
            selected = landmarks[landmarks["cancer"].eq(cancer) & landmarks["phase"].eq(phase)]
            field = draw_displacement_field(
                ax,
                selected,
                atlas,
                limits,
                PHASE_COLOR[phase],
                row=row,
                col=col,
            )
            if row == 0:
                ax.set_title(
                    PHASE_LABEL[phase],
                    loc="left",
                    fontsize=10,
                    weight="bold",
                    color=PHASE_COLOR[phase],
                    pad=10,
                )
            ax.text(
                0.03,
                0.96,
                (f"{CANCER_LABEL[cancer]}\nn={selected['_gid'].nunique()} trajectories"),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.7,
                color=INK,
                bbox={
                    "boxstyle": "round,pad=0.30",
                    "facecolor": "white",
                    "edgecolor": PHASE_COLOR[phase],
                    "alpha": 0.94,
                },
                zorder=15,
            )
            report[f"{cancer}::{phase}"] = {
                "n_trajectories": int(selected["_gid"].nunique()),
                "n_local_arrows": int(len(field)),
                "median_actual_horizon_months": float(selected["dt_months"].median()),
                "mean_dx": float(selected["dx"].mean()),
                "mean_dy": float(selected["dy"].mean()),
            }
    fig.suptitle(
        "One geography, clinically faceted six-month fields",
        x=0.04,
        y=0.995,
        ha="left",
        fontsize=19,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.04,
        0.965,
        (
            "Columns change clinical phase; rows change cancer. Thin arrows are "
            "individual generated trajectories; bold arrows are local averages. "
            "The PCA axes and scale never change."
        ),
        fontsize=9.3,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.025, 0.025, 0.995, 0.94), h_pad=2.0, w_pad=1.0)
    return save_figure(fig, "fig01_cancer_by_phase_fields"), report


def model_figure(
    landmarks: pd.DataFrame,
    atlas: pd.DataFrame,
    limits: tuple[float, float, float, float],
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    models = tuple(sorted(MODEL_LABEL))
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.5), sharex=True, sharey=True)
    fig.patch.set_facecolor(PAPER)
    report: dict[str, dict[str, float]] = {}
    pool = landmarks[
        landmarks["cancer"].isin(CANCERS) & landmarks["phase"].eq("first_line") & landmarks["model"].isin(models)
    ]
    for index, model in enumerate(models):
        row, col = divmod(index, 3)
        ax = axes[row, col]
        selected = pool[pool["model"].eq(model)]
        field = draw_displacement_field(
            ax,
            selected,
            atlas,
            limits,
            PURPLE,
            row=row,
            col=col,
            max_centres=4,
        )
        counts = selected.groupby("cancer")["_gid"].nunique().to_dict()
        support = " · ".join(f"{key} {counts.get(key, 0)}" for key in CANCERS)
        ax.set_title(
            f"{MODEL_LABEL[model]}\nn={selected['_gid'].nunique()} ({support})",
            loc="left",
            fontsize=9.4,
            weight="bold",
            color=INK,
        )
        report[model] = {
            "n_trajectories": int(selected["_gid"].nunique()),
            "n_local_arrows": int(len(field)),
            "mean_dx": float(selected["dx"].mean()),
            "mean_dy": float(selected["dy"].mean()),
        }
    fig.suptitle(
        "The field is model-specific—even on the same clinical map",
        x=0.04,
        y=0.995,
        ha="left",
        fontsize=19,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.04,
        0.955,
        (
            "Balanced LUAD + HGSOC + PAAD set · first-line, pre-progression · "
            "one earliest landmark per trajectory · six-month displacement"
        ),
        fontsize=9.3,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.025, 0.025, 0.995, 0.925), h_pad=2.2, w_pad=1.2)
    return save_figure(fig, "fig02_model_specific_fields"), report


def interpolate_map_position(course: pd.DataFrame, day: float) -> tuple[float, float]:
    """Linearly interpolate display coordinates at an exact landmark day."""
    course = collapse_same_day(course)
    days = pd.to_numeric(course["days_from_diagnosis"], errors="coerce").to_numpy(float)
    if day <= days[0]:
        row = course.iloc[0]
        return float(row["map_x"]), float(row["map_y"])
    if day >= days[-1]:
        row = course.iloc[-1]
        return float(row["map_x"]), float(row["map_y"])
    upper = int(np.searchsorted(days, day, side="left"))
    if days[upper] == day:
        row = course.iloc[upper]
        return float(row["map_x"]), float(row["map_y"])
    lower = upper - 1
    fraction = float((day - days[lower]) / (days[upper] - days[lower]))
    x = float(course.iloc[lower]["map_x"] + fraction * (course.iloc[upper]["map_x"] - course.iloc[lower]["map_x"]))
    y = float(course.iloc[lower]["map_y"] + fraction * (course.iloc[upper]["map_y"] - course.iloc[lower]["map_y"]))
    return x, y


def phase1_exact_horizon_vectors(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Build exact six-month paired-CF vectors by display-space interpolation."""
    rows: list[dict[str, object]] = []
    eligible = snapshots[
        snapshots["condition_id"].astype(str).str.contains("EGFR_1L")
        & snapshots["condition_id"].astype(str).str.endswith("__H1")
        & snapshots["policy_id"].astype(str).str.endswith(("__osimertinib", "__chemo"))
    ]
    for gid, course in eligible.groupby("_gid", sort=False):
        meta = course.iloc[0]
        assigned = events[events["_gid"].eq(gid) & events["event_type"].astype(str).eq("systemic_treatment_start")]
        if assigned.empty:
            continue
        t0 = float(pd.to_numeric(assigned["days_from_diagnosis"], errors="coerce").min())
        target = t0 + HORIZON_DAYS
        days = pd.to_numeric(course["days_from_diagnosis"], errors="coerce")
        if days.max() < target:
            continue
        x0, y0 = interpolate_map_position(course, t0)
        x1, y1 = interpolate_map_position(course, target)
        parts = str(meta["condition_id"]).split("__")
        rows.append(
            {
                "_gid": gid,
                "state": parts[-2],
                "history": parts[-1],
                "arm": str(meta["policy_id"]).split("__")[-1],
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "dx": x1 - x0,
                "dy": y1 - y0,
            }
        )
    return pd.DataFrame(rows)


def treatment_summary(vectors: pd.DataFrame) -> pd.DataFrame:
    return (
        vectors.groupby(["state", "arm"], as_index=False)
        .agg(
            x0=("x0", "median"),
            y0=("y0", "median"),
            dx=("dx", "mean"),
            dy=("dy", "mean"),
            n=("dx", "size"),
        )
        .assign(x1=lambda data: data["x0"] + data["dx"])
        .assign(y1=lambda data: data["y0"] + data["dy"])
    )


def draw_treatment_panel(
    ax: plt.Axes,
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    atlas: pd.DataFrame,
    limits: tuple[float, float, float, float],
    arm: str,
    color: str,
    col: int,
) -> None:
    selected = raw[raw["arm"].eq(arm)]
    background_points(ax, atlas)
    for item in selected.itertuples(index=False):
        add_arrow(
            ax,
            (item.x0, item.y0),
            (item.x1, item.y1),
            color,
            linewidth=0.7,
            alpha=0.18,
            mutation_scale=5,
            zorder=2,
        )
    for item in summary[summary["arm"].eq(arm)].itertuples(index=False):
        ax.scatter(
            item.x0,
            item.y0,
            s=68,
            facecolor="white",
            edgecolor=color,
            lw=1.2,
            zorder=7,
        )
        add_arrow(
            ax,
            (item.x0, item.y0),
            (item.x1, item.y1),
            color,
            linewidth=2.5,
            alpha=0.98,
            mutation_scale=14,
            zorder=8,
        )
        ax.text(
            item.x0 + 0.05,
            item.y0 + 0.05,
            item.state,
            fontsize=7.4,
            weight="bold",
            color=INK,
            zorder=10,
        )
    style_map(ax, limits, row=0, col=col)
    ax.set_title(
        f"{arm.capitalize()}\nmean six-month future from each starting state",
        loc="left",
        fontsize=10,
        weight="bold",
        color=color,
    )


def treatment_figure(
    vectors: pd.DataFrame,
    atlas: pd.DataFrame,
    limits: tuple[float, float, float, float],
) -> tuple[dict[str, str], dict[str, object]]:
    summary = treatment_summary(vectors)
    fig, axes = plt.subplots(1, 3, figsize=(15.3, 5.3), sharex=True, sharey=True)
    fig.patch.set_facecolor(PAPER)
    draw_treatment_panel(axes[0], vectors, summary, atlas, limits, "osimertinib", TEAL, 0)
    draw_treatment_panel(axes[1], vectors, summary, atlas, limits, "chemo", ORANGE, 1)

    ax = axes[2]
    background_points(ax, atlas)
    delta_rows: list[dict[str, float | str]] = []
    for state in sorted(summary["state"].unique()):
        state_rows = summary[summary["state"].eq(state)].set_index("arm")
        if not {"osimertinib", "chemo"}.issubset(state_rows.index):
            continue
        osi = state_rows.loc["osimertinib"]
        chemo = state_rows.loc["chemo"]
        x0 = float(np.mean([osi["x0"], chemo["x0"]]))
        y0 = float(np.mean([osi["y0"], chemo["y0"]]))
        ddx = float(osi["dx"] - chemo["dx"])
        ddy = float(osi["dy"] - chemo["dy"])
        ax.scatter(
            x0,
            y0,
            s=78,
            facecolor="white",
            edgecolor=PURPLE,
            lw=1.3,
            zorder=7,
        )
        add_arrow(
            ax,
            (x0, y0),
            (x0 + ddx, y0 + ddy),
            PURPLE,
            linewidth=2.7,
            alpha=0.98,
            mutation_scale=14,
            zorder=8,
        )
        ax.text(
            x0 + 0.05,
            y0 + 0.05,
            state,
            fontsize=7.4,
            weight="bold",
            color=INK,
            zorder=10,
        )
        delta_rows.append({"state": state, "delta_dx": ddx, "delta_dy": ddy})
    style_map(ax, limits, row=0, col=2)
    ax.set_title(
        "Treatment effect\nosimertinib displacement \u2212 chemotherapy",
        loc="left",
        fontsize=10,
        weight="bold",
        color=PURPLE,
    )
    fig.suptitle(
        "Treatment changes the field—not the clinical geography",
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
            "EGFR-mutant first-line NSCLC · H1 matched history · five starting "
            "states · n=3 generated futures/state/arm · exact \u03c4=6 months"
        ),
        fontsize=9.2,
        color=MUTED,
    )
    fig.text(
        0.035,
        0.045,
        (
            "Start-state key: S1 fit/low burden, ECOG 0 · S2 fit/liver "
            "micrometastasis, ECOG 1 · S3 brain + liver micrometastases, "
            "ECOG 1 · S4 liver + bone disease, ECOG 1 · S5 frail/higher "
            "burden, ECOG 2"
        ),
        fontsize=7.5,
        color=INK,
    )
    fig.text(
        0.035,
        0.012,
        (
            "Positions at exactly six months are linearly interpolated between "
            "generated snapshots. These are paired model-generated "
            "counterfactuals, not measured human causal effects."
        ),
        fontsize=8.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.095, 0.995, 0.90), w_pad=1.2)
    report = {
        "n_by_arm": vectors.groupby("arm")["_gid"].nunique().to_dict(),
        "n_starting_states": int(vectors["state"].nunique()),
        "delta_vectors": delta_rows,
        "endpoint_interpolation": "linear in frozen PCA display coordinates",
    }
    return save_figure(fig, "fig03_treatment_specific_fields"), report


def covariance_ellipse(
    ax: plt.Axes,
    points: np.ndarray,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str,
) -> None:
    if len(points) < 3:
        return
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values = np.maximum(values[order], 0)
    vectors = vectors[:, order]
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
    width, height = 2 * 1.52 * np.sqrt(values)
    centre = points.mean(axis=0)
    ax.add_patch(
        Ellipse(
            centre,
            width=float(width),
            height=float(height),
            angle=angle,
            fill=False,
            edgecolor=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=7,
        )
    )


def endpoint_density(
    ax: plt.Axes,
    endpoints: np.ndarray,
    limits: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Draw endpoint KDE contours and return the density peak."""
    xpad = 0.03 * (limits[1] - limits[0])
    ypad = 0.03 * (limits[3] - limits[2])
    xmin = max(limits[0] + xpad, float(endpoints[:, 0].min() - 0.8))
    xmax = min(limits[1] - xpad, float(endpoints[:, 0].max() + 0.8))
    ymin = max(limits[2] + ypad, float(endpoints[:, 1].min() - 0.8))
    ymax = min(limits[3] - ypad, float(endpoints[:, 1].max() + 0.8))
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 100),
        np.linspace(ymin, ymax, 100),
    )
    try:
        kde = gaussian_kde(endpoints.T)
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        positive = zz[zz > 0]
        levels = np.quantile(positive, [0.55, 0.72, 0.86, 0.95])
        levels = np.unique(levels)
        if len(levels) >= 2:
            ax.contourf(xx, yy, zz, levels=np.r_[levels, zz.max() * 1.001], cmap="Greens", alpha=0.16)
            ax.contour(xx, yy, zz, levels=levels, colors=GREEN, alpha=0.45, linewidths=0.8)
        peak_index = np.unravel_index(np.argmax(zz), zz.shape)
        return float(xx[peak_index]), float(yy[peak_index])
    except np.linalg.LinAlgError:
        centre = np.median(endpoints, axis=0)
        return float(centre[0]), float(centre[1])


def attractor_metrics(vectors: pd.DataFrame) -> dict[str, float | bool | str]:
    starts = vectors[["x0", "y0"]].to_numpy(float)
    ends = vectors[["x1", "y1"]].to_numpy(float)
    kde = gaussian_kde(ends.T)
    candidates = ends
    peak = candidates[int(np.argmax(kde(candidates.T)))]
    start_distance = np.linalg.norm(starts - peak, axis=1)
    end_distance = np.linalg.norm(ends - peak, axis=1)
    gain = start_distance - end_distance
    fraction_closer = float(np.mean(gain > 0))
    median_convergence = float(np.median(gain))
    severity_shift = float(np.median(ends[:, 0] - starts[:, 0]))
    start_spread = float(np.trace(np.cov(starts, rowvar=False)))
    end_spread = float(np.trace(np.cov(ends, rowvar=False)))
    spread_contraction = float(1 - end_spread / start_spread) if start_spread else np.nan
    candidate = bool(fraction_closer >= 0.60 and median_convergence > 0 and severity_shift < 0)
    label = (
        "finite-horizon remission attractor candidate"
        if candidate
        else "six-month destination region; convergence not established"
    )
    return {
        "peak_x": float(peak[0]),
        "peak_y": float(peak[1]),
        "fraction_moving_closer_to_peak": fraction_closer,
        "median_distance_gain": median_convergence,
        "median_pc1_shift": severity_shift,
        "spread_contraction": spread_contraction,
        "candidate": candidate,
        "label": label,
    }


def attractor_figure(
    landmarks: pd.DataFrame,
    atlas: pd.DataFrame,
    limits: tuple[float, float, float, float],
) -> tuple[dict[str, str], dict[str, dict[str, float | bool | str]]]:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.7), sharex=True, sharey=True)
    fig.patch.set_facecolor(PAPER)
    report: dict[str, dict[str, float | bool | str]] = {}
    first_treatment = (
        atlas[atlas["anchor_event_type"].astype(str).eq("systemic_treatment_start")]
        .groupby("_gid")["days_from_diagnosis"]
        .min()
    )
    for col, cancer in enumerate(CANCERS):
        ax = axes[col]
        selected = landmarks[landmarks["cancer"].eq(cancer) & landmarks["phase"].eq("diagnosis")]
        starts = selected[["x0", "y0"]].to_numpy(float)
        ends = selected[["x1", "y1"]].to_numpy(float)
        background_points(ax, atlas)
        for item in selected.itertuples(index=False):
            add_arrow(
                ax,
                (item.x0, item.y0),
                (item.x1, item.y1),
                GREEN,
                linewidth=0.8,
                alpha=0.30,
                mutation_scale=6,
                zorder=2,
            )
        peak = endpoint_density(ax, ends, limits)
        metrics = attractor_metrics(selected)
        ax.scatter(
            starts[:, 0],
            starts[:, 1],
            s=24,
            facecolor="white",
            edgecolor=BLUE,
            lw=0.9,
            alpha=0.82,
            label="early state",
            zorder=5,
        )
        ax.scatter(
            ends[:, 0],
            ends[:, 1],
            s=24,
            facecolor=GREEN,
            edgecolor="white",
            lw=0.5,
            alpha=0.80,
            label="state at 6 months",
            zorder=6,
        )
        covariance_ellipse(
            ax,
            starts,
            color=BLUE,
            linestyle="--",
            linewidth=1.5,
            label="early-state spread",
        )
        covariance_ellipse(
            ax,
            ends,
            color=GREEN,
            linestyle="-",
            linewidth=1.8,
            label="6-month destination spread",
        )
        ax.scatter(
            peak[0],
            peak[1],
            marker="*",
            s=190,
            facecolor=GOLD,
            edgecolor=INK,
            lw=0.9,
            zorder=10,
        )
        verdict_color = GREEN if metrics["candidate"] else ORANGE
        ax.text(
            0.03,
            0.97,
            (
                f"n={len(selected)} trajectories\n"
                f"{metrics['fraction_moving_closer_to_peak']:.0%} move closer "
                "to destination peak\n"
                f"median PC1 shift {metrics['median_pc1_shift']:+.2f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.7,
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.34",
                "facecolor": "white",
                "edgecolor": verdict_color,
                "linewidth": 1.3,
                "alpha": 0.95,
            },
            zorder=15,
        )
        ax.set_title(
            f"{CANCER_LABEL[cancer]}\n{metrics['label']}",
            loc="left",
            fontsize=9.4,
            weight="bold",
            color=verdict_color,
        )
        style_map(ax, limits, row=0, col=col)
        report[cancer] = {
            "n_trajectories": int(len(selected)),
            **metrics,
            "fraction_starting_treatment_by_endpoint": float(
                (
                    selected["_gid"].map(first_treatment).ge(selected["start_day"])
                    & selected["_gid"].map(first_treatment).le(selected["end_day"])
                ).mean()
            ),
            "displayed_kde_peak_x": peak[0],
            "displayed_kde_peak_y": peak[1],
        }
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle(
        "Can an early remission basin be seen?",
        x=0.035,
        y=1.015,
        ha="left",
        fontsize=18,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.035,
        0.955,
        (
            "One earliest pre-treatment landmark per trajectory. Green density "
            "shows where observable six-month futures accumulate; the star is "
            "the empirical destination-density peak."
        ),
        fontsize=9.2,
        color=MUTED,
    )
    fig.text(
        0.035,
        0.01,
        (
            "A star is only a finite-horizon destination. We call it an attractor "
            "candidate only when most paths move closer to it and PC1 severity "
            "falls; this does not prove a permanent or mechanistic attractor."
        ),
        fontsize=8.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.995, 0.90), w_pad=1.3)
    return save_figure(fig, "fig04_early_remission_attractor_candidates"), report


def display_map_report(display: DisplayMap) -> dict[str, object]:
    return {
        "features": list(display.features),
        "explained_variance": display.explained_variance.tolist(),
        "clip_low": display.clip_low.tolist(),
        "clip_high": display.clip_high.tolist(),
        "centers": display.centers.tolist(),
        "scales": display.scales.tolist(),
    }


def write_readme(report: dict[str, object]) -> None:
    attractors = report["attractor_candidates"]
    lines = [
        "# Faceted clinical field atlas — exploratory exercise",
        "",
        "## The visual contract",
        "",
        "- Every panel uses one frozen PCA fitted once from 16 contemporaneous tumour and host variables together.",
        "- Cancer, clinical phase, model, treatment, history, and time do not "
        "enter the display axes; they select the trajectories being averaged.",
        "- Every spontaneous-atlas arrow is a matched approximately six-month "
        "state displacement. Each trajectory contributes only its earliest "
        "eligible landmark per clinical phase.",
        "- Thin arrows expose the individual generated trajectories. Bold arrows "
        "are local averages and are drawn only near observed starting states.",
        "- The treatment panel uses matched Sonnet Phase-1 generated "
        "counterfactual futures. Exact six-month positions are linearly "
        "interpolated in the frozen PCA display.",
        "",
        "## What is an attractor here?",
        "",
        "The green endpoint density is an empirical six-month destination basin. "
        "It is labelled a remission-attractor candidate only when at least 60% "
        "of trajectories move closer to the endpoint-density peak, the median "
        "distance decreases, and median PC1 severity decreases. This is a "
        "finite-horizon operational definition—not evidence of an asymptotic "
        "or mechanistic attractor.",
        "",
    ]
    for cancer in CANCERS:
        item = attractors[cancer]
        lines.append(
            f"- {cancer}: **{item['label']}**; "
            f"{item['fraction_moving_closer_to_peak']:.0%} move closer; "
            f"median PC1 shift {item['median_pc1_shift']:+.2f}."
        )
    lines.extend(
        [
            "",
            "## Data status",
            "",
            "All atlas trajectories are stored model-generated trajectories, "
            "not human patients. The intervention comparison is generated "
            "counterfactual evidence and is not a calibrated clinical causal "
            "effect. These figures test the visual and analytical grammar.",
            "",
            "## Important limitations",
            "",
            "- Six-month spontaneous-state pairs require an observable "
            "nonterminal state near the horizon, so terminal exits are not "
            "represented as map destinations.",
            "- Each cancer was designed with five trajectories per model. Model "
            "facets pool three cancers (13–15 eligible trajectories/model; "
            "exact support is printed in each panel).",
            "- PCA is a lossy display. History may alter the conditional field without being a coordinate.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    atlas_tables = load_tables()
    atlas_raw = enrich_snapshots(atlas_tables)
    display, atlas = fit_display_map(atlas_raw)
    atlas = add_phase_labels(atlas)
    pairs = fixed_horizon_pairs(atlas)
    landmarks = first_landmark_per_phase(pairs)

    phase1_tables = load_derivation(PHASE1_ROOT)
    phase1 = transform_to_map(enrich_snapshots(phase1_tables), display)
    treatment_vectors = phase1_exact_horizon_vectors(phase1, phase1_tables.events)
    limits = map_limits(atlas, phase1)

    figure_1, cancer_phase = cancer_phase_figure(landmarks, atlas, limits)
    figure_2, models = model_figure(landmarks, atlas, limits)
    figure_3, treatment = treatment_figure(treatment_vectors, atlas, limits)
    figure_4, attractors = attractor_figure(landmarks, atlas, limits)

    report: dict[str, object] = {
        "contract": {
            "horizon_months": HORIZON_MONTHS,
            "spontaneous_horizon_tolerance_months": (TOLERANCE_DAYS / MONTH_DAYS),
            "state_space": "one frozen atlas PCA",
            "field_quantity": "finite-horizon state displacement",
            "spontaneous_unit": "one earliest landmark/trajectory/phase",
            "treatment_source": str(PHASE1_ROOT.relative_to(ROOT)),
        },
        "display_map": display_map_report(display),
        "figures": {
            "cancer_by_phase": figure_1,
            "model_specific": figure_2,
            "treatment_specific": figure_3,
            "attractor_candidates": figure_4,
        },
        "cancer_by_phase": cancer_phase,
        "models": models,
        "treatment": treatment,
        "attractor_candidates": attractors,
        "counts": {
            "atlas_snapshots": int(len(atlas)),
            "atlas_trajectories": int(atlas["_gid"].nunique()),
            "eligible_horizon_pairs": int(len(pairs)),
            "equal_weight_landmarks": int(len(landmarks)),
            "treatment_vectors": int(len(treatment_vectors)),
        },
    }
    (OUT / "readout.json").write_text(json.dumps(report, indent=2) + "\n")
    write_readme(report)

    for files in report["figures"].values():
        print(f"[done] {files['png']}")
    print(f"[done] {OUT.relative_to(ROOT) / 'README.md'}")
    print(f"[done] {OUT.relative_to(ROOT) / 'readout.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
