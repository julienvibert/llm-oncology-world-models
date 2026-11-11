#!/usr/bin/env python3
"""Build official individual trajectory state-space display panels.

This read-only figure generator promotes the current individual-trajectory
display grammar to the paper figure surface. It consumes the frozen state map
from `outputs/paper_state_map/` and stored derivation tables from an accepted
atlas run; it does not refit the PCA and does not call any model.

Outputs:
  outputs/paper_individual_state_trajectory_views/
    fig_complete_trajectory_full_view.{png,pdf,svg}
    fig_dot_vector_heatmaps.{png,pdf,svg}
    readout.json
    legend.md

Run:
  uv run python scripts/build_paper_individual_state_trajectory_views.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "outputs" / "paper_state_map"
DEFAULT_OUT = ROOT / "outputs" / "paper_individual_state_trajectory_views"
STATE_PATH = STATE_DIR / "state_projection.csv.gz"
LOADINGS_PATH = STATE_DIR / "feature_loadings.csv"
DEFAULT_TRAJECTORY_ID = "IDC_tumor_only__kimi_k3_off__f02"

INK = "#172033"
MUTED = "#64748b"
GRID = "#dbe3ef"
PAPER = "#fbfaf7"

FEATURES = [
    "tumour_hepatic",
    "tumour_thoracic",
    "tumour_nodal",
    "tumour_osseous",
    "tumour_marrow",
    "tumour_cns",
    "tumour_peritoneal",
    "tumour_soft_tissue_cutaneous",
    "tumour_other_anatomical",
    "tumour_extent",
    "host_ecog",
    "host_symptoms",
    "host_renal",
    "host_hepatic",
    "host_cardiac",
    "host_marrow_reserve",
    "host_pulmonary",
    "host_relative_weight_loss",
    "host_active_ae_max_grade",
    "host_active_ae_count",
]

FEATURE_SHORT_LABELS = {
    "tumour_hepatic": "liver",
    "tumour_thoracic": "thoracic",
    "tumour_nodal": "nodes",
    "tumour_osseous": "bone",
    "tumour_marrow": "marrow dz",
    "tumour_cns": "CNS",
    "tumour_peritoneal": "peritoneal",
    "tumour_soft_tissue_cutaneous": "skin/soft",
    "tumour_other_anatomical": "other",
    "tumour_extent": "extent",
    "host_ecog": "ECOG",
    "host_symptoms": "symptoms",
    "host_renal": "renal fn",
    "host_hepatic": "hepatic fn",
    "host_cardiac": "cardiac fn",
    "host_marrow_reserve": "marrow res",
    "host_pulmonary": "pulmonary fn",
    "host_relative_weight_loss": "weight loss",
    "host_active_ae_max_grade": "AE grade",
    "host_active_ae_count": "AE count",
}

EVENT_SYMBOLS = {
    "response": ("v", "#2563eb", "response"),
    "progression": ("^", "#dc2626", "progression"),
}

TREATMENT_PALETTE = [
    "#0f766e",
    "#7c2d12",
    "#1d4ed8",
    "#9333ea",
    "#be123c",
    "#15803d",
    "#a16207",
    "#0369a1",
    "#c2410c",
    "#6d28d9",
    "#475569",
]

DELTA_CMAP = ListedColormap(["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"])
DELTA_NORM = BoundaryNorm([-1.01, -0.3, -0.1, 0.1, 0.3, 1.01], DELTA_CMAP.N)
DELTA_TICKS = [-0.655, -0.2, 0.0, 0.2, 0.655]
DELTA_TICKLABELS = ["<= -2", "-1", "0", "+1", ">= +2"]
RESIDENCE_SIZE_SCALE = 14.0
RESIDENCE_SIZE_CAP_MONTHS = 24.0


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def compact(text: str, *, width: int) -> str:
    return textwrap.shorten(str(text).replace("_", " "), width=width, placeholder="...")


def load_projection() -> pd.DataFrame:
    return pd.read_csv(_require(STATE_PATH)).sort_values(
        ["trajectory_id", "snapshot_index"]
    )


def load_feature_labels() -> dict[str, str]:
    labels = pd.read_csv(_require(LOADINGS_PATH))
    return dict(zip(labels["feature"], labels["label"], strict=True))


def run_event_path(run_id: str) -> Path:
    return ROOT / "runs" / run_id / "derivation" / "event_rows.csv"


def run_baseline_path(run_id: str) -> Path:
    return ROOT / "runs" / run_id / "derivation" / "baseline_rows.csv"


def load_events(run_id: str) -> pd.DataFrame:
    path = _require(run_event_path(run_id))
    cols = [
        "trajectory_id",
        "event_index",
        "event_type",
        "days_from_diagnosis",
        "regimen",
        "line",
        "gene",
        "alteration",
        "ctcae_v5_grade",
    ]
    header = pd.read_csv(path, nrows=0).columns
    events = pd.read_csv(path, usecols=[col for col in cols if col in header])
    events["event_index"] = pd.to_numeric(events["event_index"], errors="coerce")
    events["days_from_diagnosis"] = pd.to_numeric(
        events["days_from_diagnosis"], errors="coerce"
    )
    return events


def load_baselines(run_id: str) -> pd.DataFrame:
    path = _require(run_baseline_path(run_id))
    cols = [
        "trajectory_id",
        "tumor_oncotree_code",
        "model_short",
        "histology",
        "primary_driver",
        "age_at_diagnosis",
        "sex",
    ]
    header = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(path, usecols=[col for col in cols if col in header])


def event_priority(event_type: object) -> int:
    order = {"response": 0, "progression": 1}
    return order.get(str(event_type), 99)


def attach_events_and_residence(path: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    trajectory_id = str(path["trajectory_id"].iloc[0])
    ev = events[events["trajectory_id"].eq(trajectory_id)].copy()
    ev["snapshot_index"] = pd.to_numeric(ev["event_index"], errors="coerce").astype("Int64")
    display_events = (
        ev[ev["event_type"].isin(EVENT_SYMBOLS)]
        .assign(_priority=lambda frame: frame["event_type"].map(event_priority))
        .sort_values(["snapshot_index", "_priority", "days_from_diagnosis"])
        .drop_duplicates("snapshot_index", keep="first")
        [["snapshot_index", "event_type"]]
    )
    merged = path.copy()
    merged["snapshot_index"] = merged["snapshot_index"].astype("Int64")
    merged = merged.merge(display_events, on="snapshot_index", how="left")
    merged.loc[merged["snapshot_index"].eq(0), "event_type"] = "baseline"

    death_rows = ev[ev["event_type"].eq("death")].copy()
    merged["death_recorded"] = not death_rows.empty
    merged["death_day"] = (
        float(death_rows["days_from_diagnosis"].min()) if not death_rows.empty else np.nan
    )
    merged = merged.sort_values("snapshot_index").reset_index(drop=True)
    next_day = merged["days_from_diagnosis"].shift(-1)
    next_day.iat[-1] = (
        float(death_rows["days_from_diagnosis"].min())
        if not death_rows.empty
        else float(merged["days_from_diagnosis"].iloc[-1])
    )
    residence_days = next_day.sub(merged["days_from_diagnosis"]).clip(lower=0)
    merged["residence_months"] = residence_days.div(30.44).clip(lower=0.15, upper=12.0)
    return merged


def trajectory_title(path: pd.DataFrame, baselines: pd.DataFrame, events: pd.DataFrame) -> str:
    trajectory_id = str(path["trajectory_id"].iloc[0])
    base = baselines[baselines["trajectory_id"].eq(trajectory_id)]
    first = path.iloc[0]
    bits = [str(first["tumor_oncotree_code"]), str(first["model_short"])]
    if not base.empty:
        row = base.iloc[0]
        if pd.notna(row.get("age_at_diagnosis")) or pd.notna(row.get("sex")):
            age = int(row["age_at_diagnosis"]) if pd.notna(row.get("age_at_diagnosis")) else "?"
            bits.append(f"{age}y {row.get('sex', '')}".strip())
        driver = str(row.get("primary_driver", "")).strip()
        if driver and driver.lower() != "nan":
            bits.append(driver)
    starts = events[
        events["trajectory_id"].eq(trajectory_id)
        & events["event_type"].eq("systemic_treatment_start")
    ].sort_values("days_from_diagnosis")
    if not starts.empty:
        bits.append(f"1L {compact(starts.iloc[0].get('regimen', ''), width=34)}")
    return " | ".join(bits)


def clean_regimen_label(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "no systemic treatment"
    return text.replace("_", " ")


def treatment_events(events: pd.DataFrame, trajectory_id: str) -> pd.DataFrame:
    tx = events[
        events["trajectory_id"].eq(trajectory_id)
        & events["event_type"].isin(["systemic_treatment_start", "systemic_treatment_switch"])
    ].copy()
    if tx.empty:
        return tx
    tx["regimen_clean"] = tx["regimen"].map(clean_regimen_label)
    return tx.sort_values("days_from_diagnosis")


def treatment_at_day(tx: pd.DataFrame, day: float) -> str:
    if tx.empty:
        return "no systemic treatment"
    active = tx[tx["days_from_diagnosis"].le(day)].sort_values("days_from_diagnosis")
    if active.empty:
        return "no systemic treatment"
    return clean_regimen_label(active.iloc[-1].get("regimen_clean", active.iloc[-1].get("regimen")))


def treatment_color_map(treatments: list[str]) -> dict[str, str]:
    ordered: list[str] = []
    for treatment in treatments:
        if treatment not in ordered:
            ordered.append(treatment)
    return {
        treatment: (
            "#94a3b8"
            if treatment == "no systemic treatment"
            else TREATMENT_PALETTE[index % len(TREATMENT_PALETTE)]
        )
        for index, treatment in enumerate(ordered)
    }


def plot_background(
    ax: plt.Axes,
    projection: pd.DataFrame,
    limits: tuple[float, float, float, float],
) -> None:
    sample = projection.sample(n=min(12000, len(projection)), random_state=13)
    ax.hexbin(
        sample["pc1"],
        sample["pc2"],
        gridsize=42,
        extent=limits,
        cmap="Greys",
        mincnt=1,
        linewidths=0,
        alpha=0.14,
        zorder=0,
    )


def tidy_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.7)
    ax.tick_params(labelsize=7, colors=INK)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")


def residence_to_marker_area(residence_months: np.ndarray | float) -> np.ndarray:
    residence = np.asarray(residence_months, dtype=float)
    return RESIDENCE_SIZE_SCALE * np.clip(
        residence,
        0.25,
        RESIDENCE_SIZE_CAP_MONTHS,
    )


def aspect_matched_limits(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    box_ratio: float,
    pad_fraction: float = 0.12,
) -> tuple[float, float, float, float]:
    """Return data limits whose x/y ratio matches the drawn axes box."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    xmin = float(np.nanmin(x_values))
    xmax = float(np.nanmax(x_values))
    ymin = float(np.nanmin(y_values))
    ymax = float(np.nanmax(y_values))
    x_range = max(xmax - xmin, 0.40)
    y_range = max(ymax - ymin, 0.40)
    x_range *= 1.0 + pad_fraction
    y_range *= 1.0 + pad_fraction
    current_ratio = x_range / y_range
    if current_ratio > box_ratio:
        y_range = x_range / box_ratio
    else:
        x_range = y_range * box_ratio
    x_mid = (xmin + xmax) / 2.0
    y_mid = (ymin + ymax) / 2.0
    return (
        x_mid - x_range / 2.0,
        x_mid + x_range / 2.0,
        y_mid - y_range / 2.0,
        y_mid + y_range / 2.0,
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=450 if ext == "png" else None, bbox_inches="tight")
        paths[ext] = str(path.relative_to(ROOT))
    plt.close(fig)
    return paths


def draw_missing_hatches(
    ax: plt.Axes,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    missing: np.ndarray,
) -> None:
    for row in range(missing.shape[0]):
        for col in range(missing.shape[1]):
            if not bool(missing[row, col]):
                continue
            ax.add_patch(
                Rectangle(
                    (float(x_edges[col]), float(y_edges[row])),
                    float(x_edges[col + 1] - x_edges[col]),
                    float(y_edges[row + 1] - y_edges[row]),
                    facecolor="none",
                    edgecolor="#94a3b8",
                    linewidth=0.0,
                    hatch="////",
                    zorder=5,
                )
            )


def plot_complete_trajectory(
    projection: pd.DataFrame,
    path: pd.DataFrame,
    baselines: pd.DataFrame,
    events: pd.DataFrame,
    out_dir: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    trajectory_id = str(path["trajectory_id"].iloc[0])
    times = path["days_from_diagnosis"].to_numpy(float) / 30.44
    norm = Normalize(vmin=float(np.nanmin(times)), vmax=float(np.nanmax(times)) or 1.0)
    points = path[["pc1", "pc2"]].to_numpy(float)
    residence = path["residence_months"].to_numpy(float)
    sizes = residence_to_marker_area(residence)
    tx = treatment_events(events, trajectory_id)
    segment_treatments = [
        treatment_at_day(tx, float(day))
        for day in path["days_from_diagnosis"].iloc[:-1].to_numpy(float)
    ]
    treatment_colors = treatment_color_map(segment_treatments or ["no systemic treatment"])
    segment_colors = [treatment_colors[treatment] for treatment in segment_treatments]
    axes_box = (0.705 * 9.2, 0.660 * 6.8)
    limits = aspect_matched_limits(
        path["pc1"],
        path["pc2"],
        box_ratio=axes_box[0] / axes_box[1],
    )

    fig = plt.figure(figsize=(9.2, 6.8), facecolor=PAPER)
    ax = fig.add_axes([0.075, 0.185, 0.705, 0.660])
    fig.text(
        0.075,
        0.965,
        "A complete individual trajectory in state space",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.075,
        0.925,
        compact(trajectory_title(path, baselines, events), width=100),
        ha="left",
        va="top",
        fontsize=8.2,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.888,
        "Dots are states; arrows are movements; line colour is active treatment.",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
    )
    plot_background(ax, projection, limits)

    if len(points) > 1:
        segments = np.stack([points[:-1], points[1:]], axis=1)
        ax.add_collection(
            LineCollection(segments, colors=segment_colors, linewidth=2.2, alpha=0.78, zorder=3)
        )
        delta = points[1:] - points[:-1]
        moved = np.sqrt((delta * delta).sum(axis=1)) > 0.012
        if moved.any():
            ax.quiver(
                points[:-1, 0][moved],
                points[:-1, 1][moved],
                delta[:, 0][moved],
                delta[:, 1][moved],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=[segment_colors[index] for index, keep in enumerate(moved) if keep],
                width=0.0052,
                headwidth=5.0,
                headlength=6.0,
                headaxislength=5.2,
                alpha=0.88,
                zorder=4,
            )

    scatter = ax.scatter(
        path["pc1"],
        path["pc2"],
        c=times,
        cmap="viridis",
        norm=norm,
        s=sizes,
        edgecolor="white",
        linewidth=0.65,
        zorder=7,
    )
    ax.scatter(
        path["pc1"].iloc[0],
        path["pc2"].iloc[0],
        marker="D",
        s=88,
        facecolor="#f8fafc",
        edgecolor=INK,
        linewidth=1.25,
        zorder=9,
    )
    death_day = pd.to_numeric(path["death_day"], errors="coerce").dropna()
    latest_day = float(path["days_from_diagnosis"].max())
    if bool(path["death_recorded"].any()) and not death_day.empty and 0 <= float(death_day.min()) - latest_day <= 90:
        ax.scatter(
            path["pc1"].iloc[-1],
            path["pc2"].iloc[-1],
            marker="X",
            s=170,
            facecolor="#111827",
            edgecolor="white",
            linewidth=1.2,
            zorder=10,
        )

    for event_type, group in path[path["event_type"].isin(EVENT_SYMBOLS)].groupby("event_type"):
        marker, color, _label = EVENT_SYMBOLS[event_type]
        ax.scatter(
            group["pc1"],
            group["pc2"],
            marker=marker,
            s=62,
            facecolor=color,
            edgecolor="white",
            linewidth=0.65,
            alpha=0.92,
            zorder=11,
        )

    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PC1: global tumour/host burden", fontsize=8.2, color=INK)
    ax.set_ylabel("PC2: marrow/host vs visceral pattern", fontsize=8.2, color=INK)
    tidy_axis(ax)

    cax = fig.add_axes([0.19, 0.080, 0.43, 0.024])
    cbar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=6.2)
    cbar.set_label("months since diagnosis", fontsize=6.8)

    legend_x = 0.805
    treatment_handles = [
        Line2D([0], [0], color=color, linewidth=2.8, label=compact(treatment, width=34))
        for treatment, color in treatment_colors.items()
    ]
    if treatment_handles:
        fig.add_artist(
            fig.legend(
                handles=treatment_handles,
                title="active treatment",
                loc="upper left",
                bbox_to_anchor=(legend_x, 0.875),
                frameon=False,
                fontsize=6.4,
                title_fontsize=7.0,
                borderaxespad=0,
            )
        )

    residence_values = [1, 3, 6, 12, 24]
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markeredgecolor="#334155",
            markerfacecolor="#cbd5e1",
            markersize=float(np.sqrt(residence_to_marker_area(value))),
            label=f"{value} mo",
        )
        for value in residence_values
    ]
    fig.add_artist(
        fig.legend(
            handles=size_handles,
            title="time in state",
            loc="upper left",
            bbox_to_anchor=(legend_x, 0.520),
            frameon=False,
            fontsize=6.4,
            title_fontsize=7.0,
            borderaxespad=0,
        )
    )

    event_handles = [
        Line2D([0], [0], marker="D", color="none", markeredgecolor=INK, markerfacecolor="#f8fafc", markersize=7, label="start"),
        Line2D([0], [0], marker="X", color="none", markeredgecolor="white", markerfacecolor="#111827", markersize=8, label="death"),
        Line2D([0], [0], marker="v", color="none", markeredgecolor="white", markerfacecolor="#2563eb", markersize=7, label="response"),
        Line2D([0], [0], marker="^", color="none", markeredgecolor="white", markerfacecolor="#dc2626", markersize=7, label="progression"),
    ]
    fig.legend(
        handles=event_handles,
        title="events",
        loc="upper left",
        bbox_to_anchor=(legend_x, 0.315),
        frameon=False,
        fontsize=6.4,
        title_fontsize=7.0,
        borderaxespad=0,
    )

    paths = save_figure(fig, out_dir, "fig_complete_trajectory_full_view")
    treatment_payload = {treatment: color for treatment, color in treatment_colors.items()}
    return paths, treatment_payload


def time_edges_months(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if len(times) == 0:
        return np.array([0.0, 1.0])
    if len(times) == 1:
        width = 1.0
    else:
        diffs = np.diff(times)
        positive = diffs[diffs > 0]
        width = float(np.median(positive)) if len(positive) else 1.0
    edges = np.r_[times, times[-1] + width]
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = edges[index - 1] + max(width * 0.04, 0.04)
    return edges


def plot_dot_vector_heatmaps(
    path: pd.DataFrame,
    baselines: pd.DataFrame,
    events: pd.DataFrame,
    feature_labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    view = path.sort_values("snapshot_index").reset_index(drop=True)
    state_matrix = view[FEATURES].to_numpy(float).T
    delta_matrix = (view[FEATURES].shift(-1) - view[FEATURES]).to_numpy(float).T
    state_missing = np.isnan(state_matrix)
    delta_missing = np.isnan(delta_matrix)
    state_display = np.ma.masked_invalid(state_matrix)
    delta_display = np.ma.masked_invalid(delta_matrix)
    times = view["days_from_diagnosis"].to_numpy(float) / 30.44
    x_edges = time_edges_months(times)
    y_edges = np.arange(len(FEATURES) + 1)
    row_labels = [
        FEATURE_SHORT_LABELS.get(feature, feature_labels.get(feature, feature))
        for feature in FEATURES
    ]

    fig = plt.figure(figsize=(8.6, 6.5), facecolor=PAPER)
    gs = fig.add_gridspec(
        2,
        1,
        left=0.145,
        right=0.915,
        bottom=0.105,
        top=0.835,
        hspace=0.28,
    )
    fig.text(
        0.145,
        0.965,
        "Dot and vector coordinates behind one trajectory",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.145,
        0.925,
        compact(trajectory_title(path, baselines, events), width=105),
        ha="left",
        va="top",
        fontsize=8.1,
        color=MUTED,
    )

    ax_state = fig.add_subplot(gs[0, 0])
    im_state = ax_state.pcolormesh(
        x_edges,
        y_edges,
        state_display,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        shading="flat",
    )
    draw_missing_hatches(ax_state, x_edges, y_edges, state_missing)
    ax_state.set_title(
        "a Dot: 20-dimensional state at each recorded event",
        loc="left",
        fontsize=9.2,
        fontweight="bold",
        color=INK,
    )

    ax_delta = fig.add_subplot(gs[1, 0])
    im_delta = ax_delta.pcolormesh(
        x_edges,
        y_edges,
        delta_display,
        cmap=DELTA_CMAP,
        norm=DELTA_NORM,
        shading="flat",
    )
    draw_missing_hatches(ax_delta, x_edges, y_edges, delta_missing)
    ax_delta.set_title(
        "b Vector: 20-dimensional change to the next event",
        loc="left",
        fontsize=9.2,
        fontweight="bold",
        color=INK,
    )

    for ax in [ax_state, ax_delta]:
        ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
        ax.set_ylim(len(FEATURES), 0)
        ax.set_yticks(np.arange(len(FEATURES)) + 0.5)
        ax.set_yticklabels(row_labels, fontsize=6.2)
        ax.axhline(9.5, color="white", linewidth=1.2)
        ax.set_xlabel("months since diagnosis", fontsize=7.0, color=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar_state = fig.colorbar(im_state, ax=ax_state, fraction=0.026, pad=0.012)
    cbar_state.ax.tick_params(labelsize=6.0)
    cbar_state.set_label("state [0,1]", fontsize=6.2)
    cbar_delta = fig.colorbar(im_delta, ax=ax_delta, fraction=0.026, pad=0.012)
    cbar_delta.ax.tick_params(labelsize=6.0)
    cbar_delta.set_ticks(DELTA_TICKS)
    cbar_delta.set_ticklabels(DELTA_TICKLABELS)
    cbar_delta.set_label("delta, ordinal steps", fontsize=6.2)
    fig.text(
        0.145,
        0.032,
        "Columns use absolute months since diagnosis. Irregular column widths preserve irregular event timing; the final vector column is undefined.",
        fontsize=6.5,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    return save_figure(fig, out_dir, "fig_dot_vector_heatmaps")


def write_legend(out_dir: Path) -> None:
    text = """# Individual trajectory state-space views

Official read-only display panels built from the frozen state-map projection.

- `fig_complete_trajectory_full_view`: one complete trajectory in PC1/PC2, drawn with equal data aspect so displayed directions match true PCA directions. Dots are living states, arrows connect consecutive states, dot colour is months since diagnosis, dot area is proportional to residence time until the next state and capped at 24 months for display, line colour encodes active treatment, blue downward triangles mark response, red upward triangles mark progression, a diamond marks the first state, and a black X marks terminal death after the last measured living state.
- `fig_dot_vector_heatmaps`: the same trajectory in the original 20-coordinate state contract. The top panel shows the dot, meaning the state value at each recorded event. Missing state values, if present in the source table, are hatched rather than rendered as zero. The bottom panel shows the vector, meaning the coordinate-wise change to the next event, on a discrete diverging scale labelled in approximate ordinal steps. The x-axis is absolute time in months, with irregular column widths preserving irregular event timing.

The PCA map is frozen and treatment/outcome variables are overlays, not PCA inputs.
"""
    (out_dir / "legend.md").write_text(text)


def build(out_dir: Path, trajectory_id: str) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    projection = load_projection()
    feature_labels = load_feature_labels()
    selected = projection[projection["trajectory_id"].eq(trajectory_id)].copy()
    if selected.empty:
        raise ValueError(f"trajectory_id not found in state projection: {trajectory_id}")
    run_id = str(selected["_run"].iloc[0])
    events = load_events(run_id)
    baselines = load_baselines(run_id)
    path = attach_events_and_residence(selected, events)

    complete_paths, treatment_colors = plot_complete_trajectory(
        projection, path, baselines, events, out_dir
    )
    heatmap_paths = plot_dot_vector_heatmaps(
        path, baselines, events, feature_labels, out_dir
    )
    write_legend(out_dir)

    readout = {
        "git_commit": git_commit(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "output_dir": str(out_dir.relative_to(ROOT)),
        "inputs": {
            "state_projection": {
                "path": str(STATE_PATH.relative_to(ROOT)),
                "sha256": sha256(STATE_PATH),
            },
            "feature_loadings": {
                "path": str(LOADINGS_PATH.relative_to(ROOT)),
                "sha256": sha256(LOADINGS_PATH),
            },
            "event_rows": {
                "path": str(run_event_path(run_id).relative_to(ROOT)),
                "sha256": sha256(run_event_path(run_id)),
            },
            "baseline_rows": {
                "path": str(run_baseline_path(run_id).relative_to(ROOT)),
                "sha256": sha256(run_baseline_path(run_id)),
            },
        },
        "trajectory": {
            "trajectory_id": trajectory_id,
            "run_id": run_id,
            "n_living_states": int(len(path)),
            "duration_months": float(path["days_from_diagnosis"].max() / 30.44),
            "death_recorded": bool(path["death_recorded"].any()),
        },
        "figures": {
            "complete_trajectory_full_view": complete_paths,
            "dot_vector_heatmaps": heatmap_paths,
        },
        "treatment_colors": treatment_colors,
        "plot_contract": {
            "state_space": "frozen outputs/paper_state_map PC1/PC2 projection",
            "state_space_aspect": "equal data aspect; display window expanded to match axes box ratio",
            "dot_color": "months since diagnosis",
            "dot_size": (
                "marker area proportional to residence time until the next state; "
                f"display clipped to 0.25-{RESIDENCE_SIZE_CAP_MONTHS:.0f} months"
            ),
            "line_color": "active systemic treatment at segment start",
            "event_symbols": {"response": "blue downward triangle", "progression": "red upward triangle"},
            "heatmap_x_axis": "absolute months since diagnosis; variable-width event columns",
            "heatmap_missing_values": "NaN source values are hatched and not rendered as zero",
            "vector_heatmap_scale": {
                "type": "discrete diverging ordinal-step bins",
                "bounds": [-1.01, -0.3, -0.1, 0.1, 0.3, 1.01],
                "labels": DELTA_TICKLABELS,
            },
            "heatmap_rows": FEATURES,
        },
    }
    (out_dir / "readout.json").write_text(json.dumps(readout, indent=2) + "\n")
    return readout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--trajectory-id", default=DEFAULT_TRAJECTORY_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(build(args.out_dir, args.trajectory_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
