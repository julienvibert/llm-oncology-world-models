#!/usr/bin/env python3
"""Build support-aware group vector-field views on the frozen state map.

This read-only figure generator turns a cohort of stored trajectories into an
Eulerian state-space field. Grid cells are drawn in the displayed PC1/PC2 plane
for readability, but each arrow is computed by averaging the underlying
20-coordinate finite-horizon movement and only then projecting that movement
back through the frozen PCA basis. Death before the horizon is an explicit EXIT
statistic, not a destination in the map.

Outputs:
  outputs/paper_group_state_vector_field_views/
    fig_field_<time>mo.{png,pdf,svg}
    fig_cell_heatmaps_<time>mo.{png,pdf,svg}
    cell_readouts.csv
    cell_state_medians.csv
    cell_vector_means.csv
    grid_edges.csv
    readout.json
    legend.md

Run:
  uv run python scripts/build_paper_group_state_vector_field_views.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from sklearn.cluster import AgglomerativeClustering, KMeans  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "outputs" / "paper_state_map"
DEFAULT_OUT = ROOT / "outputs" / "paper_group_state_vector_field_views"
STATE_PATH = STATE_DIR / "state_projection.csv.gz"
LOADINGS_PATH = STATE_DIR / "feature_loadings.csv"

INK = "#172033"
MUTED = "#64748b"
GRID = "#dbe3ef"
PAPER = "#fbfaf7"
BLUE = "#0f3a5f"
GREY_ARROW = "#64748b"

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

DELTA_CMAP = ListedColormap(["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"])
DELTA_NORM = BoundaryNorm([-0.45, -0.18, -0.04, 0.04, 0.18, 0.45], DELTA_CMAP.N)
DELTA_TICKS = [-0.315, -0.11, 0.0, 0.11, 0.315]
DELTA_TICKLABELS = ["large fall", "small fall", "0", "small rise", "large rise"]
SUPPORT_SIZE_SCALE = 22.0
SUPPORT_SIZE_CAP = 30
ARROW_DISPLAY_SCALE = 1.8
KDE_MIN_POINTS_DEFAULT = 12
CLUSTER_COUNT_DEFAULT = 4


@dataclass(frozen=True)
class FixedClusterModel:
    centroids: np.ndarray
    table: pd.DataFrame


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


def compact(text: object, *, width: int) -> str:
    return textwrap.shorten(str(text).replace("_", " "), width=width, placeholder="...")


def parse_months(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one frame month is required")
    return values


def event_rows_path(run_id: str) -> Path:
    return ROOT / "runs" / run_id / "derivation" / "event_rows.csv"


def load_projection() -> pd.DataFrame:
    return pd.read_csv(_require(STATE_PATH)).sort_values(
        ["trajectory_id", "snapshot_index"]
    )


def load_loadings() -> pd.DataFrame:
    return pd.read_csv(_require(LOADINGS_PATH))


def load_events(run_ids: list[str], trajectory_ids: set[str]) -> pd.DataFrame:
    cols = [
        "trajectory_id",
        "event_type",
        "days_from_diagnosis",
    ]
    frames: list[pd.DataFrame] = []
    for run_id in sorted(run_ids):
        path = _require(event_rows_path(run_id))
        header = pd.read_csv(path, nrows=0).columns
        usecols = [col for col in cols if col in header]
        events = pd.read_csv(path, usecols=usecols)
        events = events[events["trajectory_id"].isin(trajectory_ids)].copy()
        events["_run"] = run_id
        frames.append(events)
    if not frames:
        return pd.DataFrame(columns=[*cols, "_run"])
    out = pd.concat(frames, ignore_index=True)
    out["days_from_diagnosis"] = pd.to_numeric(
        out["days_from_diagnosis"], errors="coerce"
    )
    return out


def pca_delta_basis(loadings: pd.DataFrame) -> np.ndarray:
    indexed = loadings.set_index("feature").loc[FEATURES]
    return indexed[["pc1_coefficient", "pc2_coefficient"]].to_numpy(float)


def support_to_marker_area(n: int | float) -> float:
    return SUPPORT_SIZE_SCALE * float(np.clip(n, 1, SUPPORT_SIZE_CAP))


def aspect_matched_limits(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    box_ratio: float,
    pad_fraction: float = 0.10,
    quantile_low: float | None = None,
    quantile_high: float | None = None,
) -> tuple[float, float, float, float]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if quantile_low is not None and quantile_high is not None:
        xmin = float(np.nanquantile(x_values, quantile_low))
        xmax = float(np.nanquantile(x_values, quantile_high))
        ymin = float(np.nanquantile(y_values, quantile_low))
        ymax = float(np.nanquantile(y_values, quantile_high))
    else:
        xmin = float(np.nanmin(x_values))
        xmax = float(np.nanmax(x_values))
        ymin = float(np.nanmin(y_values))
        ymax = float(np.nanmax(y_values))
    x_range = max(xmax - xmin, 0.70)
    y_range = max(ymax - ymin, 0.70)
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


def tidy_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.72)
    ax.tick_params(labelsize=7, colors=INK)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")


def regular_grid_edges(
    limits: tuple[float, float, float, float], grid_size: int
) -> tuple[np.ndarray, np.ndarray]:
    x_edges = np.linspace(limits[0], limits[1], grid_size + 1)
    y_edges = np.linspace(limits[2], limits[3], grid_size + 1)
    return x_edges, y_edges


def quantile_axis_edges(
    values: pd.Series | np.ndarray,
    *,
    lower: float,
    upper: float,
    grid_size: int,
) -> np.ndarray:
    in_window = np.asarray(values, dtype=float)
    in_window = in_window[(in_window >= lower) & (in_window <= upper)]
    if len(in_window) < grid_size * 2:
        return np.linspace(lower, upper, grid_size + 1)
    edges = np.nanquantile(in_window, np.linspace(0, 1, grid_size + 1))
    edges[0] = lower
    edges[-1] = upper
    if np.any(np.diff(edges) <= 0):
        return np.linspace(lower, upper, grid_size + 1)
    return edges


def adaptive_grid_edges(
    cohort: pd.DataFrame,
    limits: tuple[float, float, float, float],
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_edges = quantile_axis_edges(
        cohort["pc1"],
        lower=limits[0],
        upper=limits[1],
        grid_size=grid_size,
    )
    y_edges = quantile_axis_edges(
        cohort["pc2"],
        lower=limits[2],
        upper=limits[3],
        grid_size=grid_size,
    )
    return x_edges, y_edges


def make_grid_edges(
    cohort: pd.DataFrame,
    limits: tuple[float, float, float, float],
    *,
    grid_size: int,
    grid_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if grid_mode == "regular":
        return regular_grid_edges(limits, grid_size)
    if grid_mode == "quantile":
        return adaptive_grid_edges(cohort, limits, grid_size)
    raise ValueError(f"unknown grid mode: {grid_mode}")


def grid_edge_table(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    grid_mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, value in enumerate(x_edges):
        rows.append({"axis": "pc1", "edge_index": index, "value": float(value), "grid_mode": grid_mode})
    for index, value in enumerate(y_edges):
        rows.append({"axis": "pc2", "edge_index": index, "value": float(value), "grid_mode": grid_mode})
    return pd.DataFrame(rows)


def assign_grid(
    x: float,
    y: float,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> tuple[int, int] | None:
    if x < x_edges[0] or x > x_edges[-1] or y < y_edges[0] or y > y_edges[-1]:
        return None
    col = int(np.searchsorted(x_edges, x, side="right") - 1)
    row = int(np.searchsorted(y_edges, y, side="right") - 1)
    col = min(max(col, 0), len(x_edges) - 2)
    row = min(max(row, 0), len(y_edges) - 2)
    return row, col


def cell_center(row: int, col: int, x_edges: np.ndarray, y_edges: np.ndarray) -> tuple[float, float]:
    return (
        float((x_edges[col] + x_edges[col + 1]) / 2.0),
        float((y_edges[row] + y_edges[row + 1]) / 2.0),
    )


def ordered_cluster_labels(labels: np.ndarray, members: pd.DataFrame) -> np.ndarray:
    label_order = (
        pd.DataFrame(
            {
                "raw_label": labels,
                "start_pc1": members["start_pc1"].to_numpy(float),
                "start_pc2": members["start_pc2"].to_numpy(float),
            }
        )
        .groupby("raw_label", sort=True)[["start_pc1", "start_pc2"]]
        .median()
        .sort_values(["start_pc1", "start_pc2"])
        .reset_index()
    )
    remap = {int(row.raw_label): index for index, row in enumerate(label_order.itertuples())}
    return np.array([remap[int(label)] for label in labels], dtype=int)


def assign_timepoint_clusters(
    members: pd.DataFrame,
    *,
    cluster_count: int,
    min_support: int,
) -> pd.DataFrame:
    if members.empty:
        return members
    out = members.copy()
    max_clusters_by_support = max(1, len(out) // max(min_support, 1))
    n_clusters = min(cluster_count, max_clusters_by_support, len(out))
    if n_clusters <= 1:
        labels = np.zeros(len(out), dtype=int)
    else:
        state_matrix = out[[f"start_{feature}" for feature in FEATURES]].to_numpy(float)
        labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(state_matrix)
    ordered = ordered_cluster_labels(labels, out)
    out["grid_row"] = ordered
    out["grid_col"] = 0
    out["cell_id"] = [f"k{label:02d}" for label in ordered]
    return out


def fixed_cluster_training_members(
    paths: dict[str, pd.DataFrame],
    deaths: dict[str, float],
    *,
    frame_months: list[float],
    horizon_day: float,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for frame_month in frame_months:
        members = frame_members(
            paths,
            deaths,
            start_day=frame_month * 30.44,
            horizon_day=horizon_day,
            x_edges=x_edges,
            y_edges=y_edges,
        )
        if members.empty:
            continue
        members = members.copy()
        members["fit_frame_month"] = frame_month
        frames.append(members)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fit_fixed_cluster_model(
    paths: dict[str, pd.DataFrame],
    deaths: dict[str, float],
    *,
    frame_months: list[float],
    horizon_day: float,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    cluster_count: int,
    min_support: int,
) -> FixedClusterModel:
    training = fixed_cluster_training_members(
        paths,
        deaths,
        frame_months=frame_months,
        horizon_day=horizon_day,
        x_edges=x_edges,
        y_edges=y_edges,
    )
    if training.empty:
        raise ValueError("fixed_cluster mode found no living states in the selected frame months")
    max_clusters_by_support = max(1, len(training) // max(min_support, 1))
    n_clusters = min(cluster_count, max_clusters_by_support, len(training))
    state_matrix = training[[f"start_{feature}" for feature in FEATURES]].to_numpy(float)
    if n_clusters <= 1:
        raw_labels = np.zeros(len(training), dtype=int)
        raw_centroids = state_matrix.mean(axis=0, keepdims=True)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=0, n_init=50)
        raw_labels = model.fit_predict(state_matrix)
        raw_centroids = model.cluster_centers_
    ordered = ordered_cluster_labels(raw_labels, training)
    raw_to_ordered = {
        int(raw): int(np.median(ordered[raw_labels == raw]))
        for raw in np.unique(raw_labels)
    }
    centroids = np.zeros_like(raw_centroids)
    rows: list[dict[str, object]] = []
    for raw, ordered_label in sorted(raw_to_ordered.items(), key=lambda item: item[1]):
        cluster = training[ordered == ordered_label]
        centroid = raw_centroids[raw]
        centroids[ordered_label, :] = centroid
        payload: dict[str, object] = {
            "cluster_id": f"k{ordered_label:02d}",
            "ordered_label": ordered_label,
            "n_fit_states": int(len(cluster)),
            "n_fit_trajectories": int(cluster["trajectory_id"].nunique()),
            "median_fit_pc1": float(cluster["start_pc1"].median()),
            "median_fit_pc2": float(cluster["start_pc2"].median()),
        }
        for feature, value in zip(FEATURES, centroid, strict=True):
            payload[f"centroid_{feature}"] = float(value)
        rows.append(payload)
    return FixedClusterModel(centroids=centroids, table=pd.DataFrame(rows))


def assign_fixed_clusters(members: pd.DataFrame, fixed_cluster_model: FixedClusterModel) -> pd.DataFrame:
    if members.empty:
        return members
    out = members.copy()
    state_matrix = out[[f"start_{feature}" for feature in FEATURES]].to_numpy(float)
    distances = ((state_matrix[:, np.newaxis, :] - fixed_cluster_model.centroids[np.newaxis, :, :]) ** 2).sum(axis=2)
    labels = distances.argmin(axis=1).astype(int)
    out["grid_row"] = labels
    out["grid_col"] = 0
    out["cell_id"] = [f"k{label:02d}" for label in labels]
    return out


def iter_aggregate_cells(
    members: pd.DataFrame,
    *,
    aggregation_mode: str,
    grid_size: int,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> list[tuple[str, int, int, float, float, pd.DataFrame]]:
    cells: list[tuple[str, int, int, float, float, pd.DataFrame]] = []
    if aggregation_mode == "grid":
        for row in range(grid_size):
            for col in range(grid_size):
                cell = members[members["grid_row"].eq(row) & members["grid_col"].eq(col)]
                if cell.empty:
                    continue
                center_x, center_y = cell_center(row, col, x_edges, y_edges)
                cells.append((f"r{row}c{col}", row, col, center_x, center_y, cell))
        return cells
    if aggregation_mode in {"cluster", "fixed_cluster"}:
        for cell_id, cell in members.groupby("cell_id", sort=True):
            row = int(cell["grid_row"].iloc[0])
            col = int(cell["grid_col"].iloc[0])
            center_x = float(cell["start_pc1"].median())
            center_y = float(cell["start_pc2"].median())
            cells.append((str(cell_id), row, col, center_x, center_y, cell))
        return cells
    raise ValueError(f"unknown aggregation mode: {aggregation_mode}")


def death_days(events: pd.DataFrame) -> dict[str, float]:
    death = events[events["event_type"].eq("death")].copy()
    if death.empty:
        return {}
    return (
        death.groupby("trajectory_id")["days_from_diagnosis"]
        .min()
        .dropna()
        .astype(float)
        .to_dict()
    )


def build_paths(projection: pd.DataFrame) -> dict[str, pd.DataFrame]:
    paths: dict[str, pd.DataFrame] = {}
    for trajectory_id, group in projection.groupby("trajectory_id", sort=False):
        paths[str(trajectory_id)] = group.sort_values("days_from_diagnosis").reset_index(drop=True)
    return paths


def state_at_day(
    path: pd.DataFrame,
    day: float,
    death_day: float | None,
) -> pd.Series | None:
    if death_day is not None and day >= death_day:
        return None
    last_state_day = float(path["days_from_diagnosis"].max())
    if day > last_state_day and death_day is None:
        return None
    eligible = path[path["days_from_diagnosis"].le(day)]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def last_living_before(path: pd.DataFrame, day: float) -> pd.Series | None:
    eligible = path[path["days_from_diagnosis"].le(day)]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def frame_members(
    paths: dict[str, pd.DataFrame],
    deaths: dict[str, float],
    *,
    start_day: float,
    horizon_day: float,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    end_day = start_day + horizon_day
    for trajectory_id, path in paths.items():
        death_day = deaths.get(trajectory_id)
        start = state_at_day(path, start_day, death_day)
        if start is None:
            continue
        assigned = assign_grid(float(start["pc1"]), float(start["pc2"]), x_edges, y_edges)
        if assigned is None:
            continue
        row, col = assigned
        outcome = "survived"
        end = None
        if death_day is not None and death_day <= end_day:
            outcome = "exit_death"
        else:
            end = state_at_day(path, end_day, death_day)
            if end is None:
                outcome = "censored"
        payload: dict[str, object] = {
            "trajectory_id": trajectory_id,
            "grid_row": row,
            "grid_col": col,
            "cell_id": f"r{row}c{col}",
            "start_pc1": float(start["pc1"]),
            "start_pc2": float(start["pc2"]),
            "outcome": outcome,
            "death_day": float(death_day) if death_day is not None else np.nan,
        }
        for feature in FEATURES:
            payload[f"start_{feature}"] = float(start[feature])
        if end is not None:
            payload["end_pc1"] = float(end["pc1"])
            payload["end_pc2"] = float(end["pc2"])
            for feature in FEATURES:
                payload[f"delta_{feature}"] = float(end[feature] - start[feature])
        rows.append(payload)
    return pd.DataFrame(rows)


def aggregate_frame(
    members: pd.DataFrame,
    *,
    frame_month: float,
    horizon_months: float,
    basis: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    grid_size: int,
    aggregation_mode: str,
    cluster_count: int,
    fixed_cluster_model: FixedClusterModel | None,
    min_support: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    readouts: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    vector_rows: list[dict[str, object]] = []
    individual_rows: list[dict[str, object]] = []
    if members.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    if aggregation_mode == "cluster":
        members = assign_timepoint_clusters(
            members,
            cluster_count=cluster_count,
            min_support=min_support,
        )
    elif aggregation_mode == "fixed_cluster":
        if fixed_cluster_model is None:
            raise ValueError("fixed_cluster mode requires a fitted cluster model")
        members = assign_fixed_clusters(members, fixed_cluster_model)

    for cell_id, row, col, center_x, center_y, cell in iter_aggregate_cells(
        members,
        aggregation_mode=aggregation_mode,
        grid_size=grid_size,
        x_edges=x_edges,
        y_edges=y_edges,
    ):
        n_start = int(len(cell))
        n_exit = int(cell["outcome"].eq("exit_death").sum())
        n_censored = int(cell["outcome"].eq("censored").sum())
        survivors = cell[cell["outcome"].eq("survived")].copy()
        n_survivor = int(len(survivors))
        state_vector = cell[[f"start_{feature}" for feature in FEATURES]].median().to_numpy(float)
        delta_pc = np.array([np.nan, np.nan])
        vector_values = np.full(len(FEATURES), np.nan)
        if n_survivor:
            delta_matrix = survivors[[f"delta_{feature}" for feature in FEATURES]].to_numpy(float)
            vector_values = np.nanmean(delta_matrix, axis=0)
            delta_pc = vector_values @ basis
            individual_pc = delta_matrix @ basis
            for index, (_idx, survivor) in enumerate(survivors.iterrows()):
                individual_rows.append(
                    {
                        "frame_month": frame_month,
                        "horizon_months": horizon_months,
                        "aggregation_mode": aggregation_mode,
                        "cell_id": cell_id,
                        "grid_row": row,
                        "grid_col": col,
                        "center_pc1": center_x,
                        "center_pc2": center_y,
                        "start_pc1": float(survivor["start_pc1"]),
                        "start_pc2": float(survivor["start_pc2"]),
                        "trajectory_id": survivor["trajectory_id"],
                        "delta_pc1": float(individual_pc[index, 0]),
                        "delta_pc2": float(individual_pc[index, 1]),
                    }
                )
        support_ok = n_start >= min_support
        survival_probability = 1.0 - (n_exit / n_start)
        readouts.append(
            {
                "frame_month": frame_month,
                "horizon_months": horizon_months,
                "aggregation_mode": aggregation_mode,
                "cell_id": cell_id,
                "grid_row": row,
                "grid_col": col,
                "center_pc1": center_x,
                "center_pc2": center_y,
                "n_start": n_start,
                "n_survivor": n_survivor,
                "n_exit_death": n_exit,
                "n_censored": n_censored,
                "support_ok": bool(support_ok),
                "survival_probability": survival_probability,
                "exit_probability": 1.0 - survival_probability,
                "mean_delta_pc1": float(delta_pc[0]) if n_survivor else np.nan,
                "mean_delta_pc2": float(delta_pc[1]) if n_survivor else np.nan,
            }
        )
        state_payload = {
            "frame_month": frame_month,
            "horizon_months": horizon_months,
            "aggregation_mode": aggregation_mode,
            "cell_id": cell_id,
            "grid_row": row,
            "grid_col": col,
            "n_start": n_start,
        }
        vector_payload = dict(state_payload)
        for feature, value in zip(FEATURES, state_vector, strict=True):
            state_payload[feature] = float(value)
        for feature, value in zip(FEATURES, vector_values, strict=True):
            vector_payload[feature] = float(value) if np.isfinite(value) else np.nan
        state_rows.append(state_payload)
        vector_rows.append(vector_payload)

    return (
        pd.DataFrame(readouts),
        pd.DataFrame(state_rows),
        pd.DataFrame(vector_rows),
        pd.DataFrame(individual_rows),
    )


def cumulative_deaths(
    paths: dict[str, pd.DataFrame],
    deaths: dict[str, float],
    frame_day: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trajectory_id, death_day in deaths.items():
        if death_day > frame_day:
            continue
        path = paths.get(trajectory_id)
        if path is None:
            continue
        state = last_living_before(path, death_day)
        if state is None:
            continue
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "death_day": death_day,
                "pc1": float(state["pc1"]),
                "pc2": float(state["pc2"]),
            }
        )
    return pd.DataFrame(rows)


def plot_same_time_kde(
    ax: plt.Axes,
    members: pd.DataFrame,
    limits: tuple[float, float, float, float],
    *,
    min_points: int,
) -> bool:
    if len(members) < min_points:
        return False
    points = members[["start_pc1", "start_pc2"]].dropna().to_numpy(float)
    if len(points) < min_points:
        return False
    if np.nanstd(points[:, 0]) <= 1e-9 or np.nanstd(points[:, 1]) <= 1e-9:
        return False
    try:
        estimator = gaussian_kde(points.T, bw_method=0.48)
    except (ValueError, np.linalg.LinAlgError):
        return False
    x_grid = np.linspace(limits[0], limits[1], 130)
    y_grid = np.linspace(limits[2], limits[3], 130)
    xx, yy = np.meshgrid(x_grid, y_grid)
    density = estimator(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    finite = density[np.isfinite(density) & (density > 0)]
    if len(finite) < min_points:
        return False
    levels = np.unique(np.quantile(finite, [0.68, 0.78, 0.86, 0.93, 0.98, 1.00]))
    if len(levels) < 2:
        return False
    ax.contourf(
        xx,
        yy,
        density,
        levels=levels,
        colors=["#eef3f8", "#dfe8f1", "#cedde9", "#bdcfdf", "#aabfd4"],
        alpha=0.46,
        antialiased=True,
        zorder=0,
    )
    ax.contour(
        xx,
        yy,
        density,
        levels=levels[1:-1],
        colors="#8ea7bd",
        linewidths=0.45,
        alpha=0.30,
        zorder=0.5,
    )
    return True


def plot_same_time_background(
    ax: plt.Axes,
    members: pd.DataFrame,
    limits: tuple[float, float, float, float],
    *,
    background_mode: str,
    kde_min_points: int,
) -> bool:
    kde_drawn = False
    if background_mode in {"kde", "dots_kde"}:
        kde_drawn = plot_same_time_kde(ax, members, limits, min_points=kde_min_points)
    if not members.empty and background_mode in {"dots", "dots_kde"}:
        ax.scatter(
            members["start_pc1"],
            members["start_pc2"],
            s=15,
            facecolor="#94a3b8",
            edgecolor="white",
            linewidth=0.35,
            alpha=0.42,
            zorder=2,
        )
    return kde_drawn


def support_layer_label(background_mode: str, kde_drawn: bool) -> str:
    if background_mode == "dots":
        return "dots"
    if background_mode == "kde":
        return "KDE" if kde_drawn else "KDE skipped"
    if background_mode == "dots_kde":
        return "dots + KDE" if kde_drawn else "dots (KDE skipped)"
    raise ValueError(f"unknown background mode: {background_mode}")


def plot_field_frame(
    cohort: pd.DataFrame,
    members: pd.DataFrame,
    readout: pd.DataFrame,
    individual_arrows: pd.DataFrame,
    death_marks: pd.DataFrame,
    *,
    tumor_code: str,
    model_label: str,
    frame_month: float,
    horizon_months: float,
    limits: tuple[float, float, float, float],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    aggregation_mode: str,
    grid_mode: str,
    cluster_count: int,
    background_mode: str,
    kde_min_points: int,
    min_support: int,
    out_dir: Path,
) -> dict[str, str]:
    fig = plt.figure(figsize=(8.2, 6.4), facecolor=PAPER)
    ax = fig.add_axes([0.085, 0.155, 0.690, 0.700])
    title_prefix = {
        "grid": "vector",
        "cluster": "profile-cluster",
        "fixed_cluster": "fixed-profile",
    }[aggregation_mode]
    title = f"{tumor_code} cohort {title_prefix} field at {frame_month:g} months"
    fig.text(0.085, 0.965, title, ha="left", va="top", fontsize=13.2, fontweight="bold", color=INK)
    fig.text(
        0.085,
        0.925,
        f"{model_label} | {horizon_months:g}-month finite-horizon movement | {len(cohort['trajectory_id'].unique())} trajectories",
        ha="left",
        va="top",
        fontsize=7.8,
        color=MUTED,
    )

    kde_drawn = plot_same_time_background(
        ax,
        members,
        limits,
        background_mode=background_mode,
        kde_min_points=kde_min_points,
    )
    if aggregation_mode == "grid":
        for x in x_edges:
            ax.axvline(float(x), color="#cbd5e1", linewidth=0.7, alpha=0.86, zorder=1)
        for y in y_edges:
            ax.axhline(float(y), color="#cbd5e1", linewidth=0.7, alpha=0.86, zorder=1)

    norm_surv = Normalize(vmin=0, vmax=1)
    cmap_surv = plt.get_cmap("RdYlGn")

    if not individual_arrows.empty:
        ax.quiver(
            individual_arrows["start_pc1"],
            individual_arrows["start_pc2"],
            individual_arrows["delta_pc1"] * ARROW_DISPLAY_SCALE,
            individual_arrows["delta_pc2"] * ARROW_DISPLAY_SCALE,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.0022,
            headwidth=3.0,
            headlength=4.0,
            headaxislength=3.2,
            color=GREY_ARROW,
            alpha=0.16,
            zorder=3,
        )

    visible = readout[readout["n_start"].ge(min_support)].copy()
    if not visible.empty:
        scatter = ax.scatter(
            visible["center_pc1"],
            visible["center_pc2"],
            s=[support_to_marker_area(value) for value in visible["n_start"]],
            c=visible["survival_probability"],
            cmap=cmap_surv,
            norm=norm_surv,
            edgecolor=INK,
            linewidth=0.72,
            alpha=0.96,
            zorder=6,
        )
        vectors = visible[visible["n_survivor"].gt(0)].copy()
        if not vectors.empty:
            ax.quiver(
                vectors["center_pc1"],
                vectors["center_pc2"],
                vectors["mean_delta_pc1"] * ARROW_DISPLAY_SCALE,
                vectors["mean_delta_pc2"] * ARROW_DISPLAY_SCALE,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.0070,
                headwidth=5.3,
                headlength=6.3,
                headaxislength=5.1,
                color=BLUE,
                alpha=0.92,
                zorder=7,
            )
    else:
        scatter = None

    if not death_marks.empty:
        ax.scatter(
            death_marks["pc1"],
            death_marks["pc2"],
            marker="x",
            s=30,
            linewidth=1.1,
            color="#111827",
            alpha=0.35,
            zorder=8,
        )

    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PC1: global tumour/host burden", fontsize=8.0, color=INK)
    ax.set_ylabel("PC2: marrow/host vs visceral pattern", fontsize=8.0, color=INK)
    tidy_axis(ax)

    if scatter is not None:
        cax = fig.add_axes([0.813, 0.280, 0.020, 0.175])
        cbar = fig.colorbar(scatter, cax=cax, orientation="vertical")
        cbar.ax.tick_params(labelsize=6.1)
        cbar.set_label(f"survival over next {horizon_months:g} mo", fontsize=6.6)

    support_values = [min_support, 10, 20, 30]
    support_values = sorted(set(value for value in support_values if value >= min_support))
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markeredgecolor=INK,
            markerfacecolor="#d9f99d",
            markersize=float(np.sqrt(support_to_marker_area(value))),
            label=f"n={value}",
        )
        for value in support_values
    ]
    arrow_handle = Line2D([0], [0], color=BLUE, linewidth=3.0, label="mean 20D movement")
    grey_handle = Line2D([0], [0], color=GREY_ARROW, linewidth=1.4, alpha=0.38, label="individual survivor movement")
    density_handle = Patch(facecolor="#bdcfdf", alpha=0.50, edgecolor="none", label="same-time density")
    dot_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markeredgecolor="white",
        markerfacecolor="#94a3b8",
        alpha=0.55,
        markersize=5,
        label="same-time living state",
    )
    death_handle = Line2D([0], [0], marker="x", color="#111827", linestyle="None", markersize=6, label="cumulative death")
    primary_handles = [arrow_handle, grey_handle]
    if kde_drawn:
        primary_handles.append(density_handle)
    primary_handles.extend([dot_handle, death_handle])

    fig.legend(
        handles=primary_handles,
        loc="upper left",
        bbox_to_anchor=(0.805, 0.840),
        frameon=False,
        fontsize=6.7,
        borderaxespad=0,
    )
    fig.legend(
        handles=size_handles,
        title="living support",
        loc="upper left",
        bbox_to_anchor=(0.805, 0.690),
        frameon=False,
        fontsize=6.7,
        title_fontsize=7.0,
        borderaxespad=0,
    )

    n_living = int(readout["n_start"].sum()) if not readout.empty else 0
    n_cells = int(readout["n_start"].ge(min_support).sum()) if not readout.empty else 0
    n_deaths = int(len(death_marks))
    fig.text(
        0.805,
        0.235,
        (
            f"visible living starts: {n_living}\n"
            f"supported aggregates: {n_cells}\n"
            f"cumulative deaths: {n_deaths}\n"
            f"aggregation: {aggregation_mode}\n"
            + (
                f"clusters target: {cluster_count}\n"
                if aggregation_mode in {"cluster", "fixed_cluster"}
                else f"grid: {grid_mode}\n"
            )
            + f"support layer: {support_layer_label(background_mode, kde_drawn)}\n"
            f"arrow scale: {ARROW_DISPLAY_SCALE:.1f}x"
        ),
        ha="left",
        va="top",
        fontsize=7.0,
        color=MUTED,
        linespacing=1.45,
    )
    fig.text(
        0.085,
        0.026,
        (
            {
                "cluster": "Aggregate dots are dynamic same-time 20D patient-profile clusters. ",
                "fixed_cluster": "Aggregate dots are fixed 20D patient-profile archetypes fitted once across selected frames. ",
                "grid": "Grid is drawn in the PC1/PC2 display plane. ",
            }[aggregation_mode]
        )
        + "Background support is same-time only; KDE is only a visual density layer. Faint arrows start at survivor dots; thick arrows average original 20-coordinate changes among survivors, then project that mean vector back to the frozen PCA plane. Death before the horizon is EXIT, not a destination.",
        ha="left",
        va="bottom",
        fontsize=6.3,
        color=MUTED,
    )

    stem = f"fig_field_{int(round(frame_month)):03d}mo"
    return save_figure(fig, out_dir, stem)


def plot_heatmap_frame(
    state_rows: pd.DataFrame,
    vector_rows: pd.DataFrame,
    readout: pd.DataFrame,
    *,
    tumor_code: str,
    frame_month: float,
    horizon_months: float,
    aggregation_mode: str,
    min_support: int,
    max_cells: int,
    out_dir: Path,
) -> dict[str, str] | None:
    if state_rows.empty:
        return None
    ordered = (
        readout[readout["n_start"].ge(min_support)]
        .sort_values(["n_start", "n_exit_death"], ascending=[False, False])
        .head(max_cells)
    )
    if ordered.empty:
        return None
    cell_ids = ordered["cell_id"].tolist()
    states = state_rows[state_rows["cell_id"].isin(cell_ids)].set_index("cell_id").loc[cell_ids]
    vectors = vector_rows[vector_rows["cell_id"].isin(cell_ids)].set_index("cell_id").loc[cell_ids]
    state_matrix = states[FEATURES].to_numpy(float)
    vector_matrix = vectors[FEATURES].to_numpy(float)
    row_labels = [
        f"{row.cell_id}  n={int(row.n_start)} exit={row.exit_probability:.0%}"
        for row in ordered.itertuples()
    ]
    col_labels = [FEATURE_SHORT_LABELS.get(feature, feature) for feature in FEATURES]
    aggregate_label = {
        "grid": "supported PCA grid cells",
        "cluster": "dynamic same-time 20D profile clusters",
        "fixed_cluster": "stable 20D profile archetypes occupied at this timepoint",
    }[aggregation_mode]

    fig = plt.figure(figsize=(9.3, 6.2), facecolor=PAPER)
    gs = fig.add_gridspec(
        2,
        1,
        left=0.185,
        right=0.910,
        bottom=0.245,
        top=0.825,
        hspace=0.34,
    )
    fig.text(
        0.185,
        0.955,
        f"{tumor_code} aggregate dot and vector coordinates at {frame_month:g} months",
        ha="left",
        va="top",
        fontsize=12.6,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.185,
        0.912,
        f"Rows are {aggregate_label}; vector is the mean change over the next {horizon_months:g} months among survivors.",
        ha="left",
        va="top",
        fontsize=7.3,
        color=MUTED,
    )

    ax_state = fig.add_subplot(gs[0, 0])
    im_state = ax_state.imshow(state_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax_state.set_title(
        "a Aggregate dot: median 20-dimensional state",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        color=INK,
    )

    ax_vector = fig.add_subplot(gs[1, 0])
    im_vector = ax_vector.imshow(
        vector_matrix,
        aspect="auto",
        cmap=DELTA_CMAP,
        norm=DELTA_NORM,
    )
    ax_vector.set_title(
        "b Aggregate vector: mean 20-dimensional change",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        color=INK,
    )

    for ax in [ax_state, ax_vector]:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=6.4)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=5.7, rotation=48, ha="right")
        ax.axvline(9.5, color="white", linewidth=1.2)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar_state = fig.colorbar(im_state, ax=ax_state, fraction=0.026, pad=0.012)
    cbar_state.ax.tick_params(labelsize=6.0)
    cbar_state.set_label("state [0,1]", fontsize=6.2)
    cbar_vector = fig.colorbar(im_vector, ax=ax_vector, fraction=0.026, pad=0.012)
    cbar_vector.set_ticks(DELTA_TICKS)
    cbar_vector.set_ticklabels(DELTA_TICKLABELS)
    cbar_vector.ax.tick_params(labelsize=6.0)
    cbar_vector.set_label("delta", fontsize=6.2)

    fig.text(
        0.185,
        0.050,
        "These heatmaps explain the aggregate map symbols: the dot is the median clinical state of patients starting in each aggregate, and the vector is the feature-wise mean movement that generated the field arrow.",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=MUTED,
    )
    stem = f"fig_cell_heatmaps_{int(round(frame_month)):03d}mo"
    return save_figure(fig, out_dir, stem)


def write_legend(out_dir: Path) -> None:
    text = f"""# Group state-vector field views

Official read-only cohort-field display built from the frozen
`outputs/paper_state_map` PCA projection.

- `fig_field_<time>mo`: support-aware vector field for one tumour-code cohort at a
  fixed time since diagnosis. By default, the grid uses PC1/PC2 quantile edges
  inside a robust display envelope so crowded regions receive finer visual
  resolution; `--grid-mode regular` gives the transparent equal-width version.
  `--aggregation-mode cluster` instead refits patient-profile clusters at each
  frame in the original 20-dimensional state space, then plots each aggregate at
  the median PC1/PC2 position of its members. Cluster labels are frame-specific
  profile identifiers, not longitudinally persistent compartments.
  `--aggregation-mode fixed_cluster` fits profile archetypes once on all selected
  living frame states in the original 20-dimensional state space, then reuses the
  same nearest-centroid assignment rule at every frame. Fixed-cluster labels are
  stable across timepoints.
  Background support is frame-specific: `--background-mode dots_kde` overlays a
  faint same-time KDE where there are enough living states and keeps the actual
  same-time dots visible; `dots` and `kde` are available for sensitivity renders.
  The KDE never defines support. Each aggregate dot is placed at the grid-cell
  centre in grid mode and at the median member position in cluster modes, dot
  area is proportional to living support, dot colour is survival probability over
  the next horizon, thin grey arrows start at each survivor's same-time state and
  show their individual finite-horizon movements, while the thick arrow is the
  mean original 20-coordinate movement projected back to PC1/PC2 and drawn from
  the aggregate dot. Deaths before the horizon enter as EXIT and are not given a
  fake map destination. Crosses mark cumulative deaths at the last living PCA
  state.
- `fig_cell_heatmaps_<time>mo`: coordinate explanation for supported cells in the
  corresponding field panel. The top heatmap is the median 20-coordinate state of
  patients starting in the cell. The bottom heatmap is the mean 20-coordinate
  movement over the horizon among survivors.

Default display scale for arrows: {ARROW_DISPLAY_SCALE:.1f}x, stated on the
panel. The scale affects only visibility; cell readouts store unscaled PC
movement.
"""
    (out_dir / "legend.md").write_text(text)


def build(
    out_dir: Path,
    *,
    tumor_code: str,
    model_short: str | None,
    grid_size: int,
    grid_mode: str,
    aggregation_mode: str,
    cluster_count: int,
    background_mode: str,
    kde_min_points: int,
    frame_months: list[float],
    horizon_months: float,
    min_support: int,
    max_heatmap_cells: int,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    projection = load_projection()
    loadings = load_loadings()
    basis = pca_delta_basis(loadings)

    cohort = projection[projection["tumor_oncotree_code"].eq(tumor_code)].copy()
    if model_short:
        cohort = cohort[cohort["model_short"].eq(model_short)].copy()
    if cohort.empty:
        raise ValueError(f"no states found for tumor_code={tumor_code!r}, model_short={model_short!r}")
    trajectory_ids = set(cohort["trajectory_id"].astype(str).unique())
    run_ids = sorted(cohort["_run"].astype(str).unique())
    events = load_events(run_ids, trajectory_ids)
    deaths = death_days(events)
    paths = build_paths(cohort)
    model_label = model_short if model_short else "six-model pooled atlas"

    axes_box = (0.690 * 8.2, 0.700 * 6.4)
    limits = aspect_matched_limits(
        cohort["pc1"],
        cohort["pc2"],
        box_ratio=axes_box[0] / axes_box[1],
        quantile_low=0.02,
        quantile_high=0.98,
    )
    x_edges, y_edges = make_grid_edges(
        cohort,
        limits,
        grid_size=grid_size,
        grid_mode=grid_mode,
    )
    horizon_day = horizon_months * 30.44
    fixed_cluster_model: FixedClusterModel | None = None
    if aggregation_mode == "fixed_cluster":
        fixed_cluster_model = fit_fixed_cluster_model(
            paths,
            deaths,
            frame_months=frame_months,
            horizon_day=horizon_day,
            x_edges=x_edges,
            y_edges=y_edges,
            cluster_count=cluster_count,
            min_support=min_support,
        )

    all_readouts: list[pd.DataFrame] = []
    all_states: list[pd.DataFrame] = []
    all_vectors: list[pd.DataFrame] = []
    all_individual: list[pd.DataFrame] = []
    figures: dict[str, object] = {}

    for frame_month in frame_months:
        frame_day = frame_month * 30.44
        members = frame_members(
            paths,
            deaths,
            start_day=frame_day,
            horizon_day=horizon_day,
            x_edges=x_edges,
            y_edges=y_edges,
        )
        readout, state_rows, vector_rows, individual_arrows = aggregate_frame(
            members,
            frame_month=frame_month,
            horizon_months=horizon_months,
            basis=basis,
            x_edges=x_edges,
            y_edges=y_edges,
            grid_size=grid_size,
            aggregation_mode=aggregation_mode,
            cluster_count=cluster_count,
            fixed_cluster_model=fixed_cluster_model,
            min_support=min_support,
        )
        death_marks = cumulative_deaths(paths, deaths, frame_day)
        field_paths = plot_field_frame(
            cohort,
            members,
            readout,
            individual_arrows,
            death_marks,
            tumor_code=tumor_code,
            model_label=model_label,
            frame_month=frame_month,
            horizon_months=horizon_months,
            limits=limits,
            x_edges=x_edges,
            y_edges=y_edges,
            aggregation_mode=aggregation_mode,
            grid_mode=grid_mode,
            cluster_count=cluster_count,
            background_mode=background_mode,
            kde_min_points=kde_min_points,
            min_support=min_support,
            out_dir=out_dir,
        )
        heatmap_paths = plot_heatmap_frame(
            state_rows,
            vector_rows,
            readout,
            tumor_code=tumor_code,
            frame_month=frame_month,
            horizon_months=horizon_months,
            aggregation_mode=aggregation_mode,
            min_support=min_support,
            max_cells=max_heatmap_cells,
            out_dir=out_dir,
        )
        frame_key = f"{int(round(frame_month)):03d}mo"
        figures[frame_key] = {"field": field_paths, "heatmaps": heatmap_paths}
        if not readout.empty:
            all_readouts.append(readout)
        if not state_rows.empty:
            all_states.append(state_rows)
        if not vector_rows.empty:
            all_vectors.append(vector_rows)
        if not individual_arrows.empty:
            all_individual.append(individual_arrows)

    readout_table = pd.concat(all_readouts, ignore_index=True) if all_readouts else pd.DataFrame()
    state_table = pd.concat(all_states, ignore_index=True) if all_states else pd.DataFrame()
    vector_table = pd.concat(all_vectors, ignore_index=True) if all_vectors else pd.DataFrame()
    individual_table = pd.concat(all_individual, ignore_index=True) if all_individual else pd.DataFrame()
    edge_table = grid_edge_table(x_edges, y_edges, grid_mode=grid_mode)
    readout_table.to_csv(out_dir / "cell_readouts.csv", index=False)
    state_table.to_csv(out_dir / "cell_state_medians.csv", index=False)
    vector_table.to_csv(out_dir / "cell_vector_means.csv", index=False)
    individual_table.to_csv(out_dir / "individual_projected_movements.csv", index=False)
    edge_table.to_csv(out_dir / "grid_edges.csv", index=False)
    if fixed_cluster_model is not None:
        fixed_cluster_model.table.to_csv(out_dir / "fixed_cluster_centroids.csv", index=False)
    write_legend(out_dir)

    event_inputs = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in [event_rows_path(run_id) for run_id in run_ids]
    }
    manifest = {
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
            "event_rows": event_inputs,
        },
        "cohort": {
            "tumor_code": tumor_code,
            "model_short": model_short,
            "model_label": model_label,
            "n_trajectories": int(len(trajectory_ids)),
            "n_living_snapshots": int(len(cohort)),
            "n_deaths": int(len(deaths)),
        },
        "field_contract": {
            "display_partition": (
                f"{grid_size}x{grid_size} {grid_mode} cells in displayed PC1/PC2"
                if aggregation_mode == "grid"
                else "no PC grid aggregation; grid_edges.csv records the display envelope only"
            ),
            "aggregation": (
                {
                    "grid": "PC1/PC2 display grid",
                    "cluster": f"timepoint-specific agglomerative clustering in original 20D state space, target k={cluster_count}",
                    "fixed_cluster": (
                        f"fixed deterministic k-means archetypes in original 20D state space, "
                        f"fit once across selected frames, target k={cluster_count}"
                    ),
                }[aggregation_mode]
            ),
            "display_envelope": "2nd-98th percentile PC1/PC2 envelope, expanded to match the axes aspect",
            "aggregate_dot_location": (
                {
                    "grid": "grid-cell centre, not fitted centroid",
                    "cluster": "median PC1/PC2 position of same-time cluster members",
                    "fixed_cluster": "median PC1/PC2 position of same-time members assigned to the fixed archetype",
                }[aggregation_mode]
            ),
            "background": f"{background_mode}; KDE minimum n={kde_min_points}; support remains defined by observed living counts",
            "support": "living trajectories with a state at frame time",
            "dot_size": f"area proportional to living support, capped at n={SUPPORT_SIZE_CAP}",
            "dot_color": f"survival probability over next {horizon_months:g} months",
            "arrow": "mean original 20-coordinate movement among survivors, projected through frozen PCA basis",
            "individual_arrows": "survivor-level 20D movement projected through the same PCA basis, drawn from each survivor's same-time start position",
            "exit": "death before horizon; counted in cell statistics and cross marks, not projected as a destination",
            "arrow_display_scale": ARROW_DISPLAY_SCALE,
            "min_support_for_visible_cell": min_support,
            "horizon_months": horizon_months,
            "frame_months": frame_months,
        },
        "figures": figures,
        "tables": {
            "cell_readouts": str((out_dir / "cell_readouts.csv").relative_to(ROOT)),
            "cell_state_medians": str((out_dir / "cell_state_medians.csv").relative_to(ROOT)),
            "cell_vector_means": str((out_dir / "cell_vector_means.csv").relative_to(ROOT)),
            "individual_projected_movements": str((out_dir / "individual_projected_movements.csv").relative_to(ROOT)),
            "grid_edges": str((out_dir / "grid_edges.csv").relative_to(ROOT)),
            "fixed_cluster_centroids": (
                str((out_dir / "fixed_cluster_centroids.csv").relative_to(ROOT))
                if fixed_cluster_model is not None
                else None
            ),
        },
    }
    (out_dir / "readout.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tumor-code", default="IDC")
    parser.add_argument("--model-short", default=None)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--grid-mode", choices=["quantile", "regular"], default="quantile")
    parser.add_argument("--aggregation-mode", choices=["grid", "cluster", "fixed_cluster"], default="grid")
    parser.add_argument("--cluster-count", type=int, default=CLUSTER_COUNT_DEFAULT)
    parser.add_argument("--background-mode", choices=["dots", "kde", "dots_kde"], default="dots_kde")
    parser.add_argument("--kde-min-points", type=int, default=KDE_MIN_POINTS_DEFAULT)
    parser.add_argument("--frame-months", default="0,6,12,18,24,36,48")
    parser.add_argument("--horizon-months", type=float, default=6.0)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--max-heatmap-cells", type=int, default=9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cluster_count < 1:
        raise ValueError("--cluster-count must be at least 1")
    if args.kde_min_points < 2:
        raise ValueError("--kde-min-points must be at least 2")
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    manifest = build(
        out_dir,
        tumor_code=args.tumor_code,
        model_short=args.model_short,
        grid_size=args.grid_size,
        grid_mode=args.grid_mode,
        aggregation_mode=args.aggregation_mode,
        cluster_count=args.cluster_count,
        background_mode=args.background_mode,
        kde_min_points=args.kde_min_points,
        frame_months=parse_months(args.frame_months),
        horizon_months=args.horizon_months,
        min_support=args.min_support,
        max_heatmap_cells=args.max_heatmap_cells,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
