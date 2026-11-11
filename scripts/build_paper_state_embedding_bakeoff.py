#!/usr/bin/env python3
"""Compare linear and nonlinear displays of the frozen paper-v1 clinical state.

This is a visualization bake-off, not a new clinical-state contract. Every
method receives the same fixed-range 20-dimensional state vectors from
``outputs/paper_state_map/state_projection.csv.gz``. Clinical estimands remain
defined in raw features.

The builder:

1. loads all 20 principal components from the frozen contract and independently
   checks the complete serialized basis against a deterministic refit;
2. shows PC1 against every remaining component using identical trajectories;
3. embeds one deterministic, weighted, deduplicated landmark set with PCA,
   UMAP, PHATE and a diffusion map;
4. overlays all PAAD, LUAD and HGSOC paths plus one algorithmically selected
   median-like path per tumour type;
5. reports neighbourhood, distance, trajectory and seed-stability diagnostics.

Nonlinear arrows are never obtained by projecting a raw vector. A displayed
segment always joins the separately embedded observed endpoints of one stored
event interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.linalg import orthogonal_procrustes  # noqa: E402
from scipy.sparse import coo_matrix, diags  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.manifold import trustworthiness  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_paper_state_map import (  # noqa: E402
    FEATURE_LABELS,
    HOST_FEATURES,
    STATE_FEATURES,
)

DEFAULT_STATE_PATH = ROOT / "outputs" / "paper_state_map" / "state_projection.csv.gz"
DEFAULT_CONTRACT_PATH = ROOT / "outputs" / "paper_state_map" / "state_map_contract.json"
DEFAULT_OUT = ROOT / "outputs" / "paper_state_embedding_bakeoff"

RNG_SEED = 20260730
N_LANDMARKS = 6000
N_NEIGHBORS = 30
EVALUATION_N = 1500
TARGET_TUMOURS: tuple[str, ...] = ("PAAD", "LUAD", "HGSOC")
TUMOUR_LABELS: Mapping[str, str] = {
    "PAAD": "pancreatic adenocarcinoma",
    "LUAD": "lung adenocarcinoma",
    "HGSOC": "high-grade serous ovarian cancer",
}
TUMOUR_COLORS: Mapping[str, str] = {
    "PAAD": "#d35f3f",
    "LUAD": "#2878b5",
    "HGSOC": "#7a5195",
}
METHOD_LABELS: Mapping[str, str] = {
    "pca": "PCA",
    "umap": "UMAP",
    "phate": "PHATE",
    "diffusion": "diffusion map",
}

INK = "#162033"
GREY = "#b6bcc6"
PALE_GREY = "#e9ebef"
GRID = "#dfe3e8"


@dataclass(frozen=True)
class FullPCAModel:
    """Complete weighted PCA transformation for the frozen state coordinates."""

    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    feature_multipliers: np.ndarray

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Project encoded state rows onto every principal component."""
        matrix = frame.loc[:, list(STATE_FEATURES)].to_numpy(float)
        transformed = matrix * self.feature_multipliers
        return (transformed - self.mean) @ self.components.T


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_state_projection(path: Path) -> pd.DataFrame:
    """Load and validate the frozen living-state projection."""
    frame = pd.read_csv(path)
    required = {
        "_gid",
        "trajectory_id",
        "model_short",
        "tumor_oncotree_code",
        "snapshot_index",
        "days_from_diagnosis",
        "observation_weight",
        *STATE_FEATURES,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"state projection is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("state projection is empty")
    matrix = frame.loc[:, list(STATE_FEATURES)].to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("state matrix contains non-finite values")
    if matrix.min() < -1e-12 or matrix.max() > 1 + 1e-12:
        raise ValueError("frozen state coordinates must remain in [0, 1]")
    weights = frame["observation_weight"].to_numpy(float)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("observation weights must be non-negative and sum to one")
    return frame.sort_values(["_gid", "snapshot_index"]).reset_index(drop=True)


def fit_full_weighted_pca(
    frame: pd.DataFrame,
    weights: np.ndarray | None = None,
) -> FullPCAModel:
    """Fit all weighted PCs using the exact frozen fixed-range convention."""
    matrix = frame.loc[:, list(STATE_FEATURES)].to_numpy(float)
    w = (
        frame["observation_weight"].to_numpy(float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    if len(w) != len(matrix):
        raise ValueError("one weight is required per state row")
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("PCA weights must be non-negative with positive total")
    w = w / w.sum()
    mean = np.average(matrix, axis=0, weights=w)
    centered = matrix - mean
    covariance = centered.T @ (centered * w[:, np.newaxis])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order].T

    # Match the frozen map's PC1/PC2 conventions. For later components, make the
    # largest absolute loading positive solely to obtain deterministic signs.
    if float(components[0].sum()) < 0:
        components[0] *= -1
    host_indices = [STATE_FEATURES.index(feature) for feature in HOST_FEATURES]
    if float(components[1, host_indices].sum()) < 0:
        components[1] *= -1
    for component_index in range(2, components.shape[0]):
        pivot = int(np.argmax(np.abs(components[component_index])))
        if components[component_index, pivot] < 0:
            components[component_index] *= -1

    total = float(eigenvalues.sum())
    ratio = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    return FullPCAModel(
        mean=mean,
        components=components,
        eigenvalues=eigenvalues,
        explained_variance_ratio=ratio,
        feature_multipliers=np.ones(len(STATE_FEATURES), dtype=float),
    )


def load_serialized_full_pca(contract_path: Path) -> FullPCAModel:
    """Load the complete PCA transform without refitting the frozen atlas."""
    contract = json.loads(contract_path.read_text())
    payload = contract["primary_model"]
    feature_names = tuple(payload["feature_names"])
    if feature_names != STATE_FEATURES:
        raise ValueError("serialized PCA feature order differs from the frozen state")
    components = np.asarray(
        payload["all_components_after_feature_multipliers"],
        dtype=float,
    )
    expected = (len(STATE_FEATURES), len(STATE_FEATURES))
    if components.shape != expected:
        raise ValueError(
            f"frozen contract does not contain a complete {expected[0]}×"
            f"{expected[1]} PCA loading matrix"
        )
    model = FullPCAModel(
        mean=np.asarray(payload["mean_after_feature_multipliers"], dtype=float),
        components=components,
        eigenvalues=np.asarray(payload["eigenvalues"], dtype=float),
        explained_variance_ratio=np.asarray(
            payload["explained_variance_ratio"],
            dtype=float,
        ),
        feature_multipliers=np.asarray(
            payload["feature_multipliers"],
            dtype=float,
        ),
    )
    if not np.allclose(
        model.components @ model.components.T,
        np.eye(len(STATE_FEATURES)),
        atol=1e-10,
    ):
        raise ValueError("serialized PCA components are not orthonormal")
    return model


def verify_serialized_full_pca(
    serialized: FullPCAModel,
    reconstructed: FullPCAModel,
) -> float:
    """Return the maximum parameter difference from an independent refit."""
    differences = (
        np.max(np.abs(serialized.mean - reconstructed.mean)),
        np.max(np.abs(serialized.components - reconstructed.components)),
        np.max(np.abs(serialized.eigenvalues - reconstructed.eigenvalues)),
        np.max(
            np.abs(
                serialized.explained_variance_ratio
                - reconstructed.explained_variance_ratio
            )
        ),
        np.max(
            np.abs(
                serialized.feature_multipliers
                - reconstructed.feature_multipliers
            )
        ),
    )
    maximum = float(max(differences))
    if maximum > 1e-10:
        raise ValueError(
            "serialized complete PCA differs from independent reconstruction "
            f"({maximum:.3g})"
        )
    return maximum


def state_keys(frame: pd.DataFrame) -> pd.Series:
    """Create deterministic exact-state keys from all frozen coordinates."""
    hashes = pd.util.hash_pandas_object(
        frame.loc[:, list(STATE_FEATURES)],
        index=False,
        categorize=False,
    )
    return hashes.map(lambda value: f"{int(value):016x}")


def summarize_trajectories(frame: pd.DataFrame, pc_scores: np.ndarray) -> pd.DataFrame:
    """Create map-independent summaries used only for exemplar selection."""
    work = frame.loc[
        :,
        [
            "_gid",
            "trajectory_id",
            "model_short",
            "tumor_oncotree_code",
            "snapshot_index",
            "days_from_diagnosis",
            *STATE_FEATURES,
        ],
    ].copy()
    work["pc1"] = pc_scores[:, 0]
    rows: list[dict[str, Any]] = []
    for gid, group in work.groupby("_gid", sort=True):
        ordered = group.sort_values("snapshot_index")
        matrix = ordered.loc[:, list(STATE_FEATURES)].to_numpy(float)
        steps = np.linalg.norm(np.diff(matrix, axis=0), axis=1)
        rows.append(
            {
                "_gid": gid,
                "trajectory_id": ordered["trajectory_id"].iloc[0],
                "model_short": ordered["model_short"].iloc[0],
                "tumor_oncotree_code": ordered["tumor_oncotree_code"].iloc[0],
                "n_snapshots": len(ordered),
                "followup_months": (
                    float(
                        ordered["days_from_diagnosis"].iloc[-1]
                        - ordered["days_from_diagnosis"].iloc[0]
                    )
                    / 30.4
                ),
                "raw_path_length": float(steps.sum()),
                "fraction_static_steps": (
                    float(np.mean(steps <= 1e-12)) if len(steps) else 1.0
                ),
                "pc1_range": float(ordered["pc1"].max() - ordered["pc1"].min()),
            }
        )
    return pd.DataFrame(rows)


def select_representative_trajectories(
    summaries: pd.DataFrame,
    tumour_types: Sequence[str] = TARGET_TUMOURS,
) -> pd.DataFrame:
    """Select one median-like trajectory per tumour using a declared rule."""
    metrics = [
        "n_snapshots",
        "followup_months",
        "raw_path_length",
        "fraction_static_steps",
        "pc1_range",
    ]
    selected: list[pd.Series] = []
    for tumour in tumour_types:
        subset = summaries[summaries["tumor_oncotree_code"].eq(tumour)].copy()
        if subset.empty:
            raise ValueError(f"no trajectories available for {tumour}")
        distance = np.zeros(len(subset), dtype=float)
        for metric in metrics:
            values = subset[metric].to_numpy(float)
            median = float(np.median(values))
            q25, q75 = np.quantile(values, [0.25, 0.75])
            scale = float(q75 - q25)
            if scale <= 1e-12:
                scale = float(np.std(values))
            if scale <= 1e-12:
                continue
            distance += ((values - median) / scale) ** 2
        subset["selection_distance"] = distance
        chosen = subset.sort_values(["selection_distance", "_gid"]).iloc[0]
        selected.append(chosen)
    return pd.DataFrame(selected).reset_index(drop=True)


def select_landmarks(
    frame: pd.DataFrame,
    *,
    target_tumours: Sequence[str] = TARGET_TUMOURS,
    n_landmarks: int = N_LANDMARKS,
    seed: int = RNG_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select weighted, deduplicated states while retaining target trajectories."""
    work = frame.copy()
    work["state_key"] = state_keys(work)
    grouped = (
        work.groupby("state_key", sort=True)
        .agg(
            observation_weight=("observation_weight", "sum"),
            first_row=("_gid", "first"),
        )
        .reset_index()
    )
    metadata = work.drop_duplicates("state_key", keep="first").set_index("state_key")
    grouped = grouped.join(
        metadata.loc[:, [*STATE_FEATURES, "model_short", "tumor_oncotree_code"]],
        on="state_key",
    )
    required_keys = set(
        work.loc[
            work["tumor_oncotree_code"].isin(target_tumours),
            "state_key",
        ]
    )
    grouped["required_target_state"] = grouped["state_key"].isin(required_keys)
    required = grouped[grouped["required_target_state"]].copy()
    remaining = grouped[~grouped["required_target_state"]].copy()
    target_n = max(n_landmarks, len(required))
    n_extra = min(max(target_n - len(required), 0), len(remaining))
    if n_extra:
        rng = np.random.default_rng(seed)
        probabilities = remaining["observation_weight"].to_numpy(float)
        probabilities /= probabilities.sum()
        positions = rng.choice(
            len(remaining),
            size=n_extra,
            replace=False,
            p=probabilities,
        )
        chosen = remaining.iloc[np.sort(positions)]
        landmarks = pd.concat([required, chosen], ignore_index=True)
    else:
        landmarks = required.copy()
    landmarks = landmarks.sort_values("state_key").reset_index(drop=True)
    work = work.merge(
        landmarks.loc[:, ["state_key"]].assign(in_landmark_set=True),
        on="state_key",
        how="left",
    )
    work["in_landmark_set"] = work["in_landmark_set"].eq(True)
    return landmarks, work


def fit_umap(matrix: np.ndarray, *, seed: int) -> np.ndarray:
    """Fit a deterministic two-dimensional UMAP display."""
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(N_NEIGHBORS, len(matrix) - 1),
        min_dist=0.12,
        metric="euclidean",
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
    )
    return np.asarray(reducer.fit_transform(matrix), dtype=float)


def fit_phate(matrix: np.ndarray, *, seed: int) -> np.ndarray:
    """Fit PHATE using its landmark implementation for the common state set."""
    import phate

    operator = phate.PHATE(
        n_components=2,
        knn=min(15, len(matrix) - 1),
        decay=40,
        n_landmark=min(1200, len(matrix)),
        t="auto",
        gamma=1,
        n_pca=min(20, matrix.shape[1]),
        # SMACOF is slower than SGD but was the solver used for the figures in
        # the PHATE paper and avoids treating a visibly unconverged SGD layout
        # as the reference display.
        mds_solver="smacof",
        knn_dist="euclidean",
        mds_dist="euclidean",
        mds="metric",
        n_jobs=-1,
        random_state=seed,
        random_landmarking=False,
        verbose=0,
    )
    return np.asarray(operator.fit_transform(matrix), dtype=float)


def fit_diffusion_map(
    matrix: np.ndarray,
    *,
    n_neighbors: int = N_NEIGHBORS,
    alpha: float = 1.0,
    diffusion_time: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an adaptive-kernel Coifman–Lafon diffusion map.

    ``alpha=1`` removes sampling-density effects before the Markov
    normalization. The returned coordinates discard the trivial eigenvector.
    """
    n = len(matrix)
    if n < 5:
        raise ValueError("diffusion map requires at least five states")
    k = min(max(3, n_neighbors), n - 1)
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    neighbors.fit(matrix)
    distances, indices = neighbors.kneighbors(matrix)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    positive = distances[distances > 1e-12]
    fallback = float(np.median(positive)) if positive.size else 1.0
    sigma = distances[:, -1].copy()
    sigma[sigma <= 1e-12] = fallback
    row_index = np.repeat(np.arange(n), k)
    col_index = indices.reshape(-1)
    distance_flat = distances.reshape(-1)
    denominator = sigma[row_index] * sigma[col_index]
    values = np.exp(-(distance_flat**2) / np.maximum(denominator, 1e-12))
    affinity = coo_matrix((values, (row_index, col_index)), shape=(n, n)).tocsr()
    affinity = affinity.maximum(affinity.T)
    affinity.setdiag(1.0)
    density = np.asarray(affinity.sum(axis=1)).ravel()
    density_scale = np.power(np.maximum(density, 1e-12), -alpha)
    corrected = diags(density_scale) @ affinity @ diags(density_scale)
    degree = np.asarray(corrected.sum(axis=1)).ravel()
    inv_sqrt_degree = np.power(np.maximum(degree, 1e-12), -0.5)
    symmetric = diags(inv_sqrt_degree) @ corrected @ diags(inv_sqrt_degree)
    eigenvalues, eigenvectors = eigsh(
        symmetric,
        k=min(4, n - 1),
        which="LA",
        tol=1e-7,
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.real(eigenvalues[order])
    eigenvectors = np.real(eigenvectors[:, order])
    markov_vectors = eigenvectors * inv_sqrt_degree[:, np.newaxis]
    coordinates = markov_vectors[:, 1:3] * (
        eigenvalues[1:3][np.newaxis, :] ** diffusion_time
    )
    return coordinates, eigenvalues


def align_to_reference(
    coordinates: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Orthogonally align arbitrary 2D coordinates to PC1/PC2 for display."""
    source = np.asarray(coordinates, dtype=float)
    target = np.asarray(reference, dtype=float)
    if source.shape != target.shape or source.shape[1] != 2:
        raise ValueError("source and reference must be matching n×2 matrices")
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    source_norm = float(np.linalg.norm(source_centered))
    target_norm = float(np.linalg.norm(target_centered))
    if source_norm <= 0 or target_norm <= 0:
        raise ValueError("cannot align a constant embedding")
    source_unit = source_centered / source_norm
    target_unit = target_centered / target_norm
    rotation, _ = orthogonal_procrustes(source_unit, target_unit)
    aligned = source_unit @ rotation * target_norm
    residual = float(np.sum((target_centered - aligned) ** 2))
    denominator = float(np.sum(target_centered**2))
    r2 = 1.0 - residual / denominator if denominator > 0 else float("nan")
    return aligned, r2


def _nearest_neighbor_overlap(
    original: np.ndarray,
    embedded: np.ndarray,
    *,
    n_neighbors: int,
) -> float:
    k = min(n_neighbors, len(original) - 1)
    high = NearestNeighbors(n_neighbors=k + 1).fit(original)
    low = NearestNeighbors(n_neighbors=k + 1).fit(embedded)
    high_indices = high.kneighbors(return_distance=False)[:, 1:]
    low_indices = low.kneighbors(return_distance=False)[:, 1:]
    overlap = [
        len(set(high_row).intersection(low_row)) / k
        for high_row, low_row in zip(high_indices, low_indices, strict=True)
    ]
    return float(np.mean(overlap))


def _distance_spearman(
    original: np.ndarray,
    embedded: np.ndarray,
    *,
    seed: int,
    n_pairs: int = 50000,
) -> float:
    rng = np.random.default_rng(seed)
    first = rng.integers(0, len(original), size=n_pairs)
    second = rng.integers(0, len(original), size=n_pairs)
    valid = first != second
    first, second = first[valid], second[valid]
    raw_distance = np.linalg.norm(original[first] - original[second], axis=1)
    embedded_distance = np.linalg.norm(embedded[first] - embedded[second], axis=1)
    return float(spearmanr(raw_distance, embedded_distance).statistic)


def _trajectory_step_correlation(
    target_rows: pd.DataFrame,
    embedding: pd.DataFrame,
) -> tuple[float, int]:
    joined = target_rows.merge(embedding, on="state_key", how="left", validate="many_to_one")
    raw_steps: list[float] = []
    display_steps: list[float] = []
    for _, group in joined.groupby("_gid"):
        ordered = group.sort_values("snapshot_index")
        raw = ordered.loc[:, list(STATE_FEATURES)].to_numpy(float)
        display = ordered.loc[:, ["axis1", "axis2"]].to_numpy(float)
        if len(raw) < 2:
            continue
        raw_steps.extend(np.linalg.norm(np.diff(raw, axis=0), axis=1))
        display_steps.extend(np.linalg.norm(np.diff(display, axis=0), axis=1))
    raw_array = np.asarray(raw_steps)
    display_array = np.asarray(display_steps)
    keep = np.isfinite(raw_array) & np.isfinite(display_array)
    if keep.sum() < 3:
        return float("nan"), int(keep.sum())
    correlation = spearmanr(raw_array[keep], display_array[keep]).statistic
    return float(correlation), int(keep.sum())


def evaluate_embeddings(
    matrix: np.ndarray,
    embeddings: Mapping[str, np.ndarray],
    target_rows: pd.DataFrame,
    landmark_keys: pd.Series,
    stability: Mapping[str, float],
    *,
    seed: int,
) -> pd.DataFrame:
    """Measure local, global and trajectory preservation on one held subset."""
    rng = np.random.default_rng(seed)
    n_eval = min(EVALUATION_N, len(matrix))
    indices = np.sort(rng.choice(len(matrix), size=n_eval, replace=False))
    original = matrix[indices]
    rows: list[dict[str, Any]] = []
    for method, coordinates in embeddings.items():
        embedded = coordinates[indices]
        coordinate_table = pd.DataFrame(
            {
                "state_key": landmark_keys,
                "axis1": coordinates[:, 0],
                "axis2": coordinates[:, 1],
            }
        )
        step_correlation, n_steps = _trajectory_step_correlation(
            target_rows,
            coordinate_table,
        )
        rows.append(
            {
                "method": method,
                "n_evaluation_states": n_eval,
                "trustworthiness_k15": float(
                    trustworthiness(
                        original,
                        embedded,
                        n_neighbors=min(15, n_eval // 2 - 1),
                    )
                ),
                "continuity_k15": float(
                    trustworthiness(
                        embedded,
                        original,
                        n_neighbors=min(15, n_eval // 2 - 1),
                    )
                ),
                "knn_overlap_k15": _nearest_neighbor_overlap(
                    original,
                    embedded,
                    n_neighbors=15,
                ),
                "distance_spearman": _distance_spearman(
                    original,
                    embedded,
                    seed=seed,
                ),
                "trajectory_step_spearman": step_correlation,
                "n_trajectory_steps": n_steps,
                "seed_stability_r2": stability.get(method, float("nan")),
            }
        )
    return pd.DataFrame(rows)


def _top_loading_phrase(model: FullPCAModel, component_index: int) -> str:
    vector = model.components[component_index]
    top = np.argsort(np.abs(vector))[::-1][:2]
    return " · ".join(
        FEATURE_LABELS[STATE_FEATURES[index]].replace(" disease", "")
        for index in top
    )


def _path_rows(
    frame_with_keys: pd.DataFrame,
    representative: pd.DataFrame,
) -> pd.DataFrame:
    gids = set(representative["_gid"])
    return frame_with_keys[frame_with_keys["_gid"].isin(gids)].copy()


def _draw_one_path(
    axis: plt.Axes,
    path: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    color: str,
    linewidth: float = 2.2,
    alpha: float = 0.95,
    zorder: int = 5,
) -> None:
    ordered = path.sort_values("snapshot_index")
    x = ordered[x_column].to_numpy(float)
    y = ordered[y_column].to_numpy(float)
    if len(ordered) > 1:
        axis.plot(x, y, color=color, lw=linewidth, alpha=alpha, zorder=zorder)
    axis.scatter(
        x,
        y,
        s=18,
        color=color,
        edgecolor="white",
        linewidth=0.45,
        alpha=alpha,
        zorder=zorder + 1,
    )
    axis.scatter(
        [x[0]],
        [y[0]],
        s=65,
        marker="o",
        color="white",
        edgecolor=color,
        linewidth=1.8,
        zorder=zorder + 2,
    )
    axis.scatter(
        [x[-1]],
        [y[-1]],
        s=65,
        marker="D",
        color=color,
        edgecolor="white",
        linewidth=0.8,
        zorder=zorder + 2,
    )


def draw_pc1_all(
    landmark_scores: np.ndarray,
    path_scores: pd.DataFrame,
    model: FullPCAModel,
    representative: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Draw the scree plot and PC1 against PC2 through PC20."""
    figure, axes = plt.subplots(5, 4, figsize=(17.0, 20.0))
    axes_flat = list(axes.flat)
    variance = model.explained_variance_ratio
    component_numbers = np.arange(1, len(variance) + 1)
    axis = axes_flat[0]
    axis.bar(component_numbers, variance * 100, color="#718096", width=0.78)
    axis.plot(
        component_numbers,
        np.cumsum(variance) * 100,
        color="#8f2636",
        marker="o",
        ms=3,
        lw=1.5,
    )
    axis.axhline(80, color=INK, ls="--", lw=0.8)
    axis.set_xlabel("principal component")
    axis.set_ylabel("variance or cumulative variance (%)")
    axis.set_title(
        "A · the spectrum\n"
        "2 PCs 39.5% · 5 PCs 64.5% · 8 PCs 80.5%",
        loc="left",
        fontsize=10,
        color=INK,
    )
    path_gids = representative.set_index("_gid")["tumor_oncotree_code"].to_dict()
    for panel_index, component_index in enumerate(range(1, 20), start=1):
        axis = axes_flat[panel_index]
        axis.scatter(
            landmark_scores[:, 0],
            landmark_scores[:, component_index],
            s=4,
            color=GREY,
            alpha=0.20,
            linewidth=0,
            rasterized=True,
        )
        for gid, group in path_scores.groupby("_gid"):
            tumour = path_gids[gid]
            _draw_one_path(
                axis,
                group,
                "pc1",
                f"pc{component_index + 1}",
                color=TUMOUR_COLORS[tumour],
                linewidth=1.8,
            )
        axis.set_xlabel(f"PC1 · {variance[0] * 100:.1f}%")
        axis.set_ylabel(f"PC{component_index + 1} · {variance[component_index] * 100:.1f}%")
        axis.set_title(
            f"PC1 × PC{component_index + 1}\n"
            f"{_top_loading_phrase(model, component_index)}",
            loc="left",
            fontsize=9.3,
            color=INK,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TUMOUR_COLORS[tumour],
            lw=2.5,
            marker="o",
            label=TUMOUR_LABELS[tumour],
        )
        for tumour in TARGET_TUMOURS
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncols=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.006),
    )
    figure.suptitle(
        "PC1 is only one view of the same twenty-dimensional clinical paths",
        fontsize=18,
        fontweight="bold",
        color=INK,
        y=0.998,
    )
    figure.text(
        0.5,
        0.978,
        "Grey = identical weighted landmark states · coloured paths = the same three algorithmically selected trajectories",
        ha="center",
        fontsize=10.5,
        color="#4b5563",
    )
    figure.tight_layout(rect=(0.02, 0.035, 0.98, 0.965), h_pad=2.0, w_pad=1.5)
    figure.savefig(out_dir / "fig_pc1_against_all_pcs.png", dpi=210, bbox_inches="tight")
    figure.savefig(out_dir / "fig_pc1_against_all_pcs.pdf", bbox_inches="tight")
    plt.close(figure)


def draw_pc_focused(
    landmark_scores: np.ndarray,
    target_scores: pd.DataFrame,
    representative: pd.DataFrame,
    model: FullPCAModel,
    out_dir: Path,
) -> None:
    """Draw three clinically distinct linear views for potential paper use."""
    pairs = ((1, "dominant variance"), (2, "toxicity-rich"), (4, "anatomical pattern"))
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.1))
    representative_gids = set(representative["_gid"])
    for axis, (component_index, message) in zip(axes, pairs, strict=True):
        axis.scatter(
            landmark_scores[:, 0],
            landmark_scores[:, component_index],
            s=5,
            color=PALE_GREY,
            alpha=0.42,
            linewidth=0,
            rasterized=True,
        )
        for tumour in TARGET_TUMOURS:
            tumour_rows = target_scores[target_scores["tumor_oncotree_code"].eq(tumour)]
            for gid, group in tumour_rows.groupby("_gid"):
                ordered = group.sort_values("snapshot_index")
                axis.plot(
                    ordered["pc1"],
                    ordered[f"pc{component_index + 1}"],
                    color=TUMOUR_COLORS[tumour],
                    alpha=0.11,
                    lw=0.7,
                    zorder=2,
                )
                if gid in representative_gids:
                    _draw_one_path(
                        axis,
                        ordered,
                        "pc1",
                        f"pc{component_index + 1}",
                        color=TUMOUR_COLORS[tumour],
                        linewidth=2.5,
                        zorder=5,
                    )
        axis.set_xlabel(f"PC1 · {model.explained_variance_ratio[0] * 100:.1f}%")
        axis.set_ylabel(
            f"PC{component_index + 1} · "
            f"{model.explained_variance_ratio[component_index] * 100:.1f}%"
        )
        axis.set_title(
            f"{message}\n{_top_loading_phrase(model, component_index)}",
            loc="left",
            fontsize=11,
            color=INK,
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color=GRID, lw=0.6, alpha=0.6)
        axis.set_axisbelow(True)
    legend_handles = [
        Line2D([0], [0], color=TUMOUR_COLORS[tumour], lw=2.5, label=tumour)
        for tumour in TARGET_TUMOURS
    ]
    axes[-1].legend(handles=legend_handles, frameon=False, loc="best")
    figure.suptitle(
        "Different linear projections expose different clinical stories",
        fontsize=16,
        fontweight="bold",
        color=INK,
        y=1.02,
    )
    figure.text(
        0.5,
        -0.01,
        "All faint paths: 30 trajectories per tumour type · bold path: declared median-like exemplar · circles start, diamonds end",
        ha="center",
        fontsize=9.5,
        color="#4b5563",
    )
    figure.tight_layout()
    figure.savefig(out_dir / "fig_pc_focused_views.png", dpi=210, bbox_inches="tight")
    figure.savefig(out_dir / "fig_pc_focused_views.pdf", bbox_inches="tight")
    plt.close(figure)


def _embedding_coordinate_table(
    landmarks: pd.DataFrame,
    embeddings: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for method, coordinates in embeddings.items():
        table = landmarks.loc[:, ["state_key"]].copy()
        table["method"] = method
        table["axis1"] = coordinates[:, 0]
        table["axis2"] = coordinates[:, 1]
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def draw_embedding_comparison(
    landmarks: pd.DataFrame,
    coordinate_table: pd.DataFrame,
    representative_rows: pd.DataFrame,
    representative: pd.DataFrame,
    diagnostics: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Draw the four coordinate systems with identical paths and state colours."""
    figure, axes = plt.subplots(1, 4, figsize=(18.0, 4.9))
    pc1_values = landmarks["pc1"].to_numpy(float)
    norm = Normalize(
        vmin=float(np.quantile(pc1_values, 0.02)),
        vmax=float(np.quantile(pc1_values, 0.98)),
        clip=True,
    )
    diagnostic_index = diagnostics.set_index("method")
    path_gids = representative.set_index("_gid")["tumor_oncotree_code"].to_dict()
    for axis, method in zip(axes, METHOD_LABELS, strict=True):
        coordinates = coordinate_table[coordinate_table["method"].eq(method)]
        merged_landmarks = landmarks.merge(coordinates, on="state_key", validate="one_to_one")
        scatter = axis.scatter(
            merged_landmarks["axis1"],
            merged_landmarks["axis2"],
            c=merged_landmarks["pc1"],
            cmap="viridis",
            norm=norm,
            s=5,
            alpha=0.36,
            linewidth=0,
            rasterized=True,
        )
        joined_paths = representative_rows.merge(
            coordinates,
            on="state_key",
            how="left",
            validate="many_to_one",
        )
        for gid, group in joined_paths.groupby("_gid"):
            tumour = path_gids[gid]
            _draw_one_path(
                axis,
                group,
                "axis1",
                "axis2",
                color=TUMOUR_COLORS[tumour],
                linewidth=2.4,
            )
        row = diagnostic_index.loc[method]
        axis.set_title(
            f"{METHOD_LABELS[method]}\n"
            f"local trust {row['trustworthiness_k15']:.2f} · "
            f"path ρ {row['trajectory_step_spearman']:.2f}",
            loc="left",
            fontsize=10.8,
            color=INK,
        )
        axis.set_xlabel(f"{METHOD_LABELS[method]} 1")
        axis.set_ylabel(f"{METHOD_LABELS[method]} 2")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[["top", "right", "bottom", "left"]].set_visible(False)
    colorbar = figure.colorbar(scatter, ax=axes, fraction=0.015, pad=0.015)
    colorbar.set_label("PC1 global-state score (colour only)", fontsize=9)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TUMOUR_COLORS[tumour],
            lw=2.5,
            marker="o",
            label=tumour,
        )
        for tumour in TARGET_TUMOURS
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncols=3,
        frameon=False,
        bbox_to_anchor=(0.48, -0.015),
    )
    figure.suptitle(
        "The same clinical states and trajectories through four coordinate systems",
        fontsize=17,
        fontweight="bold",
        color=INK,
        y=1.025,
    )
    figure.text(
        0.5,
        -0.065,
        "Nonlinear axes carry no variance percentage or clinical unit; segments join separately embedded event states and are display-only",
        ha="center",
        fontsize=9.5,
        color="#4b5563",
    )
    figure.subplots_adjust(
        left=0.025,
        right=0.92,
        bottom=0.18,
        top=0.82,
        wspace=0.20,
    )
    figure.savefig(out_dir / "fig_embedding_comparison.png", dpi=220, bbox_inches="tight")
    figure.savefig(out_dir / "fig_embedding_comparison.pdf", bbox_inches="tight")
    plt.close(figure)


def draw_embedding_by_tumour(
    landmarks: pd.DataFrame,
    coordinate_table: pd.DataFrame,
    target_rows: pd.DataFrame,
    representative: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Draw tumour-specific trajectories on every common embedding."""
    figure, axes = plt.subplots(3, 4, figsize=(17.0, 12.0))
    representative_by_tumour = representative.set_index("tumor_oncotree_code")["_gid"]
    for row_index, tumour in enumerate(TARGET_TUMOURS):
        tumour_rows = target_rows[target_rows["tumor_oncotree_code"].eq(tumour)]
        representative_gid = representative_by_tumour.loc[tumour]
        for column_index, method in enumerate(METHOD_LABELS):
            axis = axes[row_index, column_index]
            coordinates = coordinate_table[coordinate_table["method"].eq(method)]
            merged_landmarks = landmarks.merge(
                coordinates,
                on="state_key",
                validate="one_to_one",
            )
            axis.scatter(
                merged_landmarks["axis1"],
                merged_landmarks["axis2"],
                s=4,
                color=PALE_GREY,
                alpha=0.35,
                linewidth=0,
                rasterized=True,
            )
            joined = tumour_rows.merge(
                coordinates,
                on="state_key",
                how="left",
                validate="many_to_one",
            )
            for gid, group in joined.groupby("_gid"):
                ordered = group.sort_values("snapshot_index")
                axis.plot(
                    ordered["axis1"],
                    ordered["axis2"],
                    color=TUMOUR_COLORS[tumour],
                    alpha=0.16,
                    lw=0.75,
                    zorder=2,
                )
                axis.scatter(
                    ordered["axis1"],
                    ordered["axis2"],
                    color=TUMOUR_COLORS[tumour],
                    alpha=0.11,
                    s=6,
                    linewidth=0,
                    zorder=2,
                )
                if gid == representative_gid:
                    _draw_one_path(
                        axis,
                        ordered,
                        "axis1",
                        "axis2",
                        color=TUMOUR_COLORS[tumour],
                        linewidth=2.6,
                        zorder=5,
                    )
            if row_index == 0:
                axis.set_title(METHOD_LABELS[method], fontsize=12, color=INK)
            if column_index == 0:
                axis.set_ylabel(
                    f"{tumour}\n{TUMOUR_LABELS[tumour]}",
                    fontsize=10,
                    color=TUMOUR_COLORS[tumour],
                    fontweight="bold",
                )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines[["top", "right", "bottom", "left"]].set_visible(False)
    figure.suptitle(
        "Tumour-conditioned paths occupy overlapping but differently shaped regions",
        fontsize=17,
        fontweight="bold",
        color=INK,
        y=0.99,
    )
    figure.text(
        0.5,
        0.015,
        "Thirty stored trajectories per tumour (faint) · one declared median-like trajectory (bold) · identical states in every column",
        ha="center",
        fontsize=9.8,
        color="#4b5563",
    )
    figure.tight_layout(rect=(0.03, 0.04, 0.99, 0.965), h_pad=1.5, w_pad=0.8)
    figure.savefig(out_dir / "fig_embeddings_by_tumour.png", dpi=220, bbox_inches="tight")
    figure.savefig(out_dir / "fig_embeddings_by_tumour.pdf", bbox_inches="tight")
    plt.close(figure)


def draw_diagnostics(diagnostics: pd.DataFrame, out_dir: Path) -> None:
    """Draw compact diagnostics without turning them into a winner score."""
    metrics = [
        ("trustworthiness_k15", "local trust"),
        ("continuity_k15", "continuity"),
        ("knn_overlap_k15", "15-NN overlap"),
        ("distance_spearman", "global distance ρ"),
        ("trajectory_step_spearman", "trajectory-step ρ"),
        ("seed_stability_r2", "seed stability R²"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
    methods = list(METHOD_LABELS)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for axis, (metric, label) in zip(axes.flat, metrics, strict=True):
        values = diagnostics.set_index("method").loc[methods, metric]
        axis.bar(np.arange(len(methods)), values, color=colors, width=0.72)
        axis.set_xticks(np.arange(len(methods)))
        axis.set_xticklabels([METHOD_LABELS[method] for method in methods], rotation=22)
        axis.set_ylim(min(0.0, float(values.min()) - 0.05), 1.02)
        axis.set_title(label, loc="left", fontsize=11, color=INK)
        axis.grid(axis="y", color=GRID, lw=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        for position, value in enumerate(values):
            axis.text(
                position,
                value + 0.025,
                f"{value:.2f}",
                ha="center",
                fontsize=8.5,
                color=INK,
            )
    figure.suptitle(
        "No two-dimensional map preserves every property of the twenty-dimensional state",
        fontsize=16,
        fontweight="bold",
        color=INK,
        y=1.01,
    )
    figure.text(
        0.5,
        -0.015,
        "Diagnostics are complementary, not a composite score; seed stability is 1 by construction for deterministic PCA/diffusion",
        ha="center",
        fontsize=9.5,
        color="#4b5563",
    )
    figure.tight_layout()
    figure.savefig(out_dir / "fig_embedding_diagnostics.png", dpi=210, bbox_inches="tight")
    figure.savefig(out_dir / "fig_embedding_diagnostics.pdf", bbox_inches="tight")
    plt.close(figure)


def write_readout(
    out_dir: Path,
    *,
    model: FullPCAModel,
    representative: pd.DataFrame,
    diagnostics: pd.DataFrame,
    n_rows: int,
    n_unique_states: int,
    n_landmarks: int,
    frozen_max_difference: float,
    alignment: Mapping[str, float],
) -> None:
    """Write a clinician-facing interpretation of the bake-off."""
    lines = [
        "# Frozen-state display bake-off",
        "",
        "## What is held fixed",
        "",
        f"- {n_rows:,} living event snapshots and {n_unique_states:,} exact "
        "twenty-dimensional state vectors.",
        f"- {n_landmarks:,} deduplicated landmark states, sampled using the "
        "frozen equal-model/equal-trajectory observation weights.",
        "- All states use the frozen 10 tumour + 10 host fixed-range encoding.",
        "- No treatment, outcome, future event, tumour identity or model identity "
        "is an embedding input.",
        "- All PAAD, LUAD and HGSOC states are forced into the common landmark "
        "set before drawing their paths.",
        "",
        "PCA is the only method with variance percentages and directly interpretable "
        "linear loadings. UMAP, PHATE and diffusion maps preserve different graph "
        "properties and have arbitrary axes. Raw-feature clinical estimands do not "
        "change with the display.",
        "",
        "## Complete PCA spectrum",
        "",
        "| axis | variance | cumulative | three largest absolute loadings |",
        "|---|---:|---:|---|",
    ]
    cumulative = np.cumsum(model.explained_variance_ratio)
    for index, ratio in enumerate(model.explained_variance_ratio):
        vector = model.components[index]
        top = np.argsort(np.abs(vector))[::-1][:3]
        phrase = "; ".join(
            f"{FEATURE_LABELS[STATE_FEATURES[position]]} "
            f"({vector[position]:+.2f})"
            for position in top
        )
        lines.append(
            f"| PC{index + 1} | {100 * ratio:.2f}% | "
            f"{100 * cumulative[index]:.2f}% | {phrase} |"
        )
    lines += [
        "",
        f"Maximum complete-PCA parameter difference from an independent refit: "
        f"{frozen_max_difference:.3g}.",
        "",
        "## Declared representative trajectories",
        "",
        "The bold trajectory is not hand-picked. Within each tumour type it is "
        "closest to the median trajectory across five display-only summaries: "
        "snapshot count, follow-up, raw path length, fraction of unchanged steps "
        "and PC1 range.",
        "",
        "| tumour | model | trajectory | snapshots | follow-up | raw path length | selection distance |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in representative.itertuples(index=False):
        lines.append(
            f"| {row.tumor_oncotree_code} | {row.model_short} | "
            f"`{row.trajectory_id}` | {row.n_snapshots} | "
            f"{row.followup_months:.1f} mo | {row.raw_path_length:.2f} | "
            f"{row.selection_distance:.2f} |"
        )
    lines += [
        "",
        "## Projection diagnostics",
        "",
        "| method | local trust | continuity | 15-NN overlap | global distance ρ | path-step ρ | seed stability R² |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics.itertuples(index=False):
        lines.append(
            f"| {METHOD_LABELS[row.method]} | {row.trustworthiness_k15:.3f} | "
            f"{row.continuity_k15:.3f} | {row.knn_overlap_k15:.3f} | "
            f"{row.distance_spearman:.3f} | "
            f"{row.trajectory_step_spearman:.3f} | "
            f"{row.seed_stability_r2:.3f} |"
        )
    lines += [
        "",
        "The embeddings were orthogonally aligned to PC1/PC2 only so that panels "
        "have comparable visual orientation. This changes neither neighbourhoods "
        "nor within-map distances. Alignment-to-PCA R² values are:",
    ]
    for method, value in alignment.items():
        lines.append(f"- {METHOD_LABELS[method]}: {value:.3f}.")
    lines += [
        "",
        "## Decision for the paper",
        "",
        "- Keep PCA as the common quantitative geography. It best preserves global "
        "distance and the lengths of observed clinical steps, and its axes retain "
        "declared variance and loadings.",
        "- Use PC1 × PC2 for the common atlas, PC1 × PC3 when toxicity is the "
        "message, and PC1 × PC5 when anatomical pattern is the message. These are "
        "different declared views of the same frozen state vectors, not different "
        "analytic state spaces.",
        "- PHATE is the strongest nonlinear companion for a descriptive trajectory "
        "panel: it is visually smooth, nearly seed-stable here, and preserves local "
        "continuity without UMAP's stronger island fragmentation. It remains a "
        "display, not the coordinate system in which clinical effects are estimated.",
        "- UMAP has the highest local-neighbourhood score but lower seed stability "
        "and can turn continuous courses into apparent island crossings. Diffusion "
        "maps compress these data into a narrow branching continuum. Keep either "
        "for sensitivity or Extended Data rather than the main clinical trajectory "
        "figure.",
        "",
        "## Interpretation boundary",
        "",
        "- A path segment is the line between separately embedded event states. "
        "It is not the projection of a raw clinical vector through a nonlinear "
        "Jacobian.",
        "- Nonlinear arrow lengths, densities, holes and apparent basins are "
        "display properties. Attractors, occupancy and treatment effects must "
        "remain defined and tested in raw clinical variables.",
        "- Multiple PC views can legitimately be used for different messages if "
        "the selected axes and their loadings are declared before interpreting "
        "the panel.",
        "",
    ]
    (out_dir / "readout.md").write_text("\n".join(lines))


def build(
    *,
    state_path: Path = DEFAULT_STATE_PATH,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    out_dir: Path = DEFAULT_OUT,
    n_landmarks: int = N_LANDMARKS,
    seed: int = RNG_SEED,
    run_stability: bool = True,
) -> dict[str, Any]:
    """Build every display and return its provenance manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    state = load_state_projection(state_path)
    model = load_serialized_full_pca(contract_path)
    reconstructed_model = fit_full_weighted_pca(state)
    frozen_difference = verify_serialized_full_pca(model, reconstructed_model)
    all_pc_scores = model.transform(state)
    trajectory_summaries = summarize_trajectories(state, all_pc_scores)
    representative = select_representative_trajectories(trajectory_summaries)
    landmarks, state_with_keys = select_landmarks(
        state,
        n_landmarks=n_landmarks,
        seed=seed,
    )
    matrix = landmarks.loc[:, list(STATE_FEATURES)].to_numpy(float)
    landmark_pc_scores = model.transform(landmarks)
    landmarks["pc1"] = landmark_pc_scores[:, 0]
    target_rows = state_with_keys[
        state_with_keys["tumor_oncotree_code"].isin(TARGET_TUMOURS)
    ].copy()
    representative_rows = _path_rows(state_with_keys, representative)

    path_scores = representative_rows.copy()
    representative_pc_scores = model.transform(path_scores)
    for index in range(20):
        path_scores[f"pc{index + 1}"] = representative_pc_scores[:, index]
    target_scores = target_rows.copy()
    target_pc_scores = model.transform(target_scores)
    for index in (0, 1, 2, 4):
        target_scores[f"pc{index + 1}"] = target_pc_scores[:, index]

    raw_embeddings: dict[str, np.ndarray] = {
        "pca": landmark_pc_scores[:, :2],
        "umap": fit_umap(matrix, seed=seed),
        "phate": fit_phate(matrix, seed=seed),
    }
    diffusion, diffusion_eigenvalues = fit_diffusion_map(matrix)
    raw_embeddings["diffusion"] = diffusion

    embeddings: dict[str, np.ndarray] = {}
    alignment: dict[str, float] = {}
    for method, coordinates in raw_embeddings.items():
        aligned, r2 = align_to_reference(coordinates, landmark_pc_scores[:, :2])
        embeddings[method] = aligned
        alignment[method] = r2

    stability: dict[str, float] = {"pca": 1.0, "diffusion": 1.0}
    if run_stability:
        for method, fit_function in (("umap", fit_umap), ("phate", fit_phate)):
            replicate = fit_function(matrix, seed=seed + 97)
            aligned_replicate, stability_r2 = align_to_reference(
                replicate,
                embeddings[method],
            )
            del aligned_replicate
            stability[method] = stability_r2
    else:
        stability.update({"umap": float("nan"), "phate": float("nan")})

    coordinate_table = _embedding_coordinate_table(landmarks, embeddings)
    diagnostics = evaluate_embeddings(
        matrix,
        embeddings,
        target_rows,
        landmarks["state_key"],
        stability,
        seed=seed,
    )

    pca_loadings = pd.DataFrame(
        model.components,
        columns=list(STATE_FEATURES),
        index=[f"pc{index}" for index in range(1, 21)],
    )
    pca_loadings.insert(0, "explained_variance_ratio", model.explained_variance_ratio)
    pca_loadings.insert(1, "cumulative_variance_ratio", np.cumsum(model.explained_variance_ratio))
    pca_loadings.to_csv(out_dir / "full_pca_loadings.csv", index_label="axis")
    landmark_output = landmarks.loc[
        :,
        [
            "state_key",
            "observation_weight",
            "required_target_state",
            *STATE_FEATURES,
        ],
    ].copy()
    for index in range(20):
        landmark_output[f"pc{index + 1}"] = landmark_pc_scores[:, index]
    landmark_output.to_csv(out_dir / "landmark_states_and_pcs.csv.gz", index=False)
    coordinate_table.to_csv(out_dir / "embedding_coordinates.csv.gz", index=False)
    representative.to_csv(out_dir / "representative_trajectories.csv", index=False)
    target_rows.to_csv(out_dir / "target_tumour_trajectory_states.csv.gz", index=False)
    diagnostics.to_csv(out_dir / "embedding_diagnostics.csv", index=False)

    draw_pc1_all(
        landmark_pc_scores,
        path_scores,
        model,
        representative,
        out_dir,
    )
    draw_pc_focused(
        landmark_pc_scores,
        target_scores,
        representative,
        model,
        out_dir,
    )
    draw_embedding_comparison(
        landmarks,
        coordinate_table,
        representative_rows,
        representative,
        diagnostics,
        out_dir,
    )
    draw_embedding_by_tumour(
        landmarks,
        coordinate_table,
        target_rows,
        representative,
        out_dir,
    )
    draw_diagnostics(diagnostics, out_dir)
    write_readout(
        out_dir,
        model=model,
        representative=representative,
        diagnostics=diagnostics,
        n_rows=len(state),
        n_unique_states=int(state_keys(state).nunique()),
        n_landmarks=len(landmarks),
        frozen_max_difference=frozen_difference,
        alignment=alignment,
    )

    import phate
    import umap

    manifest = {
        "artifact": "paper_state_embedding_bakeoff_v1",
        "git_commit": _git_commit(),
        "script_sha256": _sha256(Path(__file__)),
        "state_projection": str(state_path.relative_to(ROOT)),
        "state_projection_sha256": _sha256(state_path),
        "state_map_contract": str(contract_path.relative_to(ROOT)),
        "state_map_contract_sha256": _sha256(contract_path),
        "n_input_snapshots": len(state),
        "n_exact_unique_states": int(state_keys(state).nunique()),
        "n_landmark_states": len(landmarks),
        "target_tumours": list(TARGET_TUMOURS),
        "target_trajectories_per_tumour": {
            tumour: int(
                target_rows.loc[
                    target_rows["tumor_oncotree_code"].eq(tumour),
                    "_gid",
                ].nunique()
            )
            for tumour in TARGET_TUMOURS
        },
        "representative_gids": representative.set_index("tumor_oncotree_code")[
            "_gid"
        ].to_dict(),
        "seed": seed,
        "state_contract_changed": False,
        "outcomes_used": False,
        "treatments_used": False,
        "nonlinear_coordinates_are_display_only": True,
        "pca": {
            "complete_serialized_basis": True,
            "serialized_component_count": len(model.components),
            "complete_pca_max_abs_parameter_difference": frozen_difference,
            "explained_variance_ratio": model.explained_variance_ratio.tolist(),
        },
        "umap": {
            "version": umap.__version__,
            "n_neighbors": N_NEIGHBORS,
            "min_dist": 0.12,
            "metric": "euclidean",
        },
        "phate": {
            "version": phate.__version__,
            "knn": 15,
            "decay": 40,
            "n_landmark": min(1200, len(landmarks)),
            "t": "auto",
            "gamma": 1,
            "mds_solver": "smacof",
        },
        "diffusion_map": {
            "n_neighbors": N_NEIGHBORS,
            "alpha": 1.0,
            "diffusion_time": 1,
            "leading_eigenvalues": diffusion_eigenvalues.tolist(),
        },
        "alignment_to_pc12_r2": alignment,
        "stability_replicate_run": run_stability,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--contract-path", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-landmarks", type=int, default=N_LANDMARKS)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument(
        "--skip-stability",
        action="store_true",
        help="Skip the second UMAP/PHATE seed used for visual stability R².",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build(
        state_path=args.state_path,
        contract_path=args.contract_path,
        out_dir=args.out_dir,
        n_landmarks=args.n_landmarks,
        seed=args.seed,
        run_stability=not args.skip_stability,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
