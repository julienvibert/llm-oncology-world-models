from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import build_paper_state_embedding_bakeoff as bakeoff  # noqa: E402


def _state_rows(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    matrix = rng.uniform(0, 1, size=(n, len(bakeoff.STATE_FEATURES)))
    rows = pd.DataFrame(matrix, columns=list(bakeoff.STATE_FEATURES))
    rows["observation_weight"] = 1.0 / n
    rows["_gid"] = [f"gid-{index // 4}" for index in range(n)]
    rows["trajectory_id"] = rows["_gid"]
    rows["model_short"] = "model"
    rows["tumor_oncotree_code"] = [
        bakeoff.TARGET_TUMOURS[index % len(bakeoff.TARGET_TUMOURS)]
        for index in range(n)
    ]
    rows["snapshot_index"] = [index % 4 for index in range(n)]
    rows["days_from_diagnosis"] = np.arange(n) * 30
    return rows


def test_full_weighted_pca_is_complete_and_orthonormal() -> None:
    rows = _state_rows(40)
    model = bakeoff.fit_full_weighted_pca(rows)

    assert model.components.shape == (20, 20)
    assert model.explained_variance_ratio.shape == (20,)
    assert np.isclose(model.explained_variance_ratio.sum(), 1.0)
    assert np.allclose(model.components @ model.components.T, np.eye(20))
    assert model.transform(rows).shape == (40, 20)


def test_complete_pca_round_trips_from_serialized_contract(
    tmp_path: Path,
) -> None:
    rows = _state_rows(40)
    fitted = bakeoff.fit_full_weighted_pca(rows)
    payload = {
        "primary_model": {
            "feature_names": list(bakeoff.STATE_FEATURES),
            "mean_after_feature_multipliers": fitted.mean.tolist(),
            "all_components_after_feature_multipliers": (
                fitted.components.tolist()
            ),
            "eigenvalues": fitted.eigenvalues.tolist(),
            "explained_variance_ratio": (
                fitted.explained_variance_ratio.tolist()
            ),
            "feature_multipliers": fitted.feature_multipliers.tolist(),
        }
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(payload))

    loaded = bakeoff.load_serialized_full_pca(contract_path)

    assert bakeoff.verify_serialized_full_pca(loaded, fitted) < 1e-12
    assert np.allclose(loaded.transform(rows), fitted.transform(rows))


def test_state_keys_match_only_exact_feature_vectors() -> None:
    rows = _state_rows(5)
    rows.loc[1, list(bakeoff.STATE_FEATURES)] = rows.loc[
        0, list(bakeoff.STATE_FEATURES)
    ].to_numpy()

    keys = bakeoff.state_keys(rows)

    assert keys.iloc[0] == keys.iloc[1]
    assert keys.nunique() == 4


def test_representative_selection_is_median_like_and_deterministic() -> None:
    summaries = pd.DataFrame(
        {
            "_gid": ["a", "b", "c"],
            "trajectory_id": ["a", "b", "c"],
            "model_short": ["m", "m", "m"],
            "tumor_oncotree_code": ["LUAD", "LUAD", "LUAD"],
            "n_snapshots": [3, 5, 10],
            "followup_months": [3.0, 5.0, 20.0],
            "raw_path_length": [0.2, 0.5, 3.0],
            "fraction_static_steps": [0.8, 0.5, 0.0],
            "pc1_range": [0.1, 0.4, 2.0],
        }
    )

    selected = bakeoff.select_representative_trajectories(
        summaries,
        tumour_types=["LUAD"],
    )

    assert selected.loc[0, "_gid"] == "b"


def test_landmarks_deduplicate_and_retain_target_states() -> None:
    rows = _state_rows(18)
    rows.loc[1, list(bakeoff.STATE_FEATURES)] = rows.loc[
        0, list(bakeoff.STATE_FEATURES)
    ].to_numpy()
    target_rows = rows["tumor_oncotree_code"].eq("PAAD")

    landmarks, keyed = bakeoff.select_landmarks(
        rows,
        target_tumours=["PAAD"],
        n_landmarks=8,
        seed=4,
    )

    target_keys = set(keyed.loc[target_rows, "state_key"])
    assert target_keys.issubset(set(landmarks["state_key"]))
    assert landmarks["state_key"].is_unique
    assert len(landmarks) >= len(target_keys)


def test_diffusion_map_is_finite_and_two_dimensional() -> None:
    rows = _state_rows(45)
    matrix = rows.loc[:, list(bakeoff.STATE_FEATURES)].to_numpy(float)

    coordinates, eigenvalues = bakeoff.fit_diffusion_map(
        matrix,
        n_neighbors=8,
    )

    assert coordinates.shape == (45, 2)
    assert np.isfinite(coordinates).all()
    assert np.all(np.diff(eigenvalues) <= 1e-10)


def test_alignment_preserves_pairwise_distances_up_to_one_scale() -> None:
    rng = np.random.default_rng(11)
    coordinates = rng.normal(size=(25, 2))
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    reference = 3.0 * coordinates @ rotation + np.asarray([9.0, -2.0])

    aligned, r2 = bakeoff.align_to_reference(coordinates, reference)

    assert r2 > 0.999999
    source_distance = pairwise_distance_vector(coordinates)
    aligned_distance = pairwise_distance_vector(aligned)
    ratio = aligned_distance / source_distance
    assert np.allclose(ratio, ratio[0])


def pairwise_distance_vector(matrix: np.ndarray) -> np.ndarray:
    differences = matrix[:, None, :] - matrix[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    return distances[np.triu_indices(len(matrix), k=1)]
