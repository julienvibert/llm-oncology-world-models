"""Tests for the frozen paper-v1 clinical state display map."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_paper_state_map as state_map  # noqa: E402


def test_primary_lung_burden_resolves_to_thoracic_without_double_counting() -> None:
    sites = pd.DataFrame(
        {
            "_gid": ["g", "g", "g"],
            "snapshot_index": [0, 0, 0],
            "site_class_canonical": ["primary_site", "lung", "liver"],
            "burden_band": ["moderate", "small", "bulky"],
        }
    )
    baselines = pd.DataFrame(
        {
            "_gid": ["g"],
            "primary_site_class_canonical": ["lung"],
            "primary_site": ["right lower lobe lung"],
        }
    )

    grouped, resolved = state_map.build_grouped_anatomy(sites, baselines)
    row = grouped.iloc[0]

    assert resolved.loc[0, "site_class_resolved"] == "lung"
    assert resolved.loc[0, "primary_resolution_source"] == (
        "baseline_primary_context"
    )
    assert row["tumour_thoracic"] == pytest.approx(3.0 / 5.0)
    assert row["tumour_hepatic"] == pytest.approx(4.0 / 5.0)
    # Generic primary_site + named lung are one resolved physical class.
    assert row["tumour_extent"] == pytest.approx(2.0 / 5.0)


def test_anatomical_extent_uses_resolved_classes_and_caps_at_five() -> None:
    classes = [
        "liver",
        "lung",
        "brain",
        "bone",
        "lymph_node",
        "skin",
        "adrenal",
    ]
    sites = pd.DataFrame(
        {
            "_gid": ["g"] * len(classes),
            "snapshot_index": [0] * len(classes),
            "site_class_canonical": classes,
            "burden_band": ["small"] * len(classes),
        }
    )
    baselines = pd.DataFrame(
        {
            "_gid": ["g"],
            "primary_site_class_canonical": ["lung"],
            "primary_site": ["lung"],
        }
    )

    grouped, _resolved = state_map.build_grouped_anatomy(sites, baselines)

    assert grouped.iloc[0]["tumour_extent"] == 1.0
    assert grouped.iloc[0]["tumour_cns"] == pytest.approx(2.0 / 5.0)
    assert grouped.iloc[0]["tumour_soft_tissue_cutaneous"] == pytest.approx(
        2.0 / 5.0
    )


def test_invalid_primary_category_uses_audited_label_fallback() -> None:
    sites = pd.DataFrame(
        {
            "_gid": ["g"],
            "snapshot_index": [0],
            "site_class_canonical": ["primary_site"],
            "burden_band": ["moderate"],
        }
    )
    baselines = pd.DataFrame(
        {
            "_gid": ["g"],
            "primary_site_class_canonical": ["invalid_primary_site_class"],
            "primary_site": ["right lung lower lobe"],
        }
    )

    grouped, resolved = state_map.build_grouped_anatomy(sites, baselines)

    assert resolved.iloc[0]["site_class_resolved"] == "lung"
    assert resolved.iloc[0]["primary_resolution_source"] == (
        "baseline_primary_label_fallback"
    )
    assert grouped.iloc[0]["tumour_thoracic"] == pytest.approx(3.0 / 5.0)


def test_weight_loss_uses_fixed_asymmetric_clinical_range() -> None:
    current = pd.Series([120.0, 100.0, 70.0, 50.0])
    baseline = pd.Series([100.0, 100.0, 100.0, 100.0])

    encoded = state_map.encode_weight_loss(current, baseline)

    assert encoded.tolist() == pytest.approx([0.0, 0.4, 1.0, 1.0])


def test_equal_model_trajectory_snapshot_weights_have_nested_equal_mass() -> None:
    frame = pd.DataFrame(
        {
            "model_short": ["m1", "m1", "m1", "m2", "m2", "m2", "m2"],
            "_gid": ["a", "a", "b", "c", "c", "c", "c"],
        }
    )

    weights = state_map.equal_model_trajectory_snapshot_weights(frame)
    weighted = frame.assign(weight=weights)

    model_mass = weighted.groupby("model_short")["weight"].sum()
    trajectory_mass = weighted.groupby(["model_short", "_gid"])["weight"].sum()
    assert model_mass.to_dict() == pytest.approx({"m1": 0.5, "m2": 0.5})
    assert trajectory_mass.loc[("m1", "a")] == pytest.approx(0.25)
    assert trajectory_mass.loc[("m1", "b")] == pytest.approx(0.25)
    assert trajectory_mass.loc[("m2", "c")] == pytest.approx(0.5)


def test_weighted_pca_projects_without_empirical_z_scoring() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.uniform(0.0, 1.0, size=(30, len(state_map.STATE_FEATURES)))
    frame = pd.DataFrame(matrix, columns=state_map.STATE_FEATURES)
    weights = np.full(len(frame), 1.0 / len(frame))

    model, scores = state_map.fit_weighted_pca(frame, weights)

    assert scores.shape == (30, 2)
    assert np.allclose(scores, model.transform(frame))
    assert np.allclose(model.feature_multipliers, 1.0)
    assert model.components.shape == (2, len(state_map.STATE_FEATURES))
    assert model.full_components.shape == (
        len(state_map.STATE_FEATURES),
        len(state_map.STATE_FEATURES),
    )
    assert model.transform_all(frame).shape == (
        len(frame),
        len(state_map.STATE_FEATURES),
    )
    assert np.allclose(
        model.transform_all(frame)[:, :2],
        scores,
    )
    assert np.allclose(
        model.full_components @ model.full_components.T,
        np.eye(len(state_map.STATE_FEATURES)),
    )
    assert model.explained_variance_ratio[:2].sum() <= 1.0

    contract = model.as_json()
    assert contract["display_component_count"] == 2
    assert contract["serialized_component_count"] == len(state_map.STATE_FEATURES)
    assert np.asarray(
        contract["all_components_after_feature_multipliers"]
    ).shape == (len(state_map.STATE_FEATURES), len(state_map.STATE_FEATURES))


def test_locf_does_not_use_a_future_snapshot() -> None:
    course = pd.DataFrame(
        {
            "days_from_diagnosis": [0.0, 100.0],
            "snapshot_index": [0, 1],
            "value": [0.0, 1.0],
        }
    )

    landmark = state_map.locf_at_day(course, 50.0)

    assert landmark["value"] == 0.0


def test_event_transition_straddles_event_without_linear_interpolation() -> None:
    rows = []
    for index, day in enumerate([0.0, 30.0]):
        row: dict[str, object] = {
            "_gid": "g",
            "days_from_diagnosis": day,
            "snapshot_index": index,
        }
        row.update({feature: 0.0 for feature in state_map.STATE_FEATURES})
        if index == 1:
            row["tumour_hepatic"] = 0.6
        rows.append(row)
    state = pd.DataFrame(rows)
    events = pd.DataFrame(
        {
            "_gid": ["g"],
            "event_type": ["progression"],
            "days_from_diagnosis": [30.0],
        }
    )

    result = state_map.event_transitions(state, events)

    assert len(result) == 1
    assert result.iloc[0]["delta_tumour"] == pytest.approx(
        0.6 / len(state_map.TUMOUR_FEATURES)
    )
    assert result.iloc[0]["delta_host"] == 0.0
    assert result.iloc[0]["delta_toxicity"] == 0.0
