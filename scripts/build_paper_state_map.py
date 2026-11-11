#!/usr/bin/env python3
"""Build and validate the frozen paper-v1 clinical state display map.

The map is deliberately a display coordinate system, not a claim that two
principal components are a sufficient patient representation. The analytic
state remains the retained 20-dimensional visit-state vector:

* nine grouped anatomical tumour-burden coordinates plus disease extent;
* ECOG, global symptoms, five organ functions, relative weight loss, maximum
  active adverse-event grade and active adverse-event count.

Every coordinate is encoded over a fixed clinical range in [0, 1]. The primary
PCA is mean-centred but is not empirically variance-standardised. The fit gives
equal total weight to every generator, every trajectory within a generator, and
every stored event snapshot within a trajectory. This prevents model sample
size, trajectory length and state residence from silently defining the map.

Recorded paths use last observation carried forward (LOCF) between generated
event snapshots. Linear interpolation is never used for quantitative movement.

Run:
  uv run python scripts/build_paper_state_map.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_paper_state_feature_ledger as ledger  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "outputs" / "paper_state_map"
REFERENCE_RUNS = ledger.REFERENCE_RUNS
PHASE1_RUN = "interventional_phase1_sonnet_full_v1c"
MONTH_DAYS = 30.4
TAU_MONTHS = 3.0
POST_EVENT_MONTHS = 3.0

BURDEN_RANK = ledger.BURDEN_RANK
SYMPTOM_RANK = ledger.SYMPTOM_RANK
ORGAN_RANK = ledger.ORGAN_RANK
ORGAN_FIELDS = ledger.ORGAN_FIELDS

ANATOMICAL_GROUPS: dict[str, tuple[str, ...]] = {
    "tumour_hepatic": ("liver",),
    "tumour_thoracic": ("lung", "pleura"),
    "tumour_nodal": ("lymph_node",),
    "tumour_osseous": ("bone",),
    "tumour_marrow": ("bone_marrow",),
    "tumour_cns": ("brain", "central_nervous_system", "leptomeningeal_cns"),
    "tumour_peritoneal": ("peritoneum", "retroperitoneum"),
    "tumour_soft_tissue_cutaneous": ("soft_tissue", "skin"),
}
OTHER_ANATOMICAL_FEATURE = "tumour_other_anatomical"
EXTENT_FEATURE = "tumour_extent"

TUMOUR_FEATURES = (
    *ANATOMICAL_GROUPS,
    OTHER_ANATOMICAL_FEATURE,
    EXTENT_FEATURE,
)
HOST_FEATURES = (
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
)
STATE_FEATURES = (*TUMOUR_FEATURES, *HOST_FEATURES)

GENERAL_HOST_FEATURES = (
    "host_ecog",
    "host_symptoms",
    "host_renal",
    "host_hepatic",
    "host_cardiac",
    "host_marrow_reserve",
    "host_pulmonary",
    "host_relative_weight_loss",
)
TOXICITY_FEATURES = (
    "host_active_ae_max_grade",
    "host_active_ae_count",
)
CHANNELS = {
    "tumour": TUMOUR_FEATURES,
    "host": GENERAL_HOST_FEATURES,
    "toxicity": TOXICITY_FEATURES,
}

FEATURE_LABELS = {
    "tumour_hepatic": "hepatic disease",
    "tumour_thoracic": "thoracic disease",
    "tumour_nodal": "nodal disease",
    "tumour_osseous": "osseous disease",
    "tumour_marrow": "marrow disease",
    "tumour_cns": "CNS disease",
    "tumour_peritoneal": "peritoneal disease",
    "tumour_soft_tissue_cutaneous": "soft-tissue / skin disease",
    "tumour_other_anatomical": "other anatomical disease",
    "tumour_extent": "disease extent",
    "host_ecog": "ECOG",
    "host_symptoms": "global symptoms",
    "host_renal": "renal function",
    "host_hepatic": "hepatic function",
    "host_cardiac": "cardiac function",
    "host_marrow_reserve": "marrow reserve",
    "host_pulmonary": "pulmonary function",
    "host_relative_weight_loss": "relative weight loss",
    "host_active_ae_max_grade": "maximum active AE grade",
    "host_active_ae_count": "active AE count",
}

PRIMARY_CLASS_NORMALIZATION = {
    "bone_marrow_blood": "bone_marrow",
    "lymphoid_tissue": "lymph_node",
    "mesothelioma": "pleura",
    "soft_tissue_extremity": "soft_tissue",
    "soft_tissue_trunk_chest_wall": "soft_tissue",
    "visceral_soft_tissue": "soft_tissue",
}

EVENT_EXPECTATION: dict[str, tuple[str, int]] = {
    "response": ("tumour", -1),
    "progression": ("tumour", 1),
    "clonal_evolution_event": ("tumour", 1),
    "local_intervention": ("tumour", -1),
    "toxicity": ("toxicity", 1),
    "patient_state_update": ("host", 1),
}
NEGATIVE_CONTROL_EVENTS = (
    "systemic_treatment_start",
    "systemic_treatment_switch",
)
EVENT_ORDER = (*EVENT_EXPECTATION, *NEGATIVE_CONTROL_EVENTS)
EVENT_LABELS = {
    "response": "response",
    "progression": "progression",
    "clonal_evolution_event": "clonal evolution",
    "local_intervention": "local intervention",
    "toxicity": "toxicity",
    "patient_state_update": "patient-state update",
    "systemic_treatment_start": "treatment start",
    "systemic_treatment_switch": "treatment switch",
}

INK = "#172033"
MUTED = "#64748b"
GRID = "#dce4ec"
TUMOUR_COLOR = "#7a5195"
HOST_COLOR = "#168a8a"
TOXICITY_COLOR = "#d98120"
CHANNEL_COLORS = {
    "tumour": TUMOUR_COLOR,
    "host": HOST_COLOR,
    "toxicity": TOXICITY_COLOR,
}
STATE_CMAP = LinearSegmentedColormap.from_list(
    "state_severity",
    ("#eaf4f2", "#5ba7a3", "#7a5195", "#8f2636"),
)


@dataclass(frozen=True)
class WeightedPCAModel:
    """Persistable fixed-range weighted PCA transformation."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    components: np.ndarray
    full_components: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    feature_multipliers: np.ndarray

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Project a state matrix onto the two declared display axes."""
        matrix = frame.loc[:, self.feature_names].to_numpy(float)
        transformed = matrix * self.feature_multipliers
        return (transformed - self.mean) @ self.components.T

    def transform_all(self, frame: pd.DataFrame) -> np.ndarray:
        """Project a state matrix onto the complete stored PCA basis."""
        matrix = frame.loc[:, self.feature_names].to_numpy(float)
        transformed = matrix * self.feature_multipliers
        return (transformed - self.mean) @ self.full_components.T

    def coefficients_in_encoded_units(self) -> np.ndarray:
        """Return PC1/PC2 coefficients multiplying original [0, 1] features."""
        return self.components * self.feature_multipliers[np.newaxis, :]

    def full_coefficients_in_encoded_units(self) -> np.ndarray:
        """Return all PC coefficients multiplying original [0, 1] features."""
        return self.full_components * self.feature_multipliers[np.newaxis, :]

    def as_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable model contract."""
        return {
            "feature_names": list(self.feature_names),
            "mean_after_feature_multipliers": self.mean.tolist(),
            "display_component_count": int(self.components.shape[0]),
            "serialized_component_count": int(self.full_components.shape[0]),
            "components_after_feature_multipliers": self.components.tolist(),
            "coefficients_in_encoded_units": (
                self.coefficients_in_encoded_units().tolist()
            ),
            "all_components_after_feature_multipliers": (
                self.full_components.tolist()
            ),
            "all_coefficients_in_encoded_units": (
                self.full_coefficients_in_encoded_units().tolist()
            ),
            "eigenvalues": self.eigenvalues.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "feature_multipliers": self.feature_multipliers.tolist(),
        }


def _accepted(frame: pd.DataFrame) -> pd.DataFrame:
    if "accepted" not in frame:
        return frame.copy()
    keep = frame["accepted"].astype(str).str.lower().isin({"true", "1", "1.0"})
    return frame.loc[keep].copy()


def _with_keys(frame: pd.DataFrame, run_id: str) -> pd.DataFrame:
    out = frame.copy()
    out["_run"] = run_id
    out["_gid"] = run_id + "::" + out["trajectory_id"].astype(str)
    return out


def load_reference_tables(
    run_ids: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Load accepted derivation tables with collision-safe trajectory keys."""
    filenames = {
        "baselines": "baseline_rows.csv",
        "snapshots": "snapshot_rows.csv",
        "sites": "disease_site_rows.csv",
        "events": "event_rows.csv",
        "endpoints": "endpoint_rows.csv",
    }
    collected: dict[str, list[pd.DataFrame]] = {key: [] for key in filenames}
    for run_id in run_ids:
        derivation = ROOT / "runs" / run_id / "derivation"
        if not derivation.exists():
            raise FileNotFoundError(derivation)
        for key, filename in filenames.items():
            path = derivation / filename
            if not path.exists():
                raise FileNotFoundError(path)
            frame = _accepted(pd.read_csv(path, low_memory=False))
            collected[key].append(_with_keys(frame, run_id))
    return {
        key: pd.concat(frames, ignore_index=True)
        for key, frames in collected.items()
    }


def _ordinal(series: pd.Series, mapping: Mapping[str, float]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    labels = series.astype(str).str.strip().str.lower().map(mapping)
    return numeric.where(numeric.notna(), labels).astype(float)


def normalize_site_class(value: object) -> str:
    """Normalize schema-level synonyms without grouping anatomical regions."""
    raw = str(value or "other").strip().lower()
    return PRIMARY_CLASS_NORMALIZATION.get(raw, raw)


def infer_primary_class_from_label(value: object) -> str:
    """Infer a schema-level primary class from a free-text anatomical label.

    This fallback is used only when the supplied baseline class failed its
    categorical validation. It is deliberately conservative and every use is
    retained in the primary-site resolution audit.
    """
    label = " ".join(
        str(value or "").lower().replace("_", " ").replace("-", " ").split()
    )
    patterns = (
        ("bone_marrow", r"\b(bone marrow|marrow|blood)\b"),
        ("leptomeningeal_cns", r"\b(leptomen|cerebrospinal|csf)\b"),
        ("brain", r"\b(brain|cerebr|cerebell|pituitary|sellar)\b"),
        ("lung", r"\b(lung|bronch|pulmonary)\b"),
        ("pleura", r"\b(pleura|mesothel)\b"),
        ("liver", r"\b(liver|hepatic|hepatocellular)\b"),
        ("lymph_node", r"\b(lymph|mediastin)\b"),
        ("bone", r"\b(bone|osseous|clivus|vertebr|skeleton)\b"),
        ("peritoneum", r"\b(peritone|omentum|mesenter)\b"),
        ("retroperitoneum", r"\bretroperitone\b"),
        ("soft_tissue", r"\b(soft tissue|muscle|back)\b"),
        ("skin", r"\b(skin|cutaneous|scalp|sole|foot|calf)\b"),
        ("colorectal", r"\b(colon|rect|anal|appendix)\b"),
        (
            "upper_gastrointestinal",
            r"\b(esoph|gastr|gastroesophageal|stomach|duoden)\b",
        ),
        ("pancreatobiliary", r"\b(pancrea|biliary|bile duct|gallbladder)\b"),
        ("kidney", r"\b(kidney|renal)\b"),
        ("urinary_tract", r"\b(bladder|ureter|urinary|urothel)\b"),
        ("prostate", r"\bprostate\b"),
        (
            "gynecologic_ovary_fallopian_peritoneal",
            r"\b(ovar|fallopian|adnex)\b",
        ),
        (
            "gynecologic_uterus_cervix",
            r"\b(uter|endometr|cervix|cervical)\b",
        ),
        ("breast", r"\b(breast|mammary)\b"),
        ("thyroid", r"\bthyroid\b"),
        ("head_neck", r"\b(head|neck|oral|tongue|parotid|pharyn|laryn)\b"),
        ("adrenal", r"\badrenal\b"),
        ("thymus", r"\bthym"),
        ("testis", r"\b(testis|testicular)\b"),
        ("penis", r"\bpeni"),
        ("vulva_vagina", r"\b(vulva|vagina)\b"),
        ("eye", r"\b(eye|ocular|uvea|retina)\b"),
    )
    for site_class, pattern in patterns:
        if re.search(pattern, label):
            return site_class
    return "other"


def resolve_primary_site_classes(
    sites: pd.DataFrame,
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve generic ``primary_site`` rows from baseline primary-site context.

    Primary cancer identity remains context. This operation only places the
    *current physical burden* at that primary site into the corresponding
    anatomical coordinate.
    """
    required_site = {
        "_gid",
        "snapshot_index",
        "site_class_canonical",
        "burden_band",
    }
    missing_site = required_site.difference(sites.columns)
    if missing_site:
        raise ValueError(f"disease-site table missing columns: {sorted(missing_site)}")
    if "primary_site_class_canonical" not in baselines:
        raise ValueError("baseline table lacks primary_site_class_canonical")

    if "primary_site" not in baselines:
        raise ValueError("baseline table lacks primary_site")
    primary = (
        baselines[["_gid", "primary_site_class_canonical", "primary_site"]]
        .drop_duplicates()
        .copy()
    )
    if primary["_gid"].duplicated().any():
        raise ValueError("trajectory has multiple baseline primary-site classes")

    out = sites.merge(primary, on="_gid", how="left", validate="many_to_one")
    out["site_class_original"] = (
        out["site_class_canonical"].fillna("other").astype(str).map(normalize_site_class)
    )
    out["primary_site_class_resolved"] = (
        out["primary_site_class_canonical"]
        .fillna("other")
        .astype(str)
        .map(normalize_site_class)
    )
    invalid_primary = out["primary_site_class_resolved"].isin(
        {
            "",
            "nan",
            "primary_site",
            "invalid_primary_site_class",
        }
    )
    label_fallback = out["primary_site"].map(infer_primary_class_from_label)
    out["primary_site_class_resolved"] = out[
        "primary_site_class_resolved"
    ].where(~invalid_primary, label_fallback)
    generic = out["site_class_original"].eq("primary_site")
    resolved_primary = out["primary_site_class_resolved"].where(
        ~out["primary_site_class_resolved"].isin({"", "nan", "primary_site"}),
        "other",
    )
    out["site_class_resolved"] = out["site_class_original"].where(
        ~generic,
        resolved_primary,
    )
    out["primary_resolution_source"] = np.select(
        [
            ~generic,
            generic & ~invalid_primary & out["site_class_resolved"].ne("other"),
            generic & invalid_primary & out["site_class_resolved"].ne("other"),
        ],
        [
            "direct_site_class",
            "baseline_primary_context",
            "baseline_primary_label_fallback",
        ],
        default="unresolved_primary_context",
    )
    return out


def anatomical_group(site_class: object) -> str:
    """Map one resolved physical class to one of nine display coordinates."""
    normalized = normalize_site_class(site_class)
    for feature, members in ANATOMICAL_GROUPS.items():
        if normalized in members:
            return feature
    return OTHER_ANATOMICAL_FEATURE


def build_grouped_anatomy(
    sites: pd.DataFrame,
    baselines: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build nine grouped burden coordinates and capped original-class extent."""
    resolved = resolve_primary_site_classes(sites, baselines)
    resolved["burden_rank"] = _ordinal(resolved["burden_band"], BURDEN_RANK)
    resolved["anatomical_group"] = resolved["site_class_resolved"].map(
        anatomical_group
    )
    active = resolved[resolved["burden_rank"].gt(0)].copy()
    keys = ["_gid", "snapshot_index"]

    if active.empty:
        columns = [*keys, *ANATOMICAL_GROUPS, OTHER_ANATOMICAL_FEATURE, EXTENT_FEATURE]
        return pd.DataFrame(columns=columns), resolved

    grouped = active.pivot_table(
        index=keys,
        columns="anatomical_group",
        values="burden_rank",
        aggfunc="max",
        fill_value=0.0,
    )
    grouped.columns.name = None
    grouped = grouped.reset_index()
    for feature in (*ANATOMICAL_GROUPS, OTHER_ANATOMICAL_FEATURE):
        if feature not in grouped:
            grouped[feature] = 0.0
        grouped[feature] = grouped[feature].astype(float) / 5.0

    extent = (
        active.groupby(keys)["site_class_resolved"]
        .nunique()
        .clip(upper=5)
        .div(5.0)
        .rename(EXTENT_FEATURE)
        .reset_index()
    )
    grouped = grouped.merge(extent, on=keys, how="left", validate="one_to_one")
    columns = [*keys, *ANATOMICAL_GROUPS, OTHER_ANATOMICAL_FEATURE, EXTENT_FEATURE]
    return grouped.loc[:, columns], resolved


def encode_weight_loss(
    current_weight: pd.Series,
    baseline_weight: pd.Series,
) -> pd.Series:
    """Encode signed relative weight loss over the fixed [-20%, +30%] range."""
    current = pd.to_numeric(current_weight, errors="coerce")
    baseline = pd.to_numeric(baseline_weight, errors="coerce")
    valid = baseline.gt(0)
    loss_pct = pd.Series(np.nan, index=current.index, dtype=float)
    loss_pct.loc[valid] = (
        100.0
        * (baseline.loc[valid] - current.loc[valid])
        / baseline.loc[valid]
    )
    return (loss_pct.clip(lower=-20.0, upper=30.0) + 20.0) / 50.0


def build_state_matrix(
    tables: Mapping[str, pd.DataFrame],
    active_conditions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the frozen 20-coordinate living visit-state matrix."""
    snapshots = tables["snapshots"].copy()
    terminal = snapshots["vital_status"].astype(str).str.lower().eq("deceased")
    terminal |= snapshots["outcome_type"].astype(str).str.lower().eq("death")
    state = snapshots.loc[~terminal].copy()
    state["snapshot_index"] = pd.to_numeric(
        state["snapshot_index"], errors="raise"
    ).astype(int)

    anatomy, resolved_sites = build_grouped_anatomy(
        tables["sites"],
        tables["baselines"],
    )
    state = state.merge(
        anatomy,
        on=["_gid", "snapshot_index"],
        how="left",
        validate="one_to_one",
    )
    for feature in TUMOUR_FEATURES:
        state[feature] = pd.to_numeric(state[feature], errors="coerce").fillna(0.0)

    if active_conditions.empty:
        raise ValueError("active-condition materialization is empty")
    state = state.merge(
        active_conditions[
            [
                "_gid",
                "snapshot_index",
                "active_condition_count",
                "active_condition_grade_max",
            ]
        ],
        on=["_gid", "snapshot_index"],
        how="left",
        validate="one_to_one",
    )
    if state[["active_condition_count", "active_condition_grade_max"]].isna().any().any():
        raise ValueError("active-condition materialization does not cover every living state")

    state["host_ecog"] = pd.to_numeric(state["ecog_ps"], errors="coerce") / 4.0
    state["host_symptoms"] = _ordinal(
        state["symptom_burden"],
        SYMPTOM_RANK,
    ) / 3.0
    for organ in ORGAN_FIELDS:
        state[f"host_{organ}"] = _ordinal(
            state[f"organ_{organ}"],
            ORGAN_RANK,
        ) / 4.0

    ordered = state.sort_values(["_gid", "snapshot_index"])
    baseline_weight = (
        ordered.groupby("_gid", sort=False)["weight_kg"]
        .first()
        .rename("_baseline_weight")
    )
    state = state.join(baseline_weight, on="_gid")
    state["host_relative_weight_loss"] = encode_weight_loss(
        state["weight_kg"],
        state["_baseline_weight"],
    )
    state["host_active_ae_max_grade"] = (
        pd.to_numeric(state["active_condition_grade_max"], errors="coerce")
        .clip(lower=0, upper=5)
        .div(5.0)
    )
    state["host_active_ae_count"] = (
        pd.to_numeric(state["active_condition_count"], errors="coerce")
        .clip(lower=0, upper=5)
        .div(5.0)
    )

    matrix = state.loc[:, STATE_FEATURES]
    if matrix.isna().any().any():
        missing = matrix.columns[matrix.isna().any()].tolist()
        raise ValueError(f"state contract contains missing values: {missing}")
    minimum = float(matrix.min().min())
    maximum = float(matrix.max().max())
    if minimum < -1e-12 or maximum > 1.0 + 1e-12:
        raise ValueError(
            f"state contract is outside [0, 1]: minimum={minimum}, maximum={maximum}"
        )
    return state, resolved_sites


def equal_model_trajectory_snapshot_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give equal mass to models, trajectories, then snapshots within trajectories."""
    if frame.empty:
        raise ValueError("cannot weight an empty state matrix")
    model_by_gid = frame[["_gid", "model_short"]].drop_duplicates()
    if model_by_gid["_gid"].duplicated().any():
        raise ValueError("one trajectory appears under multiple models")
    n_models = int(model_by_gid["model_short"].nunique())
    trajectories_per_model = model_by_gid.groupby("model_short")["_gid"].nunique()
    snapshots_per_gid = frame.groupby("_gid").size()
    model_mass = frame["model_short"].map(
        1.0 / (n_models * trajectories_per_model)
    )
    snapshot_mass = frame["_gid"].map(1.0 / snapshots_per_gid)
    weights = (model_mass * snapshot_mass).to_numpy(float)
    weights /= weights.sum()
    return weights


def fit_weighted_pca(
    frame: pd.DataFrame,
    weights: np.ndarray,
    *,
    feature_multipliers: np.ndarray | None = None,
) -> tuple[WeightedPCAModel, np.ndarray]:
    """Fit the fixed-range weighted PCA by eigendecomposition."""
    matrix = frame.loc[:, STATE_FEATURES].to_numpy(float)
    w = np.asarray(weights, dtype=float)
    if matrix.shape[0] != w.shape[0]:
        raise ValueError("one observation weight is required per state")
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("observation weights must be non-negative and sum to one")
    multipliers = (
        np.ones(matrix.shape[1], dtype=float)
        if feature_multipliers is None
        else np.asarray(feature_multipliers, dtype=float)
    )
    if multipliers.shape != (matrix.shape[1],) or np.any(multipliers <= 0):
        raise ValueError("feature multipliers must be positive and match state features")

    transformed = matrix * multipliers
    mean = np.average(transformed, axis=0, weights=w)
    centered = transformed - mean
    covariance = centered.T @ (centered * w[:, np.newaxis])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order].T
    total = float(eigenvalues.sum())
    explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)

    # All encoded coordinates increase with worsening current state. Orient PC1
    # towards global worsening, then orient PC2 towards worsening host state.
    if float(components[0].sum()) < 0:
        components[0] *= -1
    host_indices = [STATE_FEATURES.index(feature) for feature in HOST_FEATURES]
    if float(components[1, host_indices].sum()) < 0:
        components[1] *= -1
    # Later axes have no clinical orientation rule. Fix their otherwise
    # arbitrary signs by requiring the largest absolute loading to be positive,
    # so PC3-PC20 can be reproduced directly from the serialized transform.
    for component_index in range(2, components.shape[0]):
        pivot = int(np.argmax(np.abs(components[component_index])))
        if components[component_index, pivot] < 0:
            components[component_index] *= -1

    model = WeightedPCAModel(
        feature_names=STATE_FEATURES,
        mean=mean,
        components=components[:2],
        full_components=components,
        eigenvalues=eigenvalues,
        explained_variance_ratio=explained,
        feature_multipliers=multipliers,
    )
    return model, model.transform(frame)


def block_balanced_multipliers(
    frame: pd.DataFrame,
    weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Equalise tumour and host total fixed-range variance for sensitivity only."""
    matrix = frame.loc[:, STATE_FEATURES].to_numpy(float)
    mean = np.average(matrix, axis=0, weights=weights)
    variance = np.average((matrix - mean) ** 2, axis=0, weights=weights)
    tumour_trace = float(variance[: len(TUMOUR_FEATURES)].sum())
    host_trace = float(variance[len(TUMOUR_FEATURES) :].sum())
    if tumour_trace <= 0 or host_trace <= 0:
        raise ValueError("both state blocks need positive variance")
    target = 0.5 * (tumour_trace + host_trace)
    tumour_scale = float(np.sqrt(target / tumour_trace))
    host_scale = float(np.sqrt(target / host_trace))
    multipliers = np.r_[
        np.full(len(TUMOUR_FEATURES), tumour_scale),
        np.full(len(HOST_FEATURES), host_scale),
    ]
    return multipliers, {
        "tumour_total_variance": tumour_trace,
        "host_total_variance": host_trace,
        "tumour_multiplier": tumour_scale,
        "host_multiplier": host_scale,
    }


def align_sensitivity_scores(
    primary_scores: np.ndarray,
    sensitivity_scores: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Orthogonally align a two-dimensional sensitivity map to the primary map."""
    primary = np.asarray(primary_scores, dtype=float)
    sensitivity = np.asarray(sensitivity_scores, dtype=float)
    primary_centered = primary - primary.mean(axis=0)
    sensitivity_centered = sensitivity - sensitivity.mean(axis=0)
    left, _singular, right_t = np.linalg.svd(
        sensitivity_centered.T @ primary_centered
    )
    rotation = left @ right_t
    aligned = sensitivity_centered @ rotation + primary.mean(axis=0)
    residual = float(np.square(primary - aligned).sum())
    total = float(np.square(primary - primary.mean(axis=0)).sum())
    return aligned, {
        "aligned_r2": 1.0 - residual / total if total > 0 else np.nan,
        "pc1_correlation": float(np.corrcoef(primary[:, 0], aligned[:, 0])[0, 1]),
        "pc2_correlation": float(np.corrcoef(primary[:, 1], aligned[:, 1])[0, 1]),
    }


def feature_diagnostics(
    state: pd.DataFrame,
    weights: np.ndarray,
    model: WeightedPCAModel,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify realised prevalence, variance and PC contributions."""
    matrix = state.loc[:, STATE_FEATURES].to_numpy(float)
    weighted_mean = np.average(matrix, axis=0, weights=weights)
    weighted_variance = np.average(
        (matrix - weighted_mean) ** 2,
        axis=0,
        weights=weights,
    )
    coefficients = model.coefficients_in_encoded_units()
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(STATE_FEATURES):
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "block": "tumour" if feature in TUMOUR_FEATURES else "host",
                "weighted_mean": weighted_mean[index],
                "weighted_variance": weighted_variance[index],
                "nonzero_pct": 100.0 * float(state[feature].gt(0).mean()),
                "pc1_coefficient": coefficients[0, index],
                "pc2_coefficient": coefficients[1, index],
                "pc1_squared_share": (
                    coefficients[0, index] ** 2
                    / np.square(coefficients[0]).sum()
                ),
                "pc2_squared_share": (
                    coefficients[1, index] ** 2
                    / np.square(coefficients[1]).sum()
                ),
            }
        )
    features = pd.DataFrame(rows)
    blocks = (
        features.groupby("block", as_index=False)
        .agg(
            total_weighted_variance=("weighted_variance", "sum"),
            pc1_squared_share=("pc1_squared_share", "sum"),
            pc2_squared_share=("pc2_squared_share", "sum"),
        )
        .sort_values("block")
    )
    return features, blocks


def add_channel_scores(state: pd.DataFrame) -> pd.DataFrame:
    """Add raw fixed-range validation channels; these are not PCA axes."""
    out = state.copy()
    for channel, features in CHANNELS.items():
        out[f"channel_{channel}"] = out.loc[:, features].mean(axis=1)
    return out


def event_transitions(
    state: pd.DataFrame,
    events: pd.DataFrame,
    *,
    post_months: float = POST_EVENT_MONTHS,
) -> pd.DataFrame:
    """Measure the last state before to first state at/after each named event."""
    scored = add_channel_scores(state)
    channel_columns = [f"channel_{channel}" for channel in CHANNELS]
    courses = {
        gid: group.sort_values(["days_from_diagnosis", "snapshot_index"])
        for gid, group in scored.groupby("_gid", sort=False)
    }
    rows: list[dict[str, Any]] = []
    selected = events[events["event_type"].astype(str).isin(EVENT_ORDER)]
    for gid, event_group in selected.groupby("_gid", sort=False):
        course = courses.get(gid)
        if course is None or len(course) < 2:
            continue
        days = pd.to_numeric(
            course["days_from_diagnosis"], errors="coerce"
        ).to_numpy(float)
        values = course.loc[:, channel_columns].to_numpy(float)
        for event in event_group.itertuples(index=False):
            event_day = pd.to_numeric(
                pd.Series([event.days_from_diagnosis]),
                errors="coerce",
            ).iloc[0]
            if pd.isna(event_day):
                continue
            before = np.flatnonzero(days < float(event_day) - 1e-9)
            after = np.flatnonzero(
                (days >= float(event_day) - 1e-9)
                & (
                    days
                    <= float(event_day)
                    + post_months * MONTH_DAYS
                    + 1e-9
                )
            )
            if not len(before) or not len(after):
                continue
            delta = values[after[0]] - values[before[-1]]
            row: dict[str, Any] = {
                "_gid": gid,
                "event_type": str(event.event_type),
            }
            row.update(
                {
                    f"delta_{channel}": float(delta[index])
                    for index, channel in enumerate(CHANNELS)
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_event_validation(transitions: pd.DataFrame) -> pd.DataFrame:
    """Summarise prespecified event and negative-control coherence checks."""
    rows: list[dict[str, Any]] = []
    for event_type in EVENT_ORDER:
        group = transitions[transitions["event_type"].eq(event_type)]
        if group.empty:
            continue
        means = {
            channel: float(group[f"delta_{channel}"].mean())
            for channel in CHANNELS
        }
        largest = max(means, key=lambda key: abs(means[key]))
        row: dict[str, Any] = {
            "event_type": event_type,
            "event_label": EVENT_LABELS[event_type],
            "n": int(len(group)),
            **{f"mean_delta_{channel}": value for channel, value in means.items()},
            "largest_channel": largest,
        }
        if event_type in EVENT_EXPECTATION:
            expected_channel, expected_sign = EVENT_EXPECTATION[event_type]
            alternatives = [
                abs(value)
                for channel, value in means.items()
                if channel != expected_channel
            ]
            denominator = max(alternatives, default=0.0)
            specificity = (
                abs(means[expected_channel]) / denominator
                if denominator > 0
                else np.inf
            )
            passed = (
                largest == expected_channel
                and int(np.sign(means[expected_channel])) == expected_sign
            )
            row.update(
                {
                    "expected": (
                        f"{expected_channel} "
                        f"{'worse' if expected_sign > 0 else 'better'}"
                    ),
                    "specificity": specificity,
                    "passed": passed,
                }
            )
        else:
            maximum = max(abs(value) for value in means.values())
            row.update(
                {
                    "expected": "no instantaneous state movement",
                    "specificity": np.nan,
                    "passed": maximum < 0.01,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def locf_at_day(course: pd.DataFrame, day: float) -> pd.Series:
    """Return the last recorded state at or before a landmark."""
    ordered = (
        course.sort_values(["days_from_diagnosis", "snapshot_index"])
        .drop_duplicates("days_from_diagnosis", keep="last")
        .reset_index(drop=True)
    )
    observed = ordered[
        pd.to_numeric(ordered["days_from_diagnosis"], errors="coerce") <= day
    ]
    return ordered.iloc[0] if observed.empty else observed.iloc[-1]


def decision_cutoffs(
    run_id: str,
    trajectory_ids: Sequence[str],
) -> dict[str, float]:
    """Read exact decision cutoffs from persisted conditioning specifications."""
    values: dict[str, float] = {}
    run_root = ROOT / "runs" / run_id
    for trajectory_id in sorted(set(str(value) for value in trajectory_ids)):
        paths = sorted(
            (run_root / "trajectories" / trajectory_id).glob(
                "attempt_*/conditioning_spec.json"
            )
        )
        if not paths:
            raise FileNotFoundError(
                f"No conditioning specification found for {trajectory_id}"
            )
        spec = json.loads(paths[0].read_text())
        diagnosis = date.fromisoformat(
            spec["baseline_seed"]["identity_card"]["diagnosis"]["diagnosis_date"]
        )
        cutoff = date.fromisoformat(spec["observed_prefix"]["cutoff_date"])
        values[trajectory_id] = float((cutoff - diagnosis).days)
    return values


def phase1_p0_records(
    phase1_state: pd.DataFrame,
    raw_snapshots: pd.DataFrame,
    *,
    run_id: str = PHASE1_RUN,
    horizon_months: float = TAU_MONTHS,
) -> pd.DataFrame:
    """Measure no-change, jump and EXIT from each exact treatment anchor."""
    state = phase1_state.copy()
    parts = state["condition_id"].astype(str).str.split("__", expand=True)
    if parts.shape[1] < 3:
        raise ValueError("Phase-1 condition_id lacks context/state/history fields")
    state["context"] = parts[0]
    state["state_id_design"] = parts[1]
    state["history"] = parts[2]
    state["arm"] = state["policy_id"].astype(str).str.split("__").str[-1]
    cutoffs = decision_cutoffs(run_id, state["trajectory_id"].unique())

    terminal = raw_snapshots.copy()
    terminal_mask = terminal["vital_status"].astype(str).str.lower().eq("deceased")
    terminal_mask |= terminal["outcome_type"].astype(str).str.lower().eq("death")
    terminal_days = (
        terminal.loc[terminal_mask]
        .groupby("_gid")["days_from_diagnosis"]
        .min()
        .to_dict()
    )

    rows: list[dict[str, Any]] = []
    for gid, course in state.groupby("_gid", sort=False):
        trajectory_id = str(course["trajectory_id"].iloc[0])
        cutoff = cutoffs[trajectory_id]
        target = cutoff + horizon_months * MONTH_DAYS
        anchor = locf_at_day(course, cutoff)
        death_day = terminal_days.get(gid, np.inf)
        exited = bool(float(death_day) <= target)
        if exited:
            unchanged = False
        else:
            future = locf_at_day(course, target)
            unchanged = bool(
                np.allclose(
                    anchor.loc[list(STATE_FEATURES)].to_numpy(float),
                    future.loc[list(STATE_FEATURES)].to_numpy(float),
                    atol=1e-12,
                    rtol=0.0,
                )
            )
        rows.append(
            {
                "_gid": gid,
                "trajectory_id": trajectory_id,
                "context": str(course["context"].iloc[0]),
                "state": str(course["state_id_design"].iloc[0]),
                "history": str(course["history"].iloc[0]),
                "arm": str(course["arm"].iloc[0]),
                "cutoff_days": cutoff,
                "horizon_months": horizon_months,
                "no_encoded_change": unchanged,
                "jumped": bool(not unchanged and not exited),
                "exited": exited,
            }
        )
    return pd.DataFrame(rows)


def summarize_phase1_p0(records: pd.DataFrame) -> pd.DataFrame:
    """Summarise the three mutually exclusive anchor-to-horizon outcomes."""
    return (
        records.groupby(["context", "arm"], as_index=False)
        .agg(
            n=("trajectory_id", "size"),
            p0=("no_encoded_change", "mean"),
            jump_probability=("jumped", "mean"),
            exit_probability=("exited", "mean"),
        )
        .sort_values(["context", "arm"])
    )


def primary_site_resolution_audit(resolved_sites: pd.DataFrame) -> pd.DataFrame:
    """Report where generic primary-site burden records were placed."""
    generic = resolved_sites[
        resolved_sites["site_class_original"].eq("primary_site")
    ].copy()
    if generic.empty:
        return pd.DataFrame(
            columns=[
                "baseline_primary_class",
                "resolved_site_class",
                "anatomical_group",
                "resolution_source",
                "n_rows",
                "n_trajectories",
            ]
        )
    generic["anatomical_group"] = generic["site_class_resolved"].map(
        anatomical_group
    )
    return (
        generic.groupby(
            [
                "primary_site_class_resolved",
                "site_class_resolved",
                "anatomical_group",
                "primary_resolution_source",
            ],
            as_index=False,
        )
        .agg(n_rows=("_gid", "size"), n_trajectories=("_gid", "nunique"))
        .rename(
            columns={
                "primary_site_class_resolved": "baseline_primary_class",
                "primary_resolution_source": "resolution_source",
            }
        )
        .sort_values(["n_rows", "baseline_primary_class"], ascending=[False, True])
    )


def draw_map_contract(
    state: pd.DataFrame,
    features: pd.DataFrame,
    blocks: pd.DataFrame,
    model: WeightedPCAModel,
    sensitivity: Mapping[str, float],
    out_dir: Path,
) -> None:
    """Draw the map, loadings and realised influence audit."""
    sample = state
    if len(sample) > 24000:
        sample = sample.sample(24000, random_state=20260730)

    figure = plt.figure(figsize=(15.5, 9.2))
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1.45, 1.0, 0.8),
        height_ratios=(1.0, 0.9),
        hspace=0.36,
        wspace=0.34,
    )
    axis = figure.add_subplot(grid[:, 0])
    severity = sample.loc[:, STATE_FEATURES].mean(axis=1)
    axis.scatter(
        sample["pc1"],
        sample["pc2"],
        c=severity,
        cmap=STATE_CMAP,
        s=3.5,
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    axis.set_xlabel(
        f"PC1 ({100 * model.explained_variance_ratio[0]:.1f}% variance)",
        fontsize=10,
    )
    axis.set_ylabel(
        f"PC2 ({100 * model.explained_variance_ratio[1]:.1f}% variance)",
        fontsize=10,
    )
    axis.set_title(
        "A · one common map of current living state\n"
        "10 tumour + 10 host coordinates; colour = mean encoded burden",
        loc="left",
        fontsize=11,
        color=INK,
    )

    axis = figure.add_subplot(grid[0, 1])
    load = features.set_index("feature").loc[list(STATE_FEATURES)]
    image = axis.imshow(
        load[["pc1_coefficient", "pc2_coefficient"]].to_numpy(float),
        cmap="RdBu_r",
        vmin=-float(
            np.abs(
                load[["pc1_coefficient", "pc2_coefficient"]].to_numpy(float)
            ).max()
        ),
        vmax=float(
            np.abs(
                load[["pc1_coefficient", "pc2_coefficient"]].to_numpy(float)
            ).max()
        ),
        aspect="auto",
    )
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["PC1", "PC2"])
    axis.set_yticks(np.arange(len(load)))
    axis.set_yticklabels(load["label"], fontsize=7.2)
    axis.axhline(len(TUMOUR_FEATURES) - 0.5, color=INK, lw=1.2)
    axis.set_title(
        "B · what the axes learned\ncoefficients in encoded clinical units",
        loc="left",
        fontsize=10.5,
        color=INK,
    )
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)

    axis = figure.add_subplot(grid[1, 1])
    ordered = features.sort_values("weighted_variance")
    colors = [
        TUMOUR_COLOR if block == "tumour" else HOST_COLOR
        for block in ordered["block"]
    ]
    axis.barh(
        np.arange(len(ordered)),
        ordered["weighted_variance"],
        color=colors,
    )
    axis.set_yticks(np.arange(len(ordered)))
    axis.set_yticklabels(ordered["label"], fontsize=7.0)
    axis.set_xlabel("weighted variance after fixed-range encoding", fontsize=8.5)
    axis.set_title(
        "C · realised influence reflects prevalence\nreported, not variance-standardised",
        loc="left",
        fontsize=10.5,
        color=INK,
    )

    axis = figure.add_subplot(grid[0, 2])
    display = blocks.set_index("block").loc[["tumour", "host"]]
    x = np.arange(2)
    axis.bar(
        x - 0.17,
        display["pc1_squared_share"],
        width=0.34,
        color="#496a9f",
        label="PC1",
    )
    axis.bar(
        x + 0.17,
        display["pc2_squared_share"],
        width=0.34,
        color="#c8533c",
        label="PC2",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(["tumour", "host"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("squared coefficient share", fontsize=8.5)
    axis.legend(frameon=False, fontsize=8)
    axis.set_title(
        "D · realised block shares\n10 coordinates each",
        loc="left",
        fontsize=10.5,
        color=INK,
    )

    axis = figure.add_subplot(grid[1, 2])
    axis.axis("off")
    axis.text(
        0.0,
        0.96,
        "Sensitivity map",
        fontsize=10.5,
        weight="bold",
        color=INK,
        va="top",
    )
    axis.text(
        0.0,
        0.79,
        "Only the alternative map equalises\n"
        "tumour and host total variance.",
        fontsize=8.8,
        color=MUTED,
        va="top",
        linespacing=1.35,
    )
    axis.text(
        0.0,
        0.50,
        f"aligned 2D R²   {sensitivity['aligned_r2']:.3f}\n"
        f"PC1 correlation   {sensitivity['pc1_correlation']:.3f}\n"
        f"PC2 correlation   {sensitivity['pc2_correlation']:.3f}",
        fontsize=10,
        color=INK,
        va="top",
        linespacing=1.55,
    )
    axis.text(
        0.0,
        0.15,
        "Raw clinical estimands do not depend\non either display map.",
        fontsize=8.8,
        color=MUTED,
        va="top",
    )

    for axis in figure.axes:
        if axis.get_label() != "<colorbar>":
            axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "The paper-v1 clinical state map — a fixed display, not the analytic patient",
        fontsize=14,
        color=INK,
        y=0.99,
    )
    figure.savefig(
        out_dir / "fig_state_map_contract.png",
        dpi=210,
        bbox_inches="tight",
    )
    figure.savefig(
        out_dir / "fig_state_map_contract.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def draw_validation(
    event_summary: pd.DataFrame,
    p0_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Draw raw event-channel checks and the exact-anchor p0 readout."""
    figure = plt.figure(figsize=(15.0, 6.4))
    grid = figure.add_gridspec(1, 2, width_ratios=(1.55, 1.0), wspace=0.30)

    axis = figure.add_subplot(grid[0, 0])
    ordered = event_summary.set_index("event_type").reindex(EVENT_ORDER).dropna(how="all")
    positions = np.arange(len(ordered))
    width = 0.25
    for index, channel in enumerate(CHANNELS):
        axis.bar(
            positions + (index - 1) * width,
            ordered[f"mean_delta_{channel}"],
            width=width,
            color=CHANNEL_COLORS[channel],
            label=channel,
        )
    axis.axhline(0, color=INK, lw=1)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            f"{EVENT_LABELS[event]}\n(n={int(ordered.loc[event, 'n']):,})"
            for event in ordered.index
        ],
        rotation=25,
        ha="right",
        fontsize=8,
    )
    axis.set_ylabel("mean fixed-range channel change across event", fontsize=9)
    axis.legend(frameon=False, ncols=3, fontsize=8.5)
    passed = int(ordered["passed"].astype(bool).sum())
    axis.set_title(
        "A · prespecified event and negative-control gate\n"
        f"{passed}/{len(ordered)} checks pass; bars use raw state channels, not PCs",
        loc="left",
        fontsize=11,
        color=INK,
    )

    axis = figure.add_subplot(grid[0, 1])
    arms = ["supportive", "chemo", "osimertinib", "pembrolizumab"]
    arm_labels = {
        "supportive": "BSC",
        "chemo": "chemo",
        "osimertinib": "osi",
        "pembrolizumab": "pembro",
    }
    context_labels = {
        "EGFR_1L": "EGFR-mutant",
        "EGFR_WT_1L": "EGFR-wild-type",
    }
    contexts = list(p0_summary["context"].drop_duplicates())
    x_positions: list[float] = []
    labels: list[str] = []
    context_centres: list[tuple[float, str]] = []
    cursor = 0.0
    for context in contexts:
        subset = p0_summary[p0_summary["context"].eq(context)].set_index("arm")
        context_positions: list[float] = []
        for arm in arms:
            if arm not in subset.index:
                continue
            row = subset.loc[arm]
            axis.bar(
                cursor,
                row["p0"],
                color="#168a8a",
                width=0.72,
                label="no encoded change" if not x_positions else None,
            )
            axis.bar(
                cursor,
                row["jump_probability"],
                bottom=row["p0"],
                color="#7a5195",
                width=0.72,
                label="state jump" if not x_positions else None,
            )
            axis.bar(
                cursor,
                row["exit_probability"],
                bottom=row["p0"] + row["jump_probability"],
                color="#8f2636",
                width=0.72,
                label="EXIT" if not x_positions else None,
            )
            x_positions.append(cursor)
            context_positions.append(cursor)
            labels.append(arm_labels.get(arm, arm))
            cursor += 1.0
        if context_positions:
            context_centres.append(
                (float(np.mean(context_positions)), context_labels.get(context, context))
            )
        cursor += 0.45
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=22, ha="right", fontsize=7.5)
    for centre, label in context_centres:
        axis.text(
            centre,
            -0.23,
            label,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color=INK,
        )
    axis.set_ylim(0, 1)
    axis.set_ylabel("probability by 3 months", fontsize=9)
    axis.legend(
        frameon=False,
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    axis.set_title(
        "B · exact decision anchor, LOCF at 3 months\n"
        r"$p_0=P\{X(t+3\,mo)=X(t)\}$; 45 futures per arm",
        loc="left",
        fontsize=11,
        color=INK,
    )

    for axis in figure.axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color=GRID, lw=0.65, alpha=0.8)
        axis.set_axisbelow(True)
    figure.suptitle(
        "The state map passes clinical-semantic checks before any outcome analysis",
        fontsize=13.5,
        color=INK,
        y=1.01,
    )
    figure.savefig(
        out_dir / "fig_state_map_validation.png",
        dpi=210,
        bbox_inches="tight",
    )
    figure.savefig(
        out_dir / "fig_state_map_validation.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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


def write_readout(
    out_dir: Path,
    *,
    state: pd.DataFrame,
    model: WeightedPCAModel,
    features: pd.DataFrame,
    blocks: pd.DataFrame,
    event_summary: pd.DataFrame,
    p0_summary: pd.DataFrame,
    resolution: pd.DataFrame,
    sensitivity: Mapping[str, float],
) -> None:
    """Write a clinician-facing contract and validation summary."""
    largest_pc1 = features.loc[features["pc1_squared_share"].idxmax()]
    largest_pc2 = features.loc[features["pc2_squared_share"].idxmax()]
    event_passes = int(event_summary["passed"].astype(bool).sum())
    resolved_sources = {
        "baseline_primary_context",
        "baseline_primary_label_fallback",
    }
    resolution_pct = (
        100.0
        * float(
            resolution.loc[
                resolution["resolution_source"].isin(resolved_sources),
                "n_rows",
            ].sum()
        )
        / float(resolution["n_rows"].sum())
        if len(resolution) and float(resolution["n_rows"].sum()) > 0
        else np.nan
    )
    fallback_pct = (
        100.0
        * float(
            resolution.loc[
                resolution["resolution_source"].eq(
                    "baseline_primary_label_fallback"
                ),
                "n_rows",
            ].sum()
        )
        / float(resolution["n_rows"].sum())
        if len(resolution) and float(resolution["n_rows"].sum()) > 0
        else np.nan
    )

    lines = [
        "# Frozen paper-v1 clinical state map",
        "",
        "## What was fitted",
        "",
        f"- {len(state):,} living event snapshots from "
        f"{state['_gid'].nunique():,} trajectories and "
        f"{state['model_short'].nunique()} generators.",
        "- Twenty contemporaneous visit-state coordinates: ten tumour and ten host.",
        "- Every coordinate uses a fixed clinical range in [0, 1]. PCA mean-centres "
        "these values but does not divide by their empirical standard deviations.",
        "- Observation weights give equal mass to generators, trajectories within "
        "generators, and snapshots within trajectories.",
        "- Treatment, outcomes, history, tumour identity and generator identity are "
        "not PCA inputs. Death is EXIT outside the living-state map.",
        "",
        "## Frozen coordinates",
        "",
        "| coordinate | block | encoding |",
        "|---|---|---|",
    ]
    for feature in STATE_FEATURES:
        if feature.startswith("tumour_") and feature != EXTENT_FEATURE:
            encoding = "maximum grouped burden grade / 5"
        elif feature == EXTENT_FEATURE:
            encoding = "min(unique resolved anatomical classes, 5) / 5"
        elif feature == "host_ecog":
            encoding = "ECOG / 4"
        elif feature == "host_symptoms":
            encoding = "global symptom rank / 3"
        elif feature.startswith("host_") and feature in {
            "host_renal",
            "host_hepatic",
            "host_cardiac",
            "host_marrow_reserve",
            "host_pulmonary",
        }:
            encoding = "organ-impairment grade / 4"
        elif feature == "host_relative_weight_loss":
            encoding = "relative loss clipped to [-20%, +30%], then mapped to [0, 1]"
        elif feature == "host_active_ae_max_grade":
            encoding = "maximum active CTCAE grade / 5"
        else:
            encoding = "min(active AE count, 5) / 5"
        block = "tumour" if feature in TUMOUR_FEATURES else "host"
        lines.append(f"| {FEATURE_LABELS[feature]} | {block} | {encoding} |")

    lines += [
        "",
        "## PCA audit",
        "",
        f"- PC1 explains {100 * model.explained_variance_ratio[0]:.2f}% and PC2 "
        f"{100 * model.explained_variance_ratio[1]:.2f}% of weighted fixed-range "
        "variance.",
        f"- Largest PC1 squared coefficient share: {largest_pc1['label']} "
        f"({100 * largest_pc1['pc1_squared_share']:.1f}%).",
        f"- Largest PC2 squared coefficient share: {largest_pc2['label']} "
        f"({100 * largest_pc2['pc2_squared_share']:.1f}%).",
        "",
        "| block | total fixed-range variance | PC1 squared share | PC2 squared share |",
        "|---|---:|---:|---:|",
    ]
    for row in blocks.itertuples(index=False):
        lines.append(
            f"| {row.block} | {row.total_weighted_variance:.4f} | "
            f"{100 * row.pc1_squared_share:.1f}% | "
            f"{100 * row.pc2_squared_share:.1f}% |"
        )
    lines += [
        "",
        "The primary map is not block-balanced. Equal-block balancing is only a "
        "sensitivity analysis.",
        "",
        f"- Sensitivity-map aligned 2D R²: {sensitivity['aligned_r2']:.3f}.",
        f"- Aligned PC correlations: {sensitivity['pc1_correlation']:.3f}, "
        f"{sensitivity['pc2_correlation']:.3f}.",
        "",
        "## Primary-site placement",
        "",
        "Primary cancer identity remains context. Current physical burden at the "
        "primary site is resolved using baseline primary-site class before anatomical "
        "grouping. Duplicate generic and named records in the same resolved class are "
        "collapsed by maximum burden and count once in extent.",
        "",
        f"{resolution_pct:.1f}% of generic primary-site rows were resolved in "
        "total, including "
        f"{fallback_pct:.1f}% that required the audited free-text label fallback.",
        "",
        "## Event and negative-control gate",
        "",
        f"**{event_passes}/{len(event_summary)} prespecified checks pass.** These are "
        "coherence checks on raw fixed-range tumour, general-host and toxicity "
        "channels; they are not independent evidence that the generated biology is "
        "correct.",
        "",
        "| event | Δtumour | Δhost | Δtoxicity | n | expected | specificity | verdict |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in event_summary.itertuples(index=False):
        specificity = (
            "—" if pd.isna(row.specificity) else f"{row.specificity:.1f}×"
        )
        lines.append(
            f"| {row.event_label} | {row.mean_delta_tumour:+.4f} | "
            f"{row.mean_delta_host:+.4f} | {row.mean_delta_toxicity:+.4f} | "
            f"{row.n:,} | {row.expected} | {specificity} | "
            f"{'PASS' if row.passed else 'FAIL'} |"
        )
    lines += [
        "",
        "## Exact-anchor no-change mass",
        "",
        r"`p0 = P{X(t+3 months)=X(t)}` under LOCF in the frozen encoded state. "
        "It means no encoded event-driven state change, not absence of unrecorded "
        "biological change.",
        "",
        "| context | arm | n | p0 | jump | EXIT |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in p0_summary.itertuples(index=False):
        lines.append(
            f"| {row.context} | {row.arm} | {row.n:,} | {row.p0:.3f} | "
            f"{row.jump_probability:.3f} | {row.exit_probability:.3f} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "PCA is used to show where states and trajectories lie. Raw clinical "
        "variables define remission, occupancy, death phenotype, adverse-event "
        "change and treatment contrasts. Sparse GENIE patients need not contain "
        "all twenty coordinates: observed dimensions remain observed, while any "
        "hybrid projection is explicitly model-augmented and uncertain.",
        "",
    ]
    (out_dir / "readout.md").write_text("\n".join(lines))


def build(
    *,
    out_dir: Path = DEFAULT_OUT,
    run_ids: Sequence[str] = REFERENCE_RUNS,
    phase1_run: str = PHASE1_RUN,
) -> dict[str, Any]:
    """Build the frozen map, validation audit and persisted transformation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = load_reference_tables(run_ids)
    active_conditions, active_meta = ledger.load_active_condition_features(run_ids)
    state, resolved_sites = build_state_matrix(tables, active_conditions)
    weights = equal_model_trajectory_snapshot_weights(state)

    model, scores = fit_weighted_pca(state, weights)
    state["pc1"] = scores[:, 0]
    state["pc2"] = scores[:, 1]
    state["observation_weight"] = weights

    multipliers, balance_meta = block_balanced_multipliers(state, weights)
    sensitivity_model, sensitivity_scores = fit_weighted_pca(
        state,
        weights,
        feature_multipliers=multipliers,
    )
    _aligned, sensitivity_meta = align_sensitivity_scores(scores, sensitivity_scores)
    sensitivity = {**balance_meta, **sensitivity_meta}

    feature_table, block_table = feature_diagnostics(state, weights, model)
    transitions = event_transitions(state, tables["events"])
    event_summary = summarize_event_validation(transitions)
    resolution = primary_site_resolution_audit(resolved_sites)

    phase1_tables = load_reference_tables((phase1_run,))
    phase1_active, phase1_active_meta = ledger.load_active_condition_features(
        (phase1_run,)
    )
    phase1_state, phase1_resolved_sites = build_state_matrix(
        phase1_tables,
        phase1_active,
    )
    phase1_scores = model.transform(phase1_state)
    phase1_state["pc1"] = phase1_scores[:, 0]
    phase1_state["pc2"] = phase1_scores[:, 1]
    p0_records = phase1_p0_records(
        phase1_state,
        phase1_tables["snapshots"],
        run_id=phase1_run,
    )
    p0_summary = summarize_phase1_p0(p0_records)

    projection_columns = [
        "_run",
        "_gid",
        "trajectory_id",
        "model_short",
        "tumor_oncotree_code",
        "snapshot_index",
        "state_id",
        "days_from_diagnosis",
        *STATE_FEATURES,
        "pc1",
        "pc2",
        "observation_weight",
    ]
    state.loc[:, projection_columns].to_csv(
        out_dir / "state_projection.csv.gz",
        index=False,
        compression="gzip",
    )
    feature_table.to_csv(out_dir / "feature_loadings.csv", index=False)
    block_table.to_csv(out_dir / "block_shares.csv", index=False)
    transitions.to_csv(out_dir / "event_transitions.csv.gz", index=False)
    event_summary.to_csv(out_dir / "event_validation.csv", index=False)
    p0_records.to_csv(out_dir / "phase1_p0_records.csv", index=False)
    p0_summary.to_csv(out_dir / "phase1_p0_summary.csv", index=False)
    resolution.to_csv(out_dir / "primary_site_resolution.csv", index=False)
    phase1_resolution = primary_site_resolution_audit(phase1_resolved_sites)
    phase1_resolution.to_csv(
        out_dir / "phase1_primary_site_resolution.csv",
        index=False,
    )

    contract = {
        "artifact": "paper_state_map_v1",
        "role": "fixed_visual_coordinate_system_not_patient_model",
        "state_features": list(STATE_FEATURES),
        "tumour_features": list(TUMOUR_FEATURES),
        "host_features": list(HOST_FEATURES),
        "anatomical_groups": {
            key: list(value) for key, value in ANATOMICAL_GROUPS.items()
        },
        "other_anatomical_rule": "all resolved site classes not in named groups",
        "primary_site_rule": (
            "resolve generic primary_site records from baseline "
            "primary_site_class_canonical; deduplicate resolved classes"
        ),
        "encodings": {
            "anatomical_burden": "ordinal 0-5 divided by 5",
            "disease_extent": "min(unique resolved active classes, 5) divided by 5",
            "ecog": "0-4 divided by 4",
            "global_symptoms": "0-3 divided by 3",
            "organ_function": "0-4 divided by 4",
            "relative_weight_loss": (
                "100*(baseline-current)/baseline, clipped [-20,30], "
                "then (value+20)/50"
            ),
            "maximum_active_ae_grade": "min(grade,5)/5",
            "active_ae_count": "min(count,5)/5",
        },
        "scaling": "fixed clinical ranges only; mean-centred; no empirical z-scoring",
        "observation_weights": (
            "equal generator -> equal trajectory within generator -> "
            "equal snapshot within trajectory"
        ),
        "temporal_rule": (
            "event snapshots with LOCF between events; no linear interpolation; "
            "death is absorbing EXIT"
        ),
        "orientation": (
            "PC1 sign toward increasing overall encoded burden; "
            "PC2 sign toward increasing host burden; PC3-PC20 sign fixed by "
            "requiring the largest absolute loading to be positive"
        ),
        "primary_model": model.as_json(),
        "sensitivity_model": sensitivity_model.as_json(),
        "sensitivity": sensitivity,
    }
    (out_dir / "state_map_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )

    draw_map_contract(
        state,
        feature_table,
        block_table,
        model,
        sensitivity,
        out_dir,
    )
    draw_validation(event_summary, p0_summary, out_dir)
    write_readout(
        out_dir,
        state=state,
        model=model,
        features=feature_table,
        blocks=block_table,
        event_summary=event_summary,
        p0_summary=p0_summary,
        resolution=resolution,
        sensitivity=sensitivity,
    )

    source_hashes: dict[str, str] = {}
    for run_id in (*run_ids, phase1_run):
        for filename in (
            "baseline_rows.csv",
            "snapshot_rows.csv",
            "disease_site_rows.csv",
            "event_rows.csv",
            "endpoint_rows.csv",
        ):
            path = ROOT / "runs" / run_id / "derivation" / filename
            source_hashes[f"{run_id}/{filename}"] = _sha256(path)

    manifest = {
        "artifact": "paper_state_map_v1",
        "git_commit": _git_commit(),
        "script_sha256": _sha256(Path(__file__)),
        "feature_ledger_script_sha256": _sha256(
            ROOT / "scripts" / "build_paper_state_feature_ledger.py"
        ),
        "reference_runs": list(run_ids),
        "phase1_run": phase1_run,
        "source_hashes": source_hashes,
        "n_living_snapshots": int(len(state)),
        "n_trajectories": int(state["_gid"].nunique()),
        "n_models": int(state["model_short"].nunique()),
        "n_features": len(STATE_FEATURES),
        "n_tumour_features": len(TUMOUR_FEATURES),
        "n_host_features": len(HOST_FEATURES),
        "pca_display_components": int(model.components.shape[0]),
        "pca_serialized_components": int(model.full_components.shape[0]),
        "pca_explained_variance_ratio": (
            model.explained_variance_ratio[:2].tolist()
        ),
        "event_checks_passed": int(event_summary["passed"].astype(bool).sum()),
        "event_checks_total": int(len(event_summary)),
        "active_condition_coverage": active_meta,
        "phase1_active_condition_coverage": phase1_active_meta,
        "outcomes_used_to_fit_pca": False,
        "treatments_used_to_fit_pca": False,
        "future_information_used_to_fit_pca": False,
        "locf": True,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--phase1-run",
        default=PHASE1_RUN,
        help="Existing interventional run used only for the p0 gate.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build(out_dir=args.out_dir, phase1_run=args.phase1_run)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
