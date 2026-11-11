#!/usr/bin/env python3
"""Audit the native geometry of the six existing synthetic oncology atlases.

This is a ZERO-GENERATION analysis. It asks whether each model's unaided trajectories
form cancer-specific path laws, which parts of the frozen trajectory representation carry
that structure, and where the output is stereotyped. It does not compare generated paths
with real patients and therefore makes no realism claim.

The shared geometry is fitted on two SHA-selected trajectories per model x cancer cell.
All reported comparisons use three disjoint trajectories from the same cells. Cross-model
comparisons use only cancer cards with five accepted trajectories in every model.

Run:
  uv run python scripts/explore/audit_native_atlas_geometry_v1.py --self-test
  uv run python scripts/explore/audit_native_atlas_geometry_v1.py --run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, jensenshannon, pdist
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import trajectory_representation_v1 as geometry  # noqa: E402


DERIVATIONS = (
    ROOT / "runs" / "pillar1_anchored_atlas_tranche1_v4_n5_live01" / "derivation",
    ROOT / "runs" / "pillar1_anchored_atlas_gemma4_n5_live01" / "derivation",
    ROOT / "runs" / "sonnet5_mini_atlas_n100_2026_07_08" / "derivation",
    ROOT / "runs" / "sonnet5_full_atlas_topup_remaining128_n5_2026_07_09" / "derivation",
    ROOT / "runs" / "kimi_k3_off_full_atlas_n5_complete" / "derivation",
    ROOT / "runs" / "glm52_full_atlas_n5" / "derivation",
)
OUT = ROOT / "outputs" / "exploratory_native_atlas_geometry"
SEED = 20260820
HORIZON = 24.0
N_DISCOVERY = 2
N_EVALUATION = 3
N_BOOTSTRAP = 399
NEAR_DUPLICATE_VP = 0.10

MODEL_LABELS = {
    "claude_sonnet5": "Sonnet 5",
    "gemini31_lite": "Gemini 3.1 Lite",
    "gemma4": "Gemma 4",
    "glm52_fp8": "GLM-5.2",
    "kimi_k3_off": "Kimi K3",
    "qwen36_flash": "Qwen 3.6",
}
NATIVE_EVENT_TYPES = (
    "clonal_evolution_event",
    "death",
    "local_intervention",
    "patient_state_update",
    "progression",
    "response",
    "systemic_treatment_modification",
    "systemic_treatment_start",
    "systemic_treatment_stop",
    "systemic_treatment_switch",
    "toxicity",
)
REPORT_BLOCKS = (*geometry.BLOCKS, "composite")


def accepted(frame: pd.DataFrame) -> pd.DataFrame:
    if "accepted" not in frame:
        return frame.copy()
    keep = frame["accepted"].astype(str).str.lower().isin({"true", "1", "1.0"})
    return frame.loc[keep].copy()


def load_table(name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for derivation in DERIVATIONS:
        path = derivation / name
        if not path.exists():
            raise FileNotFoundError(path)
        frame = accepted(pd.read_csv(path, low_memory=False))
        frame["_run"] = derivation.parent.name
        frame["_gid"] = frame["_run"].astype(str) + "::" + frame["trajectory_id"].astype(str)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def stable_rank(value: str, split_seed: int) -> str:
    return hashlib.sha256(f"{split_seed}|{value}".encode()).hexdigest()


def split_cards(
    baseline: pd.DataFrame, split_seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    baseline = baseline.drop_duplicates(["model_short", "trajectory_id"]).copy()
    counts = baseline.groupby(["model_short", "tumor_oncotree_code"]).size()
    models = sorted(MODEL_LABELS)
    matched_codes: list[str] = []
    excluded: dict[str, dict[str, int]] = {}
    for cancer in sorted(baseline["tumor_oncotree_code"].astype(str).unique()):
        per_model = {model: int(counts.get((model, cancer), 0)) for model in models}
        if all(value >= N_DISCOVERY + N_EVALUATION for value in per_model.values()):
            matched_codes.append(cancer)
        else:
            excluded[cancer] = per_model

    matched = baseline[
        baseline["model_short"].isin(models)
        & baseline["tumor_oncotree_code"].astype(str).isin(matched_codes)
    ].copy()
    matched["_rank"] = matched["_gid"].map(lambda value: stable_rank(value, split_seed))
    matched = matched.sort_values(
        ["model_short", "tumor_oncotree_code", "_rank", "_gid"]
    )
    matched["_cell_rank"] = matched.groupby(
        ["model_short", "tumor_oncotree_code"]
    ).cumcount()
    discovery = matched[matched["_cell_rank"] < N_DISCOVERY].copy()
    evaluation = matched[
        matched["_cell_rank"].between(
            N_DISCOVERY, N_DISCOVERY + N_EVALUATION - 1
        )
    ].copy()
    expected_cells = len(models) * len(matched_codes)
    if len(discovery) != expected_cells * N_DISCOVERY:
        raise AssertionError("unbalanced discovery split")
    if len(evaluation) != expected_cells * N_EVALUATION:
        raise AssertionError("unbalanced evaluation split")
    if set(discovery["_gid"]) & set(evaluation["_gid"]):
        raise AssertionError("discovery and evaluation trajectories overlap")
    audit = {
        "models": models,
        "n_matched_cancers": len(matched_codes),
        "matched_cancers": matched_codes,
        "excluded_cancers": excluded,
        "n_discovery": len(discovery),
        "n_evaluation": len(evaluation),
        "split_seed": split_seed,
    }
    return discovery, evaluation, audit


def treatment_identity(row: pd.Series) -> str:
    event_type = str(row["event_type"])
    if event_type not in {"systemic_treatment_start", "systemic_treatment_switch"}:
        return event_type
    try:
        payload = json.loads(str(row.get("materialized_payload_json") or "{}"))
    except json.JSONDecodeError:
        return event_type
    regimen = payload.get("regimen") or {}
    classes = sorted({str(value) for value in regimen.get("component_classes") or []})
    return f"{event_type}::{'+'.join(classes)}" if classes else event_type


def build_courses(events: pd.DataFrame, gids: set[str]) -> dict[str, tuple[list[float], list[str]]]:
    events = events[events["_gid"].isin(gids)].copy()
    events["_months"] = pd.to_numeric(events["days_from_diagnosis"], errors="coerce") / 30.4
    events = events[
        events["_months"].notna()
        & events["_months"].between(0.0, HORIZON, inclusive="both")
    ].copy()
    events["_mark"] = events.apply(treatment_identity, axis=1)
    events["_event_index"] = pd.to_numeric(events["event_index"], errors="coerce")
    courses: dict[str, tuple[list[float], list[str]]] = {}
    for gid, rows in events.groupby("_gid", sort=False):
        rows = rows.sort_values(["_months", "_event_index", "event_id"])
        courses[str(gid)] = (
            rows["_months"].astype(float).tolist(),
            rows["_mark"].astype(str).tolist(),
        )
    missing = gids - set(courses)
    if missing:
        raise AssertionError(f"{len(missing)} selected trajectories have no events")
    return courses


def block_vector(
    representation: geometry.Representation,
    course: tuple[list[float], list[str]],
    block: str,
) -> np.ndarray:
    raw = geometry.components(
        course[0],
        course[1],
        representation.horizon,
        list(representation.prototypes),
        representation.time_norm,
    )[block]
    return (raw / representation.coord_scale[block]) * representation.block_scale[block]


def vector_tables(
    representation: geometry.Representation,
    evaluation: pd.DataFrame,
    courses: dict[str, tuple[list[float], list[str]]],
) -> dict[str, np.ndarray]:
    tables = {
        block: np.stack(
            [block_vector(representation, courses[gid], block) for gid in evaluation["_gid"]]
        )
        for block in geometry.BLOCKS
    }
    tables["composite"] = np.concatenate(
        [tables[block] for block in geometry.BLOCKS], axis=1
    )
    return tables


def pair_distance_summary(
    evaluation: pd.DataFrame,
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    cancers = sorted(evaluation["tumor_oncotree_code"].astype(str).unique())
    indices = {
        cancer: np.flatnonzero(
            evaluation["tumor_oncotree_code"].astype(str).to_numpy() == cancer
        )
        for cancer in cancers
    }
    within = np.array(
        [float(pdist(vectors[indices[cancer]]).mean()) for cancer in cancers]
    )
    between = np.zeros((len(cancers), len(cancers)), float)
    centroid = np.stack([vectors[indices[cancer]].mean(axis=0) for cancer in cancers])
    for left, right in combinations(range(len(cancers)), 2):
        value = float(
            cdist(vectors[indices[cancers[left]]], vectors[indices[cancers[right]]]).mean()
        )
        between[left, right] = between[right, left] = value
    return within, between, centroid, cancers


def bootstrap_specificity(
    within: np.ndarray,
    between: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    mask = np.triu(np.ones_like(between, dtype=bool), 1)
    observed_within = float(within.mean())
    observed_between = float(between[mask].mean())
    observed_ratio = observed_between / observed_within
    rng = np.random.default_rng(seed)
    samples = np.empty((N_BOOTSTRAP, 3), float)
    n = len(within)
    for replicate in range(N_BOOTSTRAP):
        draw = rng.integers(0, n, size=n)
        while len(np.unique(draw)) < 2:
            draw = rng.integers(0, n, size=n)
        sample_within = float(within[draw].mean())
        sampled = between[np.ix_(draw, draw)]
        valid = np.triu(draw[:, None] != draw[None, :], 1)
        sample_between = float(sampled[valid].mean())
        samples[replicate] = (
            sample_within,
            sample_between,
            sample_between / sample_within,
        )
    intervals = np.quantile(samples, [0.025, 0.975], axis=0)
    return {
        "within": observed_within,
        "within_ci": intervals[:, 0].tolist(),
        "between": observed_between,
        "between_ci": intervals[:, 1].tolist(),
        "specificity_ratio": observed_ratio,
        "specificity_ratio_ci": intervals[:, 2].tolist(),
    }


def composition_vector(course: tuple[list[float], list[str]]) -> np.ndarray:
    counts = Counter(mark.split("::", 1)[0] for mark in course[1])
    vector = np.array([counts[event] for event in NATIVE_EVENT_TYPES], float)
    total = vector.sum()
    return vector / total if total else np.zeros_like(vector)


def stereotypy(
    model_frame: pd.DataFrame,
    courses: dict[str, tuple[list[float], list[str]]],
    grammar_vectors: np.ndarray,
) -> dict[str, Any]:
    canonical = {
        gid: geometry.episodes(*courses[gid], HORIZON) for gid in model_frame["_gid"]
    }
    compositions = np.stack(
        [composition_vector(canonical[gid]) for gid in model_frame["_gid"]]
    )
    cancer_values = model_frame["tumor_oncotree_code"].astype(str).to_numpy()
    composition_distances: list[float] = []
    grammar_distances: list[float] = []
    exact = 0
    near = 0
    pairs = 0
    for cancer in sorted(set(cancer_values)):
        indices = np.flatnonzero(cancer_values == cancer)
        for left, right in combinations(indices, 2):
            composition_distances.append(
                float(jensenshannon(compositions[left], compositions[right], base=2.0))
            )
            grammar_distances.append(
                float(np.linalg.norm(grammar_vectors[left] - grammar_vectors[right]))
            )
            a = canonical[model_frame.iloc[left]["_gid"]]
            b = canonical[model_frame.iloc[right]["_gid"]]
            signature_a = tuple(zip(*a, strict=True))
            signature_b = tuple(zip(*b, strict=True))
            exact += int(signature_a == signature_b)
            near += int(geometry.typed_vp(*a, *b, time_norm=HORIZON) <= NEAR_DUPLICATE_VP)
            pairs += 1
    prevalence = {
        event: float(
            np.mean(
                [
                    event in {mark.split("::", 1)[0] for mark in canonical[gid][1]}
                    for gid in model_frame["_gid"]
                ]
            )
        )
        for event in NATIVE_EVENT_TYPES
    }
    return {
        "within_cancer_composition_js_distance": float(
            np.mean(composition_distances)
        ),
        "within_cancer_grammar_distance": float(np.mean(grammar_distances)),
        "exact_event_fingerprint_pairs": exact,
        "near_event_fingerprint_pairs": near,
        "n_within_cancer_pairs": pairs,
        "exact_event_fingerprint_rate": exact / pairs,
        "near_event_fingerprint_rate": near / pairs,
        "near_event_fingerprint_vp_threshold": NEAR_DUPLICATE_VP,
        "fingerprint_scope": (
            "episode-deduplicated event times/types and native treatment classes through "
            "24 months; not full-ledger identity"
        ),
        "event_presence": prevalence,
    }


def topology_agreement(
    centroids: dict[str, dict[str, np.ndarray]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for block in REPORT_BLOCKS:
        rows: list[dict[str, Any]] = []
        for left, right in combinations(sorted(centroids), 2):
            a = pdist(centroids[left][block])
            b = pdist(centroids[right][block])
            rho = float(spearmanr(a, b).statistic)
            rows.append({"model_a": left, "model_b": right, "spearman_rho": rho})
        output[block] = rows
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_provenance() -> dict[str, dict[str, str]]:
    provenance: dict[str, dict[str, str]] = {}
    for derivation in DERIVATIONS:
        run_root = derivation.parent
        manifest = run_root / "manifest.json"
        provenance[run_root.name] = {
            "manifest_sha256": sha256(manifest),
            "baseline_rows_sha256": sha256(derivation / "baseline_rows.csv"),
            "event_rows_sha256": sha256(derivation / "event_rows.csv"),
        }
    return provenance


def run(output: Path, split_seed: int = SEED) -> dict[str, Any]:
    baseline = load_table("baseline_rows.csv")
    events = load_table("event_rows.csv")
    discovery, evaluation, split = split_cards(baseline, split_seed)
    selected = set(discovery["_gid"]) | set(evaluation["_gid"])
    courses = build_courses(events, selected)
    training = [courses[gid] for gid in discovery["_gid"]]
    representation = geometry.fit(
        training,
        HORIZON,
        seed=SEED,
        time_norm="own",
    )
    all_vectors = vector_tables(representation, evaluation, courses)

    models: dict[str, Any] = {}
    centroids: dict[str, dict[str, np.ndarray]] = {}
    for model_index, model in enumerate(sorted(MODEL_LABELS)):
        model_mask = evaluation["model_short"].astype(str).to_numpy() == model
        model_frame = evaluation.loc[model_mask].reset_index(drop=True)
        centroids[model] = {}
        block_results: dict[str, Any] = {}
        grammar_for_model: np.ndarray | None = None
        for block_index, block in enumerate(REPORT_BLOCKS):
            block_values = all_vectors[block][model_mask]
            within, between, centroid, cancers = pair_distance_summary(
                model_frame, block_values
            )
            if cancers != split["matched_cancers"]:
                raise AssertionError("cancer order drifted across models")
            centroids[model][block] = centroid
            block_results[block] = bootstrap_specificity(
                within,
                between,
                SEED + 100 * model_index + block_index,
            )
            if block == "grammar":
                grammar_for_model = block_values
        if grammar_for_model is None:
            raise AssertionError("grammar vectors missing")
        models[model] = {
            "label": MODEL_LABELS[model],
            "n_trajectories": len(model_frame),
            "n_cancers": model_frame["tumor_oncotree_code"].nunique(),
            "blocks": block_results,
            "stereotypy": stereotypy(model_frame, courses, grammar_for_model),
        }

    payload = {
        "analysis": "native synthetic atlas geometry audit",
        "evidence_scope": (
            "generated expressivity, stereotypy, cancer specificity, and model topology; "
            "not real-world fidelity"
        ),
        "configuration": {
            "horizon_months": HORIZON,
            "time_normalization": "own-duration",
            "n_discovery_per_model_cancer": N_DISCOVERY,
            "n_evaluation_per_model_cancer": N_EVALUATION,
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_unit": "cancer card",
            "seed": SEED,
            "split_seed": split_seed,
            "native_event_types": list(NATIVE_EVENT_TYPES),
        },
        "split_audit": split,
        "models": models,
        "cross_model_topology": topology_agreement(centroids),
        "source_hashes": {
            "script": sha256(Path(__file__)),
            "representation": sha256(
                ROOT / "scripts" / "explore" / "trajectory_representation_v1.py"
            ),
            "ontology": sha256(ROOT / "manuscript" / "trajectory_ontology_v1.yaml"),
        },
        "input_provenance": input_provenance(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def self_test() -> int:
    toy = pd.DataFrame(
        [
            {
                "model_short": model,
                "tumor_oncotree_code": cancer,
                "trajectory_id": f"{model}-{cancer}-{index}",
                "_gid": f"run::{model}-{cancer}-{index}",
            }
            for model in MODEL_LABELS
            for cancer in ("A", "B")
            for index in range(5)
        ]
    )
    first_discovery, first_evaluation, audit = split_cards(toy)
    second_discovery, second_evaluation, _ = split_cards(toy.sample(frac=1.0, random_state=7))
    assert audit["n_matched_cancers"] == 2
    assert set(first_discovery["_gid"]) == set(second_discovery["_gid"])
    assert set(first_evaluation["_gid"]) == set(second_evaluation["_gid"])
    assert not set(first_discovery["_gid"]) & set(first_evaluation["_gid"])

    specific = np.array([[0.0], [0.1], [0.2], [3.0], [3.1], [3.2]])
    frame = pd.DataFrame({"tumor_oncotree_code": ["A"] * 3 + ["B"] * 3})
    within, between, _, _ = pair_distance_summary(frame, specific)
    summary = bootstrap_specificity(within, between, SEED)
    assert summary["specificity_ratio"] > 10.0
    print("native atlas geometry self-test OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT / "readout.json",
    )
    parser.add_argument("--split-seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.run:
        parser.error("choose --self-test or --run")
    result = run(args.out, args.split_seed)
    print(
        f"Native atlas audit: {result['split_audit']['n_matched_cancers']} cancers, "
        f"{result['split_audit']['n_evaluation']} evaluation trajectories -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
