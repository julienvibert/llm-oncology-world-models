"""SynthOncCases derivation layer.

Reads ``<run_root>/trajectories/**/{materialized_trajectory.json,
parsed_ledger.json, trajectory_result.json, result.json,
conditioning_spec.{json,yaml}}`` and materializes row-shaped analysis artifacts
under ``<run_root>/derivation/``:

- ``event_rows.csv``         — one row per ledger event (wide format)
- ``snapshot_rows.csv``      — one row per materialized snapshot
- ``treatment_rows.csv``     — one row per treatment course
- ``endpoint_rows.csv``      — one row per trajectory (OS / PFS-by-line / TTD / TTNT)
- ``baseline_rows.csv``      — one row per trajectory carrying the generated
                               baseline identity / medical-history / molecular-
                               workup fields + spec_* hashes +
                               materialized_baseline_hash (B1b)
- ``baseline_biomarker_rows.csv`` — one row per generated baseline biomarker,
                               preserving raw name/value plus deterministic
                               normalized analysis keys
- ``baseline_alteration_rows.csv`` — one row per generated baseline alteration,
                               preserving raw molecular detail and buckets
- ``treatment_component_rows.csv`` — one row per systemic treatment component,
                               preserving raw component names/classes
- ``disease_site_rows.csv``  — one row per active disease site per snapshot,
                               with site_class normalization + is_new_at_snapshot
                               provenance (B1b; powers family-2 site-tropism)
- ``attempt_rows.csv``       — one row per generation attempt (accepted + failed),
                               with cost / tokens / failure_kind / failure_rule_id /
                               failure_family / validation_warnings + spec_* hashes
                               (B1b; powers family-13 run-validity QC)
- ``field_utilization.json`` — per (model, tumor, policy, conditioning_label)
- ``run_summary.json`` + ``.md`` — derivation-level rollup

Pure read-only on trajectories. All file I/O via ``pathlib``. No global state.

Endpoint definitions (ASCO/RECIST conventions):

- ``OS``           = ``outcome_date - diagnosis_date`` in days
- ``PFS_line_N``   = ``systemic_treatment_start/switch[line=N].date`` → earliest of next
                     ``progression`` / ``death`` / ``last_followup``
- ``TTD_line_N``   = ``systemic_treatment_start/switch[line=N].date`` →
                     corresponding course end
- ``TTNT_N→N+1``   = ``systemic_treatment_start/switch[line=N].date`` →
                     ``systemic_treatment_start/switch[line=N+1].date``

Cell-key columns are present on every row of the row-level CSVs:
``trajectory_id, condition_id, policy_id, model_id, model_short, future_index,
tumor_oncotree_code, conditioning_label, conditioning_hash, accepted``.

``tumor_oncotree_code`` is read from ``baseline.identity_card.diagnosis.oncotree_code``.
``conditioning_label`` and ``conditioning_hash`` are read per trajectory from
``trajectory_result.json[cell]`` (post-Commit-B: per-cell, not manifest-root).
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path
import typing
from typing import Any

import pandas as pd
import yaml
from pydantic import ValidationError

from synthonccases.oncollm.condition_lower import lower_conditioning_cell
from synthonccases.oncollm.conditioning_audit import (
    EMPTY_CONSTRAINTS_SENTINEL,
    EMPTY_PREFIX_SENTINEL,
    ConditioningSpec,
    ConstraintMatch,
    _canonicalize_spec_value,
    _sha256_hex,
    canonical_json,
    compute_baseline_hash,
    compute_conditioning_spec_hash,
    compute_future_constraint_hash,
    compute_observed_prefix_hash,
    future_constraints_audit,
    spec_from_json_dict,
)
from synthonccases.oncollm.poc_summary import (
    AcquiredAlterationRow,
    TreatmentCourseRow,
    _mean,
    _parse_date,
    _regimen_label,
    _round_float,
    derive_treatment_courses,
)
from synthonccases.oncollm.replay import (
    BaselineObject,
    EventType,
    LedgerEvent,
    MaterializedLedgerEvent,
    canonicalize_legacy_event_dict,
)

DERIVATION_VERSION = "oncollm_derivation_v1"

DEFAULT_CONDITIONING_LABEL = "unspecified"
DEFAULT_TUMOR_ONCOTREE_CODE = "unknown"

# B1b: attempt_rows sentinel for trajectories with no accepted sibling.
# discover_attempts recovers tumor_oncotree_code from any accepted sibling
# attempt's materialized baseline; failed-only trajectories fall back to
# this documented sentinel.
DEFAULT_UNKNOWN_TUMOR = "__unknown_tumor__"

# B1b sentinels for runs without a conditioning_spec.json on disk.
LEGACY_NO_SPEC_SENTINEL = "LEGACY_NO_SPEC"
EVENT_SOURCE_OBSERVED_PREFIX = "observed_prefix"
EVENT_SOURCE_GENERATED = "generated"

# B1b: ordering for `total_burden_band` used in best_response_class derivation.
# Lower index = less disease burden.
_BURDEN_BAND_ORDER: dict[str, int] = {
    "absent": 0,
    "microscopic": 1,
    "small": 2,
    "moderate": 3,
    "bulky": 4,
    "diffuse_or_leukemic": 5,
}

# B1b: first-line guideline-canonical regimen per policy. Hardcoded for
# paper-v1; revisit in B3 when configs/treatment_guidelines_v1.yaml lands.
# Null for unknown policies (do not silently default).
_POLICY_CANONICAL_FIRST_LINE: dict[str, str] = {
    "A_THP": "THP",
    "B_TDM1": "T-DM1",
}

# B1b: regimen canonicalization (lifted from
# scripts/build_paper_treatment_pathways.py:136-153). Private to derivation
# for now — extract to a shared module only when a second caller surfaces.
_CANONICAL_OTHER = "OTHER"
_REGIMEN_TOKEN_SPLIT_RE = re.compile(r"[\s\-_/+,;\.()]+")
_CANONICAL_REGIMENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("TUC_TRIPLET", ("tucatinib", "capecitabine")),
    ("T-DXd", ("trastuzumab", "deruxtecan")),
    ("SACITUZUMAB_GOVITECAN", ("sacituzumab", "govitecan")),
    ("T-DM1", ("trastuzumab", "emtansine")),
    ("T-DM1", ("dm1",)),
    ("T-DM1", ("tdm1",)),
    ("THP", ("trastuzumab", "pertuzumab")),
    ("THP", ("thp",)),
    ("TCHP", ("tchp",)),
)


def _canonicalize_regimen(text: Any) -> str:
    """Map a heterogeneous regimen string to a canonical bucket.

    Token-bag match: split on a generous separator set, lowercase, then
    test each canonical bucket's required tokens against the produced
    token set. First match wins (``_CANONICAL_REGIMENS`` ordered
    most-specific first). Falls back to ``OTHER`` on miss / null / empty.
    """
    if text is None:
        return _CANONICAL_OTHER
    if isinstance(text, float) and pd.isna(text):
        return _CANONICAL_OTHER
    raw = str(text).strip()
    if not raw:
        return _CANONICAL_OTHER
    tokens = {t.lower() for t in _REGIMEN_TOKEN_SPLIT_RE.split(raw) if t}
    for canonical, required in _CANONICAL_REGIMENS:
        if all(req in tokens for req in required):
            return canonical
    return _CANONICAL_OTHER


# ---------- tracked field paths for field_utilization.json ----------
#
# Field set is deliberately tied to fields that actually appear on the v2.1
# BaselineObject / LedgerEvent classes (see replay.py). The grounding test in
# test_derivation_field_utilization.py instantiates representative models and
# asserts every path resolves — if a future schema edit renames a field, that
# test fails fast.

# Baseline measurement-instrument fields — populated rate measures how often
# the LLM emitted optional rich-content baseline detail.
_BASELINE_TRACKED_FIELDS: tuple[str, ...] = (
    "medical_history.comorbidities",
    "medical_history.smoking_status",
    "medical_history.tobacco_exposure_band",
    "medical_history.pack_years",
    "medical_history.family_history",
    "medical_history.contextual_factors",
    "identity_card.baseline_health.contraindications",
    "identity_card.baseline_health.frailty_band",
    "identity_card.baseline_health.autoimmune_disease_history",
    "identity_card.baseline_health.neuropathy_at_baseline",
    "identity_card.molecular_profile.alterations",
    "identity_card.molecular_profile.primary_driver_bucket",
    "identity_card.molecular_profile.biomarkers",
    "identity_card.staging.formal_stage",
    "identity_card.diagnosis.histology_description",
    "identity_card.diagnosis.histology_bucket",
    "identity_card.diagnosis.primary_site",
    "identity_card.diagnosis.primary_site_class",
)

# Per event-type measurement-instrument fields. Each entry: (event_type, field).
# Utilization = (events of that type with field populated) / (events of that type).
_EVENT_TRACKED_FIELDS: tuple[tuple[str, str], ...] = (
    ("toxicity", "recognized_date"),
    ("toxicity", "ae_category"),
    ("toxicity", "ctcae_v5_grade"),
    ("toxicity", "causes"),
    ("clonal_evolution_event", "recognized_date"),
    ("clonal_evolution_event", "gene"),
    ("clonal_evolution_event", "alteration_type"),
    ("clonal_evolution_event", "variant"),
    ("clonal_evolution_event", "driver_bucket"),
    ("clonal_evolution_event", "actionability"),
    ("clonal_evolution_event", "mechanism_bucket"),
    ("clonal_evolution_event", "detection_method"),
    ("clonal_evolution_event", "sample_source"),
    ("clonal_evolution_event", "mechanism"),
    ("patient_state_update", "state_change_cause"),
    ("local_intervention", "target_sites"),
)


# ---------- dataclasses ----------


@dataclass(frozen=True, slots=True)
class CellKey:
    """Trajectory-level cell identity used as the join key across all rows."""

    trajectory_id: str
    condition_id: str
    policy_id: str
    model_id: str
    model_short: str
    future_index: int
    tumor_oncotree_code: str
    conditioning_label: str
    conditioning_hash: str
    accepted: bool

    def as_columns(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "condition_id": self.condition_id,
            "policy_id": self.policy_id,
            "model_id": self.model_id,
            "model_short": self.model_short,
            "future_index": self.future_index,
            "tumor_oncotree_code": self.tumor_oncotree_code,
            "conditioning_label": self.conditioning_label,
            "conditioning_hash": self.conditioning_hash,
            "accepted": self.accepted,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryArtifact:
    """Wraps the per-trajectory inputs the derivation layer reads from disk.

    ``conditioning_spec`` and ``constraint_results`` are B1b additions
    populated by ``discover_trajectories`` from
    ``<attempt_dir>/conditioning_spec.json`` (with a YAML fallback). Both
    are ``None`` on legacy runs that predate the per-attempt spec write.
    Downstream column builders use these to populate ``event_source``,
    ``constraint_match_ids``, and the four ``spec_*`` hashes.
    """

    cell: CellKey
    trajectory_dir: Path
    parsed_ledger: tuple[Mapping[str, Any], ...]  # sparse LLM emission shape
    materialized: Mapping[str, Any] | None        # full materialized trajectory
    trajectory_result: Mapping[str, Any]          # trajectory_result.json
    conditioning_spec: ConditioningSpec | None = None
    constraint_results: tuple[ConstraintMatch, ...] | None = None


@dataclass(frozen=True, slots=True)
class DerivationReport:
    """Returned by ``derive_run`` so the CLI / tests can introspect outputs."""

    run_root: Path
    out_dir: Path
    event_rows_path: Path
    snapshot_rows_path: Path
    treatment_rows_path: Path
    endpoint_rows_path: Path
    field_utilization_path: Path
    run_summary_json_path: Path
    run_summary_md_path: Path
    n_trajectories: int
    n_events: int
    n_snapshots: int
    n_treatment_courses: int
    n_endpoints: int
    # B1b additions
    baseline_rows_path: Path
    n_baselines: int
    baseline_biomarker_rows_path: Path
    n_baseline_biomarkers: int
    baseline_alteration_rows_path: Path
    n_baseline_alterations: int
    treatment_component_rows_path: Path
    n_treatment_components: int
    disease_site_rows_path: Path
    n_disease_sites: int
    attempt_rows_path: Path
    n_attempts: int


# ---------- discovery ----------


def _read_json_if_exists(path: Path) -> Any | None:
    """Tolerant reader — returns None on missing or malformed JSON.

    Use ``_read_json_strict`` for accepted-trajectory artifacts where a
    missing/malformed file represents real scientific data loss.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


class DerivationDataLossError(RuntimeError):
    """Raised when an accepted trajectory is missing required on-disk artifacts.

    Distinct from FileNotFoundError so callers can surface the specific
    derivation-layer contract violation rather than a generic OS error.
    """


def _load_conditioning_spec_for_attempt(
    attempt_dir: Path,
) -> ConditioningSpec | None:
    """Load ``conditioning_spec.json`` (or ``.yaml`` fallback) from an
    accepted-attempt directory.

    Returns ``None`` when neither file is present — that's the legacy
    sentinel-backfill path for runs predating B1b's runner write. Raises
    ``DerivationDataLossError`` when a file exists but fails to parse
    (silent absence would let a malformed spec produce wrong hashes).

    Two YAML shapes are recognized:

    1. Direct serialized-spec form (the symmetric inverse of
       ``spec_to_json_dict``): parsed via ``spec_from_json_dict``.
    2. The offline-prefix-fixture wrapper shape ``{"conditioning": {...}}``
       which the upstream config-loader path uses: parsed via
       ``lower_conditioning_cell(conditioning_payload=...)``.

    JSON is always treated as direct serialized-spec form (the runner
    writes that shape via ``spec_to_json_dict``).
    """
    json_path = attempt_dir / "conditioning_spec.json"
    yaml_path = attempt_dir / "conditioning_spec.yaml"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise DerivationDataLossError(
                f"conditioning_spec.json at {json_path} is unreadable: {exc}"
            ) from exc
        loaded_path = json_path
        is_yaml = False
    elif yaml_path.exists():
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError) as exc:
            raise DerivationDataLossError(
                f"conditioning_spec.yaml at {yaml_path} is unreadable: {exc}"
            ) from exc
        loaded_path = yaml_path
        is_yaml = True
    else:
        return None
    if not isinstance(data, Mapping):
        raise DerivationDataLossError(
            f"conditioning_spec at {loaded_path} is not a JSON/YAML object"
        )
    # YAML with the {"conditioning": {...}} wrapper is the offline-
    # prefix-fixture shape; pass through the upstream lowering helper.
    if is_yaml and "conditioning" in data:
        wrapper = data["conditioning"]
        if not isinstance(wrapper, Mapping):
            raise DerivationDataLossError(
                f"conditioning_spec at {loaded_path}: 'conditioning' value "
                f"is not a mapping"
            )
        try:
            return lower_conditioning_cell(conditioning_payload=wrapper)
        except (KeyError, ValueError, TypeError, ValidationError) as exc:
            raise DerivationDataLossError(
                f"conditioning_spec at {loaded_path} failed lowering via "
                f"lower_conditioning_cell: {exc}"
            ) from exc
    try:
        return spec_from_json_dict(data)
    except (KeyError, ValueError, TypeError) as exc:
        raise DerivationDataLossError(
            f"conditioning_spec at {loaded_path} failed reconstruction via "
            f"spec_from_json_dict: {exc}"
        ) from exc


def _materialized_ledger_as_typed(
    materialized: Mapping[str, Any],
    *,
    trajectory_name: str,
) -> list[MaterializedLedgerEvent]:
    """Validate the materialized ledger's events as typed
    ``MaterializedLedgerEvent`` so they can be passed to
    ``future_constraints_audit`` (which is typed on the materialized
    ledger, not raw dicts).

    Strict: this runs on accepted-trajectory data, so any shape problem
    is data loss for ``constraint_match_ids`` downstream. Missing ``ledger``
    key, non-list ``ledger``, non-mapping entries, or
    ``MaterializedLedgerEvent`` validation failures all raise
    ``DerivationDataLossError``. The runner's own acceptance writes the
    materialized trajectory via Pydantic ``model_dump``, so any failure
    here signals real corruption.
    """
    raw_ledger = materialized.get("ledger")
    if raw_ledger is None:
        raise DerivationDataLossError(
            f"accepted trajectory {trajectory_name}: "
            f"materialized_trajectory.json has no 'ledger' key"
        )
    if not isinstance(raw_ledger, Sequence) or isinstance(raw_ledger, (str, bytes)):
        raise DerivationDataLossError(
            f"accepted trajectory {trajectory_name}: "
            f"materialized.ledger is not a list (got {type(raw_ledger).__name__})"
        )
    typed: list[MaterializedLedgerEvent] = []
    for index, entry in enumerate(raw_ledger):
        if not isinstance(entry, Mapping):
            raise DerivationDataLossError(
                f"accepted trajectory {trajectory_name}: "
                f"materialized.ledger[{index}] is not a mapping "
                f"(got {type(entry).__name__})"
            )
        try:
            typed.append(MaterializedLedgerEvent.model_validate(entry))
        except ValidationError as exc:
            raise DerivationDataLossError(
                f"accepted trajectory {trajectory_name}: "
                f"materialized.ledger[{index}] failed MaterializedLedgerEvent "
                f"validation: {exc}"
            ) from exc
    return typed


# ---------- B1b: shared spec-derived helpers ----------


_LEGACY_SPEC_HASHES: tuple[str, str, str, str] = (
    LEGACY_NO_SPEC_SENTINEL,
    EMPTY_PREFIX_SENTINEL,
    EMPTY_CONSTRAINTS_SENTINEL,
    LEGACY_NO_SPEC_SENTINEL,
)


def _spec_hashes(spec: ConditioningSpec | None) -> tuple[str, str, str, str]:
    """Return ``(spec_baseline_hash, spec_observed_prefix_hash,
    spec_future_constraint_hash, spec_conditioning_hash)``.

    Sentinel tuple when ``spec is None`` (legacy runs without
    conditioning_spec.json on disk, or attempts that don't carry one).
    """
    if spec is None:
        return _LEGACY_SPEC_HASHES
    return (
        compute_baseline_hash(spec),
        compute_observed_prefix_hash(spec),
        compute_future_constraint_hash(spec),
        compute_conditioning_spec_hash(spec),
    )


def _spec_hashes_for_artifact(
    artifact: TrajectoryArtifact,
) -> tuple[str, str, str, str]:
    """Convenience wrapper around ``_spec_hashes`` for
    ``TrajectoryArtifact``."""
    return _spec_hashes(artifact.conditioning_spec)


def _observed_prefix_event_count(artifact: TrajectoryArtifact) -> int:
    """Number of prefix events in ``artifact.conditioning_spec.observed_prefix``.

    Zero when spec is absent or the spec has no observed prefix. Used by
    ``event_source`` column derivation: events at ledger positions
    ``[0..N-1]`` are ``observed_prefix``; the rest are ``generated``.
    """
    spec = artifact.conditioning_spec
    if spec is None or spec.observed_prefix is None:
        return 0
    return len(spec.observed_prefix.events)


def _constraint_match_ids_by_event_id(
    artifact: TrajectoryArtifact,
) -> dict[str, list[str]]:
    """Invert ``constraint_results`` into ``event_id -> sorted list of
    constraint_ids whose ``matched_event_ids`` contains that event``.

    Empty dict when ``constraint_results is None``. CSV-side cells fall
    back to ``"[]"`` when an event_id is absent from this map.
    """
    if artifact.constraint_results is None:
        return {}
    by_event_id: dict[str, set[str]] = {}
    for match in artifact.constraint_results:
        for event_id in match.matched_event_ids:
            by_event_id.setdefault(event_id, set()).add(match.constraint_id)
    return {eid: sorted(cids) for eid, cids in by_event_id.items()}


def _read_json_strict(path: Path, *, kind: str) -> Any:
    """Strict reader — raises DerivationDataLossError on missing or malformed JSON.

    Use for accepted-trajectory artifacts (``materialized_trajectory.json``,
    ``parsed_ledger.json``) where silent absence would corrupt downstream
    analysis.
    """
    if not path.exists():
        raise DerivationDataLossError(
            f"accepted trajectory is missing required {kind} at {path}"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DerivationDataLossError(
            f"accepted trajectory has malformed {kind} at {path}: {exc}"
        ) from exc
    except OSError as exc:
        raise DerivationDataLossError(
            f"could not read accepted trajectory {kind} at {path}: {exc}"
        ) from exc


def _extract_tumor_oncotree_code(baseline: Mapping[str, Any] | None) -> str:
    if not isinstance(baseline, Mapping):
        return DEFAULT_TUMOR_ONCOTREE_CODE
    identity = baseline.get("identity_card")
    if not isinstance(identity, Mapping):
        return DEFAULT_TUMOR_ONCOTREE_CODE
    diagnosis = identity.get("diagnosis")
    if not isinstance(diagnosis, Mapping):
        return DEFAULT_TUMOR_ONCOTREE_CODE
    code = diagnosis.get("oncotree_code")
    return str(code) if code else DEFAULT_TUMOR_ONCOTREE_CODE


def discover_trajectories(
    run_root: Path,
    *,
    include_rejected: bool = False,
) -> list[TrajectoryArtifact]:
    """Walk ``<run_root>/trajectories/`` and build TrajectoryArtifact records.

    Behavior:
    - Skips dry-run trajectories (``failure_kind == "dry_run"``).
    - Skips trajectories missing ``trajectory_result.json`` (malformed run).
    - With ``include_rejected=False`` (default), skips trajectories whose
      ``accepted`` flag is ``False``.

    ``conditioning_label`` and ``conditioning_hash`` are read per trajectory
    from ``trajectory_result.json[cell]``, since they vary per
    (condition, policy) cell post-Commit-B.
    """
    trajectories_root = run_root / "trajectories"
    if not trajectories_root.exists() or not trajectories_root.is_dir():
        raise FileNotFoundError(
            f"trajectories directory not found at {trajectories_root}"
        )

    artifacts: list[TrajectoryArtifact] = []
    for traj_dir in sorted(trajectories_root.iterdir()):
        if not traj_dir.is_dir():
            continue
        result = _read_json_if_exists(traj_dir / "trajectory_result.json")
        if not isinstance(result, Mapping):
            continue
        if result.get("failure_kind") == "dry_run":
            continue
        accepted = bool(result.get("accepted"))
        if not accepted and not include_rejected:
            continue

        cell_data = result.get("cell") or {}
        if not isinstance(cell_data, Mapping):
            continue

        materialized: Mapping[str, Any] | None = None
        parsed_ledger: tuple[Mapping[str, Any], ...] = ()
        conditioning_spec: ConditioningSpec | None = None
        constraint_results: tuple[ConstraintMatch, ...] | None = None
        attempts_to_accept = result.get("attempts_to_accept")
        if accepted:
            # Accepted trajectories MUST have both materialized_trajectory.json
            # and parsed_ledger.json under the accepting attempt directory;
            # silent absence would drop real scientific data from the rollups.
            if attempts_to_accept is None:
                raise DerivationDataLossError(
                    f"accepted trajectory {traj_dir.name} has no "
                    "attempts_to_accept in trajectory_result.json"
                )
            attempt_dir = traj_dir / f"attempt_{int(attempts_to_accept):02d}"
            mat = _read_json_strict(
                attempt_dir / "materialized_trajectory.json",
                kind="materialized_trajectory.json",
            )
            if not isinstance(mat, Mapping):
                raise DerivationDataLossError(
                    f"materialized_trajectory.json for accepted trajectory "
                    f"{traj_dir.name} is not a JSON object"
                )
            materialized = mat
            parsed = _read_json_strict(
                attempt_dir / "parsed_ledger.json",
                kind="parsed_ledger.json",
            )
            if not isinstance(parsed, list):
                raise DerivationDataLossError(
                    f"parsed_ledger.json for accepted trajectory "
                    f"{traj_dir.name} is not a JSON array"
                )
            parsed_ledger = tuple(e for e in parsed if isinstance(e, Mapping))
            # B1b: load the generalized conditioning spec persisted per
            # attempt by the runner. None = legacy sentinel-backfill path;
            # present = downstream column builders compute real
            # ``event_source`` / ``constraint_match_ids`` / four ``spec_*``
            # hashes. Re-running ``future_constraints_audit`` here (rather
            # than threading it through the runner's result) keeps the
            # derivation layer pure-read.
            conditioning_spec = _load_conditioning_spec_for_attempt(attempt_dir)
            if conditioning_spec is not None:
                typed_ledger = _materialized_ledger_as_typed(
                    materialized, trajectory_name=traj_dir.name
                )
                constraint_results = tuple(
                    future_constraints_audit(conditioning_spec, typed_ledger)
                )

        tumor_oncotree_code = _extract_tumor_oncotree_code(
            materialized.get("baseline") if materialized else None
        )

        cell = CellKey(
            trajectory_id=str(result.get("trajectory_dir") or traj_dir.name),
            condition_id=str(cell_data.get("condition_id") or ""),
            policy_id=str(cell_data.get("policy_id") or ""),
            model_id=str(cell_data.get("model_id") or ""),
            model_short=str(cell_data.get("model_short") or ""),
            future_index=int(result.get("future_index") or 0),
            tumor_oncotree_code=tumor_oncotree_code,
            conditioning_label=str(
                cell_data.get("conditioning_label") or DEFAULT_CONDITIONING_LABEL
            ),
            conditioning_hash=str(cell_data.get("conditioning_hash") or ""),
            accepted=accepted,
        )
        artifacts.append(
            TrajectoryArtifact(
                cell=cell,
                trajectory_dir=traj_dir,
                parsed_ledger=parsed_ledger,
                materialized=materialized,
                trajectory_result=result,
                conditioning_spec=conditioning_spec,
                constraint_results=constraint_results,
            )
        )

    return artifacts


# ---------- B1b: per-attempt discovery ----------


@dataclass(frozen=True, slots=True)
class AttemptArtifact:
    """One generation attempt (accepted or failed).

    ``cell.accepted`` reflects this attempt's outcome (NOT the parent
    trajectory's overall accept/reject — failed attempts on a trajectory
    that eventually accepts still report ``cell.accepted=False`` on
    their own row).

    ``tumor_oncotree_code`` is recovered from the parent trajectory's
    accepted-attempt materialized baseline when any sibling attempt is
    accepted; failed-only trajectories use the ``DEFAULT_UNKNOWN_TUMOR``
    sentinel.

    ``conditioning_spec`` is loaded from this attempt's own
    ``conditioning_spec.json`` (or YAML fallback) — None for legacy
    attempts predating the B1b runner write.
    """

    cell: CellKey
    trajectory_dir: Path
    attempt_dir: Path
    attempt_index: int
    result: Mapping[str, Any]
    conditioning_spec: ConditioningSpec | None
    prompt_hash: str | None = None


_ATTEMPT_DIR_RE = re.compile(r"^attempt_\d+$")


def _trajectory_tumor_oncotree_code(
    traj_dir: Path,
    traj_result: Mapping[str, Any],
) -> str:
    """Recover the trajectory's tumor_oncotree_code from any accepted
    sibling attempt's materialized baseline.

    Returns ``DEFAULT_UNKNOWN_TUMOR`` sentinel when no attempt is
    accepted (so all sibling failed attempts get the sentinel) OR when
    the materialized baseline is unreadable.
    """
    attempts_to_accept = traj_result.get("attempts_to_accept")
    if attempts_to_accept is None:
        return DEFAULT_UNKNOWN_TUMOR
    try:
        accepted_index = int(attempts_to_accept)
    except (TypeError, ValueError):
        return DEFAULT_UNKNOWN_TUMOR
    accepted_attempt_dir = traj_dir / f"attempt_{accepted_index:02d}"
    mat = _read_json_if_exists(
        accepted_attempt_dir / "materialized_trajectory.json"
    )
    if not isinstance(mat, Mapping):
        return DEFAULT_UNKNOWN_TUMOR
    baseline = mat.get("baseline")
    code = _extract_tumor_oncotree_code(baseline if isinstance(baseline, Mapping) else None)
    return code or DEFAULT_UNKNOWN_TUMOR


def discover_attempts(run_root: Path) -> list[AttemptArtifact]:
    """Walk ``<run_root>/trajectories/*/attempt_*/`` for EVERY attempt
    (accepted + failed).

    Distinct from ``discover_trajectories`` (which is per-trajectory and
    skips rejected trajectories by default) — this is per-attempt and
    surfaces failed attempts on accepted trajectories AS WELL AS attempts
    on never-accepted trajectories. Powers Family 13 (Run validity and
    methods QC).

    Cell key: ``condition_id`` / ``policy_id`` / ``model_id`` /
    ``model_short`` / ``conditioning_label`` / ``conditioning_hash`` are
    read from the parent trajectory's ``trajectory_result.json[cell]``;
    ``future_index`` from the result root; ``tumor_oncotree_code`` from
    any accepted sibling attempt's materialized baseline (sentinel
    ``DEFAULT_UNKNOWN_TUMOR`` when none). ``accepted`` reflects the
    per-attempt outcome from ``result.json``.

    ``conditioning_spec`` is loaded from the attempt's own
    ``conditioning_spec.json`` (or YAML fallback). Attempts whose
    ``result.json`` file is absent (interrupted/killed attempts that never
    produced an outcome) are skipped with a warning; a present-but-malformed
    ``result.json`` still raises ``DerivationDataLossError``.
    """
    trajectories_root = run_root / "trajectories"
    if not trajectories_root.exists() or not trajectories_root.is_dir():
        raise FileNotFoundError(
            f"trajectories directory not found at {trajectories_root}"
        )

    artifacts: list[AttemptArtifact] = []
    for traj_dir in sorted(trajectories_root.iterdir()):
        if not traj_dir.is_dir():
            continue
        traj_result = _read_json_if_exists(traj_dir / "trajectory_result.json")
        if not isinstance(traj_result, Mapping):
            continue
        if traj_result.get("failure_kind") == "dry_run":
            continue
        cell_data = traj_result.get("cell") or {}
        if not isinstance(cell_data, Mapping):
            continue

        tumor_oncotree_code = _trajectory_tumor_oncotree_code(traj_dir, traj_result)

        attempt_dirs = sorted(
            p for p in traj_dir.iterdir()
            if p.is_dir() and _ATTEMPT_DIR_RE.match(p.name)
        )
        for attempt_dir in attempt_dirs:
            # attempt_rows powers QC denominators (acceptance rate,
            # failure-kind distribution). A *present-but-malformed*
            # result.json is genuine corruption and still raises (silent
            # undercount is risky). But an attempt_* dir with NO result.json
            # at all is an interrupted/killed attempt that never produced an
            # outcome (e.g. transport_or_runtime trajectories or a stale
            # second writer). Those produced nothing classifiable, so they
            # are skipped with a warning rather than aborting the whole
            # derivation — matching this function's documented intent.
            result_path = attempt_dir / "result.json"
            if not result_path.exists():
                warnings.warn(
                    f"skipping attempt with no result.json (interrupted "
                    f"attempt): {attempt_dir}",
                    stacklevel=2,
                )
                continue
            result = _read_json_strict(result_path, kind="result.json")
            if not isinstance(result, Mapping):
                raise DerivationDataLossError(
                    f"attempt {attempt_dir} has malformed result.json "
                    f"(expected JSON object, got {type(result).__name__})"
                )
            attempt_index_raw = result.get("attempt_index")
            try:
                attempt_index = int(attempt_index_raw) if attempt_index_raw is not None else int(
                    attempt_dir.name.rsplit("_", 1)[-1]
                )
            except (TypeError, ValueError) as exc:
                raise DerivationDataLossError(
                    f"attempt {attempt_dir}: result.json attempt_index "
                    f"is not int-convertible ({attempt_index_raw!r})"
                ) from exc
            cell = CellKey(
                trajectory_id=str(traj_result.get("trajectory_dir") or traj_dir.name),
                condition_id=str(cell_data.get("condition_id") or ""),
                policy_id=str(cell_data.get("policy_id") or ""),
                model_id=str(cell_data.get("model_id") or ""),
                model_short=str(cell_data.get("model_short") or ""),
                future_index=int(traj_result.get("future_index") or 0),
                tumor_oncotree_code=tumor_oncotree_code,
                conditioning_label=str(
                    cell_data.get("conditioning_label") or DEFAULT_CONDITIONING_LABEL
                ),
                conditioning_hash=str(cell_data.get("conditioning_hash") or ""),
                accepted=bool(result.get("accepted")),
            )
            conditioning_spec = _load_conditioning_spec_for_attempt(attempt_dir)
            artifacts.append(
                AttemptArtifact(
                    cell=cell,
                    trajectory_dir=traj_dir,
                    attempt_dir=attempt_dir,
                    attempt_index=attempt_index,
                    result=result,
                    conditioning_spec=conditioning_spec,
                    prompt_hash=(
                        str(traj_result.get("prompt_hash"))
                        if traj_result.get("prompt_hash") is not None
                        else None
                    ),
                )
            )

    return artifacts


# ---------- helpers ----------


def _days_from(diagnosis_date: date_cls | None, event_date: date_cls | None) -> int | None:
    if diagnosis_date is None or event_date is None:
        return None
    return (event_date - diagnosis_date).days


def _diagnosis_date(materialized: Mapping[str, Any] | None) -> date_cls | None:
    if not isinstance(materialized, Mapping):
        return None
    baseline = materialized.get("baseline")
    if not isinstance(baseline, Mapping):
        return None
    identity = baseline.get("identity_card") or {}
    if not isinstance(identity, Mapping):
        return None
    diagnosis = identity.get("diagnosis") or {}
    if not isinstance(diagnosis, Mapping):
        return None
    return _parse_date(diagnosis.get("diagnosis_date"))


def _outcome_block(materialized: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(materialized, Mapping):
        return None
    snapshots = materialized.get("snapshots") or []
    if not snapshots:
        return None
    final = snapshots[-1]
    if not isinstance(final, Mapping):
        return None
    outcome = final.get("outcome")
    return outcome if isinstance(outcome, Mapping) else None


def _materialized_ledger(materialized: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(materialized, Mapping):
        return ()
    ledger = materialized.get("ledger") or ()
    return tuple(e for e in ledger if isinstance(e, Mapping))


def _materialized_snapshots(materialized: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(materialized, Mapping):
        return ()
    snapshots = materialized.get("snapshots") or ()
    return tuple(s for s in snapshots if isinstance(s, Mapping))


def _materialized_treatment_history(
    materialized: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(materialized, Mapping):
        return ()
    history = materialized.get("treatment_history") or ()
    return tuple(c for c in history if isinstance(c, Mapping))


def _join_list(value: Any) -> str | None:
    """Stable CSV rendering for emitted string-list fields."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    items = [str(item) for item in value if item is not None]
    return ";".join(items) if items else None


def _json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, default=str)


def _toxicity_grade(event: Mapping[str, Any]) -> int | None:
    """Return the paper-v1 CTCAE v5 grade, with a legacy fallback."""
    grade = event.get("ctcae_v5_grade", event.get("ctcae_grade"))
    if isinstance(grade, bool) or not isinstance(grade, int | float):
        return None
    return int(grade)


# ---------- event_rows.csv ----------

# Stable column order. Most cells are null on any given row because event types
# only populate event-type-specific fields. Order chosen for human readability:
# cell key → universal event fields → systemic treatment lifecycle →
# response/progression (disease delta only via materialized_payload_json) →
# toxicity → clonal evolution → patient-state update → local intervention →
# death → raw payload.
_EVENT_COLUMNS: tuple[str, ...] = (
    "trajectory_id", "condition_id", "policy_id", "model_id", "model_short",
    "future_index", "tumor_oncotree_code", "conditioning_label", "conditioning_hash", "accepted",
    "event_index", "event_id", "event_type", "event_date", "days_from_diagnosis",
    "recognized_date",
    # treatment lifecycle
    "line", "regimen", "treatment_id", "stopped_treatment_id", "reason",
    "modification_type", "affected_components",
    # toxicity
    "ae", "ae_category", "ae_category_description",
    "toxicity_transition", "ctcae_v5_grade",
    "causes", "resolved_treatment_id",
    # clonal_evolution_event
    "new_clone_id", "gene", "alteration", "alteration_type", "variant",
    "driver_bucket", "actionability", "mechanism_bucket", "detection_method",
    "sample_source", "sample_source_description", "parent_clone", "mechanism",
    "assigned_clone_id", "alteration_id",
    # patient_state_update
    "state_change_cause", "rationale",
    # local_intervention
    "procedure", "target_sites", "intent", "modality", "modality_description",
    # death
    "cause_category", "cause", "eol_care_pattern", "eol_care_description",
    # raw payload for downstream re-analysis without re-walking JSONs
    "materialized_payload_json",
    # B1b: conditioning provenance per event. event_source is index-based
    # ("observed_prefix" for events at materialized.ledger[0..N-1] where N
    # is the prefix-event count, else "generated"). constraint_match_ids
    # is a JSON-encoded list of constraint_ids whose
    # ConstraintMatch.matched_event_ids contains this event's event_id.
    # Sentinels: "generated" / "[]" for legacy runs without a spec.
    "event_source", "constraint_match_ids",
)


# Canonical EventType values, sourced from the replay enum so the two stay in
# lockstep. `clinical_decision` and any other historical type that B1b removed
# fall outside this set and resolve to `event_type=None` in the derivation
# adapter below — the raw original is still recoverable via
# `materialized_payload_json`.
_CANONICAL_EVENT_TYPES: frozenset[str] = frozenset(typing.get_args(EventType))

# Snapshots additionally use the literal "baseline" anchor (the pre-event
# starting state). Not in EventType because nothing emits it as an event.
_VALID_SNAPSHOT_ANCHOR_TYPES: frozenset[str] = _CANONICAL_EVENT_TYPES | {"baseline"}


def _canonicalize_event_for_derivation(event: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize an event dict for derivation CSV emission.

    Applies the shared `canonicalize_legacy_event_dict` from replay (so the
    two paths stay coherent), then null-types any event whose `type` cannot
    be mapped to the canonical EventType enum. The raw original is preserved
    separately by the caller into `materialized_payload_json`.
    """
    canonical = canonicalize_legacy_event_dict(event)
    if canonical.get("type") not in _CANONICAL_EVENT_TYPES:
        canonical["type"] = None
    return canonical


def _canonicalize_snapshot_anchor_event_type(value: Any) -> str | None:
    """Canonicalize a snapshot's anchor event_type for CSV emission.

    Reuses the shared replay legacy-type map. Unmappable types resolve to
    None (mirroring the event-row rule). `baseline` is preserved verbatim.
    """
    if not isinstance(value, str):
        return None
    canonical = canonicalize_legacy_event_dict({"type": value}).get("type")
    if canonical not in _VALID_SNAPSHOT_ANCHOR_TYPES:
        return None
    return canonical


def _event_row(
    cell: CellKey,
    event: Mapping[str, Any],
    event_index: int,
    diagnosis_date: date_cls | None,
    *,
    event_source: str,
    constraint_match_ids_json: str,
) -> dict[str, Any]:
    canonical_event = _canonicalize_event_for_derivation(event)
    event_date = _parse_date(canonical_event.get("date"))
    row: dict[str, Any] = {col: None for col in _EVENT_COLUMNS}
    row.update(cell.as_columns())
    row["event_index"] = event_index
    row["event_id"] = canonical_event.get("event_id")
    row["event_type"] = canonical_event.get("type")
    row["event_date"] = event_date.isoformat() if event_date else None
    row["days_from_diagnosis"] = _days_from(diagnosis_date, event_date)
    rec_date = _parse_date(canonical_event.get("recognized_date"))
    row["recognized_date"] = rec_date.isoformat() if rec_date else None
    # `line` is intentionally popped by canonicalize_legacy_event_dict (so the
    # Pydantic LedgerEvent can validate without the legacy emitted field).
    # Derivation still needs the model's emitted line for provenance — fall
    # back to the raw event when the canonical copy has dropped it.
    row["line"] = canonical_event.get("line", event.get("line"))
    row["regimen"] = _regimen_label(canonical_event.get("regimen"))
    row["treatment_id"] = canonical_event.get("treatment_id")
    row["stopped_treatment_id"] = canonical_event.get("stopped_treatment_id")
    row["reason"] = canonical_event.get("reason")
    row["modification_type"] = canonical_event.get("modification_type")
    row["affected_components"] = _join_list(canonical_event.get("affected_components"))
    row["ae"] = canonical_event.get("ae")
    row["ae_category"] = canonical_event.get("ae_category")
    row["ae_category_description"] = canonical_event.get("ae_category_description")
    row["toxicity_transition"] = canonical_event.get("toxicity_transition")
    row["ctcae_v5_grade"] = _toxicity_grade(canonical_event)
    row["causes"] = _join_list(canonical_event.get("causes"))
    row["resolved_treatment_id"] = canonical_event.get("resolved_treatment_id")
    row["new_clone_id"] = canonical_event.get("new_clone_id")
    row["gene"] = canonical_event.get("gene")
    row["alteration"] = canonical_event.get("alteration")
    row["alteration_type"] = canonical_event.get("alteration_type")
    row["variant"] = canonical_event.get("variant")
    row["driver_bucket"] = canonical_event.get("driver_bucket")
    row["actionability"] = canonical_event.get("actionability")
    row["mechanism_bucket"] = canonical_event.get("mechanism_bucket")
    row["detection_method"] = canonical_event.get("detection_method")
    row["sample_source"] = canonical_event.get("sample_source")
    row["sample_source_description"] = canonical_event.get("sample_source_description")
    row["parent_clone"] = canonical_event.get("parent_clone")
    row["mechanism"] = canonical_event.get("mechanism")
    row["assigned_clone_id"] = canonical_event.get("assigned_clone_id")
    row["alteration_id"] = canonical_event.get("alteration_id")
    row["state_change_cause"] = canonical_event.get("state_change_cause")
    row["rationale"] = canonical_event.get("rationale")
    row["procedure"] = canonical_event.get("procedure")
    row["target_sites"] = _join_list(canonical_event.get("target_sites"))
    row["intent"] = canonical_event.get("intent")
    row["modality"] = canonical_event.get("modality")
    row["modality_description"] = canonical_event.get("modality_description")
    row["cause_category"] = canonical_event.get("cause_category")
    row["cause"] = canonical_event.get("cause")
    row["eol_care_pattern"] = canonical_event.get("eol_care_pattern")
    row["eol_care_description"] = canonical_event.get("eol_care_description")
    # Provenance: raw original event preserved verbatim. The legacy original
    # `type` (e.g. "clinical_decision") remains recoverable here even when
    # `event_type` above is None.
    row["materialized_payload_json"] = json.dumps(event, sort_keys=True, default=str)
    row["event_source"] = event_source
    row["constraint_match_ids"] = constraint_match_ids_json
    return row


def build_event_rows(artifacts: Sequence[TrajectoryArtifact]) -> pd.DataFrame:
    """One row per materialized ledger event across all trajectories.

    Uses the materialized ledger (not parsed_ledger) so event_id and
    pipeline-resolved fields (resolved_treatment_id, assigned_clone_id,
    alteration_id) are present.

    B1b: ``event_source`` is index-based against
    ``len(artifact.conditioning_spec.observed_prefix.events)`` (0 when no
    prefix). ``constraint_match_ids`` is inverted from
    ``artifact.constraint_results`` per event_id and JSON-encoded as a
    sorted list (``"[]"`` when no matches). Sentinels for legacy runs
    without a spec on disk.
    """
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        diagnosis_date = _diagnosis_date(artifact.materialized)
        ledger = _materialized_ledger(artifact.materialized)
        prefix_n = _observed_prefix_event_count(artifact)
        match_ids_by_event_id = _constraint_match_ids_by_event_id(artifact)
        for index, event in enumerate(ledger, start=1):
            # event_index is 1-indexed; prefix occupies materialized.ledger[0..N-1]
            # which is event_index 1..N.
            source = (
                EVENT_SOURCE_OBSERVED_PREFIX
                if index <= prefix_n
                else EVENT_SOURCE_GENERATED
            )
            event_id = event.get("event_id")
            matches = match_ids_by_event_id.get(event_id, []) if event_id else []
            match_json = json.dumps(matches)
            rows.append(
                _event_row(
                    artifact.cell,
                    event,
                    index,
                    diagnosis_date,
                    event_source=source,
                    constraint_match_ids_json=match_json,
                )
            )
    return pd.DataFrame(rows, columns=list(_EVENT_COLUMNS))


# ---------- snapshot_rows.csv ----------

_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "trajectory_id", "condition_id", "policy_id", "model_id", "model_short",
    "future_index", "tumor_oncotree_code", "conditioning_label", "conditioning_hash", "accepted",
    "snapshot_index", "state_id", "snapshot_date", "days_from_diagnosis",
    "anchor_event_type", "anchor_event_id",
    "ecog_ps", "weight_kg", "symptom_burden", "dominant_symptoms",
    "total_burden_band", "num_active_clones", "num_active_disease_sites",
    "active_disease_sites_joined", "active_treatment_regimens_joined",
    "active_treatment_lines_joined", "n_active_aes",
    "n_clones_in_registry", "n_acquired_alterations",
    "vital_status", "outcome_type",
    "organ_renal", "organ_hepatic", "organ_cardiac", "organ_marrow_reserve",
    "organ_pulmonary",
    # B1b: four conditioning-spec hashes (sentinels for legacy runs).
    "spec_baseline_hash", "spec_observed_prefix_hash",
    "spec_future_constraint_hash", "spec_conditioning_hash",
)


def _snapshot_row(
    cell: CellKey,
    snapshot: Mapping[str, Any],
    snapshot_index: int,
    diagnosis_date: date_cls | None,
    *,
    spec_hashes: tuple[str, str, str, str],
) -> dict[str, Any]:
    snap_date = _parse_date(snapshot.get("date"))
    patient_state = snapshot.get("patient_state") or {}
    if not isinstance(patient_state, Mapping):
        patient_state = {}
    organ = patient_state.get("organ_function") or {}
    if not isinstance(organ, Mapping):
        organ = {}

    clonal_state = snapshot.get("clonal_state") or []
    active_sites: list[str] = []
    clones_seen: set[str] = set()
    for entry in clonal_state:
        if not isinstance(entry, Mapping):
            continue
        clone = str(entry.get("clone") or "")
        site = str(entry.get("site") or entry.get("organ") or "")
        burden = str(entry.get("burden") or "")
        if burden and burden != "absent":
            clones_seen.add(clone)
            active_sites.append(f"{clone}:{site}({burden})")

    active_treatments = snapshot.get("active_treatments") or []
    regimens: list[str] = []
    lines: list[int] = []
    for tx in active_treatments:
        if not isinstance(tx, Mapping):
            continue
        if tx.get("regimen"):
            regimens.append(_regimen_label(tx["regimen"]))
        if tx.get("line") is not None:
            lines.append(int(tx["line"]))

    active_aes = snapshot.get("active_aes") or []
    clone_registry = snapshot.get("clone_registry") or []
    acquired = snapshot.get("acquired_alterations") or []
    outcome = snapshot.get("outcome")
    outcome_type = outcome.get("type") if isinstance(outcome, Mapping) else None

    row: dict[str, Any] = {col: None for col in _SNAPSHOT_COLUMNS}
    row.update(cell.as_columns())
    row["snapshot_index"] = snapshot_index
    row["state_id"] = snapshot.get("state_id")
    row["snapshot_date"] = snap_date.isoformat() if snap_date else None
    row["days_from_diagnosis"] = _days_from(diagnosis_date, snap_date)
    row["anchor_event_type"] = _canonicalize_snapshot_anchor_event_type(
        snapshot.get("event_type")
    )
    row["anchor_event_id"] = snapshot.get("event_id")
    row["ecog_ps"] = patient_state.get("ecog_ps", patient_state.get("ecog"))
    row["weight_kg"] = patient_state.get("weight_kg")
    row["symptom_burden"] = patient_state.get("symptom_burden")
    row["dominant_symptoms"] = _join_list(patient_state.get("dominant_symptoms"))
    row["total_burden_band"] = snapshot.get("total_burden_band")
    row["num_active_clones"] = len(clones_seen)
    row["num_active_disease_sites"] = len(active_sites)
    row["active_disease_sites_joined"] = ";".join(active_sites)
    row["active_treatment_regimens_joined"] = ";".join(regimens)
    row["active_treatment_lines_joined"] = ";".join(str(L) for L in lines)
    row["n_active_aes"] = sum(1 for ae in active_aes if isinstance(ae, Mapping))
    row["n_clones_in_registry"] = sum(1 for c in clone_registry if isinstance(c, Mapping))
    row["n_acquired_alterations"] = sum(1 for a in acquired if isinstance(a, Mapping))
    row["vital_status"] = snapshot.get("vital_status")
    row["outcome_type"] = outcome_type
    row["organ_renal"] = organ.get("renal")
    row["organ_hepatic"] = organ.get("hepatic")
    row["organ_cardiac"] = organ.get("cardiac")
    row["organ_marrow_reserve"] = organ.get("marrow_reserve")
    row["organ_pulmonary"] = organ.get("pulmonary")
    (
        row["spec_baseline_hash"],
        row["spec_observed_prefix_hash"],
        row["spec_future_constraint_hash"],
        row["spec_conditioning_hash"],
    ) = spec_hashes
    return row


def build_snapshot_rows(artifacts: Sequence[TrajectoryArtifact]) -> pd.DataFrame:
    """One row per materialized snapshot across all trajectories."""
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        diagnosis_date = _diagnosis_date(artifact.materialized)
        snapshots = _materialized_snapshots(artifact.materialized)
        spec_hashes = _spec_hashes_for_artifact(artifact)
        for index, snap in enumerate(snapshots):
            rows.append(
                _snapshot_row(
                    artifact.cell, snap, index, diagnosis_date,
                    spec_hashes=spec_hashes,
                )
            )
    return pd.DataFrame(rows, columns=list(_SNAPSHOT_COLUMNS))


# ---------- treatment_rows.csv ----------

_TREATMENT_COLUMNS: tuple[str, ...] = (
    "trajectory_id", "condition_id", "policy_id", "model_id", "model_short",
    "future_index", "tumor_oncotree_code", "conditioning_label", "conditioning_hash", "accepted",
    "course_index", "line_number", "regimen_name", "treatment_id",
    "start_date", "end_date", "duration_days", "end_reason", "was_last_line",
    # Generation contract: model-emitted per-component regimen buckets.
    "component_classes_json", "component_classes_joined",
    # B1b: regimen canonicalization (promoted from
    # scripts/build_paper_treatment_pathways.canonicalize_regimen).
    "canonical_regimen",
    # B1b: in-course dose-modification and interruption counts.
    # Matching is course date-window on event_date (not treatment_id;
    # replay does not assign treatment_id to systemic_treatment_modification
    # events, see replay.py:1287). Interruptions = subset where
    # modification_type == 'dose_hold'.
    "n_dose_modifications_in_course", "n_interruptions_in_course",
    # B1b: per-course response-like classification (Rev5 disease-state
    # deltas, NOT RECIST). best_response_class is a 5-value enum
    # {CR_like, PR_like, SD_like, PD, NE} or null when unclassifiable.
    # See _derive_course_response_columns for operational definitions.
    "best_response_class", "had_response_event",
    "n_response_events_in_course", "time_to_first_response_days",
    # B1b: four conditioning-spec hashes (sentinels for legacy runs).
    "spec_baseline_hash", "spec_observed_prefix_hash",
    "spec_future_constraint_hash", "spec_conditioning_hash",
)


@dataclass(frozen=True, slots=True)
class _CourseResponseSignals:
    """Response-related signals derived from the materialized ledger +
    snapshots within a single treatment course's date window."""

    best_response_class: str | None
    had_response_event: bool
    n_response_events: int
    time_to_first_response_days: int | None


def _derive_course_response_columns(
    artifact: TrajectoryArtifact,
    course: TreatmentCourseRow,
) -> _CourseResponseSignals:
    """B1b best_response_class derivation per the v2 plan.

    Anchors the line-end ``total_burden_band`` snapshot to the **terminal
    response event** in the course (snapshot whose ``anchor_event_id``
    matches the last response event in the course, latest such snapshot
    if multiple). Falls back to ``None`` (``__not_classifiable__``) when
    no resolvable anchor.

    PD ⇐ progression event in course.
    NE ⇐ no response AND no progression.
    CR_like ⇐ no progression AND terminal band ∈ {absent, microscopic}.
    PR_like ⇐ no progression AND terminal band strictly lower than
              line-start in BURDEN_BAND_ORDER AND not CR_like.
    SD_like ⇐ no progression AND terminal band == line-start AND
              had_response_event.
    None    ⇐ no resolvable line-start snapshot, or terminal band missing,
              or burden went UP without progression (incoherent edge case).
    """
    ledger = _materialized_ledger(artifact.materialized)
    snapshots = _materialized_snapshots(artifact.materialized)
    start = course.start_date
    end = course.end_date  # may be None for open courses

    # Events within course window
    response_events: list[tuple[date_cls, Mapping[str, Any]]] = []
    progression_in_course = False
    for event in ledger:
        evt_date = _parse_date(event.get("date"))
        if evt_date is None or start is None:
            continue
        if evt_date < start:
            continue
        if end is not None and evt_date > end:
            continue
        evt_type = event.get("type")
        if evt_type == "response":
            response_events.append((evt_date, event))
        elif evt_type == "progression":
            progression_in_course = True

    response_events.sort(key=lambda x: x[0])
    had_response = len(response_events) > 0
    n_response = len(response_events)
    time_to_first_response: int | None = None
    if had_response and start is not None:
        time_to_first_response = (response_events[0][0] - start).days

    if start is None:
        return _CourseResponseSignals(
            best_response_class=None,
            had_response_event=had_response,
            n_response_events=n_response,
            time_to_first_response_days=time_to_first_response,
        )

    if progression_in_course:
        best_class: str | None = "PD"
    elif not had_response:
        best_class = "NE"
    else:
        # Line-start snapshot: first snapshot with date >= course.start_date.
        line_start_band: str | None = None
        for snap in snapshots:
            snap_date = _parse_date(snap.get("date"))
            if snap_date is None or snap_date < start:
                continue
            band = snap.get("total_burden_band")
            if isinstance(band, str):
                line_start_band = band
            break
        # Terminal-response snapshot: snapshot whose anchor_event_id matches
        # the LAST response event in the course; take the latest such snapshot.
        terminal_event_id = response_events[-1][1].get("event_id")
        terminal_band: str | None = None
        if terminal_event_id is not None:
            anchored: list[tuple[date_cls, str | None]] = []
            for snap in snapshots:
                if snap.get("anchor_event_id") != terminal_event_id:
                    continue
                snap_date = _parse_date(snap.get("date"))
                if snap_date is None:
                    continue
                band = snap.get("total_burden_band")
                anchored.append((snap_date, band if isinstance(band, str) else None))
            if anchored:
                anchored.sort(key=lambda x: x[0])
                terminal_band = anchored[-1][1]
        if line_start_band is None or terminal_band is None:
            best_class = None
        elif terminal_band in ("absent", "microscopic"):
            best_class = "CR_like"
        else:
            start_order = _BURDEN_BAND_ORDER.get(line_start_band)
            end_order = _BURDEN_BAND_ORDER.get(terminal_band)
            if start_order is None or end_order is None:
                best_class = None
            elif end_order < start_order:
                best_class = "PR_like"
            elif end_order == start_order:
                best_class = "SD_like"
            else:
                # Burden went up without a progression event — incoherent
                # edge case; spec doesn't cover this. Mark as not
                # classifiable rather than mis-bucketing.
                best_class = None

    return _CourseResponseSignals(
        best_response_class=best_class,
        had_response_event=had_response,
        n_response_events=n_response,
        time_to_first_response_days=time_to_first_response,
    )


def _count_modifications_in_course(
    ledger: Sequence[Mapping[str, Any]],
    course: TreatmentCourseRow,
    *,
    modification_type_filter: str | None = None,
) -> int:
    """Count systemic_treatment_modification events within the course date
    window. ``modification_type_filter`` narrows to a single
    ``modification_type`` value (e.g. ``'dose_hold'`` for interruptions);
    ``None`` counts all modification types.
    """
    if course.start_date is None:
        return 0
    n = 0
    for event in ledger:
        if event.get("type") != "systemic_treatment_modification":
            continue
        evt_date = _parse_date(event.get("date"))
        if evt_date is None or evt_date < course.start_date:
            continue
        if course.end_date is not None and evt_date > course.end_date:
            continue
        if modification_type_filter is not None:
            if event.get("modification_type") != modification_type_filter:
                continue
        n += 1
    return n


def _course_to_row(
    cell: CellKey,
    course: TreatmentCourseRow,
    is_last_line: bool,
    *,
    component_classes: list[str] | None,
    response_signals: _CourseResponseSignals,
    n_dose_modifications_in_course: int,
    n_interruptions_in_course: int,
    spec_hashes: tuple[str, str, str, str],
) -> dict[str, Any]:
    duration_days: int | None = None
    if course.start_date and course.end_date:
        duration_days = (course.end_date - course.start_date).days
    row: dict[str, Any] = {col: None for col in _TREATMENT_COLUMNS}
    row.update(cell.as_columns())
    row["course_index"] = course.course_index
    row["line_number"] = course.line
    row["regimen_name"] = course.regimen
    row["treatment_id"] = course.treatment_id
    row["start_date"] = course.start_date.isoformat() if course.start_date else None
    row["end_date"] = course.end_date.isoformat() if course.end_date else None
    row["duration_days"] = duration_days
    row["was_last_line"] = is_last_line
    row["component_classes_json"] = (
        json.dumps(component_classes, sort_keys=True) if component_classes else None
    )
    row["component_classes_joined"] = _join_list(component_classes)
    row["canonical_regimen"] = _canonicalize_regimen(course.regimen)
    row["n_dose_modifications_in_course"] = n_dose_modifications_in_course
    row["n_interruptions_in_course"] = n_interruptions_in_course
    row["best_response_class"] = response_signals.best_response_class
    row["had_response_event"] = response_signals.had_response_event
    row["n_response_events_in_course"] = response_signals.n_response_events
    row["time_to_first_response_days"] = response_signals.time_to_first_response_days
    (
        row["spec_baseline_hash"],
        row["spec_observed_prefix_hash"],
        row["spec_future_constraint_hash"],
        row["spec_conditioning_hash"],
    ) = spec_hashes
    return row


def _end_reason_for_course(
    course_index: int,
    materialized: Mapping[str, Any] | None,
) -> str | None:
    """Pull end_reason from the materialized treatment_history when available.

    ``derive_treatment_courses`` does not return end_reason (the dataclass is
    older). The replay engine stores it on each closed course in
    ``materialized.treatment_history``, indexed in chronological close-order.
    Open courses (still active at the end) have no end_reason by definition.
    """
    history = _materialized_treatment_history(materialized)
    if course_index < 0 or course_index >= len(history):
        return None
    entry = history[course_index]
    reason = entry.get("end_reason")
    return str(reason) if reason else None


def _regimen_for_course(
    ledger: Sequence[Mapping[str, Any]],
    course: TreatmentCourseRow,
) -> Mapping[str, Any] | None:
    """Return the full model-emitted regimen object for this treatment course.

    ``derive_treatment_courses`` is intentionally lean and only carries the
    display regimen label. Its ``course_index`` is the chronological index of
    systemic treatment start/switch events, so use that same index to recover
    the full regimen object from the materialized ledger.
    """
    start_events = [
        event
        for event in ledger
        if event.get("type")
        in {
            "systemic_treatment_start",
            "systemic_treatment_switch",
            "treatment_start",
            "treatment_switch",
        }
    ]
    if course.course_index < 0 or course.course_index >= len(start_events):
        return None
    regimen = start_events[course.course_index].get("regimen")
    if not isinstance(regimen, Mapping):
        return None
    return regimen


def _component_classes_for_course(
    ledger: Sequence[Mapping[str, Any]],
    course: TreatmentCourseRow,
) -> list[str] | None:
    """Return model-emitted per-component classes for this treatment course."""
    regimen = _regimen_for_course(ledger, course)
    if regimen is None:
        return None
    value = regimen.get("component_classes")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        classes = [str(item) for item in value if item is not None]
        if classes:
            return classes
    return None


def build_treatment_rows(artifacts: Sequence[TrajectoryArtifact]) -> pd.DataFrame:
    """One row per treatment course across all trajectories."""
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        ledger = _materialized_ledger(artifact.materialized)
        courses = derive_treatment_courses(ledger)
        max_line = max((c.line for c in courses), default=0)
        spec_hashes = _spec_hashes_for_artifact(artifact)
        for course in courses:
            is_last_line = course.line == max_line and max_line > 0
            response_signals = _derive_course_response_columns(artifact, course)
            n_dose_mods = _count_modifications_in_course(ledger, course)
            n_interrupts = _count_modifications_in_course(
                ledger, course, modification_type_filter="dose_hold"
            )
            row = _course_to_row(
                artifact.cell, course, is_last_line,
                component_classes=_component_classes_for_course(ledger, course),
                response_signals=response_signals,
                n_dose_modifications_in_course=n_dose_mods,
                n_interruptions_in_course=n_interrupts,
                spec_hashes=spec_hashes,
            )
            row["end_reason"] = _end_reason_for_course(
                course.course_index, artifact.materialized
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=list(_TREATMENT_COLUMNS))


# ---------- treatment_component_rows.csv ----------

_TREATMENT_COMPONENT_COLUMNS: tuple[str, ...] = (
    "trajectory_id", "condition_id", "policy_id", "model_id", "model_short",
    "future_index", "tumor_oncotree_code", "conditioning_label", "conditioning_hash", "accepted",
    "spec_baseline_hash", "spec_observed_prefix_hash",
    "spec_future_constraint_hash", "spec_conditioning_hash",
    "event_index", "event_id", "event_type", "event_date", "line",
    "treatment_id", "regimen_label", "component_index", "component",
    "component_normalized", "component_class", "raw_regimen_json",
)


def _treatment_component_base_row(
    artifact: TrajectoryArtifact,
    event: Mapping[str, Any],
    event_index: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    row.update(artifact.cell.as_columns())
    (
        row["spec_baseline_hash"],
        row["spec_observed_prefix_hash"],
        row["spec_future_constraint_hash"],
        row["spec_conditioning_hash"],
    ) = _spec_hashes_for_artifact(artifact)
    row["event_index"] = event_index
    row["event_id"] = event.get("event_id")
    row["event_type"] = event.get("type")
    event_date = _parse_date(event.get("date"))
    row["event_date"] = event_date.isoformat() if event_date else None
    row["line"] = event.get("line")
    row["treatment_id"] = event.get("treatment_id")
    row["regimen_label"] = _regimen_label(event.get("regimen"))
    row["raw_regimen_json"] = _json_or_none(event.get("regimen"))
    return row


def build_treatment_component_rows(
    artifacts: Sequence[TrajectoryArtifact],
) -> pd.DataFrame:
    """One row per systemic treatment component.

    This table preserves the raw model-emitted component names and per-component
    classes so downstream treatment normalization can be revised without
    re-walking JSON artifacts.
    """
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        ledger = _materialized_ledger(artifact.materialized)
        for event_index, event in enumerate(ledger, start=1):
            if event.get("type") not in {
                "systemic_treatment_start",
                "systemic_treatment_switch",
                "treatment_start",
                "treatment_switch",
            }:
                continue
            regimen = event.get("regimen")
            if not isinstance(regimen, Mapping):
                continue
            components = regimen.get("components")
            if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
                continue
            component_classes = regimen.get("component_classes")
            class_values: Sequence[Any] = ()
            if isinstance(component_classes, Sequence) and not isinstance(
                component_classes, (str, bytes)
            ):
                class_values = component_classes
            base = _treatment_component_base_row(artifact, event, event_index)
            for component_index, component in enumerate(components, start=1):
                if component is None:
                    continue
                row = dict(base)
                row["component_index"] = component_index
                row["component"] = str(component)
                row["component_normalized"] = _normalize_text_key(component)
                class_index = component_index - 1
                row["component_class"] = (
                    str(class_values[class_index])
                    if class_index < len(class_values) and class_values[class_index] is not None
                    else None
                )
                rows.append(row)
    return pd.DataFrame(rows, columns=list(_TREATMENT_COMPONENT_COLUMNS))


# ---------- endpoint_rows.csv ----------

# Lines tracked explicitly in dedicated columns. Anything line>=4 is bucketed
# into ``pfs_line_4plus_days``. TTD/TTNT are tracked for lines 1-2 only,
# which covers the BRCA HER2+ paper-v1 anchor and the LUAD smoke (1L-2L).
# Extending is a column-add later.
_TRACKED_LINES: tuple[int, ...] = (1, 2, 3)
_TTD_LINES: tuple[int, ...] = (1, 2)
_TTNT_PAIRS: tuple[tuple[int, int], ...] = ((1, 2), (2, 3))

_ENDPOINT_COLUMNS: tuple[str, ...] = (
    "trajectory_id", "condition_id", "policy_id", "model_id", "model_short",
    "future_index", "tumor_oncotree_code", "conditioning_label", "conditioning_hash", "accepted",
    "diagnosis_date", "outcome_type", "outcome_date", "outcome_cause",
    "outcome_cause_category", "os_days",
    "pfs_line_1_days", "pfs_line_2_days", "pfs_line_3_days", "pfs_line_4plus_days",
    "ttd_line_1_days", "ttd_line_2_days",
    "ttnt_line_1_to_2_days", "ttnt_line_2_to_3_days",
    "first_progression_days", "first_grade3_ae_days", "first_dose_modification_days",
    "n_treatment_lines", "n_progressions", "n_grade3plus_aes",
    "censored",
    # B1b: per-trajectory rollups across closed courses +
    # time-to-first-treatment + 1L policy-canonical match.
    # first_line_matches_policy_canonical uses _POLICY_CANONICAL_FIRST_LINE;
    # null for unknown policies (do not silently default).
    "n_dose_modifications_total", "n_treatment_interruptions_total",
    "time_diagnosis_to_first_treatment_days",
    "first_line_matches_policy_canonical",
    # B1b: four conditioning-spec hashes (sentinels for legacy runs).
    "spec_baseline_hash", "spec_observed_prefix_hash",
    "spec_future_constraint_hash", "spec_conditioning_hash",
)


def _first_line_matches_policy_canonical(
    cell: CellKey,
    courses: Sequence[TreatmentCourseRow],
) -> bool | None:
    """True/False when the trajectory's 1L canonical regimen matches the
    policy's expected 1L canonical; None when the policy is unknown or
    there is no 1L course."""
    expected = _POLICY_CANONICAL_FIRST_LINE.get(cell.policy_id)
    if expected is None:
        return None
    first_line_courses = [c for c in courses if c.line == 1]
    if not first_line_courses:
        return None
    # Compare the first 1L course's canonical regimen.
    first_course = min(first_line_courses, key=lambda c: c.course_index)
    return _canonicalize_regimen(first_course.regimen) == expected


def _first_treatment_start_by_line(
    ledger: Sequence[Mapping[str, Any]],
) -> dict[int, date_cls]:
    """Return the date of the FIRST systemic treatment start/switch per line."""
    by_line: dict[int, date_cls] = {}
    for event in ledger:
        if event.get("type") not in {
            "systemic_treatment_start",
            "systemic_treatment_switch",
            "treatment_start",
            "treatment_switch",
        }:
            continue
        line = event.get("line")
        if line is None:
            continue
        evt_date = _parse_date(event.get("date"))
        if evt_date is None:
            continue
        line_int = int(line)
        if line_int not in by_line:
            by_line[line_int] = evt_date
    return by_line


def _pfs_for_line(
    line_start_date: date_cls,
    ledger: Sequence[Mapping[str, Any]],
) -> int | None:
    """PFS = days from this line start to earliest progression/death/last_followup."""
    for event in ledger:
        evt_date = _parse_date(event.get("date"))
        if evt_date is None or evt_date <= line_start_date:
            continue
        if event.get("type") in {"progression", "death", "last_followup"}:
            return (evt_date - line_start_date).days
    return None


def _ttd_for_line(
    line: int,
    courses: Sequence[TreatmentCourseRow],
) -> int | None:
    """Time-to-discontinuation: duration of the (first) course at this line."""
    for course in courses:
        if course.line == line and course.start_date and course.end_date:
            return (course.end_date - course.start_date).days
    return None


def _first_event_days(
    ledger: Sequence[Mapping[str, Any]],
    diagnosis_date: date_cls | None,
    predicate,
) -> int | None:
    if diagnosis_date is None:
        return None
    for event in ledger:
        if not predicate(event):
            continue
        evt_date = _parse_date(event.get("date"))
        if evt_date is None:
            continue
        return (evt_date - diagnosis_date).days
    return None


def _endpoint_row(artifact: TrajectoryArtifact) -> dict[str, Any]:
    cell = artifact.cell
    diagnosis_date = _diagnosis_date(artifact.materialized)
    ledger = _materialized_ledger(artifact.materialized)
    courses = derive_treatment_courses(ledger) if ledger else []
    outcome = _outcome_block(artifact.materialized)
    outcome_type = outcome.get("type") if outcome else None
    outcome_date = _parse_date(outcome.get("date")) if outcome else None
    outcome_cause = outcome.get("cause") if outcome else None
    outcome_cause_category = outcome.get("cause_category") if outcome else None
    os_days = _days_from(diagnosis_date, outcome_date)

    line_starts = _first_treatment_start_by_line(ledger)
    pfs_values: dict[int, int | None] = {}
    for line in _TRACKED_LINES:
        start = line_starts.get(line)
        pfs_values[line] = _pfs_for_line(start, ledger) if start else None

    pfs_line_4plus: int | None = None
    line_4plus_starts = [d for L, d in line_starts.items() if L >= 4]
    if line_4plus_starts:
        earliest = min(line_4plus_starts)
        pfs_line_4plus = _pfs_for_line(earliest, ledger)

    ttd_values: dict[int, int | None] = {
        line: _ttd_for_line(line, courses) for line in _TTD_LINES
    }

    ttnt_values: dict[tuple[int, int], int | None] = {}
    for src, dst in _TTNT_PAIRS:
        src_date = line_starts.get(src)
        dst_date = line_starts.get(dst)
        ttnt_values[(src, dst)] = (
            (dst_date - src_date).days
            if src_date is not None and dst_date is not None
            else None
        )

    first_progression_days = _first_event_days(
        ledger, diagnosis_date, lambda e: e.get("type") == "progression"
    )
    first_grade3_ae_days = _first_event_days(
        ledger,
        diagnosis_date,
        lambda e: e.get("type") == "toxicity"
        and (_toxicity_grade(e) is not None and _toxicity_grade(e) >= 3),
    )
    first_dose_modification_days = _first_event_days(
        ledger,
        diagnosis_date,
        lambda e: e.get("type") == "systemic_treatment_modification",
    )

    n_lines = len({c.line for c in courses if c.line is not None})
    n_progressions = sum(1 for e in ledger if e.get("type") == "progression")
    # Count distinct AE names that ever hit G3+. A tox emitted as G2 onset
    # then `grade_change` to G3 counts as one toxicity (matching the metric
    # name `n_grade3plus_aes`) rather than two events. Two separated episodes
    # of the same AE name (e.g., recurrent G3 neutropenia: onset → resolved →
    # onset) also count as one under this rule. Per-episode counting would
    # require either replaying active-toxicity state to attribute each G3+
    # event to its onset, or adding an `onset_event_id` column to the event
    # row schema; neither is wired up in paper-v1.
    g3plus_ae_names: set[str] = set()
    for e in ledger:
        if e.get("type") != "toxicity":
            continue
        grade = _toxicity_grade(e)
        if grade is None or grade < 3:
            continue
        ae_name = e.get("ae")
        if isinstance(ae_name, str) and ae_name.strip():
            g3plus_ae_names.add(ae_name)
    n_grade3plus_aes = len(g3plus_ae_names)

    row: dict[str, Any] = {col: None for col in _ENDPOINT_COLUMNS}
    row.update(cell.as_columns())
    row["diagnosis_date"] = diagnosis_date.isoformat() if diagnosis_date else None
    row["outcome_type"] = outcome_type
    row["outcome_date"] = outcome_date.isoformat() if outcome_date else None
    row["outcome_cause"] = outcome_cause
    row["outcome_cause_category"] = outcome_cause_category
    row["os_days"] = os_days
    row["pfs_line_1_days"] = pfs_values.get(1)
    row["pfs_line_2_days"] = pfs_values.get(2)
    row["pfs_line_3_days"] = pfs_values.get(3)
    row["pfs_line_4plus_days"] = pfs_line_4plus
    row["ttd_line_1_days"] = ttd_values.get(1)
    row["ttd_line_2_days"] = ttd_values.get(2)
    row["ttnt_line_1_to_2_days"] = ttnt_values.get((1, 2))
    row["ttnt_line_2_to_3_days"] = ttnt_values.get((2, 3))
    row["first_progression_days"] = first_progression_days
    row["first_grade3_ae_days"] = first_grade3_ae_days
    row["first_dose_modification_days"] = first_dose_modification_days
    row["n_treatment_lines"] = n_lines
    row["n_progressions"] = n_progressions
    row["n_grade3plus_aes"] = n_grade3plus_aes
    # Under terminal_contract='death_only' every accepted trajectory has
    # outcome_type='death' and censored=False. Kept as a column for runs that
    # use 'death_or_last_followup' (the default at engine level).
    row["censored"] = outcome_type == "last_followup"

    # B1b: per-trajectory rollups across closed courses +
    # time-to-first-treatment + policy match.
    closed_courses = [c for c in courses if c.end_date is not None]
    row["n_dose_modifications_total"] = sum(
        _count_modifications_in_course(ledger, c) for c in closed_courses
    )
    row["n_treatment_interruptions_total"] = sum(
        _count_modifications_in_course(
            ledger, c, modification_type_filter="dose_hold"
        )
        for c in closed_courses
    )
    line_1_start = line_starts.get(1)
    row["time_diagnosis_to_first_treatment_days"] = (
        (line_1_start - diagnosis_date).days
        if line_1_start is not None and diagnosis_date is not None
        else None
    )
    row["first_line_matches_policy_canonical"] = _first_line_matches_policy_canonical(
        cell, courses
    )
    (
        row["spec_baseline_hash"],
        row["spec_observed_prefix_hash"],
        row["spec_future_constraint_hash"],
        row["spec_conditioning_hash"],
    ) = _spec_hashes_for_artifact(artifact)
    return row


def build_endpoint_rows(artifacts: Sequence[TrajectoryArtifact]) -> pd.DataFrame:
    """One row per trajectory with OS / PFS-by-line / TTD / TTNT and event counts."""
    rows = [_endpoint_row(artifact) for artifact in artifacts]
    return pd.DataFrame(rows, columns=list(_ENDPOINT_COLUMNS))


# ---------- baseline_rows.csv ----------

# B1b: one row per accepted trajectory carrying the model's generated
# baseline content + the four conditioning-spec hashes + a content
# hash over the canonicalized materialized baseline (minus volatile
# IDs/metadata) so CF same-baseline grouping has a real handle.
_BASELINE_COLUMNS: tuple[str, ...] = (
    # Cell key (10 cols, mirrors CellKey.as_columns()).
    "trajectory_id", "condition_id", "policy_id", "model_id", "model_short",
    "future_index", "tumor_oncotree_code", "conditioning_label", "conditioning_hash", "accepted",
    # Four conditioning-spec hashes (sentinels for legacy runs).
    "spec_baseline_hash", "spec_observed_prefix_hash",
    "spec_future_constraint_hash", "spec_conditioning_hash",
    # Generated-baseline identity hash for same-patient CF grouping.
    "materialized_baseline_hash",
    # Anchor signature (the spec.pinned_paths that drove this baseline).
    "pinned_paths_joined",
    # Identity columns (replay-model paths in comments).
    "stage",                # identity_card.staging.stage_class
    "histology",            # identity_card.diagnosis.histology_class
    "histology_bucket",     # identity_card.diagnosis.histology_bucket
    "histology_bucket_canonical",
    "histology_bucket_discordance",
    "primary_site",         # identity_card.diagnosis.primary_site
    "primary_site_class",   # identity_card.diagnosis.primary_site_class
    "primary_site_class_canonical",
    "primary_site_class_discordance",
    "primary_site_class_category_error",
    "primary_driver",       # heuristic: first alteration's gene (see below)
    "primary_driver_bucket", # identity_card.molecular_profile.primary_driver_bucket
    "age_at_diagnosis",     # identity_card.demographics.age_at_diagnosis
    "sex",                  # identity_card.demographics.sex
    # Baseline health constraints (identity_card.baseline_health).
    "organ_renal", "organ_hepatic", "organ_cardiac", "organ_marrow_reserve",
    "organ_pulmonary",
    "contraindications_json", "contraindications_joined",
    "frailty_band", "autoimmune_disease_history", "neuropathy_at_baseline",
    # Medical history (all in baseline.medical_history; may be absent on legacy roots).
    "smoking_status",
    "tobacco_exposure_band", "pack_years",
    "n_comorbidities", "comorbidity_categories_joined",
    "comorbidity_controlled_fraction",
    "family_history_categories_joined",
    "n_contextual_factors", "contextual_factor_categories_joined",
    # Baseline molecular workup (powers perfect-patient component 3).
    # Reads .alterations / .biomarkers from the replay model's
    # MolecularProfile (replay.py:291-293) — NOT .key_alterations
    # (legacy schema). Defensive fallback handles either shape.
    "n_baseline_alterations", "n_baseline_biomarkers",
)


_BASELINE_VOLATILE_KEYS: frozenset[str] = frozenset({"baseline_id", "metadata"})


def _materialized_baseline_hash(baseline: Mapping[str, Any]) -> str:
    """SHA-256 over the canonicalized materialized baseline minus
    ``{baseline_id, metadata}`` (volatile IDs / pipeline-owned blocks).

    Includes identity_card + clone_registry + baseline_disease_burden +
    baseline_patient_state + medical_history. Excluding any of these
    would risk joining different generated patients on CF same-baseline
    grouping (per B1b plan v2).
    """
    payload = {
        k: v for k, v in baseline.items() if k not in _BASELINE_VOLATILE_KEYS
    }
    return _sha256_hex(canonical_json(_canonicalize_spec_value(payload)))


def _safe_get_mapping(obj: Any) -> Mapping[str, Any]:
    """Return ``obj`` if it's a Mapping, else an empty dict — for safe
    dotted-path reads against partial baselines."""
    return obj if isinstance(obj, Mapping) else {}


def _alterations_list(molecular_profile: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the alterations list from molecular_profile.

    Reads ``alterations`` (Rev5 replay model) with a defensive fallback
    to ``key_alterations`` (legacy schema shape). Filters to mappings.
    """
    raw = molecular_profile.get("alterations")
    if raw is None:
        raw = molecular_profile.get("key_alterations")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [a for a in raw if isinstance(a, Mapping)]


def _biomarkers_list(molecular_profile: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = molecular_profile.get("biomarkers")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [b for b in raw if isinstance(b, Mapping)]


def _derive_primary_driver(alterations: Sequence[Mapping[str, Any]]) -> str | None:
    """Gene of the alteration marked as the baseline primary driver.

    New generation requires exactly one ``is_primary_driver=true`` alteration.
    The first-gene fallback preserves derivation on legacy materialized roots
    that predate the explicit marker.
    """
    if not alterations:
        return None
    for alteration in alterations:
        if alteration.get("is_primary_driver") is True:
            gene = alteration.get("gene")
            return str(gene) if gene else None
    first = alterations[0]
    gene = first.get("gene")
    return str(gene) if gene else None


_HISTOLOGY_BUCKET_CANONICAL_ALIASES: dict[str, str] = {}

_PRIMARY_SITE_CLASS_CANONICAL_ALIASES: dict[str, str] = {
    # Bidirectional anatomy-vocabulary collision surfaced in the 148-code
    # breadth screen: these are valid SiteClass-style words, but the primary
    # organ analysis wants the stable PrimarySiteClass buckets.
    "bone_marrow": "bone_marrow_blood",
    "lymph_node": "lymphoid_tissue",
    "brain": "central_nervous_system",
    # Mesothelioma is a histology/cancer-type word. Keep it measurable as raw,
    # but count it under the roster-backed pleura primary site.
    "mesothelioma": "pleura",
    # This is a category error for primary_site_class, not an organ. It is
    # accepted in v4 only so the run can measure and flag it instead of
    # hard-rejecting otherwise coherent trajectories.
    "primary_site": "invalid_primary_site_class",
}


def _canonical_bucket(
    value: Any,
    aliases: Mapping[str, str],
) -> tuple[str | None, str | None, bool]:
    raw = str(value).strip() if value is not None else ""
    if not raw:
        return None, None, False
    canonical = aliases.get(raw, raw)
    return raw, canonical, canonical != raw


def _baseline_row(artifact: TrajectoryArtifact) -> dict[str, Any]:
    cell = artifact.cell
    materialized = artifact.materialized or {}
    baseline = _safe_get_mapping(materialized.get("baseline"))
    identity = _safe_get_mapping(baseline.get("identity_card"))
    demographics = _safe_get_mapping(identity.get("demographics"))
    diagnosis = _safe_get_mapping(identity.get("diagnosis"))
    staging = _safe_get_mapping(identity.get("staging"))
    baseline_health = _safe_get_mapping(identity.get("baseline_health"))
    molecular_profile = _safe_get_mapping(identity.get("molecular_profile"))
    medical_history = _safe_get_mapping(baseline.get("medical_history"))

    comorbidities_raw = medical_history.get("comorbidities") or []
    comorbidities = [
        c for c in comorbidities_raw if isinstance(c, Mapping)
    ] if isinstance(comorbidities_raw, Sequence) else []
    family_history_raw = medical_history.get("family_history") or []
    family_history = [
        f for f in family_history_raw if isinstance(f, Mapping)
    ] if isinstance(family_history_raw, Sequence) else []
    contextual_factors_raw = medical_history.get("contextual_factors") or []
    contextual_factors = [
        f for f in contextual_factors_raw if isinstance(f, Mapping)
    ] if isinstance(contextual_factors_raw, Sequence) else []

    alterations = _alterations_list(molecular_profile)
    biomarkers = _biomarkers_list(molecular_profile)

    comorbidity_categories = _join_list(
        sorted({str(c.get("category")) for c in comorbidities if c.get("category")})
    )
    n_comorb = len(comorbidities)
    controlled_fraction: float | None
    if n_comorb > 0:
        controlled = sum(1 for c in comorbidities if c.get("status") == "controlled")
        controlled_fraction = _round_float(controlled / n_comorb)
    else:
        controlled_fraction = None

    fh_categories = _join_list(
        sorted({str(f.get("cancer_type")) for f in family_history if f.get("cancer_type")})
    )
    contextual_factor_categories = _join_list(
        sorted({
            str(f.get("category"))
            for f in contextual_factors
            if f.get("category")
        })
    )

    spec_hashes = _spec_hashes_for_artifact(artifact)
    if artifact.conditioning_spec is not None:
        pinned_paths_joined = _join_list(
            list(artifact.conditioning_spec.pinned_paths)
        )
    else:
        pinned_paths_joined = LEGACY_NO_SPEC_SENTINEL

    row: dict[str, Any] = {col: None for col in _BASELINE_COLUMNS}
    row.update(cell.as_columns())
    (
        row["spec_baseline_hash"],
        row["spec_observed_prefix_hash"],
        row["spec_future_constraint_hash"],
        row["spec_conditioning_hash"],
    ) = spec_hashes
    row["materialized_baseline_hash"] = (
        _materialized_baseline_hash(baseline) if baseline else LEGACY_NO_SPEC_SENTINEL
    )
    row["pinned_paths_joined"] = pinned_paths_joined
    row["stage"] = staging.get("stage_class")
    row["histology"] = diagnosis.get("histology_class")
    (
        row["histology_bucket"],
        row["histology_bucket_canonical"],
        row["histology_bucket_discordance"],
    ) = _canonical_bucket(
        diagnosis.get("histology_bucket"),
        _HISTOLOGY_BUCKET_CANONICAL_ALIASES,
    )
    row["primary_site"] = diagnosis.get("primary_site")
    (
        row["primary_site_class"],
        row["primary_site_class_canonical"],
        row["primary_site_class_discordance"],
    ) = _canonical_bucket(
        diagnosis.get("primary_site_class"),
        _PRIMARY_SITE_CLASS_CANONICAL_ALIASES,
    )
    row["primary_site_class_category_error"] = (
        row["primary_site_class_canonical"] == "invalid_primary_site_class"
    )
    row["primary_driver"] = _derive_primary_driver(alterations)
    row["primary_driver_bucket"] = molecular_profile.get("primary_driver_bucket")
    row["age_at_diagnosis"] = demographics.get("age_at_diagnosis")
    row["sex"] = demographics.get("sex")
    row["organ_renal"] = baseline_health.get("renal")
    row["organ_hepatic"] = baseline_health.get("hepatic")
    row["organ_cardiac"] = baseline_health.get("cardiac")
    row["organ_marrow_reserve"] = baseline_health.get("marrow_reserve")
    row["organ_pulmonary"] = baseline_health.get("pulmonary")
    row["contraindications_json"] = _json_or_none(
        baseline_health.get("contraindications")
    )
    row["contraindications_joined"] = _join_list(
        baseline_health.get("contraindications")
    )
    row["frailty_band"] = baseline_health.get("frailty_band")
    row["autoimmune_disease_history"] = baseline_health.get("autoimmune_disease_history")
    row["neuropathy_at_baseline"] = baseline_health.get("neuropathy_at_baseline")
    row["smoking_status"] = medical_history.get("smoking_status")
    row["tobacco_exposure_band"] = medical_history.get("tobacco_exposure_band")
    row["pack_years"] = medical_history.get("pack_years")
    row["n_comorbidities"] = n_comorb
    row["comorbidity_categories_joined"] = comorbidity_categories
    row["comorbidity_controlled_fraction"] = controlled_fraction
    row["family_history_categories_joined"] = fh_categories
    row["n_contextual_factors"] = len(contextual_factors)
    row["contextual_factor_categories_joined"] = contextual_factor_categories
    row["n_baseline_alterations"] = len(alterations)
    row["n_baseline_biomarkers"] = len(biomarkers)
    return row


def build_baseline_rows(artifacts: Sequence[TrajectoryArtifact]) -> pd.DataFrame:
    """One row per artifact (one row per accepted trajectory).

    Each row carries the generated baseline's identity / medical-history /
    molecular-workup fields plus the four conditioning-spec hashes and
    the ``materialized_baseline_hash`` content hash for CF same-patient
    grouping.
    """
    rows = [_baseline_row(artifact) for artifact in artifacts]
    return pd.DataFrame(rows, columns=list(_BASELINE_COLUMNS))


# ---------- baseline_biomarker_rows.csv / baseline_alteration_rows.csv ----------

_TOKEN_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")

_BIOMARKER_NAME_ALIASES: dict[str, str] = {
    "estrogen_receptor": "er",
    "er_status": "er",
    "progesterone_receptor": "pr",
    "pr_status": "pr",
    "her_2": "her2",
    "her2_status": "her2",
    "human_epidermal_growth_factor_receptor_2": "her2",
    "mismatch_repair": "mmr",
    "mismatch_repair_status": "mmr",
    "microsatellite_instability": "msi",
    "microsatellite_instability_status": "msi",
    "tumor_mutational_burden": "tmb",
    "tumour_mutational_burden": "tmb",
    "pd_l1_expression": "pd_l1",
    "pdl1": "pd_l1",
    "programmed_death_ligand_1": "pd_l1",
    "homologous_recombination_deficiency": "hrd",
}

_BIOMARKER_VALUE_ALIASES: dict[str, str] = {
    "pos": "positive",
    "positive": "positive",
    "detected": "positive",
    "present": "positive",
    "neg": "negative",
    "negative": "negative",
    "not_detected": "negative",
    "absent": "negative",
    "equivocal": "equivocal",
    "borderline": "equivocal",
    "high": "high",
    "low": "low",
    "intermediate": "intermediate",
    "stable": "stable",
    "mss": "mss",
    "microsatellite_stable": "mss",
    "msi_high": "msi_high",
    "msi_h": "msi_high",
    "unstable": "msi_high",
    "deficient": "deficient",
    "dmmr": "deficient",
    "proficient": "proficient",
    "pmmr": "proficient",
    "unknown": "unknown",
}


def _normalize_text_key(value: Any) -> str | None:
    if value is None:
        return None
    token = _TOKEN_NORMALIZE_RE.sub("_", str(value).strip().lower()).strip("_")
    return token or None


def _normalize_biomarker_name(value: Any) -> str | None:
    token = _normalize_text_key(value)
    if token is None:
        return None
    return _BIOMARKER_NAME_ALIASES.get(token, token)


def _normalize_biomarker_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return None
        return str(value)
    token = _normalize_text_key(value)
    if token is None:
        return None
    return _BIOMARKER_VALUE_ALIASES.get(token, token)


_BASELINE_BIOMARKER_COLUMNS: tuple[str, ...] = (
    "trajectory_id", "condition_id", "policy_id", "model_id", "model_short",
    "future_index", "tumor_oncotree_code", "conditioning_label", "conditioning_hash", "accepted",
    "spec_baseline_hash", "spec_observed_prefix_hash",
    "spec_future_constraint_hash", "spec_conditioning_hash",
    "materialized_baseline_hash",
    "biomarker_index", "biomarker_name", "biomarker_name_normalized",
    "biomarker_value", "biomarker_value_normalized", "biomarker_value_type",
    "raw_biomarker_json",
)

_BASELINE_ALTERATION_COLUMNS: tuple[str, ...] = (
    "trajectory_id", "condition_id", "policy_id", "model_id", "model_short",
    "future_index", "tumor_oncotree_code", "conditioning_label", "conditioning_hash", "accepted",
    "spec_baseline_hash", "spec_observed_prefix_hash",
    "spec_future_constraint_hash", "spec_conditioning_hash",
    "materialized_baseline_hash",
    "alteration_index", "gene", "gene_normalized", "alteration_type",
    "alteration", "variant", "driver_bucket", "actionability",
    "is_primary_driver", "raw_alteration_json",
)


def _baseline_detail_base_row(artifact: TrajectoryArtifact) -> dict[str, Any]:
    materialized = artifact.materialized or {}
    baseline = _safe_get_mapping(materialized.get("baseline"))
    row: dict[str, Any] = {}
    row.update(artifact.cell.as_columns())
    (
        row["spec_baseline_hash"],
        row["spec_observed_prefix_hash"],
        row["spec_future_constraint_hash"],
        row["spec_conditioning_hash"],
    ) = _spec_hashes_for_artifact(artifact)
    row["materialized_baseline_hash"] = (
        _materialized_baseline_hash(baseline) if baseline else LEGACY_NO_SPEC_SENTINEL
    )
    return row


def _baseline_molecular_profile(artifact: TrajectoryArtifact) -> Mapping[str, Any]:
    materialized = artifact.materialized or {}
    baseline = _safe_get_mapping(materialized.get("baseline"))
    identity = _safe_get_mapping(baseline.get("identity_card"))
    return _safe_get_mapping(identity.get("molecular_profile"))


def build_baseline_biomarker_rows(
    artifacts: Sequence[TrajectoryArtifact],
) -> pd.DataFrame:
    """One row per generated baseline biomarker.

    Keeps the raw name/value emitted by the model while adding deterministic
    normalized keys for paper-v1 biomarker analyses.
    """
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        base = _baseline_detail_base_row(artifact)
        biomarkers = _biomarkers_list(_baseline_molecular_profile(artifact))
        for index, biomarker in enumerate(biomarkers, start=1):
            value = biomarker.get("value")
            row = dict(base)
            row["biomarker_index"] = index
            row["biomarker_name"] = biomarker.get("name")
            row["biomarker_name_normalized"] = _normalize_biomarker_name(
                biomarker.get("name")
            )
            row["biomarker_value"] = value
            row["biomarker_value_normalized"] = _normalize_biomarker_value(value)
            row["biomarker_value_type"] = type(value).__name__ if value is not None else None
            row["raw_biomarker_json"] = _json_or_none(biomarker)
            rows.append(row)
    return pd.DataFrame(rows, columns=list(_BASELINE_BIOMARKER_COLUMNS))


def build_baseline_alteration_rows(
    artifacts: Sequence[TrajectoryArtifact],
) -> pd.DataFrame:
    """One row per generated baseline molecular alteration."""
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        base = _baseline_detail_base_row(artifact)
        alterations = _alterations_list(_baseline_molecular_profile(artifact))
        for index, alteration in enumerate(alterations, start=1):
            row = dict(base)
            gene = alteration.get("gene")
            row["alteration_index"] = index
            row["gene"] = gene
            row["gene_normalized"] = str(gene).upper() if gene else None
            row["alteration_type"] = alteration.get("alteration_type")
            row["alteration"] = alteration.get("alteration")
            row["variant"] = alteration.get("variant")
            row["driver_bucket"] = alteration.get("driver_bucket")
            row["actionability"] = alteration.get("actionability")
            row["is_primary_driver"] = alteration.get("is_primary_driver")
            row["raw_alteration_json"] = _json_or_none(alteration)
            rows.append(row)
    return pd.DataFrame(rows, columns=list(_BASELINE_ALTERATION_COLUMNS))


# ---------- disease_site_rows.csv ----------

# B1b: one row per active disease site per snapshot. Powers family-2
# (Disease burden and sites) metrics: baseline site-tropism per cancer ×
# model, brain-met-at-baseline rates, longitudinal new-site emergence.
# Replaces string-parsing snapshot_rows.active_disease_sites_joined,
# which the registry forbids going forward.

_DISEASE_SITE_COLUMNS: tuple[str, ...] = (
    # Cell key (10 cols).
    "trajectory_id", "condition_id", "policy_id", "model_id", "model_short",
    "future_index", "tumor_oncotree_code", "conditioning_label", "conditioning_hash", "accepted",
    # Four conditioning-spec hashes (sentinels for legacy runs).
    "spec_baseline_hash", "spec_observed_prefix_hash",
    "spec_future_constraint_hash", "spec_conditioning_hash",
    # Snapshot reference.
    "snapshot_index", "state_id", "snapshot_date", "days_from_diagnosis",
    # Site columns. ``site_id`` is a within-snapshot ordinal — NOT a
    # persistent lesion ID. ``site_class`` preserves the raw emitted enum
    # value; ``site_class_canonical`` is the analysis bucket after v4
    # cross-alias normalization. ``site_signature`` is the SHA-256 hex of
    # canonical (site_class_canonical, site_anatomic_label) and is
    # the equality key powering ``is_new_at_snapshot`` — clone_id is
    # deliberately excluded (metastatic-site-tropism convention) so
    # the same anatomic site under two clones counts as one site.
    "site_id", "clone_id", "site_anatomic_label", "site_class",
    "site_class_canonical", "site_class_discordance",
    "burden_band", "site_signature",
    # Provenance flags.
    "is_baseline", "is_new_at_snapshot",
)


_SITE_CLASS_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Order matters: bone_marrow must be tested before bone;
    # leptomeningeal_cns before brain.
    ("primary_site", ("primary",)),
    ("bone_marrow", ("marrow", "bone marrow", "bm")),
    ("bone", ("bone", "osseous", "vertebr", "skeletal")),
    ("liver", ("liver", "hepatic")),
    ("lung", ("lung", "pulmonary")),
    ("pleura", ("pleura", "pleural")),
    ("leptomeningeal_cns", ("leptomening", "lmd", "csf", "spinal cord", "cns")),
    ("brain", ("brain", "cerebr", "intracrani")),
    ("lymph_node", ("node", "lymph", "ln")),
    ("soft_tissue", ("soft tissue", "muscle", "subcutaneous")),
    ("retroperitoneum", ("retroperitone",)),
    ("adrenal", ("adrenal",)),
    ("peritoneum", ("peritone", "ascit", "omentum", "omental")),
    ("skin", ("skin", "cutaneous", "dermal")),
)

_SITE_CLASS_OTHER = "other"

_SITE_CLASS_CANONICAL_ALIASES: dict[str, str] = {
    # PrimarySiteClass terms leaking into disease-burden site_class.
    "bone_marrow_blood": "bone_marrow",
    "lymphoid_tissue": "lymph_node",
    "central_nervous_system": "brain",
    "soft_tissue_extremity": "soft_tissue",
    "soft_tissue_trunk_chest_wall": "soft_tissue",
    "visceral_soft_tissue": "soft_tissue",
}


def _normalize_site_class(raw_label: Any) -> str:
    """Deterministic mapping from raw anatomic label to a stable site bucket.

    Order-sensitive: ``bone_marrow`` is tested before ``bone``;
    ``leptomeningeal_cns`` before ``brain``. Fallback ``"other"`` on
    miss / null / non-string. Unit-tested against representative real
    labels in the B1b test suite (Step 8).
    """
    if raw_label is None:
        return _SITE_CLASS_OTHER
    text = str(raw_label).lower()
    if not text.strip():
        return _SITE_CLASS_OTHER
    for site_class, keywords in _SITE_CLASS_KEYWORDS:
        for keyword in keywords:
            if keyword in text:
                return site_class
    return _SITE_CLASS_OTHER


def _site_signature(site_class: str, site_anatomic_label: str) -> str:
    """SHA-256 hex over canonical (site_class, site_anatomic_label).

    Equality key for ``is_new_at_snapshot``. ``clone_id`` deliberately
    excluded so the same anatomic site under two clones counts as one
    site for metastatic-tropism analyses.
    """
    payload = [site_class, site_anatomic_label.lower().strip()]
    return _sha256_hex(canonical_json(_canonicalize_spec_value(payload)))


def _disease_site_rows_for_artifact(
    artifact: TrajectoryArtifact,
) -> list[dict[str, Any]]:
    """One row per active disease site (``burden != 'absent'``) per
    snapshot, with rolling per-trajectory ``is_new_at_snapshot``
    bookkeeping."""
    cell = artifact.cell
    cell_columns = cell.as_columns()
    spec_hashes = _spec_hashes_for_artifact(artifact)
    diagnosis_date = _diagnosis_date(artifact.materialized)
    snapshots = _materialized_snapshots(artifact.materialized)

    # Per-trajectory rolling set of site_signatures from PRIOR snapshots
    # only. Within-snapshot duplicates (same site_class+label across
    # multiple clones, where clone_id is excluded from the signature) all
    # count as "not new" if the signature already appeared earlier, and
    # all count as "new" if this is the signature's first appearance —
    # the registry semantic is "not seen in any earlier snapshot," not
    # "not seen in any earlier ROW within the same snapshot."
    seen_signatures: set[str] = set()
    rows: list[dict[str, Any]] = []
    for snapshot_index, snapshot in enumerate(snapshots):
        snap_date = _parse_date(snapshot.get("date"))
        days_from_dx = _days_from(diagnosis_date, snap_date)
        clonal_state = snapshot.get("clonal_state") or []
        if not isinstance(clonal_state, Sequence) or isinstance(
            clonal_state, (str, bytes)
        ):
            continue
        snapshot_signatures: set[str] = set()
        site_id = 0
        for entry in clonal_state:
            if not isinstance(entry, Mapping):
                continue
            burden = entry.get("burden")
            if burden is None or burden == "absent":
                continue
            burden_str = str(burden)
            site_anatomic_label = str(entry.get("site") or entry.get("organ") or "")
            site_class_raw = entry.get("site_class")
            site_class = (
                str(site_class_raw)
                if isinstance(site_class_raw, str) and site_class_raw.strip()
                else _normalize_site_class(site_anatomic_label)
            )
            (
                site_class,
                site_class_canonical,
                site_class_discordance,
            ) = _canonical_bucket(site_class, _SITE_CLASS_CANONICAL_ALIASES)
            site_class_for_signature = site_class_canonical or _SITE_CLASS_OTHER
            site_signature = _site_signature(
                site_class_for_signature,
                site_anatomic_label,
            )
            is_new = site_signature not in seen_signatures
            snapshot_signatures.add(site_signature)

            row: dict[str, Any] = {col: None for col in _DISEASE_SITE_COLUMNS}
            row.update(cell_columns)
            (
                row["spec_baseline_hash"],
                row["spec_observed_prefix_hash"],
                row["spec_future_constraint_hash"],
                row["spec_conditioning_hash"],
            ) = spec_hashes
            row["snapshot_index"] = snapshot_index
            row["state_id"] = snapshot.get("state_id")
            row["snapshot_date"] = snap_date.isoformat() if snap_date else None
            row["days_from_diagnosis"] = days_from_dx
            row["site_id"] = site_id
            row["clone_id"] = entry.get("clone")
            row["site_anatomic_label"] = site_anatomic_label
            row["site_class"] = site_class
            row["site_class_canonical"] = site_class_canonical
            row["site_class_discordance"] = site_class_discordance
            row["burden_band"] = burden_str
            row["site_signature"] = site_signature
            row["is_baseline"] = snapshot_index == 0
            row["is_new_at_snapshot"] = is_new
            rows.append(row)
            site_id += 1
        # Merge this snapshot's signatures AFTER finishing the snapshot so
        # within-snapshot rows are evaluated against prior snapshots only.
        seen_signatures.update(snapshot_signatures)
    return rows


def build_disease_site_rows(
    artifacts: Sequence[TrajectoryArtifact],
) -> pd.DataFrame:
    """One row per active disease site per snapshot across all trajectories.

    ``is_new_at_snapshot`` is computed via a per-trajectory rolling set
    of ``site_signature`` values — true iff this signature has not been
    seen in any earlier snapshot of the same trajectory. Active = burden
    not in {None, 'absent'}, matching the snapshot_rows.csv semantic.
    """
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        rows.extend(_disease_site_rows_for_artifact(artifact))
    return pd.DataFrame(rows, columns=list(_DISEASE_SITE_COLUMNS))


# ---------- attempt_rows.csv ----------

# B1b: one row per generation attempt (accepted + failed). Powers
# Family 13 (Run validity and methods QC) — acceptance rate, attempts-
# to-accept distribution, failure-kind distribution, validation-warning
# rate per rule, cost/tokens per accepted trajectory.

_ATTEMPT_COLUMNS: tuple[str, ...] = (
    # Cell key (9 cols — `accepted` is NOT here because it's at the row
    # level reflecting per-attempt outcome, not cell-level).
    "trajectory_id", "condition_id", "policy_id", "model_id", "model_short",
    "future_index", "tumor_oncotree_code", "conditioning_label", "conditioning_hash",
    # Attempt identity.
    "attempt_index", "attempt_dir",
    "prompt_hash", "prompt_path", "raw_response_path", "result_path",
    # Outcome (per-attempt, not per-trajectory).
    "accepted", "failure_kind", "failure_message",
    "failure_rule_id", "failure_family",
    # Validation warnings: semicolon-joined rule_id list. Populated from
    # result.json.validation_warning_rule_ids on attempts that reached
    # replay (accepted + post-replay-failed); null on attempts that
    # didn't reach replay (transport, parse, baseline/ledger validation,
    # observed_prefix_audit, replay itself). Legacy attempts predating
    # the B1b runner write are also null.
    "validation_warnings_joined",
    # Cost / usage (required from result.json).
    "cost_usd", "tokens_input", "tokens_output", "reasoning_tokens",
    "response_model", "response_id", "provider_retries",
    "max_tokens_sent", "max_tokens_param", "reasoning_mode",
    "temperature_sent", "top_p_sent", "seed_sent", "finish_reason",
    # Timing (optional / nullable — best-effort; no locked metric reads).
    "latency_ms",
    # Four conditioning-spec hashes (sentinels for legacy attempts).
    "spec_baseline_hash", "spec_observed_prefix_hash",
    "spec_future_constraint_hash", "spec_conditioning_hash",
)

_REPLAY_RULE_PREFIX_RE = re.compile(r"^([A-Z][A-Z0-9_]+):\s+")

_REPLAY_RULE_FAMILY: dict[str, str] = {
    "CLONE_REGISTRY_UNIQUE": "clone_state",
    "CLONE_REGISTRY_NAMING": "clone_state",
    "CLONE_REFERENCE": "clone_state",
    "CLONAL_EVOLUTION_CLONE_ID": "clone_state",
    "CLONAL_EVOLUTION_PARENT_REFERENCE": "clone_state",
    "EVENT_CONTRACT": "event_contract",
    "RECOGNITION_TIMING": "chronology",
    "TERMINAL_CONTRACT": "terminality",
    "TERMINALITY": "terminality",
    "TOXICITY_GRADE_RANGE": "toxicity_contract",
    "TREATMENT_LIFECYCLE": "treatment_lifecycle",
}

_FAILURE_KIND_FAMILY: dict[str, str] = {
    "baseline_validation": "baseline_schema",
    "conditioning_audit": "conditioning",
    "dry_run": "dry_run",
    "future_constraints_audit": "conditioning",
    "generation_contract": "generation_contract",
    "intervention_audit": "conditioning",
    "ledger_validation": "ledger_schema",
    "observed_prefix_audit": "conditioning",
    "parse": "response_shape",
    "terminal_contract": "terminality",
    "transport_or_runtime": "provider_runtime",
}


def _attempt_validation_warnings_joined(result: Mapping[str, Any]) -> str | None:
    """Read ``result.json.validation_warning_rule_ids`` and join with ';'.

    Returns null when the field is null (legacy attempts predating B1b
    runner write, or attempts that didn't reach replay) AND when the
    field is an empty list (no warnings on a replayed attempt — still
    null rather than empty string so the column distinguishes "no
    warnings" via the empty-list semantic in result.json from "didn't
    reach replay" via null; consumers that need the distinction read
    the bool ``accepted`` + ``failure_kind`` columns).
    """
    rule_ids = result.get("validation_warning_rule_ids")
    if rule_ids is None:
        return None
    if not isinstance(rule_ids, Sequence) or isinstance(rule_ids, (str, bytes)):
        return None
    items = [str(r) for r in rule_ids if r is not None]
    return ";".join(items) if items else None


def _failure_rule_id(result: Mapping[str, Any]) -> str | None:
    """Derive a compact hard-failure rule ID when the runner exposes one.

    Replay failures are emitted as ``RULE_ID: human message`` by
    ``ReplayError``. Other failure kinds have heterogeneous messages
    (provider exceptions, Pydantic validation blocks, free-text audit
    details), so this stays intentionally narrow instead of inventing
    unstable IDs.
    """
    failure_kind = result.get("failure_kind")
    if not isinstance(failure_kind, str) or not failure_kind:
        return None
    if failure_kind == "terminal_contract":
        return "TERMINAL_CONTRACT"
    if failure_kind != "replay":
        return None

    failure_message = result.get("failure_message")
    if not isinstance(failure_message, str):
        return None
    match = _REPLAY_RULE_PREFIX_RE.match(failure_message)
    return match.group(1) if match else None


def _failure_family(result: Mapping[str, Any]) -> str | None:
    """Derive a stable analysis family for hard-failure stratification."""
    failure_kind = result.get("failure_kind")
    if not isinstance(failure_kind, str) or not failure_kind:
        return None
    if failure_kind == "replay":
        rule_id = _failure_rule_id(result)
        if rule_id is None:
            return "replay"
        return _REPLAY_RULE_FAMILY.get(rule_id, "replay")
    return _FAILURE_KIND_FAMILY.get(failure_kind, failure_kind)


def _attempt_latency_ms(result: Mapping[str, Any]) -> int | None:
    """Derive latency_ms from result.json.usage.duration_s when present."""
    usage = result.get("usage")
    if not isinstance(usage, Mapping):
        return None
    duration = usage.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        return None
    if not math.isfinite(duration):
        return None
    return int(duration * 1000)


def _attempt_tokens(result: Mapping[str, Any], key: str) -> int | None:
    usage = result.get("usage")
    if not isinstance(usage, Mapping):
        return None
    value = usage.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return int(value)


def _attempt_usage_value(result: Mapping[str, Any], key: str) -> Any:
    usage = result.get("usage")
    if not isinstance(usage, Mapping):
        return None
    value = usage.get(key)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _attempt_row(artifact: AttemptArtifact) -> dict[str, Any]:
    cell = artifact.cell
    result = artifact.result
    spec_hashes = _spec_hashes(artifact.conditioning_spec)
    # Store attempt_dir as a path relative to the run root
    # (parent of the trajectories/ dir) so the CSV is portable across
    # machines. Concretely: "trajectories/<traj_dir>/attempt_XX".
    run_root = artifact.trajectory_dir.parent.parent
    try:
        relative_attempt_dir = artifact.attempt_dir.relative_to(run_root)
    except ValueError as exc:
        raise DerivationDataLossError(
            f"attempt_dir {artifact.attempt_dir} is not under run root "
            f"{run_root}; cannot compute relative path"
        ) from exc
    assert not relative_attempt_dir.is_absolute(), (
        f"attempt_dir relative path is unexpectedly absolute: "
        f"{relative_attempt_dir}"
    )
    row: dict[str, Any] = {col: None for col in _ATTEMPT_COLUMNS}
    row["trajectory_id"] = cell.trajectory_id
    row["condition_id"] = cell.condition_id
    row["policy_id"] = cell.policy_id
    row["model_id"] = cell.model_id
    row["model_short"] = cell.model_short
    row["future_index"] = cell.future_index
    row["tumor_oncotree_code"] = cell.tumor_oncotree_code
    row["conditioning_label"] = cell.conditioning_label
    row["conditioning_hash"] = cell.conditioning_hash
    row["attempt_index"] = artifact.attempt_index
    row["attempt_dir"] = str(relative_attempt_dir)
    row["prompt_hash"] = artifact.prompt_hash
    row["prompt_path"] = (
        str(relative_attempt_dir / "prompt.txt")
        if (artifact.attempt_dir / "prompt.txt").exists()
        else None
    )
    row["raw_response_path"] = (
        str(relative_attempt_dir / "raw_response.txt")
        if (artifact.attempt_dir / "raw_response.txt").exists()
        else None
    )
    row["result_path"] = str(relative_attempt_dir / "result.json")
    row["accepted"] = bool(result.get("accepted"))
    row["failure_kind"] = result.get("failure_kind")
    row["failure_message"] = result.get("failure_message")
    row["failure_rule_id"] = _failure_rule_id(result)
    row["failure_family"] = _failure_family(result)
    row["validation_warnings_joined"] = _attempt_validation_warnings_joined(result)
    row["cost_usd"] = result.get("cost_usd")
    row["tokens_input"] = _attempt_tokens(result, "input_tokens")
    row["tokens_output"] = _attempt_tokens(result, "output_tokens")
    row["reasoning_tokens"] = _attempt_tokens(result, "reasoning_tokens")
    row["response_model"] = _attempt_usage_value(result, "response_model")
    row["response_id"] = _attempt_usage_value(result, "response_id")
    row["provider_retries"] = _attempt_tokens(result, "retries")
    row["max_tokens_sent"] = _attempt_tokens(result, "max_tokens_sent")
    row["max_tokens_param"] = _attempt_usage_value(result, "max_tokens_param")
    row["reasoning_mode"] = _attempt_usage_value(result, "reasoning_mode")
    row["temperature_sent"] = _attempt_usage_value(result, "temperature_sent")
    row["top_p_sent"] = _attempt_usage_value(result, "top_p_sent")
    row["seed_sent"] = _attempt_usage_value(result, "seed_sent")
    row["finish_reason"] = _attempt_usage_value(result, "finish_reason")
    row["latency_ms"] = _attempt_latency_ms(result)
    (
        row["spec_baseline_hash"],
        row["spec_observed_prefix_hash"],
        row["spec_future_constraint_hash"],
        row["spec_conditioning_hash"],
    ) = spec_hashes
    return row


def build_attempt_rows(attempts: Sequence[AttemptArtifact]) -> pd.DataFrame:
    """One row per attempt (accepted + failed) across all trajectories."""
    rows = [_attempt_row(a) for a in attempts]
    return pd.DataFrame(rows, columns=list(_ATTEMPT_COLUMNS))


# ---------- field_utilization.json ----------


def _resolve_nested(obj: Mapping[str, Any] | None, path: str) -> Any:
    """Walk dotted path on a nested mapping. Returns None on any missing link."""
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


def _is_populated(value: Any) -> bool:
    """A field is populated iff value is non-None and (for containers) non-empty."""
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, str)):
        return len(value) > 0
    return True


def _group_key_for_artifact(artifact: TrajectoryArtifact) -> tuple[str, str, str, str]:
    cell = artifact.cell
    return (cell.model_id, cell.tumor_oncotree_code, cell.policy_id, cell.conditioning_label)


def build_field_utilization(
    artifacts: Sequence[TrajectoryArtifact],
) -> dict[str, Any]:
    """Per (model, tumor, policy, conditioning_label) field-populated rate.

    Baseline fields: rate = (trajectories with baseline.<path> populated) /
                            (trajectories in group)

    Event fields: rate = (events of that type with field populated) /
                         (events of that type in group)
                  null when zero events of that type appear in the group.

    Uses parsed_ledger.json (sparse LLM emission) for event-field rates, NOT
    the materialized ledger. The point of this artifact is to measure what
    the model emitted; the materialized ledger fills in pipeline-resolved
    fields that the model never wrote.
    """
    # group → (n_trajectories, baseline_pop_counts_by_path,
    #          event_count_by_type, event_pop_counts_by_(type,field))
    groups: dict[
        tuple[str, str, str, str],
        dict[str, Any],
    ] = {}

    for artifact in artifacts:
        key = _group_key_for_artifact(artifact)
        bucket = groups.setdefault(
            key,
            {
                "n_trajectories": 0,
                "baseline_pop": dict.fromkeys(_BASELINE_TRACKED_FIELDS, 0),
                "event_count_by_type": {},
                "event_pop": {ef: 0 for ef in _EVENT_TRACKED_FIELDS},
            },
        )
        bucket["n_trajectories"] += 1
        baseline = (
            artifact.materialized.get("baseline") if artifact.materialized else None
        )
        if isinstance(baseline, Mapping):
            for path in _BASELINE_TRACKED_FIELDS:
                value = _resolve_nested(baseline, path)
                if _is_populated(value):
                    bucket["baseline_pop"][path] += 1
        for event in artifact.parsed_ledger:
            evt_type = event.get("type")
            if not isinstance(evt_type, str):
                continue
            bucket["event_count_by_type"][evt_type] = (
                bucket["event_count_by_type"].get(evt_type, 0) + 1
            )
            for evt_type_field, field_name in _EVENT_TRACKED_FIELDS:
                if evt_type != evt_type_field:
                    continue
                if _is_populated(event.get(field_name)):
                    bucket["event_pop"][(evt_type_field, field_name)] += 1

    output: dict[str, Any] = {}
    for (model_id, tumor_oncotree_code, policy_id, conditioning_label), bucket in sorted(groups.items()):
        n = bucket["n_trajectories"]
        block: dict[str, float | None] = {}
        for path in _BASELINE_TRACKED_FIELDS:
            block[f"baseline.{path}"] = (
                _round_float(bucket["baseline_pop"][path] / n, 4) if n else None
            )
        for (evt_type, field) in _EVENT_TRACKED_FIELDS:
            evt_n = bucket["event_count_by_type"].get(evt_type, 0)
            block[f"{evt_type}.{field}"] = (
                _round_float(bucket["event_pop"][(evt_type, field)] / evt_n, 4)
                if evt_n else None
            )
        # Nest: model_id → tumor_oncotree_code → policy_id → conditioning_label → {field: rate}
        (
            output
            .setdefault(model_id, {})
            .setdefault(tumor_oncotree_code, {})
            .setdefault(policy_id, {})
        )[conditioning_label] = block

    return output


def _pydantic_dotted_paths(model_cls: type) -> set[str]:
    """Return every valid dotted attribute path reachable from a Pydantic model.

    Recurses through nested BaseModel fields (including those wrapped in
    Optional / Union annotations) so that paths like
    ``identity_card.baseline_health.frailty_band`` are validated end-to-end,
    not just at the top level.
    """
    import typing
    from pydantic import BaseModel

    paths: set[str] = set()
    if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
        return paths

    for name, field in model_cls.model_fields.items():
        paths.add(name)
        annotation = field.annotation
        candidates: list[Any] = [annotation]
        # Unwrap Optional[X] / Union[X, None] into the inner candidates.
        for arg in typing.get_args(annotation):
            candidates.append(arg)
        for candidate in candidates:
            if (
                isinstance(candidate, type)
                and candidate is not model_cls
                and issubclass(candidate, BaseModel)
            ):
                for sub in _pydantic_dotted_paths(candidate):
                    paths.add(f"{name}.{sub}")
                break
    return paths


def _verify_tracked_fields_resolve() -> None:
    """Grounding check: every tracked dotted path must exist on the schema.

    Called from build_derivation_summary so a future schema rename fails the
    derivation build fast, with a specific diagnostic, rather than silently
    producing utilization=0.0 across the board. Walks the full nested path
    (not just the top-level field) so that renames anywhere under
    ``medical_history.*`` or ``identity_card.baseline_health.*`` are caught.
    """
    baseline_paths = _pydantic_dotted_paths(BaselineObject)
    event_paths = _pydantic_dotted_paths(LedgerEvent)

    for path in _BASELINE_TRACKED_FIELDS:
        if path not in baseline_paths:
            raise RuntimeError(
                f"_BASELINE_TRACKED_FIELDS entry {path!r} does not resolve "
                f"against the current BaselineObject schema."
            )
    for evt_type, field in _EVENT_TRACKED_FIELDS:
        if field not in event_paths:
            raise RuntimeError(
                f"_EVENT_TRACKED_FIELDS entry ({evt_type!r}, {field!r}) does "
                f"not resolve: {field!r} not in LedgerEvent schema."
            )


# ---------- run_summary ----------


def build_derivation_summary(
    artifacts: Sequence[TrajectoryArtifact],
    event_rows: pd.DataFrame,
    snapshot_rows: pd.DataFrame,
    treatment_rows: pd.DataFrame,
    endpoint_rows: pd.DataFrame,
    field_utilization: Mapping[str, Any],
    run_root: Path,
) -> tuple[dict[str, Any], str]:
    """Build the derivation-level summary JSON + markdown.

    This summary rolls up the row-level artifacts so a reader can scan the
    derivation outputs without joining the CSVs by hand.
    """
    _verify_tracked_fields_resolve()

    n_trajectories = len(artifacts)
    n_events = len(event_rows)
    n_snapshots = len(snapshot_rows)
    n_treatment_courses = len(treatment_rows)
    n_endpoints = len(endpoint_rows)

    by_cell_medians: list[dict[str, Any]] = []
    if not endpoint_rows.empty:
        group_cols = ["tumor_oncotree_code", "model_short", "policy_id", "conditioning_label"]
        grouped = endpoint_rows.groupby(group_cols, dropna=False)
        for keys, group in grouped:
            os_values = [v for v in group["os_days"].tolist() if isinstance(v, (int, float))]
            pfs1_values = [
                v for v in group["pfs_line_1_days"].tolist()
                if isinstance(v, (int, float))
            ]
            by_cell_medians.append(
                {
                    "tumor_oncotree_code": keys[0],
                    "model_short": keys[1],
                    "policy_id": keys[2],
                    "conditioning_label": keys[3],
                    "n": int(len(group)),
                    "os_days_median": (
                        _round_float(_percentile(os_values, 0.5), 1) if os_values else None
                    ),
                    "os_days_mean": _round_float(_mean(os_values), 1) if os_values else None,
                    "pfs_line_1_days_median": (
                        _round_float(_percentile(pfs1_values, 0.5), 1) if pfs1_values else None
                    ),
                }
            )

    payload = {
        "derivation_version": DERIVATION_VERSION,
        "run_root": str(run_root.name),
        "n_trajectories": n_trajectories,
        "n_accepted": sum(1 for a in artifacts if a.cell.accepted),
        "n_rejected": sum(1 for a in artifacts if not a.cell.accepted),
        "n_events": n_events,
        "n_snapshots": n_snapshots,
        "n_treatment_courses": n_treatment_courses,
        "n_endpoints": n_endpoints,
        "endpoints_by_cell_medians": by_cell_medians,
        "field_utilization_summary": _field_utilization_top_rows(field_utilization),
    }
    md = _format_derivation_markdown(payload, field_utilization)
    return payload, md


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _field_utilization_top_rows(
    field_utilization: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Flatten the nested field_utilization dict for a compact summary table."""
    rows: list[dict[str, Any]] = []
    for model_id, by_tumor in field_utilization.items():
        for tumor_oncotree_code, by_policy in by_tumor.items():
            for policy_id, by_label in by_policy.items():
                for label, fields in by_label.items():
                    rows.append(
                        {
                            "model_id": model_id,
                            "tumor_oncotree_code": tumor_oncotree_code,
                            "policy_id": policy_id,
                            "conditioning_label": label,
                            "fields": dict(fields),
                        }
                    )
    return rows


def _format_derivation_markdown(
    payload: Mapping[str, Any],
    field_utilization: Mapping[str, Any],
) -> str:
    lines: list[str] = []
    lines.append(f"# SynthOncCases derivation — {payload['run_root']}")
    lines.append("")
    lines.append(f"- Derivation version: `{payload['derivation_version']}`")
    lines.append(
        f"- Trajectories: {payload['n_trajectories']} "
        f"(accepted={payload['n_accepted']}, rejected={payload['n_rejected']})"
    )
    lines.append(
        f"- Events: {payload['n_events']} | "
        f"Snapshots: {payload['n_snapshots']} | "
        f"Treatment courses: {payload['n_treatment_courses']} | "
        f"Endpoints: {payload['n_endpoints']}"
    )
    lines.append("")
    lines.append("## Endpoints by cell")
    lines.append("")
    if payload["endpoints_by_cell_medians"]:
        lines.append(
            "| tumor | model | policy | label | n | OS median | OS mean | PFS L1 median |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in payload["endpoints_by_cell_medians"]:
            lines.append(
                f"| {row['tumor_oncotree_code']} | {row['model_short']} | {row['policy_id']} | "
                f"{row['conditioning_label']} | {row['n']} | {row['os_days_median']} | "
                f"{row['os_days_mean']} | {row['pfs_line_1_days_median']} |"
            )
        lines.append("")
    else:
        lines.append("(no endpoint rows)")
        lines.append("")

    lines.append("## Field utilization")
    lines.append("")
    if not field_utilization:
        lines.append("(no trajectories — field utilization empty)")
        lines.append("")
    else:
        for model_id, by_tumor in field_utilization.items():
            for tumor_oncotree_code, by_policy in by_tumor.items():
                for policy_id, by_label in by_policy.items():
                    for label, fields in by_label.items():
                        lines.append(
                            f"### `{model_id}` · {tumor_oncotree_code} · {policy_id} · "
                            f"label=`{label}`"
                        )
                        lines.append("")
                        lines.append("| field | utilization |")
                        lines.append("|---|---|")
                        for field, rate in fields.items():
                            display = "n/a" if rate is None else f"{rate:.2f}"
                            lines.append(f"| `{field}` | {display} |")
                        lines.append("")
    return "\n".join(lines) + "\n"


# ---------- top-level ----------


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def derive_run(
    run_root: Path,
    out_dir: Path | None = None,
    *,
    force: bool = False,
    include_rejected: bool = False,
) -> DerivationReport:
    """Top-level derivation orchestration.

    Discovers trajectories under ``<run_root>/trajectories/``, builds the six
    artifacts, and writes them under ``out_dir`` (default
    ``<run_root>/derivation/``). Refuses to overwrite an existing output
    directory unless ``force=True``.
    """
    target_dir = out_dir if out_dir is not None else run_root / "derivation"
    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        raise FileExistsError(
            f"derivation output directory {target_dir} is non-empty; "
            "pass force=True to overwrite"
        )
    target_dir.mkdir(parents=True, exist_ok=True)

    artifacts = discover_trajectories(run_root, include_rejected=include_rejected)

    event_rows = build_event_rows(artifacts)
    snapshot_rows = build_snapshot_rows(artifacts)
    treatment_rows = build_treatment_rows(artifacts)
    endpoint_rows = build_endpoint_rows(artifacts)
    baseline_rows = build_baseline_rows(artifacts)
    baseline_biomarker_rows = build_baseline_biomarker_rows(artifacts)
    baseline_alteration_rows = build_baseline_alteration_rows(artifacts)
    treatment_component_rows = build_treatment_component_rows(artifacts)
    disease_site_rows = build_disease_site_rows(artifacts)
    # B1b: attempt_rows walks EVERY attempt dir (accepted + failed), not
    # just the accepted-trajectory subset, so it runs against the
    # full attempt corpus rather than the TrajectoryArtifact list.
    attempts = discover_attempts(run_root)
    attempt_rows = build_attempt_rows(attempts)
    field_utilization = build_field_utilization(artifacts)

    summary_payload, summary_md = build_derivation_summary(
        artifacts,
        event_rows,
        snapshot_rows,
        treatment_rows,
        endpoint_rows,
        field_utilization,
        run_root,
    )

    event_rows_path = target_dir / "event_rows.csv"
    snapshot_rows_path = target_dir / "snapshot_rows.csv"
    treatment_rows_path = target_dir / "treatment_rows.csv"
    endpoint_rows_path = target_dir / "endpoint_rows.csv"
    baseline_rows_path = target_dir / "baseline_rows.csv"
    baseline_biomarker_rows_path = target_dir / "baseline_biomarker_rows.csv"
    baseline_alteration_rows_path = target_dir / "baseline_alteration_rows.csv"
    treatment_component_rows_path = target_dir / "treatment_component_rows.csv"
    disease_site_rows_path = target_dir / "disease_site_rows.csv"
    attempt_rows_path = target_dir / "attempt_rows.csv"
    field_utilization_path = target_dir / "field_utilization.json"
    run_summary_json_path = target_dir / "run_summary.json"
    run_summary_md_path = target_dir / "run_summary.md"

    _write_csv(event_rows, event_rows_path)
    _write_csv(snapshot_rows, snapshot_rows_path)
    _write_csv(treatment_rows, treatment_rows_path)
    _write_csv(endpoint_rows, endpoint_rows_path)
    _write_csv(baseline_rows, baseline_rows_path)
    _write_csv(baseline_biomarker_rows, baseline_biomarker_rows_path)
    _write_csv(baseline_alteration_rows, baseline_alteration_rows_path)
    _write_csv(treatment_component_rows, treatment_component_rows_path)
    _write_csv(disease_site_rows, disease_site_rows_path)
    _write_csv(attempt_rows, attempt_rows_path)
    _write_json(field_utilization_path, field_utilization)
    _write_json(run_summary_json_path, summary_payload)
    _write_text(run_summary_md_path, summary_md)

    return DerivationReport(
        run_root=run_root,
        out_dir=target_dir,
        event_rows_path=event_rows_path,
        snapshot_rows_path=snapshot_rows_path,
        treatment_rows_path=treatment_rows_path,
        endpoint_rows_path=endpoint_rows_path,
        field_utilization_path=field_utilization_path,
        run_summary_json_path=run_summary_json_path,
        run_summary_md_path=run_summary_md_path,
        n_trajectories=len(artifacts),
        n_events=len(event_rows),
        n_snapshots=len(snapshot_rows),
        n_treatment_courses=len(treatment_rows),
        n_endpoints=len(endpoint_rows),
        baseline_rows_path=baseline_rows_path,
        n_baselines=len(baseline_rows),
        baseline_biomarker_rows_path=baseline_biomarker_rows_path,
        n_baseline_biomarkers=len(baseline_biomarker_rows),
        baseline_alteration_rows_path=baseline_alteration_rows_path,
        n_baseline_alterations=len(baseline_alteration_rows),
        treatment_component_rows_path=treatment_component_rows_path,
        n_treatment_components=len(treatment_component_rows),
        disease_site_rows_path=disease_site_rows_path,
        n_disease_sites=len(disease_site_rows),
        attempt_rows_path=attempt_rows_path,
        n_attempts=len(attempt_rows),
    )


# Re-exports for tests / downstream importers.
__all__: tuple[str, ...] = (
    "DERIVATION_VERSION",
    "DEFAULT_CONDITIONING_LABEL",
    "DEFAULT_TUMOR_ONCOTREE_CODE",
    "CellKey",
    "TrajectoryArtifact",
    "DerivationReport",
    "discover_trajectories",
    "build_event_rows",
    "build_snapshot_rows",
    "build_treatment_rows",
    "build_treatment_component_rows",
    "build_endpoint_rows",
    "build_baseline_rows",
    "build_baseline_biomarker_rows",
    "build_baseline_alteration_rows",
    "build_disease_site_rows",
    "build_attempt_rows",
    "discover_attempts",
    "AttemptArtifact",
    "build_field_utilization",
    "build_derivation_summary",
    "derive_run",
    "TreatmentCourseRow",
    "AcquiredAlterationRow",
)
