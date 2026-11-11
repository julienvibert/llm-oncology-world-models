"""SynthOncCases generation runner.

Orchestrates lean-ledger trajectory generation across conditioning cells,
models, and futures. It loads a YAML config, renders the active prompt with the
injected schemas and conditioning payload, dispatches one fresh provider call
per trajectory attempt, validates the response through the acceptance gates,
and writes reproducible on-disk artifacts.

Each cell carries a deterministic ``conditioning_label`` and
``conditioning_hash`` derived from baseline pins, optional fixed-prefix events,
and future constraints. The declared response contract selects either
``{"baseline_patch": {...}, "ledger": [...]}`` or continuation-only
``{"ledger": [...]}``; the runner enforces the same shape the prompt requested
before deterministic state reconstruction.

The provider call is injectable via ``generate_fn`` so unit tests stay
offline. Treatment-course derivation and paper figure construction are separate
downstream steps.
"""

from __future__ import annotations

import hashlib
import atexit
import csv
import json
import multiprocessing as mp
import os
import re
import subprocess
from copy import deepcopy
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from synthonccases.generate.providers import (
    DEFAULT_REQUEST_SEED,
    DEFAULT_REQUEST_TEMPERATURE,
    DEFAULT_REQUEST_TOP_P,
    PROVIDER_INTERNAL_MAX_ATTEMPTS,
    GenerationResult,
    get_provider,
)
from synthonccases.oncollm.parse_utils import strip_code_fence
from synthonccases.oncollm.conditioning_audit import (
    BaselineResolutionError,
    ConditioningSpec,
    assert_no_pin_collision,
    assert_no_prefix_collision,
    canonical_json,
    compute_conditioning_hash,
    compute_conditioning_spec_hash,
    compute_pinned_paths,
    compute_required_patch_leaves,
    compute_seed_hash,
    derive_conditioning_label,
    future_constraints_audit,
    resolve_and_validate,
    spec_to_json_dict,
    validate_baseline_seed,
    validate_observed_prefix,
)
from synthonccases.oncollm.condition_lower import lower_conditioning_cell
from synthonccases.oncollm.generation_contract import check_generation_contract
from synthonccases.oncollm.replay import (
    BaselineObject,
    LedgerEvent,
    MaterializedTrajectory,
    ReplayError,
    canonicalize_legacy_event_dict,
    replay,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

_SYSTEM_PROMPT = (
    "You are generating a compact JSON event ledger for deterministic replay. "
    "Output valid JSON only. Do not include markdown or explanatory prose."
)

NO_ADDITIONAL_POLICY_TEXT = "No additional run-specific instructions."
_NO_ADDITIONAL_POLICY_TEXTS = frozenset(
    {
        NO_ADDITIONAL_POLICY_TEXT,
        "No additional operator constraints.",
    }
)

_PROVIDER_CALL_TIMEOUT_ENV = "ONCOLLM_PROVIDER_CALL_TIMEOUT_SECONDS"
_DEFAULT_PROVIDER_CALL_TIMEOUT_SECONDS = 600.0

FAILURE_KINDS = frozenset(
    {
        "parse",
        "conditioning_audit",
        "generation_contract",
        "baseline_validation",
        "ledger_validation",
        "intervention_audit",
        # Gate A3.5b: generalized conditioning-spec audits. Fire only when a
        # ``ConditioningSpec`` is threaded into ``classify_response``; the
        # legacy ``intervention_audit`` keeps firing on the legacy path
        # (``conditioning_spec is None``).
        "observed_prefix_audit",
        "future_constraints_audit",
        "replay",
        "terminal_contract",
        "transport_or_runtime",
    }
)

# Study-level contract on what counts as a valid terminal event. Lives at
# run/config level, NOT policy level — this is an experimental-design decision,
# not a treatment choice. Default keeps both terminal types valid (matches the
# replay engine's underlying contract). Production runs can flip this to
# "death_only" so censoring becomes an opt-in for continuation/longitudinal
# studies, not a default trajectory exit.
TerminalContract = Literal["death_or_last_followup", "death_only"]

#: WHICH response shape the prompt asked for, and therefore which one the parser enforces.
#: `baseline_patch_and_ledger` is the frozen default every atlas run used. `ledger_only` is the
#: continuation contract, whose prompt states there is no baseline_patch and that emitting one is
#: rejected -- so under it an ABSENT patch is the correct answer.
#: Declared in the config rather than sniffed from the template, and then CHECKED against the
#: template at load time: a silent disagreement between what the model was told and what the
#: parser demands rejects every correct answer, which is how one smoke batch lost 143 attempts.
ResponseContract = Literal["baseline_patch_and_ledger", "ledger_only"]
DEFAULT_RESPONSE_CONTRACT: ResponseContract = "baseline_patch_and_ledger"
#: The sentence the continuation template uses to forbid a patch. Present in the template iff
#: the run is `ledger_only`; the load-time check compares the two and refuses a mismatch.
_LEDGER_ONLY_MARKER = "There is no `baseline_patch` in this contract"
DEFAULT_TERMINAL_CONTRACT: TerminalContract = "death_or_last_followup"
_STUDY_REFERENCE_DIAGNOSIS_DATE_PATH = "identity_card.diagnosis.diagnosis_date"
DEFAULT_STUDY_KNOWLEDGE_CUTOFF = "2024-12-31"
EnumOrderMode = Literal["schema_order", "deterministic_shuffle"]

# ---------- config dataclasses ----------


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    short: str


@dataclass(frozen=True, slots=True)
class RegimenLine:
    line: int
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FixedFirstLineTreatment:
    """Structured first-line pin for treatment-CF runs.

    The ``regimen_label`` is the canonical short name (used in the
    ``+1L=<label>`` suffix on the human-readable conditioning_label). The
    full canonical payload — including ``drugs`` and optional ``schedule``
    — flows into ``conditioning_hash`` so two policies sharing only the
    same ``regimen_label`` still hash differently when their drug list
    or schedule differs.
    """

    regimen_label: str
    drugs: tuple[str, ...]
    schedule: str | None = None


@dataclass(frozen=True, slots=True)
class StudyReference:
    """Study-level anchors applied to every generation cell.

    ``diagnosis_date`` is a patient-data anchor: it is merged into the
    baseline seed before validation and hashing so normal pinning/audit
    machinery enforces it.

    ``knowledge_cutoff`` is protocol metadata for the prompt instrument. It is
    deliberately not merged into patient state; it defines the clinical
    knowledge horizon the model is instructed to respect.
    """

    diagnosis_date: str | None = None
    knowledge_cutoff: str = DEFAULT_STUDY_KNOWLEDGE_CUTOFF


@dataclass(frozen=True, slots=True)
class PromptEnumOrder:
    """Prompt-visible schema enum ordering policy.

    This is an experimental-instrument control only. It changes the order of
    enum arrays in the rendered JSON schemas the LLM sees, while the real
    Pydantic schemas and acceptance gates stay unchanged.
    """

    mode: EnumOrderMode = "schema_order"
    seed: str = ""


@dataclass(frozen=True, slots=True)
class PolicySpec:
    id: str
    text: str
    regimen_sequence: tuple[RegimenLine, ...]
    unconstrained_after_line: int | None
    fixed_first_line_treatment: FixedFirstLineTreatment | None = None


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    """One conditioning regime: a (partial-or-empty-or-full) baseline seed.

    Resolved at config-load time: ``baseline_seed`` is the JSON-parsed
    seed dict (from ``seed_path`` or ``inline_seed`` or ``{}`` for
    spontaneous), ``pinned_paths`` is the sorted dotted leaf set, and
    ``seed_hash`` is the deterministic identifier of that seed shape.
    The full ``conditioning_hash`` is per-cell (combines this with the
    policy payload) and lives on the ``CellKey``.

    ``label_override`` is optional cosmetic per-condition; a global
    ``label_prefix`` on ``POCConfig`` prepends to every derived label.
    Neither affects ``conditioning_hash``.
    """

    id: str
    baseline_seed: Mapping[str, Any]
    pinned_paths: tuple[str, ...]
    seed_hash: str
    # True when ``BaselineObject.model_validate(baseline_seed)`` succeeds —
    # i.e., the seed already specifies a complete valid baseline. In that
    # regime, ``classify_response`` requires ``baseline_patch == {}`` (any
    # non-empty patch — even on optional unpinned leaves — is rejected with
    # failure_kind="conditioning_audit").
    is_full_baseline: bool = False
    label_override: str | None = None


@dataclass(frozen=True, slots=True)
class ConditioningCellSpec:
    """One user-facing conditioning cell.

    Conceptually this is the flexible public contract:
    ``baseline_seed`` fixes any subset of baseline state and the optional
    ``policy.fixed_first_line_treatment`` fixes the first treatment event.
    Internally we keep the existing ConditionSpec / PolicySpec split because
    baseline pins and event pins are audited by different mechanisms.

    Gate A3.5b adds ``conditioning_spec``: the generalized three-axis
    ``ConditioningSpec`` for this cell, built at cell-resolution time from
    the SAME resolved baseline_seed that flows into ``condition.baseline_seed``.
    The invariant ``conditioning_spec.baseline_seed is condition.baseline_seed``
    prevents prompt rendering and audit logic from operating on diverging
    copies of the seed after study-reference application. ``None`` only for
    legacy code paths that haven't migrated to ``_build_cell``.
    """

    id: str
    condition: ConditionSpec
    policy: PolicySpec
    conditioning_spec: ConditioningSpec | None = None


@dataclass(frozen=True, slots=True)
class POCConfig:
    schema_version: int
    config_id: str
    conditions_count: int
    futures_per_cell: int
    max_attempts: int
    prompt_template_path: Path
    use_openrouter: bool
    response_format: Mapping[str, Any]
    models: tuple[ModelSpec, ...]
    conditions: tuple[ConditionSpec, ...]
    policies: tuple[PolicySpec, ...]
    require_parameters: bool = False
    prompt_enum_order: PromptEnumOrder = PromptEnumOrder()
    terminal_contract: TerminalContract = DEFAULT_TERMINAL_CONTRACT
    response_contract: ResponseContract = DEFAULT_RESPONSE_CONTRACT
    study_reference: StudyReference = StudyReference()
    # Cosmetic prefix prepended to every cell's derived conditioning_label
    # (e.g. "pilot-1:"). Does NOT affect conditioning_hash.
    label_prefix: str = ""
    # New primary config contract. When populated, matrix expansion uses these
    # explicit cells instead of the legacy conditions x policies cross-product.
    cells: tuple[ConditioningCellSpec, ...] = ()
    # Optional acceptance-target mode for definitive tranches. When set,
    # ``futures_per_cell`` is the target accepted count and matrix expansion
    # uses ``max_total_futures_per_cell`` as the per-cell/model attempt cap.
    accepted_futures_per_cell: int | None = None
    max_total_futures_per_cell: int | None = None

    def selected_conditions(self) -> tuple[ConditionSpec, ...]:
        if not self.cells:
            return self.conditions[: self.conditions_count]
        selected: list[ConditionSpec] = []
        seen: set[str] = set()
        for cell in self.selected_cells():
            if cell.condition.id in seen:
                continue
            selected.append(cell.condition)
            seen.add(cell.condition.id)
        return tuple(selected)

    def selected_cells(self) -> tuple[ConditioningCellSpec, ...]:
        if self.cells:
            return self.cells[: self.conditions_count]
        return tuple(
            ConditioningCellSpec(
                id=f"{condition.id}__{policy.id}",
                condition=condition,
                policy=policy,
            )
            for condition in self.selected_conditions()
            for policy in self.policies
        )


# ---------- matrix keys ----------


@dataclass(frozen=True, slots=True)
class CellKey:
    cell_id: str
    condition_id: str
    policy_id: str
    model_id: str
    model_short: str
    conditioning_label: str
    conditioning_hash: str

    def dir_prefix(self) -> str:
        return f"{self.cell_id}__{self.model_short}"


@dataclass(frozen=True, slots=True)
class TrajectoryKey:
    cell: CellKey
    future_index: int  # 1-indexed

    @property
    def dir_name(self) -> str:
        return f"{self.cell.dir_prefix()}__f{self.future_index:02d}"


# ---------- result dataclasses ----------


@dataclass(frozen=True, slots=True)
class AttemptResult:
    attempt_index: int  # 1-indexed
    accepted: bool
    failure_kind: str | None
    failure_message: str | None
    event_count: int | None
    snapshot_count: int | None
    validation_log_count: int | None
    # Per-rule validation-warning IDs, populated from
    # ``materialized.validation_log[].rule_id`` on every attempt that
    # reached replay (accepted + post-replay failures:
    # ``future_constraints_audit`` and ``terminal_contract``). Empty tuple
    # when replay succeeded with zero warnings. ``None`` on attempts that
    # didn't reach replay (transport, parse, baseline/ledger validation,
    # observed_prefix_audit, replay itself). B1b consumes this via
    # ``attempt_rows.validation_warnings_joined``.
    validation_warning_rule_ids: tuple[str, ...] | None
    cost_usd: float | None
    usage: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TrajectoryResult:
    key: TrajectoryKey
    accepted: bool
    attempts_to_accept: int | None
    failure_kind: str | None
    attempts: tuple[AttemptResult, ...]


@dataclass(frozen=True, slots=True)
class POCRunReport:
    run_id: str
    run_root: Path
    cells_planned: int
    trajectories_planned: int
    trajectories_accepted: int
    trajectories_failed: int
    trajectories_reused: int
    started_at: str
    completed_at: str
    dry_run: bool
    resumed: bool = False
    max_parallel_trajectories: int = 1
    max_parallel_trajectories_per_model: int | None = None


@dataclass(frozen=True, slots=True)
class _RunDirectoryLock:
    path: Path
    pid: int

    def release(self) -> None:
        try:
            payload = json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return
        if payload.get("pid") != self.pid:
            return
        self.path.unlink(missing_ok=True)


# ---------- config loading ----------


def _resolve_response_contract(prompt_block: Mapping[str, Any]) -> ResponseContract:
    """Read the declared response contract and REFUSE if the template disagrees with it.

    The declaration and the template have to say the same thing. A config that demands a
    baseline_patch while the prompt forbids one rejects every correct answer, and the failure
    looks exactly like model incompetence: a whole smoke batch came back with 143 attempts
    rejected for "response must contain a 'baseline_patch' object" while every one of those
    responses was a valid future-only ledger. Cheaper to fail at load time than after spending.
    """
    declared = str(prompt_block.get("response_contract", DEFAULT_RESPONSE_CONTRACT))
    if declared not in ("baseline_patch_and_ledger", "ledger_only"):
        raise ValueError(
            f"prompt.response_contract must be 'baseline_patch_and_ledger' or 'ledger_only'; "
            f"got {declared!r}")
    template_path = _resolve_repo_path(Path(prompt_block["template_path"]))
    forbids_patch = _LEDGER_ONLY_MARKER in template_path.read_text(encoding="utf-8")
    if forbids_patch and declared != "ledger_only":
        raise ValueError(
            f"{template_path.name} forbids a baseline_patch (continuation contract) but the "
            f"config declares response_contract={declared!r}. Every valid response would be "
            "rejected as a parse failure. Set prompt.response_contract: ledger_only.")
    if declared == "ledger_only" and not forbids_patch:
        raise ValueError(
            f"config declares response_contract='ledger_only' but {template_path.name} does not "
            "forbid a baseline_patch, so the model will emit one and it will be rejected.")
    return declared  # type: ignore[return-value]


def load_config(path: Path) -> POCConfig:
    """Load and validate a POC YAML config."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"config root must be a mapping; got {type(raw).__name__}")
    return _build_config(raw)


def _normalize_date_string(value: Any, *, context: str) -> str:
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise ValueError(f"{context}: expected ISO date string; got {type(value).__name__}")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{context}: expected ISO date string; got {value!r}") from exc
    return value


def _build_study_reference(raw: Mapping[str, Any]) -> StudyReference:
    block = raw.get("study_reference", {})
    if block is None:
        return StudyReference()
    if not isinstance(block, Mapping):
        raise ValueError(
            f"study_reference must be a mapping; got {type(block).__name__}"
        )
    allowed = {"diagnosis_date", "knowledge_cutoff"}
    unknown = sorted(str(k) for k in block if k not in allowed)
    if unknown:
        raise ValueError(f"study_reference has unknown key(s): {', '.join(unknown)}")
    diagnosis_date = block.get("diagnosis_date")
    knowledge_cutoff = block.get(
        "knowledge_cutoff", DEFAULT_STUDY_KNOWLEDGE_CUTOFF
    )
    return StudyReference(
        diagnosis_date=(
            None
            if diagnosis_date is None
            else _normalize_date_string(
                diagnosis_date, context="study_reference.diagnosis_date"
            )
        ),
        knowledge_cutoff=_normalize_date_string(
            knowledge_cutoff, context="study_reference.knowledge_cutoff"
        ),
    )


def _get_nested(mapping: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _set_nested(mapping: dict[str, Any], path: str, value: Any) -> None:
    current = mapping
    parts = path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise ValueError(
                f"cannot apply study_reference at {path!r}: "
                f"{part!r} already contains a non-mapping value"
            )
        current = child
    current[parts[-1]] = value


def _apply_study_reference_to_seed(
    seed: Mapping[str, Any],
    *,
    study_reference: StudyReference,
    context: str,
) -> dict[str, Any]:
    """Return seed with study-level data anchors merged in.

    Conflicting explicit seed values fail fast. Matching explicit values are
    allowed and canonicalized to the study-reference representation.
    """
    out = deepcopy(dict(seed))
    if study_reference.diagnosis_date is None:
        return out

    exists, existing = _get_nested(out, _STUDY_REFERENCE_DIAGNOSIS_DATE_PATH)
    if exists:
        existing_date = _normalize_date_string(
            existing, context=f"{context}.{_STUDY_REFERENCE_DIAGNOSIS_DATE_PATH}"
        )
        if existing_date != study_reference.diagnosis_date:
            raise ValueError(
                f"{context}: study_reference.diagnosis_date="
                f"{study_reference.diagnosis_date!r} conflicts with "
                f"{_STUDY_REFERENCE_DIAGNOSIS_DATE_PATH}={existing_date!r}"
            )
    _set_nested(
        out,
        _STUDY_REFERENCE_DIAGNOSIS_DATE_PATH,
        study_reference.diagnosis_date,
    )
    return out


def _study_reference_label_ignored_paths(
    study_reference: StudyReference,
) -> tuple[str, ...]:
    if study_reference.diagnosis_date is None:
        return ()
    return (_STUDY_REFERENCE_DIAGNOSIS_DATE_PATH,)


def _build_condition(
    raw: Mapping[str, Any],
    *,
    study_reference: StudyReference,
) -> ConditionSpec:
    """Resolve a single YAML condition block into a ConditionSpec.

    YAML shapes:
    - ``seed_path: <path-to-json>`` (relative paths resolved against REPO_ROOT)
    - ``inline_seed: {...}`` (literal partial-or-full baseline dict)
    - Neither → spontaneous condition (empty seed)
    - Both → ValueError

    Validates the resolved seed via ``validate_baseline_seed`` so a typo
    fails before any LLM call. Computes pinned_paths + seed_hash.
    """
    cid = str(raw["id"])
    seed_path_raw = raw.get("seed_path")
    inline_seed = raw.get("inline_seed")
    if seed_path_raw is not None and inline_seed is not None:
        raise ValueError(
            f"condition {cid!r}: set seed_path OR inline_seed, not both"
        )
    if seed_path_raw is not None:
        seed_path = _resolve_repo_path(Path(seed_path_raw))
        seed: Mapping[str, Any] = json.loads(seed_path.read_text(encoding="utf-8"))
    elif inline_seed is not None:
        if not isinstance(inline_seed, Mapping):
            raise ValueError(
                f"condition {cid!r}: inline_seed must be a mapping; "
                f"got {type(inline_seed).__name__}"
            )
        seed = dict(inline_seed)
    else:
        seed = {}
    if not isinstance(seed, Mapping):
        raise ValueError(
            f"condition {cid!r}: resolved seed must be a mapping; "
            f"got {type(seed).__name__}"
        )
    seed = _apply_study_reference_to_seed(
        seed,
        study_reference=study_reference,
        context=f"condition {cid!r}",
    )
    validate_baseline_seed(seed)
    pinned = compute_pinned_paths(seed)
    try:
        BaselineObject.model_validate(seed)
        is_full = True
    except ValidationError:
        is_full = False
    return ConditionSpec(
        id=cid,
        baseline_seed=seed,
        pinned_paths=pinned,
        seed_hash=compute_seed_hash(seed),
        is_full_baseline=is_full,
        label_override=(
            str(raw["label_override"]) if raw.get("label_override") else None
        ),
    )


def _build_condition_from_seed_fields(
    raw: Mapping[str, Any],
    *,
    default_id: str,
    study_reference: StudyReference,
) -> ConditionSpec:
    """Build a ConditionSpec from either legacy or cell-style seed fields."""
    condition_raw: dict[str, Any] = {"id": str(raw.get("condition_id") or default_id)}
    if "label_override" in raw:
        condition_raw["label_override"] = raw["label_override"]
    if "seed_path" in raw:
        condition_raw["seed_path"] = raw["seed_path"]
    if "inline_seed" in raw:
        condition_raw["inline_seed"] = raw["inline_seed"]
    if "baseline_seed_path" in raw:
        if "seed_path" in condition_raw:
            raise ValueError(
                f"cell {default_id!r}: set seed_path OR baseline_seed_path, not both"
            )
        condition_raw["seed_path"] = raw["baseline_seed_path"]
    if "baseline_seed" in raw:
        if "inline_seed" in condition_raw:
            raise ValueError(
                f"cell {default_id!r}: set inline_seed OR baseline_seed, not both"
            )
        condition_raw["inline_seed"] = raw["baseline_seed"]
    return _build_condition(condition_raw, study_reference=study_reference)


def _build_fixed_first_line(raw: Mapping[str, Any] | None) -> FixedFirstLineTreatment | None:
    if raw is None:
        return None
    return FixedFirstLineTreatment(
        regimen_label=str(raw["regimen_label"]),
        drugs=tuple(str(d) for d in raw.get("drugs", [])),
        schedule=str(raw["schedule"]) if raw.get("schedule") else None,
    )


def _build_prompt_enum_order(raw: Mapping[str, Any] | None) -> PromptEnumOrder:
    if raw is None:
        return PromptEnumOrder()
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"prompt.enum_order must be a mapping; got {type(raw).__name__}"
        )
    mode = str(raw.get("mode", "schema_order"))
    if mode not in ("schema_order", "deterministic_shuffle"):
        raise ValueError(
            "prompt.enum_order.mode must be 'schema_order' or "
            f"'deterministic_shuffle'; got {mode!r}"
        )
    seed = str(raw.get("seed", "")).strip()
    if mode == "deterministic_shuffle" and not seed:
        raise ValueError("prompt.enum_order.seed is required for deterministic_shuffle")
    return PromptEnumOrder(mode=mode, seed=seed)


def _first_treatment_policy_text(
    treatment: FixedFirstLineTreatment,
    aliases: tuple[str, ...],
) -> str:
    label = aliases[0] if aliases else treatment.regimen_label
    parenthetical = "/".join(treatment.drugs) if treatment.drugs else treatment.regimen_label
    return (
        f"The first systemic_treatment_start must be {label}\n"
        f"({parenthetical}).\n\n"
        "No other run-specific instructions."
    )


def _build_policy_from_cell(raw: Mapping[str, Any], *, default_id: str) -> PolicySpec:
    first_treatment_raw = raw.get("first_treatment")
    if first_treatment_raw is not None and not isinstance(first_treatment_raw, Mapping):
        raise ValueError(
            f"cell {default_id!r}: first_treatment must be a mapping or null; "
            f"got {type(first_treatment_raw).__name__}"
        )

    fixed_first_line = _build_fixed_first_line(first_treatment_raw)
    if fixed_first_line is None:
        policy_id = str(raw.get("policy_id") or "unconstrained")
        return PolicySpec(
            id=policy_id,
            text=str(raw.get("operator_constraints") or NO_ADDITIONAL_POLICY_TEXT),
            regimen_sequence=(),
            unconstrained_after_line=None,
            fixed_first_line_treatment=None,
        )

    aliases_raw = first_treatment_raw.get("aliases") if first_treatment_raw else None
    aliases = (
        tuple(str(a) for a in aliases_raw)
        if isinstance(aliases_raw, Sequence) and not isinstance(aliases_raw, (str, bytes))
        else ()
    )
    if not aliases:
        aliases = (fixed_first_line.regimen_label, *fixed_first_line.drugs)
    policy_id = str(
        raw.get("policy_id") or f"first_treatment_{fixed_first_line.regimen_label}"
    )
    regimen_sequence = tuple(
        RegimenLine(line=row["line"], aliases=tuple(row["aliases"]))
        for row in raw.get("regimen_sequence", [])
    )
    if not regimen_sequence:
        regimen_sequence = (RegimenLine(line=1, aliases=aliases),)
    return PolicySpec(
        id=policy_id,
        text=str(
            raw.get("operator_constraints")
            or _first_treatment_policy_text(fixed_first_line, aliases)
        ),
        regimen_sequence=regimen_sequence,
        unconstrained_after_line=raw.get("unconstrained_after_line"),
        fixed_first_line_treatment=fixed_first_line,
    )


def _build_cell(
    raw: Mapping[str, Any],
    *,
    study_reference: StudyReference,
) -> ConditioningCellSpec:
    cell_id = str(raw["id"])
    condition = _build_condition_from_seed_fields(
        raw,
        default_id=cell_id,
        study_reference=study_reference,
    )
    policy = _build_policy_from_cell(raw, default_id=cell_id)
    conditioning_spec = _build_conditioning_spec_for_cell(raw, condition, policy)

    # Preflight validation of the observed prefix (if any) fires once per
    # cell at config-resolution time, not per attempt. Uses the final
    # resolved baseline seed (after study-reference application) so the
    # diagnosis floor matches what the runner will use at generation time.
    diagnosis_date = _extract_diagnosis_date_from_seed(condition.baseline_seed)
    validate_observed_prefix(
        conditioning_spec, baseline_diagnosis_date=diagnosis_date
    )

    return ConditioningCellSpec(
        id=cell_id,
        condition=condition,
        policy=policy,
        conditioning_spec=conditioning_spec,
    )


def _build_conditioning_spec_for_cell(
    raw: Mapping[str, Any],
    condition: ConditionSpec,
    policy: PolicySpec,
) -> ConditioningSpec:
    """Build the generalized ``ConditioningSpec`` for a cell.

    Invariant: ``ConditionSpec.baseline_seed`` and ``ConditioningSpec.baseline_seed``
    are the SAME resolved seed (after study-reference application). We
    enforce this by injecting ``condition.baseline_seed`` into whichever
    payload we route through, and dropping ``baseline_seed_path`` so
    ``lower_conditioning_cell`` cannot re-load from disk and silently diverge.

    Routing:

    - When the cell carries a nested ``conditioning:`` block (the generalized
      three-axis YAML shape, e.g. with ``fixed_events`` for observed-prefix
      mode), the block is passed through to ``lower_conditioning_cell`` as
      ``conditioning_payload=``. This is the only path that yields a
      non-empty ``observed_prefix`` on the resulting spec.
    - Otherwise (the universal case in current generation configs), the legacy
      lowering path captures the ``+1L`` declaration as a
      ``FutureEventConstraint`` (per A1) and the resulting spec has
      ``observed_prefix=None``.

    A non-mapping ``conditioning`` value is a YAML typo and raises
    ``ValueError`` rather than silently falling back to the legacy path.
    """
    nested = raw.get("conditioning")
    if nested is not None and not isinstance(nested, Mapping):
        raise ValueError(
            f"cell {raw.get('id')!r}: `conditioning` must be a mapping when "
            f"present; got {type(nested).__name__}. A YAML typo here would "
            "otherwise fall back to legacy lowering silently."
        )
    if isinstance(nested, Mapping) and nested:
        conditioning = dict(nested)
        conditioning["baseline_seed"] = condition.baseline_seed
        return lower_conditioning_cell(conditioning_payload=conditioning)

    resolved_cell: dict[str, Any] = dict(raw)
    resolved_cell["baseline_seed"] = condition.baseline_seed
    resolved_cell.pop("baseline_seed_path", None)
    return lower_conditioning_cell(
        condition_payload=resolved_cell,
        policy_payload=resolved_cell,
    )


def _extract_diagnosis_date_from_seed(
    baseline_seed: Mapping[str, Any],
) -> date | None:
    try:
        raw = baseline_seed["identity_card"]["diagnosis"]["diagnosis_date"]
    except (KeyError, TypeError):
        return None
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str):
        try:
            return date.fromisoformat(raw)
        except ValueError:
            return None
    return None


def _dedupe_conditions(cells: Sequence[ConditioningCellSpec]) -> tuple[ConditionSpec, ...]:
    out: list[ConditionSpec] = []
    seen: dict[str, str] = {}
    for cell in cells:
        condition = cell.condition
        previous_hash = seen.get(condition.id)
        if previous_hash is not None:
            if previous_hash != condition.seed_hash:
                raise ValueError(
                    f"condition_id {condition.id!r} is used by multiple cells "
                    "with different baseline seeds"
                )
            continue
        seen[condition.id] = condition.seed_hash
        out.append(condition)
    return tuple(out)


def _dedupe_policies(cells: Sequence[ConditioningCellSpec]) -> tuple[PolicySpec, ...]:
    out: list[PolicySpec] = []
    seen: dict[str, dict[str, Any]] = {}
    for cell in cells:
        policy = cell.policy
        payload = _canonical_policy_payload(policy)
        previous_payload = seen.get(policy.id)
        if previous_payload is not None:
            if canonical_json(previous_payload) != canonical_json(payload):
                raise ValueError(
                    f"policy_id {policy.id!r} is used by multiple cells with "
                    "different policy payloads"
                )
            continue
        seen[policy.id] = payload
        out.append(policy)
    return tuple(out)


def _build_config(raw: Mapping[str, Any]) -> POCConfig:
    defaults = raw.get("run_defaults", {})
    prompt_block = raw.get("prompt", {})
    provider_block = raw.get("provider", {})
    study_reference = _build_study_reference(raw)
    prompt_enum_order = _build_prompt_enum_order(prompt_block.get("enum_order"))

    models = tuple(ModelSpec(id=m["id"], short=m["short"]) for m in raw.get("models", []))
    explicit_cells = tuple(
        _build_cell(c, study_reference=study_reference) for c in raw.get("cells", [])
    )
    if explicit_cells:
        if raw.get("conditions") or raw.get("policies"):
            raise ValueError(
                "configs with cells: must not also define top-level conditions: "
                "or policies:"
            )
        conditions = _dedupe_conditions(explicit_cells)
        policies = _dedupe_policies(explicit_cells)
    else:
        conditions = tuple(
            _build_condition(c, study_reference=study_reference)
            for c in raw.get("conditions", [])
        )
        policies = tuple(
            PolicySpec(
                id=p["id"],
                text=p["text"],
                regimen_sequence=tuple(
                    RegimenLine(line=row["line"], aliases=tuple(row["aliases"]))
                    for row in p.get("regimen_sequence", [])
                ),
                unconstrained_after_line=p.get("unconstrained_after_line"),
                fixed_first_line_treatment=_build_fixed_first_line(
                    p.get("fixed_first_line_treatment")
                ),
            )
            for p in raw.get("policies", [])
        )

    raw_contract = defaults.get("terminal_contract", DEFAULT_TERMINAL_CONTRACT)
    if raw_contract not in ("death_or_last_followup", "death_only"):
        raise ValueError(
            f"terminal_contract must be 'death_or_last_followup' or 'death_only'; "
            f"got {raw_contract!r}"
        )

    raw_accepted_target = defaults.get("accepted_futures_per_cell")
    accepted_target = (
        int(raw_accepted_target) if raw_accepted_target is not None else None
    )
    raw_futures = defaults.get("futures_per_cell", accepted_target)
    if raw_futures is None:
        raise ValueError(
            "run_defaults must define futures_per_cell, or "
            "accepted_futures_per_cell with max_total_futures_per_cell"
        )
    raw_cap = defaults.get("max_total_futures_per_cell")
    max_total_futures_per_cell = int(raw_cap) if raw_cap is not None else None

    return POCConfig(
        schema_version=int(raw["schema_version"]),
        config_id=str(raw["config_id"]),
        conditions_count=int(
            defaults.get(
                "cells_count",
                defaults.get("conditions_count", len(explicit_cells) if explicit_cells else len(conditions)),
            )
        ),
        futures_per_cell=int(raw_futures),
        max_attempts=int(defaults["max_attempts"]),
        prompt_template_path=Path(prompt_block["template_path"]),
        use_openrouter=bool(provider_block.get("use_openrouter", True)),
        response_format=dict(provider_block.get("response_format", {})),
        models=models,
        conditions=conditions,
        policies=policies,
        require_parameters=bool(provider_block.get("require_parameters", False)),
        prompt_enum_order=prompt_enum_order,
        terminal_contract=raw_contract,
        response_contract=_resolve_response_contract(prompt_block),
        study_reference=study_reference,
        label_prefix=str(raw.get("label_prefix", "")),
        cells=explicit_cells,
        accepted_futures_per_cell=accepted_target,
        max_total_futures_per_cell=max_total_futures_per_cell,
    )


def apply_overrides(
    config: POCConfig,
    *,
    conditions_count: int | None = None,
    futures_per_cell: int | None = None,
    accepted_futures_per_cell: int | None = None,
    max_total_futures_per_cell: int | None = None,
    max_attempts: int | None = None,
    terminal_contract: TerminalContract | None = None,
    label_prefix: str | None = None,
) -> POCConfig:
    """Return a new config with selected fields overridden."""
    resolved_accepted_target = (
        accepted_futures_per_cell
        if accepted_futures_per_cell is not None
        else config.accepted_futures_per_cell
    )
    resolved_futures_per_cell = (
        futures_per_cell
        if futures_per_cell is not None
        else (
            accepted_futures_per_cell
            if accepted_futures_per_cell is not None
            else config.futures_per_cell
        )
    )
    return POCConfig(
        schema_version=config.schema_version,
        config_id=config.config_id,
        conditions_count=(
            conditions_count if conditions_count is not None else config.conditions_count
        ),
        futures_per_cell=resolved_futures_per_cell,
        max_attempts=max_attempts if max_attempts is not None else config.max_attempts,
        prompt_template_path=config.prompt_template_path,
        use_openrouter=config.use_openrouter,
        response_format=config.response_format,
        models=config.models,
        conditions=config.conditions,
        policies=config.policies,
        require_parameters=config.require_parameters,
        prompt_enum_order=config.prompt_enum_order,
        terminal_contract=(
            terminal_contract if terminal_contract is not None else config.terminal_contract
        ),
        response_contract=config.response_contract,
        study_reference=config.study_reference,
        label_prefix=label_prefix if label_prefix is not None else config.label_prefix,
        cells=config.cells,
        accepted_futures_per_cell=resolved_accepted_target,
        max_total_futures_per_cell=(
            max_total_futures_per_cell
            if max_total_futures_per_cell is not None
            else config.max_total_futures_per_cell
        ),
    )


# ---------- matrix expansion ----------


def _trajectory_slots_per_cell(config: POCConfig) -> int:
    if config.accepted_futures_per_cell is None:
        return config.futures_per_cell
    if config.max_total_futures_per_cell is None:
        raise ValueError(
            "max_total_futures_per_cell must be set when "
            "accepted_futures_per_cell is set"
        )
    return config.max_total_futures_per_cell


def expand_matrix(config: POCConfig) -> tuple[tuple[CellKey, tuple[TrajectoryKey, ...]], ...]:
    """Expand (conditioning cell x model x future).

    New-style configs provide explicit ``cells:`` where each cell already
    combines baseline pins and optional first-treatment pins. Legacy configs
    are adapted into explicit cells from the old condition x policy product.
    """
    if config.conditions_count <= 0:
        raise ValueError(f"cells_count/conditions_count must be >= 1; got {config.conditions_count}")
    if config.futures_per_cell <= 0:
        raise ValueError(f"futures_per_cell must be >= 1; got {config.futures_per_cell}")
    if config.accepted_futures_per_cell is not None:
        if config.accepted_futures_per_cell <= 0:
            raise ValueError(
                "accepted_futures_per_cell must be >= 1; "
                f"got {config.accepted_futures_per_cell}"
            )
        if config.max_total_futures_per_cell is None:
            raise ValueError(
                "max_total_futures_per_cell must be set when "
                "accepted_futures_per_cell is set"
            )
        if config.max_total_futures_per_cell < config.accepted_futures_per_cell:
            raise ValueError(
                "max_total_futures_per_cell must be >= accepted_futures_per_cell; "
                f"got {config.max_total_futures_per_cell} < "
                f"{config.accepted_futures_per_cell}"
            )
    if config.max_attempts <= 0:
        raise ValueError(f"max_attempts must be >= 1; got {config.max_attempts}")
    available_cells = len(config.cells) if config.cells else len(config.conditions)
    label = "cells" if config.cells else "conditions"
    if config.conditions_count > available_cells:
        raise ValueError(
            f"cells_count/conditions_count={config.conditions_count} exceeds available "
            f"{label} ({available_cells}). Add a {label[:-1]} block to extend the matrix."
        )

    selected = config.selected_cells()
    slots_per_cell = _trajectory_slots_per_cell(config)
    cells: list[tuple[CellKey, tuple[TrajectoryKey, ...]]] = []
    for conditioning_cell in selected:
        condition = conditioning_cell.condition
        policy = conditioning_cell.policy
        label_ = _cell_label(
            config,
            condition,
            policy,
            conditioning_cell.conditioning_spec,
        )
        hash_ = _cell_hash(condition, policy, conditioning_cell.conditioning_spec)
        for model in config.models:
            cell = CellKey(
                cell_id=conditioning_cell.id,
                condition_id=condition.id,
                policy_id=policy.id,
                model_id=model.id,
                model_short=model.short,
                conditioning_label=label_,
                conditioning_hash=hash_,
            )
            trajectories = tuple(
                TrajectoryKey(cell=cell, future_index=i)
                for i in range(1, slots_per_cell + 1)
            )
            cells.append((cell, trajectories))
    return tuple(cells)


def _canonical_policy_payload(policy: PolicySpec) -> dict[str, Any]:
    """Canonical hash-input for a policy. Excludes ``id`` (metadata)."""
    ffl = policy.fixed_first_line_treatment
    return {
        "text": policy.text,
        "regimen_sequence": [
            {"line": row.line, "aliases": list(row.aliases)}
            for row in policy.regimen_sequence
        ],
        "unconstrained_after_line": policy.unconstrained_after_line,
        "fixed_first_line_treatment": (
            None if ffl is None
            else {
                "regimen_label": ffl.regimen_label,
                "drugs": list(ffl.drugs),
                "schedule": ffl.schedule,
            }
        ),
    }


def _cell_hash(
    condition: ConditionSpec,
    policy: PolicySpec,
    conditioning_spec: ConditioningSpec | None = None,
) -> str:
    if conditioning_spec is not None:
        return compute_conditioning_spec_hash(conditioning_spec)
    return compute_conditioning_hash(
        seed_hash=condition.seed_hash,
        canonical_policy_payload=_canonical_policy_payload(policy),
    )


def _label_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unspecified"


def _fixed_prefix_treatment_label(spec: ConditioningSpec | None) -> str | None:
    if spec is None or spec.observed_prefix is None:
        return None
    label: str | None = None
    for event in spec.observed_prefix.events:
        if event.type not in {"systemic_treatment_start", "systemic_treatment_switch"}:
            continue
        if event.regimen is None or not event.regimen.label:
            continue
        label = event.regimen.label
    return _label_slug(label) if label else None


def _cell_label(
    config: POCConfig,
    condition: ConditionSpec,
    policy: PolicySpec,
    conditioning_spec: ConditioningSpec | None,
) -> str:
    """Compose the per-cell label as ``[prefix:]<core>[+treatment=<regimen>]``.

    ``label_override`` replaces the condition-derived core only; treatment
    suffixes are appended from either the legacy first-line policy object or
    the last systemic treatment in a fixed prefix. Hash is unaffected either
    way and remains the authoritative identity.
    """
    if condition.label_override is not None:
        core = condition.label_override
    else:
        core = derive_conditioning_label(
            baseline_seed=condition.baseline_seed,
            pinned_paths=condition.pinned_paths,
            fixed_first_line_regimen_label=None,
            label_ignored_pinned_paths=_study_reference_label_ignored_paths(
                config.study_reference
            ),
        )
    ffl = policy.fixed_first_line_treatment
    if ffl is not None:
        core = f"{core}+1L={ffl.regimen_label}"
    else:
        fixed_prefix_label = _fixed_prefix_treatment_label(conditioning_spec)
        if fixed_prefix_label:
            core = f"{core}+fixed_tx={fixed_prefix_label}"
    if config.label_prefix:
        return f"{config.label_prefix}:{core}"
    return core


def total_trajectories(matrix: Sequence[tuple[CellKey, Sequence[TrajectoryKey]]]) -> int:
    return sum(len(trajs) for _, trajs in matrix)


# ---------- prompt rendering ----------


def render_prompt(
    template_text: str,
    *,
    baseline_seed: Mapping[str, Any],
    pinned_paths: tuple[str, ...],
    required_patch_leaves: tuple[str, ...],
    policy_text: str,
    knowledge_cutoff: str = DEFAULT_STUDY_KNOWLEDGE_CUTOFF,
    terminal_contract: TerminalContract = "death_only",
    conditioning_spec: ConditioningSpec | None = None,
    prompt_enum_order: PromptEnumOrder = PromptEnumOrder(),
) -> str:
    """Substitute the prompt-template placeholders for the rich lean-ledger prompt.

    Placeholders:
    - ``{{LEDGER_EVENT_SCHEMA}}`` — Pydantic schema of ``LedgerEvent``.
    - ``{{BASELINE_SCHEMA}}`` — Pydantic schema of ``BaselineObject``.
    - ``{{BASELINE_SEED}}`` — the partial-or-empty-or-full seed (pinned values).
    - ``{{PINNED_FIELDS}}`` — bulleted list of dotted leaf paths the model
      MUST NOT write in its ``baseline_patch``.
    - ``{{REQUIRED_PATCH_LEAVES}}`` — bulleted list of required leaves the
      model MUST fill in its ``baseline_patch`` (list-valued required fields
      appear as whole leaves, not pseudo-indexed).
    - ``{{TASK_OPENING}}`` — from-diagnosis wording for ordinary cells;
      continuation wording when a fixed prefix is present.
    - ``{{KNOWLEDGE_CUTOFF}}`` — study-level clinical knowledge horizon.
    - ``{{OBSERVED_PREFIX}}`` — block describing fixed prefix events.
      Empty string when ``conditioning_spec`` is ``None`` or its
      ``observed_prefix`` is ``None`` (no-behavior-change for legacy cells).
      Populated ``## Fixed Prefix`` block otherwise — see
      :func:`_render_observed_prefix_block`.
    - ``{{POLICY_TEXT}}`` — operator-supplied policy narrative plus any
      generalized ``ConditioningSpec.free_text_operator_constraints`` that
      would otherwise be audit-only.
    """
    ledger_schema_blob = json.dumps(
        _apply_prompt_enum_order(
            _ledger_event_schema_for_prompt(terminal_contract),
            prompt_enum_order,
        ),
        indent=2,
        sort_keys=True,
    )
    baseline_schema_blob = json.dumps(
        _apply_prompt_enum_order(
            BaselineObject.model_json_schema(),
            prompt_enum_order,
        ),
        indent=2,
        sort_keys=True,
    )
    seed_blob = json.dumps(baseline_seed, indent=2, sort_keys=True)
    pinned_block = (
        "\n".join(f"- `{p}`" for p in pinned_paths)
        if pinned_paths
        else "_(none — seed is empty; you may write anywhere in baseline_patch)_"
    )
    required_block = (
        "\n".join(f"- `{p}`" for p in required_patch_leaves)
        if required_patch_leaves
        else "_(none — the seed already covers every required leaf; emit `{}` for baseline_patch)_"
    )
    observed_prefix_block = _render_observed_prefix_block(conditioning_spec)
    task_opening = _task_opening_for(conditioning_spec)
    effective_policy_text = _effective_policy_text(policy_text, conditioning_spec)
    return (
        template_text
        .replace("{{TASK_OPENING}}", task_opening)
        .replace("{{KNOWLEDGE_CUTOFF}}", knowledge_cutoff)
        .replace("{{LEDGER_EVENT_SCHEMA}}", ledger_schema_blob)
        .replace("{{BASELINE_SCHEMA}}", baseline_schema_blob)
        .replace("{{BASELINE_SEED}}", seed_blob)
        .replace("{{PINNED_FIELDS}}", pinned_block)
        .replace("{{REQUIRED_PATCH_LEAVES}}", required_block)
        # Gate A3.5b: conditional expansion. Empty string for no-prefix
        # cells (preserves A3.5a no-behavior-change promise); populated
        # ``## Fixed Prefix`` block when ``conditioning_spec.observed_prefix``
        # is non-None.
        .replace("{{OBSERVED_PREFIX}}", observed_prefix_block)
        .replace("{{POLICY_TEXT}}", effective_policy_text)
    )


def _apply_prompt_enum_order(
    schema: dict[str, Any],
    enum_order: PromptEnumOrder,
) -> dict[str, Any]:
    if enum_order.mode == "schema_order":
        return schema
    if enum_order.mode != "deterministic_shuffle":
        raise ValueError(f"unsupported prompt enum order mode: {enum_order.mode!r}")
    out = deepcopy(schema)
    _shuffle_schema_enums(out, enum_order.seed, path=())
    return out


def _shuffle_schema_enums(node: Any, seed: str, *, path: tuple[str, ...]) -> None:
    if isinstance(node, dict):
        for key, value in list(node.items()):
            child_path = (*path, str(key))
            if key == "enum" and isinstance(value, list) and len(value) > 1:
                node[key] = sorted(
                    value,
                    key=lambda item: sha256_hex(
                        "\x1f".join(
                            (
                                seed,
                                ".".join(path),
                                json.dumps(item, sort_keys=True),
                            )
                        )
                    ),
                )
            else:
                _shuffle_schema_enums(value, seed, path=child_path)
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            _shuffle_schema_enums(value, seed, path=(*path, str(idx)))


def _task_opening_for(conditioning_spec: ConditioningSpec | None) -> str:
    if conditioning_spec is not None and conditioning_spec.observed_prefix is not None:
        return (
            "Continue this latent cancer-patient trajectory from the fixed "
            "prefix until death."
        )
    return (
        "Generate one complete latent cancer-patient trajectory from diagnosis "
        "until death."
    )


def _effective_policy_text(
    policy_text: str,
    conditioning_spec: ConditioningSpec | None,
) -> str:
    """Return the prompt-visible run-specific instructions.

    The generalized ``ConditioningSpec`` is persisted and audited, but the LLM
    only sees text that flows through ``{{POLICY_TEXT}}``. Keep that boundary
    explicit: when a generalized spec carries free-text constraints, surface
    them in the prompt even if the cell omitted top-level
    ``operator_constraints``.
    """
    text = str(policy_text or "").strip()
    spec_text = (
        ""
        if conditioning_spec is None
        else str(conditioning_spec.free_text_operator_constraints or "").strip()
    )
    if not spec_text:
        return text
    if not text or text in _NO_ADDITIONAL_POLICY_TEXTS:
        return spec_text

    normalized_text = " ".join(text.split())
    normalized_spec = " ".join(spec_text.split())
    if normalized_text == normalized_spec or normalized_spec in normalized_text:
        return text
    if normalized_text in normalized_spec:
        return spec_text

    return (
        f"{text}\n\n"
        "Conditioning-spec constraints audited after replay:\n"
        f"{spec_text}"
    )


def _render_observed_prefix_block(spec: ConditioningSpec | None) -> str:
    """Render the ``{{OBSERVED_PREFIX}}`` substitution.

    Returns ``""`` when no prefix is in scope (the A3.5a contract). Returns
    a populated ``## Fixed Prefix`` block when ``spec`` carries a
    non-empty observed/fixed prefix.

    Anti-coaching discipline: the block contains schema-control framing
    only (cutoff date rule, "do not restate prefix events"). No clinical
    guidance, no behavioral cues — the lint test
    ``test_rich_prompt_forbids_behavioral_coaching_phrases`` must stay
    green.
    """
    if spec is None or spec.observed_prefix is None:
        return ""
    prefix = spec.observed_prefix
    events_json = json.dumps(
        [ev.model_dump(mode="json", exclude_none=True) for ev in prefix.events],
        indent=2,
        sort_keys=True,
    )
    cutoff_iso = prefix.cutoff_date.isoformat()
    return (
        "## Fixed Prefix\n"
        "\n"
        "These events are fixed for this run: observed history and/or "
        "operator-imposed counterfactual interventions. Your ledger MUST "
        f"contain ONLY events with dates strictly after cutoff_date={cutoff_iso}. "
        "The first event you emit must also be strictly after the cutoff date. "
        "Do not restate or alter these events.\n"
        "\n"
        f"cutoff_date: {cutoff_iso}\n"
        "events:\n"
        f"{events_json}"
    )


def _ledger_event_schema_for_prompt(
    terminal_contract: TerminalContract,
) -> dict[str, Any]:
    schema = LedgerEvent.model_json_schema()
    if terminal_contract != "death_only":
        return schema
    type_schema = schema.get("properties", {}).get("type")
    if isinstance(type_schema, dict) and isinstance(type_schema.get("enum"), list):
        type_schema["enum"] = [v for v in type_schema["enum"] if v != "last_followup"]
    return schema


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


# ---------- attempt classification ----------


@dataclass(frozen=True, slots=True)
class ParsedAttempt:
    ledger_dicts: list[dict[str, Any]]
    ledger_events: list[LedgerEvent]
    materialized: MaterializedTrajectory


def _canonical_ledger_shape_failure(
    ledger: Sequence[Mapping[str, Any]],
) -> str | None:
    """Reject new LLM emissions that rely on historical event-shape rewrites."""
    for index, event in enumerate(ledger):
        original = dict(event)
        canonical = canonicalize_legacy_event_dict(original)
        if canonical == original:
            continue
        affected = sorted(set(original) | set(canonical))
        return (
            f"ledger[{index}] uses an old compatibility event shape that would "
            "be rewritten before validation; emit the current canonical schema "
            f"directly. Affected key(s): {', '.join(affected)}"
        )
    return None


def classify_response(
    raw_text: str,
    *,
    baseline_seed: Mapping[str, Any],
    pinned_paths: tuple[str, ...],
    seed_is_complete: bool = False,
    policy: PolicySpec | None = None,
    terminal_contract: TerminalContract = DEFAULT_TERMINAL_CONTRACT,
    conditioning_spec: ConditioningSpec | None = None,
    response_contract: ResponseContract = "baseline_patch_and_ledger",
) -> tuple[str | None, str | None, ParsedAttempt | None]:
    """Classify a raw model response. Returns (failure_kind, message, parsed).

    ``response_contract`` selects WHICH contract is enforced, and the two are not
    interchangeable. Under ``ledger_only`` the prompt tells the model there is no
    ``baseline_patch`` in this contract and that emitting one is hard-rejected, so demanding it
    here would reject every correct answer -- which is exactly what happened to a whole smoke
    batch: the model obeyed the continuation prompt and the parser insisted on the old shape.

    Enforces the patch-emission contract + acceptance gates:
    1. Response must be ``{"baseline_patch": {...}, "ledger": [...]}`` under the default
       contract, or ``{"ledger": [...]}`` under ``ledger_only``.
    2. When ``seed_is_complete`` (full-baseline regime), ``baseline_patch``
       MUST be ``{}``. A non-empty patch — even on optional unpinned leaves
       — is rejected (failure_kind="conditioning_audit"). This closes the
       loophole where the seed validates but omits optional leaves.
    3. ``baseline_patch`` MUST NOT collide with any pinned path
       (failure_kind="conditioning_audit").
    4. Deep-merge(seed, patch) MUST validate as ``BaselineObject``
       (failure_kind="baseline_validation").
    5. Ledger events use the current canonical schema directly and parse as
       ``LedgerEvent`` (failure_kind="ledger_validation").
    6. The generation contract passes
       (failure_kind="generation_contract").
    7. When ``policy.fixed_first_line_treatment`` is set, the first
       ``systemic_treatment_start`` event's regimen MUST match one of
       ``policy.regimen_sequence[line=1].aliases``
       (failure_kind="intervention_audit"). This makes the structured 1L
       pin a hard acceptance gate, not a downstream summary signal.
    8. Replay succeeds (failure_kind="replay").
    9. Terminal-contract gate (failure_kind="terminal_contract").
    """
    try:
        parsed = json.loads(strip_code_fence(raw_text))
    except json.JSONDecodeError as exc:
        return ("parse", f"{type(exc).__name__}: {exc}", None)

    if not isinstance(parsed, dict):
        return ("parse", f"response root must be a JSON object; got {type(parsed).__name__}", None)
    if response_contract == "ledger_only":
        # The continuation prompt requires exactly one top-level key. Checking the key set,
        # rather than ``parsed.get()``, also rejects ``{"baseline_patch": null}``, which is
        # present even though ``get`` makes it indistinguishable from absence.
        unexpected = sorted(set(parsed) - {"ledger"})
        if unexpected:
            return ("parse",
                    "this run uses the continuation contract: the response must contain only "
                    f"'ledger'; unexpected top-level key(s): {', '.join(unexpected)}",
                    None)
        patch = {}
    else:
        patch = parsed.get("baseline_patch")
        if not isinstance(patch, dict):
            return ("parse", "response must contain a 'baseline_patch' object", None)
    ledger = parsed.get("ledger")
    if not isinstance(ledger, list):
        return ("parse", "response must contain a 'ledger' array", None)
    if not all(isinstance(item, dict) for item in ledger):
        return ("parse", "every ledger item must be a JSON object", None)

    if seed_is_complete and patch:
        # Full-baseline regime: seed already specifies a complete valid
        # BaselineObject. Any non-empty patch (even on optional unpinned
        # leaves like diagnosis_label or biomarkers) violates the contract.
        return (
            "conditioning_audit",
            "baseline_patch must be {} when the condition seed already "
            f"specifies a complete BaselineObject; got patch with "
            f"{len(patch)} top-level key(s)",
            None,
        )

    collisions = assert_no_pin_collision(patch, pinned_paths)
    if collisions:
        descriptions = "; ".join(
            f"patch {c.patch_path!r} vs pinned {c.pinned_path!r} ({c.relation})"
            for c in collisions
        )
        return (
            "conditioning_audit",
            f"baseline_patch collides with pinned path(s): {descriptions}",
            None,
        )

    try:
        resolved = resolve_and_validate(baseline_seed, patch)
    except BaselineResolutionError as exc:
        return ("baseline_validation", str(exc), None)

    canonical_shape_failure = _canonical_ledger_shape_failure(ledger)
    if canonical_shape_failure is not None:
        return ("ledger_validation", canonical_shape_failure, None)

    try:
        events = [LedgerEvent.model_validate(event) for event in ledger]
    except ValidationError as exc:
        return ("ledger_validation", str(exc), None)

    generation_contract_failure = check_generation_contract(resolved, events)
    if generation_contract_failure is not None:
        return ("generation_contract", generation_contract_failure, None)

    if conditioning_spec is None:
        # Legacy path: the original ``+1L`` intervention audit fires on the
        # raw emitted ledger. Unchanged.
        if policy is not None and policy.fixed_first_line_treatment is not None:
            violation = _check_first_line_match(events, policy)
            if violation is not None:
                return ("intervention_audit", violation, None)
    else:
        # Generalized path: ``future_constraints_audit`` (post-replay) is
        # the single audit gate for future-event constraints; it subsumes
        # the legacy ``intervention_audit``. The legacy gate is explicitly
        # skipped to avoid dual-audit failure-message confusion.
        if conditioning_spec.observed_prefix is not None:
            prefix_collisions = assert_no_prefix_collision(conditioning_spec, events)
            if prefix_collisions:
                detail = "; ".join(c.detail for c in prefix_collisions)
                return ("observed_prefix_audit", detail, None)

    # Build the event list for replay. In prefix mode the materialized
    # trajectory is prefix + suffix so downstream artifacts (figures,
    # viewer) see the whole patient timeline. ``parsed_ledger.json``
    # remains the emitted suffix only — that split happens in
    # ``run_one_attempt``.
    if (
        conditioning_spec is not None
        and conditioning_spec.observed_prefix is not None
    ):
        events_for_replay = list(conditioning_spec.observed_prefix.events) + events
    else:
        events_for_replay = events

    try:
        materialized = replay(resolved.model_dump(mode="json"), events_for_replay)
    except ReplayError as exc:
        return ("replay", str(exc), None)

    # Post-replay failures (future_constraints_audit, terminal_contract)
    # return the ``ParsedAttempt`` alongside the failure tuple so the
    # runner can preserve replay-derived signals (validation_log,
    # event_count, snapshot_count) on the failed attempt's result.json.
    # B1b uses this for ``attempt_rows.validation_warnings_joined`` on
    # rejected-but-replayed attempts.
    parsed = ParsedAttempt(
        ledger_dicts=ledger, ledger_events=events, materialized=materialized
    )

    if (
        conditioning_spec is not None
        and conditioning_spec.future_event_constraints
    ):
        constraint_results = future_constraints_audit(
            conditioning_spec, materialized.ledger
        )
        failures = [r for r in constraint_results if not r.passed]
        if failures:
            detail = "; ".join(
                f"{r.constraint_id}: {r.failure_reason}" for r in failures
            )
            return ("future_constraints_audit", detail, parsed)

    if terminal_contract == "death_only":
        # The terminal check operates on the emitted suffix (``events``),
        # not on the materialized ledger. ``validate_observed_prefix`` has
        # already rejected prefixes ending in death (events past a death
        # are incoherent), so the suffix's last event is the trajectory's
        # actual terminal.
        terminal_type = events[-1].type if events else None
        if terminal_type != "death":
            return (
                "terminal_contract",
                f"terminal_contract='death_only' but trajectory ends in "
                f"{terminal_type!r}",
                parsed,
            )

    return (None, None, parsed)


def _check_first_line_match(
    events: list[LedgerEvent], policy: PolicySpec,
) -> str | None:
    """Enforce the fixed-1L gate. Returns None on match, error message on
    mismatch. Called only when ``policy.fixed_first_line_treatment`` is set.

    Rule: the first ``systemic_treatment_start`` event's regimen must match
    one of the policy's line-1 aliases (case-insensitive, punctuation-
    normalized; either by label-token-superset or by single-drug-component
    membership). Aliases live in ``policy.regimen_sequence[line=1].aliases``;
    when that block is absent, fall back to ``fixed_first_line_treatment``'s
    ``regimen_label`` + ``drugs`` as the alias set.
    """
    first_start = next(
        (e for e in events if e.type == "systemic_treatment_start"), None,
    )
    if first_start is None:
        return (
            "fixed_first_line_treatment is set but ledger contains no "
            "systemic_treatment_start event"
        )
    regimen = first_start.regimen
    if regimen is None:
        return (
            "first systemic_treatment_start has no regimen; cannot verify "
            "against fixed_first_line_treatment"
        )
    aliases = _resolve_line1_aliases(policy)
    observed_label = regimen.label or ""
    observed_components = list(regimen.components or [])
    for alias in aliases:
        if _regimen_matches_alias(observed_label, observed_components, alias):
            return None
    return (
        f"first systemic_treatment_start regimen "
        f"(label={observed_label!r}, components={observed_components!r}) "
        f"does not match fixed_first_line_treatment "
        f"({policy.fixed_first_line_treatment.regimen_label!r}); "
        f"accepted aliases: {list(aliases)!r}"
    )


def _resolve_line1_aliases(policy: PolicySpec) -> tuple[str, ...]:
    for row in policy.regimen_sequence:
        if row.line == 1:
            return row.aliases
    # Policy lacks an explicit line-1 row; use the FFL's regimen_label +
    # drug list as the accepted alias set.
    ffl = policy.fixed_first_line_treatment
    if ffl is None:
        return ()
    return (ffl.regimen_label, *ffl.drugs)


def _normalize_regimen_str(text: str) -> str:
    """Lowercase, normalize punctuation, collapse whitespace.

    Strict-enough for the legacy first-line acceptance gate. This helper stays
    local so the runner does not depend on downstream derivation/readout code."""
    out = text.strip().lower()
    for ch in ("-", "/", "+", ","):
        out = out.replace(ch, " ")
    return " ".join(out.split())


def _regimen_matches_alias(
    observed_label: str, observed_components: list[str], alias: str,
) -> bool:
    a = _normalize_regimen_str(alias)
    if not a:
        return False
    obs_label = _normalize_regimen_str(observed_label)
    if obs_label == a:
        return True
    obs_label_tokens = set(obs_label.split()) if obs_label else set()
    a_tokens = set(a.split())
    if a_tokens and a_tokens.issubset(obs_label_tokens):
        return True
    obs_comp_tokens = {_normalize_regimen_str(c) for c in observed_components}
    obs_comp_tokens.discard("")
    if a in obs_comp_tokens:
        return True
    if a_tokens.issubset({t for c in obs_comp_tokens for t in c.split()}):
        return True
    return False


# ---------- generate-fn dispatchers ----------


GenerateFn = Callable[[ModelSpec, str, str], GenerationResult]


def _provider_call_timeout_seconds() -> float | None:
    """Live provider-call watchdog timeout.

    ``None`` disables the watchdog. The default is intentionally generous; it
    protects throughput from provider-side hangs without converting ordinary
    slow calls into failures.
    """
    raw = os.environ.get(_PROVIDER_CALL_TIMEOUT_ENV)
    if raw is None or raw.strip() == "":
        return _DEFAULT_PROVIDER_CALL_TIMEOUT_SECONDS
    token = raw.strip().lower()
    if token in {"0", "none", "off", "false"}:
        return None
    try:
        value = float(token)
    except ValueError as exc:
        raise RuntimeError(
            f"{_PROVIDER_CALL_TIMEOUT_ENV} must be a positive number of seconds "
            "or one of: 0, none, off, false"
        ) from exc
    if value <= 0:
        raise RuntimeError(
            f"{_PROVIDER_CALL_TIMEOUT_ENV} must be a positive number of seconds "
            "or one of: 0, none, off, false"
        )
    return value


def _provider_generate_child(
    conn: Any,
    *,
    model_id: str,
    use_openrouter: bool,
    response_format: dict | None,
    require_parameters: bool,
    system: str,
    user: str,
) -> None:
    try:
        provider = get_provider(
            model_id,
            use_openrouter=use_openrouter,
            response_format=response_format,
            require_parameters=require_parameters,
        )
        conn.send(("ok", provider.generate(system, user)))
    except BaseException as exc:  # noqa: BLE001 - parent reclassifies as runtime
        conn.send(("error", type(exc).__name__, str(exc)))
    finally:
        conn.close()


def _generate_with_provider_watchdog(
    *,
    model: ModelSpec,
    system: str,
    user: str,
    use_openrouter: bool,
    response_format: dict | None,
    require_parameters: bool,
    timeout_seconds: float,
) -> GenerationResult:
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_provider_generate_child,
        kwargs={
            "conn": child_conn,
            "model_id": model.id,
            "use_openrouter": use_openrouter,
            "response_format": response_format,
            "require_parameters": require_parameters,
            "system": system,
            "user": user,
        },
    )
    proc.start()
    child_conn.close()
    try:
        if not parent_conn.poll(timeout_seconds):
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
            raise RuntimeError(
                f"provider call exceeded {timeout_seconds:.0f}s watchdog "
                f"timeout for model {model.id}; set "
                f"{_PROVIDER_CALL_TIMEOUT_ENV}=0 to disable"
            )
        try:
            status, *payload = parent_conn.recv()
        except EOFError as exc:
            raise RuntimeError(
                f"provider child exited without a result for model {model.id}"
            ) from exc
    finally:
        parent_conn.close()
    proc.join(timeout=5)
    if status == "ok":
        result = payload[0]
        if not isinstance(result, GenerationResult):
            raise RuntimeError(
                "provider child returned unexpected payload type "
                f"{type(result).__name__}"
            )
        return result
    if status == "error":
        exc_type, message = payload
        raise RuntimeError(f"{exc_type}: {message}")
    raise RuntimeError(f"provider child returned unexpected status {status!r}")


def default_generate_fn(config: POCConfig) -> GenerateFn:
    """Build the live provider-backed generate_fn from config provider settings."""
    use_openrouter = config.use_openrouter
    response_format = dict(config.response_format) if config.response_format else None
    require_parameters = config.require_parameters
    provider_call_timeout = _provider_call_timeout_seconds()

    def _generate(model: ModelSpec, system: str, user: str) -> GenerationResult:
        if provider_call_timeout is not None:
            return _generate_with_provider_watchdog(
                model=model,
                system=system,
                user=user,
                use_openrouter=use_openrouter,
                response_format=response_format,
                require_parameters=require_parameters,
                timeout_seconds=provider_call_timeout,
            )
        provider = get_provider(
            model.id,
            use_openrouter=use_openrouter,
            response_format=response_format,
            require_parameters=require_parameters,
        )
        return provider.generate(system, user)

    return _generate


# ---------- per-attempt / per-trajectory orchestration ----------


def _usage_dict(gen: GenerationResult | None) -> dict[str, Any]:
    if gen is None:
        return {}
    return {
        "input_tokens": gen.input_tokens,
        "output_tokens": gen.output_tokens,
        "reasoning_tokens": gen.reasoning_tokens,
        "response_id": gen.response_id,
        "response_model": gen.response_model,
        "retries": gen.retries,
        "max_tokens_sent": gen.max_tokens_sent,
        "max_tokens_param": gen.max_tokens_param,
        "reasoning_mode": gen.reasoning_mode,
        "temperature_sent": gen.temperature_sent,
        "top_p_sent": gen.top_p_sent,
        "seed_sent": gen.seed_sent,
        "finish_reason": gen.finish_reason,
        "duration_s": gen.duration_s,
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_or_verify_bytes(path: Path, data: bytes, *, resume: bool) -> None:
    """Write an immutable run-input snapshot, or verify it on resume."""
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        if path.read_bytes() != data:
            raise ValueError(f"resume refused because run-input snapshot drifted: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_json(path: Path, obj: Any) -> None:
    _write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _attempt_result_to_dict(result: AttemptResult) -> dict[str, Any]:
    return {
        "attempt_index": result.attempt_index,
        "accepted": result.accepted,
        "failure_kind": result.failure_kind,
        "failure_message": result.failure_message,
        "event_count": result.event_count,
        "snapshot_count": result.snapshot_count,
        "validation_log_count": result.validation_log_count,
        "validation_warning_rule_ids": (
            list(result.validation_warning_rule_ids)
            if result.validation_warning_rule_ids is not None
            else None
        ),
        "cost_usd": result.cost_usd,
        "usage": dict(result.usage),
    }


def _attempt_result_from_dict(payload: Mapping[str, Any]) -> AttemptResult:
    warnings = payload.get("validation_warning_rule_ids")
    return AttemptResult(
        attempt_index=int(payload["attempt_index"]),
        accepted=bool(payload["accepted"]),
        failure_kind=payload.get("failure_kind"),
        failure_message=payload.get("failure_message"),
        event_count=payload.get("event_count"),
        snapshot_count=payload.get("snapshot_count"),
        validation_log_count=payload.get("validation_log_count"),
        validation_warning_rule_ids=(
            tuple(str(rule_id) for rule_id in warnings)
            if warnings is not None
            else None
        ),
        cost_usd=payload.get("cost_usd"),
        usage=dict(payload.get("usage", {})),
    )


def _trajectory_result_to_dict(
    result: TrajectoryResult,
    *,
    prompt_hash: str | None = None,
    seed_hash: str | None = None,
    policy_hash: str | None = None,
) -> dict[str, Any]:
    cell = result.key.cell
    return {
        "trajectory_dir": result.key.dir_name,
        "cell": {
            "cell_id": cell.cell_id,
            "condition_id": cell.condition_id,
            "policy_id": cell.policy_id,
            "model_id": cell.model_id,
            "model_short": cell.model_short,
            "conditioning_label": cell.conditioning_label,
            "conditioning_hash": cell.conditioning_hash,
        },
        "future_index": result.key.future_index,
        "accepted": result.accepted,
        "attempts_to_accept": result.attempts_to_accept,
        "failure_kind": result.failure_kind,
        "attempts": [_attempt_result_to_dict(a) for a in result.attempts],
        "prompt_hash": prompt_hash,
        "seed_hash": seed_hash,
        "policy_hash": policy_hash,
    }


def _trajectory_result_from_existing(
    *,
    trajectory_key: TrajectoryKey,
    result_path: Path,
) -> TrajectoryResult:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    attempts = tuple(
        _attempt_result_from_dict(attempt) for attempt in payload.get("attempts", [])
    )
    attempts_to_accept = payload.get("attempts_to_accept")
    return TrajectoryResult(
        key=trajectory_key,
        accepted=bool(payload["accepted"]),
        attempts_to_accept=(
            int(attempts_to_accept) if attempts_to_accept is not None else None
        ),
        failure_kind=payload.get("failure_kind"),
        attempts=attempts,
    )


def run_one_attempt(
    *,
    attempt_index: int,
    attempt_dir: Path,
    rendered_prompt: str,
    baseline_seed: Mapping[str, Any],
    pinned_paths: tuple[str, ...],
    seed_is_complete: bool,
    policy: PolicySpec | None,
    model: ModelSpec,
    generate_fn: GenerateFn,
    terminal_contract: TerminalContract = DEFAULT_TERMINAL_CONTRACT,
    conditioning_spec: ConditioningSpec | None = None,
    response_contract: ResponseContract = DEFAULT_RESPONSE_CONTRACT,
) -> AttemptResult:
    """Run a single attempt: write prompt, call provider, classify, write artifacts."""
    attempt_dir.mkdir(parents=True, exist_ok=False)
    _write_text(attempt_dir / "prompt.txt", rendered_prompt)
    # Persist the generalized conditioning spec per attempt (accepted +
    # failed) so the B1b derivation layer can reconstruct it from disk,
    # recompute the four spec_* hashes, and re-run future_constraints_audit
    # to recover per-event constraint_match_ids. Written before the
    # provider call so transport failures still get spec.json on disk.
    if conditioning_spec is not None:
        _write_json(
            attempt_dir / "conditioning_spec.json",
            spec_to_json_dict(conditioning_spec),
        )

    generation: GenerationResult | None = None
    try:
        generation = generate_fn(model, _SYSTEM_PROMPT, rendered_prompt)
    except Exception as exc:  # noqa: BLE001 — transport failures must be captured
        result = AttemptResult(
            attempt_index=attempt_index,
            accepted=False,
            failure_kind="transport_or_runtime",
            failure_message=f"{type(exc).__name__}: {exc}",
            event_count=None,
            snapshot_count=None,
            validation_log_count=None,
            validation_warning_rule_ids=None,
            cost_usd=None,
            usage={},
        )
        _write_json(
            attempt_dir / "failed_attempt.json",
            {
                "failure_kind": result.failure_kind,
                "failure_message": result.failure_message,
                "raw_response": None,
                "cost_usd": None,
                "usage": {},
            },
        )
        _write_json(attempt_dir / "result.json", _attempt_result_to_dict(result))
        return result

    _write_text(attempt_dir / "raw_response.txt", generation.text)
    # OpenRouter can acknowledge `reasoning: none` yet return hidden reasoning tokens on an
    # intermittent route. The request metadata then says forced_off while the response proves
    # otherwise. Treating that answer as accepted mixes two decoding regimes inside one
    # preregistered arm, so this is a provider-parameter violation before clinical parsing.
    if generation.reasoning_mode == "forced_off" and generation.reasoning_tokens > 0:
        message = (
            "provider returned "
            f"{generation.reasoning_tokens} reasoning tokens although reasoning_mode was "
            "forced_off"
        )
        result = AttemptResult(
            attempt_index=attempt_index,
            accepted=False,
            failure_kind="provider_parameter_violation",
            failure_message=message,
            event_count=None,
            snapshot_count=None,
            validation_log_count=None,
            validation_warning_rule_ids=None,
            cost_usd=generation.reported_cost,
            usage=_usage_dict(generation),
        )
        _write_json(
            attempt_dir / "failed_attempt.json",
            {
                "failure_kind": result.failure_kind,
                "failure_message": message,
                "raw_response": generation.text,
                "cost_usd": generation.reported_cost,
                "usage": _usage_dict(generation),
            },
        )
        _write_json(attempt_dir / "result.json", _attempt_result_to_dict(result))
        return result
    failure_kind, failure_message, parsed = classify_response(
        generation.text,
        baseline_seed=baseline_seed,
        pinned_paths=pinned_paths,
        seed_is_complete=seed_is_complete,
        policy=policy,
        terminal_contract=terminal_contract,
        conditioning_spec=conditioning_spec,
        response_contract=response_contract,
    )

    if failure_kind is None and parsed is not None:
        # parsed_ledger.json mirrors the LLM's emitted ledger shape — sparse,
        # no null padding. The wide-format LedgerEvent class has ~25 optional
        # fields shared across the 12 event types, but per-event only a handful
        # are populated. exclude_none keeps the on-disk shape identical to the
        # LLM's raw_response.txt (no Pydantic-introduced null bloat).
        # materialized_trajectory.json is the downstream replay artifact and is
        # intentionally left in its verbose form — null fields there are
        # information for analysis code.
        _write_json(
            attempt_dir / "parsed_ledger.json",
            [event.model_dump(mode="json", exclude_none=True) for event in parsed.ledger_events],
        )
        _write_json(
            attempt_dir / "materialized_trajectory.json",
            parsed.materialized.model_dump(mode="json"),
        )
        result = AttemptResult(
            attempt_index=attempt_index,
            accepted=True,
            failure_kind=None,
            failure_message=None,
            event_count=len(parsed.ledger_events),
            snapshot_count=len(parsed.materialized.snapshots),
            validation_log_count=len(parsed.materialized.validation_log),
            validation_warning_rule_ids=tuple(
                v.rule_id for v in parsed.materialized.validation_log
            ),
            cost_usd=generation.reported_cost,
            usage=_usage_dict(generation),
        )
    else:
        # Post-replay failures (future_constraints_audit, terminal_contract)
        # still carry a ParsedAttempt — preserve replay-derived signals
        # (event count, snapshot count, validation log) on the failed
        # attempt's result.json so attempt_rows.csv doesn't drop warnings
        # for replay-successful-but-audit-failed attempts. Pre-replay
        # failures (parse, baseline/ledger validation, observed_prefix_audit,
        # replay) have parsed=None and keep the historic null fields.
        if parsed is not None:
            replay_event_count: int | None = len(parsed.ledger_events)
            replay_snapshot_count: int | None = len(parsed.materialized.snapshots)
            replay_validation_log_count: int | None = len(
                parsed.materialized.validation_log
            )
            replay_warning_rule_ids: tuple[str, ...] | None = tuple(
                v.rule_id for v in parsed.materialized.validation_log
            )
        else:
            replay_event_count = None
            replay_snapshot_count = None
            replay_validation_log_count = None
            replay_warning_rule_ids = None
        result = AttemptResult(
            attempt_index=attempt_index,
            accepted=False,
            failure_kind=failure_kind,
            failure_message=failure_message,
            event_count=replay_event_count,
            snapshot_count=replay_snapshot_count,
            validation_log_count=replay_validation_log_count,
            validation_warning_rule_ids=replay_warning_rule_ids,
            cost_usd=generation.reported_cost,
            usage=_usage_dict(generation),
        )
        _write_json(
            attempt_dir / "failed_attempt.json",
            {
                "failure_kind": failure_kind,
                "failure_message": failure_message,
                "raw_response": generation.text,
                "cost_usd": generation.reported_cost,
                "usage": _usage_dict(generation),
            },
        )

    _write_json(attempt_dir / "result.json", _attempt_result_to_dict(result))
    return result


def run_one_trajectory(
    *,
    trajectory_key: TrajectoryKey,
    trajectory_dir: Path,
    rendered_prompt: str,
    baseline_seed: Mapping[str, Any],
    pinned_paths: tuple[str, ...],
    seed_is_complete: bool,
    policy: PolicySpec | None,
    model: ModelSpec,
    max_attempts: int,
    generate_fn: GenerateFn,
    terminal_contract: TerminalContract = DEFAULT_TERMINAL_CONTRACT,
    prompt_hash: str | None = None,
    seed_hash: str | None = None,
    policy_hash: str | None = None,
    conditioning_spec: ConditioningSpec | None = None,
    response_contract: ResponseContract = DEFAULT_RESPONSE_CONTRACT,
) -> TrajectoryResult:
    """Run up to max_attempts for one trajectory. Writes trajectory_result.json."""
    trajectory_dir.mkdir(parents=True, exist_ok=False)
    attempts: list[AttemptResult] = []
    accepted = False
    attempts_to_accept: int | None = None
    final_failure_kind: str | None = None

    for i in range(1, max_attempts + 1):
        attempt = run_one_attempt(
            attempt_index=i,
            attempt_dir=trajectory_dir / f"attempt_{i:02d}",
            rendered_prompt=rendered_prompt,
            baseline_seed=baseline_seed,
            pinned_paths=pinned_paths,
            seed_is_complete=seed_is_complete,
            policy=policy,
            model=model,
            generate_fn=generate_fn,
            terminal_contract=terminal_contract,
            conditioning_spec=conditioning_spec,
            response_contract=response_contract,
        )
        attempts.append(attempt)
        if attempt.accepted:
            accepted = True
            attempts_to_accept = i
            break
        final_failure_kind = attempt.failure_kind

    result = TrajectoryResult(
        key=trajectory_key,
        accepted=accepted,
        attempts_to_accept=attempts_to_accept,
        failure_kind=None if accepted else final_failure_kind,
        attempts=tuple(attempts),
    )
    _write_json(
        trajectory_dir / "trajectory_result.json",
        _trajectory_result_to_dict(
            result,
            prompt_hash=prompt_hash,
            seed_hash=seed_hash,
            policy_hash=policy_hash,
        ),
    )
    return result


# ---------- top-level orchestration ----------


@dataclass(frozen=True, slots=True)
class _TrajectoryWorkItem:
    trajectory_key: TrajectoryKey
    trajectory_dir: Path
    rendered_prompt: str
    baseline_seed: Mapping[str, Any]
    pinned_paths: tuple[str, ...]
    seed_is_complete: bool
    policy: PolicySpec | None
    model: ModelSpec
    max_attempts: int
    generate_fn: GenerateFn
    terminal_contract: TerminalContract
    prompt_hash: str
    seed_hash: str
    policy_hash: str
    # Gate A3.5b: threaded through to ``classify_response`` so prefix-mode
    # audit gates fire. ``None`` for legacy cells that have not been
    # migrated to the generalized spec.
    conditioning_spec: ConditioningSpec | None = None
    #: Which response shape the prompt asked for. Threaded from the config so the parser and
    #: the prompt cannot disagree per-trajectory.
    response_contract: ResponseContract = DEFAULT_RESPONSE_CONTRACT


@dataclass(slots=True)
class _AcceptedTargetState:
    cell: CellKey
    work_items: tuple[_TrajectoryWorkItem, ...]
    accepted_target: int
    next_index: int = 0
    accepted_count: int = 0
    in_flight_count: int = 0

    @property
    def model_id(self) -> str:
        return self.cell.model_id

    def can_launch(self) -> bool:
        return (
            self.next_index < len(self.work_items)
            and self.accepted_count + self.in_flight_count < self.accepted_target
        )

    def take_next(self) -> _TrajectoryWorkItem:
        item = self.work_items[self.next_index]
        self.next_index += 1
        self.in_flight_count += 1
        return item

    def record_result(self, result: TrajectoryResult) -> None:
        self.in_flight_count -= 1
        if result.accepted:
            self.accepted_count += 1


def _trajectory_runtime_failure_result(
    item: _TrajectoryWorkItem,
    exc: BaseException,
) -> TrajectoryResult:
    """Persist a trajectory-level failure when orchestration itself raises."""
    item.trajectory_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = item.trajectory_dir / "attempt_01"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = attempt_dir / "prompt.txt"
    if not prompt_path.exists():
        _write_text(prompt_path, item.rendered_prompt)

    failure_message = f"{type(exc).__name__}: {exc}"
    attempt = AttemptResult(
        attempt_index=1,
        accepted=False,
        failure_kind="transport_or_runtime",
        failure_message=failure_message,
        event_count=None,
        snapshot_count=None,
        validation_log_count=None,
        validation_warning_rule_ids=None,
        cost_usd=None,
        usage={},
    )
    _write_json(
        attempt_dir / "failed_attempt.json",
        {
            "failure_kind": attempt.failure_kind,
            "failure_message": failure_message,
            "raw_response": None,
            "cost_usd": None,
            "usage": {},
        },
    )
    _write_json(attempt_dir / "result.json", _attempt_result_to_dict(attempt))

    result = TrajectoryResult(
        key=item.trajectory_key,
        accepted=False,
        attempts_to_accept=None,
        failure_kind="transport_or_runtime",
        attempts=(attempt,),
    )
    _write_json(
        item.trajectory_dir / "trajectory_result.json",
        _trajectory_result_to_dict(
            result,
            prompt_hash=item.prompt_hash,
            seed_hash=item.seed_hash,
            policy_hash=item.policy_hash,
        ),
    )
    return result


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _acquire_run_directory_lock(actual_root: Path, run_id: str) -> _RunDirectoryLock:
    """Prevent two live runner processes from writing the same run directory."""
    lock_path = actual_root / ".run_paper_v1.lock"
    payload = {
        "pid": os.getpid(),
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(lock_path, flags, 0o644)
    except FileExistsError as exc:
        existing: dict[str, Any] = {}
        try:
            existing = json.loads(lock_path.read_text())
        except json.JSONDecodeError:
            pass
        existing_pid = existing.get("pid")
        if isinstance(existing_pid, int) and _pid_is_alive(existing_pid):
            raise RuntimeError(
                "run lock refused because another run_paper_v1 process appears "
                f"active for run_id={run_id!r}: pid={existing_pid}. "
                "Stop that process or use a different run_id."
            ) from exc
        lock_path.unlink(missing_ok=True)
        fd = os.open(lock_path, flags, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return _RunDirectoryLock(path=lock_path, pid=payload["pid"])


def _result_cell_value(
    result: Mapping[str, Any],
    planned_key: TrajectoryKey | None,
    key: str,
) -> Any:
    cell = result.get("cell")
    if isinstance(cell, Mapping) and key in cell:
        return cell[key]
    if planned_key is None:
        return None
    planned_cell = planned_key.cell
    return {
        "cell_id": planned_cell.cell_id,
        "condition_id": planned_cell.condition_id,
        "policy_id": planned_cell.policy_id,
        "model_id": planned_cell.model_id,
        "model_short": planned_cell.model_short,
        "conditioning_label": planned_cell.conditioning_label,
        "conditioning_hash": planned_cell.conditioning_hash,
    }.get(key)


def _future_index_from_dir_name(name: str) -> int | None:
    match = re.search(r"__f(\d+)$", name)
    if not match:
        return None
    return int(match.group(1))


def _empty_inventory_row(
    *,
    run_id: str,
    trajectory_dir: str,
    planned_key: TrajectoryKey | None,
    result: Mapping[str, Any] | None,
    has_result: bool,
) -> dict[str, Any]:
    future_index = (
        result.get("future_index")
        if result is not None and result.get("future_index") is not None
        else (
            planned_key.future_index
            if planned_key is not None
            else _future_index_from_dir_name(trajectory_dir)
        )
    )
    return {
        "run_id": run_id,
        "trajectory_dir": trajectory_dir,
        "cell_id": (
            _result_cell_value(result or {}, planned_key, "cell_id")
            or (planned_key.cell.cell_id if planned_key is not None else "")
        ),
        "condition_id": _result_cell_value(result or {}, planned_key, "condition_id") or "",
        "policy_id": _result_cell_value(result or {}, planned_key, "policy_id") or "",
        "model_id": _result_cell_value(result or {}, planned_key, "model_id") or "",
        "model_short": _result_cell_value(result or {}, planned_key, "model_short") or "",
        "conditioning_label": (
            _result_cell_value(result or {}, planned_key, "conditioning_label") or ""
        ),
        "conditioning_hash": (
            _result_cell_value(result or {}, planned_key, "conditioning_hash") or ""
        ),
        "future_index": future_index if future_index is not None else "",
        "has_result": has_result,
        "accepted": bool(result.get("accepted")) if result is not None else False,
        "failure_kind": result.get("failure_kind") if result is not None else "partial_no_result",
        "attempts_to_accept": (
            result.get("attempts_to_accept") if result is not None else ""
        ),
        "attempt_count": (
            len(result.get("attempts", []))
            if result is not None and isinstance(result.get("attempts"), list)
            else ""
        ),
        "accepted_rank_in_cell": "",
        "primary_cohort": False,
        "cohort_role": "",
        "exclusion_reason": "",
        "prompt_hash": result.get("prompt_hash") if result is not None else "",
        "seed_hash": result.get("seed_hash") if result is not None else "",
        "policy_hash": result.get("policy_hash") if result is not None else "",
    }


def _inventory_cell_group(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("cell_id") or ""),
        str(row.get("condition_id") or ""),
        str(row.get("policy_id") or ""),
        str(row.get("model_short") or ""),
    )


def write_run_inventory(run_root: Path, config: POCConfig) -> dict[str, Any]:
    """Write run-level inventory files labeling primary vs nonprimary outputs."""
    trajectories_root = run_root / "trajectories"
    planned_lookup = {
        key.dir_name: key
        for _, trajectories in expand_matrix(config)
        for key in trajectories
    }
    rows: list[dict[str, Any]] = []
    if trajectories_root.exists():
        for trajectory_dir in sorted(p for p in trajectories_root.iterdir() if p.is_dir()):
            result_path = trajectory_dir / "trajectory_result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            rows.append(
                _empty_inventory_row(
                    run_id=run_root.name,
                    trajectory_dir=trajectory_dir.name,
                    planned_key=planned_lookup.get(trajectory_dir.name),
                    result=result,
                    has_result=result is not None,
                )
            )

    accepted_target = config.accepted_futures_per_cell or config.futures_per_cell
    accepted_by_cell: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["accepted"]:
            accepted_by_cell.setdefault(_inventory_cell_group(row), []).append(row)

    for accepted_rows in accepted_by_cell.values():
        accepted_rows.sort(
            key=lambda row: (
                int(row["future_index"]) if row["future_index"] != "" else 10**9,
                str(row["trajectory_dir"]),
            )
        )
        for rank, row in enumerate(accepted_rows, start=1):
            row["accepted_rank_in_cell"] = rank

    for row in rows:
        failure_kind = str(row["failure_kind"] or "")
        if not row["has_result"]:
            row["cohort_role"] = "partial"
            row["exclusion_reason"] = "partial_no_result"
        elif failure_kind == "dry_run":
            row["cohort_role"] = "dry_run"
            row["exclusion_reason"] = "dry_run"
        elif row["accepted"]:
            rank = int(row["accepted_rank_in_cell"])
            if rank <= accepted_target:
                row["primary_cohort"] = True
                row["cohort_role"] = "primary"
            else:
                row["cohort_role"] = "overshoot_nonprimary"
                row["exclusion_reason"] = "accepted_beyond_prespecified_target"
        else:
            row["cohort_role"] = "failed"
            row["exclusion_reason"] = failure_kind or "failed_unknown"

    fieldnames = [
        "run_id",
        "trajectory_dir",
        "cell_id",
        "condition_id",
        "policy_id",
        "model_id",
        "model_short",
        "conditioning_label",
        "conditioning_hash",
        "future_index",
        "has_result",
        "accepted",
        "failure_kind",
        "attempts_to_accept",
        "attempt_count",
        "accepted_rank_in_cell",
        "primary_cohort",
        "cohort_role",
        "exclusion_reason",
        "prompt_hash",
        "seed_hash",
        "policy_hash",
    ]
    inventory_path = run_root / "run_inventory.csv"
    with inventory_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_model: dict[str, dict[str, int]] = {}
    for row in rows:
        model = str(row["model_short"] or "unknown")
        model_counts = by_model.setdefault(
            model,
            {
                "total": 0,
                "accepted": 0,
                "primary": 0,
                "overshoot_nonprimary": 0,
                "failed": 0,
                "partial": 0,
                "dry_run": 0,
            },
        )
        model_counts["total"] += 1
        if row["accepted"]:
            model_counts["accepted"] += 1
        role = str(row["cohort_role"])
        if role in model_counts:
            model_counts[role] += 1

    summary = {
        "run_id": run_root.name,
        "config_id": config.config_id,
        "inventory_path": str(inventory_path),
        "accepted_target_per_cell": accepted_target,
        "sampling_mode": (
            "accepted_target"
            if config.accepted_futures_per_cell is not None
            else "fixed_futures"
        ),
        "rows_total": len(rows),
        "result_files": sum(1 for row in rows if row["has_result"]),
        "partial_dirs": sum(1 for row in rows if row["cohort_role"] == "partial"),
        "accepted_total": sum(1 for row in rows if row["accepted"]),
        "primary_accepted_total": sum(1 for row in rows if row["primary_cohort"]),
        "accepted_nonprimary_total": sum(
            1 for row in rows if row["cohort_role"] == "overshoot_nonprimary"
        ),
        "failed_completed_total": sum(1 for row in rows if row["cohort_role"] == "failed"),
        "dry_run_total": sum(1 for row in rows if row["cohort_role"] == "dry_run"),
        "by_model": by_model,
    }
    _write_json(run_root / "run_inventory_summary.json", summary)
    return summary


def _run_trajectory_work_item(item: _TrajectoryWorkItem) -> TrajectoryResult:
    try:
        return run_one_trajectory(
            trajectory_key=item.trajectory_key,
            trajectory_dir=item.trajectory_dir,
            rendered_prompt=item.rendered_prompt,
            baseline_seed=item.baseline_seed,
            pinned_paths=item.pinned_paths,
            seed_is_complete=item.seed_is_complete,
            policy=item.policy,
            model=item.model,
            max_attempts=item.max_attempts,
            generate_fn=item.generate_fn,
            terminal_contract=item.terminal_contract,
            prompt_hash=item.prompt_hash,
            seed_hash=item.seed_hash,
            policy_hash=item.policy_hash,
            conditioning_spec=item.conditioning_spec,
            response_contract=item.response_contract,
        )
    except Exception as exc:  # noqa: BLE001 — isolate one trajectory failure
        return _trajectory_runtime_failure_result(item, exc)


def _run_live_trajectories(
    work_items: Sequence[_TrajectoryWorkItem],
    *,
    max_parallel_trajectories: int,
    max_parallel_trajectories_per_model: int | None = None,
) -> tuple[TrajectoryResult, ...]:
    if max_parallel_trajectories_per_model is not None:
        return _run_live_trajectories_per_model(
            work_items,
            max_parallel_trajectories_per_model=max_parallel_trajectories_per_model,
        )
    if max_parallel_trajectories == 1:
        return tuple(_run_trajectory_work_item(item) for item in work_items)

    results: list[TrajectoryResult] = []
    with ThreadPoolExecutor(max_workers=max_parallel_trajectories) as executor:
        futures = [executor.submit(_run_trajectory_work_item, item) for item in work_items]
        for future in as_completed(futures):
            results.append(future.result())
    return tuple(results)


def _run_live_trajectories_per_model(
    work_items: Sequence[_TrajectoryWorkItem],
    *,
    max_parallel_trajectories_per_model: int,
) -> tuple[TrajectoryResult, ...]:
    """Run fixed-future work with an independent concurrency lane per model."""
    if not work_items:
        return ()
    model_queues: dict[str, deque[_TrajectoryWorkItem]] = {}
    for item in work_items:
        model_queues.setdefault(item.model.id, deque()).append(item)
    model_ids = sorted(model_queues)
    max_workers = max_parallel_trajectories_per_model * len(model_ids)
    active_by_model = {model_id: 0 for model_id in model_ids}
    results: list[TrajectoryResult] = []
    pending: dict[Future[TrajectoryResult], str] = {}
    cursor = 0

    def launch_ready(executor: ThreadPoolExecutor) -> None:
        nonlocal cursor
        while len(pending) < max_workers:
            launched = False
            for _ in range(len(model_ids)):
                model_id = model_ids[cursor % len(model_ids)]
                cursor += 1
                queue = model_queues[model_id]
                if active_by_model[model_id] >= max_parallel_trajectories_per_model:
                    continue
                if not queue:
                    continue
                item = queue.popleft()
                active_by_model[model_id] += 1
                pending[executor.submit(_run_trajectory_work_item, item)] = model_id
                launched = True
                break
            if not launched:
                break

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        launch_ready(executor)
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                model_id = pending.pop(future)
                active_by_model[model_id] -= 1
                results.append(future.result())
            launch_ready(executor)
    return tuple(results)


def _resolve_repo_path(p: Path) -> Path:
    return p if p.is_absolute() else REPO_ROOT / p


def _policy_for(config: POCConfig, policy_id: str) -> PolicySpec:
    for p in config.policies:
        if p.id == policy_id:
            return p
    raise KeyError(f"policy_id not in config: {policy_id!r}")


def _cell_spec_for(config: POCConfig, cell_id: str) -> ConditioningCellSpec:
    for cell in config.selected_cells():
        if cell.id == cell_id:
            return cell
    raise KeyError(f"cell_id not in config: {cell_id!r}")


def _model_for(config: POCConfig, model_id: str) -> ModelSpec:
    for m in config.models:
        if m.id == model_id:
            return m
    raise KeyError(f"model_id not in config: {model_id!r}")


@dataclass(frozen=True, slots=True)
class _PreflightedInputs:
    template_path: Path
    template_text: str


def _preflight(config: POCConfig) -> _PreflightedInputs:
    """Resolve and validate template inputs BEFORE the run directory exists.

    Condition seeds are already validated at ``load_config`` via
    ``validate_baseline_seed``; this preflight only resolves the prompt
    template path (so a typo there fails before any directory is created).
    """
    template_path = _resolve_repo_path(config.prompt_template_path)
    template_text = template_path.read_text(encoding="utf-8")
    return _PreflightedInputs(
        template_path=template_path,
        template_text=template_text,
    )


def default_run_id() -> str:
    return datetime.now(UTC).strftime("synthonccases_%Y_%m_%d_%H%M%S")


def _git_commit_or_none() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip() or None


def _replay_module_source() -> str:
    from synthonccases.oncollm import replay as _replay_module
    return Path(_replay_module.__file__).read_text(encoding="utf-8")


def _serialize_prompt_enum_order(enum_order: PromptEnumOrder) -> dict[str, str]:
    return {"mode": enum_order.mode, "seed": enum_order.seed}


def _serialize_conditioning_for_echo(spec: ConditioningSpec) -> dict[str, Any]:
    """Return the generalized YAML shape accepted by ``load_config``.

    ``conditioning_spec.json`` uses the dataclass field names, while the public YAML uses
    ``fixed_events`` and ``future_event_constraints``. Keeping this conversion explicit makes
    the run's resolved config executable instead of a lossy human summary.
    """
    canonical = spec_to_json_dict(spec)
    payload: dict[str, Any] = {
        "free_text_operator_constraints": spec.free_text_operator_constraints,
    }
    observed = canonical.get("observed_prefix")
    if isinstance(observed, Mapping):
        payload["fixed_events"] = {
            "origin": observed.get("origin", "observed_prefix"),
            "cutoff_date": observed["cutoff_date"],
            "events": observed.get("events", []),
        }
    constraints = canonical.get("future_event_constraints")
    if constraints:
        payload["future_event_constraints"] = constraints
    return payload


def _serialize_config_for_echo(
    config: POCConfig,
    *,
    prompt_template_path: Path | None = None,
) -> dict[str, Any]:
    prompt_payload: dict[str, Any] = {
        "template_path": str(prompt_template_path or config.prompt_template_path),
        "response_contract": config.response_contract,
    }
    if config.prompt_enum_order != PromptEnumOrder():
        prompt_payload["enum_order"] = _serialize_prompt_enum_order(
            config.prompt_enum_order
        )

    payload: dict[str, Any] = {
        "schema_version": config.schema_version,
        "config_id": config.config_id,
        "label_prefix": config.label_prefix,
        "study_reference": {
            "diagnosis_date": config.study_reference.diagnosis_date,
            "knowledge_cutoff": config.study_reference.knowledge_cutoff,
        },
        "run_defaults": {
            "cells_count": config.conditions_count,
            "futures_per_cell": config.futures_per_cell,
            "max_attempts": config.max_attempts,
            "terminal_contract": config.terminal_contract,
        },
        "prompt": prompt_payload,
        "provider": {
            "use_openrouter": config.use_openrouter,
            "response_format": dict(config.response_format),
            "require_parameters": config.require_parameters,
        },
        "models": [{"id": m.id, "short": m.short} for m in config.models],
    }
    if config.accepted_futures_per_cell is not None:
        payload["run_defaults"]["accepted_futures_per_cell"] = (
            config.accepted_futures_per_cell
        )
        payload["run_defaults"]["max_total_futures_per_cell"] = (
            config.max_total_futures_per_cell
        )

    if config.cells:
        payload["cells"] = [_serialize_cell_for_echo(c) for c in config.selected_cells()]
    else:
        payload["conditions"] = [
            {
                "id": c.id,
                "inline_seed": dict(c.baseline_seed),
                "label_override": c.label_override,
            }
            for c in config.conditions
        ]
        payload["policies"] = [
            {
                "id": p.id,
                "text": p.text,
                "regimen_sequence": [
                    {"line": row.line, "aliases": list(row.aliases)}
                    for row in p.regimen_sequence
                ],
                "unconstrained_after_line": p.unconstrained_after_line,
                "fixed_first_line_treatment": (
                    None
                    if p.fixed_first_line_treatment is None
                    else {
                        "regimen_label": p.fixed_first_line_treatment.regimen_label,
                        "drugs": list(p.fixed_first_line_treatment.drugs),
                        "schedule": p.fixed_first_line_treatment.schedule,
                    }
                ),
            }
            for p in config.policies
        ]
    return payload


def _serialize_cell_for_echo(cell: ConditioningCellSpec) -> dict[str, Any]:
    policy = cell.policy
    out: dict[str, Any] = {
        "id": cell.id,
        "condition_id": cell.condition.id,
        "policy_id": policy.id,
        "baseline_seed": dict(cell.condition.baseline_seed),
    }
    if cell.condition.label_override is not None:
        out["label_override"] = cell.condition.label_override
    if policy.text not in _NO_ADDITIONAL_POLICY_TEXTS:
        out["operator_constraints"] = policy.text
    if policy.regimen_sequence:
        out["regimen_sequence"] = [
            {"line": row.line, "aliases": list(row.aliases)}
            for row in policy.regimen_sequence
        ]
    if policy.unconstrained_after_line is not None:
        out["unconstrained_after_line"] = policy.unconstrained_after_line
    if policy.fixed_first_line_treatment is not None:
        line_1_aliases = next(
            (
                tuple(row.aliases)
                for row in policy.regimen_sequence
                if row.line == 1
            ),
            (),
        )
        out["first_treatment"] = {
            "regimen_label": policy.fixed_first_line_treatment.regimen_label,
            "aliases": list(line_1_aliases),
            "drugs": list(policy.fixed_first_line_treatment.drugs),
            "schedule": policy.fixed_first_line_treatment.schedule,
        }
    if cell.conditioning_spec is not None:
        out["conditioning"] = _serialize_conditioning_for_echo(
            cell.conditioning_spec
        )
    return out


def _config_hash(config: POCConfig) -> str:
    """sha256 of the canonical serialized config echo.

    Captures every field that affects experimental design (including the
    new conditioning_label) without depending on the user's on-disk YAML
    formatting. Stable across tests that build POCConfig directly.
    """
    blob = yaml.safe_dump(
        _serialize_config_for_echo(config), sort_keys=True
    ).encode("utf-8")
    return sha256_hex(blob)


def _build_manifest(
    *,
    report: POCRunReport,
    config: POCConfig,
    run_start_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_template_hash = str(run_start_provenance["prompt_template_sha256"])
    replay_module_hash = str(run_start_provenance["replay_module_sha256"])
    git_commit_at_start = run_start_provenance.get("git_commit_at_start")
    git_commit_at_completion = _git_commit_or_none()
    return {
        "schema_version": 1,
        "run_id": report.run_id,
        "started_at": report.started_at,
        "completed_at": report.completed_at,
        "dry_run": report.dry_run,
        "resumed": report.resumed,
        "config_id": config.config_id,
        "config_schema_version": config.schema_version,
        "config_hash": run_start_provenance["config_hash"],
        "source_config_path": run_start_provenance.get("source_config_path"),
        "source_config_recorded_path": (
            "config.source.yaml"
            if run_start_provenance.get("source_config_sha256") is not None
            else None
        ),
        "source_config_sha256": run_start_provenance.get("source_config_sha256"),
        "preregistration": run_start_provenance.get("preregistration"),
        "resolved_config_recorded_path": "config.yaml",
        "resolved_config_sha256": run_start_provenance["resolved_config_sha256"],
        "label_prefix": config.label_prefix,
        "study_reference": {
            "diagnosis_date": config.study_reference.diagnosis_date,
            "knowledge_cutoff": config.study_reference.knowledge_cutoff,
        },
        "prompt_template_path": str(config.prompt_template_path),
        "prompt_enum_order": _serialize_prompt_enum_order(config.prompt_enum_order),
        "prompt_template_hash": prompt_template_hash,
        "prompt_template_sha256": prompt_template_hash,
        "replay_module_hash": replay_module_hash,
        "replay_module_sha256": replay_module_hash,
        "git_commit": git_commit_at_start,
        "git_commit_at_start": git_commit_at_start,
        "git_commit_at_completion": git_commit_at_completion,
        "git_commit_changed_during_run": (
            git_commit_at_start != git_commit_at_completion
        ),
        "cells_planned": report.cells_planned,
        "trajectories_planned": report.trajectories_planned,
        "trajectories_accepted": report.trajectories_accepted,
        "trajectories_failed": report.trajectories_failed,
        "trajectories_reused": report.trajectories_reused,
        "max_parallel_trajectories": report.max_parallel_trajectories,
        "max_parallel_trajectories_per_model": (
            report.max_parallel_trajectories_per_model
        ),
        "generation_parameter_policy": {
            "temperature_when_sent": DEFAULT_REQUEST_TEMPERATURE,
            "top_p_when_sent": DEFAULT_REQUEST_TOP_P,
            "seed_when_sent": DEFAULT_REQUEST_SEED,
            "provider_internal_max_attempts": PROVIDER_INTERNAL_MAX_ATTEMPTS,
            "provider_call_watchdog_timeout_seconds": (
                _provider_call_timeout_seconds()
            ),
            "trajectory_max_attempts": config.max_attempts,
            "sampling_mode": (
                "accepted_target"
                if config.accepted_futures_per_cell is not None
                else "fixed_futures"
            ),
            "accepted_futures_per_cell": config.accepted_futures_per_cell,
            "max_total_futures_per_cell": config.max_total_futures_per_cell,
            "response_format": dict(config.response_format),
            "require_parameters": config.require_parameters,
            "actual_per_attempt_fields": [
                "max_tokens_sent",
                "max_tokens_param",
                "reasoning_mode",
                "temperature_sent",
                "top_p_sent",
                "seed_sent",
                "finish_reason",
                "response_model",
                "response_id",
                "reasoning_tokens",
                "retries",
                "duration_s",
            ],
            "notes": (
                "Per-attempt usage records the exact parameters sent to the "
                "provider. Null means the parameter was intentionally not sent "
                "for that provider/model family, e.g. native OpenAI reasoning "
                "models reject temperature."
            ),
        },
        "models": [{"id": m.id, "short": m.short} for m in config.models],
        "policies": [
            {
                "id": p.id,
                "regimen_sequence_len": len(p.regimen_sequence),
                "fixed_first_line_treatment": (
                    None
                    if p.fixed_first_line_treatment is None
                    else {
                        "regimen_label": p.fixed_first_line_treatment.regimen_label,
                        "drugs": list(p.fixed_first_line_treatment.drugs),
                        "schedule": p.fixed_first_line_treatment.schedule,
                    }
                ),
            }
            for p in config.policies
        ],
        "conditioning_cells_used": [
            {
                "id": c.id,
                "condition_id": c.condition.id,
                "policy_id": c.policy.id,
                "seed_hash": c.condition.seed_hash,
                "policy_hash": sha256_hex(
                    canonical_json(_canonical_policy_payload(c.policy))
                ),
                "conditioning_hash": _cell_hash(
                    c.condition, c.policy, c.conditioning_spec
                ),
                "conditioning_spec_hash": (
                    compute_conditioning_spec_hash(c.conditioning_spec)
                    if c.conditioning_spec is not None
                    else None
                ),
                "pinned_paths": list(c.condition.pinned_paths),
                "study_reference_label_ignored_paths": list(
                    _study_reference_label_ignored_paths(config.study_reference)
                ),
                "label_override": c.condition.label_override,
                "fixed_first_line_treatment": (
                    None
                    if c.policy.fixed_first_line_treatment is None
                    else {
                        "regimen_label": c.policy.fixed_first_line_treatment.regimen_label,
                        "drugs": list(c.policy.fixed_first_line_treatment.drugs),
                        "schedule": c.policy.fixed_first_line_treatment.schedule,
                    }
                ),
            }
            for c in config.selected_cells()
        ],
        "conditions_used": [
            {
                "id": c.id,
                "seed_hash": c.seed_hash,
                "pinned_paths": list(c.pinned_paths),
                "study_reference_label_ignored_paths": list(
                    _study_reference_label_ignored_paths(config.study_reference)
                ),
                "label_override": c.label_override,
            }
            for c in config.selected_conditions()
        ],
        "provider": {
            "use_openrouter": config.use_openrouter,
            "response_format": dict(config.response_format),
            "require_parameters": config.require_parameters,
        },
        "matrix": {
            "cells_count": config.conditions_count,
            "conditions_count": config.conditions_count,
            "futures_per_cell": config.futures_per_cell,
            "accepted_futures_per_cell": config.accepted_futures_per_cell,
            "max_total_futures_per_cell": config.max_total_futures_per_cell,
            "max_attempts": config.max_attempts,
            "terminal_contract": config.terminal_contract,
            "response_contract": config.response_contract,
        },
    }


def _assert_resume_manifest_compatible(
    *,
    run_root: Path,
    config: POCConfig,
    prompt_template_text: str,
) -> None:
    manifest_path = run_root / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dry_run") is True:
        raise ValueError("resume refused because existing manifest is from a dry-run")
    expected = {
        "config_hash": _config_hash(config),
        "prompt_template_sha256": sha256_hex(prompt_template_text),
        "replay_module_sha256": sha256_hex(_replay_module_source()),
    }
    mismatches = [
        key
        for key, expected_value in expected.items()
        if manifest.get(key) != expected_value
    ]
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(
            "resume refused because existing manifest does not match the "
            f"current generation contract: {joined}"
        )


def _run_live_accepted_target_matrix(
    cell_work_items: Sequence[tuple[CellKey, Sequence[_TrajectoryWorkItem]]],
    *,
    accepted_target_per_cell: int,
    max_parallel_trajectories: int,
    max_parallel_trajectories_per_model: int | None = None,
    existing_results_by_cell: Mapping[CellKey, Sequence[TrajectoryResult]] | None = None,
) -> tuple[TrajectoryResult, ...]:
    """Run accepted-target sampling across all model-cells.

    The scheduler keeps every model-cell eligible for top-up until it reaches
    the accepted target or exhausts its cap. It never launches more in-flight
    work for a cell than could still be needed, so a perfectly successful
    batch cannot overshoot the accepted target.
    """
    if max_parallel_trajectories_per_model is not None:
        model_ids = sorted({cell.model_id for cell, work_items in cell_work_items if work_items})
        max_workers = max_parallel_trajectories_per_model * len(model_ids)
        model_limit = max_parallel_trajectories_per_model
    else:
        max_workers = max_parallel_trajectories
        model_limit = max_parallel_trajectories

    if max_workers <= 1:
        return _run_live_accepted_target_matrix_sequential(
            cell_work_items,
            accepted_target_per_cell=accepted_target_per_cell,
            existing_results_by_cell=existing_results_by_cell,
        )

    states = [
        _AcceptedTargetState(
            cell=cell,
            work_items=tuple(work_items),
            accepted_target=accepted_target_per_cell,
            accepted_count=sum(
                1
                for result in (existing_results_by_cell or {}).get(cell, ())
                if result.accepted
            ),
        )
        for cell, work_items in cell_work_items
        if work_items
    ]
    if not states:
        return ()

    active_by_model = {model_id: 0 for model_id in {state.model_id for state in states}}
    results: list[TrajectoryResult] = []
    pending: dict[Future[TrajectoryResult], _AcceptedTargetState] = {}
    cursor = 0

    def launch_ready(executor: ThreadPoolExecutor) -> None:
        nonlocal cursor
        while len(pending) < max_workers:
            launched = False
            for _ in range(len(states)):
                state = states[cursor % len(states)]
                cursor += 1
                if active_by_model[state.model_id] >= model_limit:
                    continue
                if not state.can_launch():
                    continue
                item = state.take_next()
                active_by_model[state.model_id] += 1
                pending[executor.submit(_run_trajectory_work_item, item)] = state
                launched = True
                break
            if not launched:
                break

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        launch_ready(executor)
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                state = pending.pop(future)
                active_by_model[state.model_id] -= 1
                result = future.result()
                state.record_result(result)
                results.append(result)
            launch_ready(executor)
    return tuple(results)


def _run_live_accepted_target_matrix_sequential(
    cell_work_items: Sequence[tuple[CellKey, Sequence[_TrajectoryWorkItem]]],
    *,
    accepted_target_per_cell: int,
    existing_results_by_cell: Mapping[CellKey, Sequence[TrajectoryResult]] | None = None,
) -> tuple[TrajectoryResult, ...]:
    results: list[TrajectoryResult] = []
    for cell, work_items in cell_work_items:
        accepted_for_cell = sum(
            1
            for result in (existing_results_by_cell or {}).get(cell, ())
            if result.accepted
        )
        next_index = 0
        while next_index < len(work_items) and accepted_for_cell < accepted_target_per_cell:
            remaining_target = accepted_target_per_cell - accepted_for_cell
            chunk_size = min(
                1,
                remaining_target,
                len(work_items) - next_index,
            )
            chunk = work_items[next_index : next_index + chunk_size]
            chunk_results = _run_live_trajectories(
                chunk,
                max_parallel_trajectories=chunk_size,
            )
            results.extend(chunk_results)
            accepted_for_cell += sum(1 for result in chunk_results if result.accepted)
            next_index += chunk_size
    return tuple(results)


def run_poc(
    *,
    config: POCConfig,
    run_root: Path,
    run_id: str | None = None,
    dry_run: bool = False,
    generate_fn: GenerateFn | None = None,
    max_parallel_trajectories: int = 1,
    max_parallel_trajectories_per_model: int | None = None,
    resume: bool = False,
    source_config_path: Path | None = None,
) -> POCRunReport:
    """Top-level POC orchestration. Creates run_root/<run_id>/ and returns the report.

    Preflight (matrix expansion + path resolution + baseline JSON + schema
    validation via BaselineObject) happens BEFORE any directory is created,
    so a misconfigured matrix or a typo in baselines[].path never leaves
    orphan run directories.
    """
    if max_parallel_trajectories < 1:
        raise ValueError(
            "max_parallel_trajectories must be >= 1; "
            f"got {max_parallel_trajectories}"
        )
    if (
        max_parallel_trajectories_per_model is not None
        and max_parallel_trajectories_per_model < 1
    ):
        raise ValueError(
            "max_parallel_trajectories_per_model must be >= 1 when set; "
            f"got {max_parallel_trajectories_per_model}"
        )
    if resume and dry_run:
        raise ValueError("resume is only supported for live runs, not dry-runs")
    if run_id is None:
        run_id = default_run_id()

    # Capture executable provenance before any provider call. In particular, do not ask Git for
    # HEAD only after a long run: another process may commit while generation is in flight.
    git_commit_at_start = _git_commit_or_none()
    config_hash_at_start = _config_hash(config)
    replay_module_hash_at_start = sha256_hex(_replay_module_source())
    source_config_bytes = (
        source_config_path.read_bytes() if source_config_path is not None else None
    )
    source_config_payload = (
        yaml.safe_load(source_config_bytes) if source_config_bytes is not None else {}
    )
    source_preregistration = (
        dict(source_config_payload.get("preregistration", {}))
        if isinstance(source_config_payload, Mapping)
        else {}
    )

    matrix = expand_matrix(config)
    n_cells = len(matrix)
    n_trajectories = total_trajectories(matrix)

    preflighted = _preflight(config)
    template_text = preflighted.template_text
    prompt_template_hash_at_start = sha256_hex(template_text)

    actual_root = run_root / run_id
    if actual_root.exists():
        if not resume:
            raise FileExistsError(actual_root)
        if not actual_root.is_dir():
            raise ValueError(f"run root exists but is not a directory: {actual_root}")
        _assert_resume_manifest_compatible(
            run_root=actual_root,
            config=config,
            prompt_template_text=template_text,
        )
    else:
        actual_root.mkdir(parents=True, exist_ok=False)

    run_lock: _RunDirectoryLock | None = None
    if not dry_run:
        run_lock = _acquire_run_directory_lock(actual_root, run_id)
        atexit.register(run_lock.release)

    if generate_fn is None and not dry_run:
        generate_fn = default_generate_fn(config)

    prompt_snapshot_path = actual_root / "prompt_template.md"
    _write_or_verify_bytes(
        prompt_snapshot_path,
        template_text.encode("utf-8"),
        resume=resume,
    )
    if source_config_bytes is not None:
        _write_or_verify_bytes(
            actual_root / "config.source.yaml",
            source_config_bytes,
            resume=resume,
        )
    resolved_config_text = yaml.safe_dump(
        _serialize_config_for_echo(
            config,
            prompt_template_path=prompt_snapshot_path.resolve(),
        ),
        sort_keys=False,
    )
    _write_or_verify_bytes(
        actual_root / "config.yaml",
        resolved_config_text.encode("utf-8"),
        resume=resume,
    )
    run_start_provenance = {
        "git_commit_at_start": git_commit_at_start,
        "config_hash": config_hash_at_start,
        "source_config_path": (
            str(source_config_path.resolve()) if source_config_path is not None else None
        ),
        "source_config_sha256": (
            sha256_hex(source_config_bytes) if source_config_bytes is not None else None
        ),
        "preregistration": source_preregistration or None,
        "resolved_config_sha256": sha256_hex(resolved_config_text),
        "prompt_template_sha256": prompt_template_hash_at_start,
        "replay_module_sha256": replay_module_hash_at_start,
    }

    started_at = datetime.now(UTC).isoformat()
    trajectories_root = actual_root / "trajectories"
    trajectories_root.mkdir(exist_ok=resume)

    accepted_count = 0
    failed_count = 0
    reused_count = 0
    existing_results: list[TrajectoryResult] = []
    existing_results_by_cell: dict[CellKey, list[TrajectoryResult]] = {}
    live_work_items: list[_TrajectoryWorkItem] = []
    live_cell_work_items: list[tuple[CellKey, list[_TrajectoryWorkItem]]] = []

    for cell, trajectories in matrix:
        conditioning_cell = _cell_spec_for(config, cell.cell_id)
        condition = conditioning_cell.condition
        seed = condition.baseline_seed
        pinned_paths = condition.pinned_paths
        required_patch_leaves = compute_required_patch_leaves(seed)

        policy = conditioning_cell.policy
        policy_hash = sha256_hex(canonical_json(_canonical_policy_payload(policy)))

        prompt = render_prompt(
            template_text,
            baseline_seed=seed,
            pinned_paths=pinned_paths,
            required_patch_leaves=required_patch_leaves,
            policy_text=policy.text,
            terminal_contract=config.terminal_contract,
            conditioning_spec=conditioning_cell.conditioning_spec,
            knowledge_cutoff=config.study_reference.knowledge_cutoff,
            prompt_enum_order=config.prompt_enum_order,
        )
        prompt_hash = sha256_hex(prompt)
        model = _model_for(config, cell.model_id)
        cell_work_items: list[_TrajectoryWorkItem] = []

        for traj_key in trajectories:
            trajectory_dir = trajectories_root / traj_key.dir_name
            existing_result_path = trajectory_dir / "trajectory_result.json"

            if dry_run:
                trajectory_dir.mkdir(parents=True, exist_ok=False)
                _write_text(trajectory_dir / "prompt.txt", prompt)
                _write_json(
                    trajectory_dir / "trajectory_result.json",
                    {
                        "trajectory_dir": traj_key.dir_name,
                        "cell": {
                            "cell_id": cell.cell_id,
                            "condition_id": cell.condition_id,
                            "policy_id": cell.policy_id,
                            "model_id": cell.model_id,
                            "model_short": cell.model_short,
                            "conditioning_label": cell.conditioning_label,
                            "conditioning_hash": cell.conditioning_hash,
                        },
                        "future_index": traj_key.future_index,
                        "accepted": False,
                        "attempts_to_accept": None,
                        "failure_kind": "dry_run",
                        "attempts": [],
                        "prompt_hash": prompt_hash,
                        "seed_hash": condition.seed_hash,
                        "policy_hash": policy_hash,
                    },
                )
                continue

            if resume and existing_result_path.exists():
                existing_result = _trajectory_result_from_existing(
                    trajectory_key=traj_key,
                    result_path=existing_result_path,
                )
                existing_results.append(existing_result)
                existing_results_by_cell.setdefault(cell, []).append(existing_result)
                continue

            if resume and trajectory_dir.exists():
                # A previous process may have died mid-trajectory. Leave its
                # partial artifacts untouched and rely on unused future slots
                # for accepted-target top-up. If the cap is exhausted, the run
                # report will expose the shortfall.
                continue

            assert generate_fn is not None  # narrow for type-checkers
            cell_work_items.append(
                _TrajectoryWorkItem(
                    trajectory_key=traj_key,
                    trajectory_dir=trajectory_dir,
                    rendered_prompt=prompt,
                    baseline_seed=seed,
                    pinned_paths=pinned_paths,
                    seed_is_complete=condition.is_full_baseline,
                    policy=policy,
                    model=model,
                    max_attempts=config.max_attempts,
                    generate_fn=generate_fn,
                    terminal_contract=config.terminal_contract,
                    prompt_hash=prompt_hash,
                    seed_hash=condition.seed_hash,
                    policy_hash=policy_hash,
                    conditioning_spec=conditioning_cell.conditioning_spec,
                    response_contract=config.response_contract,
                )
            )
        if cell_work_items:
            live_cell_work_items.append((cell, cell_work_items))
            live_work_items.extend(cell_work_items)

    results: tuple[TrajectoryResult, ...] = ()
    if live_work_items and config.accepted_futures_per_cell is not None:
        new_results = _run_live_accepted_target_matrix(
            live_cell_work_items,
            accepted_target_per_cell=config.accepted_futures_per_cell,
            max_parallel_trajectories=max_parallel_trajectories,
            max_parallel_trajectories_per_model=max_parallel_trajectories_per_model,
            existing_results_by_cell=existing_results_by_cell,
        )
        results = tuple(existing_results) + new_results
        accepted_count = sum(1 for result in results if result.accepted)
        failed_count = len(results) - accepted_count
        n_trajectories = len(results)
    elif live_work_items:
        new_results = _run_live_trajectories(
            live_work_items,
            max_parallel_trajectories=max_parallel_trajectories,
            max_parallel_trajectories_per_model=max_parallel_trajectories_per_model,
        )
        results = tuple(existing_results) + new_results
        accepted_count = sum(1 for result in results if result.accepted)
        failed_count = len(results) - accepted_count
    elif existing_results:
        results = tuple(existing_results)
        accepted_count = sum(1 for result in results if result.accepted)
        failed_count = len(results) - accepted_count
        if config.accepted_futures_per_cell is not None:
            n_trajectories = len(results)

    reused_count = len(existing_results)

    completed_at = datetime.now(UTC).isoformat()
    report = POCRunReport(
        run_id=run_id,
        run_root=actual_root,
        cells_planned=n_cells,
        trajectories_planned=n_trajectories,
        trajectories_accepted=accepted_count,
        trajectories_failed=failed_count,
        trajectories_reused=reused_count,
        started_at=started_at,
        completed_at=completed_at,
        dry_run=dry_run,
        resumed=resume,
        max_parallel_trajectories=max_parallel_trajectories,
        max_parallel_trajectories_per_model=max_parallel_trajectories_per_model,
    )

    _write_json(
        actual_root / "manifest.json",
        _build_manifest(
            report=report,
            config=config,
            run_start_provenance=run_start_provenance,
        ),
    )
    write_run_inventory(actual_root, config)

    if run_lock is not None:
        run_lock.release()

    return report
