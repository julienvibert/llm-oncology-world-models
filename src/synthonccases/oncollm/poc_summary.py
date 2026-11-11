"""Rev5 POC summary builder — read-only readout of a finished POC run.

Reads `run_root/trajectories/**` and writes `run_root/summary/`. Pure read with
respect to everything outside `summary/`: it never re-invokes the provider,
never edits trajectory artifacts, and never touches the run-root manifest.

Public entry points:
  build_summary(run_root)            — top-level orchestration
  derive_treatment_courses(ledger)   — pure: courses from materialized ledger
  check_policy_adherence(courses, policy)
                                     — pure: line + regimen + unconstrained_after_line
  normalize_regimen_string(text)     — pure: case + separator-insensitive
  soft_no_progression_before_death_from_progression(materialized)
                                     — pure: cause mentions progression but no event

Determinism contract: `summary.json` is byte-identical on re-run for an
unchanged run root. Per-trajectory figures and the markdown readout are
NOT byte-deterministic (matplotlib metadata varies).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend; safe under pytest and headless.
import matplotlib.pyplot as plt  # noqa: E402

from synthonccases.oncollm.poc_runner import PolicySpec, load_config  # noqa: E402

_SEPARATOR_RE = re.compile(r"[\s\-/+_,]")
# Token-split treats hyphen and comma as regular separators alongside slash,
# plus, underscore, and whitespace. The substring-match path still catches
# acronym variants (T-DM1 ↔ TDM1) via separator-stripping in
# normalize_regimen_string; the token-bag path catches multi-drug regimens
# written with any of these separators in any order.
_TOKEN_SPLIT_RE = re.compile(r"[\s\-/+_,]+")
_PARENTHETICAL_RE = re.compile(r"\([^)]*\)")
_PROGRESSION_CAUSE_RE = re.compile(r"progress(?:ion|ive)", re.IGNORECASE)

RICH_EVENT_TYPES: tuple[str, ...] = (
    "toxicity",
    "clonal_evolution_event",
    "patient_state_update",
    "local_intervention",
)

# Brand → generic mapping for HER2+ BRCA POC. Tiny + explicit by design:
# this is NOT a general drug vocabulary module; it is a small fix for the
# 3 alias-map false positives observed in runs/rev5_poc_2026_05_20_pm_live/.
# Applied case-insensitively as a pre-normalization step on observed
# regimens only — aliases in policy config are not rewritten.
BRAND_TO_GENERIC: tuple[tuple[str, str], ...] = (
    ("Herceptin", "trastuzumab"),
    ("Perjeta", "pertuzumab"),
    ("Taxotere", "docetaxel"),
    ("Kadcyla", "ado-trastuzumab emtansine"),
    ("Tukysa", "tucatinib"),
    ("Xeloda", "capecitabine"),
)
_BRAND_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    # Non-letter lookarounds — not \b — so brand names also match when bordered
    # by underscores (which are word-chars and so do not produce a \b boundary).
    (re.compile(rf"(?<![A-Za-z]){re.escape(brand)}(?![A-Za-z])", re.IGNORECASE), generic)
    for brand, generic in BRAND_TO_GENERIC
)

MismatchKind = Literal["line", "regimen", "out_of_policy_continuation"]


# ---------- dataclasses ----------


@dataclass(frozen=True, slots=True)
class TreatmentCourseRow:
    course_index: int  # 0-indexed in chronological order
    treatment_id: str
    line: int
    regimen: str
    start_date: date_cls
    end_date: date_cls | None  # null when open at last_followup


@dataclass(frozen=True, slots=True)
class PolicyMismatch:
    mismatch_kind: MismatchKind
    course_index: int
    line: int | None
    expected_line: int | None
    observed_line: int | None
    expected_aliases: tuple[str, ...] | None
    observed_regimen: str | None


@dataclass(frozen=True, slots=True)
class SoftWarning:
    rule_id: str
    message: str
    event_id: str | None


@dataclass(frozen=True, slots=True)
class AcquiredAlterationRow:
    alteration_id: str
    alteration: str
    supporting_clone: str
    detection_date: date_cls | None


@dataclass(frozen=True, slots=True)
class CellKeyTuple:
    condition_id: str
    policy_id: str
    model_id: str
    model_short: str
    conditioning_label: str
    conditioning_hash: str


@dataclass(frozen=True, slots=True)
class TrajectorySummary:
    trajectory_dir: str
    cell: CellKeyTuple
    future_index: int
    accepted: bool
    attempts_to_accept: int | None
    failure_kind: str | None
    event_count: int | None
    snapshot_count: int | None
    progression_count: int | None
    response_count: int | None
    toxicity_count: int | None
    molecular_lead_event_count: int | None
    pro_update_count: int | None
    clinical_decision_count: int | None
    local_intervention_count: int | None
    acquired_alterations_count: int | None
    treatment_course_count: int | None
    diagnosis_date: date_cls | None
    outcome_type: str | None
    outcome_date: date_cls | None
    outcome_cause: str | None
    os_days: int | None
    treatment_courses: tuple[TreatmentCourseRow, ...]
    acquired_alterations: tuple[AcquiredAlterationRow, ...]
    ctcae_grades: tuple[int, ...]
    treatment_reasons: tuple[str, ...]
    clinical_decision_types: tuple[str, ...]
    policy_mismatches: tuple[PolicyMismatch, ...]
    soft_no_progression_before_death_from_progression: bool
    replay_warnings: tuple[SoftWarning, ...]
    cost_usd: float | None
    output_tokens: int | None


@dataclass(frozen=True, slots=True)
class SkippedTrajectory:
    trajectory_dir: str
    reason: str


@dataclass(frozen=True, slots=True)
class SummaryArtifacts:
    summary_json_path: Path
    summary_md_path: Path
    figures: tuple[Path, ...]


# ---------- pure helpers ----------


def normalize_regimen_string(text: str) -> str:
    """Lowercase and strip [-/+_ \\s] for robust alias matching."""
    return _SEPARATOR_RE.sub("", text.lower())


def apply_brand_map(text: str) -> str:
    """Replace known brand names with their generic equivalents (case-insensitive).

    Applied only to the OBSERVED regimen string before matching. Policy aliases
    are taken at face value. Idempotent; safe to apply to already-genericised text.
    """
    out = text
    for pattern, generic in _BRAND_PATTERNS:
        out = pattern.sub(generic, out)
    return out


def strip_observed_regimen_annotation(text: str) -> str:
    """Remove parenthetical dose/admin annotations from observed regimen text."""
    return _PARENTHETICAL_RE.sub("", text)


def _regimen_tokens(text: str) -> set[str]:
    """Tokenise a regimen string for token-bag matching.

    Strategy: split on the unified separator set including comma, hyphen,
    whitespace, slash, plus, and underscore. Lowercase. Empty tokens dropped.

    Examples:
      ``T-DM1``                                  → ``{'t', 'dm1'}``
      ``trastuzumab/pertuzumab/docetaxel``       → ``{'trastuzumab', 'pertuzumab', 'docetaxel'}``
      ``docetaxel_trastuzumab_pertuzumab``       → same (reorder-invariant)
      ``docetaxel-trastuzumab-pertuzumab``       → same (hyphen-separated)
      ``Docetaxel, Trastuzumab, Pertuzumab``     → same (comma-separated)
      ``ado-trastuzumab emtansine``              → ``{'ado', 'trastuzumab', 'emtansine'}``

    Acronyms like ``T-DM1`` vs ``TDM1`` are NOT equivalent under token-bag
    (the hyphen-form yields two tokens, the no-hyphen form yields one) —
    they are handled by the parallel substring match in
    ``_regimen_matches_any_alias`` via ``normalize_regimen_string`` which
    strips all separators including hyphens. The OR of the two strategies
    covers both classes of variation.
    """
    return {
        tok.lower()
        for tok in _TOKEN_SPLIT_RE.split(text)
        if tok.strip()
    }


def _parse_date(value: Any) -> date_cls | None:
    if value is None:
        return None
    if isinstance(value, date_cls):
        return value
    if isinstance(value, str):
        try:
            return date_cls.fromisoformat(value)
        except ValueError:
            return None
    return None


def _regimen_label(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("label") or "")
    if value is None:
        return ""
    return str(value)


def derive_treatment_courses(
    ledger: Sequence[Mapping[str, Any]],
) -> list[TreatmentCourseRow]:
    """Walk MaterializedTrajectory.ledger to extract chronological treatment courses.

    Behavior per memo §7:
      - systemic_treatment_start    -> open a new course
      - systemic_treatment_switch   -> close the current open course; open a new course
      - systemic_treatment_stop     -> close the current open course (end_date = event.date)
      - death              -> close all open courses (end_date = event.date)
      - last_followup      -> leave open courses open (end_date = null)
    """
    courses: list[TreatmentCourseRow] = []
    open_course: dict[str, Any] | None = None
    next_index = 0

    def _close(open_c: dict[str, Any], end_date: date_cls | None) -> None:
        courses.append(
            TreatmentCourseRow(
                course_index=open_c["course_index"],
                treatment_id=open_c["treatment_id"],
                line=open_c["line"],
                regimen=open_c["regimen"],
                start_date=open_c["start_date"],
                end_date=end_date,
            )
        )

    for event in ledger:
        evt_type = event.get("type")
        evt_date = _parse_date(event.get("date"))
        if evt_type in {"systemic_treatment_start", "treatment_start"}:
            open_course = {
                "course_index": next_index,
                "treatment_id": event.get("treatment_id") or "",
                "line": int(event.get("line") or (next_index + 1)),
                "regimen": _regimen_label(event.get("regimen")),
                "start_date": evt_date,
            }
            next_index += 1
        elif evt_type in {"systemic_treatment_switch", "treatment_switch"}:
            if open_course is not None:
                _close(open_course, evt_date)
            open_course = {
                "course_index": next_index,
                "treatment_id": event.get("treatment_id") or "",
                "line": int(event.get("line") or (next_index + 1)),
                "regimen": _regimen_label(event.get("regimen")),
                "start_date": evt_date,
            }
            next_index += 1
        elif evt_type in {"systemic_treatment_stop", "treatment_stop"}:
            if open_course is not None:
                _close(open_course, evt_date)
                open_course = None
        elif evt_type == "death":
            if open_course is not None:
                _close(open_course, evt_date)
                open_course = None
        elif evt_type == "last_followup":
            if open_course is not None:
                _close(open_course, None)
                open_course = None
        # response / progression do not change treatment-course state.

    # Defensive: any course still open at the end of an iteration that didn't
    # see a terminal event (malformed ledger) is recorded with end_date=null.
    if open_course is not None:
        _close(open_course, None)

    return courses


def check_policy_adherence(
    courses: Sequence[TreatmentCourseRow],
    policy: PolicySpec,
) -> list[PolicyMismatch]:
    """Return the list of policy mismatches per memo §6.3."""
    mismatches: list[PolicyMismatch] = []
    sequence = policy.regimen_sequence
    unconstrained_after = policy.unconstrained_after_line

    for course in courses:
        i = course.course_index
        L = course.line
        R = course.regimen

        if i < len(sequence):
            expected = sequence[i]
            # Line mismatch
            if L != expected.line:
                mismatches.append(
                    PolicyMismatch(
                        mismatch_kind="line",
                        course_index=i,
                        line=None,
                        expected_line=expected.line,
                        observed_line=L,
                        expected_aliases=None,
                        observed_regimen=R,
                    )
                )
            # Regimen mismatch
            if not _regimen_matches_any_alias(R, expected.aliases):
                mismatches.append(
                    PolicyMismatch(
                        mismatch_kind="regimen",
                        course_index=i,
                        line=L,
                        expected_line=None,
                        observed_line=None,
                        expected_aliases=tuple(expected.aliases),
                        observed_regimen=R,
                    )
                )
        else:
            # Beyond the policy's prescribed sequence.
            if _is_same_line_same_regimen_modification(course, courses):
                continue
            if unconstrained_after is not None and L > unconstrained_after:
                # Clinician-chosen / unconstrained — not scored.
                continue
            mismatches.append(
                PolicyMismatch(
                    mismatch_kind="out_of_policy_continuation",
                    course_index=i,
                    line=L,
                    expected_line=None,
                    observed_line=None,
                    expected_aliases=None,
                    observed_regimen=R,
                )
            )

    return mismatches


def _is_same_line_same_regimen_modification(
    course: TreatmentCourseRow,
    courses: Sequence[TreatmentCourseRow],
) -> bool:
    """Return True for same-line dose/admin changes represented as new courses."""
    if course.course_index <= 0 or course.course_index >= len(courses):
        return False
    previous = courses[course.course_index - 1]
    return (
        course.line == previous.line
        and _regimen_matches_any_alias(course.regimen, (previous.regimen,))
    )


def _regimen_matches_any_alias(observed: str, aliases: Iterable[str]) -> bool:
    """Match observed regimen against any alias by either substring or token-bag.

    Two-stage match, BOTH applied to brand-mapped observed text:
      1. Substring after separator-strip. Catches acronym matches
         (``T-DM1`` → ``tdm1`` substring of ``tdm1``) and suffix matches
         (``trastuzumab emtansine`` substring of ``adotrastuzumabemtansine``).
      2. Token-bag subset. Catches arbitrary word-order variants
         (``docetaxel_trastuzumab_pertuzumab`` against alias
         ``trastuzumab/pertuzumab/docetaxel``).

    Either match wins. Backward-compatible with the prior substring-only check.
    """
    observed_branded = apply_brand_map(strip_observed_regimen_annotation(observed))
    observed_normalized = normalize_regimen_string(observed_branded)
    observed_tokens = _regimen_tokens(observed_branded)

    for alias in aliases:
        if not alias:
            continue
        # Stage 1 — substring match on the separator-stripped form.
        if normalize_regimen_string(alias) in observed_normalized:
            return True
        # Stage 2 — token-bag subset (every alias token present in observed).
        alias_tokens = _regimen_tokens(alias)
        if alias_tokens and alias_tokens.issubset(observed_tokens):
            return True
    return False


def soft_no_progression_before_death_from_progression(materialized: Mapping[str, Any]) -> bool:
    """Fires when outcome.cause mentions progression but no progression event occurred."""
    snapshots = materialized.get("snapshots") or []
    if not snapshots:
        return False
    final = snapshots[-1]
    outcome = final.get("outcome") if isinstance(final, Mapping) else None
    if not isinstance(outcome, Mapping):
        return False
    if outcome.get("type") != "death":
        return False
    cause = outcome.get("cause") or ""
    if not _PROGRESSION_CAUSE_RE.search(str(cause)):
        return False
    # Cause mentions progression. Check if any progression event fired.
    ledger = materialized.get("ledger") or []
    for event in ledger:
        if isinstance(event, Mapping) and event.get("type") == "progression":
            return False
    return True


# ---------- run-root loading ----------


def _load_config_echo(run_root: Path) -> dict[str, PolicySpec]:
    config_path = run_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config echo not found at {config_path}; "
            "is this run root a finished POC run?"
        )
    config = load_config(config_path)
    return {p.id: p for p in config.policies}


def _read_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def summarize_trajectory(
    trajectory_dir: Path,
    policies_by_id: Mapping[str, PolicySpec],
) -> TrajectorySummary | SkippedTrajectory:
    """Build a TrajectorySummary by reading the on-disk trajectory artifacts."""
    result_path = trajectory_dir / "trajectory_result.json"
    if not result_path.exists():
        return SkippedTrajectory(
            trajectory_dir=trajectory_dir.name,
            reason="missing trajectory_result.json",
        )
    try:
        result_data = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return SkippedTrajectory(
            trajectory_dir=trajectory_dir.name,
            reason=f"trajectory_result.json malformed: {exc}",
        )

    cell_data = result_data.get("cell") or {}
    try:
        cell = CellKeyTuple(
            condition_id=str(cell_data["condition_id"]),
            policy_id=str(cell_data["policy_id"]),
            model_id=str(cell_data["model_id"]),
            model_short=str(cell_data["model_short"]),
            conditioning_label=str(cell_data.get("conditioning_label", "")),
            conditioning_hash=str(cell_data.get("conditioning_hash", "")),
        )
    except KeyError as exc:
        return SkippedTrajectory(
            trajectory_dir=trajectory_dir.name,
            reason=f"trajectory_result.json missing cell field: {exc}",
        )

    accepted = bool(result_data.get("accepted", False))
    failure_kind = result_data.get("failure_kind")
    attempts_to_accept = result_data.get("attempts_to_accept")

    materialized: dict[str, Any] | None = None
    cost_usd: float | None = None
    output_tokens: int | None = None
    if accepted and attempts_to_accept is not None:
        # Accepted attempts are always written under attempt_NN/.
        attempt_dir = trajectory_dir / f"attempt_{int(attempts_to_accept):02d}"
        materialized = _read_json_if_exists(attempt_dir / "materialized_trajectory.json")
        attempt_result = _read_json_if_exists(attempt_dir / "result.json")
        if isinstance(attempt_result, Mapping):
            cost_usd = attempt_result.get("cost_usd")
            usage = attempt_result.get("usage") or {}
            output_tokens = usage.get("output_tokens") if isinstance(usage, Mapping) else None

    # Sad-path branch: accepted=True but materialized missing.
    if accepted and materialized is None:
        return SkippedTrajectory(
            trajectory_dir=trajectory_dir.name,
            reason="accepted but materialized_trajectory.json missing",
        )

    treatment_courses: tuple[TreatmentCourseRow, ...] = ()
    policy_mismatches: tuple[PolicyMismatch, ...] = ()
    replay_warnings: tuple[SoftWarning, ...] = ()
    soft_no_prog = False
    event_count: int | None = None
    snapshot_count: int | None = None
    progression_count: int | None = None
    response_count: int | None = None
    toxicity_count: int | None = None
    molecular_lead_event_count: int | None = None
    pro_update_count: int | None = None
    clinical_decision_count: int | None = None
    local_intervention_count: int | None = None
    acquired_alterations_count: int | None = None
    acquired_alterations: tuple[AcquiredAlterationRow, ...] = ()
    ctcae_grades: tuple[int, ...] = ()
    treatment_reasons: tuple[str, ...] = ()
    clinical_decision_types: tuple[str, ...] = ()
    diagnosis_date: date_cls | None = None
    outcome_type: str | None = None
    outcome_date: date_cls | None = None
    outcome_cause: str | None = None
    os_days: int | None = None

    if materialized is not None:
        ledger = materialized.get("ledger") or []
        snapshots = materialized.get("snapshots") or []
        event_count = len(ledger)
        snapshot_count = len(snapshots)
        progression_count = sum(1 for e in ledger if isinstance(e, Mapping) and e.get("type") == "progression")
        response_count = sum(1 for e in ledger if isinstance(e, Mapping) and e.get("type") == "response")
        toxicity_count = sum(1 for e in ledger if isinstance(e, Mapping) and e.get("type") == "toxicity")
        molecular_lead_event_count = sum(
            1 for e in ledger
            if isinstance(e, Mapping) and e.get("type") in {"clonal_evolution_event", "molecular_lead_event"}
        )
        pro_update_count = sum(
            1 for e in ledger
            if isinstance(e, Mapping) and e.get("type") in {"patient_state_update", "pro_update"}
        )
        clinical_decision_count = 0
        # Count both the new canonical type and the legacy "surgery" name so
        # summary builds still work against pre-rename run-root artifacts.
        local_intervention_count = sum(
            1 for e in ledger
            if isinstance(e, Mapping) and e.get("type") in {"local_intervention", "surgery"}
        )
        ctcae_grades = tuple(
            int(grade)
            for e in ledger
            if (
                isinstance(e, Mapping)
                and e.get("type") == "toxicity"
                and isinstance(e.get("ctcae_v5_grade", e.get("ctcae_grade")), int | float)
            )
            for grade in [e.get("ctcae_v5_grade", e.get("ctcae_grade"))]
        )
        clinical_decision_types = ()

        baseline = materialized.get("baseline") or {}
        identity = baseline.get("identity_card") or {} if isinstance(baseline, Mapping) else {}
        diagnosis = identity.get("diagnosis") or {}
        diagnosis_date = _parse_date(diagnosis.get("diagnosis_date"))

        if snapshots:
            final = snapshots[-1]
            outcome = final.get("outcome") if isinstance(final, Mapping) else None
            if isinstance(outcome, Mapping):
                outcome_type = outcome.get("type")
                outcome_date = _parse_date(outcome.get("date"))
                outcome_cause = outcome.get("cause")

        if diagnosis_date is not None and outcome_date is not None:
            os_days = (outcome_date - diagnosis_date).days

        treatment_courses = tuple(derive_treatment_courses(ledger))
        acquired_rows: list[AcquiredAlterationRow] = []
        for alteration in materialized.get("acquired_alterations") or []:
            if not isinstance(alteration, Mapping):
                continue
            acquired_rows.append(
                AcquiredAlterationRow(
                    alteration_id=str(alteration.get("alteration_id") or ""),
                    alteration=str(alteration.get("alteration") or ""),
                    supporting_clone=str(alteration.get("supporting_clone") or ""),
                    detection_date=(
                        _parse_date(alteration.get("recognized_date"))
                        or _parse_date(alteration.get("detection_date"))
                        or _parse_date(alteration.get("first_presence_date"))
                    ),
                )
            )
        acquired_alterations = tuple(acquired_rows)
        acquired_alterations_count = len(acquired_alterations)
        treatment_reasons = tuple(
            str(course.get("end_reason"))
            for course in materialized.get("treatment_history") or []
            if isinstance(course, Mapping) and course.get("end_reason")
        )

        policy = policies_by_id.get(cell.policy_id)
        if policy is not None:
            policy_mismatches = tuple(check_policy_adherence(treatment_courses, policy))

        warnings = materialized.get("validation_log") or []
        replay_warnings = tuple(
            SoftWarning(
                rule_id=str(w.get("rule_id", "")),
                message=str(w.get("message", "")),
                event_id=w.get("event_id"),
            )
            for w in warnings
            if isinstance(w, Mapping)
        )
        soft_no_prog = soft_no_progression_before_death_from_progression(materialized)

    return TrajectorySummary(
        trajectory_dir=trajectory_dir.name,
        cell=cell,
        future_index=int(result_data.get("future_index", 0)),
        accepted=accepted,
        attempts_to_accept=attempts_to_accept,
        failure_kind=failure_kind,
        event_count=event_count,
        snapshot_count=snapshot_count,
        progression_count=progression_count,
        response_count=response_count,
        toxicity_count=toxicity_count,
        molecular_lead_event_count=molecular_lead_event_count,
        pro_update_count=pro_update_count,
        clinical_decision_count=clinical_decision_count,
        local_intervention_count=local_intervention_count,
        acquired_alterations_count=acquired_alterations_count,
        treatment_course_count=len(treatment_courses) if materialized is not None else None,
        diagnosis_date=diagnosis_date,
        outcome_type=outcome_type,
        outcome_date=outcome_date,
        outcome_cause=outcome_cause,
        os_days=os_days,
        treatment_courses=treatment_courses,
        acquired_alterations=acquired_alterations,
        ctcae_grades=ctcae_grades,
        treatment_reasons=treatment_reasons,
        clinical_decision_types=clinical_decision_types,
        policy_mismatches=policy_mismatches,
        soft_no_progression_before_death_from_progression=soft_no_prog,
        replay_warnings=replay_warnings,
        cost_usd=cost_usd,
        output_tokens=output_tokens,
    )


def load_trajectory_results(
    run_root: Path,
    policies_by_id: Mapping[str, PolicySpec],
) -> tuple[list[TrajectorySummary], list[SkippedTrajectory]]:
    trajectories_root = run_root / "trajectories"
    if not trajectories_root.exists() or not trajectories_root.is_dir():
        raise FileNotFoundError(
            f"trajectories directory not found at {trajectories_root}"
        )

    summaries: list[TrajectorySummary] = []
    skipped: list[SkippedTrajectory] = []
    for traj_dir in sorted(trajectories_root.iterdir()):
        if not traj_dir.is_dir():
            continue
        outcome = summarize_trajectory(traj_dir, policies_by_id)
        if isinstance(outcome, SkippedTrajectory):
            skipped.append(outcome)
        else:
            summaries.append(outcome)
    # Sort deterministically for downstream consumers.
    summaries.sort(key=lambda s: s.trajectory_dir)
    skipped.sort(key=lambda s: s.trajectory_dir)
    return summaries, skipped


# ---------- aggregation ----------


def _round_float(value: float | None, places: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), places)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _accepted_rich_content_aggregate(rows: Sequence[TrajectorySummary]) -> dict[str, Any]:
    accepted = [r for r in rows if r.accepted]

    def _count_values(field_name: str) -> list[int]:
        return [
            int(value)
            for r in accepted
            if (value := getattr(r, field_name)) is not None
        ]

    clonal_counts = _count_values("molecular_lead_event_count")
    patient_state_counts = _count_values("pro_update_count")
    legacy_decision_counts = _count_values("clinical_decision_count")
    return {
        "toxicity_count_mean": _round_float(_mean(_count_values("toxicity_count")), 3),
        "clonal_evolution_event_count_mean": _round_float(_mean(clonal_counts), 3),
        "patient_state_update_count_mean": _round_float(_mean(patient_state_counts), 3),
        "legacy_clinical_decision_count_mean": _round_float(
            _mean(legacy_decision_counts), 3
        ),
        # Backward-compatible aliases for pre-paper-v1 consumers.
        "molecular_lead_event_count_mean": _round_float(
            _mean(clonal_counts), 3
        ),
        "pro_update_count_mean": _round_float(_mean(patient_state_counts), 3),
        "clinical_decision_count_mean": _round_float(
            _mean(legacy_decision_counts), 3
        ),
        "local_intervention_count_mean": _round_float(_mean(_count_values("local_intervention_count")), 3),
        "acquired_alterations_total": sum(
            r.acquired_alterations_count or 0 for r in accepted
        ),
        "ctcae_grade_distribution": _distribution(
            grade for r in accepted for grade in r.ctcae_grades
        ),
        "treatment_reason_distribution": _distribution(
            reason for r in accepted for reason in r.treatment_reasons
        ),
        "clinical_decision_type_distribution": _distribution(
            decision for r in accepted for decision in r.clinical_decision_types
        ),
    }


def _trajectory_row_for_json(t: TrajectorySummary) -> dict[str, Any]:
    return {
        "trajectory_dir": t.trajectory_dir,
        "cell": {
            "condition_id": t.cell.condition_id,
            "policy_id": t.cell.policy_id,
            "model_id": t.cell.model_id,
            "model_short": t.cell.model_short,
            "conditioning_label": t.cell.conditioning_label,
            "conditioning_hash": t.cell.conditioning_hash,
        },
        "future_index": t.future_index,
        "accepted": t.accepted,
        "attempts_to_accept": t.attempts_to_accept,
        "failure_kind": t.failure_kind,
        "event_count": t.event_count,
        "snapshot_count": t.snapshot_count,
        "progression_count": t.progression_count,
        "response_count": t.response_count,
        "toxicity_count": t.toxicity_count,
        "clonal_evolution_event_count": t.molecular_lead_event_count,
        "patient_state_update_count": t.pro_update_count,
        "legacy_clinical_decision_count": t.clinical_decision_count,
        # Backward-compatible aliases for pre-paper-v1 consumers.
        "molecular_lead_event_count": t.molecular_lead_event_count,
        "pro_update_count": t.pro_update_count,
        "clinical_decision_count": t.clinical_decision_count,
        "local_intervention_count": t.local_intervention_count,
        "acquired_alterations_count": t.acquired_alterations_count,
        "treatment_course_count": t.treatment_course_count,
        "diagnosis_date": t.diagnosis_date.isoformat() if t.diagnosis_date else None,
        "outcome_type": t.outcome_type,
        "outcome_date": t.outcome_date.isoformat() if t.outcome_date else None,
        "outcome_cause": t.outcome_cause,
        "os_days": t.os_days,
        "treatment_courses": [
            {
                "course_index": c.course_index,
                "treatment_id": c.treatment_id,
                "line": c.line,
                "regimen": c.regimen,
                "start_date": c.start_date.isoformat() if c.start_date else None,
                "end_date": c.end_date.isoformat() if c.end_date else None,
            }
            for c in t.treatment_courses
        ],
        "acquired_alterations": [
            {
                "alteration_id": a.alteration_id,
                "alteration": a.alteration,
                "supporting_clone": a.supporting_clone,
                "detection_date": a.detection_date.isoformat() if a.detection_date else None,
            }
            for a in t.acquired_alterations
        ],
        "ctcae_grades": list(t.ctcae_grades),
        "treatment_reasons": list(t.treatment_reasons),
        "clinical_decision_types": list(t.clinical_decision_types),
        "policy_mismatches": [
            {
                "mismatch_kind": m.mismatch_kind,
                "course_index": m.course_index,
                "line": m.line,
                "expected_line": m.expected_line,
                "observed_line": m.observed_line,
                "expected_aliases": list(m.expected_aliases) if m.expected_aliases else None,
                "observed_regimen": m.observed_regimen,
            }
            for m in t.policy_mismatches
        ],
        "soft_no_progression_before_death_from_progression": t.soft_no_progression_before_death_from_progression,
        "replay_warning_count": len(t.replay_warnings),
        "cost_usd": _round_float(t.cost_usd),
        "output_tokens": t.output_tokens,
    }


def _category_aggregate(rows: Sequence[TrajectorySummary]) -> dict[str, Any]:
    accepted = [r for r in rows if r.accepted]
    failed = [r for r in rows if not r.accepted and r.failure_kind not in (None, "dry_run")]
    dry_run = [r for r in rows if r.failure_kind == "dry_run"]

    failure_kinds: dict[str, int] = {}
    for r in failed:
        if r.failure_kind:
            failure_kinds[r.failure_kind] = failure_kinds.get(r.failure_kind, 0) + 1

    event_counts = [r.event_count for r in accepted if r.event_count is not None]
    progression_counts = [r.progression_count for r in accepted if r.progression_count is not None]
    os_days_list = [r.os_days for r in accepted if r.os_days is not None]
    output_tokens_list = [r.output_tokens for r in accepted if r.output_tokens is not None]
    cost_list = [r.cost_usd for r in accepted if r.cost_usd is not None]

    n_courses_with_mismatch = sum(
        1 for r in accepted if r.policy_mismatches
    )
    aggregate = {
        "trajectories_planned": len(rows),
        "trajectories_accepted": len(accepted),
        "trajectories_failed": len(failed),
        "trajectories_dry_run": len(dry_run),
        "failure_kinds": dict(sorted(failure_kinds.items())),
        "event_count_mean": _round_float(_mean(event_counts), 3),
        "progression_count_mean": _round_float(_mean(progression_counts), 3),
        "os_days_mean": _round_float(_mean(os_days_list), 1),
        "output_tokens_mean": _round_float(_mean(output_tokens_list), 1),
        "cost_usd_total": _round_float(sum(cost_list) if cost_list else 0.0, 6),
        "cost_usd_mean": _round_float(_mean(cost_list), 6),
        "soft_no_progression_before_death_from_progression": sum(
            1 for r in accepted if r.soft_no_progression_before_death_from_progression
        ),
        "policy_mismatch_rate": (
            _round_float(n_courses_with_mismatch / len(accepted), 4)
            if accepted else None
        ),
    }
    aggregate.update(_accepted_rich_content_aggregate(rows))
    return aggregate


def _cell_level_counts(
    summaries: Sequence[TrajectorySummary],
) -> tuple[int, int, int, int]:
    """Return (cells_planned, fully_accepted, partially_accepted, fully_failed)."""
    by_cell: dict[tuple[str, str, str], list[TrajectorySummary]] = {}
    for s in summaries:
        if s.failure_kind == "dry_run":
            continue
        key = (s.cell.condition_id, s.cell.policy_id, s.cell.model_short)
        by_cell.setdefault(key, []).append(s)

    fully_accepted = 0
    partially_accepted = 0
    fully_failed = 0
    for rows in by_cell.values():
        accepted_count = sum(1 for r in rows if r.accepted)
        if accepted_count == len(rows):
            fully_accepted += 1
        elif accepted_count == 0:
            fully_failed += 1
        else:
            partially_accepted += 1
    return len(by_cell), fully_accepted, partially_accepted, fully_failed


def _attempts_histogram(summaries: Sequence[TrajectorySummary]) -> dict[str, int]:
    hist: dict[str, int] = {"1": 0, "2": 0, "null_failed": 0, "dry_run": 0}
    for s in summaries:
        if s.failure_kind == "dry_run":
            hist["dry_run"] += 1
        elif s.accepted and s.attempts_to_accept is not None:
            key = str(s.attempts_to_accept)
            hist[key] = hist.get(key, 0) + 1
        else:
            hist["null_failed"] += 1
    return dict(sorted(hist.items()))


def build_summary_payload(
    summaries: Sequence[TrajectorySummary],
    skipped: Sequence[SkippedTrajectory],
    run_root: Path,
) -> dict[str, Any]:
    by_model: dict[str, list[TrajectorySummary]] = {}
    by_policy: dict[str, list[TrajectorySummary]] = {}
    by_condition: dict[str, list[TrajectorySummary]] = {}
    by_conditioning_label: dict[str, list[TrajectorySummary]] = {}
    for s in summaries:
        by_model.setdefault(s.cell.model_id, []).append(s)
        by_policy.setdefault(s.cell.policy_id, []).append(s)
        by_condition.setdefault(s.cell.condition_id, []).append(s)
        if s.cell.conditioning_label:
            by_conditioning_label.setdefault(s.cell.conditioning_label, []).append(s)

    cells_planned, fully_accepted, partially_accepted, fully_failed = _cell_level_counts(summaries)

    # Read run-root manifest for config_id provenance, but tolerate absence.
    manifest_path = run_root / "manifest.json"
    manifest = _read_json_if_exists(manifest_path) or {}
    config_id = manifest.get("config_id") if isinstance(manifest, Mapping) else None
    run_id = manifest.get("run_id") if isinstance(manifest, Mapping) else None
    # Conditioning labels are per-cell (post-Commit-B). Surface the set of
    # distinct labels observed across this run's trajectories.
    conditioning_labels = sorted(by_conditioning_label.keys())
    config_hash = (
        manifest.get("config_hash") if isinstance(manifest, Mapping) else None
    )
    prompt_template_hash = (
        manifest.get("prompt_template_hash") if isinstance(manifest, Mapping) else None
    )

    overall = _category_aggregate(summaries)

    # Censoring breakdown — censored-at-last-follow-up trajectories are NOT
    # failures and are NOT incomplete. They are valid terminal events under
    # the default terminal_contract; production paper-v1 runs will flip the
    # contract to "death_only" to exclude them from the cohort by design.
    accepted = [s for s in summaries if s.accepted]
    n_death = sum(1 for s in accepted if s.outcome_type == "death")
    n_censored = sum(1 for s in accepted if s.outcome_type == "last_followup")
    terminal_contract_from_manifest = None
    if isinstance(manifest, Mapping):
        matrix_block = manifest.get("matrix")
        if isinstance(matrix_block, Mapping):
            terminal_contract_from_manifest = matrix_block.get("terminal_contract")

    scope_note = _build_scope_note(
        n_death=n_death,
        n_censored=n_censored,
        n_total_accepted=len(accepted),
        terminal_contract=terminal_contract_from_manifest,
    )

    return {
        "run_id": run_id,
        "config_id": config_id,
        "conditioning_labels": conditioning_labels,
        "config_hash": config_hash,
        "prompt_template_hash": prompt_template_hash,
        "run_root": str(run_root.name),
        "scope_note": scope_note,
        "terminal_contract": terminal_contract_from_manifest,
        "cells_planned": cells_planned,
        "cells_fully_accepted": fully_accepted,
        "cells_partially_accepted": partially_accepted,
        "cells_fully_failed": fully_failed,
        "trajectories_planned": overall["trajectories_planned"],
        "trajectories_accepted": overall["trajectories_accepted"],
        "trajectories_failed": overall["trajectories_failed"],
        "trajectories_dry_run": overall["trajectories_dry_run"],
        "trajectories_skipped": len(skipped),
        "trajectories_death_terminated": n_death,
        "trajectories_censored_at_last_followup": n_censored,
        "attempts_to_accept_hist": _attempts_histogram(summaries),
        "rich_content": _accepted_rich_content_aggregate(summaries),
        "by_model": {k: _category_aggregate(v) for k, v in sorted(by_model.items())},
        "by_policy": {k: _category_aggregate(v) for k, v in sorted(by_policy.items())},
        "by_condition": {k: _category_aggregate(v) for k, v in sorted(by_condition.items())},
        "by_conditioning_label": {
            k: _category_aggregate(v) for k, v in sorted(by_conditioning_label.items())
        },
        "trajectories": [_trajectory_row_for_json(t) for t in summaries],
        "skipped": [{"trajectory_dir": s.trajectory_dir, "reason": s.reason} for s in skipped],
    }


def _build_scope_note(
    *,
    n_death: int,
    n_censored: int,
    n_total_accepted: int,
    terminal_contract: str | None,
) -> str:
    """Compose the deterministic scope-note string for summary.json/md.

    Wording deliberately uses "censored at last follow-up" — never
    "incomplete", "failed", or "missing" — to keep the scientific framing
    correct: last_followup is a valid terminal event under the lean ledger,
    not a failure mode.
    """
    parts = [
        f"Rev5 POC prototype — {n_total_accepted} accepted trajectories",
        f"({n_death} death-terminated, {n_censored} censored at last follow-up).",
    ]
    if terminal_contract == "death_only":
        parts.append("terminal_contract=death_only enforced (no censored accepted).")
    elif terminal_contract == "death_or_last_followup":
        parts.append("terminal_contract=death_or_last_followup (both terminal types accepted).")
    return " ".join(parts)


# ---------- figures ----------


def _figure_horizon_date(t: TrajectorySummary) -> date_cls | None:
    """Return the date the per-trajectory figure should clip at."""
    if t.outcome_date is not None:
        return t.outcome_date
    if t.treatment_courses:
        # Fall back to the last start_date if no outcome is recorded.
        return max((c.start_date for c in t.treatment_courses if c.start_date), default=None)
    return None


def _render_trajectory_figure(
    t: TrajectorySummary,
    materialized: Mapping[str, Any],
    out_path: Path,
) -> None:
    """Render the 4-panel per-trajectory figure: treatment / burden / ECOG / weight."""
    snapshots = materialized.get("snapshots") or []
    snap_dates: list[date_cls] = []
    burden_band: list[str] = []
    ecog: list[int] = []
    weight: list[float] = []
    for s in snapshots:
        if not isinstance(s, Mapping):
            continue
        d = _parse_date(s.get("date"))
        if d is None:
            continue
        snap_dates.append(d)
        burden_band.append(str(s.get("total_burden_band") or ""))
        patient_state = s.get("patient_state") or {}
        ecog_v = patient_state.get("ecog") if isinstance(patient_state, Mapping) else None
        ecog.append(int(ecog_v) if isinstance(ecog_v, int | float) else 0)
        weight_v = patient_state.get("weight_kg") if isinstance(patient_state, Mapping) else None
        weight.append(float(weight_v) if isinstance(weight_v, int | float) else 0.0)

    horizon = _figure_horizon_date(t)

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(t.trajectory_dir, fontsize=10)

    # Panel 0: treatment timeline (Gantt by line).
    ax_tx = axes[0]
    for course in t.treatment_courses:
        start = course.start_date
        end = course.end_date or horizon
        if start is None or end is None:
            continue
        ax_tx.barh(
            y=course.line,
            width=(end - start).days,
            left=start,
            height=0.6,
            label=course.regimen,
        )
        ax_tx.text(
            start, course.line + 0.35, course.regimen,
            fontsize=7, verticalalignment="bottom",
        )
    ax_tx.set_ylabel("Line")
    ax_tx.set_title("Treatment timeline", fontsize=9, loc="left")
    if t.treatment_courses:
        ax_tx.set_yticks(sorted({c.line for c in t.treatment_courses}))

    # Panel 1: burden band (categorical ↦ numeric for plotting).
    band_order = ["absent", "microscopic", "small", "moderate", "bulky", "diffuse_or_leukemic"]
    band_rank = {b: i for i, b in enumerate(band_order)}
    ax_b = axes[1]
    if snap_dates:
        ax_b.step(
            snap_dates,
            [band_rank.get(b, -1) for b in burden_band],
            where="post",
        )
    ax_b.set_yticks(range(len(band_order)))
    ax_b.set_yticklabels(band_order, fontsize=7)
    ax_b.set_ylabel("Total burden")
    ax_b.set_title("Total burden band", fontsize=9, loc="left")

    # Panel 2: ECOG.
    ax_e = axes[2]
    if snap_dates:
        ax_e.step(snap_dates, ecog, where="post")
    ax_e.set_yticks(range(6))
    ax_e.set_ylim(-0.5, 5.5)
    ax_e.set_ylabel("ECOG")
    ax_e.set_title("ECOG performance status", fontsize=9, loc="left")

    # Panel 3: weight.
    ax_w = axes[3]
    if snap_dates:
        ax_w.plot(snap_dates, weight, marker="o", markersize=3)
    ax_w.set_ylabel("Weight (kg)")
    ax_w.set_xlabel("Date")
    ax_w.set_title("Weight", fontsize=9, loc="left")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _render_aggregate_figures(
    summaries: Sequence[TrajectorySummary],
    out_dir: Path,
) -> list[Path]:
    """Render the five aggregate figures. Returns the list of written PNG paths."""
    written: list[Path] = []

    # 1. OS by policy x model (accepted, death-terminated only).
    fig, ax = plt.subplots(figsize=(8, 5))
    groups: dict[str, list[int]] = {}
    for s in summaries:
        if s.accepted and s.outcome_type == "death" and s.os_days is not None:
            key = f"{s.cell.policy_id}\n{s.cell.model_short}"
            groups.setdefault(key, []).append(s.os_days)
    if groups:
        labels = sorted(groups)
        ax.boxplot([groups[k] for k in labels], tick_labels=labels)
    ax.set_ylabel("OS (days from diagnosis)")
    ax.set_title("Overall survival — accepted, death-terminated")
    fig.tight_layout()
    p = out_dir / "aggregate_os_by_policy.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p)

    # 2. Event count by model.
    fig, ax = plt.subplots(figsize=(8, 5))
    model_events: dict[str, list[int]] = {}
    for s in summaries:
        if s.accepted and s.event_count is not None:
            model_events.setdefault(s.cell.model_short, []).append(s.event_count)
    if model_events:
        labels = sorted(model_events)
        ax.hist([model_events[m] for m in labels], label=labels)
        ax.legend()
    ax.set_xlabel("Event count")
    ax.set_ylabel("Trajectories")
    ax.set_title("Ledger event-count distribution by model")
    fig.tight_layout()
    p = out_dir / "aggregate_event_counts.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p)

    # 3. Progression count by model.
    fig, ax = plt.subplots(figsize=(8, 5))
    model_prog: dict[str, list[int]] = {}
    for s in summaries:
        if s.accepted and s.progression_count is not None:
            model_prog.setdefault(s.cell.model_short, []).append(s.progression_count)
    if model_prog:
        labels = sorted(model_prog)
        ax.hist([model_prog[m] for m in labels], label=labels, bins=range(0, 12))
        ax.legend()
    ax.set_xlabel("Progression events")
    ax.set_ylabel("Trajectories")
    ax.set_title("Progression-event count distribution by model")
    fig.tight_layout()
    p = out_dir / "aggregate_progression_counts.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p)

    # 4. Cost / output tokens by model.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    model_cost: dict[str, list[float]] = {}
    model_tok: dict[str, list[int]] = {}
    for s in summaries:
        if not s.accepted:
            continue
        if s.cost_usd is not None:
            model_cost.setdefault(s.cell.model_short, []).append(s.cost_usd)
        if s.output_tokens is not None:
            model_tok.setdefault(s.cell.model_short, []).append(s.output_tokens)
    if model_cost:
        labels = sorted(model_cost)
        axes[0].bar(labels, [sum(model_cost[m]) for m in labels])
    axes[0].set_ylabel("Total cost USD")
    axes[0].set_title("Total cost by model")
    if model_tok:
        labels = sorted(model_tok)
        axes[1].bar(labels, [sum(model_tok[m]) / max(1, len(model_tok[m])) for m in labels])
    axes[1].set_ylabel("Mean output tokens")
    axes[1].set_title("Mean output tokens by model")
    fig.tight_layout()
    p = out_dir / "aggregate_cost_tokens.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p)

    # 5. Acceptance rate by cell.
    fig, ax = plt.subplots(figsize=(10, 5))
    by_cell: dict[str, tuple[int, int]] = {}
    for s in summaries:
        if s.failure_kind == "dry_run":
            continue
        key = f"{s.cell.condition_id}\n{s.cell.policy_id}\n{s.cell.model_short}"
        accepted_n, total_n = by_cell.get(key, (0, 0))
        by_cell[key] = (accepted_n + (1 if s.accepted else 0), total_n + 1)
    if by_cell:
        labels = sorted(by_cell)
        rates = [by_cell[k][0] / by_cell[k][1] if by_cell[k][1] else 0.0 for k in labels]
        ax.bar(range(len(labels)), rates)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Acceptance rate")
        ax.set_title("Acceptance rate by cell")
    fig.tight_layout()
    p = out_dir / "aggregate_acceptance.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p)

    return written


def _render_per_trajectory_figures(
    summaries: Sequence[TrajectorySummary],
    run_root: Path,
    out_dir: Path,
) -> list[Path]:
    written: list[Path] = []
    for t in summaries:
        if not t.accepted or t.attempts_to_accept is None:
            continue
        attempt_dir = run_root / "trajectories" / t.trajectory_dir / f"attempt_{int(t.attempts_to_accept):02d}"
        materialized = _read_json_if_exists(attempt_dir / "materialized_trajectory.json")
        if not isinstance(materialized, Mapping):
            continue
        out_path = out_dir / f"trajectory_{t.trajectory_dir}.png"
        try:
            _render_trajectory_figure(t, materialized, out_path)
        except Exception:  # noqa: BLE001 — figure failure should not crash the summary build
            continue
        written.append(out_path)
    return written


# ---------- markdown readout ----------


def _format_markdown(payload: Mapping[str, Any], figure_dir_name: str = "figures") -> str:
    lines: list[str] = []
    lines.append(f"# Rev5 POC summary — {payload.get('run_id') or '(unknown run)'}")
    lines.append("")
    lines.append(f"- Config: `{payload.get('config_id') or '?'}`")
    lines.append(f"- Run root: `{payload.get('run_root')}`")
    lines.append("")

    lines.append("## Scope")
    lines.append("")
    lines.append(payload.get("scope_note") or "(scope note unavailable)")
    lines.append("")
    n_death = payload.get("trajectories_death_terminated")
    n_censored = payload.get("trajectories_censored_at_last_followup")
    if n_death is not None and n_censored is not None:
        lines.append(
            f"- Death-terminated: **n={n_death}**  |  "
            f"Censored at last follow-up: **n={n_censored}**"
        )
        lines.append("")
    contract = payload.get("terminal_contract")
    if contract:
        lines.append(f"- `terminal_contract`: `{contract}`")
        lines.append("")
    lines.append(
        "*Censored* trajectories ended in a clean `last_followup` event and are "
        "valid POC outputs under the current contract. Production paper-v1 runs "
        "will flip `terminal_contract` to `death_only` so censored trajectories "
        "become an opt-in for continuation / longitudinal studies rather than a "
        "default cohort exit.",
    )
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Cells planned: {payload['cells_planned']}")
    lines.append(
        f"  - Fully accepted: {payload['cells_fully_accepted']} | "
        f"Partial: {payload['cells_partially_accepted']} | "
        f"Fully failed: {payload['cells_fully_failed']}"
    )
    lines.append(f"- Trajectories planned: {payload['trajectories_planned']}")
    lines.append(
        f"  - Accepted: {payload['trajectories_accepted']} | "
        f"Failed: {payload['trajectories_failed']} | "
        f"Dry-run: {payload['trajectories_dry_run']} | "
        f"Skipped: {payload['trajectories_skipped']}"
    )
    hist = payload["attempts_to_accept_hist"]
    lines.append(f"- Attempts-to-accept histogram: {hist}")
    lines.append("")

    def _rich_lines(prefix: str, v: Mapping[str, Any]) -> None:
        lines.append(f"- Rich event means per accepted trajectory ({prefix}): "
                     f"toxicity={v.get('toxicity_count_mean')}, "
                     f"clonal_evolution={v.get('clonal_evolution_event_count_mean')}, "
                     f"patient_state_update={v.get('patient_state_update_count_mean')}, "
                     f"legacy_decision={v.get('legacy_clinical_decision_count_mean')}, "
                     f"local_intervention={v.get('local_intervention_count_mean')}")
        lines.append(f"- Acquired alterations total: {v.get('acquired_alterations_total')}")
        lines.append(f"- CTCAE grade distribution: {v.get('ctcae_grade_distribution')}")
        lines.append(f"- Treatment reason distribution: {v.get('treatment_reason_distribution')}")
        lines.append(
            f"- Clinical decision type distribution: {v.get('clinical_decision_type_distribution')}"
        )

    lines.append("## Rich Content Surface")
    lines.append("")
    _rich_lines("overall", payload.get("rich_content") or {})
    lines.append("")

    def _block(title: str, mapping: Mapping[str, Mapping[str, Any]]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        for k, v in mapping.items():
            lines.append(f"### {k}")
            lines.append("")
            lines.append(f"- Trajectories: planned={v['trajectories_planned']} "
                         f"accepted={v['trajectories_accepted']} "
                         f"failed={v['trajectories_failed']}")
            lines.append(f"- Failure kinds: {v['failure_kinds']}")
            lines.append(f"- Event-count mean: {v['event_count_mean']}")
            lines.append(f"- Progression-count mean: {v['progression_count_mean']}")
            lines.append(f"- OS days mean (death-terminated): {v['os_days_mean']}")
            lines.append(f"- Mean output tokens: {v['output_tokens_mean']}")
            lines.append(f"- Total cost USD: {v['cost_usd_total']}")
            lines.append(
                "- SOFT_NO_PROGRESSION_BEFORE_DEATH_FROM_PROGRESSION: "
                f"{v['soft_no_progression_before_death_from_progression']}"
            )
            lines.append(f"- Policy mismatch rate: {v['policy_mismatch_rate']}")
            _rich_lines(k, v)
            lines.append("")

    _block("By model", payload["by_model"])
    _block("By policy", payload["by_policy"])
    _block("By condition", payload["by_condition"])
    if payload.get("by_conditioning_label"):
        _block("By conditioning label", payload["by_conditioning_label"])

    lines.append("## Figures")
    lines.append("")
    for fname in (
        "aggregate_os_by_policy.png",
        "aggregate_event_counts.png",
        "aggregate_progression_counts.png",
        "aggregate_cost_tokens.png",
        "aggregate_acceptance.png",
    ):
        lines.append(f"- ![{fname}]({figure_dir_name}/{fname})")
    lines.append("")

    # Notable trajectories: policy-mismatch and SOFT-no-progression flags.
    notable_mismatch = [
        t for t in payload["trajectories"] if t["policy_mismatches"]
    ]
    notable_soft = [
        t for t in payload["trajectories"]
        if t["soft_no_progression_before_death_from_progression"]
    ]
    if notable_mismatch:
        lines.append("## Notable: policy mismatches")
        lines.append("")
        for t in notable_mismatch[:25]:
            lines.append(f"- `{t['trajectory_dir']}`: {len(t['policy_mismatches'])} mismatch(es)")
        lines.append("")
    if notable_soft:
        lines.append("## Notable: SOFT_NO_PROGRESSION_BEFORE_DEATH_FROM_PROGRESSION")
        lines.append("")
        for t in notable_soft[:25]:
            lines.append(f"- `{t['trajectory_dir']}`: cause={t['outcome_cause']!r}")
        lines.append("")
    if payload["skipped"]:
        lines.append("## Skipped trajectories")
        lines.append("")
        for s in payload["skipped"]:
            lines.append(f"- `{s['trajectory_dir']}`: {s['reason']}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------- top-level build_summary ----------


def build_summary(run_root: Path) -> SummaryArtifacts:
    """Read run_root/trajectories/**, write run_root/summary/. Idempotent."""
    policies_by_id = _load_config_echo(run_root)
    summaries, skipped = load_trajectory_results(run_root, policies_by_id)
    payload = build_summary_payload(summaries, skipped, run_root)

    summary_dir = run_root / "summary"
    figures_dir = summary_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = summary_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary_md_path = summary_dir / "summary.md"
    summary_md_path.write_text(_format_markdown(payload), encoding="utf-8")

    aggregate_figures = _render_aggregate_figures(summaries, figures_dir)
    traj_figures = _render_per_trajectory_figures(summaries, run_root, figures_dir)
    figures = tuple(sorted(aggregate_figures + traj_figures))

    return SummaryArtifacts(
        summary_json_path=summary_json_path,
        summary_md_path=summary_md_path,
        figures=figures,
    )
