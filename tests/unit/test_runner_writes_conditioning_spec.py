"""Gate B1b — runner persists ``conditioning_spec.json`` per attempt and
``validation_warning_rule_ids`` in ``result.json``.

Tests cover the two B1b-required runner contracts:

- ``conditioning_spec.json`` is written next to ``prompt.txt`` for every
  attempt where ``conditioning_spec is not None`` — accepted, classify-
  failure, and transport-failure paths all write it (so
  ``attempt_rows.csv`` can compute the four spec hashes on failed
  attempts too).
- ``conditioning_spec.json`` is round-trippable via
  ``spec_from_json_dict`` — all four existing hash helpers produce the
  same digest on the original spec and on the reconstructed one.
- ``result.json`` carries ``validation_warning_rule_ids`` populated from
  ``parsed.materialized.validation_log[].rule_id`` on every attempt that
  reached replay — accepted plus post-replay failures
  (``future_constraints_audit``, ``terminal_contract``). Empty list when
  replay succeeded with zero warnings. ``None`` on attempts that didn't
  reach replay (transport, parse, baseline/ledger validation,
  observed_prefix_audit, replay itself).
- When ``conditioning_spec is None`` (legacy path), no
  ``conditioning_spec.json`` is written.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

from synthonccases.generate.providers import GenerationResult
from synthonccases.oncollm.conditioning_audit import (
    ConditioningSpec,
    FutureEventConstraint,
    FutureEventSelector,
    ObservedPrefix,
    compute_baseline_hash,
    compute_conditioning_spec_hash,
    compute_future_constraint_hash,
    compute_observed_prefix_hash,
    compute_pinned_paths,
    spec_from_json_dict,
)
from synthonccases.oncollm.poc_runner import (
    ModelSpec,
    run_one_attempt,
)
from synthonccases.oncollm.replay import LedgerEvent


REPO_ROOT = Path(__file__).resolve().parents[2]
# Pair fixtures kept together for back-compat with the existing
# ``replay_spike`` paired baseline + ledger that the rev5 POC runner tests
# use; that pair is the canonical accepted-attempt fixture.
BRCA_BASELINE_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "replay_spike" / "her2_brca_baseline_001.json"
)
LEDGER_001_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "replay_spike" / "her2_brca_ledger_001.json"
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _stub_generation(text: str) -> GenerationResult:
    return GenerationResult(
        text=text,
        input_tokens=4000,
        output_tokens=600,
        reasoning_tokens=0,
        model_id="stub-model",
        response_model="stub-model",
        response_id="stub-resp",
        reported_cost=0.001,
        max_tokens_sent=16000,
        max_tokens_param="max_tokens",
        reasoning_mode="off",
        duration_s=0.1,
    )


def _stub_generate_fn(*responses: GenerationResult | Exception) -> Callable:
    iterator: Iterator = iter(responses)

    def _fn(model: ModelSpec, system: str, user: str):
        nxt = next(iterator)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    return _fn


def _load_baseline() -> dict[str, Any]:
    return json.loads(BRCA_BASELINE_PATH.read_text(encoding="utf-8"))


def _load_ledger() -> list[dict[str, Any]]:
    return json.loads(LEDGER_001_PATH.read_text(encoding="utf-8"))


def _valid_response_text() -> str:
    return json.dumps({"baseline_patch": {}, "ledger": _load_ledger()})


def _stub_spec_baseline_only() -> ConditioningSpec:
    seed = _load_baseline()
    return ConditioningSpec(
        baseline_seed=seed,
        pinned_paths=compute_pinned_paths(seed),
        observed_prefix=None,
        future_event_constraints=(),
        free_text_operator_constraints="",
    )


def _stub_spec_with_prefix_and_constraint() -> ConditioningSpec:
    seed = _load_baseline()
    prefix_event = LedgerEvent.model_validate(
        {
            "date": "2024-04-15",
            "type": "systemic_treatment_start",
            "regimen": {
                "label": "THP",
                "components": ["trastuzumab", "pertuzumab", "docetaxel"],
            },
        }
    )
    constraint = FutureEventConstraint(
        constraint_id="post_cutoff_progression",
        selector=FutureEventSelector(
            event_type="progression", ordinal_after_cutoff="first"
        ),
        constraint={"event_type_is_one_of": ["progression"]},
    )
    return ConditioningSpec(
        baseline_seed=seed,
        pinned_paths=compute_pinned_paths(seed),
        observed_prefix=ObservedPrefix(
            events=(prefix_event,),
            cutoff_date=date(2024, 7, 15),
        ),
        future_event_constraints=(constraint,),
        free_text_operator_constraints="prefer oral therapy",
    )


def _run_attempt(
    tmp_path: Path,
    *,
    response_text: str | Exception,
    conditioning_spec: ConditioningSpec | None,
) -> tuple[Path, Any]:
    attempt_dir = tmp_path / "attempt_01"
    seed = _load_baseline()
    response: GenerationResult | Exception
    if isinstance(response_text, Exception):
        response = response_text
    else:
        response = _stub_generation(response_text)
    result = run_one_attempt(
        attempt_index=1,
        attempt_dir=attempt_dir,
        rendered_prompt="prompt body",
        baseline_seed=seed,
        pinned_paths=compute_pinned_paths(seed),
        seed_is_complete=True,
        policy=None,
        model=ModelSpec(id="stub/model-a", short="stub_a"),
        generate_fn=_stub_generate_fn(response),
        conditioning_spec=conditioning_spec,
    )
    return attempt_dir, result


def _read_result_json(attempt_dir: Path) -> dict[str, Any]:
    return json.loads((attempt_dir / "result.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# conditioning_spec.json is written per attempt
# ---------------------------------------------------------------------------


def test_accepted_attempt_writes_conditioning_spec_json(tmp_path: Path) -> None:
    spec = _stub_spec_baseline_only()
    attempt_dir, result = _run_attempt(
        tmp_path, response_text=_valid_response_text(), conditioning_spec=spec
    )
    assert result.accepted is True
    spec_path = attempt_dir / "conditioning_spec.json"
    assert spec_path.exists()


def test_classify_failure_attempt_writes_conditioning_spec_json(
    tmp_path: Path,
) -> None:
    # Garbage JSON → parse failure inside classify_response.
    spec = _stub_spec_baseline_only()
    attempt_dir, result = _run_attempt(
        tmp_path, response_text="not valid json", conditioning_spec=spec
    )
    assert result.accepted is False
    assert result.failure_kind == "parse"
    assert (attempt_dir / "conditioning_spec.json").exists()


def test_transport_failure_attempt_writes_conditioning_spec_json(
    tmp_path: Path,
) -> None:
    spec = _stub_spec_baseline_only()
    attempt_dir, result = _run_attempt(
        tmp_path,
        response_text=RuntimeError("provider down"),
        conditioning_spec=spec,
    )
    assert result.accepted is False
    assert result.failure_kind == "transport_or_runtime"
    assert (attempt_dir / "conditioning_spec.json").exists()


def test_forced_off_response_with_reasoning_tokens_is_rejected(tmp_path: Path) -> None:
    seed = _load_baseline()
    generation = replace(
        _stub_generation(_valid_response_text()),
        reasoning_mode="forced_off",
        reasoning_tokens=17,
    )
    attempt_dir = tmp_path / "attempt_01"
    result = run_one_attempt(
        attempt_index=1,
        attempt_dir=attempt_dir,
        rendered_prompt="prompt body",
        baseline_seed=seed,
        pinned_paths=compute_pinned_paths(seed),
        seed_is_complete=True,
        policy=None,
        model=ModelSpec(id="stub/model-a", short="stub_a"),
        generate_fn=_stub_generate_fn(generation),
        conditioning_spec=_stub_spec_baseline_only(),
    )

    assert result.accepted is False
    assert result.failure_kind == "provider_parameter_violation"
    assert result.usage["reasoning_tokens"] == 17
    failure = json.loads((attempt_dir / "failed_attempt.json").read_text())
    assert failure["failure_kind"] == "provider_parameter_violation"
    assert not (attempt_dir / "parsed_ledger.json").exists()


def test_no_spec_no_conditioning_spec_json(tmp_path: Path) -> None:
    """Legacy path (``conditioning_spec=None``) must NOT write spec.json."""
    attempt_dir, result = _run_attempt(
        tmp_path, response_text=_valid_response_text(), conditioning_spec=None
    )
    assert result.accepted is True
    assert not (attempt_dir / "conditioning_spec.json").exists()


# ---------------------------------------------------------------------------
# conditioning_spec.json round-trips back to the same hashes
# ---------------------------------------------------------------------------


def test_spec_json_round_trip_preserves_all_four_hashes(tmp_path: Path) -> None:
    spec = _stub_spec_with_prefix_and_constraint()
    attempt_dir, _ = _run_attempt(
        tmp_path, response_text=_valid_response_text(), conditioning_spec=spec
    )
    spec_path = attempt_dir / "conditioning_spec.json"
    on_disk = json.loads(spec_path.read_text(encoding="utf-8"))
    reconstructed = spec_from_json_dict(on_disk)
    assert compute_baseline_hash(reconstructed) == compute_baseline_hash(spec)
    assert (
        compute_observed_prefix_hash(reconstructed)
        == compute_observed_prefix_hash(spec)
    )
    assert (
        compute_future_constraint_hash(reconstructed)
        == compute_future_constraint_hash(spec)
    )
    assert (
        compute_conditioning_spec_hash(reconstructed)
        == compute_conditioning_spec_hash(spec)
    )


# ---------------------------------------------------------------------------
# validation_warning_rule_ids on result.json
# ---------------------------------------------------------------------------


def test_accepted_attempt_carries_validation_warning_rule_ids_list(
    tmp_path: Path,
) -> None:
    """Accepted: list[str] (possibly empty), same length as validation_log_count."""
    spec = _stub_spec_baseline_only()
    attempt_dir, result = _run_attempt(
        tmp_path, response_text=_valid_response_text(), conditioning_spec=spec
    )
    payload = _read_result_json(attempt_dir)
    rule_ids = payload["validation_warning_rule_ids"]
    assert isinstance(rule_ids, list)
    assert all(isinstance(r, str) for r in rule_ids)
    assert len(rule_ids) == payload["validation_log_count"]
    # AttemptResult holds a tuple of the same content.
    assert result.validation_warning_rule_ids is not None
    assert tuple(rule_ids) == result.validation_warning_rule_ids


def test_classify_failure_attempt_validation_warning_rule_ids_null(
    tmp_path: Path,
) -> None:
    spec = _stub_spec_baseline_only()
    attempt_dir, result = _run_attempt(
        tmp_path, response_text="not valid json", conditioning_spec=spec
    )
    payload = _read_result_json(attempt_dir)
    assert payload["validation_warning_rule_ids"] is None
    assert result.validation_warning_rule_ids is None


def test_transport_failure_attempt_validation_warning_rule_ids_null(
    tmp_path: Path,
) -> None:
    spec = _stub_spec_baseline_only()
    attempt_dir, result = _run_attempt(
        tmp_path,
        response_text=RuntimeError("provider down"),
        conditioning_spec=spec,
    )
    payload = _read_result_json(attempt_dir)
    assert payload["validation_warning_rule_ids"] is None
    assert result.validation_warning_rule_ids is None


def _stub_spec_with_must_not_occur_response_constraint() -> ConditioningSpec:
    """Spec with a future constraint the fixture ledger will violate.

    The fixture ``her2_brca_ledger_001.json`` contains at least one
    ``response`` event, so a ``must_not_occur=True`` constraint on
    ``event_type='response'`` is guaranteed to fail post-replay.
    """
    seed = _load_baseline()
    constraint = FutureEventConstraint(
        constraint_id="no_response_events",
        selector=FutureEventSelector(event_type="response"),
        constraint={"must_not_occur": True},
    )
    return ConditioningSpec(
        baseline_seed=seed,
        pinned_paths=compute_pinned_paths(seed),
        observed_prefix=None,
        future_event_constraints=(constraint,),
        free_text_operator_constraints="",
    )


def test_post_replay_failure_preserves_validation_warning_rule_ids(
    tmp_path: Path,
) -> None:
    """``future_constraints_audit`` failure — replay succeeded, so warnings
    are recoverable. Result.json must carry the warning-id list (possibly
    empty) rather than null. Defends against silent loss of warning IDs
    on replay-successful-but-audit-failed attempts (which the registry
    requires for ``attempt_rows.validation_warnings_joined``)."""
    spec = _stub_spec_with_must_not_occur_response_constraint()
    attempt_dir, result = _run_attempt(
        tmp_path, response_text=_valid_response_text(), conditioning_spec=spec
    )
    assert result.accepted is False
    assert result.failure_kind == "future_constraints_audit"
    payload = _read_result_json(attempt_dir)
    rule_ids = payload["validation_warning_rule_ids"]
    # Critical: a LIST (possibly empty), never None for a post-replay failure.
    assert isinstance(rule_ids, list)
    assert all(isinstance(r, str) for r in rule_ids)
    # Event count + snapshot count are also recoverable post-replay.
    assert payload["event_count"] is not None
    assert payload["snapshot_count"] is not None
    assert payload["validation_log_count"] == len(rule_ids)
