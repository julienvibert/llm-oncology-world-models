"""Gate B1b — ``discover_attempts`` + ``build_attempt_rows`` semantics.

Locks: per-attempt row emission (accepted + failed), strict
``result.json`` handling, ``tumor_oncotree_code`` recovery from accepted
sibling vs ``__unknown_tumor__`` sentinel, relative ``attempt_dir``
path, spec-hash population per-attempt, ``validation_warning_rule_ids``
flow, and the cost/usage/latency reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from synthonccases.oncollm.derivation import (
    DEFAULT_UNKNOWN_TUMOR,
    DerivationDataLossError,
    LEGACY_NO_SPEC_SENTINEL,
    _attempt_latency_ms,
    _attempt_tokens,
    _attempt_validation_warnings_joined,
    _failure_family,
    _failure_rule_id,
    build_attempt_rows,
    discover_attempts,
)


# ---------------------------------------------------------------------------
# Tiny synth run-root builder for attempt_rows scenarios
# ---------------------------------------------------------------------------


def _write_traj(
    trajs_dir: Path,
    traj_dir_name: str,
    *,
    accepted: bool,
    attempts_to_accept: int | None,
    attempts: list[dict[str, Any]],
    materialized_for_attempt: int | None = None,
    failure_kind_traj: str | None = None,
    cell_override: dict[str, Any] | None = None,
) -> Path:
    """Write a one-trajectory layout under trajs_dir/<traj_dir_name>/.

    Each entry in ``attempts`` is a result.json payload (dict). When
    ``materialized_for_attempt`` is set, also writes a tiny
    materialized_trajectory.json under that attempt's dir (used by
    discover_attempts to recover tumor_oncotree_code).
    """
    td = trajs_dir / traj_dir_name
    td.mkdir(parents=True, exist_ok=True)
    cell = {
        "condition_id": "c1", "policy_id": "A_THP",
        "model_id": "m/x", "model_short": "x",
        "conditioning_label": "cl", "conditioning_hash": "h",
    }
    if cell_override:
        cell.update(cell_override)
    traj_result: dict[str, Any] = {
        "trajectory_dir": traj_dir_name,
        "accepted": accepted,
        "attempts_to_accept": attempts_to_accept,
        "future_index": 0,
        "cell": cell,
    }
    if failure_kind_traj is not None:
        traj_result["failure_kind"] = failure_kind_traj
    (td / "trajectory_result.json").write_text(json.dumps(traj_result))
    for i, result in enumerate(attempts, start=1):
        adir = td / f"attempt_{i:02d}"
        adir.mkdir()
        full_result = {"attempt_index": i, **result}
        (adir / "result.json").write_text(json.dumps(full_result))
        (adir / "prompt.txt").write_text(f"prompt for {traj_dir_name} attempt {i}")
        if full_result.get("accepted") or full_result.get("failure_kind") != "transport_or_runtime":
            (adir / "raw_response.txt").write_text("{}")
        if materialized_for_attempt == i:
            (adir / "materialized_trajectory.json").write_text(json.dumps({
                "baseline": {"identity_card": {"diagnosis": {
                    "oncotree_code": "BRCA",
                }}},
                "ledger": [], "snapshots": [], "treatment_history": [],
            }))
    return td


def _basic_accepted_attempt_result(**overrides: Any) -> dict[str, Any]:
    base = {
        "accepted": True, "failure_kind": None, "failure_message": None,
        "event_count": 8, "snapshot_count": 5, "validation_log_count": 0,
        "validation_warning_rule_ids": [],
        "cost_usd": 0.0019,
        "usage": {
            "input_tokens": 1200,
            "output_tokens": 800,
            "reasoning_tokens": 11,
            "response_model": "provider/model-routed",
            "response_id": "resp_123",
            "retries": 1,
            "max_tokens_sent": 16384,
            "max_tokens_param": "max_tokens",
            "reasoning_mode": "off",
            "temperature_sent": 1.0,
            "top_p_sent": None,
            "seed_sent": None,
            "finish_reason": "stop",
            "duration_s": 3.2,
        },
    }
    base.update(overrides)
    return base


def _basic_failed_attempt_result(**overrides: Any) -> dict[str, Any]:
    base = {
        "accepted": False, "failure_kind": "parse",
        "failure_message": "bad json",
        "event_count": None, "snapshot_count": None,
        "validation_log_count": None, "validation_warning_rule_ids": None,
        "cost_usd": 0.0012,
        "usage": {"input_tokens": 1000, "output_tokens": 50, "duration_s": 1.5},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Per-helper unit tests
# ---------------------------------------------------------------------------


def test_validation_warnings_joined_list_to_semicolon_string() -> None:
    result = {"validation_warning_rule_ids": ["R001", "R042", "R007"]}
    assert _attempt_validation_warnings_joined(result) == "R001;R042;R007"


def test_validation_warnings_joined_empty_list_is_null() -> None:
    """Empty list ('no warnings on a replayed attempt') maps to None
    so the column distinguishes legacy / unreached-replay null from
    no-warnings null only via the accompanying accepted/failure_kind
    columns (per the column doc)."""
    assert _attempt_validation_warnings_joined({"validation_warning_rule_ids": []}) is None


def test_validation_warnings_joined_null_field_is_null() -> None:
    assert _attempt_validation_warnings_joined({"validation_warning_rule_ids": None}) is None


def test_validation_warnings_joined_missing_field_is_null() -> None:
    assert _attempt_validation_warnings_joined({}) is None


def test_validation_warnings_joined_non_sequence_is_null() -> None:
    """Defensive: garbage non-sequence value → None, not crash."""
    assert _attempt_validation_warnings_joined({"validation_warning_rule_ids": "R001"}) is None


def test_failure_rule_id_extracts_replay_rule_prefix() -> None:
    result = {
        "failure_kind": "replay",
        "failure_message": (
            "TREATMENT_LIFECYCLE: systemic_treatment_start cannot occur "
            "while treatment is active"
        ),
    }
    assert _failure_rule_id(result) == "TREATMENT_LIFECYCLE"
    assert _failure_family(result) == "treatment_lifecycle"


def test_failure_rule_id_is_narrow_for_non_replay_messages() -> None:
    result = {
        "failure_kind": "ledger_validation",
        "failure_message": "ValidationError: 2 validation errors for LedgerEvent",
    }
    assert _failure_rule_id(result) is None
    assert _failure_family(result) == "ledger_schema"


def test_failure_family_terminal_contract_without_replay_prefix() -> None:
    result = {
        "failure_kind": "terminal_contract",
        "failure_message": "terminal_contract='death_only' but trajectory ends in 'last_followup'",
    }
    assert _failure_rule_id(result) == "TERMINAL_CONTRACT"
    assert _failure_family(result) == "terminality"


def test_attempt_latency_ms_from_duration_s() -> None:
    assert _attempt_latency_ms({"usage": {"duration_s": 3.2}}) == 3200


def test_attempt_latency_ms_missing_usage_is_null() -> None:
    assert _attempt_latency_ms({}) is None
    assert _attempt_latency_ms({"usage": {}}) is None


def test_attempt_latency_ms_bool_rejected() -> None:
    """Pythonic gotcha: bool is a subclass of int. Reject explicitly."""
    assert _attempt_latency_ms({"usage": {"duration_s": True}}) is None


def test_attempt_tokens_reads_usage() -> None:
    result = {"usage": {"input_tokens": 1200, "output_tokens": 800}}
    assert _attempt_tokens(result, "input_tokens") == 1200
    assert _attempt_tokens(result, "output_tokens") == 800


# ---------------------------------------------------------------------------
# discover_attempts — strictness on result.json
# ---------------------------------------------------------------------------


def test_discover_attempts_raises_on_malformed_result_json(tmp_path: Path) -> None:
    """attempt_rows powers QC denominators; silent skip on malformed
    result.json would undercount. Must raise DerivationDataLossError."""
    trajs = tmp_path / "trajectories"
    _write_traj(
        trajs, "traj_a", accepted=True, attempts_to_accept=1,
        attempts=[_basic_accepted_attempt_result()],
        materialized_for_attempt=1,
    )
    # Corrupt the result.json.
    (trajs / "traj_a" / "attempt_01" / "result.json").write_text("not json {")
    with pytest.raises(DerivationDataLossError, match="result.json"):
        discover_attempts(tmp_path)


def test_discover_attempts_raises_on_non_object_result_json(tmp_path: Path) -> None:
    trajs = tmp_path / "trajectories"
    _write_traj(
        trajs, "traj_a", accepted=True, attempts_to_accept=1,
        attempts=[_basic_accepted_attempt_result()],
        materialized_for_attempt=1,
    )
    (trajs / "traj_a" / "attempt_01" / "result.json").write_text(json.dumps([1, 2, 3]))
    with pytest.raises(DerivationDataLossError, match="malformed result.json"):
        discover_attempts(tmp_path)


def test_discover_attempts_skips_attempt_dir_without_result_json(
    tmp_path: Path,
) -> None:
    """An attempt_* dir whose result.json file is ABSENT is an interrupted
    attempt (killed mid-write / stale second writer): it produced no
    classifiable outcome, so it is skipped with a warning rather than
    aborting derivation. A present-but-malformed result.json still raises
    (covered above)."""
    trajs = tmp_path / "trajectories"
    _write_traj(
        trajs, "traj_a", accepted=True, attempts_to_accept=1,
        attempts=[_basic_accepted_attempt_result()],
        materialized_for_attempt=1,
    )
    # Add a second attempt dir that never wrote a result.json.
    interrupted = trajs / "traj_a" / "attempt_02"
    interrupted.mkdir()
    (interrupted / "prompt.txt").write_text("interrupted prompt")

    with pytest.warns(UserWarning, match="no result.json"):
        attempts = discover_attempts(tmp_path)
    # Only the completed attempt is surfaced; the interrupted one is dropped.
    assert len(attempts) == 1
    assert attempts[0].attempt_index == 1


def test_discover_attempts_skips_dry_run_trajectories(tmp_path: Path) -> None:
    """Mirror discover_trajectories dry-run skip."""
    trajs = tmp_path / "trajectories"
    _write_traj(
        trajs, "traj_dry", accepted=False, attempts_to_accept=None,
        attempts=[_basic_failed_attempt_result()],
        failure_kind_traj="dry_run",
    )
    assert discover_attempts(tmp_path) == []


# ---------------------------------------------------------------------------
# build_attempt_rows — mixed-outcome trajectory + tumor recovery
# ---------------------------------------------------------------------------


def test_attempt_rows_failed_sibling_inherits_tumor_from_accepted(
    tmp_path: Path,
) -> None:
    """Failed attempt on a trajectory that eventually accepted inherits
    the tumor_oncotree_code from the accepted sibling's materialized
    baseline."""
    trajs = tmp_path / "trajectories"
    _write_traj(
        trajs, "traj_mixed", accepted=True, attempts_to_accept=2,
        attempts=[
            _basic_failed_attempt_result(),
            _basic_accepted_attempt_result(),
        ],
        materialized_for_attempt=2,
    )
    attempts = discover_attempts(tmp_path)
    df = build_attempt_rows(attempts)
    assert len(df) == 2
    # Both rows have tumor=BRCA (recovered from accepted sibling).
    assert all(t == "BRCA" for t in df["tumor_oncotree_code"])
    # Per-row accepted reflects per-attempt outcome.
    assert list(df["accepted"]) == [False, True]
    assert list(df["failure_kind"].fillna("__null__")) == ["parse", "__null__"]


def test_attempt_rows_all_failed_trajectory_uses_unknown_tumor_sentinel(
    tmp_path: Path,
) -> None:
    """No accepted sibling → DEFAULT_UNKNOWN_TUMOR sentinel."""
    trajs = tmp_path / "trajectories"
    _write_traj(
        trajs, "traj_all_failed", accepted=False, attempts_to_accept=None,
        attempts=[_basic_failed_attempt_result(failure_kind="transport_or_runtime")],
    )
    attempts = discover_attempts(tmp_path)
    df = build_attempt_rows(attempts)
    assert len(df) == 1
    assert df.iloc[0]["tumor_oncotree_code"] == DEFAULT_UNKNOWN_TUMOR


def test_attempt_rows_attempt_dir_is_relative_path(tmp_path: Path) -> None:
    """attempt_dir is stored relative to run_root (portable across machines)."""
    trajs = tmp_path / "trajectories"
    _write_traj(
        trajs, "traj_a", accepted=True, attempts_to_accept=1,
        attempts=[_basic_accepted_attempt_result()],
        materialized_for_attempt=1,
    )
    attempts = discover_attempts(tmp_path.resolve())  # absolute root
    df = build_attempt_rows(attempts)
    assert df.iloc[0]["attempt_dir"] == "trajectories/traj_a/attempt_01"


def test_attempt_rows_spec_hashes_sentinel_when_no_spec(tmp_path: Path) -> None:
    """No conditioning_spec.json on disk → 4 sentinel hashes."""
    trajs = tmp_path / "trajectories"
    _write_traj(
        trajs, "traj_a", accepted=True, attempts_to_accept=1,
        attempts=[_basic_accepted_attempt_result()],
        materialized_for_attempt=1,
    )
    attempts = discover_attempts(tmp_path)
    df = build_attempt_rows(attempts)
    assert df.iloc[0]["spec_baseline_hash"] == LEGACY_NO_SPEC_SENTINEL
    assert df.iloc[0]["spec_conditioning_hash"] == LEGACY_NO_SPEC_SENTINEL


def test_attempt_rows_cost_tokens_latency_populated(tmp_path: Path) -> None:
    trajs = tmp_path / "trajectories"
    _write_traj(
        trajs, "traj_a", accepted=True, attempts_to_accept=1,
        attempts=[_basic_accepted_attempt_result()],
        materialized_for_attempt=1,
    )
    attempts = discover_attempts(tmp_path)
    df = build_attempt_rows(attempts)
    r = df.iloc[0]
    assert r["cost_usd"] == 0.0019
    assert r["tokens_input"] == 1200
    assert r["tokens_output"] == 800
    assert r["latency_ms"] == 3200


def test_attempt_rows_provider_metadata_and_artifact_paths_populated(
    tmp_path: Path,
) -> None:
    trajs = tmp_path / "trajectories"
    _write_traj(
        trajs, "traj_a", accepted=True, attempts_to_accept=1,
        attempts=[_basic_accepted_attempt_result()],
        materialized_for_attempt=1,
    )
    attempts = discover_attempts(tmp_path)
    df = build_attempt_rows(attempts)
    r = df.iloc[0]
    assert r["prompt_path"] == "trajectories/traj_a/attempt_01/prompt.txt"
    assert r["raw_response_path"] == "trajectories/traj_a/attempt_01/raw_response.txt"
    assert r["result_path"] == "trajectories/traj_a/attempt_01/result.json"
    assert r["failure_message"] is None
    assert r["failure_rule_id"] is None
    assert r["failure_family"] is None
    assert r["reasoning_tokens"] == 11
    assert r["response_model"] == "provider/model-routed"
    assert r["response_id"] == "resp_123"
    assert r["provider_retries"] == 1
    assert r["max_tokens_sent"] == 16384
    assert r["max_tokens_param"] == "max_tokens"
    assert r["reasoning_mode"] == "off"
    assert r["temperature_sent"] == 1.0
    assert r["finish_reason"] == "stop"


def test_attempt_rows_failure_rule_and_family_populated_for_replay(
    tmp_path: Path,
) -> None:
    trajs = tmp_path / "trajectories"
    _write_traj(
        trajs, "traj_a", accepted=False, attempts_to_accept=None,
        attempts=[_basic_failed_attempt_result(
            failure_kind="replay",
            failure_message=(
                "CLONE_REFERENCE: disease delta references unknown clone 'C1'"
            ),
        )],
    )
    attempts = discover_attempts(tmp_path)
    df = build_attempt_rows(attempts)
    r = df.iloc[0]
    assert r["failure_kind"] == "replay"
    assert r["failure_rule_id"] == "CLONE_REFERENCE"
    assert r["failure_family"] == "clone_state"


def test_attempt_rows_validation_warnings_joined_populated(tmp_path: Path) -> None:
    """Accepted attempt with warnings → semicolon-joined cell."""
    trajs = tmp_path / "trajectories"
    _write_traj(
        trajs, "traj_a", accepted=True, attempts_to_accept=1,
        attempts=[_basic_accepted_attempt_result(
            validation_log_count=2, validation_warning_rule_ids=["R001", "R042"]
        )],
        materialized_for_attempt=1,
    )
    attempts = discover_attempts(tmp_path)
    df = build_attempt_rows(attempts)
    assert df.iloc[0]["validation_warnings_joined"] == "R001;R042"


def test_attempt_rows_failed_attempt_null_warnings(tmp_path: Path) -> None:
    """Failed attempt that didn't reach replay → null warnings."""
    trajs = tmp_path / "trajectories"
    _write_traj(
        trajs, "traj_a", accepted=False, attempts_to_accept=None,
        attempts=[_basic_failed_attempt_result()],
    )
    attempts = discover_attempts(tmp_path)
    df = build_attempt_rows(attempts)
    import pandas as pd
    assert pd.isna(df.iloc[0]["validation_warnings_joined"])
