"""Gate B1b — ``discover_trajectories`` loads ``conditioning_spec.json``
per accepted attempt and re-runs ``future_constraints_audit`` for
``constraint_match_ids``.

Tests the two extensions to ``discover_trajectories``:

- When ``<attempt_dir>/conditioning_spec.json`` is absent (legacy
  sentinel-backfill path), ``TrajectoryArtifact.conditioning_spec`` and
  ``.constraint_results`` are both ``None``.
- When the spec file is present, the loaded spec round-trips back to a
  ``ConditioningSpec`` whose four hashes match the original, and
  ``constraint_results`` is populated from
  ``future_constraints_audit(spec, materialized.ledger)``.
- YAML fallback (``conditioning_spec.yaml``) works for the offline-
  prefix-fixture path.
- A malformed spec file raises ``DerivationDataLossError`` — silent
  acceptance would produce wrong hashes downstream.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from synthonccases.generate.providers import GenerationResult
from synthonccases.oncollm.conditioning_audit import (
    EMPTY_PREFIX_SENTINEL,
    ConditioningSpec,
    FutureEventConstraint,
    FutureEventSelector,
    compute_baseline_hash,
    compute_conditioning_spec_hash,
    compute_future_constraint_hash,
    compute_observed_prefix_hash,
    compute_pinned_paths,
    spec_to_json_dict,
)
from synthonccases.oncollm.derivation import (
    DerivationDataLossError,
    discover_trajectories,
)
from synthonccases.oncollm.poc_runner import (
    ModelSpec,
    run_one_attempt,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
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


def _spec_baseline_only() -> ConditioningSpec:
    seed = _load_baseline()
    return ConditioningSpec(
        baseline_seed=seed,
        pinned_paths=compute_pinned_paths(seed),
        observed_prefix=None,
        future_event_constraints=(),
        free_text_operator_constraints="",
    )


def _spec_with_satisfiable_constraint() -> ConditioningSpec:
    """Spec whose constraint the fixture ledger satisfies (so the audit
    passes when classified — but for `discover_trajectories` we re-run it
    independently of the runner's accept/reject decision)."""
    seed = _load_baseline()
    constraint = FutureEventConstraint(
        constraint_id="any_response",
        selector=FutureEventSelector(event_type="response", ordinal="first"),
        constraint={"event_type_is_one_of": ["response"]},
    )
    return ConditioningSpec(
        baseline_seed=seed,
        pinned_paths=compute_pinned_paths(seed),
        observed_prefix=None,
        future_event_constraints=(constraint,),
        free_text_operator_constraints="",
    )


def _build_minimal_run_root(
    tmp_path: Path,
    *,
    conditioning_spec: ConditioningSpec | None,
) -> Path:
    """Use ``run_one_attempt`` to materialize a one-trajectory, one-attempt
    run root that ``discover_trajectories`` can walk."""
    run_root = tmp_path / "run"
    traj_dir = run_root / "trajectories" / "traj_001"
    attempt_dir = traj_dir / "attempt_01"
    seed = _load_baseline()
    result = run_one_attempt(
        attempt_index=1,
        attempt_dir=attempt_dir,
        rendered_prompt="prompt body",
        baseline_seed=seed,
        pinned_paths=compute_pinned_paths(seed),
        seed_is_complete=True,
        policy=None,
        model=ModelSpec(id="stub/model-a", short="stub_a"),
        generate_fn=_stub_generate_fn(_stub_generation(_valid_response_text())),
        conditioning_spec=conditioning_spec,
    )
    assert result.accepted, "fixture must produce an accepted attempt"
    # Write a minimal trajectory_result.json so discover_trajectories
    # recognizes the trajectory.
    (traj_dir / "trajectory_result.json").write_text(
        json.dumps(
            {
                "trajectory_dir": "traj_001",
                "accepted": True,
                "attempts_to_accept": 1,
                "cell": {
                    "condition_id": "cond_a",
                    "policy_id": "pol_a",
                    "model_id": "stub/model-a",
                    "model_short": "stub_a",
                    "conditioning_label": "test",
                    "conditioning_hash": "abc123",
                },
                "future_index": 0,
            }
        ),
        encoding="utf-8",
    )
    return run_root


# ---------------------------------------------------------------------------
# No spec on disk → both fields are None (legacy sentinel-backfill path)
# ---------------------------------------------------------------------------


def test_discover_trajectories_no_spec_yields_none(tmp_path: Path) -> None:
    run_root = _build_minimal_run_root(tmp_path, conditioning_spec=None)
    # The runner did NOT write conditioning_spec.json (spec was None).
    assert not (
        run_root / "trajectories" / "traj_001" / "attempt_01" / "conditioning_spec.json"
    ).exists()
    artifacts = discover_trajectories(run_root)
    assert len(artifacts) == 1
    assert artifacts[0].conditioning_spec is None
    assert artifacts[0].constraint_results is None


# ---------------------------------------------------------------------------
# JSON spec on disk → loaded + constraint_results populated
# ---------------------------------------------------------------------------


def test_discover_trajectories_loads_spec_and_constraint_results(
    tmp_path: Path,
) -> None:
    spec = _spec_with_satisfiable_constraint()
    run_root = _build_minimal_run_root(tmp_path, conditioning_spec=spec)
    artifacts = discover_trajectories(run_root)
    assert len(artifacts) == 1
    art = artifacts[0]
    assert art.conditioning_spec is not None
    # Round-trip preserves all four hashes.
    assert compute_baseline_hash(art.conditioning_spec) == compute_baseline_hash(spec)
    assert (
        compute_observed_prefix_hash(art.conditioning_spec)
        == compute_observed_prefix_hash(spec)
    )
    assert (
        compute_future_constraint_hash(art.conditioning_spec)
        == compute_future_constraint_hash(spec)
    )
    assert (
        compute_conditioning_spec_hash(art.conditioning_spec)
        == compute_conditioning_spec_hash(spec)
    )
    # constraint_results is a tuple with one ConstraintMatch per spec
    # constraint, in input order.
    assert art.constraint_results is not None
    assert len(art.constraint_results) == 1
    assert art.constraint_results[0].constraint_id == "any_response"


def test_discover_trajectories_loads_baseline_only_spec(tmp_path: Path) -> None:
    """Spec with zero future_event_constraints still populates
    conditioning_spec (so spec_* hashes can be computed downstream)
    and yields an empty constraint_results tuple."""
    spec = _spec_baseline_only()
    run_root = _build_minimal_run_root(tmp_path, conditioning_spec=spec)
    artifacts = discover_trajectories(run_root)
    art = artifacts[0]
    assert art.conditioning_spec is not None
    assert art.constraint_results == ()


# ---------------------------------------------------------------------------
# YAML fallback works
# ---------------------------------------------------------------------------


def test_discover_trajectories_loads_yaml_wrapper_shape(tmp_path: Path) -> None:
    """The offline-prefix-fixture path ships
    ``conditioning_spec.yaml`` with a ``{conditioning: {...}}`` wrapper
    (the upstream config-loader shape). The loader must lower it via
    ``lower_conditioning_cell`` — NOT pass it directly to
    ``spec_from_json_dict``, which would silently produce an empty spec.

    Uses the real LUAD offline fixture YAML to defend against shape
    drift between B1b and the fixture builder.
    """
    real_fixture_yaml = (
        REPO_ROOT
        / "tests"
        / "fixtures"
        / "prefix_eval"
        / "offline"
        / "luad_tumor_only__flash_lite__f01__split_3"
        / "conditioning_spec.yaml"
    )
    assert real_fixture_yaml.exists(), (
        "real offline-prefix fixture must exist — defends against the "
        "wrapper-shape regression"
    )
    run_root = _build_minimal_run_root(tmp_path, conditioning_spec=None)
    attempt_dir = run_root / "trajectories" / "traj_001" / "attempt_01"
    shutil.copy(real_fixture_yaml, attempt_dir / "conditioning_spec.yaml")

    artifacts = discover_trajectories(run_root)
    art = artifacts[0]
    assert art.conditioning_spec is not None
    # Critical: the wrapper-aware path produces a NON-empty spec.
    # spec_from_json_dict on the wrapped shape would silently yield a
    # spec with no prefix and EMPTY_PREFIX_SENTINEL hash.
    assert art.conditioning_spec.observed_prefix is not None
    assert len(art.conditioning_spec.observed_prefix.events) > 0
    assert compute_observed_prefix_hash(art.conditioning_spec) != EMPTY_PREFIX_SENTINEL
    # Baseline seed populated too (tumor-only-style seed in the LUAD fixture).
    assert art.conditioning_spec.baseline_seed != {}


def test_discover_trajectories_loads_direct_yaml_spec(tmp_path: Path) -> None:
    """A YAML file without the ``conditioning`` wrapper is treated as
    direct serialized-spec form (symmetric inverse of
    ``spec_to_json_dict``)."""
    spec = _spec_baseline_only()
    run_root = _build_minimal_run_root(tmp_path, conditioning_spec=None)
    attempt_dir = run_root / "trajectories" / "traj_001" / "attempt_01"
    (attempt_dir / "conditioning_spec.yaml").write_text(
        yaml.safe_dump(spec_to_json_dict(spec)), encoding="utf-8"
    )
    artifacts = discover_trajectories(run_root)
    art = artifacts[0]
    assert art.conditioning_spec is not None
    assert compute_baseline_hash(art.conditioning_spec) == compute_baseline_hash(spec)


# ---------------------------------------------------------------------------
# Malformed spec → DerivationDataLossError (silent acceptance would
# produce wrong hashes downstream)
# ---------------------------------------------------------------------------


def test_discover_trajectories_raises_on_malformed_spec_json(
    tmp_path: Path,
) -> None:
    run_root = _build_minimal_run_root(tmp_path, conditioning_spec=None)
    attempt_dir = run_root / "trajectories" / "traj_001" / "attempt_01"
    (attempt_dir / "conditioning_spec.json").write_text(
        "not valid json {", encoding="utf-8"
    )
    with pytest.raises(DerivationDataLossError, match="conditioning_spec.json"):
        discover_trajectories(run_root)


def test_discover_trajectories_raises_on_non_object_spec(tmp_path: Path) -> None:
    run_root = _build_minimal_run_root(tmp_path, conditioning_spec=None)
    attempt_dir = run_root / "trajectories" / "traj_001" / "attempt_01"
    (attempt_dir / "conditioning_spec.json").write_text(
        json.dumps([1, 2, 3]), encoding="utf-8"
    )
    with pytest.raises(DerivationDataLossError, match="not a JSON/YAML object"):
        discover_trajectories(run_root)


# ---------------------------------------------------------------------------
# Typed-ledger strict handling (silent dropping would corrupt
# constraint_match_ids downstream)
# ---------------------------------------------------------------------------


def _corrupt_materialized_ledger(
    run_root: Path, *, corruption: Callable[[dict[str, Any]], None]
) -> None:
    """Mutate the trajectory's materialized_trajectory.json in place."""
    mat_path = (
        run_root
        / "trajectories"
        / "traj_001"
        / "attempt_01"
        / "materialized_trajectory.json"
    )
    payload = json.loads(mat_path.read_text(encoding="utf-8"))
    corruption(payload)
    mat_path.write_text(json.dumps(payload), encoding="utf-8")


def test_discover_trajectories_raises_when_ledger_key_missing(
    tmp_path: Path,
) -> None:
    spec = _spec_baseline_only()
    run_root = _build_minimal_run_root(tmp_path, conditioning_spec=spec)

    def _drop_ledger(payload: dict[str, Any]) -> None:
        payload.pop("ledger", None)

    _corrupt_materialized_ledger(run_root, corruption=_drop_ledger)
    with pytest.raises(DerivationDataLossError, match="no 'ledger' key"):
        discover_trajectories(run_root)


def test_discover_trajectories_raises_when_ledger_not_list(
    tmp_path: Path,
) -> None:
    spec = _spec_baseline_only()
    run_root = _build_minimal_run_root(tmp_path, conditioning_spec=spec)

    def _scalarize_ledger(payload: dict[str, Any]) -> None:
        payload["ledger"] = "this-is-not-a-list"

    _corrupt_materialized_ledger(run_root, corruption=_scalarize_ledger)
    with pytest.raises(DerivationDataLossError, match="ledger is not a list"):
        discover_trajectories(run_root)


def test_discover_trajectories_raises_on_non_mapping_ledger_entry(
    tmp_path: Path,
) -> None:
    spec = _spec_baseline_only()
    run_root = _build_minimal_run_root(tmp_path, conditioning_spec=spec)

    def _inject_non_mapping(payload: dict[str, Any]) -> None:
        payload["ledger"].append("not-a-mapping")

    _corrupt_materialized_ledger(run_root, corruption=_inject_non_mapping)
    with pytest.raises(DerivationDataLossError, match="is not a mapping"):
        discover_trajectories(run_root)


def test_discover_trajectories_raises_on_invalid_materialized_event(
    tmp_path: Path,
) -> None:
    spec = _spec_baseline_only()
    run_root = _build_minimal_run_root(tmp_path, conditioning_spec=spec)

    def _replace_first_event(payload: dict[str, Any]) -> None:
        # A dict that's a Mapping but won't validate as MaterializedLedgerEvent.
        payload["ledger"][0] = {"bogus_field": "no required fields here"}

    _corrupt_materialized_ledger(run_root, corruption=_replace_first_event)
    with pytest.raises(
        DerivationDataLossError, match="MaterializedLedgerEvent validation"
    ):
        discover_trajectories(run_root)
