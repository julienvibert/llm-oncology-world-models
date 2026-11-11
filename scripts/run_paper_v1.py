"""SynthOncCases generation CLI.

One Typer command, one YAML config. The preferred YAML surface is `cells:`:
each cell can pin any subset of the baseline, provide fixed/observed prefix
events, and specify future-event constraints. Legacy `conditions:` +
`policies:` configs are still accepted for compatibility. Each cell evaluates
to a deterministic `conditioning_label` + `conditioning_hash` that flows
through the manifest, per-trajectory `trajectory_result.json`, and derivation
tables.

The same CLI handles every generation regime: spontaneous, tumor-type-only,
full-baseline, anchor-sensitivity, treatment-CF, continuation, etc. The
variation is expressed in the YAML, not the CLI flags.

Examples:
    # Dry-run a single-condition full-baseline regime
    uv run python scripts/run_paper_v1.py \\
        --config configs/paper_v1_brca_full_baseline.yaml \\
        --dry-run --out-root /tmp/paper_v1_smoke

    # Real run, override cells_count, prepend a run-tag to every
    # cell's human-readable label (hash unaffected)
    uv run python scripts/run_paper_v1.py \\
        --config configs/paper_v1_brca_anchor_sensitivity.yaml \\
        --out-root runs \\
        --cells-count 6 \\
        --max-parallel-trajectories 8 \\
        --label-prefix pilot-1
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from synthonccases.oncollm.poc_runner import (  # noqa: E402
    apply_overrides,
    default_run_id,
    load_config,
    run_poc,
    write_run_inventory,
)


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config: Path = typer.Option(  # noqa: B008
        ...,
        "--config",
        help="Path to the conditioning YAML config.",
    ),
    out_root: Path = typer.Option(  # noqa: B008
        REPO_ROOT / "runs",
        "--out-root",
        help="Directory where <out_root>/<run_id>/ is created.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Optional explicit run id (defaults to a UTC-stamped id).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Write prompts + manifest + per-trajectory result stubs but skip provider calls.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help=(
            "Resume an existing live run with matching config/prompt/replay "
            "hashes, reusing completed trajectory_result.json files and "
            "top-up unused future slots."
        ),
    ),
    conditions_count: int | None = typer.Option(
        None,
        "--cells-count",
        "--conditions-count",
        help=(
            "Override run_defaults.cells_count / conditions_count "
            "(must be ≤ available explicit cells or conditions)."
        ),
    ),
    futures_per_cell: int | None = typer.Option(
        None,
        "--futures-per-cell",
        help="Override run_defaults.futures_per_cell (≥ 1).",
    ),
    accepted_futures_per_cell: int | None = typer.Option(
        None,
        "--accepted-futures-per-cell",
        min=1,
        help=(
            "Override run_defaults.accepted_futures_per_cell. Useful for "
            "cheap broad smokes from a larger accepted-target config."
        ),
    ),
    max_total_futures_per_cell: int | None = typer.Option(
        None,
        "--max-total-futures-per-cell",
        min=1,
        help=(
            "Override run_defaults.max_total_futures_per_cell, the generated "
            "future cap used by accepted-target sampling."
        ),
    ),
    max_attempts: int | None = typer.Option(
        None,
        "--max-attempts",
        help="Override run_defaults.max_attempts (≥ 1).",
    ),
    max_parallel_trajectories: int = typer.Option(
        1,
        "--max-parallel-trajectories",
        min=1,
        help=(
            "Maximum number of live trajectories to run concurrently. "
            "Dry-runs remain sequential."
        ),
    ),
    max_parallel_trajectories_per_model: int | None = typer.Option(
        None,
        "--max-parallel-trajectories-per-model",
        min=1,
        help=(
            "When set, run up to this many live trajectories per model. "
            "This is the preferred high-throughput mode for all-4-model "
            "paper runs and overrides the global concurrency limit for "
            "live dispatch."
        ),
    ),
    terminal_contract: str | None = typer.Option(
        None,
        "--terminal-contract",
        help="'death_or_last_followup' or 'death_only'.",
    ),
    label_prefix: str | None = typer.Option(
        None,
        "--label-prefix",
        help=(
            "Cosmetic prefix prepended to every cell's derived "
            "conditioning_label. Does NOT affect conditioning_hash."
        ),
    ),
    inventory_only: bool = typer.Option(
        False,
        "--inventory-only",
        help=(
            "Rebuild run_inventory.csv and run_inventory_summary.json for an "
            "existing run without launching provider calls. Requires --run-id."
        ),
    ),
) -> None:
    if terminal_contract is not None and terminal_contract not in (
        "death_or_last_followup", "death_only",
    ):
        raise typer.BadParameter(
            f"terminal_contract must be 'death_or_last_followup' or 'death_only'; "
            f"got {terminal_contract!r}",
            param_hint="--terminal-contract",
        )
    cfg = load_config(config)
    cfg = apply_overrides(
        cfg,
        conditions_count=conditions_count,
        futures_per_cell=futures_per_cell,
        accepted_futures_per_cell=accepted_futures_per_cell,
        max_total_futures_per_cell=max_total_futures_per_cell,
        max_attempts=max_attempts,
        terminal_contract=terminal_contract,  # type: ignore[arg-type]
        label_prefix=label_prefix,
    )

    chosen_run_id = run_id or default_run_id()
    if inventory_only:
        if run_id is None:
            raise typer.BadParameter(
                "--inventory-only requires an explicit --run-id",
                param_hint="--run-id",
            )
        summary = write_run_inventory(out_root / chosen_run_id, cfg)
        typer.echo(_format_inventory_summary(summary))
        return

    report = run_poc(
        config=cfg,
        run_root=out_root,
        run_id=chosen_run_id,
        dry_run=dry_run,
        resume=resume,
        max_parallel_trajectories=max_parallel_trajectories,
        max_parallel_trajectories_per_model=max_parallel_trajectories_per_model,
        source_config_path=config,
    )
    report_dict = asdict(report)
    report_dict["run_root"] = str(report.run_root)
    typer.echo(_format_report(report_dict))


def _format_report(report: dict) -> str:
    return (
        f"run_id:               {report['run_id']}\n"
        f"run_root:             {report['run_root']}\n"
        f"dry_run:              {report['dry_run']}\n"
        f"cells_planned:        {report['cells_planned']}\n"
        f"trajectories_planned: {report['trajectories_planned']}\n"
        f"trajectories_accepted: {report['trajectories_accepted']}\n"
        f"trajectories_failed:  {report['trajectories_failed']}\n"
        f"trajectories_reused:  {report['trajectories_reused']}\n"
        f"resumed:              {report['resumed']}\n"
        f"max_parallel_trajectories: {report['max_parallel_trajectories']}\n"
        f"max_parallel_trajectories_per_model: "
        f"{report['max_parallel_trajectories_per_model']}\n"
        f"started_at:           {report['started_at']}\n"
        f"completed_at:         {report['completed_at']}"
    )


def _format_inventory_summary(summary: dict) -> str:
    return (
        f"run_id:                   {summary['run_id']}\n"
        f"inventory_path:           {summary['inventory_path']}\n"
        f"rows_total:               {summary['rows_total']}\n"
        f"result_files:             {summary['result_files']}\n"
        f"accepted_total:           {summary['accepted_total']}\n"
        f"primary_accepted_total:   {summary['primary_accepted_total']}\n"
        f"accepted_nonprimary_total: {summary['accepted_nonprimary_total']}\n"
        f"failed_completed_total:   {summary['failed_completed_total']}\n"
        f"partial_dirs:             {summary['partial_dirs']}"
    )


if __name__ == "__main__":
    app()
