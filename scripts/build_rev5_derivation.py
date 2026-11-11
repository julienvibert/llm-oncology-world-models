#!/usr/bin/env python3
"""Operator CLI for the SynthOncCases derivation layer.

Wraps ``synthonccases.oncollm.derivation.derive_run``. Reads
``<run_root>/trajectories/**`` and writes analysis artifacts under
``<run_root>/derivation/`` (or a user-specified ``--out-dir``):

- ``event_rows.csv``
- ``snapshot_rows.csv``
- ``treatment_rows.csv``
- ``endpoint_rows.csv``
- ``baseline_rows.csv``
- ``baseline_biomarker_rows.csv``
- ``baseline_alteration_rows.csv``
- ``treatment_component_rows.csv``
- ``disease_site_rows.csv``
- ``attempt_rows.csv``
- ``field_utilization.json``
- ``run_summary.json`` + ``run_summary.md``

Read-only on everything outside the derivation output directory. Refuses to
overwrite an existing non-empty output directory unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# Make src/ importable without a package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synthonccases.oncollm.derivation import derive_run  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Path to a SynthOncCases generation run root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for derivation artifacts (default: <run-root>/derivation/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already contains files.",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include trajectories where accepted=False (default: accepted-only).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.run_root.exists():
        print(f"run root does not exist: {args.run_root}", file=sys.stderr)
        return 2
    if not (args.run_root / "trajectories").exists():
        print(
            f"trajectories directory not found under {args.run_root}; "
            "is this a SynthOncCases generation run root?",
            file=sys.stderr,
        )
        return 2
    try:
        report = derive_run(
            args.run_root,
            out_dir=args.out_dir,
            force=args.force,
            include_rejected=args.include_rejected,
        )
    except FileExistsError as exc:
        print(f"{exc}", file=sys.stderr)
        return 3

    print(f"derivation dir:        {report.out_dir}")
    print(f"event_rows.csv:        {report.n_events} rows")
    print(f"snapshot_rows.csv:     {report.n_snapshots} rows")
    print(f"treatment_rows.csv:    {report.n_treatment_courses} rows")
    print(f"endpoint_rows.csv:     {report.n_endpoints} rows")
    print(f"baseline_biomarker_rows.csv:  {report.n_baseline_biomarkers} rows")
    print(f"baseline_alteration_rows.csv: {report.n_baseline_alterations} rows")
    print(f"treatment_component_rows.csv: {report.n_treatment_components} rows")
    print(f"disease_site_rows.csv:        {report.n_disease_sites} rows")
    print(f"attempt_rows.csv:             {report.n_attempts} rows")
    print("field_utilization.json + run_summary.{json,md} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
