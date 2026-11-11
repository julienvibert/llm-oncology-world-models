#!/usr/bin/env python3
"""Rebuild the combined Kimi K3 EGFR-check table from its two run folders.

`outputs/exploratory_interventional_phase1_kimi_k3_combined/` holds the Kimi K3 rows of the
EGFR checks (Fig. 3e,f). It was first assembled on 28 July 2026; this script rebuilds it
deterministically:

  1. Read `derivation/endpoint_rows.csv` and `derivation/event_rows.csv` from the main run
     (`interventional_phase1_kimi_k3_off_full_v1c`, 285 accepted futures) and from the
     top-up run (`interventional_phase1_kimi_k3_off_topup75_v1c_retry2`, 75 accepted futures).
  2. Append `__topup_retry2` to every top-up `trajectory_id` (the top-up futures are
     numbered f01 and would otherwise collide with main-run IDs), and concatenate
     main rows first, then top-up rows.
  3. Rebuild the arm manifest (arm, role, state, history, decision cutoff of each cell) with
     the committed builder `build_interventional_phase1.py`. The original manifest was
     written to /private/tmp and is lost. The rebuilt cells are checked against the
     `cells` of both run configs; any difference stops the script.
  4. Score the merged derivation with the unchanged scorer `score_interventional_phase1.py`.

The run folders are only read. The derivation CSVs are the ones already in the run folders
(made by `scripts/build_rev5_derivation.py`).

Examples:
  uv run python scripts/explore/build_kimi_k3_combined.py            # scratch dir + compare
  uv run python scripts/explore/build_kimi_k3_combined.py --out-dir /some/dir
  uv run python scripts/explore/build_kimi_k3_combined.py --in-place # only if identical
"""

from __future__ import annotations

import argparse
import filecmp
import json
from pathlib import Path
import shutil
import sys
import tempfile

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import score_interventional_phase1 as scorer  # noqa: E402

MAIN_RUN = ROOT / "runs" / "interventional_phase1_kimi_k3_off_full_v1c"
RETRY_RUN = ROOT / "runs" / "interventional_phase1_kimi_k3_off_topup75_v1c_retry2"
COMBINED_DIR = ROOT / "outputs" / "exploratory_interventional_phase1_kimi_k3_combined"
RETRY_SUFFIX = "__topup_retry2"
DERIVATION_FILES = ("endpoint_rows.csv", "event_rows.csv")
OUTPUT_FILES = (
    "phase1_cell_arm_medians.csv",
    "phase1_context_arm_medians.csv",
    "phase1_endpoint_long.csv",
    "phase1_gate_readout.md",
    "phase1_gate_scores.csv",
    "phase1_supportive_ttp_trajectory.csv",
)
CELL_KEYS = ("id", "condition_id", "policy_id", "baseline_seed", "operator_constraints")


def rebuild_arm_manifest() -> tuple[dict[str, dict], list[dict]]:
    """Arm manifest and prompt cells from the committed Phase-1 builder (no files written)."""
    import build_interventional_phase1 as builder

    manifest: dict[str, dict] = {}
    cells: list[dict] = []
    for context_id in builder.CONTEXTS:
        for state_id in builder.STATE_GRID:
            for history_id in builder.HISTORY_GRID:
                card = builder._build_card(context_id, state_id, history_id)
                prefix = builder.observed_prefix_phase1(card)
                for arm_key in builder.CONTEXTS[context_id]["arms"]:
                    cell, row = builder.build_cell_phase1(card, arm_key, prefix)
                    cells.append(cell)
                    manifest[cell["id"]] = row
    return manifest, cells


def check_cells_match_config(cells: list[dict], config_path: Path) -> int:
    """Every config cell must equal the rebuilt cell with the same id. Returns the count."""
    config = yaml.safe_load(config_path.read_text())
    rebuilt = {c["id"]: c for c in cells}
    bad: list[str] = []
    for cell in config["cells"]:
        ref = rebuilt.get(cell["id"])
        if ref is None or any(ref.get(k) != cell.get(k) for k in CELL_KEYS):
            bad.append(cell["id"])
    if bad:
        raise SystemExit(f"[fatal] {len(bad)} cells of {config_path} differ from the builder (e.g. {bad[0]})")
    return len(config["cells"])


def merge_derivations(main_run: Path, retry_run: Path, dest_run: Path, *, suffix: str = RETRY_SUFFIX) -> dict[str, int]:
    """Write main + suffixed retry derivation CSVs under dest_run/derivation. Returns row counts."""
    out = dest_run / "derivation"
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name in DERIVATION_FILES:
        main = pd.read_csv(main_run / "derivation" / name)
        retry = pd.read_csv(retry_run / "derivation" / name)
        retry["trajectory_id"] = retry["trajectory_id"].astype(str) + suffix
        clash = set(main["trajectory_id"]) & set(retry["trajectory_id"])
        if clash:
            raise SystemExit(f"[fatal] trajectory_id collision in {name}: {sorted(clash)[:3]}")
        merged = pd.concat([main, retry], ignore_index=True)
        merged.to_csv(out / name, index=False)
        counts[name] = len(merged)
    endpoint = pd.read_csv(out / "endpoint_rows.csv")
    counts["accepted_futures"] = int(scorer._accepted_mask(endpoint).sum())
    return counts


def build_combined(main_run: Path, retry_run: Path, out_dir: Path, arm_manifest: dict[str, dict]) -> dict[str, int]:
    """Merge the two runs and score them into out_dir (the six phase1_* files only)."""
    with tempfile.TemporaryDirectory(prefix="kimi_k3_combined_") as td:
        work = Path(td)
        counts = merge_derivations(main_run, retry_run, work / "run")
        manifest_path = work / "arm_manifest.json"
        manifest_path.write_text(json.dumps(arm_manifest, indent=2))
        scorer.score_phase1(work / "run", manifest_path, out_dir)
    return counts


def compare_dirs(new: Path, ref: Path, names: tuple[str, ...] = OUTPUT_FILES) -> dict[str, bool | None]:
    """Byte comparison of each output file.

    True, identical; False, different or not produced; None, absent from the reference folder (the public
    deposit holds only some of the reference files), so not compared.
    """
    return {
        n: None if not (ref / n).is_file()
        else (new / n).is_file() and filecmp.cmp(new / n, ref / n, shallow=False)
        for n in names
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-run", type=Path, default=MAIN_RUN)
    ap.add_argument("--retry-run", type=Path, default=RETRY_RUN)
    ap.add_argument("--out-dir", type=Path, help="target directory (default: a new scratch directory)")
    ap.add_argument("--reference", type=Path, default=COMBINED_DIR, help="folder to compare against")
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="write into the reference folder; refused if any file would change",
    )
    args = ap.parse_args()

    arm_manifest, cells = rebuild_arm_manifest()
    n_main = check_cells_match_config(cells, args.main_run / "config.yaml")
    n_retry = check_cells_match_config(cells, args.retry_run / "config.yaml")
    print(f"[cells] builder matches config cells: main {n_main}, retry {n_retry}")

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="kimi_k3_combined_"))
    if args.in_place:
        out_dir = Path(tempfile.mkdtemp(prefix="kimi_k3_combined_"))
    counts = build_combined(args.main_run, args.retry_run, out_dir, arm_manifest)
    print(f"[merge] {counts}")
    print(f"[out] {out_dir}")

    if not args.reference.is_dir():
        if args.in_place:
            args.reference.mkdir(parents=True)
            for n in OUTPUT_FILES:
                shutil.copy2(out_dir / n, args.reference / n)
            print(f"[in-place] created {args.reference}")
        else:
            print(f"[compare] no reference folder at {args.reference}")
        return 0

    result = compare_dirs(out_dir, args.reference)
    for name, same in result.items():
        label = "not in the reference folder (not compared)" if same is None else (
            "identical" if same else "DIFFERENT")
        print(f"[compare] {name}: {label}")
    if any(same is False for same in result.values()) or all(same is None for same in result.values()):
        if args.in_place:
            print("[in-place] refused: output differs from the existing folder; nothing written")
        return 1
    if args.in_place:
        print(f"[in-place] {args.reference} already identical; nothing to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
