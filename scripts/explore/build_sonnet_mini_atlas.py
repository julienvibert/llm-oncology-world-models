#!/usr/bin/env python3
"""Build a budget-capped Sonnet 5 mini-atlas config.

This is a model-quality control for the anchored atlas: keep the locked
paper-v1 tumor-type-only generation contract, but narrow the matrix to a
clinically broad 20-code subset and a single Sonnet 5 model.

The builder emits config/manifests only. It does not launch paid generation.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs" / "paper_v1_pillar1_anchored_atlas_tranche1_n5.yaml"
DEFAULT_OUT = Path("/private/tmp/sonnet5_mini_atlas")
CONFIG_ID = "paper_v1_sonnet5_mini_atlas_n5_v1"
SONNET_MODEL = {"id": "anthropic/claude-sonnet-5", "short": "claude_sonnet5"}

# Broad, deterministic cross-tissue subset from the locked 148-code roster.
DEFAULT_CODES = [
    "LUAD",  # lung adenocarcinoma
    "LUSC",  # lung squamous
    "SCLC",  # small-cell lung
    "IDC",  # breast ductal
    "PAAD",  # pancreatic adenocarcinoma
    "COAD",  # colon adenocarcinoma
    "ESCA",  # esophageal adenocarcinoma
    "HCC",  # hepatocellular carcinoma
    "CCRCC",  # renal clear cell carcinoma
    "BLCA",  # bladder urothelial carcinoma
    "PRAD",  # prostate adenocarcinoma
    "HGSOC",  # high-grade serous ovarian cancer
    "UEC",  # uterine endometrioid carcinoma
    "SKCM",  # cutaneous melanoma
    "GB",  # glioblastoma, IDH-wildtype
    "THPA",  # papillary thyroid cancer
    "OS",  # osteosarcoma
    "CUP",  # cancer of unknown primary
    "AML",  # acute myeloid leukemia
    "DLBCLNOS",  # diffuse large B-cell lymphoma, NOS
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config-id", default=CONFIG_ID)
    parser.add_argument("--accepted-per-code", type=int, default=5)
    parser.add_argument("--codes", nargs="+", default=None)
    parser.add_argument(
        "--exclude-codes-from-run",
        type=Path,
        default=None,
        help=(
            "Run root whose derivation/endpoint_rows.csv contains already "
            "generated OncoTree codes. When set without --codes, selects all "
            "base-config codes except those observed in that run."
        ),
    )
    parser.add_argument("--max-total-futures-per-cell", type=int, default=24)
    args = parser.parse_args()

    if args.accepted_per_code < 1:
        raise SystemExit("--accepted-per-code must be >= 1")
    if args.max_total_futures_per_cell < args.accepted_per_code:
        raise SystemExit("--max-total-futures-per-cell must be >= --accepted-per-code")

    cfg = yaml.safe_load(args.base_config.read_text())
    cells_by_code = {
        cell["baseline_seed"]["identity_card"]["diagnosis"]["oncotree_code"]: cell
        for cell in cfg["cells"]
    }
    requested_codes = args.codes
    if requested_codes is None:
        requested_codes = list(cells_by_code) if args.exclude_codes_from_run else DEFAULT_CODES

    excluded_codes: set[str] = set()
    if args.exclude_codes_from_run is not None:
        excluded_codes = _codes_from_endpoint_rows(args.exclude_codes_from_run)
        requested_codes = [code for code in requested_codes if code not in excluded_codes]

    missing = [code for code in requested_codes if code not in cells_by_code]
    if missing:
        raise SystemExit(f"Requested code(s) not in base config: {', '.join(missing)}")

    cells = [deepcopy(cells_by_code[code]) for code in requested_codes]
    out_cfg = deepcopy(cfg)
    out_cfg["config_id"] = args.config_id
    out_cfg["run_defaults"]["cells_count"] = len(cells)
    out_cfg["run_defaults"]["accepted_futures_per_cell"] = args.accepted_per_code
    out_cfg["run_defaults"]["max_total_futures_per_cell"] = args.max_total_futures_per_cell
    out_cfg["models"] = [SONNET_MODEL]
    out_cfg["cells"] = cells

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.out_dir / f"{args.config_id}.yaml"
    manifest_name = (
        "sonnet5_mini_atlas_manifest.json"
        if args.config_id == CONFIG_ID
        else f"{args.config_id}_manifest.json"
    )
    manifest_path = args.out_dir / manifest_name
    readme_path = args.out_dir / "README.md"

    config_path.write_text(yaml.safe_dump(out_cfg, sort_keys=False, width=110))
    manifest = {
        "config_id": args.config_id,
        "base_config": str(args.base_config),
        "model": SONNET_MODEL,
        "accepted_per_code": args.accepted_per_code,
        "planned_accepted": len(cells) * args.accepted_per_code,
        "excluded_codes_from_run": str(args.exclude_codes_from_run)
        if args.exclude_codes_from_run
        else None,
        "excluded_codes_count": len(excluded_codes),
        "codes": [
            {
                "code": cell["baseline_seed"]["identity_card"]["diagnosis"]["oncotree_code"],
                "diagnosis_label": cell["baseline_seed"]["identity_card"]["diagnosis"]["diagnosis_label"],
                "cell_id": cell["id"],
            }
            for cell in cells
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    readme_path.write_text(readme(config_path, manifest))

    print("Sonnet 5 mini-atlas config")
    if excluded_codes:
        print(f"  excluded: {len(excluded_codes)} already generated code(s)")
    print(f"  codes:    {len(cells)}")
    print(f"  model:    {SONNET_MODEL['short']}")
    print(f"  accepted: {len(cells)} x {args.accepted_per_code} = {manifest['planned_accepted']}")
    print(f"  out:      {args.out_dir}")
    print(f"\nRun:\n  uv run python scripts/run_paper_v1.py --config {config_path}")
    return 0


def _codes_from_endpoint_rows(run_root: Path) -> set[str]:
    endpoint_rows = run_root / "derivation" / "endpoint_rows.csv"
    if not endpoint_rows.exists():
        raise SystemExit(f"endpoint_rows.csv not found under run root: {endpoint_rows}")
    with endpoint_rows.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"endpoint_rows.csv has no header: {endpoint_rows}")
        code_col = (
            "tumor_oncotree_code"
            if "tumor_oncotree_code" in reader.fieldnames
            else "tumor_type"
            if "tumor_type" in reader.fieldnames
            else None
        )
        if code_col is None:
            raise SystemExit(
                "endpoint_rows.csv must contain tumor_oncotree_code or tumor_type: "
                f"{endpoint_rows}"
            )
        return {row[code_col] for row in reader if row.get(code_col)}


def readme(config_path: Path, manifest: dict[str, object]) -> str:
    model = manifest["model"]  # type: ignore[assignment]
    rows = "\n".join(
        f"- `{row['code']}`: {row['diagnosis_label']}"
        for row in manifest["codes"]  # type: ignore[index]
    )
    return f"""# Sonnet 5 Mini-Atlas

Generated by `scripts/explore/build_sonnet_mini_atlas.py`.

Purpose: budget-capped Sonnet 5 control for the anchored n=5 atlas. The prompt,
schema, terminal contract, enum shuffle, study reference, and tumor-type-only
baseline seed contract are inherited from the locked anchored atlas config.
Only the cell subset and model are narrowed.

## Scale

- Codes: {len(manifest["codes"])}
- Model: `{model["short"]}`
- Planned accepted trajectories: {manifest["planned_accepted"]}

## Codes

{rows}

## Run

```bash
uv run python scripts/run_paper_v1.py --config {config_path}
```
"""


if __name__ == "__main__":
    raise SystemExit(main())
