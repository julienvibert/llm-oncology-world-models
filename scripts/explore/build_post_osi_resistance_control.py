#!/usr/bin/env python3
"""Control: does an EXPLICIT osi-resistance clone stop the model re-sensitising osimertinib?

The post-osi finding (model re-imagines a RESPONSE to osimertinib after documented
progression) is confounded: our prefix recorded `progression` but no resistance clone,
so the recorded state still held only the founder EGFR-sensitive clone. This control
adds the decisive variable — an explicit `clonal_evolution_event` (EGFR C797S, the
model's own resistance mechanism) inside the fixed prefix — and holds everything else
fixed.

  IMPLICIT prefix : osi start -> response -> progression            (current; re-responds)
  EXPLICIT prefix : osi start -> response -> C797S clone -> progression

Two arms each (continue-osi = the test, chemo = stable comparator). Read-out after
generation: fraction of continue-osi trajectories that imagine a RESPONSE after cutoff.
  * explicit kills the re-response  -> representation issue: model won't apply
    progression->resistance unless the clone is explicit (won't infer the default).
  * re-response persists            -> deeper: model overrides an explicit resistance flag.

Built on the runner's core observed-prefix path (conditioning.fixed_events), independent
of the battery builder. Validates seed + prefix + load_config before emitting. No spend.
  uv run python scripts/explore/build_post_osi_resistance_control.py
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from synthonccases.oncollm.condition_lower import lower_conditioning_cell  # noqa: E402
from synthonccases.oncollm.conditioning_audit import (  # noqa: E402
    BaselineObject,
    validate_baseline_seed,
    validate_observed_prefix,
)
from synthonccases.oncollm.poc_runner import load_config  # noqa: E402

CONFIG_ID = "post_osi_resistance_control_v1"
SEED_PATH = Path(__file__).resolve().parents[2] / "runs" / \
    "sentinel_usecase_egfr_post_osi_v5_2026_07_13" / "config.yaml"  # source of the validated seed
DX = date(2018, 3, 1)
CUTOFF = date(2019, 6, 1)
DEFAULT_OUT = Path(f"/private/tmp/{CONFIG_ID}")
MODELS = [{"id": "google/gemini-3.1-flash-lite", "short": "gemini31_lite"},
          {"id": "qwen/qwen3.6-flash", "short": "qwen36_flash"},
          {"id": "google/gemma-4-31b-it", "short": "gemma4"},
          {"id": "anthropic/claude-sonnet-5", "short": "claude_sonnet5"}]

RESISTANCE_CLONE = {
    "type": "clonal_evolution_event", "date": date(2018, 11, 1).isoformat(),
    "gene": "EGFR", "alteration": "EGFR C797S", "alteration_type": "mutation",
    "driver_bucket": "egfr", "actionability": "resistance_marker",
    "mechanism_bucket": "on_target_secondary_alteration",
    "mechanism": "on-target secondary mutation conferring resistance to osimertinib",
    "new_clone_id": "C1", "parent_clone": "C0"}
ARMS = {"continue_osi": ("osimertinib", ("osimertinib",)),
        "chemo": ("carboplatin + pemetrexed", ("carboplatin", "pemetrexed"))}


def prefix(explicit: bool) -> dict:
    events = [
        {"type": "systemic_treatment_start", "date": DX.isoformat(),
         "regimen": {"label": "osimertinib", "components": ["osimertinib"]}},
        {"type": "response", "date": date(2018, 5, 1).isoformat()},
    ]
    if explicit:
        events.append(RESISTANCE_CLONE)
    events.append({"type": "progression", "date": CUTOFF.isoformat()})
    return {"origin": "observed_prefix", "cutoff_date": CUTOFF.isoformat(), "events": events}


def load_seed() -> dict:
    cfg = yaml.safe_load(SEED_PATH.read_text())
    cell = next(c for c in cfg["cells"] if c["id"].endswith("continue_osi"))
    return cell["baseline_seed"]


def build_cell(seed: dict, cond: str, arm: str, pfx: dict) -> dict:
    label, components = ARMS[arm]
    op = (f"The next systemic_treatment_start after the fixed prior course MUST be {label} "
          "(no additional systemic anticancer agents in that same event). Response, toxicity, "
          "progression, later treatment changes, and terminal events are unconstrained.")
    return {"id": f"{cond}__{arm}", "condition_id": cond, "policy_id": f"{cond}__{arm}",
            "baseline_seed": seed, "operator_constraints": op,
            "conditioning": {"free_text_operator_constraints": op, "fixed_events": pfx,
                             "future_event_constraints": [{
                                 "constraint_id": f"force_{arm}",
                                 "selector": {"event_type": "systemic_treatment_start",
                                              "ordinal_after_cutoff": "first"},
                                 "constraint": {"regimen_matches": list(components)}}]}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--accepted-per-cell", type=int, default=3)
    args = ap.parse_args()

    seed = load_seed()
    BaselineObject.model_validate(seed)
    validate_baseline_seed(seed)

    cells = []
    for cond, explicit in (("postosi_implicit", False), ("postosi_explicit", True)):
        pfx = prefix(explicit)
        probe = lower_conditioning_cell(conditioning_payload={
            "baseline_seed": seed, "fixed_events": pfx, "free_text_operator_constraints": ""})
        validate_observed_prefix(probe, baseline_diagnosis_date=DX)
        for arm in ARMS:
            cells.append(build_cell(seed, cond, arm, pfx))

    config = {"schema_version": 1, "config_id": CONFIG_ID,
              "study_reference": {"diagnosis_date": DX.isoformat(), "knowledge_cutoff": "2024-12-31"},
              "run_defaults": {"cells_count": len(cells), "accepted_futures_per_cell": args.accepted_per_cell,
                               "max_total_futures_per_cell": max(8, args.accepted_per_cell * 3),
                               "max_attempts": 4, "terminal_contract": "death_only"},
              "prompt": {"template_path": "prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md",
                         "enum_order": {"mode": "deterministic_shuffle", "seed": f"{CONFIG_ID}_enum_v1"}},
              "provider": {"use_openrouter": True, "response_format": {"type": "json_object"}},
              "models": MODELS, "cells": cells}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = args.out_dir / f"{CONFIG_ID}.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False, width=110))
    loaded = load_config(cfg_path)
    for lc in loaded.cells:
        ic = dict(lc.condition.baseline_seed).get("identity_card", {})
        assert ic.get("molecular_profile", {}).get("primary_driver_bucket"), f"seed not pinned: {lc.id}"

    n = len(cells) * len(MODELS) * args.accepted_per_cell
    print(f"post-osi resistance control: {len(cells)} cells "
          f"(2 conditions x 2 arms) x {len(MODELS)} models x {args.accepted_per_cell} = {n} futures (~$4)")
    print("  conditions: postosi_implicit (progression only) | postosi_explicit (+EGFR C797S clone)")
    print("  load_config + observed-prefix + seed-pinned: PASS")
    print(f"  config -> {cfg_path}")
    print(f"\nRun:\n  uv run python scripts/run_paper_v1.py --config {cfg_path}")
    print("Then (renewed-response readout, implicit vs explicit):")
    print("  build derivation, then per condition count continue-osi trajectories with a post-cutoff `response`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
