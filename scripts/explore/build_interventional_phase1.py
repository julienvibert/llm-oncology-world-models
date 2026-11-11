#!/usr/bin/env python3
"""Build the v1 Phase-1 EGFR interventional design (no model call).

This is the generation-facing artifact for docs/interventional_run_spec_v1.md:

  * EGFR-mut LUAD, 1L/current-decision positive context.
  * EGFR-WT LUAD negative-control context.
  * 5 current clinical states x 3 raw pre-treatment histories x 4 forced arms.
  * Sonnet only by default, 3 accepted futures/cell.

The script emits config/seeds/manifests/pre-registration artifacts only. It
launches no provider calls. The generated YAML is intended to be inspected with
`scripts/run_paper_v1.py --dry-run` before any paid run.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import date
from datetime import timedelta
import csv
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_sentinel_battery as sb  # noqa: E402
from synthonccases.oncollm.condition_lower import lower_conditioning_cell  # noqa: E402
from synthonccases.oncollm.conditioning_audit import (  # noqa: E402
    BaselineObject,
    validate_baseline_seed,
    validate_observed_prefix,
)
from synthonccases.oncollm.poc_runner import load_config  # noqa: E402


CONFIG_ID = "interventional_phase1_egfr_v1c"
DEFAULT_OUT = Path("/private/tmp") / CONFIG_ID
DIAGNOSIS_DATE = sb.DIAGNOSIS_DATE

BURDEN_SCORE = {
    "absent": 0,
    "microscopic": 1,
    "small": 2,
    "moderate": 3,
    "bulky": 4,
    "diffuse_or_leukemic": 5,
}
HEALTH_SCORE = {
    "normal": 0,
    "mild_impairment": 1,
    "moderate_impairment": 2,
    "severe_impairment": 3,
    "failure": 4,
}

LUNG_PRIMARY = {
    "site": "primary tumour",
    "site_class": "primary_site",
    "burden": "small",
    "kinetic": "growing",
}


def _site(site: str, site_class: str, burden: str) -> dict:
    return {"site": site, "site_class": site_class, "burden": burden, "kinetic": "growing"}


def _health(*, frailty: str = "fit", hepatic: str = "normal", pulmonary: str = "normal") -> dict:
    return {
        "renal": "normal",
        "hepatic": hepatic,
        "cardiac": "normal",
        "marrow_reserve": "normal",
        "pulmonary": pulmonary,
        "frailty_band": frailty,
    }


STATE_GRID = {
    "S1": {
        "name": "low_burden_fit",
        "age": 58,
        "sex": "female",
        "ecog": 0,
        "health": _health(frailty="fit"),
        "current_burden": [
            {**LUNG_PRIMARY, "burden": "small"},
            _site("contralateral lung metastases", "lung", "small"),
        ],
    },
    "S2": {
        "name": "standard_metastatic",
        "age": 65,
        "sex": "female",
        "ecog": 1,
        "health": _health(frailty="fit"),
        "current_burden": [
            {**LUNG_PRIMARY, "burden": "moderate"},
            _site("liver metastases", "liver", "small"),
        ],
    },
    "S3": {
        "name": "cns_met_fit",
        "age": 63,
        "sex": "female",
        "ecog": 1,
        "health": _health(frailty="fit", pulmonary="mild_impairment"),
        "current_burden": [
            {**LUNG_PRIMARY, "burden": "moderate"},
            _site("brain metastases", "brain", "small"),
            _site("liver metastases", "liver", "small"),
        ],
    },
    "S4": {
        "name": "high_burden_fit",
        "age": 69,
        "sex": "male",
        "ecog": 1,
        "health": _health(frailty="fit"),
        "current_burden": [
            {**LUNG_PRIMARY, "burden": "moderate"},
            _site("liver metastases", "liver", "moderate"),
            _site("bone metastases", "bone", "moderate"),
        ],
    },
    "S5": {
        "name": "high_burden_frail",
        "age": 78,
        "sex": "male",
        "ecog": 2,
        "health": _health(frailty="frail", hepatic="mild_impairment", pulmonary="mild_impairment"),
        "current_burden": [
            {**LUNG_PRIMARY, "burden": "moderate"},
            _site("liver metastases", "liver", "bulky"),
            _site("bone metastases", "bone", "moderate"),
        ],
    },
}

# These names never reach the prompt. The model sees only raw event dates and
# disease deltas. H0 = immediate decision; H1/H2 = same current state reached
# after different pre-treatment tempos.
HISTORY_GRID = {
    "H0": {"name": "immediate_decision", "progression_month": None},
    "H1": {"name": "rapid_pre_treatment_growth", "progression_month": 1},
    "H2": {"name": "slower_pre_treatment_growth", "progression_month": 4},
}


def _molecular_profile(*, egfr_mut: bool) -> dict:
    if egfr_mut:
        return {
            "alterations": [
                {
                    "gene": "EGFR",
                    "alteration": "EGFR exon 19 deletion",
                    "driver_bucket": "egfr",
                    "is_primary_driver": True,
                    "actionability": "standard_of_care",
                },
                {
                    "gene": "TP53",
                    "alteration": "TP53 mutation",
                    "driver_bucket": "tp53",
                    "is_primary_driver": False,
                    "actionability": "prognostic_only",
                },
            ],
            "biomarkers": [{"name": "PD-L1", "value": "low"}],
            "primary_driver_bucket": "egfr",
        }
    return {
        "alterations": [
            {
                "gene": "TP53",
                "alteration": "TP53 mutation",
                "driver_bucket": "tp53",
                "is_primary_driver": False,
                "actionability": "prognostic_only",
            }
        ],
        "biomarkers": [{"name": "PD-L1", "value": "low"}],
        "primary_driver_bucket": "none",
    }


CONTEXTS = {
    "EGFR_1L": {
        "name": "EGFR-mut LUAD",
        "egfr_mut": True,
        "arms": {
            "osimertinib": "matched_standard",
            "chemo": "active_alternative",
            "pembrolizumab": "weak_io",
            "supportive": "no_systemic_control",
        },
    },
    "EGFR_WT_1L": {
        "name": "EGFR-WT LUAD",
        "egfr_mut": False,
        "arms": {
            "chemo": "active_standard",
            "osimertinib": "wrong_target",
            "pembrolizumab": "weak_io",
            "supportive": "no_systemic_control",
        },
    },
}

PREREG_RUBRIC = {
    "endpoint": (
        "Active arms: forced-line PFS (endpoint_rows.csv:pfs_line_1_days). "
        "Supportive arm: disease-worsening time = first post-cutoff progression minus cutoff; "
        "if no progression is emitted before death, use death time as a conservative fallback and flag it."
    ),
    "phase1_gate": [
        "EGFR-mut: median osimertinib forced-line PFS must exceed median chemo forced-line PFS.",
        "EGFR-mut: median osimertinib forced-line PFS must exceed median supportive disease-worsening time.",
        "EGFR-WT specificity: osimertinib must not rank first among active arms by median forced-line PFS.",
        "BSC control: supportive disease-worsening time must be lower than the best active-arm PFS in each context.",
        "No impossible reversal: median (osimertinib PFS - chemo PFS) must be larger in EGFR-mut than EGFR-WT.",
    ],
    "scoreability": {
        "pfs": "endpoint_rows.csv:pfs_line_1_days for active forced arms",
        "supportive_ttp": (
            "event_rows.csv first post-cutoff progression for supportive arms; "
            "fallback to endpoint_rows.csv:os_days if no progression is emitted"
        ),
        "os": "endpoint_rows.csv:os_days for secondary/descriptive readouts",
        "field": "derived snapshots/events grouped by context, state_id, history_id, arm_key",
        "field_shift": "descriptive only for F5; not part of the hard Phase-1 pass/fail gate",
    },
}


def _months_after_diagnosis(months: int) -> date:
    y, m = divmod(DIAGNOSIS_DATE.month - 1 + months, 12)
    return date(DIAGNOSIS_DATE.year + y, m + 1, DIAGNOSIS_DATE.day)


def _less_burden(sites: Iterable[dict]) -> list[dict]:
    """Construct a lower-burden starting point for pre-treatment tempo histories."""
    order = ["microscopic", "small", "moderate", "bulky", "diffuse_or_leukemic"]
    idx = {name: i for i, name in enumerate(order)}
    out = []
    for site in sites:
        b = site["burden"]
        if b == "small":
            nb = "microscopic"
        elif b in idx and idx[b] > 0:
            nb = order[idx[b] - 1]
        else:
            nb = b
        out.append({**site, "burden": nb, "kinetic": "growing"})
    return out


def _disease_delta_from_sites(sites: Iterable[dict]) -> list[dict]:
    return [{"clone": "C0", **site} for site in sites]


def _tumour_index(sites: Iterable[dict]) -> int:
    return sum(BURDEN_SCORE.get(str(site.get("burden")), 0) for site in sites)


def _host_index(card: dict) -> int:
    health = card["baseline_health"]
    organ = max(
        HEALTH_SCORE.get(str(health.get(k)), 0)
        for k in ("renal", "hepatic", "cardiac", "marrow_reserve", "pulmonary")
    )
    return int(card["demographics"]["ecog_ps_at_diagnosis"]) + organ


def _build_card(context_id: str, state_id: str, history_id: str) -> dict:
    context = CONTEXTS[context_id]
    state = STATE_GRID[state_id]
    history = HISTORY_GRID[history_id]
    current = list(state["current_burden"])
    baseline_burden = current if history["progression_month"] is None else _less_burden(current)
    return {
        "_id": f"{context_id}__{state_id}__{history_id}",
        "context": context_id,
        "state_id": state_id,
        "state_name": state["name"],
        "history_id": history_id,
        "history_name": history["name"],
        "demographics": {
            "age_at_diagnosis": state["age"],
            "sex": state["sex"],
            "ecog_ps_at_diagnosis": state["ecog"],
        },
        "diagnosis": {
            "oncotree_code": "LUAD",
            "histology_class": "adenocarcinoma",
            "primary_site": "lung",
        },
        "molecular_profile": _molecular_profile(egfr_mut=bool(context["egfr_mut"])),
        "baseline_health": state["health"],
        "baseline_burden": baseline_burden,
        "current_burden": current,
        "progression_month": history["progression_month"],
    }


def lower_seed_phase1(card: dict) -> dict:
    seed = sb.lower_seed(card)
    seed["baseline_disease_burden"] = [{"clone": "C0", **b} for b in card["baseline_burden"]]
    return seed


def observed_prefix_phase1(card: dict) -> dict | None:
    progression_month = card.get("progression_month")
    if progression_month is None:
        return None
    progression_date = _months_after_diagnosis(int(progression_month))
    events = [
        {
            "type": "progression",
            "date": progression_date.isoformat(),
            "disease": _disease_delta_from_sites(card["current_burden"]),
        }
    ]
    return {"origin": "observed_prefix", "cutoff_date": progression_date.isoformat(), "events": events}


def decision_date_phase1(prefix: dict | None) -> date:
    """Cell-wide clinical decision date.

    The decision date is the common t0 for all arms. When a history prefix is
    non-empty, t0 is strictly after the last fixed history event so generated
    futures cannot collide with the fixed event boundary.
    """
    if prefix is None or not prefix.get("events"):
        return DIAGNOSIS_DATE
    last_event_date = max(date.fromisoformat(event["date"]) for event in prefix["events"])
    return last_event_date + timedelta(days=1)


def supportive_decision_prefix_phase1(prefix: dict | None) -> dict:
    """Prompt-facing fixed prefix for supportive-only arms.

    For immediate-decision patients there is no prior history, but we still pin
    an empty observed prefix at the decision date so the rendered prompt carries
    a concrete cutoff and the strict post-cutoff event rule.
    """
    decision_date = decision_date_phase1(prefix)
    events = [] if prefix is None else list(prefix.get("events", []))
    return {"origin": "observed_prefix", "cutoff_date": decision_date.isoformat(), "events": events}


def fixed_intervention_prefix_phase1(
    prefix: dict | None, *, label: str, components: tuple[str, ...]
) -> dict:
    """Append the assigned active treatment at the shared decision date."""
    decision_date = decision_date_phase1(prefix)
    events = [] if prefix is None else list(prefix.get("events", []))
    event = {
        "type": "systemic_treatment_start",
        "date": decision_date.isoformat(),
        "regimen": {"label": label, "components": list(components)},
    }
    return {
        "origin": "observed_prefix",
        "cutoff_date": decision_date.isoformat(),
        "events": events + [event],
    }


def operator_text_phase1(label: str, components: tuple[str, ...]) -> str:
    if not components:
        return (
            "The assigned management after the current decision point is supportive-care-only "
            "cancer management, with NO systemic anticancer therapy. You MUST NOT emit "
            "systemic_treatment_start, systemic_treatment_switch, or systemic_treatment_stop "
            "events after the cutoff date under any circumstance. If the cancer worsens, "
            "represent that as progression, patient_state_update, local palliation, supportive "
            "medications/procedures, hospice or palliative-care transitions, and/or death as "
            "clinically appropriate. Do not invent chemotherapy, targeted therapy, immunotherapy, "
            "maintenance therapy, or trial systemic therapy. The patient is not automatically "
            "enrolled in hospice at cutoff; symptoms, disease course, local palliation, and "
            "terminal timing remain unconstrained."
        )
    return (
        "The assigned systemic intervention at the current decision point is fixed in the "
        f"Fixed Prefix as systemic_treatment_start = {label}. Do not choose, omit, or "
        "restate that treatment. Generate only downstream response, toxicity, duration, "
        "dose modifications, later treatment changes, progression, and terminal events "
        "after the fixed intervention."
    )


def forbid_systemic_supportive_phase1(cutoff: date) -> list[dict]:
    """Acceptance-audited hard constraints for the supportive-only arm."""
    rng = {"date_range": [cutoff.isoformat(), "2035-01-01"]}
    return [
        {
            "constraint_id": f"forbid_{event_type}",
            "selector": {"event_type": event_type, **rng},
            "constraint": {"must_not_occur": True},
        }
        for event_type in (
            "systemic_treatment_start",
            "systemic_treatment_switch",
            "systemic_treatment_stop",
        )
    ]


def build_cell_phase1(card: dict, arm_key: str, prefix: dict | None) -> tuple[dict, dict]:
    label, components = sb.arm_regimen(arm_key, "LUAD")
    seed = lower_seed_phase1(card)
    op = operator_text_phase1(label, components)
    conditioning: dict = {"free_text_operator_constraints": op}
    decision_date = decision_date_phase1(prefix)
    if components:
        conditioning["fixed_events"] = fixed_intervention_prefix_phase1(prefix, label=label, components=components)
    else:
        fixed_supportive_prefix = supportive_decision_prefix_phase1(prefix)
        cutoff = date.fromisoformat(fixed_supportive_prefix["cutoff_date"])
        conditioning["fixed_events"] = fixed_supportive_prefix
        conditioning["future_event_constraints"] = forbid_systemic_supportive_phase1(cutoff)
    policy_id = f"{card['context']}__{arm_key}"
    cell = {
        "id": f"{card['_id']}__{arm_key}",
        "condition_id": card["_id"],
        "policy_id": policy_id,
        "baseline_seed": seed,
        "operator_constraints": op,
        "conditioning": conditioning,
    }
    manifest = {
        "arm_key": arm_key,
        "policy_id": policy_id,
        "regimen_label": label,
        "intervention_fixed_in_prefix": bool(components),
        "context": card["context"],
        "condition_id": card["_id"],
        "state_id": card["state_id"],
        "state_name": card["state_name"],
        "history_id": card["history_id"],
        "history_name": card["history_name"],
        "role": CONTEXTS[card["context"]]["arms"][arm_key],
        "tumour_index_current": _tumour_index(card["current_burden"]),
        "host_index": _host_index(card),
        "decision_cutoff_date": decision_date.isoformat(),
        "decision_cutoff_days": (decision_date - DIAGNOSIS_DATE).days,
    }
    return cell, manifest


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--accepted-per-cell", type=int, default=3)
    ap.add_argument("--models", nargs="+", choices=sorted(sb.MODEL_LIBRARY), default=["claude_sonnet5"])
    ap.add_argument(
        "--provider",
        choices=["openrouter", "kather"],
        default="openrouter",
        help=(
            "Provider settings to emit in the YAML. Kather uses native provider "
            "routing with provider.use_openrouter=false and no response_format."
        ),
    )
    args = ap.parse_args()

    cells: list[dict] = []
    arm_manifest: dict[str, dict] = {}
    seed_files: dict[str, dict] = {}
    clinical_grid_rows: list[dict] = []
    leaks: list[tuple[str, list[str]]] = []
    prompt_review_cells: list[dict] = []

    for context_id in CONTEXTS:
        for state_id in STATE_GRID:
            for history_id in HISTORY_GRID:
                card = _build_card(context_id, state_id, history_id)
                prefix = observed_prefix_phase1(card)
                seed = lower_seed_phase1(card)
                BaselineObject.model_validate(seed)
                validate_baseline_seed(seed)
                sb.validate_replay_ready_seed(seed, label=card["_id"])
                if prefix is not None:
                    sb.validate_replay_ready_prefix(prefix, label=card["_id"])
                    probe = lower_conditioning_cell(
                        conditioning_payload={
                            "baseline_seed": seed,
                            "fixed_events": prefix,
                            "free_text_operator_constraints": "",
                        }
                    )
                    validate_observed_prefix(probe, baseline_diagnosis_date=DIAGNOSIS_DATE)
                seed_files[card["_id"]] = seed
                clinical_grid_rows.append(
                    {
                        "condition_id": card["_id"],
                        "context": card["context"],
                        "state_id": state_id,
                        "state_name": card["state_name"],
                        "history_id": history_id,
                        "history_name": card["history_name"],
                        "age": card["demographics"]["age_at_diagnosis"],
                        "sex": card["demographics"]["sex"],
                        "ecog": card["demographics"]["ecog_ps_at_diagnosis"],
                        "progression_month": card["progression_month"] or "",
                        "n_current_sites": len(card["current_burden"]),
                        "current_sites": ";".join(s["site_class"] for s in card["current_burden"]),
                        "current_burdens": ";".join(s["burden"] for s in card["current_burden"]),
                        "tumour_index_current": _tumour_index(card["current_burden"]),
                        "host_index": _host_index(card),
                    }
                )
                for arm_key in CONTEXTS[context_id]["arms"]:
                    cell, manifest = build_cell_phase1(card, arm_key, prefix)
                    fixed = cell["conditioning"].get("fixed_events")
                    if fixed is not None:
                        sb.validate_replay_ready_prefix(fixed, label=cell["id"])
                        probe = lower_conditioning_cell(
                            conditioning_payload={
                                "baseline_seed": seed,
                                "fixed_events": fixed,
                                "free_text_operator_constraints": "",
                            }
                        )
                        validate_observed_prefix(probe, baseline_diagnosis_date=DIAGNOSIS_DATE)
                    found = sb.leak_lint(cell)
                    if found:
                        leaks.append((cell["id"], found))
                    cells.append(cell)
                    arm_manifest[cell["id"]] = manifest

    if leaks:
        for cell_id, tokens in leaks:
            print(f"[LEAK] {cell_id}: {tokens}")
        raise SystemExit("[fatal] answer-label leakage into prompt-facing text — aborting")

    cutoffs_by_condition: dict[str, set[str]] = {}
    for row in arm_manifest.values():
        cutoffs_by_condition.setdefault(row["condition_id"], set()).add(row["decision_cutoff_date"])
    nonshared_cutoffs = {condition_id: sorted(cutoffs) for condition_id, cutoffs in cutoffs_by_condition.items() if len(cutoffs) != 1}
    if nonshared_cutoffs:
        first_condition, first_cutoffs = next(iter(nonshared_cutoffs.items()))
        raise SystemExit(
            "[fatal] decision cutoff must be shared across arms within a condition; "
            f"{first_condition} has {first_cutoffs}"
        )

    models = [sb.MODEL_LIBRARY[m] for m in args.models]
    accepted_future_count = len(cells) * len(models) * args.accepted_per_cell
    accepted_cost = sum(len(cells) * args.accepted_per_cell * sb.COST_PER_FUTURE[m["short"]] for m in models)
    retry_cost_20 = accepted_cost * 1.20

    out = args.out_dir
    (out / "seeds").mkdir(parents=True, exist_ok=True)
    for condition_id, seed in seed_files.items():
        (out / "seeds" / f"{condition_id}.json").write_text(json.dumps(seed, indent=2))

    provider_block = (
        {"use_openrouter": True, "response_format": {"type": "json_object"}}
        if args.provider == "openrouter"
        else {"use_openrouter": False, "response_format": {}}
    )

    config = {
        "schema_version": 1,
        "config_id": CONFIG_ID,
        "study_reference": {
            "diagnosis_date": DIAGNOSIS_DATE.isoformat(),
            "knowledge_cutoff": "2024-12-31",
        },
        "run_defaults": {
            "cells_count": len(cells),
            "accepted_futures_per_cell": args.accepted_per_cell,
            "max_total_futures_per_cell": args.accepted_per_cell,
            "max_attempts": 4,
            "terminal_contract": "death_only",
        },
        "prompt": {
            "template_path": "prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md",
            "enum_order": {"mode": "deterministic_shuffle", "seed": f"{CONFIG_ID}_enum_shuffle_v1"},
        },
        "provider": provider_block,
        "models": models,
        "cells": cells,
    }
    config_path = out / f"{CONFIG_ID}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, width=110))

    loaded = load_config(config_path)
    unpinned, replay_bad = [], []
    for loaded_cell in loaded.cells:
        resolved_seed = sb._as_plain_dict(loaded_cell.condition.baseline_seed)
        if not resolved_seed.get("identity_card", {}).get("molecular_profile", {}).get("primary_driver_bucket"):
            unpinned.append(loaded_cell.id)
        try:
            sb.validate_replay_ready_seed(resolved_seed, label=loaded_cell.id)
        except SystemExit as exc:
            replay_bad.append(str(exc))
    if unpinned:
        raise SystemExit(f"[fatal] resolved baseline not pinned for {len(unpinned)} cells (e.g. {unpinned[0]})")
    if replay_bad:
        raise SystemExit(f"[fatal] resolved baseline not replay-ready (first: {replay_bad[0]})")

    for cell_id in [
        "EGFR_1L__S2__H0__osimertinib",
        "EGFR_1L__S5__H1__osimertinib",
        "EGFR_WT_1L__S2__H0__osimertinib",
        "EGFR_WT_1L__S5__H1__osimertinib",
        "EGFR_WT_1L__S5__H1__supportive",
    ]:
        if cell_id in arm_manifest:
            prompt_review_cells.append(arm_manifest[cell_id])

    seed_hashes = {
        condition_id: hashlib.sha256(json.dumps(seed, sort_keys=True).encode()).hexdigest()[:16]
        for condition_id, seed in seed_files.items()
    }
    (out / "arm_manifest.json").write_text(json.dumps(arm_manifest, indent=2))
    (out / "seed_hashes.json").write_text(json.dumps(seed_hashes, indent=2))
    _write_csv(out / "clinical_grid.csv", clinical_grid_rows)
    (out / "prereg_phase1_rubric.yaml").write_text(yaml.safe_dump(PREREG_RUBRIC, sort_keys=False))
    (out / "prompt_review_cells.json").write_text(json.dumps(prompt_review_cells, indent=2))
    (out / "dry_scope_summary.md").write_text(
        "\n".join(
            [
                "# Interventional Phase 1 dry scope",
                "",
                "No provider calls were made.",
                "",
                f"- Contexts: {', '.join(CONTEXTS)}",
                f"- State grid: {len(STATE_GRID)} current states",
                f"- History grid: {len(HISTORY_GRID)} raw pre-treatment histories",
                f"- Arms/context: {', '.join(next(iter(CONTEXTS.values()))['arms'])}",
                f"- Provider: {args.provider}",
                f"- Models: {', '.join(m['short'] for m in models)}",
                f"- Cells: {len(cells)}",
                f"- Accepted futures/cell: {args.accepted_per_cell}",
                f"- Planned accepted futures: {accepted_future_count}",
                f"- Accepted-future cost estimate: ${accepted_cost:.0f}",
                f"- Approximate estimate with 20% retry cushion: ${retry_cost_20:.0f}",
                "",
                "Pre-registered gate:",
                *[f"- {line}" for line in PREREG_RUBRIC["phase1_gate"]],
                "",
                "Generated artifacts:",
                f"- `{config_path}`",
                "- `arm_manifest.json`",
                "- `clinical_grid.csv`",
                "- `prereg_phase1_rubric.yaml`",
                "- `prompt_review_cells.json`",
            ]
        )
        + "\n"
    )

    print("Interventional Phase 1 dry scope: PASS")
    print(f"  contexts: {', '.join(CONTEXTS)}")
    print(f"  grid: {len(STATE_GRID)} states x {len(HISTORY_GRID)} histories x 2 contexts")
    print(f"  provider: {args.provider}")
    print(f"  cells: {len(cells)}  |  models: {', '.join(m['short'] for m in models)}")
    print(f"  futures: {len(cells)} cells x {len(models)} model(s) x {args.accepted_per_cell}/cell = "
          f"{accepted_future_count}")
    print(f"  est. accepted-future cost ~${accepted_cost:.0f}; ~${retry_cost_20:.0f} with 20% retry cushion")
    print(f"  config load_config(): PASS ({len(loaded.cells)} cells)")
    print("  leakage lint: PASS | seeds replay-ready: PASS | resolved seed pinned: PASS")
    print("  shared decision cutoff: PASS")
    print(f"  out: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
