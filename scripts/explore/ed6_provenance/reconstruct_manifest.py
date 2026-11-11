"""Reconstruct the lost sentinel arm_manifest.json from the run config.yaml files.

Follows build_sentinel_battery.build_cell (manifest entry keyed by cell id with
arm_key, policy_id, regimen_label, intervention_fixed_in_prefix, context,
condition_id) + main() (adds role = spec[context].arms[arm_key]).
policy_id = f"{context}__{arm_key}" (builder line 364), so context/arm_key are
recovered by splitting the policy_id that each run config records.
"""
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(sys.argv[1])
spec = yaml.safe_load((ROOT / "scripts/explore/sentinel_benchmark_spec_v1.yaml").read_text())["contexts"]
runs = sorted((ROOT / "runs").glob("sentinel_usecase_*"))
entries: dict = {}
sources: dict = {}
for r in runs:
    cfg = yaml.safe_load((r / "config.yaml").read_text())
    for cell in cfg["cells"]:
        pid = cell["policy_id"]
        ctx, arm = pid.split("__", 1)
        assert ctx in spec, (r.name, pid)
        assert arm in spec[ctx]["arms"], (r.name, pid)
        m = re.search(r"systemic_treatment_start = (.+?)\. Do not", cell["operator_constraints"])
        label = m.group(1) if m else "best supportive care"
        e = {"arm_key": arm, "policy_id": pid, "regimen_label": label,
             "intervention_fixed_in_prefix": arm != "supportive",
             "context": ctx, "condition_id": cell["condition_id"],
             "role": spec[ctx]["arms"][arm]}
        if cell["id"] in entries:
            assert entries[cell["id"]] == e, cell["id"]
            sources[cell["id"]].append(r.name)
        else:
            entries[cell["id"]] = e
            sources[cell["id"]] = [r.name]
for k in entries:
    entries[k]["_source_runs"] = sources[k]
header = {
    "_reconstruction": {
        "status": "RECONSTRUCTION, not the original file",
        "date": "2026-10-14",
        "replaces": "/private/tmp/sentinel_reasoning_battery_v5_remaining_source/arm_manifest.json (lost)",
        "method": ("Entry schema from scripts/explore/build_sentinel_battery.py build_cell()+main(). "
                   "policy_id read from runs/sentinel_usecase_*/config.yaml cells; context and arm_key "
                   "= policy_id split on '__' (builder: policy_id = f'{context}__{arm_key}'); role = "
                   "sentinel_benchmark_spec_v1.yaml contexts[context].arms[arm_key]; regimen_label parsed "
                   "from the cell operator_constraints text (supportive = 'best supportive care'); "
                   "intervention_fixed_in_prefix = arm != supportive (builder logic; not recorded in run "
                   "configs). _source_runs lists the run folders that contain the cell id."),
        "scope": "Superset: every cell of all 11 sentinel_usecase_* run folders (original likely held a subset).",
        "use": ("Scoring uses only policy_id -> (context, arm_key). Keys starting with '_' are metadata and "
                "must be dropped before passing this file to plot_sentinel_trust_map.py / "
                "score_sentinel_benchmark.py (they iterate manifest.values()). See "
                "docs/run_provenance/sentinel_benchmark_scoring.md."),
    }
}
OUT.write_text(json.dumps(header | entries, indent=2) + "\n")
print(len(entries), "cell entries;", len({e["policy_id"] for e in entries.values()}), "policy ids")
