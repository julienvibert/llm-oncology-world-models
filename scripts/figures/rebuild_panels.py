"""Draw every panel of scripts/figures/layouts.json, one script per panel, and report which ones could not be drawn.

Most panels are drawn by scripts/figures/panels/<id>.py or by a script that names the id in quotes; the pages below
build their id in the script, so they are mapped here. Without GENIE-BPC or PRIME access, the panels that need them
fail and are listed; every other panel must be drawn.

Run: uv run python scripts/figures/rebuild_panels.py [--public] [--only id ...]
--public: skip the panels that need GENIE-BPC or PRIME data (RESTRICTED, listed below) and fail only if another
panel fails. Without it, every panel is attempted and the exit code is 1 if any script failed.
Output: one line per script (ok, failed or skipped) and a summary.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "scripts" / "figures" / "panels"
# panel ids built inside their script (f-strings): id -> (script, arguments)
# cand_css_pages.py draws one group of pages per argument, so that a restricted page cannot stop a synthetic one
DYNAMIC = {
    "cand_final_genie_nsclc": ("cand_css_pages.py", ["genie"]),
    "cand_final_genie_crc": ("cand_css_pages.py", ["genie"]),
    "prop_fig5_genie": ("cand_css_pages.py", ["genie"]),
    "cand_final_prime_wt": ("cand_css_pages.py", ["prime"]),
    "cand_final_prime_mut": ("cand_css_pages.py", ["prime"]),
    "prop_fig6_prime": ("cand_css_pages.py", ["prime"]),
    "cand_fig3_four": ("cand_css_pages.py", ["egfr_four"]),
    "cand_final_atlas": ("cand_css_pages.py", ["atlas"]),
    "cand_cc_css_plm_genie_nsclc": ("cand_cc_css_plm_genie.py", ["NSCLC"]),
}

# panels that need GENIE-BPC or PRIME data, or tables derived from them (checked on a fresh checkout)
RESTRICTED = frozenset({
    "cand_cc_css_plm_genie_nsclc",
    "cand_final_genie_crc",
    "cand_final_genie_nsclc",
    "cand_final_prime_mut",
    "cand_final_prime_wt",
    "ed07a_km_at_risk",
    "ed07b_gap_by_horizon",
    "ed07e_donor_agreement",
    "ed08a_block_contrasts",
    "ed08b_cancer_contrasts",
    "ed08c_sensitivity",
    "ed08d_regimen_timing",
    "ed09a_forest_hr",
    "f4a_design",
    "f4b_early_deaths",
    "f4c_survival_12m",
    "f4d_bias_deaths",
    "f4e_fan_coverage",
    "f4f_c_index",
    "f4g_calibration",
    "f4h_joint_score",
    "f4i_treatment",
    "f5a_case_fans",
    "n14_support_gradient",  # aggregates of the GENIE-BPC registry experiment (n14_support_gradient_v1.py)
    "f5c_course_disagreement",
    "f5d_course_contrasts",
    "f6a_prime_rmst",
    "n12a_prime_km",
    "n2_state_occupancy_all_arms",
    "n2_state_occupancy_main",
    "n6_support_ladder",
    "n8_prime_time_in_state",
    "prop_fig5_genie",
    "prop_fig6_prime",
    "si08a_lung_fans",
    "si09a_colorectal_fans",
})


def script_for(pid: str) -> tuple[str, tuple[str, ...]]:
    if pid in DYNAMIC:
        s, args = DYNAMIC[pid]
        return s, tuple(args)
    if (P / f"{pid}.py").exists():
        return f"{pid}.py", ()
    hits = [s.name for s in sorted(P.glob("[a-z]*.py")) if f'"{pid}"' in s.read_text()]
    if not hits:
        raise SystemExit(f"no script draws panel {pid}")
    return hits[0], ()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--public", action="store_true", help="skip the restricted panels")
    a = ap.parse_args()
    layout = json.loads((ROOT / "scripts/figures/layouts.json").read_text())
    ids = sorted({p["id"] for k, v in layout.items() if not k.startswith("_") for p in v["panels"]})
    if a.only:
        ids = [i for i in ids if i in a.only]
    jobs: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for i in ids:
        jobs.setdefault(script_for(i), []).append(i)
    failed, skipped = [], []
    for (s, args), panels in sorted(jobs.items()):
        restricted = [p for p in panels if p in RESTRICTED]
        if restricted and len(restricted) != len(panels):
            raise SystemExit(f"{s} {' '.join(args)} draws restricted and public panels together: {panels}")
        if a.public and restricted:
            skipped += panels
            print(f"skip   {s} {' '.join(args)} -> {', '.join(panels)} (needs GENIE-BPC or PRIME data)")
            continue
        r = subprocess.run([sys.executable, str(P / s), *args], cwd=ROOT, capture_output=True, text=True)
        ok = r.returncode == 0
        print(f"{'ok    ' if ok else 'FAILED'} {s} {' '.join(args)} -> {', '.join(panels)}")
        if not ok:
            failed.append((s, panels, (r.stderr.strip().splitlines() or [""])[-1][-200:]))
    print(f"{len(jobs)} scripts for {len(ids)} panels; skipped (restricted): {len(skipped)}; failed: {len(failed)}")
    for s, panels, err in failed:
        print(f"  {s}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
