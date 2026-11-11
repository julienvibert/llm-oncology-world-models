"""Rescore the scenario benchmark (Extended Data Fig. 4a,b) from saved sentinel runs with the
ORIGINAL scoring functions.

Imports plot_sentinel_trust_map / score_sentinel_benchmark unmodified, feeds them the
reconstructed manifest (metadata keys dropped), reproduces main()'s run selection and
correctness() loop without drawing, and adds denominators. Writes to outputs/ed6_provenance_2026-10-14/.
"""
import glob
import json
import os
import sys

# The original scorer breaks an exact tie through a Python set (discrepancy D1); fix the hash
# seed so reruns are deterministic and leave tracked outputs unchanged.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "outputs" / "ed6_provenance_2026-10-14"
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
import plot_sentinel_trust_map as P  # noqa: E402
from score_sentinel_benchmark import PREFIX_CONTEXTS, _meds  # noqa: E402

man_raw = json.loads((ROOT / "scripts/explore/sentinel_battery_manifest_reconstructed_2026-10-14.json").read_text())
manifest = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in man_raw.items() if not k.startswith("_")}

spec = yaml.safe_load(P.SPEC.read_text())["contexts"]
all_runs = [Path(p) for p in sorted(glob.glob(str(ROOT / "runs" / "sentinel_usecase_*")))]
runs = [r for r in all_runs if (r / "derivation" / "endpoint_rows.csv").exists()]
if any("post_osi_explicit" in r.name for r in runs):
    runs = [r for r in runs if not r.name.startswith("sentinel_usecase_egfr_post_osi_v")]
print("runs used:", [r.name for r in runs])
print("runs excluded:", [r.name for r in all_runs if r not in runs])
df = P.combined_table(runs, manifest)

present = [c for c in spec if (df["context"] == c).any()]
dynamic = {c for c in present if spec[c].get("core_signal") in ("history_delta", "host_intensity")}
rows = [c for c in present if c not in dynamic] + [c for c in present if c in dynamic]
models = [m for m in P.MODEL_ORDER if (df["model"] == m).any()]

out = []
for c in rows:
    for m in models:
        s, g = P.correctness(c, spec[c], df, m)
        out.append({"context_id": c, "model": m, "primary": s, "guardrail": g})
S = pd.DataFrame(out)

# ---- denominators from the raw endpoint rows (same filters as combined_table)
frames = []
for r in runs:
    e = pd.read_csv(r / "derivation" / "endpoint_rows.csv")
    e["run"] = r.name
    frames.append(e)
E = pd.concat(frames, ignore_index=True)
pol2 = {m["policy_id"]: (m["context"], m["arm_key"]) for m in manifest.values()}
E = E[E["policy_id"].isin(pol2)].copy()
E["context"] = E["policy_id"].map(lambda p: pol2[p][0])
E["arm_key"] = E["policy_id"].map(lambda p: pol2[p][1])
l1 = pd.to_numeric(E["pfs_line_1_days"], errors="coerce")
l2 = pd.to_numeric(E["pfs_line_2_days"], errors="coerce")
E["pfs_days_forced"] = np.where(E["context"].isin(PREFIX_CONTEXTS), l2, l1)
E["acc"] = E["accepted"].astype(bool)
den = (E.groupby(["context", "condition_id", "arm_key", "model_short"])
       .agg(rows=("trajectory_id", "size"), accepted=("acc", "sum"),
            pfs_nonnull_accepted=("pfs_days_forced", lambda s: int(s[E.loc[s.index, "acc"]].notna().sum())))
       .reset_index())
den.to_csv(HERE / "denominators_per_arm.csv", index=False)

# ---- per-check detail (which comparisons were scored)
checks = []
for c in rows:
    ctx = spec[c]
    for m in models:
        if c == "HGSOC_relapse":
            hg = P.score_hgsoc_delta(df, m)
            checks += [{"context_id": c, "model": m, "check": f"platinum sens-res delta {hg['platinum_delta']:.2f} mo > {P.HGSOC_DELTA_MIN}",
                        "pass": hg["platinum_delta"] > P.HGSOC_DELTA_MIN},
                       {"context_id": c, "model": m, "check": "platinum not top (resistant)", "pass": hg["platinum_not_top_resistant"]}]
        elif c == "pancreas_fit_frail":
            pa = P.score_pancreas_host(df, m)
            checks += [{"context_id": c, "model": m, "check": "folfirinox top (fit)", "pass": pa["folfirinox_top_fit"]},
                       {"context_id": c, "model": m, "check": "folfirinox not top (frail)", "pass": pa["folfirinox_not_top_frail"]}]
        else:
            mp = _meds(df, c, m, "pfs_mo")
            bsc = ctx.get("bsc")
            for a, b in ctx.get("expected_signs", []):
                if a == bsc or b == bsc:
                    continue
                if a in mp and b in mp:
                    checks.append({"context_id": c, "model": m, "check": f"{a} {mp[a]:.1f} > {b} {mp[b]:.1f}", "pass": mp[a] > mp[b]})
                else:
                    checks.append({"context_id": c, "model": m, "check": f"{a} > {b} NOT SCORED (missing PFS)", "pass": None})
C = pd.DataFrame(checks)
C.to_csv(HERE / "checks_detail.csv", index=False)

# ---- compare with saved table
T = pd.read_csv(ROOT / "outputs/exploratory_sentinel_benchmark/sentinel_trust_map_table.csv")
cmp = []
for _, r in S.iterrows():
    t = T[T.context_id == r.context_id].iloc[0]
    saved = t[f"{r.model}_primary_pct"]
    saved_g = t[f"{r.model}_guardrail_pct"]
    new = round(100 * r.primary) if pd.notna(r.primary) else None
    new_g = None if r.guardrail is None or (isinstance(r.guardrail, float) and np.isnan(r.guardrail)) else 100 * r.guardrail
    n_ch = int(C[(C.context_id == r.context_id) & (C.model == r.model) & C["pass"].notna()].shape[0])
    n_pass = int(C[(C.context_id == r.context_id) & (C.model == r.model) & (C["pass"] == True)].shape[0])  # noqa: E712
    cmp.append({"context_id": r.context_id, "model": r.model, "saved_primary": saved, "rescored_primary": new,
                "checks_passed": n_pass, "checks_scored": n_ch,
                "saved_guardrail": saved_g, "rescored_guardrail": new_g,
                "match_primary": saved == new,
                "match_guardrail": (pd.isna(saved_g) and new_g is None) or (new_g is not None and saved_g == new_g)})
CMP = pd.DataFrame(cmp)
CMP.to_csv(HERE / "comparison.csv", index=False)
print(CMP.to_string(index=False))
print("primary mismatches:", int((~CMP.match_primary).sum()), " guardrail mismatches:", int((~CMP.match_guardrail).sum()))

# axis means
ro = json.loads((ROOT / "outputs/paper_interventional_trust_map/readout.json").read_text())["sentinel"]["axis_means"]
for m in models:
    st = S[(S.model == m) & (~S.context_id.isin(dynamic))]["primary"].mean() * 100
    dy = S[(S.model == m) & (S.context_id.isin(dynamic))]["primary"].mean() * 100
    print(f"{m}: static {st:.4f} (readout {ro[m]['static_mean_pct']:.4f})  dynamic {dy:.4f} (readout {ro[m]['dynamic_mean_pct']:.4f})")
tot_pass = int((C["pass"] == True).sum())  # noqa: E712
tot = int(C["pass"].notna().sum())
print("total checks passed/scored:", tot_pass, "/", tot)
for ax_, ids in (("static", [c for c in rows if c not in dynamic]), ("dynamic", sorted(dynamic))):
    sub = C[C.context_id.isin(ids) & C["pass"].notna()]
    print(ax_, "checks passed/scored by model:", sub.groupby("model")["pass"].agg(["sum", "count"]).to_dict("index"))
