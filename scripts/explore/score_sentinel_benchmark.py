#!/usr/bin/env python3
"""Score a generated sentinel CF run against the reasoning-benchmark spec.

Reads sentinel_benchmark_spec_v1.yaml + the battery's arm_manifest.json (policy_id ->
arm/context) + a run's derivation, and scores each (context, model) in THREE SEPARATE
TIERS (never merged):

  primary   pre-registered pairwise expected_signs, on FORCED-LINE PFS ...... the biology
  guardrail `not_top`: wrong-target arms must not rank first (by PFS) .... hallucination
  control   BSC ranked last, on OS, ONLY where active tx is clearly appropriate

PRIMARY ENDPOINT = forced-line PFS, NOT OS. Each arm only fixes the FIRST line; everything
downstream is free, so the model rescues (e.g. gives the wrong drug, it fails at ~2mo, it
switches to the right drug) and OS washes the intervention out. PFS of the forced line
isolates the model's belief about THAT drug. OS is kept as a SECONDARY readout (the OS-PFS
gap measures imagined salvage). Forced line:
  * 1L contexts               -> pfs_line_1_days
  * history contexts (post-osi, HGSOC relapse; the prefix consumes L1) -> pfs_line_2_days
BSC/supportive has no forced treatment => no PFS; its "ranked last" control is scored on OS.

Core-signal primitives (all on PFS): direction (pairwise signs) · biomarker_flip (the
wrong-target arm's PFS must be LOWER here than in its flip partner = uses the biomarker) ·
history_delta (HGSOC platinum PFS sensitive>resistant; post-osi chemo>continue_osi) ·
host_intensity (pancreas FOLFIRINOX PFS top in fit / not-top in frail).

Real-derived side stays aggregate/non-git; these seeds are synthetic so scores are git-ok.

  uv run python scripts/explore/score_sentinel_benchmark.py --self-test          # no run needed
  uv run python scripts/explore/score_sentinel_benchmark.py --run <run_root> \\
        --manifest /private/tmp/sentinel_reasoning_battery_v1/arm_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
SPEC = Path(__file__).resolve().parent / "sentinel_benchmark_spec_v1.yaml"
OUT = ROOT / "outputs" / "exploratory_sentinel_benchmark"
# contexts whose forced arm is the line AFTER the fixed prefix (prefix consumes L1)
PREFIX_CONTEXTS = {"EGFR_post_progression", "HGSOC_relapse"}


# --------------------------------------------------------------------- IO layer
def endpoint_table(run: Path, manifest: dict) -> pd.DataFrame:
    """Per (context, condition_id, arm, model): median forced-line PFS + OS (months).

    endpoint_rows.csv carries per-trajectory pfs_line_{1,2}_days / os_days plus
    condition_id / policy_id / model_short. arm_key + context come from the arm_manifest
    (policy_id -> arm/context). Forced line = L2 for prefix contexts, else L1.
    """
    e = pd.read_csv(run / "derivation" / "endpoint_rows.csv")
    if "accepted" in e:
        e = e[e["accepted"].astype(bool)]
    pol2 = {m["policy_id"]: (m["context"], m["arm_key"]) for m in manifest.values()}
    e = e[e["policy_id"].isin(pol2)].copy()
    e["context"] = e["policy_id"].map(lambda p: pol2[p][0])
    e["arm_key"] = e["policy_id"].map(lambda p: pol2[p][1])
    e["model"] = e["model_short"]
    l1 = pd.to_numeric(e["pfs_line_1_days"], errors="coerce")
    l2 = pd.to_numeric(e["pfs_line_2_days"], errors="coerce")
    e["pfs_mo"] = np.where(e["context"].isin(PREFIX_CONTEXTS), l2, l1) / 30.4
    e["os_mo"] = pd.to_numeric(e["os_days"], errors="coerce") / 30.4
    return e.groupby(["context", "condition_id", "arm_key", "model"], as_index=False)[["pfs_mo", "os_mo"]].median()


# ----------------------------------------------------------------- pure scoring
def _meds(df: pd.DataFrame, context: str, model: str, col: str, *, condition: str | None = None) -> dict:
    d = df[(df["context"] == context) & (df["model"] == model)]
    if condition is not None:
        d = d[d["condition_id"] == condition]
    return {a: v for a, v in zip(d["arm_key"], d[col]) if pd.notna(v)}


def score_tiers(ctx: dict, mp: dict, mo: dict) -> dict:
    """Three tiers: primary/guardrail/two_attractor on forced-line PFS (`mp`); control on OS (`mo`)."""
    bsc = ctx.get("bsc")
    active = set(mp)  # active arms have a forced-line PFS; BSC does not
    # primary: active-vs-active pairwise signs on PFS. BSC-involving signs are the trivial
    # "knows to treat" check and belong to the control tier, so they are dropped here.
    signs = [(a, b) for a, b in ctx.get("expected_signs", [])
             if a != bsc and b != bsc and a in active and b in active]
    primary = float(np.mean([mp[a] > mp[b] for a, b in signs])) if signs else None
    top = max(active, key=lambda a: mp[a]) if active else None
    nt = [a for a in ctx.get("not_top", []) if a in active]
    guardrail = float(np.mean([a != top for a in nt])) if nt else None
    two = ctx.get("two_attractor")
    two_ok = None
    if two and set(two) <= active:
        weakest = min(mp[a] for a in active)
        two_ok = bool(all(mp[a] > weakest for a in two))
    control = None
    if bsc in mo and not ctx.get("bsc_suspended", False):
        control = bool(min(mo, key=mo.get) == bsc)
    soc = ctx.get("soc")
    return {"primary": primary, "guardrail": guardrail, "control": control, "two_attractor": two_ok,
            "soc_top": bool(soc in active and top == soc)}


def score_flip_delta(ctx_id: str, ctx: dict, df: pd.DataFrame, model: str, spec: dict) -> float | None:
    """Wrong-target arm's forced-line PFS should be LOWER here than in its flip partner."""
    partner_id = ctx.get("flip_pair_with")
    if not partner_id or partner_id not in spec:
        return None
    wrong = [a for a, role in ctx.get("arms", {}).items() if role == "wrong_target"]
    if not wrong:
        return None
    here = _meds(df, ctx_id, model, "pfs_mo").get(wrong[0])
    there = _meds(df, partner_id, model, "pfs_mo").get(wrong[0])
    if here is None or there is None:
        return None
    return float(there - here)  # >0 = the arm is worth less (shorter PFS) in the flipped context


def score_hgsoc_delta(df: pd.DataFrame, model: str) -> dict | None:
    """platinum forced-line PFS(sensitive) - (resistant) > 0; platinum not top in resistant."""
    sens = _meds(df, "HGSOC_relapse", model, "pfs_mo", condition="hgsoc_platinum_sensitive")
    res = _meds(df, "HGSOC_relapse", model, "pfs_mo", condition="hgsoc_platinum_resistant")
    if "platinum" not in sens or "platinum" not in res:
        return None
    return {"platinum_delta": float(sens["platinum"] - res["platinum"]),
            "platinum_not_top_resistant": bool(res and max(res, key=res.get) != "platinum")}


def score_pancreas_host(df: pd.DataFrame, model: str) -> dict | None:
    """FOLFIRINOX forced-line PFS top in fit; not top in frail."""
    fit = _meds(df, "pancreas_fit_frail", model, "pfs_mo", condition="pancreas_fit")
    frail = _meds(df, "pancreas_fit_frail", model, "pfs_mo", condition="pancreas_frail")
    if "folfirinox" not in fit or "folfirinox" not in frail:
        return None
    return {"folfirinox_top_fit": bool(max(fit, key=fit.get) == "folfirinox"),
            "folfirinox_not_top_frail": bool(max(frail, key=frail.get) != "folfirinox")}


def score_run(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    models = sorted(df["model"].dropna().unique())
    rows = []
    for cid, ctx in spec.items():
        if not (df["context"] == cid).any():
            continue
        dyn = "seed_variants" in ctx
        for m in models:
            if dyn:
                for cond in sorted(df[df["context"] == cid]["condition_id"].unique()):
                    ctx_v = dict(ctx, bsc_suspended=ctx.get("bsc_suspended", False) or ("frail" in cond))
                    mp = _meds(df, cid, m, "pfs_mo", condition=cond)
                    mo = _meds(df, cid, m, "os_mo", condition=cond)
                    rows.append({"context": cid, "condition": cond, "model": m, **score_tiers(ctx_v, mp, mo)})
            else:
                mp, mo = _meds(df, cid, m, "pfs_mo"), _meds(df, cid, m, "os_mo")
                extra = {}
                if ctx.get("core_signal") == "biomarker_flip":
                    extra["flip_delta"] = score_flip_delta(cid, ctx, df, m, spec)
                rows.append({"context": cid, "condition": cid, "model": m, **score_tiers(ctx, mp, mo), **extra})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- self-test
def _mock_frame() -> pd.DataFrame:
    """Mock (context, arm) -> forced-line PFS + OS a competent model would produce.
    Note the deliberate OS/PFS divergence: the wrong drug has a SHORT PFS but similar OS
    (imagined rescue). Scoring on PFS must recover the biology; OS alone would not."""
    r = []

    def add(ctx, cond, arm, pfs, os):
        r.append({"context": ctx, "condition_id": cond, "arm_key": arm, "model": "good",
                  "pfs_mo": pfs, "os_mo": os})
    # EGFR_1L: osi PFS >> pembro PFS (OS close = rescue). supportive: no PFS.
    add("EGFR_1L", "egfr_mut_luad", "osimertinib", 28, 48)
    add("EGFR_1L", "egfr_mut_luad", "pembrolizumab", 3, 43)
    add("EGFR_1L", "egfr_mut_luad", "supportive", np.nan, 12)
    # EGFR_WT flip: chemo PFS > osi PFS (osi has no target -> short PFS); osi not top by PFS
    add("EGFR_WT_1L", "egfr_wt_luad", "chemo", 11, 32)
    add("EGFR_WT_1L", "egfr_wt_luad", "osimertinib", 2.6, 25)
    add("EGFR_WT_1L", "egfr_wt_luad", "supportive", np.nan, 11)
    # HGSOC: platinum PFS helps in sensitive, not resistant
    for arm, pfs, os in [("platinum", 20, 24), ("non_platinum", 10, 14), ("parp", 8, 10), ("supportive", np.nan, 6)]:
        add("HGSOC_relapse", "hgsoc_platinum_sensitive", arm, pfs, os)
    for arm, pfs, os in [("platinum", 7, 9), ("non_platinum", 11, 12), ("parp", 6, 8), ("supportive", np.nan, 6)]:
        add("HGSOC_relapse", "hgsoc_platinum_resistant", arm, pfs, os)
    # pancreas host: folfirinox PFS top in fit, not in frail
    for arm, pfs, os in [("folfirinox", 12, 11), ("gem_nab", 8, 9), ("gem", 6, 7), ("supportive", np.nan, 3)]:
        add("pancreas_fit_frail", "pancreas_fit", arm, pfs, os)
    for arm, pfs, os in [("folfirinox", 3, 4), ("gem_nab", 6, 6), ("gem", 5, 6), ("supportive", np.nan, 5)]:
        add("pancreas_fit_frail", "pancreas_frail", arm, pfs, os)
    return pd.DataFrame(r)


def _self_test() -> int:
    spec = yaml.safe_load(SPEC.read_text())["contexts"]
    df = _mock_frame()
    S = score_run(df, spec)
    egfr = S[S.context == "EGFR_1L"].iloc[0]
    wt = S[S.context == "EGFR_WT_1L"].iloc[0]
    assert egfr["primary"] == 1.0 and egfr["soc_top"], egfr           # osi>pembro by PFS, osi top
    assert wt["primary"] == 1.0 and wt["guardrail"] == 1.0, wt        # chemo>osi by PFS, osi not top
    assert egfr["control"] and wt["control"], (egfr, wt)              # BSC last by OS
    assert score_flip_delta("EGFR_WT_1L", spec["EGFR_WT_1L"], df, "good", spec) > 0
    hg = score_hgsoc_delta(df, "good")
    assert hg["platinum_delta"] > 0 and hg["platinum_not_top_resistant"], hg
    pan = score_pancreas_host(df, "good")
    assert pan["folfirinox_top_fit"] and pan["folfirinox_not_top_frail"], pan
    frail = S[(S.context == "pancreas_fit_frail") & (S.condition == "pancreas_frail")].iloc[0]
    assert frail["control"] is None, frail                            # BSC-last suspended in frail
    print("[self-test] PASS — PFS-primary tiers, OS control, flip Δ, HGSOC Δ, pancreas host all correct")
    print(S.to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test or not args.run:
        return _self_test()

    OUT.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(SPEC.read_text())["contexts"]
    manifest = json.loads(args.manifest.read_text())
    df = endpoint_table(args.run, manifest)
    df.round(1).to_csv(OUT / "sentinel_endpoint_medians.csv", index=False)  # PFS + OS, secondary
    S = score_run(df, spec)
    S.to_csv(OUT / "sentinel_scores.csv", index=False)

    deltas = []
    for m in sorted(df["model"].dropna().unique()):
        hg, pan = score_hgsoc_delta(df, m), score_pancreas_host(df, m)
        if hg:
            deltas.append({"model": m, "primitive": "hgsoc_platinum", **hg})
        if pan:
            deltas.append({"model": m, "primitive": "pancreas_host", **pan})
    if deltas:
        pd.DataFrame(deltas).to_csv(OUT / "sentinel_dynamic_deltas.csv", index=False)

    # The figure lives in plot_sentinel_trust_map.py (combined, static/dynamic banded).
    # This scorer only writes CSVs so a single-run / all-NaN-primary context never
    # produces a misleading "empty" heatmap.
    print(f"[sentinel] scored {S['context'].nunique()} contexts x {S['model'].nunique()} models (PFS-primary)")
    print(f"[done] wrote {OUT / 'sentinel_scores.csv'} (+ endpoint medians, dynamic deltas)")
    print("       -> render the figure with: uv run python scripts/explore/plot_sentinel_trust_map.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
