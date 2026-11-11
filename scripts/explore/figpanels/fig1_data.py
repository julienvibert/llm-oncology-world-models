#!/usr/bin/env python3
"""Aggregate tables for Figure 1 panels, from the six-model native atlas (no model call).

Writes to outputs/exploratory_figure_panels/fig1/data/:
  survival_by_family.csv  median generated survival (months from diagnosis to death) per
                          model x tumour family, n, and a 2,000-replicate percentile
                          bootstrap interval resampling trajectories within the cell.
                          Reproduces `median_os_by_family` in
                          outputs/exploratory_atlas_dissection/readout.json (checked).
  driver_entropy.csv      driver diversity per model (Shannon entropy, bits): mean within a
                          tumour code (about five courses per code), with a bootstrap
                          interval over codes; pooled across codes (all courses); and an
                          EQUAL-SAMPLE reference: the same cell sizes drawn at random from
                          the model's courses across all codes, averaged over cells, 2,000
                          draws (mean and 2.5-97.5% range). Entropy of n observations cannot
                          exceed log2(n), so only the equal-sample reference is a fair
                          comparison for the within-code value. Point estimates for within
                          and pooled reproduce exploratory_atlas_archetype/readout.json.
  exemplar.json           one generated course chosen by a fixed rule (Sonnet, lung
                          adenocarcinoma, at most 30 ledger events, contains a progression, a
                          second line and death; lowest trajectory-id hash among those),
                          with its baseline fields, dated events, treatment lines and the
                          replayed tumour-burden and ECOG path. Synthetic data only.

Run:  uv run python scripts/explore/figpanels/fig1_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "scripts"))
from build_atlas_model_dissection import MODEL_LABEL  # noqa: E402
from build_field_trellis import CODE2FAM  # noqa: E402
from composite_tumor_host_phase_space import (  # noqa: E402
    ATLAS_DERIVS,
    MONTH_DAYS,
    accepted,
    enrich_snapshots,
    load_tables,
)

OUT = ROOT / "outputs" / "exploratory_figure_panels" / "fig1" / "data"
N_BOOT = 2000
SEED = 20260923


def survival_table(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["family"] = s["tumor_oncotree_code"].astype(str).map(CODE2FAM)
    s["model"] = s["model_short"].astype(str).map(lambda m: MODEL_LABEL.get(m, m))
    os_tab = (s.groupby(["model", "_gid"])["month"].max().reset_index()
              .merge(s.groupby("_gid")["family"].first().rename("family"), on="_gid")
              .dropna(subset=["family"]))
    all_gids = s["_gid"].nunique()
    mapped_codes = s.dropna(subset=["family"])["tumor_oncotree_code"].nunique()
    all_codes = s["tumor_oncotree_code"].nunique()
    print(f"survival table: {os_tab['_gid'].nunique()} of {all_gids} courses, "
          f"{mapped_codes} of {all_codes} codes mapped to 8 families")
    rng = np.random.default_rng(SEED)
    rows = []
    for (fam, model), g in os_tab.groupby(["family", "model"]):
        v = g["month"].to_numpy(float)
        boots = np.median(v[rng.integers(0, len(v), (N_BOOT, len(v)))], axis=1)
        rows.append({"family": fam, "model": model, "n": len(v),
                     "median": float(np.median(v)),
                     "ci_low": float(np.quantile(boots, 0.025)),
                     "ci_high": float(np.quantile(boots, 0.975))})
    return pd.DataFrame(rows)


def _entropy_bits(values: pd.Series) -> float:
    p = values.value_counts(normalize=True).to_numpy()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def entropy_table() -> pd.DataFrame:
    import build_paper_state_map as smap
    b = smap.load_reference_tables(smap.REFERENCE_RUNS)["baselines"].groupby("_gid").first()
    b = b[["model_short", "tumor_oncotree_code", "primary_driver_bucket"]].astype(str)
    rng = np.random.default_rng(SEED)
    rows = []
    for model, g in b.groupby("model_short"):
        groups = [gc for _c, gc in g.groupby("tumor_oncotree_code") if len(gc) >= 2]
        cells = np.array([_entropy_bits(gc["primary_driver_bucket"]) for gc in groups])
        sizes = np.array([len(gc) for gc in groups])
        wb = rng.choice(cells, size=(N_BOOT, len(cells)), replace=True).mean(axis=1)
        v = g["primary_driver_bucket"].to_numpy()
        matched = np.array([
            np.mean([_entropy_bits(pd.Series(v[rng.choice(len(v), k, replace=False)]))
                     for k in sizes])
            for _ in range(N_BOOT)])
        pb = [_entropy_bits(pd.Series(v[rng.integers(0, len(v), len(v))])) for _ in range(N_BOOT)]
        rows.append({"model": MODEL_LABEL.get(model, model), "n_codes": len(cells),
                     "n_trajectories": len(g),
                     "within": float(cells.mean()),
                     "within_low": float(np.quantile(wb, 0.025)),
                     "within_high": float(np.quantile(wb, 0.975)),
                     "pooled": _entropy_bits(g["primary_driver_bucket"]),
                     "pooled_low": float(np.quantile(pb, 0.025)),
                     "pooled_high": float(np.quantile(pb, 0.975)),
                     "matched": float(matched.mean()),
                     "matched_low": float(np.quantile(matched, 0.025)),
                     "matched_high": float(np.quantile(matched, 0.975)),
                     "median_cell_size": int(np.median(sizes)),
                     "max_bits_at_median_size": float(np.log2(np.median(sizes)))})
    return pd.DataFrame(rows)


def _read(name: str) -> pd.DataFrame:
    frames = []
    for d in ATLAS_DERIVS:
        p = d / name
        if p.exists():
            x = accepted(pd.read_csv(p, low_memory=False))
            x["_gid"] = d.parent.name + "::" + x["trajectory_id"].astype(str)
            frames.append(x)
    return pd.concat(frames, ignore_index=True)


def exemplar(s: pd.DataFrame) -> dict:
    ev = _read("event_rows.csv")
    tx = _read("treatment_rows.csv")
    base = _read("baseline_rows.csv")
    cand = ev[(ev["model_short"] == "claude_sonnet5") & (ev["tumor_oncotree_code"] == "LUAD")]
    ok = []
    for gid, g in cand.groupby("_gid"):
        types = set(g["event_type"].astype(str))
        lines = tx.loc[tx["_gid"] == gid, "line_number"]
        if len(g) <= 30 and "progression" in types and "death" in types and lines.max() >= 2:
            ok.append(gid)
    if not ok:
        raise SystemExit("no exemplar matches the rule")
    gid = min(ok, key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    g_ev = ev[ev["_gid"] == gid].sort_values("event_index")
    g_tx = tx[tx["_gid"] == gid].sort_values("course_index")
    g_b = base[base["_gid"] == gid].iloc[0]
    g_s = s[s["_gid"] == gid].sort_values(["month", "snapshot_index"])
    events = []
    for r in g_ev.itertuples(index=False):
        label = str(r.event_type)
        for extra in ("regimen", "reason"):
            val = getattr(r, extra, None)
            if isinstance(val, str) and val and val != "nan":
                label = f"{label}: {val}"
                break
        events.append({"month": round(float(r.days_from_diagnosis) / MONTH_DAYS, 2),
                       "type": str(r.event_type), "label": label})
    lines = [{"line": int(r.line_number), "regimen": str(r.regimen_name),
              "start": round((pd.Timestamp(r.start_date) - pd.Timestamp(g_b.get("diagnosis_date", r.start_date))).days / MONTH_DAYS, 2)
              if "diagnosis_date" in g_b.index else None,
              "duration_months": round(float(r.duration_days) / MONTH_DAYS, 2)
              if pd.notna(r.duration_days) else None}
             for r in g_tx.itertuples(index=False)]
    return {
        "gid_hash": hashlib.sha256(gid.encode()).hexdigest()[:12],
        "model": "Claude Sonnet 5", "oncotree": "LUAD",
        "baseline": {k: (None if pd.isna(g_b.get(k)) else g_b.get(k))
                     for k in ("stage", "primary_driver", "age_at_diagnosis", "sex",
                               "frailty_band", "smoking_status")},
        "events": events,
        "lines": lines,
        "path": [{"month": round(float(r.month), 2),
                  "burden": None if pd.isna(r.tumour_burden_score) else float(r.tumour_burden_score),
                  "ecog": None if pd.isna(r.ecog) else float(r.ecog),
                  "terminal": bool(r.terminal)}
                 for r in g_s.itertuples(index=False)],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s = enrich_snapshots(load_tables())
    surv = survival_table(s)
    ref = json.loads((ROOT / "outputs/exploratory_atlas_dissection/readout.json").read_text())
    worst = 0.0
    for model, fams in ref["median_os_by_family"].items():
        for fam, med in fams.items():
            if med is None or (isinstance(med, float) and np.isnan(med)):
                continue
            row = surv[(surv.model == model) & (surv.family == fam)]
            worst = max(worst, abs(float(row["median"].iloc[0]) - med))
    print(f"max |median - readout| = {worst:.3f} months")
    if worst > 0.01:
        raise SystemExit("medians do not reproduce the dissection readout")
    surv.to_csv(OUT / "survival_by_family.csv", index=False)
    ent = entropy_table()
    arch = json.loads((ROOT / "outputs/exploratory_atlas_archetype/readout.json").read_text())
    inv = {v: k for k, v in MODEL_LABEL.items()}
    ref_e = {r["model"]: r for r in arch["entropy"]}
    werr = max(abs(r.within - ref_e[inv[r.model]]["within_code_entropy"])
               + abs(r.pooled - ref_e[inv[r.model]]["pooled_entropy"]) for r in ent.itertuples())
    print(f"max |entropy - readout| = {werr:.4f} bits")
    if werr > 1e-6:
        raise SystemExit("entropies do not reproduce the archetype readout")
    ent.to_csv(OUT / "driver_entropy.csv", index=False)
    pd.DataFrame(arch["archetype"]).assign(
        model=lambda d: d["model"].map(lambda m: MODEL_LABEL.get(m, m))
    ).to_csv(OUT / "modal_share.csv", index=False)
    ex = exemplar(s)
    (OUT / "exemplar.json").write_text(json.dumps(ex, indent=2, default=str) + "\n")
    print(f"wrote {OUT}; exemplar {ex['gid_hash']} with {len(ex['events'])} events, "
          f"{len(ex['lines'])} lines, {len(ex['path'])} snapshots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
