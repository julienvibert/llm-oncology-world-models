"""Crossover to osimertinib in the EGFR experiment: who receives it, when, and first-line control with and without an early switch.

Output: outputs/exploratory_cand_css_four_arm/crossover_numbers.csv (aggregate). Run: uv run python -W ignore scripts/explore/cand_css_crossover_numbers.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import build_paper_precision_ladder_paired_views as p  # noqa: E402

M = 30.4375
st, en, _ = p.load_phase1()
en = en.set_index("_gid")
out = []
for ctx in ("EGFR_1L", "EGFR_WT_1L"):
    for arm in ("chemo", "osimertinib"):
        q = st[st.context.eq(ctx) & st.arm.eq(arm)].copy()
        q["t"] = (q.days_from_diagnosis - q.cutoff_days) / M
        q["osi"] = q.active_treatment_regimens_joined.fillna("").str.lower().str.contains("osimertinib")
        ids = q._gid.unique()
        cut = q.groupby("_gid").cutoff_days.first()
        pfs, os_ = en.pfs_line_1_days.reindex(ids) / M, (en.os_days.reindex(ids) - cut.reindex(ids)) / M
        first = q[q.osi & q.t.ge(0)].groupby("_gid").t.min().reindex(ids)
        early = first < pfs
        row = {"context": ctx, "arm": arm, "n": len(ids), "median_pfs": pfs.median(), "median_os": os_.median(), "ever_osimertinib": int(first.notna().sum()),
               "osimertinib_before_first_line_progression": int(early.sum()), "median_pfs_no_early_switch": pfs[~early].median()}
        for t in (3, 6, 12, 24):
            alive = [g for g in ids if os_[g] > t]
            s = q[q.t.le(t + 1e-9)].sort_values(["_gid", "t"]).groupby("_gid").tail(1).set_index("_gid").osi.reindex(alive)
            row[f"on_osimertinib_m{t}"], row[f"alive_m{t}"] = int(s.sum()), len(alive)
        out.append(row)
(ROOT / "outputs/exploratory_cand_css_four_arm").mkdir(parents=True, exist_ok=True)
pd.DataFrame(out).round(2).to_csv(ROOT / "outputs/exploratory_cand_css_four_arm/crossover_numbers.csv", index=False)
