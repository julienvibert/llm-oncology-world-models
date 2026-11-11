"""EGFR phase-1 medians in months from saved days (30.4375 days per month), for the EGFR panels (f3b, f3d, n3, ed04d).

The scorer (scripts/explore/score_interventional_phase1.py) writes months at 30.4 days. The manuscript uses
30.4375 days per month, so these panels read the saved day-level medians and convert them here. The scorer
and its outputs are not changed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT  # noqa: E402

MONTH_DAYS = 30.4375
RUNS = {"claude_sonnet5": "outputs/exploratory_interventional_phase1",
        "kimi_k3_off": "outputs/exploratory_interventional_phase1_kimi_k3_combined",
        "glm52_fp8": "outputs/exploratory_interventional_phase1_glm52_full"}
CONTEXT = {"mut": "EGFR_1L", "wt": "EGFR_WT_1L"}
SOURCES = [f"{d}/phase1_context_arm_medians.csv" for d in RUNS.values()]


def context_medians() -> pd.DataFrame:
    """One row per model, status and arm: median days, median months (30.4375) and n futures."""
    frames = [pd.read_csv(ROOT / d / "phase1_context_arm_medians.csv") for d in RUNS.values()]
    t = pd.concat(frames, ignore_index=True)
    t["months"] = t["median_endpoint_days"] / MONTH_DAYS
    return t[["context", "arm_key", "model", "median_endpoint_days", "months", "n"]]


def value(t: pd.DataFrame, model: str, status: str, arm: str, col: str = "months") -> float:
    """Median months (or n with col='n') for one model, status ('mut' or 'wt') and arm."""
    row = t[t.model.eq(model) & t.context.eq(CONTEXT[status]) & t.arm_key.eq(arm)]
    assert len(row) == 1, (model, status, arm)
    return float(row[col].iloc[0])
