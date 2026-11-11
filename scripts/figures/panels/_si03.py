"""Toxicity tables of the six-model atlas for the SI3 panels (new adverse events only).

A new adverse event is a toxicity record with transition 'onset' (15,946 of the 30,396 toxicity
records; the others are resolutions and grade changes of an existing event). Denominator per model: all
accepted generated courses of that model. Generated data only."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "explore" / "figpanels"))
sys.path.insert(0, str(HERE.parents[1] / "explore"))
import numpy as np  # noqa: E402
from fig1_data import _read  # noqa: E402

SHORT = {"claude_sonnet5": "Sonnet", "gemini31_lite": "Gemini 3.1 L", "gemma4": "Gemma 4",
         "glm52_fp8": "GLM-5.2", "kimi_k3_off": "Kimi K3", "qwen36_flash": "Qwen 3.6"}
SEED = 20261012

ev = _read("event_rows.csv")
ev["model"] = ev["model_short"].map(SHORT)
onset = ev[(ev.event_type == "toxicity") & (ev.toxicity_transition == "onset")].copy()
courses = {m: sorted(g._gid.unique()) for m, g in ev.groupby("model")}


def per_course_counts(m: str) -> np.ndarray:
    c = onset[onset.model == m].groupby("_gid").size()
    return c.reindex(courses[m], fill_value=0).to_numpy(float)


def mean_ci(m: str, n_boot: int = 2000) -> tuple[float, float, float]:
    x = per_course_counts(m)
    rng = np.random.default_rng(SEED)
    b = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))
