"""Shared data and colours for the scenario-benchmark panels (ed04a, ed04b, ed04d)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import build_paper_interventional_trust_map as o  # noqa: E402  (sets rcParams; common re-applies house style after)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODEL_COLOURS  # noqa: E402

# score colours: red (none followed) - gold (half) - teal (all followed)
SCORE = {0: "#C8443A", 50: "#F0C75E", 100: "#2A8C82"}
SCORE_TEXT = {0: "white", 50: "black", 100: "white"}
SENT = o.load_sentinel(o.DEFAULT_SENTINEL)
P1 = o.load_phase1(tuple(o.DEFAULT_PHASE1))
MODELS = list(o.SENTINEL_MODELS)
MODEL_NAME = {"gemini31_lite": "Gemini 3.1 Flash-Lite", "qwen36_flash": "Qwen 3.6 Flash", "gemma4": "Gemma 4",
              "claude_sonnet5": "Sonnet 5", "kimi_k3_off": "Kimi K3", "glm52_fp8": "GLM-5.2"}
MODEL_KEY = {"gemini31_lite": "Gemini 3.1 L", "qwen36_flash": "Qwen 3.6", "gemma4": "Gemma 4",
             "claude_sonnet5": "Sonnet", "kimi_k3_off": "Kimi K3", "glm52_fp8": "GLM-5.2"}
ROWS = ["EGFR-mutant lung, first line", "PD-L1-high lung, first line", "EGFR wild-type lung, first line",
        "MSI-high colorectal", "RAS-mutant colorectal", "BRAF V600E melanoma", "HER2-positive breast",
        "HER2-negative breast", "Clear-cell kidney", "Ovarian, relapsed", "Pancreatic, fit versus frail",
        "EGFR-mutant lung, after progression"]


def colour(m: str) -> str:
    return MODEL_COLOURS[MODEL_KEY[m]]
