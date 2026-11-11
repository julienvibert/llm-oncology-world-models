"""Shared helpers for Extended Data figures 1 and 2 (state-map pages). Read-only use of saved outputs."""
import sys
from pathlib import Path

import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODEL_COLOURS, ROOT  # noqa: E402,F401

D = ROOT / "outputs" / "paper_state_map"
TUMOUR, HOST, TOX = "#7a5195", "#E08A3C", "#56B4E9"
CMAP = LinearSegmentedColormap.from_list("state_severity", ("#eaf4f2", "#5ba7a3", "#7a5195", "#8f2636"))
MODEL_KEYS = {"claude_sonnet5": "Sonnet", "gemini31_lite": "Gemini 3.1 L", "gemma4": "Gemma 4",
              "glm52_fp8": "GLM-5.2", "kimi_k3_off": "Kimi K3", "qwen36_flash": "Qwen 3.6"}
NOT_COORD = {"_run", "_gid", "trajectory_id", "model_short", "tumor_oncotree_code", "snapshot_index",
             "state_id", "days_from_diagnosis", "pc1", "pc2", "observation_weight"}

_PROJ = None


def projection() -> pd.DataFrame:
    global _PROJ
    if _PROJ is None:
        _PROJ = pd.read_csv(D / "state_projection.csv.gz")
    return _PROJ


def coord_cols(p: pd.DataFrame) -> list[str]:
    return [c for c in p.columns if c not in NOT_COORD]


AX1 = "State-map PC1 (a.u.)"
AX2 = "State-map PC2 (a.u.)"
