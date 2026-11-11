"""Panel ed02d_specificity: how cancer-specific each model's generated courses are."""
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
from _ed12 import MODEL_COLOURS, ROOT  # noqa: E402
from common import MODEL_ORDER, GREY, panel, save_panel  # noqa: E402

r = json.loads((ROOT / "outputs/exploratory_native_atlas_geometry/readout.json").read_text())
KEY = {"Sonnet": "claude_sonnet5", "Gemini 3.1 L": "gemini31_lite", "Gemma 4": "gemma4", "GLM-5.2": "glm52_fp8",
       "Kimi K3": "kimi_k3_off", "Qwen 3.6": "qwen36_flash"}
rows = []
for m in MODEL_ORDER:
    b = r["models"][KEY[m]]["blocks"]["composite"]
    rows.append((m, b["specificity_ratio"], *b["specificity_ratio_ci"]))
rows.sort(key=lambda x: x[1])
fig, ax = panel(70, 42)
for i, (m, v, lo, hi) in enumerate(rows):
    ax.plot([lo, hi], [i, i], color=MODEL_COLOURS[m], lw=1.2)
    ax.plot(v, i, "o", color=MODEL_COLOURS[m], ms=4.5, mew=0)
    ax.text(hi + 0.012, i, f"{v:.2f}", va="center", fontsize=6)
ax.axvline(1.0, color=GREY, ls="--", lw=0.7)
ax.set_yticks(range(len(rows)), [x[0] for x in rows])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0.95, max(x[3] for x in rows) + 0.1)
ax.set_xlabel("Between-cancer / within-cancer distance")
ax.spines["left"].set_visible(False)
n = r["split_audit"]["n_matched_cancers"]
save_panel(
    fig, "ed02d_specificity",
    legend=("How cancer-specific each model's generated courses are. For each model, the average distance between "
            "courses generated for different cancers divided by the average distance between courses generated for the "
            "same cancer (held-out generated paths over 24 months, composite ruler). A ratio of 1 (dashed line) means "
            f"the model's cancers are not distinguishable from one another. Dots, estimate; bars, bootstrap interval over "
            f"cancers ({n} cancers; 399 resamples). Models are ordered by the estimate. Descriptive, not a measure of "
            "clinical fidelity."),
    sources=["outputs/exploratory_native_atlas_geometry/readout.json"],
    confirm=["Interval level: readout stores bootstrap CIs from 399 resamples."],
    message="All six models generate cancer-specific courses, but to different degrees.")
