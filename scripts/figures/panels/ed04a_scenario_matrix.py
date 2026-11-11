"""Panel ed04a_scenario_matrix: 12 treatment scenarios x 4 models, share of specified comparisons each model got right.
Run: uv run python scripts/figures/panels/ed04a_scenario_matrix.py"""
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ed04 import MODEL_NAME, MODELS, ROWS, SCORE, SCORE_TEXT, SENT  # noqa: E402
from _f4 import fix_axes  # noqa: E402
from common import panel, save_panel  # noqa: E402

M = SENT[[f"{m}_primary_pct" for m in MODELS]].to_numpy(float)
n_static = int(SENT["axis"].eq("static").sum())
assert M.shape == (12, 4) and n_static == 9
fig, ax = panel(98, 78)
for r in range(12):
    for c in range(4):
        v = int(round(M[r, c]))
        ax.add_patch(__import__("matplotlib").patches.Rectangle((c - 0.5, r - 0.5), 1, 1, fc=SCORE[v], ec="white", lw=0.8))
        ax.text(c, r, f"{v}%", ha="center", va="center", fontsize=6, color=SCORE_TEXT[v])
ax.set_xlim(-0.5, 3.5); ax.set_ylim(11.5, -0.5)  # noqa: E702
ax.axhline(n_static - 0.5, color="black", lw=1.0)
ax.set_yticks(range(12), ROWS)
ax.xaxis.tick_top()
ax.set_xticks(range(4), [{"Gemini 3.1 Flash-Lite": "Gemini\n3.1\nFlash-Lite", "Qwen 3.6 Flash": "Qwen\n3.6\nFlash", "Gemma 4": "Gemma\n4", "Sonnet 5": "Sonnet\n5"}.get(MODEL_NAME[m], MODEL_NAME[m]) for m in MODELS])
ax.tick_params(length=0, labelsize=6)

for s in ax.spines.values():
    s.set_visible(False)
ax.text(3.58, 4, "Treatment or\nbiomarker\n(9 scenarios)", ha="left", va="center", fontsize=6)
ax.text(3.58, 10, "Needs history\nor host\n(3 scenarios)", ha="left", va="center", fontsize=6)
fix_axes(fig, ax, left=36, right=19, bottom=1, top=10.5)
LEGEND = ("Twelve clinical scenarios, each asking a model to rank treatments for a specified patient. Each cell is the share of the "
          "specified comparisons in that scenario that the model ranked correctly (red, 0%; gold, 50%; teal, 100%), for four "
          "generating models. The first nine scenarios depend on the biomarker or treatment class; the last three also need previous treatment or the "
          "patient's physical condition (relapsed ovarian cancer, fit versus frail pancreatic cancer, EGFR-mutant lung cancer after progression). "
          "The only values observed are 0%, 50% and 100%. This is a model-implied reasoning benchmark; it does not estimate an observed treatment effect. "
          "Abbreviations: PD-L1, programmed death-ligand 1; MSI, microsatellite instability; HER2, human epidermal growth factor receptor 2.")
save_panel(fig, "ed04a_scenario_matrix", legend=LEGEND,
           sources=["outputs/exploratory_sentinel_benchmark/sentinel_trust_map_table.csv", "scripts/build_paper_interventional_trust_map.py"],
           confirm=["Scenario names are plain-language renderings of the labels in the table (e.g. 'EGFR-mut LUAD, 1L').",
                    "A 50% cell: half of the specified comparisons followed."],
           message="Treatment ranking is often right for biomarker-driven scenarios and weaker when history or host condition matters.")
