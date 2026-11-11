"""Panel si03c_toxicity_grades: severity (CTCAE grade) of the new adverse events, by model.

Message: the share of severe (grade 3 or higher) events differs by model. Data: _si03.py.
Run: uv run python scripts/figures/panels/si03c_toxicity_grades.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import _si03 as T  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _f3util import axes_mm, new_fig  # noqa: E402
from common import MODEL_COLOURS, MODEL_ORDER, save_panel  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

fig = new_fig(96, 47)
ax = axes_mm(fig, 5, 9, 66, 34)
cols = plt.get_cmap("YlOrRd")(np.linspace(0.2, 0.95, 5))
grades = [1, 2, 3, 4, 5]
n_tot = 0
for i, m in enumerate(MODEL_ORDER):
    o = T.onset[T.onset.model == m]
    n = len(o)
    n_tot += n
    left = 0.0
    for gi, g in enumerate(grades):
        w = (o.ctcae_v5_grade == g).sum() / n
        ax.barh(i, w, left=left, height=0.62, color=cols[gi], lw=0)
        left += w
    ax.plot([-0.025], [i], marker="s", ms=4, color=MODEL_COLOURS[m], clip_on=False, ls="none")
    sev = (o.ctcae_v5_grade >= 3).mean()
    ax.text(1.03, i, f"{sev:.0%}", va="center", fontsize=6)
    ax.text(1.2, i, f"{n:,}", va="center", fontsize=6)
ax.text(1.03, -0.55, "Grade 3+", fontsize=6, va="bottom")
ax.text(1.2, -0.55, "Events", fontsize=6, va="bottom")
ax.set_ylim(len(MODEL_ORDER) - 0.5, -0.5)
ax.set_yticks([])
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
ax.set_xlabel("Share of the model's new adverse events")
for s in ("left",):
    ax.spines[s].set_visible(False)
ax.legend(handles=[Patch(color=cols[k], label=f"Grade {g}") for k, g in enumerate(grades)], ncol=5,
          loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=6, handlelength=0.9,
          columnspacing=0.6, handletextpad=0.3)
save_panel(fig, "si03c_toxicity_grades",
           legend=("Severity of the new adverse events of panel b, by model (same rows). Each bar is 100% of the "
                   "model's events, split by CTCAE v5 grade at onset (light yellow, grade 1, to dark red, grade 5; grade 5 is too rare to see). "
                   "Squares at the left give the model colour of panel b. Right of the bars: the share of events of grade 3 or higher, and the number of events. "
                   f"Total {n_tot:,} events from six models. Generated records; not validated against real "
                   "toxicity rates."),
           sources=["runs/*/derivation/event_rows.csv for the six atlas runs (scripts/explore/figpanels/fig1_data.py)"],
           confirm=["Grade at onset only; later grade changes are ignored."],
           message="The share of severe adverse events differs by model.")
