"""Panel si03b_toxicity_per_course: new adverse events per generated course, by model.

Message: models differ several-fold in how many adverse events they generate per course.
Data: _si03.py (six-model atlas). Bootstrap over courses (2,000 draws, fixed seed) is computed here.
Run: uv run python scripts/figures/panels/si03b_toxicity_per_course.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import _si03 as T  # noqa: E402
from _f3util import axes_mm, new_fig  # noqa: E402
from common import INK, MODEL_COLOURS, MODEL_ORDER, save_panel  # noqa: E402

fig = new_fig(84, 47)
ax = axes_mm(fig, 16, 9, 66, 34)
res = {m: T.mean_ci(m) for m in MODEL_ORDER}
for i, m in enumerate(MODEL_ORDER):
    mu, lo, hi = res[m]
    ax.barh(i, mu, height=0.62, color=MODEL_COLOURS[m])
    ax.plot([lo, hi], [i, i], color=INK, lw=0.8)
    ax.text(hi + 0.08, i, f"{mu:.1f}", va="center", fontsize=6)
ax.set_yticks(range(len(MODEL_ORDER)), [f"{m}\n(n = {len(T.courses[m])})" for m in MODEL_ORDER])
ax.set_ylim(len(MODEL_ORDER) - 0.5, -0.5)
ax.set_xlim(0, 9.5)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("New adverse events per generated course")
save_panel(fig, "si03b_toxicity_per_course",
           legend=("New adverse events per generated course, by model. A new adverse event is a toxicity record whose transition is an onset (not a resolution or a change of grade of an existing event). Bars, mean number of new adverse "
                   "events (toxicity onsets) per course; black lines, 95% bootstrap interval, resampling whole generated courses (the bootstrap unit) with replacement "
                   "(2,000 resamples, fixed seed). n, accepted generated courses per model; courses without any adverse "
                   "event are counted as zero. Generated records, not observed patients."),
           sources=["runs/*/derivation/event_rows.csv for the six atlas runs (scripts/explore/figpanels/fig1_data.py)"],
           confirm=["Bootstrap intervals computed here from the event tables (fixed seed).",
                    "Counts only toxicity onsets (15,946 events); resolutions and grade changes are not new events."],
           message="Models differ several-fold in how many adverse events they generate per course.")
