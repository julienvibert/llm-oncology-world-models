"""Panel si03d_toxicity_organ_systems: which organ systems the adverse events involve, by model.

Message: the mix of adverse-event types differs between models; one dot per model shows the spread.
Data: _si03.py. Run: uv run python scripts/figures/panels/si03d_toxicity_organ_systems.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import _si03 as T  # noqa: E402
from _f3util import axes_mm, new_fig  # noqa: E402
from common import INK, LIGHT, MODEL_COLOURS, MODEL_ORDER, save_panel  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

o = T.onset
cats = o.ae_category.value_counts().index.tolist()
share = {m: o[o.model == m].ae_category.value_counts(normalize=True) for m in MODEL_ORDER}
NAME = {"hematologic": "Blood count (hematologic)", "gastrointestinal": "Gastrointestinal",
        "neurologic": "Neurological", "dermatologic": "Skin", "general_systemic": "General (such as fatigue)",
        "cardiac": "Cardiac", "infectious": "Infection", "immune_related": "Immune-related",
        "hepatic": "Liver", "metabolic_endocrine": "Metabolic or endocrine", "renal": "Kidney",
        "pulmonary": "Lung", "other_specified": "Other"}
H = 4.6 * len(cats) + 16
fig = new_fig(180, H)
ax = axes_mm(fig, 31, 9, 146, H - 9 - 9)
for i, c in enumerate(cats):
    vals = [share[m].get(c, 0.0) for m in MODEL_ORDER]
    ax.plot([min(vals), max(vals)], [i, i], color=LIGHT, lw=2.2, zorder=1, solid_capstyle="round")
    for m, v in zip(MODEL_ORDER, vals, strict=True):
        ax.scatter(v, i, s=22, color=MODEL_COLOURS[m], edgecolor=INK, lw=0.3, zorder=3)
ax.set_yticks(range(len(cats)), [NAME.get(c, c.replace("_", " ").capitalize()) for c in cats])
ax.set_ylim(len(cats) - 0.5, -0.5)
ax.set_xlim(0, 0.5)
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], ["0%", "10%", "20%", "30%", "40%", "50%"])
ax.tick_params(axis="y", length=0)
ax.grid(axis="y", color=LIGHT, lw=0.3)
ax.set_xlabel("Share of the model's new adverse events")
fig.legend(handles=[Line2D([], [], marker="o", ls="none", color=MODEL_COLOURS[m], mec=INK, mew=0.3, ms=4, label=m)
                    for m in MODEL_ORDER], ncol=6, loc="upper left", bbox_to_anchor=(0.2, 0.995), frameon=False,
           fontsize=6, handletextpad=0.2, columnspacing=1.0)
save_panel(fig, "si03d_toxicity_organ_systems",
           legend=("Organ system of the new adverse events, by model, as assigned by each model from the 13 categories of the output schema (plain-word labels; not recoded or reviewed). Each dot is one model's share of its own new "
                   "adverse events falling in the category (dot colours as in panel b; the grey bar spans the lowest to "
                   f"the highest model). Categories are ordered by pooled frequency. n = {len(o):,} events from six "
                   "models (per model in panel c). Generated records; not validated against real toxicity "
                   "frequencies."),
           sources=["runs/*/derivation/event_rows.csv for the six atlas runs (scripts/explore/figpanels/fig1_data.py)"],
           confirm=["Labels are plain-word names of the 13 ae_category values of the output schema, chosen by each model "
                    "for each event; checked 11 Nov 2026 against the event terms (e.g. fatigue is 856 of 1,383 'general' events)."],
           message="The mix of adverse-event types differs between models.")
