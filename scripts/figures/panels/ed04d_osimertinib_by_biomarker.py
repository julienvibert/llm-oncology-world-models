"""Panel ed04d_osimertinib_by_biomarker: median time to progression on osimertinib, EGFR-mutant versus wild-type, per model.
Medians are recomputed from the saved day-level medians at 30.4375 days per month (helper _egfr_days.py);
the trust-map table (30.4-day months) is read only to confirm the model order and the month-length identity.
Run: uv run python scripts/figures/panels/ed04d_osimertinib_by_biomarker.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ed04 import MODEL_NAME, P1, colour  # noqa: E402
from _egfr_days import SOURCES as DAY_SOURCES, context_medians, value  # noqa: E402
from _f4 import fix_axes  # noqa: E402
from common import panel, save_panel  # noqa: E402
import build_paper_interventional_trust_map as o  # noqa: E402

fig, ax = panel(78, 50)
p = P1.set_index("model")
MED = context_medians()
labs = {}
for m in o.PHASE1_ORDER:
    mut, wt = value(MED, m, "mut", "osimertinib"), value(MED, m, "wt", "osimertinib")
    for v, col in ((mut, "egfr_mut_osi_mo"), (wt, "egfr_wt_osi_mo")):
        assert abs(v * 30.4375 / 30.4 - float(p.loc[m, col])) < 1e-9, (m, col)
    assert value(MED, m, "mut", "osimertinib", "n") == value(MED, m, "wt", "osimertinib", "n") == 45
    c = colour(m)
    ax.plot([0, 1], [mut, wt], color=c, lw=1.1, zorder=2)
    ax.scatter([0, 1], [mut, wt], s=14, color=c, zorder=3, lw=0)
    labs[m] = (wt, f"{MODEL_NAME[m]}: {mut - wt:+.1f} months")
ax.set_xlim(-0.15, 2.0); ax.set_ylim(0, 48)  # noqa: E702
for m, (wt, t) in labs.items():
    y = wt
    ax.text(1.08, y, t.replace("+", "difference +").replace("-", "−") if False else t, va="center", fontsize=6)
ax.set_xticks([0, 1], ["EGFR-mutant", "EGFR wild-type"])
ax.set_ylabel("Median time to progression\non osimertinib (months)")
ax.set_xlabel("Generated disease")
fix_axes(fig, ax, left=14, right=1.5, bottom=10, top=2)
LEGEND = ("Median time to progression on osimertinib in generated EGFR-mutant and EGFR wild-type disease (each median summarises 45 generated futures per model "
          "and status: 15 designed cells, three repeats each), for the three models of panel c. Labels give the gain in EGFR-mutant relative to wild-type disease. A model can pass the directional checks and "
          "still differ widely in magnitude: among models that pass, the difference ranges from 1.6 to 21.1 months. The biomarker-blind model (GLM-5.2) predicts "
          "a long time to progression in both groups. Model-implied; not an observed treatment effect. No intervals are shown. Months are days / 30.4375.")
save_panel(fig, "ed04d_osimertinib_by_biomarker", legend=LEGEND,
           sources=[*DAY_SOURCES, "outputs/paper_interventional_trust_map/egfr_gate_scores_used.csv", "scripts/build_paper_interventional_trust_map.py"],
           confirm=[
                    "Medians are median_endpoint_days / 30.4375 (manuscript convention). Mutant / wild-type: Sonnet 23.43 / 2.30, "
                    "Kimi 23.95 / 22.31, GLM 44.25 / 33.12; differences 21.1, 1.6, 11.1.",
                    "n: 45 osimertinib futures per model and status (3 per designed cell, all 15 cells), checked; GLM's missing "
                    "futures are in chemotherapy and pembrolizumab arms only, which this panel does not use."],
           message="Passing models differ widely in how large they make the osimertinib benefit of EGFR mutation.")
