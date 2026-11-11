"""Panel f2d_trajectory_gallery.

Message: each generated record is a rich linear trajectory: treatment lines (by class, with drug
names), time in each disease state, toxicity, molecular events, radiotherapy and death.
Data: atlas event and treatment tables (event_rows.csv, treatment_rows.csv in the six atlas
derivation folders; loader _read from scripts/explore/figpanels/fig1_data.py). Lung
adenocarcinoma (OncoTree LUAD); per model the accepted trajectory with the lowest SHA-256 hash of
its identifier (fixed rule, no selection by content). Generated data only.
Run: uv run python scripts/figures/panels/f2d_trajectory_gallery.py
"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "explore" / "figpanels"))
sys.path.insert(0, str(HERE.parents[1] / "explore" / "figpanels"))
sys.path.insert(0, str(HERE.parents[1] / "explore"))
sys.path.insert(0, str(HERE.parents[1]))
from _f2 import MODEL_COLOURS, MODEL_ORDER, axes_mm, canvas  # noqa: E402
from common import INK, LIGHT, MODALITY_COLOURS, STATE_COLOURS, save_panel  # noqa: E402
import pandas as pd  # noqa: E402
from fig1_data import _read  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

MD = 30.4
SHORT = {"claude_sonnet5": "Sonnet", "gemini31_lite": "Gemini 3.1 L", "gemma4": "Gemma 4",
         "glm52_fp8": "GLM-5.2", "kimi_k3_off": "Kimi K3", "qwen36_flash": "Qwen 3.6"}
# display names for the regimen identifiers in the tables: (long, short)
NAMES = {
    "osimertinib": ("Osimertinib", "Osimertinib"),
    "carboplatin_pemetrexed": ("Carboplatin + pemetrexed", "Carbo + pem"),
    "platinum-pemetrexed": ("Platinum + pemetrexed", "Plat + pem"),
    "carboplatin_etoposide": ("Carboplatin + etoposide", "Carbo + etop"),
    "osimertinib_chemotherapy": ("Osimertinib + chemotherapy", "Osi + chemo"),
    "osimertinib_capmatinib": ("Osimertinib + capmatinib", "Osi + capma"),
    "carboplatin_pemetrexed_pemetrexed_maintenance": ("Carboplatin + pemetrexed, maintenance", "Carbo + pem"),
    "docetaxel": ("Docetaxel", "Docetaxel"),
    "osimertinib_carboplatin_pemetrexed": ("Osimertinib + carbo + pem", "Osi + carbo + pem"),
    "docetaxel_ramucirumab": ("Docetaxel + ramucirumab", "Docetax + ramuc"),
}
MOLNAME = {"MET amplification": "MET amp.", "small cell histologic transformation": "small-cell transformation",
           "EGFR C797S mutation": "EGFR C797S"}
TARGETED, CHEMO = MODALITY_COLOURS["targeted"], MODALITY_COLOURS["chemotherapy"]
TOX = "#8E44AD"
MOL = "#B8860B"

ev = _read("event_rows.csv")
tx = _read("treatment_rows.csv")
ev = ev[ev.tumor_oncotree_code == "LUAD"]
picked = {SHORT[ms]: min(g._gid.unique(), key=lambda x: hashlib.sha256(x.encode()).hexdigest())
          for ms, g in ev.groupby("model_short")}
assert set(picked) == set(MODEL_ORDER)

W, H = 180, 76
fig = canvas(W, H)
AX_X, AX_W, AX_Y, AX_H = 36.0, 141.0, 7.5, 61.0
ax = axes_mm(fig, AX_X, AX_Y, AX_W, AX_H)
XMAX = 42.0
mm_per_month = AX_W / XMAX
ax.set_xlim(0, XMAX)
ax.set_ylim(0, AX_H)
BLOCK = AX_H / 6
# lane offsets (mm) from the bottom of a block, and lane heights
LANE = {"molecular": (0.2, 1.8), "toxicity": (2.3, 1.8), "disease": (4.3, 2.2), "treatment": (6.8, 3.2)}
summary = []
for i, m in enumerate(MODEL_ORDER):
    gid = picked[m]
    e = ev[ev._gid == gid].sort_values("event_index").copy()
    e["month"] = e.days_from_diagnosis.astype(float) / MD
    t0 = pd.to_datetime(e.event_date.iloc[0]) - pd.Timedelta(days=float(e.days_from_diagnosis.iloc[0]))
    base = AX_H - (i + 1) * BLOCK + 0.4
    d = tx[tx._gid == gid].sort_values("course_index")
    death = float(e[e.event_type == "death"].month.iloc[0])

    def yc(lane):
        return base + LANE[lane][0] + LANE[lane][1] / 2

    # model colour bar and name
    ax.add_patch(Rectangle((-AX_X / mm_per_month + 0.3 / mm_per_month, base), 1.4 / mm_per_month, BLOCK - 1.0,
                           fc=MODEL_COLOURS[m], ec="none", clip_on=False))
    ax.text(-AX_X / mm_per_month + 2.6 / mm_per_month, base + BLOCK / 2 - 0.5, m, ha="left", va="center",
            fontsize=6.5, color=INK, clip_on=False)
    if i < 5:
        ax.plot([0, XMAX], [base - 0.4] * 2, color=LIGHT, lw=0.4, zorder=0)
    # treatment lines
    for r in d.itertuples(index=False):
        s = (pd.to_datetime(r.start_date) - t0).days / MD
        en = (pd.to_datetime(r.end_date) - t0).days / MD
        assert en >= s
        cl = str(r.component_classes_joined)
        has_chemo = "cytotoxic" in cl
        has_targ = any(k in cl for k in ("egfr_tki", "other_targeted", "monoclonal"))
        y0, hh = base + LANE["treatment"][0], LANE["treatment"][1]
        if has_chemo and has_targ:
            # combined regimen: blue bar with a pink stripe underneath
            ax.add_patch(Rectangle((s, y0), en - s, hh, fc=TARGETED, ec="white", lw=0.4))
            ax.add_patch(Rectangle((s, y0), en - s, 0.8, fc=CHEMO, ec="white", lw=0.4))
            tcol, stroke = "white", False
        else:
            col = CHEMO if has_chemo else TARGETED
            ax.add_patch(Rectangle((s, y0), en - s, hh, fc=col, ec="white", lw=0.4))
            tcol, stroke = ("white" if col == TARGETED else INK), False
        long_, short_ = NAMES[str(r.regimen_name).lower()]
        wmm = (en - s) * mm_per_month
        label = long_ if len(long_) * 1.12 + 1 < wmm else short_ if len(short_) * 1.12 + 1 < wmm else None
        if label:
            t = ax.text((s + en) / 2, y0 + hh / 2 + (0.35 if has_chemo and has_targ else 0), label, ha="center", va="center", fontsize=6, color=tcol)
            if stroke:
                import matplotlib.patheffects as pe
                t.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])
    # disease state bars from response / progression events
    seg, state, a = [], "unresolved", 0.0
    for r in e[e.event_type.isin(["response", "progression"])].itertuples(index=False):
        if r.event_type != state:
            seg.append((a, r.month, state))
            state, a = r.event_type, r.month
    seg.append((a, death, state))
    for a_, b_, st in seg:
        ax.add_patch(Rectangle((a_, base + LANE["disease"][0]), b_ - a_, LANE["disease"][1],
                               fc=STATE_COLOURS[st], ec="white", lw=0.3))
    ax.plot(death, yc("disease"), "X", color=STATE_COLOURS["death"], ms=4.2, mec="none", zorder=4)
    # toxicity onsets
    tox = e[(e.event_type == "toxicity") & (e.toxicity_transition == "onset")]
    for r in tox.itertuples(index=False):
        hi = r.ctcae_v5_grade >= 3
        ax.plot(r.month, yc("toxicity"), "^", mfc=TOX if hi else "white", mec=TOX, mew=0.8, ms=3.4, zorder=3)
    # molecular events and radiotherapy
    rt = list(e[e.event_type == "local_intervention"].month)
    for r in e[e.event_type == "clonal_evolution_event"].itertuples(index=False):
        ax.plot(r.month, yc("molecular"), "D", color=MOL, ms=3.2, mec="none", zorder=3)
        lab = MOLNAME.get(str(r.alteration), str(r.alteration))
        wm = len(lab) * 1.05 / mm_per_month
        clash = any(r.month - 0.3 < x < r.month + 0.7 + wm for x in rt)
        if clash:
            ax.text(r.month - 0.6, yc("molecular"), lab, fontsize=6, va="center", ha="right", color=INK)
        else:
            ax.text(r.month + 0.6, yc("molecular"), lab, fontsize=6, va="center", ha="left", color=INK)
    for x in rt:
        ax.plot(x, yc("molecular"), "s", color=INK, ms=2.8, mec="none", zorder=3)
    summary.append((m, len(e), len(d), round(death, 1), int((tox.ctcae_v5_grade >= 3).sum()), len(tox)))
    if i == 0:
        for lane, name in (("treatment", "Treatment"), ("disease", "Disease state"), ("toxicity", "Toxicity"),
                           ("molecular", "Molecular, RT")):
            ax.text(-0.8 / mm_per_month, yc(lane), name, ha="right", va="center", fontsize=6, color=INK)
for s in ("left", "top", "right"):
    ax.spines[s].set_visible(False)
ax.set_yticks([])
ax.set_xticks([0, 6, 12, 18, 24, 30, 36, 42])
ax.set_xlabel("Months from diagnosis")
H_ = [Patch(fc=TARGETED, label="Targeted therapy"), Patch(fc=CHEMO, label="Chemotherapy (stripe: added to targeted)"),
      Patch(fc=STATE_COLOURS["unresolved"], ec=LIGHT, lw=0.4, label="Not yet assessed"),
      Patch(fc=STATE_COLOURS["response"], label="Response"),
      Patch(fc=STATE_COLOURS["progression"], label="Progression"),
      Line2D([], [], marker="X", ls="", color=STATE_COLOURS["death"], ms=4.2, label="Death"),
      Line2D([], [], marker="^", ls="", mfc=TOX, mec=TOX, ms=3.4, label="Toxicity grade 3-4"),
      Line2D([], [], marker="^", ls="", mfc="white", mec=TOX, mew=0.8, ms=3.4, label="Toxicity grade 1-2"),
      Line2D([], [], marker="D", ls="", color=MOL, ms=3.2, label="Molecular event"),
      Line2D([], [], marker="s", ls="", color=INK, ms=2.8, label="Radiotherapy")]
fig.legend(handles=H_, loc="upper left", bbox_to_anchor=(1 / W, 1.0), ncol=5, frameon=False, fontsize=6,
           handlelength=1.2, handletextpad=0.4, columnspacing=1.4, labelspacing=0.4, borderaxespad=0)
for s in summary:
    print(s)
save_panel(
    fig, "f2d_trajectory_gallery",
    legend=("One generated lung adenocarcinoma record per model, drawn as parallel tracks against months from "
            "diagnosis. Each case is the accepted record with the lowest SHA-256 hash of its identifier among that "
            "model's five lung adenocarcinoma records (fixed rule, not selected for content or outcome). Treatment: one bar per "
            "treatment line, blue for targeted therapy and pink for chemotherapy (a blue bar with a pink stripe is targeted therapy combined with chemotherapy), "
            "with the regimen name. Disease state: bars show the time spent in each state, set by the generated "
            "response and progression events; the state before the first assessment is shown as not yet assessed; "
            "the cross marks death. Toxicity: triangle at each adverse-event onset (filled, grade 3-4; open, grade 1-2). "
            "Molecular: diamonds mark new molecular alterations; squares mark radiotherapy. Track names are shown "
            "once, for the first case; the colour bar at the left identifies the model. Generated data."),
    sources=["runs/*/derivation/event_rows.csv", "runs/*/derivation/treatment_rows.csv (via scripts/explore/figpanels/fig1_data.py _read)"],
    confirm=["Six cases (lowest hash per model among 5 LUAD records each); the gallery shows the rule, not the median patient.",
             "Disease-state bars are derived from response and progression events only (no assessment before the first event); "
             "the state after a response event stays 'response' until progression.",
             "Toxicity: only onset events are drawn (grade changes and resolutions omitted). Dose modifications and patient-state updates are not drawn.",
             "Drug names are the regimen identifiers in the tables, written in clinical form; abbreviations are used when the bar is too short; very short bars carry no name.",
             "New colours for toxicity (violet) and molecular events (dark gold) are local to this panel and not in common.py.",
             "Treatment bars use dates converted at 30.4 days per month."],
    message="Each generated record is a rich linear trajectory of treatment, disease state, toxicity, molecular events and death.")
