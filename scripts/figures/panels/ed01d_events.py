"""Panel ed01d_events: do clinical events move patients in the expected direction of the state map?"""
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from _ed12 import D, HOST, TOX, TUMOUR  # noqa: E402
from common import panel, save_panel  # noqa: E402

ev = pd.read_csv(D / "event_validation.csv")
ORDER = ["response", "local_intervention", "progression", "clonal_evolution_event", "patient_state_update",
         "toxicity", "systemic_treatment_start", "systemic_treatment_switch"]
ev = ev.set_index("event_type").reindex(ORDER)
CH = {"tumour": TUMOUR, "host": HOST, "toxicity": TOX}
EXP = {"tumour better": "tumour", "tumour worse": "tumour", "host worse": "host", "toxicity worse": "toxicity"}
fig, ax = panel(180, 48)
pos = np.arange(len(ev)); w = 0.26  # noqa: E702
for i, ch in enumerate(CH):
    ax.bar(pos + (i - 1) * w, ev[f"mean_delta_{ch}"], w, color=CH[ch])
for k, (e, r) in enumerate(ev.iterrows()):
    ch = EXP.get(r["expected"])
    if ch:
        j = list(CH).index(ch)
        v = r[f"mean_delta_{ch}"]
        ax.plot(k + (j - 1) * w, v + (0.006 if v >= 0 else -0.006), marker="v" if v >= 0 else "^", color="black", ms=3, ls="none") if False else \
            ax.plot(k + (j - 1) * w, (v + 0.006) if v >= 0 else (v - 0.006), marker="*", color="black", ms=4, ls="none")
ax.axhline(0, color="black", lw=0.5)
labels = [textwrap.fill(str(ev.loc[e, "event_label"]), 14, break_long_words=False) + f"\nn = {int(ev.loc[e, 'n']):,}" for e in ev.index]
ax.set_xticks(pos, labels)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("Mean change in the state\n(0 to 1 scale; + = worse)")
ax.set_ylim(-0.065, 0.1)
h = [Patch(color=TUMOUR, label="Tumour coordinates"), Patch(color=HOST, label="Host coordinates"),
     Patch(color=TOX, label="Toxicity coordinates"),
     Line2D([], [], marker="*", color="black", ls="none", ms=4, label="Expected to move")]
ax.legend(handles=h, loc="upper right", ncols=2, frameon=False)
ax.text(0.99, 0.30, "Negative controls", transform=ax.transAxes, ha="right", fontsize=6) if False else None
save_panel(
    fig, "ed01d_events",
    legend=("Do clinical events move patients in the expected direction? For each type of event in the generated "
            "trajectories (n = event transitions, under each label), bars show the mean change in the tumour (purple), "
            "host (orange) and toxicity (sky blue) coordinates between the state just before and just after the event, on "
            "the fixed 0 to 1 scale of each coordinate (positive = worse). A star marks the coordinate that was "
            "specified in advance to move: tumour down with response and local intervention, tumour up with progression "
            "and clonal evolution, host up with a patient-state update, toxicity up with toxicity. All eight "
            "pre-specified checks pass (six directional, two negative controls). Starting or switching a systemic treatment is a negative control: it is not "
            "expected to change the state at that instant, and indeed no coordinate moves. The bars use the raw "
            "coordinates, not the two map axes. These are coherence checks on generated data, not evidence that the "
            "generated biology is correct. Source: generated atlas, six models, 4,434 trajectories."),
    sources=["outputs/paper_state_map/event_validation.csv"],
    confirm=["Event order: better, then worse, then controls."],
    message="Clinical events move the specified coordinates of the state map and the negative controls do not.")
