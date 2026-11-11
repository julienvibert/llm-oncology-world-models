"""Panel ed10b_index_patients: three index patients, population median versus the median of their finest comparators.

Source: outputs/exploratory_precision_ladder/readout.json (index_patients).
Run:  uv run python scripts/figures/panels/ed10b_index_patients.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BLUE, INK, ORANGE, RED, ROOT, fk, save_panel  # noqa: E402

src = json.loads((ROOT / "outputs/exploratory_precision_ladder/readout.json").read_text())
pats = src["index_patients"]
keys = ["pancreatic adenocarcinoma (PAAD)", "a patient who looks average", "basal cell carcinoma (BCC)"]
names = ["Pancreatic\nadenocarcinoma", "Typical member of the\nlargest matched group", "Basal cell\ncarcinoma"]
W, H = 88.0, 66.0
fig = fk.figure(width=W, height_mm=H)
ax = fig.add_axes([32 / W, 22 / H, 54 / W, 42 / H])
y = np.arange(3)[::-1]
pop, fine, obs, cnt = [], [], [], []
for k in keys:
    r = pats[k]
    obs.append(float(r["os_months"])); pop.append(float(r["by_rung"][0]["median"]))  # noqa: E702
    fine.append(float(r["by_rung"][-1]["median"])); cnt.append(int(r["by_rung"][-1]["n"]))  # noqa: E702
for yi, p, f, o, c in zip(y, pop, fine, obs, cnt):
    ax.plot([p, o], [yi, yi], color=INK, lw=0.4, zorder=1)
    ax.scatter(p, yi, color=ORANGE, marker="s", s=22, zorder=3)
    ax.scatter(o, yi, color=RED, marker="D", s=24, zorder=4)
    ax.scatter(f, yi, facecolor="white" if c < 3 else BLUE, edgecolor=BLUE, marker="o", lw=0.9, s=26, zorder=5)
    ax.text(f, yi - 0.2, f"n = {c}", ha="center", va="top", fontsize=6)
ax.set_yticks(y); ax.set_yticklabels(names)  # noqa: E702
ax.set_ylim(-0.6, 2.5); ax.set_xlim(0, 10 * np.ceil(max(pop + fine + obs) / 10 + 0.3))  # noqa: E702
ax.set_xlabel("Overall survival (months)")
ax.scatter([], [], color=ORANGE, marker="s", s=22, label="Median of all atlas patients")
ax.scatter([], [], color=RED, marker="D", s=24, label="This patient's generated survival")
ax.scatter([], [], color=BLUE, marker="o", s=26, label="Median of the most closely matched patients")
fig.legend(*ax.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.02, 0.0), fontsize=6, handletextpad=0.2, frameon=False, ncol=1, labelspacing=0.3)
code = {k: pats[k]["code"] for k in keys}
assert src.get("ladder_definition", "").startswith("cumulative")


def mo(v: float) -> str:
    return f"{v:.1f}"


legend = ("The same atlas-wide estimate misses individual patients, and close matching helps only in some. Three index patients "
          "from the generated six-model atlas (not observed patients), chosen by baseline covariates only, never by outcome: for "
          "two tumour types named in advance (pancreatic adenocarcinoma, basal cell carcinoma), the member nearest that type's "
          "centroid; for the middle row, the member nearest the centroid of the largest cell at the finest matching level "
          f"(OncoTree code {code[keys[1]]}{', hairy cell leukaemia' if code[keys[1]] == 'HCL' else ''}), not a patient typical of the whole atlas. Orange squares, median overall survival of "
          "all atlas patients; red diamonds, the patient's own generated overall survival; blue circles, median survival of the "
          "comparators matched on all criteria (n, number of comparators left; open circle, fewer than 3, so not an estimate). "
          f"Pancreatic: matched median {mo(fine[0])} months against an outcome of {mo(obs[0])} (n = {cnt[0]}). Middle row: the "
          f"matched median ({mo(fine[1])} months, n = {cnt[1]}) was longer than the patient's survival ({mo(obs[1])}). Basal "
          f"cell carcinoma: {cnt[2]} comparator{'s' if cnt[2] != 1 else ''} left. Three patients illustrate the mechanism; they "
          "are not a performance estimate.")
save_panel(fig, "ed10b_index_patients", legend=legend,
           sources=["outputs/exploratory_precision_ladder/readout.json"],
           confirm=[f"Middle-row patient: OncoTree {code[keys[1]]}; matched median {mo(fine[1])} vs survival {mo(obs[1])}, n = "
                    f"{cnt[1]}.",
                    "The matched MEDIAN was longer (not 'all comparators'); the middle patient is near the centroid of the "
                    "largest finest cell, not of the whole atlas.",
                    "Open circle rule: fewer than 3 comparators."],
           message="A population median misses individuals in opposite directions; matching helps only while comparators remain.")
