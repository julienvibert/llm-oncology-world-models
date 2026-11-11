"""Panel f6b_patient_paths: six individual PRIME patients, state-map path next to linear tumour timeline.

Message: averages hide individual paths: each trial patient wanders through the clinical state
map while the linear record shows only tumour size and progression.

Selection: declared rule (patients with >= 5 lesion assessments; three drawn at random per arm
with the seed of scripts/explore/pds_prime_trial_geometry.py), re-run on the saved tables.
Renumbered 1-6; no
identifiers are printed. Map positions come from the restricted, trial-centred projection used
for the PRIME displays, read from the assessment table prime_state_visits.csv.gz (every assessment, first to
last); tumour sizes and progression from state_snapshots.csv.gz and subject_table.csv.

Run:  uv run python scripts/figures/panels/f6b_patient_paths.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "scripts" / "explore"))
import pds_prime_trial_geometry as src  # noqa: E402  (constants only)

sys.path.insert(0, str(HERE.parents[1]))
from common import BLUE, CHEMO, GREY, INK, ROOT, STATE_COLOURS, fk, save_panel  # noqa: E402

D = ROOT / "outputs" / "exploratory_pds_prime_trial_geometry"
states = pd.read_csv(D / "state_snapshots.csv.gz", dtype={"SUBJID": str})
subjects = pd.read_csv(D / "subject_table.csv", dtype={"SUBJID": str})
VIS = "outputs/exploratory_prime_trial_geometry/prime_state_visits.csv.gz"
visits = pd.read_csv(ROOT / VIS)
visits["SUBJID"] = visits["SUBJID"].astype(int)
ARM = {src.ARM_CONTROL: ("FOLFOX", CHEMO), src.ARM_EXPERIMENTAL: ("Panitumumab + FOLFOX", BLUE)}

# declared selection rule (identical to s38_prime_example_paths.py)
eligible = states.groupby("SUBJID").filter(lambda g: len(g) >= 5)
examples: list[str] = []
rng = np.random.default_rng(src.SEED)
for _, g in eligible.groupby("arm_raw"):
    ids = g["SUBJID"].unique()
    examples.extend(rng.choice(ids, min(3, len(ids)), replace=False))
assert len(examples) == 6

W, H = 180.0, 48.0
fig = fk.figure(width=W, height_mm=H)
MAP = (13.0, 10.5, 70.0, 31.5)          # left, bottom, width, height (mm) of the state map
LANE_L, LANE_W, LANE_H, LANE_GAP, LANE_B = 106.0, 72.0, 4.6, 1.1, 8.0


def mm(l, b, w, h):  # noqa: E741
    return fig.add_axes([l / W, b / H, w / W, h / H])


axm = mm(*MAP)
lanes = [mm(LANE_L, LANE_B + (5 - i) * (LANE_H + LANE_GAP), LANE_W, LANE_H) for i in range(6)]
ratio_max = 0
rows = []
for sid in examples:
    g = states[states.SUBJID.eq(sid)].sort_values("VISITDY")
    r = g.target_sld / g.target_sld.iloc[0]
    ratio_max = max(ratio_max, r.max())
    rows.append((sid, g, r))

OFF = {2: (-6, -4), 3: (-5, 3), 4: (5, 3), 5: (3, 4)}
OFFA = {2: "right", 3: "right", 4: "left", 5: "left"}
OFFV = {2: "top", 3: "bottom", 4: "bottom", 5: "bottom"}
for i, (sid, g, r) in enumerate(rows):
    name, col = ARM[g["arm_raw"].iloc[0]]
    s = visits[visits.SUBJID.eq(int(sid))].sort_values("day")
    assert len(s) == len(g) and np.allclose(s.day.to_numpy(), g.VISITDY.to_numpy())  # same assessments
    axm.plot(s.pc1, s.pc2, color=col, lw=0.9, alpha=0.9, zorder=2)
    axm.plot(s.pc1, s.pc2, "o", color=col, ms=2.2, mec="white", mew=0.3, zorder=3)
    axm.plot(s.pc1.iloc[0], s.pc2.iloc[0], "o", ms=4.8, mfc="white", mec=INK, mew=0.7, zorder=4)
    axm.plot(s.pc1.iloc[-1], s.pc2.iloc[-1], "X", ms=5, color=INK, mec="white", mew=0.3, zorder=4)
    axm.annotate(str(i + 1), (s.pc1.iloc[0], s.pc2.iloc[0]), xytext=OFF.get(i, (-4.5, 3.5)), textcoords="offset points",
                 fontsize=6, ha=OFFA.get(i, "right"), va=OFFV.get(i, "bottom"), zorder=5)
    a = lanes[i]
    t = g.VISITDY / src.MONTH_DAYS
    a.axhline(100, color=GREY, lw=0.4, ls=(0, (2, 2)), zorder=1)
    a.plot(t, 100 * r, "-o", color=col, lw=1.0, ms=2.0, mec="white", mew=0.2, zorder=3)
    p = subjects[subjects.SUBJID.eq(sid)].iloc[0]
    if p.pfs_event == 1:
        a.plot(p.pfs_months, 100 * ratio_max * 1.05 * 0.88, "v", color=STATE_COLOURS["progression"], ms=4.2,
               mec="white", mew=0.3, clip_on=False, zorder=4)
    a.set_xlim(-1.5, 31)
    a.set_ylim(0, 100 * ratio_max * 1.05)
    a.set_yticks([100])
    a.spines["bottom"].set_visible(i == 5)
    a.spines["left"].set_visible(False)
    a.tick_params(axis="y", length=0)
    if i < 5:
        a.set_xticks([])
    a.set_ylabel(f"{i + 1}", rotation=0, ha="right", va="center", fontsize=6, labelpad=6)

lanes[5].set_xticks([0, 6, 12, 18, 24, 30])
lanes[5].set_xlabel("Months since randomisation")
fig.text((LANE_L - 9.5) / W, (LANE_B + 3 * (LANE_H + LANE_GAP) - LANE_GAP / 2) / H,
         "Target-lesion size (% of first)", rotation=90, fontsize=7, ha="center", va="center")

axm.set_xlabel("Restricted PRIME projection PC1 (a.u.)")
axm.set_ylabel("Restricted PRIME projection PC2 (a.u.)")
axm.margins(0.12)
axm.set_xticks([-0.2, 0, 0.2, 0.4])
axm.set_yticks([-0.1, 0, 0.1, 0.2])
fig.canvas.draw()
for ax_, f in ((axm, axm.set_xticklabels), (axm, axm.set_yticklabels)):
    pass
axm.set_xticklabels([f"{v:g}".replace("-", "−") for v in axm.get_xticks()])
axm.set_yticklabels([f"{v:g}".replace("-", "−") for v in axm.get_yticks()])

handles = [Line2D([], [], color=BLUE, lw=1.2, label="Panitumumab + FOLFOX"),
           Line2D([], [], color=CHEMO, lw=1.2, label="FOLFOX"),
           Line2D([], [], marker="o", mfc="white", mec=INK, ls="", ms=4, label="First assessment"),
           Line2D([], [], marker="X", color=INK, ls="", ms=4.5, label="Last assessment"),
           Line2D([], [], marker="v", color=STATE_COLOURS["progression"], ls="", ms=4, label="Progression")]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(14 / W, 1.0), ncol=5, frameon=False, fontsize=6,
           handletextpad=0.3, columnspacing=1.2, handlelength=1.4)

arms = [ARM[g["arm_raw"].iloc[0]][0] for _, g, _ in rows]
legend = (
    "Six individual patients from the PRIME trial (NCT00364013; Project Data Sphere; randomised, first-line "
    "metastatic colorectal cancer), numbered 1 to 6 in this figure, three per arm "
    f"({', '.join(f'{k}, {v}' for k, v in enumerate(arms, 1))}). They were drawn at random, with a fixed seed, "
    "among patients with at least five lesion assessments; they are "
    "not selected for outcome. Left: each patient's path through the clinical state map, from the first lesion "
    "assessment (open circle, patient number) to the last (cross); dots, assessments; colour, trial arm. Axes are the "
    "restricted PRIME projection (Methods; mean positions by arm in Supplementary Fig. 14) (arbitrary units; they combine tumour and host measurements and are not "
    "validated clinical scales): PC1 rises mainly with liver tumour burden, disease extent and hepatic laboratory "
    "impairment; PC2 rises with marrow impairment grade (lower haemoglobin, platelets or white cells) and falls with "
    "liver and other tumour burden. Right: the same patients as a conventional timeline, the sum of the target-lesion "
    "diameters as a percentage of the first assessment (dashed line, 100%); triangle, progression recorded by the "
    "trial (absent when none was recorded). Observed individual data, not generated and not counterfactual."
)
save_panel(fig, "f6b_patient_paths", legend=legend,
           sources=["outputs/exploratory_pds_prime_trial_geometry/state_snapshots.csv.gz",
                    "outputs/exploratory_pds_prime_trial_geometry/subject_table.csv", VIS],
           confirm=["Selection rule re-run on the saved tables, not imported.",
                    "Paths run from the first to the last assessment, read from the assessment table (asserted to be "
                    "the same assessments as the timeline).",
                    "No axis direction words: marrow reserve is an impairment grade. Loadings from restricted_axis_correlations.json: PC1 r = 0.84 liver "
                    "tumour, 0.51 extent, 0.50 hepatic labs; PC2 r = 0.67 marrow grade, -0.59 liver tumour, -0.39 extent.",
                    "Time is months since randomisation (VISITDY/30.44) so that progression (PFS months) sits on the "
                    "same clock.",
                    "Patient numbers 1-6 are display labels, not trial identifiers."],
           message="Individual trial patients move irregularly through the state map while a linear record shows "
                   "only tumour size and progression.")
