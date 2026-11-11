"""Panel ed07a_km_at_risk: generated survival against real GENIE-BPC survival, by cancer, with numbers at risk.

Kaplan-Meier curves by cancer in the arm colours of the main figures, numbers at risk under each curve.
Run: uv run python scripts/figures/panels/ed07a_km_at_risk.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "explore"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import build_query_universe_v1 as QU  # noqa: E402
import genie_operator_registry_v2 as GV2  # noqa: E402
from common import PAL, ROOT, fk, save_panel  # noqa: E402

DEEP = ROOT / "outputs" / "exploratory_four_arm_deep"
gen = pd.read_csv(DEEP / "generated_side.csv")
real = pd.read_csv(DEEP / "real_side.csv")
ARMS = [
    ("patient_only", "Record alone", "arm_patient_only"),
    ("setting", "+ care setting", "arm_setting"),
    ("generic_cohort", "+ generic examples", "arm_generic"),
    ("patients_like_me", "+ matched examples", "arm_matched"),
]
CANCERS = [("BRCA", "Breast cancer"), ("CRC", "Colorectal cancer"), ("NSCLC", "Non-small cell lung cancer")]

queries = {}
for line in (ROOT / "outputs" / "exploratory_query_universe" / "queries.jsonl").open():
    q = json.loads(line)
    queries[q["query_id"]] = q
followup = {}
for cancer, folder in QU.COHORTS.items():
    genie = QU.GENIE_ROOT / folder
    if not genie.exists():
        continue
    patients = {p.pid: p for p in GV2.load(genie)}
    for qid, q in queries.items():
        if q["cancer"] == cancer and q["patient"] in patients:
            followup[qid] = patients[q["patient"]].last_contact - q["tau_months"]

XMAX = 36.0
TICKS = [0, 12, 24, 36]


def curve(km):
    sf = km.survival_function_.iloc[:, 0]
    t, v = sf.index.to_numpy(float), sf.to_numpy(float)
    keep = t <= XMAX
    t, v = t[keep], v[keep]
    return np.append(t, XMAX), np.append(v, v[-1])


def band(km):
    ci = km.confidence_interval_
    t = ci.index.to_numpy(float)
    keep = t <= XMAX
    t = np.append(t[keep], XMAX)
    lo = ci.iloc[:, 0].to_numpy(float)[keep]
    hi = ci.iloc[:, 1].to_numpy(float)[keep]
    return t, np.append(lo, lo[-1]), np.append(hi, hi[-1])


W, H = 180, 80
fig = fk.figure(width=W, height_mm=H)
L, R, GAP = 0.125, 0.975, 0.05
aw = (R - L - 2 * GAP) / 3
rows = ["Observed (real)"] + [a[1] for a in ARMS]
cols = ["black"] + [PAL[a[2]] for a in ARMS]
axes, tabs = [], []
ns = {}
for i, (cancer, cname) in enumerate(CANCERS):
    x0 = L + i * (aw + GAP)
    ax = fig.add_axes([x0, 0.41, aw, 0.45])
    tb = fig.add_axes([x0, 0.02, aw, 0.24])
    sub = real[real.cancer == cancer]
    dur = np.where(sub.died, sub.real_os, [followup.get(q, np.nan) for q in sub.query_id])
    ok = ~np.isnan(dur)
    km = KaplanMeierFitter().fit(dur[ok], sub.died.astype(int).to_numpy()[ok])
    t, lo, hi = band(km)
    ax.fill_between(t, lo, hi, step="post", color="black", alpha=0.15, lw=0)
    ax.step(*curve(km), where="post", color="black", lw=1.6, zorder=5)
    risk = [[int((dur[ok] >= m).sum()) for m in TICKS]]
    ns[cancer] = {"real": int(ok.sum())}
    for arm, lab, c in ARMS:
        g = gen[(gen.cancer == cancer) & (gen.arm == arm)].dropna(subset=["gen_os"])
        ax.step(*curve(KaplanMeierFitter().fit(g.gen_os, np.ones(len(g)))), where="post", color=PAL[c], lw=1.1)
        risk.append([int((g.gen_os >= m).sum()) for m in TICKS])
        ns[cancer][arm] = len(g)
    ax.set_xlim(0, XMAX)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xticks(TICKS)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "50", "100"])
    ax.set_title(f"{cname}", fontsize=7)
    ax.set_xlabel("Months from the landmark")
    if i == 0:
        ax.set_ylabel("Patients alive (%)")
    else:
        ax.tick_params(axis="y", labelleft=False)
    # numbers at risk
    tb.set_xlim(0, XMAX)
    tb.set_ylim(len(rows) - 0.5, -0.5)
    tb.axis("off")
    for r, vals in enumerate(risk):
        for m, v in zip(TICKS, vals):
            tb.text(m, r, f"{v:,}", color=cols[r], ha="center" if 0 < m < XMAX else ("left" if m == 0 else "right"),
                    va="center", fontsize=6)
        if i == 0:
            tb.text(-0.02, r, rows[r], color=cols[r], transform=tb.get_yaxis_transform(), ha="right", va="center", fontsize=6)
    tb.text(0.5, 1.02, "Number at risk (real: patients; generated: futures)" if i == 1 else "", transform=tb.transAxes,
            ha="center", va="bottom", fontsize=6)
    axes.append(ax)
    tabs.append(tb)

h = [Line2D([], [], color="black", lw=1.6)] + [Line2D([], [], color=PAL[a[2]], lw=1.4) for a in ARMS]
fig.legend(h, ["Observed: real GENIE-BPC patients (95% interval, shaded)"] + [a[1] for a in ARMS], loc="upper center",
           bbox_to_anchor=(0.54, 1.0), ncol=5, frameon=False, fontsize=6, handlelength=1.8, columnspacing=1.0)

rowsn = "; ".join(f"{c}: {v['real']} real patients, generated futures " + ", ".join(str(v[a[0]]) for a in ARMS)
                  for c, v in ns.items())
legend = (
    "Survival over 36 months from the landmark (the start of the treatment line being forecast): generated futures against real "
    "patients, in three cancers. Black line, Kaplan-Meier survival of the real GENIE-BPC patients, censored at last contact, shaded "
    "band 95% confidence interval. Coloured lines, survival curves of the generated futures in the four evidence conditions (light "
    "amber, patient record only, to wine, matched examples added). Generated times are all counted as deaths, so these curves show "
    "the distribution of generated survival times and carry no censoring. Table: patients still at risk in the real data and "
    "generated futures still alive at 0, 12, 24 and 36 months. n (real patients; generated futures in the order patient record, "
    "care setting, generic, matched): " + rowsn + "."
)
save_panel(
    fig, "ed07a_km_at_risk", legend=legend,
    sources=["outputs/exploratory_four_arm_deep/generated_side.csv", "outputs/exploratory_four_arm_deep/real_side.csv",
             "outputs/exploratory_query_universe/queries.jsonl", "GENIE-BPC (aggregate Kaplan-Meier curves only)"],
    confirm=[
        "Numbers at risk computed from the same durations as the curves (real: duration at or above the month; generated: generated survival time at or above the month).",
        "Real patients with no usable follow-up (missing duration) are dropped from the Kaplan-Meier fit (n printed in legend).",
        "Generated times are treated as events without censoring.",
    ],
    message="Generated survival is above real survival early and falls below it at long follow-up in colorectal and lung cancer.",
)
