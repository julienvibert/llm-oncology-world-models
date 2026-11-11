"""Panel si05b_arrow_anatomy: what changed behind each arrow, and how much each change moved it.

Message: each arrow is the exact sum of the moves caused by the clinical variables that changed; the
same variable moves PC1 and PC2 by different amounts, so the arrow cannot be read back into one variable.
Data: _si05.py (frozen state map loadings x change in each encoded feature). Generated data only.
Run: uv run python scripts/figures/panels/si05b_arrow_anatomy.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import _si05 as S  # noqa: E402
from _f3util import axes_mm, new_fig  # noqa: E402
from _si05_style import EVENT_COLOURS  # noqa: E402
from common import INK, save_panel  # noqa: E402

d = S.load()
A = d["arrows"]
rows = []  # (kind, arrow index, text, pc1, pc2)
for n, a in enumerate(A, 1):
    rows.append(("head", n, a, 0, 0))
    for f, b, c, p1, p2 in a["changes"]:
        rows.append(("var", n, f"{S.name(f, d['labels'])}: {b} → {c}", p1, p2))
    rows.append(("sum", n, "Whole arrow", a["dx"], a["dy"]))
    rows.append(("gap", n, "", 0, 0))
rows.pop()
RH = 4.0
W, H = 108, RH * len(rows) + 14
fig = new_fig(W, H)
TOP = H - 1.5
LEFT, BW, GAP = 1.0, 24.0, 5.0
x1 = 55.0
x2 = x1 + BW + GAP
axs = [axes_mm(fig, x, 9, BW, TOP - 9 - 0.5) for x in (x1, x2)]
LIM = 0.14
for ax, key, xl in zip(axs, (3, 4), ("State-map PC1\nmove (a.u.)", "State-map PC2\nmove (a.u.)")):
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.1, 0.2])
    ax.set_xticklabels(["−0.1", "0", "0.1", "0.2"])
    ax.axvline(0, color=INK, lw=0.5)
    ax.set_xlabel(xl, fontsize=6, labelpad=1)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    for i, (kind, n, txt, p1, p2) in enumerate(rows):
        if kind in ("head", "gap"):
            continue
        col = EVENT_COLOURS[A[n - 1]["event"]]
        v = p1 if key == 3 else p2
        ax.barh(i, v, height=0.7, color=col, edgecolor=INK if kind == "sum" else col, lw=0.8 if kind == "sum" else 0)
        ax.text(v + (0.008 if v >= 0 else -0.008), i, ("0.00" if abs(v) < 0.005 else f"{v:+.2f}".replace("-", "−")), fontsize=6, ha="left" if v >= 0 else "right",
                va="center", color=INK)
    ax.set_xlim(-LIM, 0.31)
for i, (kind, n, txt, p1, p2) in enumerate(rows):
    yy = TOP - (i + 0.5) * (TOP - 9 - 0.5) / len(rows) - 0.0
    ymm = 9 + (len(rows) - i - 0.5) * (TOP - 9 - 0.5) / len(rows)
    fx = lambda x: x / W  # noqa: E731
    fy = ymm / H
    if kind == "head":
        a = txt
        col = EVENT_COLOURS[a["event"]]
        fig.text(fx(LEFT + 2.2), fy, str(n), fontsize=6, color="white", ha="center", va="center", fontweight="bold",
                 bbox=dict(boxstyle="circle,pad=0.25", fc=col, ec="none"))
        lab = a["event"] + (f" ({a['detail']})" if a["detail"] else "") + f", month {a['month']:.1f}"
        fig.text(fx(LEFT + 6), fy, lab, fontsize=6, ha="left", va="center", fontweight="bold")
    elif kind == "var":
        fig.text(fx(LEFT + 6), fy, txt, fontsize=6, ha="left", va="center")
    elif kind == "sum":
        fig.text(fx(LEFT + 6), fy, txt, fontsize=6, ha="left", va="center", style="italic")
legend = (
    "What changed behind each arrow of panel a, and how much each change moved it. Each group is one arrow "
    "(number and colour as in panel a) with the clinical variables that changed at that event, written as value "
    "before → value after (burden bands from absent to diffuse; ECOG performance status; symptom and "
    "side-effect grades). Bars: how far each variable's change moved the patient along PC1 (left) and along "
    "PC2 (right), in arbitrary units of the state map; both axes use the same scale. A positive move is a higher "
    "coordinate, not a better or worse condition. The outlined bar is the whole arrow and equals the "
    "sum of the bars above it, exactly: the contributions sum by linear algebra, because the map is a fixed "
    "linear combination of the 20 encoded features (loadings in Extended Data Fig. 1). A change in one variable moves a patient along both axes, so a "
    "position on the map cannot be turned back into one clinical variable. The panel describes what changed in "
    "the generated record; it identifies no biological mechanism. Same generated course as panel a."
)
save_panel(fig, "si05b_arrow_anatomy", legend=legend,
           sources=["outputs/paper_state_map/state_projection.csv.gz", "outputs/paper_state_map/feature_loadings.csv",
                    "scripts/figures/panels/_si05.py"],
           confirm=["Exactness checked in code (|sum of bars - arrow| < 1e-9 for PC1 and PC2).",
                    "Plain wording of encoded values (for example 'grade 0 to grade 2' for the worst "
                    "active side effect; 'ECOG 1 to ECOG 2') follows the encoding in state_map_contract.json."],
           message="Each arrow is the exact sum of the moves caused by the variables that changed.")
