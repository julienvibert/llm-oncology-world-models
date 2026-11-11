"""Shared helpers for the Figure 4 panels (rows, colours, point-and-interval drawing, aggregates)."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PANELS, PAL, INK, GREY, LIGHT, ROOT  # noqa: E402,F401

S = ROOT / "outputs" / "exploratory_four_arm_deep"      # exploratory; survival-time panels only
F = ROOT / "outputs" / "four_arm_preprint_20260928"     # final scored tables
F_REL = "outputs/four_arm_preprint_20260928"

# key, label, face colour, marker
ARMS = [("patient_only", "Record alone", PAL["arm_patient_only"], "o"),
        ("setting", "+ care setting", PAL["arm_setting"], "o"),
        ("generic_cohort", "+ generic examples", PAL["arm_generic"], "o"),
        ("patients_like_me", "+ matched examples", PAL["arm_matched"], "o")]
ROWS = ARMS + [("donor_generic", "Generic donors, direct", "white", "v"),
               ("donor_matched", "Matched donors, direct", "white", "s"),
               ("cox_cv", "Cox model", "white", "D")]
# Figure 4 row with panels b (f4c_survival_12m) and f4f_c_index side by side: same height and
# vertical margins (mm) so their seven forecaster rows line up; only b carries the row labels.
R2 = {"height": 44.0, "bottom": 10.5, "top": 1.0, "label_col": 24.5, "b_width": 106.0}

# the two donor cards: the donors shown to the generic and the matched arm, used directly
CARDS = [("generic", "generic_cohort", "donor_generic"), ("matched", "patients_like_me", "donor_matched")]


def final_table(name: str):
    """Read one of the final aggregate tables at full precision."""
    import pandas as pd
    return pd.read_csv(F / name, float_precision="round_trip")


def ptxt(p: float) -> str:
    """P value as printed in the figures."""
    if p < 0.001:
        return "P < 0.001"
    if p < 0.01:
        return f"P = {p:.3f}"
    return f"P = {p:.2f}"


def dot(ax, y, est, lo, hi, col, mk="o", ms=5.0):
    """Point with a 95% interval; black outline, colour = entity."""
    ax.plot([lo, hi], [y, y], color=INK, lw=0.75, zorder=2, solid_capstyle="butt")
    ax.plot(est, y, mk, ms=ms if mk == "o" else ms - 0.5, mfc=col, mec=INK, mew=0.6, zorder=3)


def point(ax, y, est, col, mk="o", ms=5.0):
    """Point without interval (for estimates saved without one)."""
    ax.plot(est, y, mk, ms=ms if mk == "o" else ms - 0.5, mfc=col, mec=INK, mew=0.6, zorder=3)


def write_aggregates(panel_id: str, header: list[str], rows: list[list]) -> Path:
    """Save the numbers a panel plots (estimates and bootstrap intervals) next to its outputs."""
    out = PANELS / panel_id
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{panel_id}_aggregates.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return p


def fix_axes(fig, ax, left, right=1.5, bottom=13.0, top=2.0):
    """Place the axes at fixed margins (mm) so rows of panels line up across the figure."""
    W, H = (fig.get_size_inches() * 25.4)
    fig.set_layout_engine("none")
    ax.set_position([left / W, bottom / H, (W - left - right) / W, (H - bottom - top) / H])
