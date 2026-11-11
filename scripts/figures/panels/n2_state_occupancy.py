"""Panel n2_state_occupancy: disease state month by month, registry against generated futures.

Message: generated futures are 'responding' far more often, and 'stable' almost never, compared
with the registry course of the same patients.
Data: outputs/exploratory_n2_state_occupancy/n2_state_occupancy_v1.csv (aggregate), cancer all;
observed = registry reading; arms = calendar reading (futures read on each patient's own
assessment dates, as in the scored comparison).
Shares are distributions among the patients retained at each month (dead by then, or alive and
still followed); not censoring-adjusted. Same 437 patients as f5c_course_disagreement.
Run: uv run python scripts/figures/panels/n2_state_occupancy.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, STATE_COLOURS, save_panel  # noqa: E402
from _f3util import axes_mm, new_fig  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "release"))
import build_source_data as SD  # noqa: E402
import mask_small_cells as MS  # noqa: E402

FULL = pd.read_csv(ROOT / "outputs" / "exploratory_n2_state_occupancy" / "n2_state_occupancy_v1.csv")
# Same suppression as the Source Data (scripts/release/build_source_data.py, SPECS): registry shares implying 1 to 4
# patients, plus one complementary state, are combined into one neutral band (disclosure control).
_masked, _ = MS.mask(FULL, SD.SPECS["n2_state_occupancy_v1.csv"])
SUPP = FULL["share"].where(pd.to_numeric(_masked["share"], errors="coerce").isna() & FULL["share"].notna())
d = FULL.assign(suppressed=SUPP.notna())
d = d[d.cancer.eq("all")]
SOURCES = [("observed", "registry", "Observed"), ("patient_only", "calendar", "Record alone"),
           ("patients_like_me", "calendar", "Matched examples")]
# bottom to top; death kept as its own band at the top
STATES = [("no_active_disease", "no evidence of disease", "No disease"), ("responding", "response", "Response"),
          ("stable", "stable", "Stable"), ("progressing", "progression", "Progression"),
          ("unresolved", "unresolved", "Unresolved"), ("dead", "death", "Dead")]
months = np.arange(0, 25)

ARMS_ALL = [("observed", "registry", "Observed"), ("patient_only", "calendar", "Record alone"),
            ("setting", "calendar", "+ care setting"), ("generic_cohort", "calendar", "+ generic examples"),
            ("patients_like_me", "calendar", "+ matched examples")]


def build(panel_id: str, sources: list, W: float, H: float, scope: str, message: str, confirm: list) -> None:
    """Stacked monthly state shares, one small multiple per source, key at the right."""
    fig = new_fig(W, H)
    L, B, PH, GAP, KEY = 8.0, 7.5, H - 10.5, 3.4, 19.7
    PW = (W - L - KEY - (len(sources) - 1) * GAP) / len(sources)
    tab, n_supp = {}, {}
    for j, (src, reading, title) in enumerate(sources):
        ax = axes_mm(fig, L + j * (PW + GAP), B, PW, PH)
        sel = d[d.source.eq(src) & d.reading.eq(reading)]
        x = sel.pivot_table(index="month", columns="state", values="share")
        x = x.reindex(index=months, columns=[s for s, *_ in STATES]).fillna(0.0)
        assert np.allclose(x.sum(axis=1), 1.0, atol=1e-6), src
        sup = sel[sel.suppressed].pivot_table(index="month", columns="state", values="share", aggfunc="sum")
        sup = sup.reindex(index=months, columns=x.columns).fillna(0.0)
        x = x - sup  # suppressed shares leave their state ...
        x["suppressed"] = sup.sum(axis=1)  # ... and are drawn together in one neutral band
        tab[src] = x + sup.reindex(columns=x.columns, fill_value=0.0)  # numbers in the legend use unsuppressed months only
        tab[src]["suppressed"] = x["suppressed"]
        n_supp[src] = int((x["suppressed"] > 0).sum())
        order = [s for s, *_ in STATES[:-1]] + ["suppressed", "dead"]
        polys = ax.stackplot(months, *(100 * x[s].to_numpy() for s in order),
                             colors=[STATE_COLOURS[c] for _, c, _ in STATES[:-1]] + ["white", STATE_COLOURS["death"]], lw=0)
        polys[order.index("suppressed")].set_hatch("//////")
        polys[order.index("suppressed")].set_edgecolor("#8A8A8A")
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 12, 24])
        ax.set_yticks([0, 50, 100])
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Patients (%)", labelpad=1)
        ax.text(0.5, 1.02, title, transform=ax.transAxes, fontsize=6, ha="center", va="bottom")
        ax.set_xlabel("Months", labelpad=0.5)
    n = d[d.source.eq("observed")].groupby("month").n_patients.first()
    kx = W - KEY + 2.0
    fig.text(kx / W, 0.6 / H, f"n = {n[0]}, {n[12]},\n{n[24]} at months\n0, 12, 24",
             fontsize=6, ha="left", va="bottom", linespacing=1.0)
    keys = [(STATE_COLOURS[c], lab, None) for _, c, lab in STATES[::-1]]
    if any(n_supp.values()):
        keys.insert(1, ("white", "Suppressed", "//////"))
    for i, (col, lab, hatch) in enumerate(keys):
        y = B + PH - 1.8 - i * 2.9
        fig.add_artist(Rectangle((kx / W, y / H), 2.2 / W, 2.0 / H, transform=fig.transFigure, facecolor=col,
                                 edgecolor="#9A9A9A" if lab in ("Unresolved", "Suppressed") else "none", lw=0.3,
                                 hatch=hatch))
        fig.text((kx + 3.0) / W, (y + 1.0) / H, lab, fontsize=6, va="center", ha="left")

    def pct(src: str, state: str, lo: int, hi: int) -> str:
        v = 100 * tab[src].loc[lo:hi, state]
        return f"{v.min():.0f}-{v.max():.0f}%"

    shown = [s for s, *_ in sources if s != "observed"]
    numbers = " ".join(
        f"{t}: responding {pct(s, 'responding', 6, 12)}, stable {pct(s, 'stable', 6, 12)} in months 6-12; dead "
        f"{100 * tab[s].loc[12, 'dead']:.0f}% and {100 * tab[s].loc[24, 'dead']:.0f}% at 12 and 24 months."
        for s, _, t in sources)
    legend = (
        "Disease state month by month over 24 months after the landmark, in the registry course and in the generated "
        "futures of the same patients (GENIE-BPC; the 437 patients of Fig. 5c). Observed: last documented state carried "
        "forward on the registry assessments. Generated: each future read on the patient's own assessment dates, as in "
        "the scored comparison; shares averaged within patient first (7 to 10 accepted futures per arm), then over "
        f"patients. {scope} Each month shows the distribution among the patients retained at that month (dead by then, "
        "or alive and still followed); censored patients leave the denominator, so the curves are not "
        f"censoring-adjusted estimates (n = {n[0]}, {n[12]} and {n[24]} at months 0, 12 and 24). {numbers} "
        "Unresolved, time with no documented or generated state at that date. "
        + ("Suppressed (hatched), state breakdown suppressed for disclosure control: registry states with 1 to 4 "
           "patients and one complementary state are drawn together, preserving their combined share; this is not an "
           "unknown or unassessed clinical state, and the Source Data blank the same cells. " if any(n_supp.values()) else "")
        + "DeepSeek V4 Flash. "
        f"Arms shown: {len(shown)}.")
    save_panel(fig, panel_id, legend=legend,
               sources=["outputs/exploratory_n2_state_occupancy/n2_state_occupancy_v1.csv",
                        "outputs/exploratory_n2_state_occupancy/manifest.json"],
               confirm=confirm, message=message)


if __name__ == "__main__":
    build("n2_state_occupancy", SOURCES, 96.0, 33.0,
          scope="Patient record only and matched examples shown; all four arms in n2_state_occupancy_all_arms.",
          message="Generated futures are 'responding' far more, and 'stable' almost never, compared with the registry.",
          confirm=["Small-multiple labels (Observed, Record alone, Matched examples) identify the source, not a conclusion.",
                   "Two of the four arms shown for space; all four in n2_state_occupancy_all_arms."])
