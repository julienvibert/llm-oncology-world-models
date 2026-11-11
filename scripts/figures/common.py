"""Shared settings for the manuscript figure panels (Nature Medicine).

Every panel is its own small figure at its printed size, with no panel letter. Panels are
assembled into figures by scripts/figures/assemble.py, which adds the letters. This keeps
figures easy to reorganise: change the layout file, not the panels.

Usage in a panel script:

    from scripts.figures.common import fk, panel, save_panel, PAL
    fig, ax = panel(width_mm=58, height_mm=45)
    ...
    save_panel(fig, "f2a_modal_share", legend="...", sources=["outputs/..."], confirm=[...])

figkit.py is the figure helper library (journal presets, plotting helpers, audit); it is
included so that the panels can be rebuilt with no other install.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib  # noqa: E402

# Panels are files, never windows. Agg on every platform, so a panel's page size does not
# depend on the interactive backend (the macOS backend snaps figure size to whole pixels).
matplotlib.use("Agg")
import figkit as fk  # noqa: E402

fk.use("nrj")  # Nature Medicine: Nature Research journals, 88/180 mm, 5-7 pt (house minimum 6 pt)
import matplotlib as _mpl  # noqa: E402

# no implicit colours: anything drawn without an explicit colour is black, then grey
_mpl.rcParams["axes.prop_cycle"] = _mpl.cycler(color=[fk.INK, fk.GREY])

PANELS = ROOT / "outputs" / "figures" / "panels"
ASSEMBLED = ROOT / "outputs" / "figures" / "assembled"

# ---------------------------------------------------------------------------------------
# Colour plan for the whole paper. A colour means one thing everywhere.
#   black   observed patients and real data
#   no greyscale for data: every entity has a colour; grey only for axes, reference lines
#   and true background
#   arms    four-arm experiment, amber to wine (PAL["arm_*"]); models: MODEL_COLOURS
#   treatments by class (MODALITY_COLOURS): targeted blue (osimertinib), chemotherapy purple
#   disease states: STATE_COLOURS, used in timelines, fans and state maps
# ---------------------------------------------------------------------------------------
INK = fk.INK
GREY = fk.GREY
LIGHT = fk.LIGHT
ACCENT = fk.ACCENT
BLUE, RED, ORANGE, SKY = fk.CATEGORICAL

PAL = {
    "observed": "#000000",
    "generated": GREY,
    "context": LIGHT,
    "accent": ACCENT,
    # treatments named in comparisons take the colour of their class (MODALITY_COLOURS)
    "osimertinib": "#2A6FBF",      # targeted therapy
    "chemotherapy": "#CC79A7",
    # four-arm experiment: increasing patient evidence, light amber to deep wine (no greys).
    # Observed patients stay black; Cox and donors are open black markers.
    "arm_patient_only": "#F2C57C",
    "arm_setting": "#E08A3C",
    "arm_generic": "#B5532A",
    "arm_matched": "#7B1E3A",
}

# Six generating models (identity only; chosen to avoid the state and treatment colours)
MODEL_COLOURS = {
    "Sonnet": "#E69F00",
    "Gemini 3.1 L": "#0072B2",
    "Gemma 4": "#009E73",
    "GLM-5.2": "#999933",
    "Kimi K3": "#882255",
    "Qwen 3.6": "#332288",
}

# Disease states (teal family; progression in vermilion so that it stays distinct from no
# evidence of disease under protanopia)
STATE_COLOURS = {
    "no evidence of disease": "#1F6F66",
    "response": "#6DB3A6",
    "stable": "#DFC782",
    "progression": "#D55E00",
    "death": "#000000",
    "unresolved": "#E3E7EA",
}

# Systemic treatment classes, for treatment strips in trajectory displays
MODALITY_COLOURS = {
    "chemotherapy": "#CC79A7",
    "endocrine": "#56B4E9",
    "targeted": "#2A6FBF",
    "immunotherapy": "#8C6D31",
    "other": "#3D3D3D",
}

# Cancer families, wherever a family is an entity in its own right (chosen to avoid models, arms, states)
FAMILY_COLOURS = {"lung": "#00798C", "gi": "#8F6BB3", "colorectal": "#8F6BB3", "sarcoma": "#A6761D", "breast": "#C44E52"}

CHEMO = PAL["chemotherapy"]  # chemotherapy and treatment B in schematics

MODEL_ORDER = ["Sonnet", "Gemini 3.1 L", "Gemma 4", "GLM-5.2", "Kimi K3", "Qwen 3.6"]


def panel(width_mm: float, height_mm: float, nrows: int = 1, ncols: int = 1, **kw):
    """A single panel at its printed size (mm). Returns (fig, ax or axes)."""
    fig, ax = fk.subplots(nrows, ncols, width=width_mm, height_mm=height_mm, **kw)
    return fig, ax


US_SPELLING = [("umour", "umor"), ("olour", "olor"), ("andomis", "andomiz"), ("aematolog", "ematolog"), ("ynaecolog", "ynecolog"), ("eukaemia", "eukemia"),
               ("grey", "gray"), ("FOLFOX alone", "FOLFOX")]


MODEL_DISPLAY = [(r"(?<!Claude\s)\bSonnet\b(?!\s5)", "Sonnet 5")]  # short model names; Sonnet with its version


def us_spelling(text: str) -> str:
    """Display text in US spelling (Nature Medicine) and with exact model names. Data keys and code names
    stay as they are."""
    for a, b in US_SPELLING:
        text = text.replace(a, b)
    for a, b in MODEL_DISPLAY:
        text = re.sub(a, b, text)
    return text


def _us_figure(fig) -> None:
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        t.set_text(us_spelling(t.get_text()))
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            formatter = axis.get_major_formatter()
            seq = getattr(formatter, "seq", None)
            if seq is not None:
                formatter.seq = [us_spelling(str(s)) for s in seq]
            # set_yticks(ticks, labels) stores the labels in a dict behind a FuncFormatter
            for held in getattr(getattr(formatter, "func", None), "args", ()):
                if isinstance(held, dict):
                    held.update({k: us_spelling(str(v)) for k, v in held.items()})


def save_panel(fig, panel_id: str, legend: str, sources: list[str], confirm: list[str] | None = None,
               message: str = "", strict: bool = True):
    """Audit and save a panel to outputs/figures/panels/<id>/<id>.{pdf,svg,png} with its
    handover note, and record its message and sources in <id>_meta.json."""
    _us_figure(fig)
    legend, message = us_spelling(legend), us_spelling(message)
    out = PANELS / panel_id
    out.mkdir(parents=True, exist_ok=True)
    w, h = (fig.get_size_inches() * 25.4).round(1)
    (out / f"{panel_id}_meta.json").write_text(json.dumps(
        {"id": panel_id, "message": message, "width_mm": float(w), "height_mm": float(h),
         "sources": sources, "confirm": confirm or []}, indent=2) + "\n")
    return fk.save(fig, str(out / panel_id), legend=legend, confirm=confirm or [], strict=strict)
