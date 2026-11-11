"""Small helpers shared by the Figure 2 panels (millimetre axes, model-ordered tables)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODEL_COLOURS, MODEL_ORDER, ROOT, fk, panel  # noqa: E402,F401
import pandas as pd  # noqa: E402

DATA = ROOT / "outputs" / "exploratory_figure_panels" / "fig1" / "data"


def canvas(width_mm: float, height_mm: float):
    fig, unused = panel(width_mm, height_mm)
    fig.delaxes(unused)
    fig.set_layout_engine(None)
    return fig


def axes_mm(fig, x: float, y: float, w: float, h: float):
    fw, fh = fig.get_size_inches() * 25.4
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])


def table(name: str) -> pd.DataFrame:
    t = pd.read_csv(DATA / name)
    assert set(t.model if "model" in t else []) == set(MODEL_ORDER)
    return t
