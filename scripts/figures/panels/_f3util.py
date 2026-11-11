"""Small helpers shared by the Figure 3 panels (axes placed in millimetres)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import fk  # noqa: E402


def new_fig(width_mm: float, height_mm: float):
    fig = fk.figure(width=width_mm, height_mm=height_mm)
    fig.set_layout_engine(None)
    return fig


def axes_mm(fig, left: float, bottom: float, width: float, height: float):
    """Axes with its lower-left corner at (left, bottom) mm and the given size in mm."""
    w, h = fig.get_size_inches() * 25.4
    return fig.add_axes([left / w, bottom / h, width / w, height / h])


def minus(v: float, fmt: str = "+.1f") -> str:
    return format(v, fmt).replace("-", "−")
