"""Panel n2_state_occupancy_main: disease state month by month, registry and two settings, full width.

Same data, readings and denominators as n2_state_occupancy.py (437 patients of all three cancers).
Run: uv run python scripts/figures/panels/n2_state_occupancy_main.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from n2_state_occupancy import SOURCES, build  # noqa: E402

build("n2_state_occupancy_main", SOURCES, 180.0, 44.0,
      scope="Observed, record alone and matched examples (all four settings in Extended Data).",
      message="Generated futures are in response far more often, and stable almost never, compared with the registry.",
      confirm=["Main-figure version of Extended Data panel n2_state_occupancy_all_arms."])
