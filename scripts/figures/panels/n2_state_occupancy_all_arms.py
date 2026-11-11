"""Panel n2_state_occupancy_all_arms: the four-arm companion of n2_state_occupancy.

Message: in every arm, generated futures are 'responding' far more often and 'stable' almost never compared with the
registry course of the same patients. Same data, readings and denominators as n2_state_occupancy.py.
Run: uv run python scripts/figures/panels/n2_state_occupancy_all_arms.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from n2_state_occupancy import ARMS_ALL, build  # noqa: E402

build("n2_state_occupancy_all_arms", ARMS_ALL, 180.0, 40.0,
      scope="All four arms shown (n2_state_occupancy shows the patient record and matched-example arms).",
      message="In every arm, generated futures are 'responding' far more, and 'stable' almost never, than the registry.",
      confirm=[])
