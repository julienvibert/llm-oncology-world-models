from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "explore" / "audit_native_atlas_geometry_v1.py"


def test_native_atlas_geometry_self_test() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--self-test"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "self-test OK" in completed.stdout
