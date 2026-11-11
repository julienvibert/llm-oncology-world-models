import sys
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/explore"))
import plot_sentinel_trust_map as P  # noqa: E402
man = {k: v for k, v in json.loads((ROOT/"scripts/explore/sentinel_battery_manifest_reconstructed_2026-10-14.json").read_text()).items() if not k.startswith("_")}
df = P.combined_table([ROOT/"runs/sentinel_usecase_hgsoc_v5_2026_07_13"], man)
for m in P.MODEL_ORDER:
    pf = P.score_hgsoc_delta(df, m)
    d2 = df.copy()
    d2["pfs_mo"] = d2["os_mo"]
    os_ = P.score_hgsoc_delta(d2, m)
    print(m, "PFS delta %.2f" % pf["platinum_delta"], "| OS delta %.2f" % os_["platinum_delta"])
