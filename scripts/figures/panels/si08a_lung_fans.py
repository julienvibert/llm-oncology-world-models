"""si08a_lung_fans: lung-cancer case, linear fans (companion to f5a_case_fans).

Case: the declared rule of f5a_case_fans restricted to non-small-cell lung cancer landmarks
(_fans.select_eventful_query(cancer="NSCLC")). Run: uv run python scripts/figures/panels/si08a_lung_fans.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fans  # noqa: E402
from f5a_case_fans import SELECTION  # noqa: E402,F401

CANCER = "NSCLC"
ID = "si08a_lung_fans"

if __name__ == "__main__":
    qid, info = _fans.select_eventful_query(cancer=CANCER)
    case = _fans.load_query(qid)
    _fans.build(CANCER, ID, case=case, selection=SELECTION,
                lead="A second patient, with lung cancer, shown as in Fig. 5a: one observed course and four sets of generated futures.",
                tail="The same futures are drawn as dials in the next panel. The colorectal case is Supplementary Fig. 9.",
                confirm=[f"Case chosen by the f5a_case_fans rule restricted to NSCLC (landmark tag {info['tag']}; {info['checked_in_hash_order']} landmarks "
                         f"checked in hash order; 18-month threshold met). Observed course followed for {case['real']['end']:.0f} months.",
                         "The rule reads the observed course, not generated output; stated in the legend.",
                         "Treatment strips from per-agent / per-component exposure."])
