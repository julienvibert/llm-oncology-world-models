"""f5a_case_fans: one observed patient and four distributions of clinical trajectories (linear view).

Message: for the same patient, the four ways of supplying information give visibly different
distributions of disease and treatment trajectories, shown against the observed trajectory.
Case: declared rule (_fans.select_eventful_query): among landmarks with 10 accepted futures in
every arm and an observed progression inside the 24-month window, the lowest SHA-256 of
"meeting-display-v2:" + query_id. Drawn by _fans.py (shared with si08a_lung_fans and si09a_colorectal_fans),
from the four-arm readers (four_arm_eligibility_v2 / four_arm_readers_v2).

Run: uv run python scripts/figures/panels/f5a_case_fans.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fans  # noqa: E402

SELECTION = ("the case was selected for a long, eventful observed course: ten accepted futures in every arm, "
             "at least 18 months of observed follow-up after the landmark, and at least two changes of documented "
             "disease state in the 24-month window, one of them a progression, then the lowest SHA-256 hash of the "
             "query identifier (the rule reads the observed course, not the generated futures)")

if __name__ == "__main__":
    qid, info = _fans.select_eventful_query()
    case = _fans.load_query(qid)
    _fans.build(case["cancer"], "f5a_case_fans", case=case, selection=SELECTION,
                lead="One observed patient and four distributions of generated clinical trajectories, linear view.",
                tail="The same futures are drawn as dials in Supplementary Fig. 12. Lung and colorectal cases, "
                     "Supplementary Figs. 10 and 11.",
                confirm=[f"Case chosen by the declared (tightened) rule (landmark tag {info['tag']}; {info['complete_landmarks']} landmarks "
                         f"have 10 futures in every arm; third in hash order qualified; the 18-month threshold was met, no relaxation to 12). Observed course: "
                         f"followed for {case['real']['end']:.0f} months, six documented state intervals including progression.",
                         "The rule reads the observed course (not generated output); stated in the legend.",
                         "Treatment strips from per-component exposure (generated) and per-agent dates (observed; an "
                         "investigational agent from month 11 is drawn as exposure of undocumented duration)."])
