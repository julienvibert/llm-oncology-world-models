"""Gate B1b — per-trajectory rollups on endpoint_rows.csv.

Locks: ``n_dose_modifications_total`` / ``n_treatment_interruptions_total``
via course-date-window matching; ``time_diagnosis_to_first_treatment_days``;
``first_line_matches_policy_canonical`` with hardcoded mapping
+ null for unknown policies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from synthonccases.oncollm.derivation import (
    CellKey,
    TrajectoryArtifact,
    _first_line_matches_policy_canonical,
    build_endpoint_rows,
)
from synthonccases.oncollm.poc_summary import TreatmentCourseRow


def _make_artifact(
    *,
    baseline: dict[str, Any] | None = None,
    ledger: list[dict[str, Any]] | None = None,
    snapshots: list[dict[str, Any]] | None = None,
    policy_id: str = "A_THP",
) -> TrajectoryArtifact:
    cell = CellKey(
        trajectory_id="t1", condition_id="c", policy_id=policy_id,
        model_id="m", model_short="m", future_index=1,
        tumor_oncotree_code="BRCA", conditioning_label="x",
        conditioning_hash="h", accepted=True,
    )
    materialized = {
        "baseline": baseline or {},
        "ledger": ledger or [],
        "snapshots": snapshots or [
            {
                "date": "2024-12-31",
                "state_id": "s_final",
                "outcome": {"type": "death", "date": "2024-12-31", "cause": "disease"},
            }
        ],
        "treatment_history": [],
    }
    return TrajectoryArtifact(
        cell=cell, trajectory_dir=Path("/tmp"),
        parsed_ledger=(), materialized=materialized,
        trajectory_result={},
    )


def _course(
    line: int, start: str, end: str | None = None, regimen: str = "THP",
) -> TreatmentCourseRow:
    """Helper to build a TreatmentCourseRow for unit tests."""
    from datetime import date
    return TreatmentCourseRow(
        course_index=line - 1,
        line=line,
        regimen=regimen,
        treatment_id=f"T{line}",
        start_date=date.fromisoformat(start),
        end_date=date.fromisoformat(end) if end else None,
    )


# ---------------------------------------------------------------------------
# _first_line_matches_policy_canonical
# ---------------------------------------------------------------------------


def test_first_line_match_a_thp_matches() -> None:
    cell = CellKey(
        trajectory_id="t", condition_id="c", policy_id="A_THP",
        model_id="m", model_short="m", future_index=1,
        tumor_oncotree_code="BRCA", conditioning_label="x",
        conditioning_hash="h", accepted=True,
    )
    courses = [_course(line=1, start="2024-02-15", end="2024-08-15",
                       regimen="trastuzumab/pertuzumab/docetaxel")]
    assert _first_line_matches_policy_canonical(cell, courses) is True


def test_first_line_match_b_tdm1_matches() -> None:
    cell = CellKey(
        trajectory_id="t", condition_id="c", policy_id="B_TDM1",
        model_id="m", model_short="m", future_index=1,
        tumor_oncotree_code="BRCA", conditioning_label="x",
        conditioning_hash="h", accepted=True,
    )
    courses = [_course(line=1, start="2024-02-15", regimen="trastuzumab-emtansine")]
    assert _first_line_matches_policy_canonical(cell, courses) is True


def test_first_line_match_a_thp_mismatch_returns_false() -> None:
    cell = CellKey(
        trajectory_id="t", condition_id="c", policy_id="A_THP",
        model_id="m", model_short="m", future_index=1,
        tumor_oncotree_code="BRCA", conditioning_label="x",
        conditioning_hash="h", accepted=True,
    )
    # Policy expects THP, course is T-DXd
    courses = [_course(line=1, start="2024-02-15", regimen="trastuzumab/deruxtecan")]
    assert _first_line_matches_policy_canonical(cell, courses) is False


def test_first_line_match_unknown_policy_returns_null() -> None:
    """Policy not in the hardcoded mapping → null. Do not silently
    default to a class."""
    cell = CellKey(
        trajectory_id="t", condition_id="c", policy_id="C_unknown_policy",
        model_id="m", model_short="m", future_index=1,
        tumor_oncotree_code="BRCA", conditioning_label="x",
        conditioning_hash="h", accepted=True,
    )
    courses = [_course(line=1, start="2024-02-15", regimen="THP")]
    assert _first_line_matches_policy_canonical(cell, courses) is None


def test_first_line_match_no_1L_course_returns_null() -> None:
    cell = CellKey(
        trajectory_id="t", condition_id="c", policy_id="A_THP",
        model_id="m", model_short="m", future_index=1,
        tumor_oncotree_code="BRCA", conditioning_label="x",
        conditioning_hash="h", accepted=True,
    )
    courses = [_course(line=2, start="2024-08-15", regimen="THP")]  # 2L only
    assert _first_line_matches_policy_canonical(cell, courses) is None


# ---------------------------------------------------------------------------
# build_endpoint_rows integration — rollup columns populate correctly
# ---------------------------------------------------------------------------


def test_endpoint_rows_time_to_first_treatment_days() -> None:
    """diagnosis_date → 1L start_date delta."""
    ledger = [
        {"date": "2024-02-15", "type": "systemic_treatment_start",
         "line": 1, "regimen": {"label": "THP",
                                "components": ["trastuzumab", "pertuzumab", "docetaxel"]}},
        {"date": "2024-12-31", "type": "death", "cause_category": "cancer",
         "cause": "disease progression"},
    ]
    baseline = {"identity_card": {"diagnosis": {
        "oncotree_code": "BRCA", "diagnosis_date": "2024-01-15"}}}
    art = _make_artifact(baseline=baseline, ledger=ledger)
    df = build_endpoint_rows([art])
    # 2024-01-15 → 2024-02-15 = 31 days
    assert df.iloc[0]["time_diagnosis_to_first_treatment_days"] == 31


def test_endpoint_rows_dose_mod_rollup_across_courses() -> None:
    """n_dose_modifications_total sums across closed courses."""
    ledger = [
        {"date": "2024-02-15", "type": "systemic_treatment_start",
         "line": 1, "regimen": {"label": "THP",
                                "components": ["trastuzumab", "pertuzumab", "docetaxel"]}},
        # Modification during course 1
        {"date": "2024-04-15", "type": "systemic_treatment_modification",
         "modification_type": "dose_reduction"},
        # Hold during course 1 (counts as both dose_mod AND interruption)
        {"date": "2024-05-15", "type": "systemic_treatment_modification",
         "modification_type": "dose_hold"},
        {"date": "2024-08-15", "type": "systemic_treatment_switch",
         "line": 2, "regimen": {"label": "T-DM1",
                                "components": ["trastuzumab", "emtansine"]}},
        # Modification during course 2
        {"date": "2024-10-15", "type": "systemic_treatment_modification",
         "modification_type": "dose_hold"},
        {"date": "2024-12-31", "type": "death", "cause_category": "cancer"},
    ]
    baseline = {"identity_card": {"diagnosis": {
        "oncotree_code": "BRCA", "diagnosis_date": "2024-01-15"}}}
    art = _make_artifact(baseline=baseline, ledger=ledger)
    df = build_endpoint_rows([art])
    # 2 mods in course 1, 1 mod in course 2 → 3 total.
    assert df.iloc[0]["n_dose_modifications_total"] == 3
    # 1 dose_hold in course 1, 1 dose_hold in course 2 → 2 interruptions total.
    assert df.iloc[0]["n_treatment_interruptions_total"] == 2


def test_endpoint_rows_policy_match_propagates_to_row() -> None:
    """first_line_matches_policy_canonical column populated via the helper."""
    ledger = [
        {"date": "2024-02-15", "type": "systemic_treatment_start",
         "line": 1, "regimen": {"label": "THP",
                                "components": ["trastuzumab", "pertuzumab", "docetaxel"]}},
        {"date": "2024-12-31", "type": "death", "cause_category": "cancer"},
    ]
    baseline = {"identity_card": {"diagnosis": {
        "oncotree_code": "BRCA", "diagnosis_date": "2024-01-15"}}}
    art = _make_artifact(baseline=baseline, ledger=ledger, policy_id="A_THP")
    df = build_endpoint_rows([art])
    assert bool(df.iloc[0]["first_line_matches_policy_canonical"]) is True


def test_endpoint_rows_unknown_policy_match_is_null() -> None:
    ledger = [
        {"date": "2024-02-15", "type": "systemic_treatment_start",
         "line": 1, "regimen": {"label": "THP", "components": ["trastuzumab", "pertuzumab", "docetaxel"]}},
        {"date": "2024-12-31", "type": "death", "cause_category": "cancer"},
    ]
    baseline = {"identity_card": {"diagnosis": {
        "oncotree_code": "BRCA", "diagnosis_date": "2024-01-15"}}}
    art = _make_artifact(
        baseline=baseline, ledger=ledger, policy_id="C_unrecognized",
    )
    df = build_endpoint_rows([art])
    assert pd.isna(df.iloc[0]["first_line_matches_policy_canonical"])
