"""Gate B1b — treatment_rows columns: ``canonical_regimen``,
``component_classes_*``,
``n_dose_modifications_in_course``, ``n_interruptions_in_course``,
``best_response_class`` (5-value enum), ``had_response_event``,
``n_response_events_in_course``, ``time_to_first_response_days``.

Locks each of the 5 best_response_class buckets + the null
"not-classifiable" path, the course-date-window matching for dose-mod
counting, the model-emitted component-classes passthrough, and the
canonical-regimen lift.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from synthonccases.oncollm.derivation import (
    CellKey,
    TrajectoryArtifact,
    _canonicalize_regimen,
    _count_modifications_in_course,
    _derive_course_response_columns,
    build_treatment_rows,
)
from synthonccases.oncollm.poc_summary import TreatmentCourseRow


# ---------------------------------------------------------------------------
# _canonicalize_regimen — token-bag matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regimen,expected", [
    # TUC_TRIPLET
    ("tucatinib + capecitabine + trastuzumab", "TUC_TRIPLET"),
    # T-DXd
    ("trastuzumab-deruxtecan", "T-DXd"),
    # SACITUZUMAB_GOVITECAN
    ("sacituzumab govitecan", "SACITUZUMAB_GOVITECAN"),
    # T-DM1 (multiple fallback rules)
    ("trastuzumab-emtansine", "T-DM1"),
    ("DM1", "T-DM1"),
    ("TDM1", "T-DM1"),
    # THP
    ("trastuzumab/pertuzumab/docetaxel", "THP"),
    ("THP", "THP"),
    # TCHP
    ("tchp", "TCHP"),
    # OTHER fallback
    ("some_unknown_regimen", "OTHER"),
    ("", "OTHER"),
    (None, "OTHER"),
])
def test_canonicalize_regimen(regimen, expected) -> None:
    assert _canonicalize_regimen(regimen) == expected


# ---------------------------------------------------------------------------
# _count_modifications_in_course — course-date-window matching
# ---------------------------------------------------------------------------


def _course(line: int, start: str, end: str | None = None) -> TreatmentCourseRow:
    return TreatmentCourseRow(
        course_index=line - 1, line=line, regimen="THP", treatment_id=f"T{line}",
        start_date=date.fromisoformat(start),
        end_date=date.fromisoformat(end) if end else None,
    )


def test_count_modifications_in_course_window_filtering() -> None:
    """Counts mods with event_date in [start, end] inclusive."""
    ledger = [
        # Before course
        {"date": "2024-01-15", "type": "systemic_treatment_modification",
         "modification_type": "dose_reduction"},
        # In course
        {"date": "2024-03-15", "type": "systemic_treatment_modification",
         "modification_type": "dose_hold"},
        {"date": "2024-05-01", "type": "systemic_treatment_modification",
         "modification_type": "dose_reduction"},
        # After course
        {"date": "2024-09-15", "type": "systemic_treatment_modification",
         "modification_type": "dose_hold"},
        # Wrong event type
        {"date": "2024-04-01", "type": "response"},
    ]
    course = _course(1, "2024-02-15", "2024-08-15")
    assert _count_modifications_in_course(ledger, course) == 2


def test_count_modifications_in_course_with_filter() -> None:
    """modification_type_filter narrows to a single value (interruptions
    use 'dose_hold')."""
    ledger = [
        {"date": "2024-03-15", "type": "systemic_treatment_modification",
         "modification_type": "dose_hold"},
        {"date": "2024-04-15", "type": "systemic_treatment_modification",
         "modification_type": "dose_reduction"},
        {"date": "2024-05-15", "type": "systemic_treatment_modification",
         "modification_type": "dose_hold"},
    ]
    course = _course(1, "2024-02-15", "2024-08-15")
    assert _count_modifications_in_course(
        ledger, course, modification_type_filter="dose_hold"
    ) == 2


def test_count_modifications_open_course_no_end_date() -> None:
    """Open courses (end_date=None) count mods from start_date onward."""
    ledger = [
        {"date": "2024-01-15", "type": "systemic_treatment_modification",
         "modification_type": "dose_hold"},  # before
        {"date": "2024-03-15", "type": "systemic_treatment_modification",
         "modification_type": "dose_hold"},  # after start, no end bound
        {"date": "2025-12-31", "type": "systemic_treatment_modification",
         "modification_type": "dose_hold"},  # also after start
    ]
    course = _course(1, "2024-02-15", end=None)
    assert _count_modifications_in_course(ledger, course) == 2


# ---------------------------------------------------------------------------
# _derive_course_response_columns — the 5 best_response_class buckets
# ---------------------------------------------------------------------------


def _make_artifact(
    *,
    ledger: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
) -> TrajectoryArtifact:
    cell = CellKey(
        trajectory_id="t1", condition_id="c", policy_id="A_THP",
        model_id="m", model_short="m", future_index=1,
        tumor_oncotree_code="BRCA", conditioning_label="x",
        conditioning_hash="h", accepted=True,
    )
    materialized = {
        "baseline": {}, "ledger": ledger,
        "snapshots": snapshots, "treatment_history": [],
    }
    return TrajectoryArtifact(
        cell=cell, trajectory_dir=Path("/tmp"),
        parsed_ledger=(), materialized=materialized,
        trajectory_result={},
    )


def _snapshot(
    date_str: str, *, total_burden_band: str, anchor_event_id: str | None = None,
    state_id: str | None = None,
) -> dict[str, Any]:
    return {
        "date": date_str,
        "state_id": state_id or f"s_{date_str}",
        "anchor_event_id": anchor_event_id,
        "total_burden_band": total_burden_band,
    }


def test_best_response_class_pd_on_progression_in_course() -> None:
    """Progression event during course → PD, regardless of any response."""
    course = _course(1, "2024-02-15", "2024-08-15")
    art = _make_artifact(
        ledger=[
            {"event_id": "r1", "date": "2024-04-15", "type": "response"},
            {"event_id": "p1", "date": "2024-06-15", "type": "progression"},
        ],
        snapshots=[
            _snapshot("2024-02-20", total_burden_band="moderate"),
            _snapshot("2024-04-15", total_burden_band="small", anchor_event_id="r1"),
        ],
    )
    sig = _derive_course_response_columns(art, course)
    assert sig.best_response_class == "PD"


def test_best_response_class_ne_no_response_no_progression() -> None:
    course = _course(1, "2024-02-15", "2024-08-15")
    art = _make_artifact(
        ledger=[],  # no response, no progression events
        snapshots=[_snapshot("2024-02-20", total_burden_band="moderate")],
    )
    sig = _derive_course_response_columns(art, course)
    assert sig.best_response_class == "NE"
    assert sig.had_response_event is False
    assert sig.n_response_events == 0


def test_best_response_class_cr_like_terminal_absent() -> None:
    """Terminal-response-anchored band ∈ {absent, microscopic} →
    CR_like."""
    course = _course(1, "2024-02-15", "2024-08-15")
    art = _make_artifact(
        ledger=[
            {"event_id": "r1", "date": "2024-04-15", "type": "response"},
            {"event_id": "r2", "date": "2024-07-15", "type": "response"},
        ],
        snapshots=[
            _snapshot("2024-02-20", total_burden_band="bulky"),  # line-start
            _snapshot("2024-04-15", total_burden_band="moderate", anchor_event_id="r1"),
            _snapshot("2024-07-15", total_burden_band="absent", anchor_event_id="r2"),
        ],
    )
    sig = _derive_course_response_columns(art, course)
    assert sig.best_response_class == "CR_like"


def test_best_response_class_cr_like_terminal_microscopic() -> None:
    course = _course(1, "2024-02-15", "2024-08-15")
    art = _make_artifact(
        ledger=[{"event_id": "r1", "date": "2024-07-15", "type": "response"}],
        snapshots=[
            _snapshot("2024-02-20", total_burden_band="bulky"),
            _snapshot("2024-07-15", total_burden_band="microscopic", anchor_event_id="r1"),
        ],
    )
    sig = _derive_course_response_columns(art, course)
    assert sig.best_response_class == "CR_like"


def test_best_response_class_pr_like_strictly_lower_band() -> None:
    course = _course(1, "2024-02-15", "2024-08-15")
    art = _make_artifact(
        ledger=[{"event_id": "r1", "date": "2024-07-15", "type": "response"}],
        snapshots=[
            _snapshot("2024-02-20", total_burden_band="bulky"),
            _snapshot("2024-07-15", total_burden_band="small", anchor_event_id="r1"),
        ],
    )
    sig = _derive_course_response_columns(art, course)
    assert sig.best_response_class == "PR_like"


def test_best_response_class_sd_like_band_unchanged() -> None:
    course = _course(1, "2024-02-15", "2024-08-15")
    art = _make_artifact(
        ledger=[{"event_id": "r1", "date": "2024-07-15", "type": "response"}],
        snapshots=[
            _snapshot("2024-02-20", total_burden_band="moderate"),
            _snapshot("2024-07-15", total_burden_band="moderate", anchor_event_id="r1"),
        ],
    )
    sig = _derive_course_response_columns(art, course)
    assert sig.best_response_class == "SD_like"


def test_best_response_class_null_when_no_anchored_snapshot() -> None:
    """Response event exists but no snapshot anchors to it →
    __not_classifiable__ (null)."""
    course = _course(1, "2024-02-15", "2024-08-15")
    art = _make_artifact(
        ledger=[{"event_id": "r1", "date": "2024-07-15", "type": "response"}],
        snapshots=[
            _snapshot("2024-02-20", total_burden_band="moderate"),
            # No snapshot anchored to r1
        ],
    )
    sig = _derive_course_response_columns(art, course)
    assert sig.best_response_class is None


def test_best_response_class_null_when_no_line_start_snapshot() -> None:
    """No snapshot with date >= course.start_date → null."""
    course = _course(1, "2024-02-15", "2024-08-15")
    art = _make_artifact(
        ledger=[{"event_id": "r1", "date": "2024-07-15", "type": "response"}],
        snapshots=[
            # Only pre-course snapshots
            _snapshot("2024-01-15", total_burden_band="moderate"),
        ],
    )
    sig = _derive_course_response_columns(art, course)
    assert sig.best_response_class is None


def test_best_response_class_null_on_burden_up_without_progression() -> None:
    """Burden went UP without a progression event — incoherent edge
    case; classify as None rather than mis-bucket."""
    course = _course(1, "2024-02-15", "2024-08-15")
    art = _make_artifact(
        ledger=[{"event_id": "r1", "date": "2024-07-15", "type": "response"}],
        snapshots=[
            _snapshot("2024-02-20", total_burden_band="small"),
            _snapshot("2024-07-15", total_burden_band="bulky", anchor_event_id="r1"),
        ],
    )
    sig = _derive_course_response_columns(art, course)
    assert sig.best_response_class is None


def test_best_response_class_takes_latest_snapshot_for_anchor() -> None:
    """When multiple snapshots anchor to the same response event, the
    LATEST one defines terminal_band."""
    course = _course(1, "2024-02-15", "2024-08-15")
    art = _make_artifact(
        ledger=[{"event_id": "r1", "date": "2024-07-15", "type": "response"}],
        snapshots=[
            _snapshot("2024-02-20", total_burden_band="bulky"),
            _snapshot("2024-07-15", total_burden_band="small", anchor_event_id="r1"),
            _snapshot("2024-08-01", total_burden_band="absent", anchor_event_id="r1"),
        ],
    )
    sig = _derive_course_response_columns(art, course)
    # Latest anchored snapshot is "absent" → CR_like, not PR_like.
    assert sig.best_response_class == "CR_like"


# ---------------------------------------------------------------------------
# had_response_event / n_response_events / time_to_first_response
# ---------------------------------------------------------------------------


def test_had_response_event_and_count_and_time_to_first() -> None:
    course = _course(1, "2024-02-15", "2024-08-15")
    art = _make_artifact(
        ledger=[
            {"event_id": "r1", "date": "2024-04-15", "type": "response"},
            {"event_id": "r2", "date": "2024-06-15", "type": "response"},
            {"event_id": "r3", "date": "2024-07-15", "type": "response"},
        ],
        snapshots=[
            _snapshot("2024-02-20", total_burden_band="moderate"),
            _snapshot("2024-07-15", total_burden_band="small", anchor_event_id="r3"),
        ],
    )
    sig = _derive_course_response_columns(art, course)
    assert sig.had_response_event is True
    assert sig.n_response_events == 3
    # 2024-02-15 → 2024-04-15 = 60 days
    assert sig.time_to_first_response_days == 60


def test_time_to_first_response_null_when_no_response() -> None:
    course = _course(1, "2024-02-15", "2024-08-15")
    art = _make_artifact(ledger=[], snapshots=[])
    sig = _derive_course_response_columns(art, course)
    assert sig.time_to_first_response_days is None


# ---------------------------------------------------------------------------
# build_treatment_rows integration — canonical_regimen column populated
# ---------------------------------------------------------------------------


def test_treatment_rows_canonical_regimen_populated() -> None:
    """End-to-end: component_classes and canonical_regimen both populate."""
    ledger = [
        {"date": "2024-02-15", "type": "systemic_treatment_start",
         "line": 1, "regimen": {"label": "THP",
                                "components": ["trastuzumab", "pertuzumab", "docetaxel"],
                                "component_classes": [
                                    "anti_her2",
                                    "anti_her2",
                                    "cytotoxic_chemotherapy",
                                ]}},
        {"date": "2024-08-15", "type": "systemic_treatment_switch",
         "line": 2, "regimen": {"label": "T-DM1",
                                "components": ["trastuzumab", "emtansine"],
                                "component_classes": [
                                    "antibody_drug_conjugate",
                                    "antibody_drug_conjugate",
                                ]}},
        {"date": "2024-12-31", "type": "death", "cause_category": "cancer"},
    ]
    art = _make_artifact(ledger=ledger, snapshots=[])
    df = build_treatment_rows([art])
    # Two courses; model component buckets and downstream canonical helper both propagate.
    assert list(df["component_classes_joined"]) == [
        "anti_her2;anti_her2;cytotoxic_chemotherapy",
        "antibody_drug_conjugate;antibody_drug_conjugate",
    ]
    assert "regimen_class" not in df.columns
    assert "THP" in list(df["canonical_regimen"])
    assert "T-DM1" in list(df["canonical_regimen"])
