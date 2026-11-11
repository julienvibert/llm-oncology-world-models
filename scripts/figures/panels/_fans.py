"""Shared drawing for the four-arm case fans (panels f5a_case_fans, si08a_lung_fans, si09a_colorectal_fans).

Four arms side by side, every accepted future, the observed trajectory repeated above each column,
disease state as a bar and systemic treatment classes as thin strips beneath. The data come from
the four-arm readers, read-only on the frozen run, in the pattern of
`scripts/score_four_arm_final_v1.py` (`traced`, `traced_scores`):

    observed course   `four_arm_eligibility_v2.real_course` under `E.convention("primary")`
                      (disease path from the registry, treatment from
                      `four_arm_readers_v2.load_real_agents`: per-agent exposure segments)
    generated future  `four_arm_eligibility_v2.generated_course` (cumulative disease truth,
                      per-component exposure), drawn as the ASSERTED course, unmasked:
                      not re-observed on the registry assessment calendar
    treatment strips  documented exposure segments of each agent / component, classed by
                      `four_arm_readers_v2.classify` (one versioned drug table); an agent of
                      unrecognised class is drawn as "other", adjuncts are not drawn

Case rules and the saved forecast order (repeat index) are fixed.
"""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import INK, LIGHT, MODALITY_COLOURS, STATE_COLOURS, fk, save_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "scripts", ROOT / "scripts" / "explore", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import four_arm_eligibility_v2 as E  # noqa: E402
import four_arm_readers_v2 as R  # noqa: E402
import genie_operator_registry_v2 as GV2  # noqa: E402

from synthonccases.oncollm.parse_utils import strip_code_fence  # noqa: E402

HORIZON = 24.0
ARMS = ("patient_only", "setting", "generic_cohort", "patients_like_me")
RUN = ROOT / "runs" / "four_arm_v1_definitive_p40b20_20260826" / "attempts.jsonl"
QUERIES = ROOT / "outputs" / "exploratory_query_universe" / "queries.jsonl"
PROMPTS = ROOT / "outputs" / "exploratory_four_arm_prompts" / "quartets.jsonl"
STATE_KEY = {
    "no_active_disease": "no evidence of disease",
    "responding": "response",
    "stable": "stable",
    "progressing": "progression",
    "unknown": "unresolved",
}
STATE_LABEL = {
    "no evidence of disease": "No evidence of disease",
    "response": "Response",
    "stable": "Stable",
    "progression": "Progression",
    "unresolved": "Unresolved",
}
ARM_NAME = {
    "patient_only": "Patient record",
    "setting": "+ care setting",
    "generic_cohort": "+ generic examples",
    "patients_like_me": "+ matched examples",
}
CANCER = {"BRCA": "breast cancer", "NSCLC": "non-small-cell lung cancer", "CRC": "colorectal cancer"}
SOURCES = [
    "runs/four_arm_v1_definitive_p40b20_20260826/attempts.jsonl",
    "outputs/exploratory_query_universe/queries.jsonl",
    "outputs/exploratory_four_arm_prompts/quartets.jsonl",
    "GENIE-BPC (regimen_cancer_level_dataset.csv, data_timeline_treatment.txt; under DUA)",
    "scripts/four_arm_eligibility_v2.py (real_course, generated_course)",
    "scripts/four_arm_readers_v2.py (load_real_agents, classify)",
]
#: Exposure whose duration the source does not establish (real: no documented stop or last
#: administration, investigational duration redacted; generated: ambiguous modification).
UNKNOWN_DURATION = "unknown_duration"
TX_COLOURS = {**MODALITY_COLOURS, UNKNOWN_DURATION: "#AEB4BA"}
TX_LABEL = {"chemotherapy": "Chemotherapy", "endocrine": "Endocrine", "targeted": "Targeted",
            "immunotherapy": "Immunotherapy", "other": "Other or unclassified", UNKNOWN_DURATION: "Duration not documented"}
#: Prespecified strata of the display rule.
STRATA = {"NSCLC": "regimen 2", "BRCA": "regimen 1", "CRC": "regimen 3+"}


def _digest(query_id: str) -> str:
    """Short non-identifying tag of a landmark for handovers (never the query id itself)."""
    return hashlib.sha256(("fans-display-tag:" + query_id).encode()).hexdigest()[:10]


def _jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(x) for x in handle if x.strip()]


_COHORTS: dict[str, tuple[dict, dict]] = {}


def _cohort(cancer: str) -> tuple[dict, dict]:
    """GENIE patients and their curated agents (v2 reader), cached per cancer."""
    if cancer not in _COHORTS:
        genie = GV2.GENIE_ROOT / GV2.COHORTS[cancer]
        _COHORTS[cancer] = ({p.pid: p for p in GV2.load(genie)}, R.load_real_agents(genie)[0])
    return _COHORTS[cancer]


def _real(q: dict) -> E.Course:
    patients, agents = _cohort(q["cancer"])
    p = patients[q["patient"]]
    with E.convention("primary"):
        return E.real_course(p, float(q["tau_months"]), HORIZON, agents.get(q["patient"], []))


def _treatment_spans(tx: R.Treatment, end: float) -> list[tuple]:
    """Exposure per agent, unioned into (start, stop, classes) spans.

    Documented segments carry the agent's drug classes (unrecognised class: "other");
    unresolved intervals (exposure of unknown duration) carry UNKNOWN_DURATION. Adjuncts are
    not drawn (recognised, no modality)."""
    spans = []
    for ag in tx.agents:
        if ag.cls.kind == R.ADJUNCT:
            continue
        mods = set(ag.cls.modalities) if ag.cls.kind == "modality" else {"other"}
        for pieces, m in ((ag.segments, mods), (ag.unresolved, {UNKNOWN_DURATION})):
            for a, b in pieces:
                a, b = max(0.0, a), min(end, b)
                if b > a + 1e-9:
                    spans.append((a, b, m))
    edges = sorted({v for a, b, _ in spans for v in (a, b)})
    out: list[tuple] = []
    for a, b in zip(edges, edges[1:]):
        active = [m for lo, hi, m in spans if lo < b - 1e-9 and hi > a + 1e-9]
        if active:
            mods = tuple(sorted(set().union(*active)))
            if out and out[-1][2] == mods and abs(out[-1][1] - a) < 1e-9:
                out[-1] = (out[-1][0], b, mods)
            else:
                out.append((a, b, mods))
    return out


def course_dict(c: E.Course) -> dict:
    """The drawing dict of one course: disease path, end, closing reason, treatment strips."""
    died = c.death <= HORIZON + 1e-9
    end = min(c.death, c.follow_end, HORIZON)
    reason = "closed by death" if died else ("censored" if end < HORIZON - 1e-9 else "complete")
    path = [(a, min(b, end), s) for a, b, s in c.path if min(b, end) > a]
    return {"path": path, "end": end, "reason": reason, "treatment": _treatment_spans(c.treatment, end)}


def select_eventful_query(min_follow_up: float = 18.0, cancer: str | None = None) -> tuple[str, dict]:
    """Declared case rule for Figure 5 (reads the observed course, no generated output).

    Among landmarks with 10 accepted futures in every one of the four arms, observed follow-up of
    at least `min_follow_up` months after the landmark (alive and followed, or death after that
    time), and at least two changes of documented disease state in the 24-month window, at least
    one of them a progression: the lowest SHA-256 of "meeting-display-v2:" + query_id.
    With `cancer` given, only landmarks of that cancer are considered (si08a_lung_fans, si09a_colorectal_fans).
    The observed course is read by the corrected reader (`E.real_course`, primary convention).
    """
    counts: Counter = Counter()
    for row in _jsonl(RUN):
        if row.get("accepted"):
            counts[(row["query_id"], row["arm"])] += 1
    complete = {q for q, _ in counts if all(counts[(q, a)] == 10 for a in ARMS)}
    queries = [q for q in _jsonl(QUERIES) if q["query_id"] in complete and (cancer is None or q["cancer"] == cancer)]
    queries.sort(key=lambda q: hashlib.sha256(("meeting-display-v2:" + q["query_id"]).encode()).hexdigest())
    n_checked = 0
    for q in queries:
        real = course_dict(_real(q))
        n_checked += 1
        if real["end"] < min_follow_up:
            continue
        segs = [(max(0.0, a), min(sp, real["end"]), st) for a, sp, st in merged(real["path"])]
        segs = [x for x in segs if x[1] > x[0]]
        if len(segs) - 1 >= 2 and any(st == "progressing" for _, _, st in segs[1:]):
            return q["query_id"], {"complete_landmarks": len(complete), "checked_in_hash_order": n_checked,
                                   "min_follow_up": min_follow_up, "tag": _digest(q["query_id"])}
    raise ValueError("no landmark satisfies the case rule")


def stratum_query(cancer: str) -> str:
    """Prespecified display rule: lowest SHA-256 of "meeting-display-v2:" + query_id
    in the prespecified cancer/depth stratum; reads no outcome."""
    cands = [q for q in _jsonl(QUERIES) if q["cancer"] == cancer and q["depth"] == STRATA[cancer]]
    return min(cands, key=lambda q: hashlib.sha256(("meeting-display-v2:" + q["query_id"]).encode()).hexdigest())[
        "query_id"]


def load_query(query_id: str) -> dict:
    """One landmark: the observed course and every accepted future per arm, in saved order."""
    q = next(x for x in _jsonl(QUERIES) if x["query_id"] == query_id)
    qt = next(x for x in _jsonl(PROMPTS) if x["query_id"] == query_id)
    ledgers: dict[str, list] = defaultdict(list)
    for row in _jsonl(RUN):
        if row["query_id"] == query_id and row.get("accepted"):
            ledgers[row["arm"]].append(row)
    case = {"query_id": query_id, "tag": _digest(query_id), "cancer": q["cancer"], "depth": q["depth"],
            "landmark_state": qt.get("landmark_state", "unknown"), "index_regimen": qt["index_regimen"],
            "real": course_dict(_real(q)), "arms": {}, "not_drawable": 0}
    for arm in ARMS:
        kw = {"cutoff": qt["arms"][arm]["conditioning_payload"]["fixed_events"]["cutoff_date"],
              "landmark_state": qt["landmark_state"], "landmark_presence": qt["landmark_presence"],
              "landmark_sites": qt["landmark_sites"], "landmark_extent": qt["landmark_extent"],
              "index_regimen": qt["index_regimen"]}
        rows = sorted(ledgers[arm], key=lambda r: int(r["repeat_index"]))
        if not rows:
            raise ValueError("a displayed landmark has no accepted futures in one arm")
        fans = []
        for row in rows:
            ledger = json.loads(strip_code_fence(row["raw_response"]))["ledger"]
            try:
                c = E.generated_course(ledger, HORIZON, **kw)
            except R.NotScoreable:
                case["not_drawable"] += 1
                continue
            fans.append({"repeat": int(row["repeat_index"]), "terminal": c.terminal, **course_dict(c)})
        case["arms"][arm] = fans
    return case


def load_case(cancer: str) -> dict:
    """The display case of one cancer (prespecified stratum, hash rule)."""
    return load_query(stratum_query(cancer))


def merged(path: list) -> list:
    """Join consecutive intervals with the same state (avoids hairline seams in print)."""
    out: list = []
    for start, stop, state in path:
        if out and out[-1][2] == state and abs(out[-1][1] - start) < 1e-9:
            out[-1] = (out[-1][0], stop, state)
        else:
            out.append((start, stop, state))
    return out


def draw_course(ax, course: dict, y: float) -> None:
    for start, stop, state in merged(course["path"]):
        lo, hi = max(0, start), min(HORIZON, course["end"], stop)
        if hi > lo:
            ax.broken_barh(
                [(lo, hi - lo)], (y - 0.08, 0.40), facecolor=STATE_COLOURS[STATE_KEY[state]], edgecolor="none"
            )
    for start, stop, mods in course["treatment"]:
        h = 0.24 / len(mods)
        for k, mod in enumerate(mods):
            ax.broken_barh(
                [(start, stop - start)],
                (y - 0.36 + k * h, h),
                facecolor=TX_COLOURS.get(mod, TX_COLOURS["other"]),
                edgecolor="none",
            )
    if course["reason"] == "closed by death":
        ax.plot(course["end"], y + 0.12, marker="x", color=INK, ms=3.0, mew=0.75, clip_on=False)
    elif course["reason"] == "censored" and course["end"] < HORIZON:
        ax.plot(course["end"], y + 0.12, marker="|", color=INK, ms=4.5, mew=0.75, clip_on=False)


def draw_case(case: dict):
    n_rows = max(len(case["arms"][a]) for a in ARMS)
    top = n_rows + 1.0
    h_mm = 25 + 3.9 * n_rows  # rows plus title, axis and two-row key
    fig = fk.figure(width=180, height_mm=h_mm)
    fig.set_layout_engine(None)
    gs = fig.add_gridspec(1, 4, wspace=0.05, left=0.075, right=0.995, top=1 - 4.5 / h_mm, bottom=17.5 / h_mm)
    for j, arm in enumerate(ARMS):
        ax = fig.add_subplot(gs[j])
        futures = case["arms"][arm]
        ax.set_xlim(-0.25, 24.25)
        ax.set_ylim(-0.6, top + 0.5)
        draw_course(ax, case["real"], top)
        ax.axhline(top - 0.62, color=LIGHT, lw=0.5)
        ax.axvline(case["real"]["end"], color=LIGHT, lw=0.5, ls=(0, (2, 2)))
        for k, course in enumerate(futures):
            draw_course(ax, course, top - 1.4 - k)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Months after regimen start", fontsize=6.5, labelpad=1.5)
        ax.set_title(f"{ARM_NAME[arm]} ({len(futures)} futures)", fontsize=7, loc="left", pad=3)
        if j == 0:
            ax.text(-1.0, top + 0.1, "Observed", fontsize=6.5, ha="right", va="center")
            ax.text(-1.0, top - 1.4 - (n_rows - 1) / 2, "Generated\nfutures", fontsize=6.5, ha="right", va="center")
    state_h = [Patch(color=STATE_COLOURS[k], label=v) for k, v in STATE_LABEL.items()]
    state_h += [
        Line2D([], [], marker="x", ls="none", color=INK, ms=3.0, mew=0.75, label="Death"),
        Line2D([], [], marker="|", ls="none", color=INK, ms=4.5, mew=0.75, label="Last follow-up"),
    ]
    treat_h = [Patch(color=c, label=TX_LABEL[k]) for k, c in TX_COLOURS.items()]
    kw = dict(
        fontsize=6,
        handlelength=1.0,
        handleheight=0.7,
        columnspacing=1.2,
        frameon=False,
        handletextpad=0.4,
        borderaxespad=0,
    )
    for y_mm, title, hs in ((4.6, "Disease state", state_h), (1.0, "Treatment class", treat_h)):
        fig.text(0.075, y_mm / h_mm, title, fontsize=6.5, va="bottom")
        fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.175, (y_mm - 0.35) / h_mm), ncol=len(hs), **kw)
    return fig


SELECTION_FIXED = (
    "the case was chosen by a fixed rule that reads no outcome (lowest SHA-256 hash of the query "
    "identifier in a prespecified stratum)"
)


def legend_text(case: dict, lead: str, tail: str = "", selection: str = SELECTION_FIXED) -> str:
    return (
        f"{lead} {CANCER[case['cancer']].capitalize()}, GENIE-BPC, at the start of a recorded "
        f"regimen ({case['depth']}; {case['index_regimen'].replace('_', ' ')}); {selection}. Each column is one arm of the four-arm "
        "experiment (DeepSeek V4 Flash): patient record only; + care setting; + generic "
        "examples; + matched examples. The observed trajectory is repeated at the top of each "
        "column; below it, every accepted generated future, one row each, in saved order. Thick "
        "bars, disease state: for the observed patient, the state documented in the registry "
        "(last recorded state carried forward, the primary reading); for generated futures, the "
        "course asserted by the model, each event applied in order to one cumulative disease "
        "picture, drawn unmasked (not re-observed on the registry assessment calendar used for "
        "scoring); unresolved, no state documented or asserted. Thin strips, systemic treatment "
        "class of each agent's exposure (observed: curated agent start and stop or last "
        "administration dates; generated: each regimen component from its start to its switch, "
        "stop, discontinuation or hold of eight weeks or more), classes from one drug table, "
        "stacked when given together; grey strip, exposure whose duration is not documented "
        "(no stop or last administration date, investigational agent with redacted duration, or "
        "an ambiguous generated modification); adjuncts not drawn. ×, death; |, last "
        "follow-up before 24 months; dotted line, end of the observed window. Blank time after "
        "death is not missing follow-up. Rows are not paired across arms. The display is an "
        "illustration, not a "
        f"comparative estimate.{(' ' + tail) if tail else ''}"
    )


def build(cancer: str, panel_id: str, lead: str, tail: str = "", confirm=(), case: dict | None = None,
          selection: str = SELECTION_FIXED) -> None:
    case = case or load_case(cancer)
    fig = draw_case(case)
    save_panel(
        fig,
        panel_id,
        legend=legend_text(case, lead, tail, selection),
        sources=SOURCES,
        message=f"For one observed patient with {CANCER[cancer]}, the four arms generate "
        "visibly different trajectory distributions.",
        confirm=[
            "One de-identified GENIE-BPC case is displayed (months from regimen "
            "start only).",
            *confirm,
        ],
    )


R0 = -5.0  # empty centre of the dials so that inner months stay readable


def _spoke(ax, course: dict, theta: float, width: float) -> None:
    import numpy as np  # noqa: F401
    for start, stop, state in merged(course["path"]):
        lo, hi = max(0, start), min(HORIZON, course["end"], stop)
        if hi > lo:
            ax.bar(theta, hi - lo, width=width, bottom=lo, color=STATE_COLOURS[STATE_KEY[state]],
                   edgecolor="none", linewidth=0)
    if course["reason"] == "closed by death":
        ax.plot(theta, course["end"] + 1.3, marker="x", color=INK, ms=3.0, mew=0.75, clip_on=False)


def draw_radial(case: dict, height_mm: float = 42.0):
    """Dials for one case (same drawing as f5b_radial_fans, which keeps its own copy): the
    observed course and one dial per arm, one spoke per future, radius = months."""
    import numpy as np

    fig = fk.figure(width=180, height_mm=height_mm)
    fig.set_layout_engine(None)
    titles = ["Observed"] + [ARM_NAME[a] for a in ARMS]
    courses = [[case["real"]]] + [case["arms"][a] for a in ARMS]
    w = 36.0 / 180
    for j, (t, cs) in enumerate(zip(titles, courses)):
        ax = fig.add_axes([j * w + 0.002, 0.17, w - 0.004, 0.70], projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(R0, HORIZON + 2)
        width = np.deg2rad(23)
        for k, c in enumerate(cs):
            _spoke(ax, c, np.deg2rad(15 + 30 * k), width)
        ax.set_xticks([])
        ax.set_yticks([0, 6, 12, 18, 24])
        ax.set_yticklabels([])
        ax.grid(color=LIGHT, lw=0.4)
        ax.spines["polar"].set_visible(False)
        ax.set_rlabel_position(0)
        ax.set_title(t, fontsize=7, pad=-2)
        gap = np.deg2rad(-30)
        for r in (6, 12, 18, 24):
            ax.text(gap, r, str(r), fontsize=6, ha="center", va="center", color=INK)
        if j == 0:
            ax.text(gap, 29, "months", fontsize=6, ha="center", va="center", color=INK)
    hs = [Patch(color=STATE_COLOURS[k], label=v) for k, v in STATE_LABEL.items()]
    hs.append(Line2D([], [], marker="x", ls="none", color=INK, ms=3.0, mew=0.75, label="Death"))
    fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.005, 0.0), ncol=6, fontsize=6, handlelength=1.0,
               handleheight=0.7, columnspacing=1.2, frameon=False, handletextpad=0.4, borderaxespad=0)
    return fig
