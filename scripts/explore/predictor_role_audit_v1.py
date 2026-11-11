#!/usr/bin/env python3
"""Audit every predictor's role, then rerun the ladder with the corrected boundary.

A `now_*` prefix is not evidence that a feature is current state. Three history-derived
variables were sitting in the state rung because of their names, and the fix is not to move
three columns but to give every predictor a declared role that a reader can check.

THE CLINICAL GEOMETRY IS CHAX. The query is q_tau = (C, X_tau, H_tau, A_tau). The
measurement process M_tau -- scan frequency, assessment recency, documentation intensity --
is NOT a fifth clinical concept. It describes how GENIE observes the patient, not the
patient, and it enters both models identically as a nuisance adjustment:

    snapshot model   f0(C, X_tau, A_tau | M_tau)
    history model    f1(C, X_tau, H_tau, A_tau | M_tau)

The history effect is f1 - f0. M is never a rung of the geometry; its own predictive power
is a diagnostic about the registry, not a finding about patients. M is deliberately not
called O, because O already denotes the observation operator that projects trajectories
into a dataset.

THE TEST, applied to each feature:

    Could this be known from a complete examination of the patient at the landmark,
    without inspecting anything that happened before it?

    yes, and it is unchanging          -> C  fixed context
    yes, and it describes the patient  -> X  current clinical state
    yes, and it describes the therapy  -> A  current action
    no, it needs earlier events        -> H  accumulated history
    it describes our looking, not them -> M  measurement process (nuisance, in both)

Two entanglements the audit exposes and repairs:

  * the fresh/stale status encoding mixes X with M. What the patient's disease is doing
    is state; whether anyone looked recently is measurement. They are split here into a
    pure status indicator and separate staleness measures.
  * months on the current line and months since the last line start are DETERMINISTIC
    DUPLICATES whenever the patient is on treatment, differing only off treatment. Both
    are history, and carrying both is redundant rather than informative.

Reported as two models adjusted identically for M, not as a five-rung ladder.

GOVERNANCE. GENIE-BPC under DUA. Aggregate output only, gitignored.

Run:  uv run python scripts/explore/predictor_role_audit_v1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import genie_operator_registry_v2 as GV2  # noqa: E402
from synthonccases import disease_state as DS  # noqa: E402
import line_anchored_diagnostic_v1 as LD  # noqa: E402
import trajectory_feature_poc_v1 as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GENIE_ROOT = Path.home() / "GenieBPC"
OUT = ROOT / "outputs" / "exploratory_predictor_role_audit"
H, ALPHA, SEED, FOLDS, N_PERM = 12.0, 100.0, 20260824, 5, 199
STATES = GV2.STATES

#: role, source window, and whether the value requires knowing earlier events.
REGISTRY: dict[str, tuple[str, str, bool]] = {
    "age_at_diagnosis": ("C", "diagnosis", False),
    "sex_female": ("C", "diagnosis", False),
    "histology_adeno": ("C", "diagnosis", False),
    "histology_squamous": ("C", "diagnosis", False),
    "stage_at_diagnosis_iv": ("C", "diagnosis", False),
    "stage_at_diagnosis_unknown": ("C", "diagnosis", False),

    # Metastatic extent at the LANDMARK is X, not C. It passes the snapshot test -- a
    # complete examination at the landmark would establish it -- and it is emphatically
    # not the same variable as diagnosis stage.
    "metastatic_documented_by_landmark": ("X", "landmark", False),
    "metastatic_extent_unknown": ("X", "landmark", False),
    "n_metastatic_sites_by_landmark": ("X", "landmark", False),

    "months_since_last_assessment": ("M", "prefix", True),
    "n_assessments_prefix": ("M", "prefix", True),
    "assessments_per_month": ("M", "prefix", True),
    "months_documented_prefix": ("M", "prefix", True),
    "now_never_observed": ("M", "prefix", True),
    "now_status_is_stale": ("M", "landmark", True),

    # SCAN_SITES is the region imaged, not the site of disease. These count how broadly
    # the patient was scanned, so they are measurement, not clinical state.
    "now_n_regions_imaged": ("M", "landmark", False),
    "n_regions_imaged_ever": ("M", "prefix", True),
    "n_new_regions_imaged_prefix": ("M", "prefix", True),
    "now_on_treatment": ("A", "landmark", False),
    "now_tx_chemotherapy": ("A", "landmark", False),
    "now_tx_immunotherapy": ("A", "landmark", False),
    "now_tx_targeted": ("A", "landmark", False),
    "now_tx_endocrine": ("A", "landmark", False),

    # Named now_*, but none of these survive the snapshot test.
    "now_line_number": ("H", "prefix", True),
    "now_months_on_current_line": ("H", "prefix", True),
    "months_since_last_line_start": ("H", "prefix", True),
}
# Canonical five-state vocabulary. The state itself is X -- what the patient's disease is
# doing. Whether we know it by observation or by inference is MEASUREMENT, so it sits in M
# and enters both models identically as a nuisance adjustment.
for state in DS.State.__args__:
    REGISTRY[f"now_state_{state}"] = ("X", "landmark", False)
REGISTRY["now_state_inferred"] = ("M", "landmark", False)
for name in GV2.LADDER["L3 + treatment history"] + GV2.LADDER["L4 + disease history"]:
    REGISTRY[name] = ("H", "prefix", True)


def clean_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Formerly split the fresh/stale encoding into status plus staleness.

    The canonical operator already emits a pure state and a separate evidence flag, so
    there is nothing left to disentangle. Kept as an identity so the call sites do not
    have to change, and so this file records WHY the split disappeared rather than
    leaving a reader to wonder where it went.
    """
    return X.copy()


def roles_of(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"C": [], "M": [], "X": [], "A": [], "H": [],
                                    "UNCLASSIFIED": []}
    for c in columns:
        groups[REGISTRY.get(c, ("UNCLASSIFIED",))[0]].append(c)
    return groups


def duplicate_pairs(X: pd.DataFrame, columns: list[str]) -> list[tuple[str, str, float]]:
    out = []
    values = {c: X[c].fillna(-999.0).to_numpy(float) for c in columns}
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            same = float(np.mean(np.isclose(values[a], values[b])))
            if same > 0.90:
                out.append((a, b, same))
    return out


def ladder_scores(X: pd.DataFrame, Y: pd.DataFrame, groups: np.ndarray,
                  rungs: list[tuple[str, list[str]]]) -> dict:
    ymat = Y.to_numpy(float)
    ok = np.isfinite(ymat).all(axis=1)
    sd = np.where(ymat[ok].std(axis=0) > 1e-9, ymat[ok].std(axis=0), 1.0)
    ystd = (ymat[ok] - ymat[ok].mean(axis=0)) / sd
    g = groups[ok]
    folds = M.group_folds(g, min(FOLDS, len(set(g))), SEED)
    blocks = [[GV2.OUTCOMES.index(o) for o in names]
              for names in GV2.OUTCOME_BLOCKS.values()]
    alphas = [ALPHA] * ystd.shape[1]
    scores, cumulative = {}, []
    for name, cols in rungs:
        cumulative = cumulative + [c for c in cols if c in X.columns]
        raw = np.nan_to_num(X.loc[ok, cumulative].to_numpy(float))
        scores[name] = M.block_joint_score(raw, ystd, blocks, folds, g, SEED, alphas)
    return {"scores": scores, "n": int(ok.sum())}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"registry": {k: {"role": v[0], "window": v[1],
                                     "requires_prior_events": v[2]}
                                 for k, v in REGISTRY.items()}}
    print("PREDICTOR ROLE AUDIT\n")

    genie = GENIE_ROOT / "NSCLC_Genie"
    patients = GV2.load(genie)

    for landmark_kind in ("calendar 12 months", "line-anchored"):
        if landmark_kind.startswith("calendar"):
            records = [{"patient": p, "tau": 12.0} for p in patients]
        else:
            records = LD.assign_landmarks(patients, SEED)
        keep = [r for r in records
                if r["patient"].death > r["tau"]
                and (r["patient"].last_contact >= r["tau"] + H
                     or r["patient"].death <= r["tau"] + H)
                and any(w <= r["tau"] for w, _ in r["patient"].status)
                and (GV2.observed_span(r["patient"], r["tau"], r["tau"] + H) > 0
                     or r["patient"].death <= r["tau"] + H)]
        X = clean_frame(pd.DataFrame([GV2.prefix_features(r["patient"], r["tau"])
                                      for r in keep]))
        Y = pd.DataFrame([GV2.future_outcomes(r["patient"], r["tau"], H)
                          for r in keep])[GV2.OUTCOMES]
        groups = np.array([r["patient"].pid for r in keep])
        groups_by_role = roles_of(list(X.columns))

        print(f"=== {landmark_kind}: n={len(keep)}")
        print("    role counts: " + ", ".join(
            f"{role} {len(cols)}" for role, cols in groups_by_role.items() if cols))
        if groups_by_role["UNCLASSIFIED"]:
            print(f"    UNCLASSIFIED: {groups_by_role['UNCLASSIFIED']}")

        dups = duplicate_pairs(X, groups_by_role["H"] + groups_by_role["X"]
                               + groups_by_role["A"])
        for a, b, share in dups:
            print(f"    duplicate: {a} == {b} on {share:.0%} of patients")

        moved = ["now_line_number", "now_months_on_current_line",
                 "months_since_last_line_start"]
        # Two models, both adjusted identically for M. C alone and C+M are reported
        # only as diagnostics -- they are not rungs of the clinical geometry.
        old = [("C", groups_by_role["C"]),
               ("C+M", groups_by_role["M"]),
               ("+S+A (old boundary)", groups_by_role["X"] + groups_by_role["A"] + moved),
               ("+H (old boundary)", [c for c in groups_by_role["H"] if c not in moved])]
        new = [("C", groups_by_role["C"]),
               ("C+M (diagnostic)", groups_by_role["M"]),
               ("f0 snapshot: C+X+A | M", groups_by_role["X"] + groups_by_role["A"]),
               ("f1 history: C+X+H+A | M", groups_by_role["H"])]
        before, after = (ladder_scores(X, Y, groups, old),
                         ladder_scores(X, Y, groups, new))
        b_gain = (before["scores"]["+H (old boundary)"]
                  - before["scores"]["+S+A (old boundary)"])
        a_gain = (after["scores"]["f1 history: C+X+H+A | M"]
                  - after["scores"]["f0 snapshot: C+X+A | M"])
        print(f"    old boundary: state {before['scores']['+S+A (old boundary)']:.3f} -> "
              f"history {before['scores']['+H (old boundary)']:.3f}   gain {b_gain:+.4f}")
        print(f"    corrected   : f0 {after['scores']['f0 snapshot: C+X+A | M']:.3f}"
              f" -> f1 {after['scores']['f1 history: C+X+H+A | M']:.3f}   history effect "
              f"{a_gain:+.4f}")
        print(f"    diagnostics only: C {after['scores']['C']:.3f}, "
              f"C+M {after['scores']['C+M (diagnostic)']:.3f}\n")
        report[landmark_kind] = {"n": len(keep), "roles": {k: len(v) for k, v
                                                           in groups_by_role.items()},
                                 "duplicates": [[a, b, s] for a, b, s in dups],
                                 "old_boundary": before["scores"],
                                 "corrected": after["scores"],
                                 "old_history_gain": b_gain,
                                 "corrected_history_gain": a_gain}

    (OUT / "readout.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
