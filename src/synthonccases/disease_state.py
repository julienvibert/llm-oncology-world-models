"""The canonical disease-state operator, and the clinical truth table it implements.

At least three different vocabularies for this one concept are in use across the analysis
scripts -- {responding, stable, progressing}, {responding, progressing, neither}, and
{responding, stable, mixed, progressing} -- plus seven- and eight-state variants elsewhere.
None of them has a NO ACTIVE DISEASE state, so a patient in postoperative remission has
never had anywhere to be. Results derived under different vocabularies were never
comparable, and rewriting one function would have left the rest.

THE RULE, in one line each:

    burden       says WHERE the patient is
    transitions  say HOW it is changing
    history      says HOW THEY ARRIVED
    evidence     says HOW CONFIDENTLY the record lets us know

FIVE STATES, and they apply to localized and metastatic disease alike:

    no_active_disease   no measurable burden at any site. Reached by resection, by
                        complete response to systemic therapy, or by both. The path is
                        recorded in history, not in the state.
    responding          active disease, shrinking
    stable              active disease, neither shrinking nor growing
    progressing         active disease growing, or new disease appearing
    unknown             cannot be established from the record

TWO OPERATORS, one vocabulary. The generated operator reads materialized burden, which is
complete. The GENIE operator reads an incomplete projection and must say how it knows --
GENIE records no surgery and no radiotherapy, so it can never witness the path into
no_active_disease and must sometimes infer the state from sustained negative follow-up.
That inference is labelled, never silent, and a strict mode masks it entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

State = Literal["no_active_disease", "responding", "stable", "progressing", "unknown"]
Evidence = Literal["directly_observed", "clinically_inferred", "unknown"]

#: Burden bands that mean disease is present and measurable. "microscopic" counts as
#: active: minimal residual disease is not remission, and treating it as no_active_disease
#: would silently merge two clinically distinct situations.
ACTIVE_BANDS = {"microscopic", "small", "moderate", "bulky", "diffuse_or_leukemic"}

#: Worst-wins ordering. A patient with one shrinking site and one growing site is managed
#: as progressing, because the growing site drives the decision. The raw per-site kinetics
#: are kept alongside so "mixed" is never lost, only resolved.
SEVERITY: dict[State, int] = {"no_active_disease": 0, "responding": 1, "stable": 2,
                              "progressing": 3, "unknown": -1}


@dataclass(frozen=True)
class DiseaseState:
    """A state, how we know it, and why. The basis is not decoration: an inferred
    remission and an observed one occupy the same coordinate and must stay tellable
    apart in any sensitivity analysis."""

    state: State
    evidence: Evidence
    basis: str
    mixed: bool = False          # site kinetics disagreed and were resolved worst-wins

    def strict(self) -> DiseaseState:
        """The strict view: inferred states become unknown.

        Inferred remission will be the MAJORITY state for localized disease, so the primary
        analysis rests on it. If the strict and primary analyses disagree, that disagreement
        is the finding, and it can only be seen if the inference is separable.
        """
        if self.evidence == "clinically_inferred":
            return DiseaseState("unknown", "unknown",
                                f"masked in strict mode ({self.basis})", self.mixed)
        return self


def defined_window(start: float, horizon: float, death: float,
                   last_contact: float) -> tuple[float, float, str]:
    """The interval over which disease state is DEFINED at all.

    Death is vital status, not a disease state, and the difference matters for occupancy.
    After death there is no disease state to be unknown -- the question stops applying.
    Letting a deceased patient accrue "unknown" months would then invite that time to be
    masked, imputed, or counted as unobserved follow-up, all of which are wrong in the same
    direction: they treat a completed course as an incomplete one.

    Loss to follow-up is the opposite case. There the question still applies and we simply
    cannot answer it, so the interval ends and the remainder is genuinely unknown.
    """
    end = start + horizon
    if death <= end:
        return start, max(start, death), "closed by death"
    if last_contact < end:
        return start, max(start, last_contact), "closed by loss to follow-up"
    return start, end, "complete"


def from_burden(bands: list[str], kinetics: list[str]) -> DiseaseState:
    """The generated-trajectory operator. Materialized burden is complete, so every state
    here is directly observed.

    Burden decides whether disease is ACTIVE; kinetics decide how it is CHANGING. A
    treatment change alters neither and therefore cannot alter the state -- which is the
    property that keeps a planned chemotherapy-to-endocrine handover from reading as
    progression.
    """
    active = [(b, k) for b, k in zip(bands, kinetics, strict=True) if b in ACTIVE_BANDS]
    if not active:
        return DiseaseState("no_active_disease", "directly_observed",
                            "no measurable burden at any site")

    moves = {k for _b, k in active}
    growing = bool(moves & {"appearing", "growing"})
    shrinking = bool(moves & {"shrinking", "cleared"})
    mixed = growing and shrinking

    if growing:
        # Worst-wins, including the mixed case.
        return DiseaseState("progressing", "directly_observed",
                            "at least one site appearing or growing", mixed)
    if shrinking:
        return DiseaseState("responding", "directly_observed",
                            "active disease, all changing sites shrinking", mixed)
    return DiseaseState("stable", "directly_observed",
                        "active disease, no site changing", mixed)


#: GENIE's curated vocabulary -> canonical. GENIE has no no_active_disease category, which
#: is exactly the gap this table cannot fill: a curated "responding" on a patient with no
#: measurable disease is not remission, it is a label applied to something else.
FROM_GENIE_STATUS: dict[str, State] = {
    "responding": "responding",
    "stable": "stable",
    "progressing": "progressing",
    "mixed": "progressing",          # worst-wins; the mixed flag is retained
}


def from_genie(
    *,
    presence: str | None,
    curated_status: str | None,
    new_distant_site: bool = False,
) -> DiseaseState:
    """The real-data operator. Two questions, answered separately, nothing inferred.

    GENIE answers TWO things at each visit and we had only read the second:

        PRESENCE   md_ca -- is there cancer? evidence / no evidence / equivocal
        DYNAMICS   what is it doing? responding / stable / mixed / progressing

    Both arguments are already RECENCY-FILTERED by the caller: a stale assessment is
    passed as None, never carried forward.

    THERE IS NO INFERENCE HERE ANY MORE. The previous version read a patient's subsequent
    follow-up to decide their state AT the landmark -- "because they stayed relapse-free
    for twelve months, they were in remission when they started" -- which rewrites the
    prefix using the answer. It also gated that inference on a stub that meant only "three
    or fewer regimens so far". Absence of evidence is now simply unknown.

    CONFLICTS ARE NAMED, NOT SILENTLY RESOLVED. An earlier version claimed to work
    worst-first and then tested "no evidence" before the curated status, so a patient
    recorded as having no evidence of cancer AND a progressing status came out as
    remission -- the two most opposite readings resolved in favour of the reassuring one.
    Genuinely discordant evidence now yields an explicit conflict, because a case where the
    oncologist and the response record disagree is one where we do not know.
    """
    active = curated_status in ("responding", "stable", "progressing", "mixed")
    if presence == "no evidence" and (active or new_distant_site):
        detail = ("a distant site was newly documented" if new_distant_site
                  else f"the curated status is {curated_status}")
        return DiseaseState("unknown", "unknown",
                            f"CONFLICT: oncologist records no evidence of cancer while "
                            f"{detail}")

    if new_distant_site:
        return DiseaseState("progressing", "directly_observed",
                            "a distant site was newly documented")

    if presence == "no evidence":
        # The ONLY way to reach this state in GENIE. The response vocabulary has no
        # complete-response category, so without md_ca a patient in remission is invisible.
        return DiseaseState("no_active_disease", "directly_observed",
                            "oncologist records no evidence of cancer")

    if curated_status:
        state = FROM_GENIE_STATUS.get(curated_status, "unknown")
        mixed = curated_status == "mixed"
        return DiseaseState(state, "directly_observed",
                            f"curated assessment: {curated_status}", mixed)

    if presence == "evidence":
        # Cancer is present but nobody recorded what it is doing. This is NOT the same as
        # not knowing whether there is cancer, and reporting both as "unknown" concealed
        # that the old inference had labelled documented active disease as remission. The
        # presence field still reaches the prefix, so the model is told.
        return DiseaseState("unknown", "unknown",
                            "cancer present, but its trajectory is not assessed")
    if presence == "equivocal":
        return DiseaseState("unknown", "unknown", "assessment equivocal")
    return DiseaseState("unknown", "unknown", "no recent assessment on record")
