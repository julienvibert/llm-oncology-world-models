"""What "patients like me" means, as data rather than as code.

Every open question about anchoring -- which patients are eligible, how similarity is
measured, how many are retrieved, how many are shown, when to refuse, and what the model is
told about each one -- is a field on :class:`AnchorPolicy`. Nothing about those choices is
hard-coded anywhere else in this package, so a new answer is a new policy object and never a
new code path.

The policy is hashed into every result. Two runs that used different anchoring cannot be
mistaken for one experiment, and a result cannot be re-attributed to a policy it was not
produced under.

This package is ADDITIVE to the frozen generation contract. It builds a block of context;
it never edits a prompt template, a schema, an enum, or the conditioning spec. The frozen
pipeline continues to produce byte-identical prompts when no anchor block is supplied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from typing import Literal

Metric = Literal["euclidean", "supervised", "archetype"]
Selection = Literal["nearest", "stratified", "diverse"]
SupportLabel = Literal["direct", "transport", "aggregate", "unsupported", "sham"]

#: Rungs of the evidence ladder, best first. `aggregate` has no retrieval implementation
#: yet; a query that would land there is reported as `unsupported` until one exists, so the
#: label can never overstate what was actually retrieved.
LADDER: tuple[SupportLabel, ...] = ("direct", "transport", "aggregate", "unsupported")


@dataclass(frozen=True, slots=True)
class AnchorPolicy:
    """A complete, hashable statement of how anchors are chosen and described.

    Defaults encode what has been measured rather than what is convenient:

    * ``k = 40`` -- neighbourhoods below about forty score WORSE than predicting the base
      rate (k=1 skill -0.730, k=5 -0.022); skill peaks at 40-80. A small anchor set is not
      a weaker version of a large one, it is actively misleading.
    * ``metric = "supervised"`` -- a distance learned from outcomes beat plain standardised
      Euclidean (0.116 vs 0.099). Equal weight on institution and on last-scan status is not
      a clinical notion of similarity.
    * ``min_support = 10`` -- at cluster level, 85% of archetypes carry two treatment arms
      with at least ten members each; the radius rule reached 7%.
    """

    # --- who is eligible to be an anchor -------------------------------------------------
    same_action: bool = True
    """Restrict anchors to the query's own action. False for counterfactual retrieval."""
    target_action: str | None = None
    """Retrieve anchors on THIS action instead of the query's. Sets up a contrast."""
    match_exact: tuple[str, ...] = ()
    """Fields an anchor must match exactly: the STRICT SET.

    The patients a clinician would accept without argument. Not a fixed size -- three or two
    hundred -- and the size is itself information about how well-trodden the decision is.

    Hard match FIRST, distance SECOND. Same tumour, stage, line and regimen can still
    contain very different patients, so eligibility never replaces ranking: within the
    strict set the nearest are chosen, and the whole strict set is summarised separately."""
    tiers: tuple[tuple[str, tuple[str, ...]], ...] = ()
    """Ordered fallback rules, each (label, exact-match fields), tried when the strict set
    cannot fill the display.

    Relaxation is BY RULE and LABELLED, never a silent nearest-neighbour top-up. "Same
    cancer and regimen but a different line" is a clinically different claim from "same
    everything", and a model shown ten rows without being told which is which cannot weigh
    them differently. Empty means no relaxation: a short strict set stays short."""

    exclude_same_subject: bool = True
    """A patient is never their own anchor. Their other lines are excluded too."""

    # --- how similarity is measured ------------------------------------------------------
    metric: Metric = "supervised"
    channels: tuple[str, ...] = ()
    """Channels entering the distance. Empty means every numeric channel in the corpus."""

    # --- how many ------------------------------------------------------------------------
    k: int = 40
    """Cap on the strict set. Retrieval takes at most this many of the patients that match
    exactly; the strict set is often smaller and that is information, not a shortfall."""
    render_target: int = 10
    """How many anchors go in front of the model, in detail.

    When the strict set is smaller, the shortfall is made up from patients outside it,
    nearest first, and those are LABELLED rather than blended in. A single fixed k forces
    exactly k patients whether or not k of them deserve the name."""
    relax_to_fill: bool = True
    """Top up a short strict set from outside it. False leaves it short."""
    selection: Selection = "nearest"

    # --- when to refuse ------------------------------------------------------------------
    min_support: int = 10
    max_distance: float | None = None
    """Hard ceiling on the distance to the k-th anchor. None means no ceiling."""

    # --- what the model is told about each anchor ----------------------------------------
    fields: tuple[str, ...] = ()
    """Per-anchor fields to render. Empty means the renderer chooses (see `compact`)."""
    context_fields: tuple[str, ...] = ()
    """Always rendered, whatever else is chosen: the axes that say WHICH WORLD this is —
    cancer, driver, line, action, era. Retrieval may be broad; the rendering is not."""
    n_varying: int = 5
    """How many of the remaining channels to show, chosen as those that differ most between
    the query and the anchors. Twenty columns of which fifteen are identical is noise that
    buries the five that matter."""
    n_strata: int = 3
    """How many axes an evidence set is split on, at most, in the conditional renderer.
    Three keeps the block short and each subgroup large enough to mean something."""
    include_residual: bool = True
    """How the query differs from the anchor centre, channel by channel."""
    include_aggregate: bool = True
    """A summary of all k anchors, not only the rendered few."""
    include_outcomes: bool = True
    """Anchors' observed outcomes. False isolates whether the model uses them at all."""

    # --- controls, which must be as easy to run as the real thing ------------------------
    shuffle_anchors: bool = False
    """Replace the anchors with unrelated patients. The decisive negative control: if a run
    with this set performs as well as the real one, the model is reacting to prompt volume
    rather than to the evidence in it."""
    shuffle_scope: str = "corpus"
    """Where the sham anchors come from.

    ``corpus`` (default) — any patient at all, uniformly at random. This is the control the
    experiment needs: it asks whether retrieved evidence matters, and with thousands of
    patients available it is ALWAYS constructible.

    ``eligible`` — restricted to rows passing the same filters (same regimen, same cancer)
    but outside the neighbourhood. A narrower question: does PROXIMITY matter, holding
    relevance constant. Weaker as a control, since far-but-same-regimen patients are still
    informative, and it cannot be built where the eligible pool is no larger than k --
    exactly the sparse cases that matter most. Available, not the default."""
    shuffle_seed: int = 0

    label: str = "default"
    """Human-readable name, carried into provenance so runs are distinguishable at a glance."""

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be at least 1")
        if self.render_target < 1:
            raise ValueError("render_target must be at least 1")
        if self.min_support > self.k:
            raise ValueError(f"min_support {self.min_support} exceeds k {self.k}")
        if self.target_action is not None and self.same_action:
            raise ValueError("target_action requires same_action=False")

    # -------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def hash(self) -> str:
        """Stable over field ORDER, sensitive to every value. Goes into every result."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def variant(self, **changes) -> AnchorPolicy:
        """A copy with fields changed -- the way ablation arms are declared."""
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """What retrieval produced, and everything needed to audit or reproduce it."""

    query_id: str
    anchor_index: tuple[int, ...]
    """Positional indices into the corpus. Empty when the query is unsupported."""
    distances: tuple[float, ...]
    n_eligible: int
    """How many rows passed the eligibility filters, before k was applied."""
    support_label: SupportLabel
    residual: dict[str, float] = field(default_factory=dict)
    """Query channel minus anchor-centroid channel. Positive means the query is higher."""

    # --- the support audit: what this anchor set is worth, recorded with it ---------------
    anchor_spread: float = float("nan")
    """Mean distance of anchors from their own centre. How alike they actually are — a wide
    set is a loose analogy however many members it has."""
    outcome_dispersion: float = float("nan")
    """Interquartile range of the anchors' outcomes. Together with spread this is the
    roughness measure: anchors that are close but disagree are the case where averaging
    them fails and reasoning has to do the work."""
    max_distance_used: float = float("nan")
    """Distance to the furthest anchor actually retained."""
    relaxed_axes: tuple[str, ...] = ()
    """Which axes were relaxed to find these anchors, e.g. ("action",). Empty means none —
    the difference between rung 1 and rung 2 of the evidence ladder, recorded rather than
    inferred from the label."""
    support_space: tuple[str, ...] = ()
    """The channels the anchors were actually matched and ranked on. "High support" is a
    statement about THIS space and no other: a hundred neighbours matched on cancer, stage,
    line and regimen say nothing about whether the patient's performance status is typical
    of them, because nobody recorded it."""
    unobserved_axes: tuple[str, ...] = ()
    """Channels the SOURCE never records, for anchors or query alike."""
    query_unobserved: tuple[str, ...] = ()
    """Channels not assessed for THIS query, though the source records them in general.
    Distinct from `unobserved_axes`: one is a property of the data source, the other of the
    patient's own record."""
    corpus_id: str = ""
    """Hash of the corpus this was retrieved from. Two results from different corpora are
    not comparable and must not be silently pooled."""
    n_strict: int = 0
    """How many anchors came from the strict set."""
    tier_labels: tuple[str, ...] = ()
    """One label per anchor, in the same order: which rule admitted it."""
    tier_counts: tuple[tuple[str, int], ...] = ()
    """(label, count) per tier, in order, for the prompt and the audit."""
    censored_fraction: float = float("nan")
    """Share of anchors whose primary endpoint is censored. A set that is numerous but
    mostly censored carries far less than its count suggests."""
    is_sham: bool = False
    """True when the anchors are a negative control rather than retrieved evidence.

    A sham result carries support_label "sham" and must never be counted as direct or
    transport support: a random draw is not evidence, and letting it inherit a real label
    would put the control inside the support and quadrant analyses as though it were data."""

    policy_hash: str = ""
    policy_label: str = ""
    note: str = ""
    """Why a query was refused, when it was."""

    @property
    def supported(self) -> bool:
        """Has usable anchors. A sham control has anchors but is NOT support."""
        return self.support_label not in ("unsupported", "sham")

    @property
    def has_anchors(self) -> bool:
        """Anchors exist to render, whether or not they are evidence."""
        return len(self.anchor_index) > 0

    @property
    def n_anchors(self) -> int:
        return len(self.anchor_index)
