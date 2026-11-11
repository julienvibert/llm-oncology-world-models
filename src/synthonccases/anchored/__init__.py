"""Anchored generation: retrieve real comparable patients, then condition a generation on them.

Two steps, deliberately separable:

    1. RETRIEVAL   deterministic, pure, no model call. Which real patients stand behind this
                   generation, or a labelled refusal when too few exist.
    2. RENDERING   those anchors as a block of prompt text. Swappable, because what the model
                   is told about its anchors is as open a question as which anchors it gets.
    3. QUERY       who the generation is FOR: a pinned baseline seed and a dated observed
                   prefix, built once per landmark and identical across every arm, so the
                   anchor block is the only thing that varies between them.
    4. CONTRACT    what the model may write back. `continuation` is the primary one for
                   real-patient prediction -- future events only, no baseline. Under the
                   inherited `latent_completion` contract the model completes the missing
                   baseline, which changes the statistical object: a course conditioned on
                   an invented ECOG is a course for a different patient. That contract is
                   kept as a named ablation because "does letting it fill in the unobserved
                   help?" is a question worth answering, not one to assume.

Both are driven by one hashable :class:`~.policy.AnchorPolicy`, so every open design question
-- how many neighbours, which distance, what is shown, when to refuse -- is a parameter and
never a code path. The policy hash travels with every result, so two runs under different
anchoring cannot be confused for one experiment.

ADDITIVE TO THE FROZEN PIPELINE. Nothing here edits a prompt template, schema, enum,
conditioning spec, or the runner. The block this package produces is appended by a caller;
supply no block and the frozen pipeline emits byte-identical prompts. Reproducibility of
every existing result is unaffected.

    >>> corpus = Corpus(frame, channels=[...], subject_col="patient", action_col="arm")
    >>> policy = AnchorPolicy(k=40, render_target=10, label="v1-direct")
    >>> result = retrieve(corpus, query, policy)
    >>> block  = render("table", corpus, result, policy)
"""

from .contract import (
    compose_prompt, provenance, render_template, verify_anchored_template,
    verify_continuation_template,
)
from .endpoint import Endpoint, extract_many, extract_pfs
from .policy import LADDER, AnchorPolicy, RetrievalResult, SupportLabel
from .future import (
    Endpoints, FutureLedger, endpoints, pfs_from_future, validate_future_ledger,
)
from .query import (
    baseline_seed, conditioning_payload, diagnosis_date, oncotree_code, prefix_events,
)
from .render import available, register, render
from .summary import Survival, choose_strata, format_km, km_summary, stratify
from .retrieval import Corpus, retrieve, retrieve_pair

__all__ = ["LADDER", "AnchorPolicy", "Corpus", "Endpoint", "Endpoints", "FutureLedger",
           "RetrievalResult", "Survival", "SupportLabel", "available", "baseline_seed",
           "choose_strata", "format_km", "km_summary", "stratify",
           "compose_prompt", "conditioning_payload", "diagnosis_date", "extract_many",
           "endpoints", "extract_pfs", "oncotree_code", "pfs_from_future", "prefix_events", "provenance",
           "register", "render", "render_template", "retrieve", "retrieve_pair",
           "validate_future_ledger", "verify_anchored_template",
           "verify_continuation_template"]
