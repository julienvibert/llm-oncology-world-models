"""Which frozen prompt this pipeline is derived from, asserted rather than assumed.

The anchored template is a four-line diff of the prompt the paper-v1 runs actually used. That
claim is only worth anything if it is checked, so the parent's hash is pinned here and
verified before a prompt is composed.

The parent template is the one that generated the deposited runs: every run manifest records
`prompt_template_sha256 = 3109486f...`, which is `experimental_lean_ledger_v2_rich_neutralized.md`
(not `prompts/oncollm_v1/unified_minimal.md`, which the generic prompt loader reads). The check
below pins that parent by hash.

`verify_parent` compares against the run-recorded hash, not against whatever is on disk, so
editing the parent breaks the check instead of silently redefining the baseline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

#: The prompt every paper-v1 atlas run recorded in its manifest as
#: `prompt_template_sha256`. Pinned, not computed: a computed value would follow the file
#: if the file changed, which is the failure this guards against.
FROZEN_PARENT_SHA256 = "3109486f7e2f96f94c61685bd000a15fdef52e6759b13b11517af15407d0bac6"
FROZEN_PARENT = ROOT / "prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md"

ANCHORED_TEMPLATE = (ROOT / "prompts/anchored_generation_v1"
                     / "lean_ledger_v2_rich_neutralized_anchored.md")

#: The CONTINUATION contract: a fixed patient in, future events out.
#:
#: The wording is deliberately SILENT ON PROVENANCE. It was first written for the real-patient
#: GENIE work and said so ("this run describes a REAL patient"), which is false when the same
#: contract carries a constructed battery anchor. Its function -- these facts are fixed,
#: absence means unspecified -- is identical either way, so the neutral phrasing lets both
#: surfaces share one contract without either being told something untrue about itself.
#:
#: The anchored template above inherits the frozen synthetic-trajectory contract, under
#: which the model completes a partial baseline through `baseline_patch`. For a real GENIE
#: landmark that changes the statistical object: a model that invents ECOG, organ function
#: and burden, then emits a course consistent with the patient it just invented, is not
#: predicting this patient, and the spread over repeats measures how much its story varies
#: rather than uncertainty about the real one. Since we score a DISTRIBUTION with proper
#: scoring rules, that spread is the estimand -- so this is not a tidiness question.
#:
#: The alternative considered and rejected was a deterministic completer filling the
#: missing leaves. The schema has no `unknown` member -- `renal`/`hepatic`/`cardiac` run
#: `normal ... failure`, `ecog_ps_at_diagnosis` is an int -- so a completer must write
#: "normal", which is how the hybrid bridge ended up with every patient at an identical
#: host coordinate: a pipeline constant wearing a patient's clothes. Missingness is stated
#: in prose instead, and stays missing.
CONTINUATION_TEMPLATE = (ROOT / "prompts/anchored_generation_v1"
                         / "lean_ledger_v2_rich_neutralized_continuation.md")

#: The continuation template is the anchored template with EXACTLY these edits applied, and
#: `verify_continuation_template` re-applies them and compares. The diff is therefore code
#: rather than a claim about a file: an edit to either template that is not also an edit
#: here fails the check. Each `old` must occur exactly `count` times.
CONTINUATION_EDITS: tuple[tuple[str, str, str, int], ...] = (
    (
        "title names the contract",
        "# SynthOncCases Latent Trajectory Prompt v1 - Neutralized Framing Control",
        "# SynthOncCases Latent Trajectory Prompt v1 - Neutralized Framing Control "
        "(continuation contract)",
        1,
    ),
    (
        "output is the future ledger alone",
        '{ "baseline_patch": { ... }, "ledger": [ ... ] }',
        '{ "ledger": [ ... ] }',
        2,
    ),
    (
        "the baseline is given, not written",
        """`baseline_patch` is the at-diagnosis baseline JSON, restricted to the leaves
NOT supplied in `## Baseline Seed`. The pipeline deep-merges seed + patch into
the final baseline before deterministic state reconstruction. See `## Pinned
Fields` and `## Required Patch Leaves` for the exact split for this run.

- If the seed already provides a complete valid baseline, emit `{}` for
  `baseline_patch`.
- If the seed is partial, fill every required leaf listed under
  `## Required Patch Leaves` with a valid, non-placeholder value compatible
  with the schema and pinned seed.
- If the seed is empty, generate the entire baseline yourself, matching the
  shape in `## Baseline Schema`.

You MUST NOT write any dotted path listed under `## Pinned Fields`. A patch
that overlaps a pinned path is hard-rejected. A dotted path is a JSON field
path such as `identity_card.diagnosis.histology_bucket`.

`ledger` is the chronological event list. End with exactly one `death` event.""",
        """There is no `baseline_patch` in this contract. The patient facts for this
run are given under `## Baseline Seed` and the observed history under
`## Fixed Prefix`. Both are fixed. Emitting a baseline, or any object other
than `ledger`, is hard-rejected.

The seed is deliberately partial. Fields it does not mention were NOT OBSERVED
for this patient by the source; they are not normal, not absent, and not yours
to decide. Reason about the future under that uncertainty rather than resolving
it: where an unobserved factor would change the course, that shows as a wider
range of possible courses, which is the correct answer.

`ledger` is the chronological event list, and it contains ONLY events strictly
after the cutoff date given under `## Fixed Prefix`. Do not restate, revise or
re-date anything already in the prefix. End with exactly one `death` event.""",
        1,
    ),
    (
        "the trajectory starts at the landmark, not at diagnosis",
        """The trajectory starts at diagnosis. Do not emit treatment history for the
generated index cancer before baseline; all index-cancer therapy starts in the
ledger after diagnosis.""",
        """The trajectory continues from the cutoff date in `## Fixed Prefix`, which is
the day the index treatment started. Everything before it already happened and
is given; your first event must be strictly after it.""",
        1,
    ),
    (
        "the seed is observation, not a patch boundary",
        """These fields are pinned for this run. Treat their values as fixed; do NOT
include them in your `baseline_patch`. Empty `{}` means the entire baseline
is yours to fill. The `## Pinned Fields` section below is the machine-readable
path list derived from this seed; it exists to remove ambiguity.""",
        """Patient facts supplied for this run. These values are fixed, not
suggestions: reason from them and never contradict them. Fields absent here are
unspecified and must not be invented, and the trajectory you emit must not
depend on having decided them.""",
        1,
    ),
    (
        "pinned fields become observed facts",
        """Your `baseline_patch` MUST NOT contain any of these dotted leaf paths (a
patch that overlaps any of them is hard-rejected):""",
        """The observed leaves, as machine-readable paths. Nothing you emit may
contradict them:""",
        1,
    ),
    (
        "required patch leaves have no meaning without a patch",
        """## Required Patch Leaves

Your `baseline_patch` MUST cover every required leaf listed here that the
seed has not already supplied. List-valued entries (`clone_registry`,
`baseline_disease_burden`, ...) are required as whole arrays, not per-index:

{{REQUIRED_PATCH_LEAVES}}

""",
        "",
        1,
    ),
    (
        "the task is a continuation",
        """Now emit the JSON for this run using the resolved baseline, fixed prefix if
present, run-specific instructions, and schema contract above.""",
        """Now emit the JSON for this run: the future of the patient described by the
baseline seed and fixed prefix above, under the run-specific instructions and
schema contract. Do not emit a baseline and do not restate the prefix.""",
        1,
    ),
)

#: The one placeholder this pipeline adds, on its own line beside {{OBSERVED_PREFIX}}.
ANCHOR_PLACEHOLDER = "{{ANCHOR_BLOCK}}"
#: The placeholder WITH its surrounding newlines. Substituting this form with the empty
#: string removes the line entirely and reproduces the frozen parent byte for byte -- which
#: is what makes the unanchored arm a control by construction. Replacing only the
#: placeholder would leave a blank line behind and the identity claim would be false.
ANCHOR_LINE = "\n" + ANCHOR_PLACEHOLDER + "\n"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_parent() -> str:
    """Raise unless the frozen parent on disk is the one the runs actually used."""
    if not FROZEN_PARENT.exists():
        raise FileNotFoundError(f"frozen parent prompt is missing: {FROZEN_PARENT}")
    got = sha256_of(FROZEN_PARENT)
    if got != FROZEN_PARENT_SHA256:
        raise ValueError(
            f"frozen parent prompt has changed.\n"
            f"  expected {FROZEN_PARENT_SHA256}  (recorded by every paper-v1 run manifest)\n"
            f"  found    {got}\n"
            f"  file     {FROZEN_PARENT.relative_to(ROOT)}\n"
            "The anchored template is a diff of that file. If the parent legitimately "
            "changed, the anchored template must be re-derived and this pin updated in the "
            "same commit — never one without the other.")
    return got


def verify_anchored_template() -> str:
    """Raise unless the anchored template is the frozen parent plus the anchor block only."""
    verify_parent()
    if not ANCHORED_TEMPLATE.exists():
        raise FileNotFoundError(f"anchored template is missing: {ANCHORED_TEMPLATE}")
    text = ANCHORED_TEMPLATE.read_text()
    if ANCHOR_LINE not in text:
        raise ValueError(
            f"anchored template has no {ANCHOR_PLACEHOLDER} on its own line; the empty-block "
            "case cannot then reproduce the parent")
    # the ONLY admissible difference is the anchor line. Anything else and the claim that
    # this template differs from the frozen prompt by anchoring alone is false.
    if text.replace(ANCHOR_LINE, "", 1) != FROZEN_PARENT.read_text():
        raise ValueError(
            "anchored template is not the frozen parent plus the anchor line alone. "
            "Removing the anchor line must reproduce the parent byte for byte; it does not. "
            "Re-derive the template.")
    return sha256_of(ANCHORED_TEMPLATE)


def _apply_continuation_edits(text: str) -> str:
    """The anchored template turned into the continuation template, edit by edit."""
    for label, old, new, count in CONTINUATION_EDITS:
        found = text.count(old)
        if found != count:
            raise ValueError(
                f"continuation edit {label!r} expected {count} occurrence(s) of its source "
                f"text and found {found}. The anchored template changed under it; re-derive "
                "the edit rather than loosening the count.")
        text = text.replace(old, new, count)
    return text


def verify_continuation_template() -> str:
    """Raise unless the continuation template is the anchored one plus the declared edits."""
    verify_anchored_template()
    if not CONTINUATION_TEMPLATE.exists():
        raise FileNotFoundError(f"continuation template is missing: {CONTINUATION_TEMPLATE}")
    expected = _apply_continuation_edits(ANCHORED_TEMPLATE.read_text())
    if CONTINUATION_TEMPLATE.read_text() != expected:
        raise ValueError(
            "continuation template does not match the declared edit list. Either the file "
            "was hand-edited or an edit was changed without regenerating it. The edit list "
            "in this module is the contract; the file is its output.")
    # exactly one mention survives, and it is the sentence forbidding it
    text = CONTINUATION_TEMPLATE.read_text()
    if text.count("baseline_patch") != 1 or "There is no `baseline_patch`" not in text:
        raise ValueError(
            "the continuation template should mention baseline_patch exactly once, in the "
            "sentence that forbids it")
    return sha256_of(CONTINUATION_TEMPLATE)


def write_continuation_template() -> Path:
    """Regenerate the continuation template from the anchored one. Run after an edit."""
    CONTINUATION_TEMPLATE.write_text(_apply_continuation_edits(ANCHORED_TEMPLATE.read_text()))
    return CONTINUATION_TEMPLATE


def provenance() -> dict:
    """What a run must record so its prompt lineage can be reconstructed later."""
    return {
        "frozen_parent": str(FROZEN_PARENT.relative_to(ROOT)),
        "frozen_parent_sha256": verify_parent(),
        "anchored_template": str(ANCHORED_TEMPLATE.relative_to(ROOT)),
        "anchored_template_sha256": verify_anchored_template(),
        "continuation_template": str(CONTINUATION_TEMPLATE.relative_to(ROOT)),
        "continuation_template_sha256": verify_continuation_template(),
        "continuation_edits": [label for label, *_ in CONTINUATION_EDITS],
        "anchor_placeholder": ANCHOR_PLACEHOLDER,
        "empty_block_reproduces_parent": (
            render_template("") == FROZEN_PARENT.read_text()),
        "note": ("The anchored template is the frozen parent plus one conditional line. "
                 "An empty block reproduces the parent byte for byte, so the unanchored "
                 "arm IS the frozen prompt. Anchors are substituted before the frozen "
                 "runner's render_prompt, which is never modified."),
    }


#: `latent_completion` -- the frozen synthetic contract: the model completes the baseline.
#: Kept as a NAMED ABLATION, not deleted: "would it predict better if allowed to fill in the
#: unobserved?" is the first question a reader asks, and this is the arm that answers it.
#: `continuation` -- the primary contract for real-patient prediction.
Contract = str


def render_template(anchor_block: str, contract: Contract = "latent_completion") -> str:
    """The template with the block substituted, under the named contract.

    Under `latent_completion` an empty or whitespace-only block removes the line entirely
    and returns the frozen parent byte for byte, so the unanchored arm is literally the
    frozen prompt. Under `continuation` the same removal returns the continuation template
    without an anchor block -- still the identical prompt for every arm, which is what makes
    R3 a control, but no longer byte-identical to the parent. That is the price of asking a
    different question and it is stated rather than glossed.
    """
    if contract == "continuation":
        verify_continuation_template()
        text = CONTINUATION_TEMPLATE.read_text()
    elif contract == "latent_completion":
        verify_anchored_template()
        text = ANCHORED_TEMPLATE.read_text()
    else:
        raise ValueError(f"unknown contract {contract!r}; "
                         "have 'continuation' and 'latent_completion'")
    if not anchor_block.strip():
        return text.replace(ANCHOR_LINE, "", 1)
    return text.replace(ANCHOR_PLACEHOLDER, anchor_block, 1)


def compose_prompt(anchor_block: str, *, contract: Contract = "latent_completion",
                   **runner_kwargs) -> str:
    """The full prompt a provider receives: anchors substituted, then the runner's own render.

    The frozen runner's `render_prompt` takes the template TEXT, so the anchor block is
    substituted into the template before it is handed over. The runner then fills every
    placeholder it knows about and sees none it does not. Nothing in the frozen pipeline is
    modified, and pointing a config at this template without going through here would leave
    {{ANCHOR_BLOCK}} unsubstituted in the prompt sent to the model.

    `runner_kwargs` are passed straight through: baseline_seed, pinned_paths,
    required_patch_leaves, policy_text, knowledge_cutoff, terminal_contract,
    conditioning_spec, prompt_enum_order.
    """
    from synthonccases.oncollm.poc_runner import render_prompt
    if contract == "continuation":
        # no patch means no required leaves. Passing them would print a section the
        # continuation template no longer has, and asking for leaves it forbids.
        runner_kwargs["required_patch_leaves"] = ()
    return render_prompt(render_template(anchor_block, contract), **runner_kwargs)


if __name__ == "__main__":
    print(json.dumps(provenance(), indent=2))
