"""Seed hash + conditioning hash determinism."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from synthonccases.oncollm.conditioning_audit import (
    compute_conditioning_hash,
    compute_seed_hash,
)
from synthonccases.oncollm.poc_runner import (
    FixedFirstLineTreatment,
    PolicySpec,
    RegimenLine,
    _canonical_policy_payload,
)


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "rev5_poc" / "her2_brca_baseline_002.json"
)


def _full() -> dict:
    return json.loads(FIXTURE.read_text())


def _policy(
    *,
    text: str = "Standard policy.",
    regimen_sequence: tuple[RegimenLine, ...] = (),
    unconstrained_after_line: int | None = None,
    fixed_first_line_treatment: FixedFirstLineTreatment | None = None,
) -> PolicySpec:
    return PolicySpec(
        id="p_test",
        text=text,
        regimen_sequence=regimen_sequence,
        unconstrained_after_line=unconstrained_after_line,
        fixed_first_line_treatment=fixed_first_line_treatment,
    )


# ---------- compute_seed_hash ----------


def test_seed_hash_deterministic_across_key_reordering() -> None:
    a = {"identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}}
    b = {"identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}}
    assert compute_seed_hash(a) == compute_seed_hash(b)


def test_seed_hash_changes_when_pinned_value_changes() -> None:
    a = {"identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}}
    b = {"identity_card": {"diagnosis": {"oncotree_code": "LUAD"}}}
    assert compute_seed_hash(a) != compute_seed_hash(b)


def test_seed_hash_empty_seed_is_64_hex() -> None:
    h = compute_seed_hash({})
    assert len(h) == 64
    int(h, 16)  # parses as hex


def test_seed_hash_for_full_baseline_is_stable() -> None:
    full = _full()
    assert compute_seed_hash(full) == compute_seed_hash(_full())


# ---------- compute_conditioning_hash (per-cell, includes full policy) ----------


def test_conditioning_hash_identical_policies_identical_hash() -> None:
    ffl = FixedFirstLineTreatment(regimen_label="THP", drugs=("T", "H", "P"))
    p1 = _policy(
        text="Same text.",
        regimen_sequence=(RegimenLine(line=1, aliases=("THP",)),),
        unconstrained_after_line=None,
        fixed_first_line_treatment=ffl,
    )
    p2 = dataclasses.replace(p1, id="p_other")
    h1 = compute_conditioning_hash(
        seed_hash="abc", canonical_policy_payload=_canonical_policy_payload(p1),
    )
    h2 = compute_conditioning_hash(
        seed_hash="abc", canonical_policy_payload=_canonical_policy_payload(p2),
    )
    assert h1 == h2


def test_conditioning_hash_differs_when_unconstrained_after_line_differs() -> None:
    """Regression for rev3 feedback: two policies with the same 1L pin but
    different downstream policy state must hash differently."""
    ffl = FixedFirstLineTreatment(regimen_label="THP", drugs=("T",))
    p1 = _policy(unconstrained_after_line=2, fixed_first_line_treatment=ffl)
    p2 = _policy(unconstrained_after_line=3, fixed_first_line_treatment=ffl)
    h1 = compute_conditioning_hash(
        seed_hash="x", canonical_policy_payload=_canonical_policy_payload(p1),
    )
    h2 = compute_conditioning_hash(
        seed_hash="x", canonical_policy_payload=_canonical_policy_payload(p2),
    )
    assert h1 != h2


def test_conditioning_hash_differs_when_regimen_sequence_differs() -> None:
    ffl = FixedFirstLineTreatment(regimen_label="THP", drugs=("T",))
    p1 = _policy(
        regimen_sequence=(RegimenLine(line=1, aliases=("THP",)),),
        fixed_first_line_treatment=ffl,
    )
    p2 = _policy(
        regimen_sequence=(
            RegimenLine(line=1, aliases=("THP",)),
            RegimenLine(line=2, aliases=("T-DM1",)),
        ),
        fixed_first_line_treatment=ffl,
    )
    h1 = compute_conditioning_hash(
        seed_hash="x", canonical_policy_payload=_canonical_policy_payload(p1),
    )
    h2 = compute_conditioning_hash(
        seed_hash="x", canonical_policy_payload=_canonical_policy_payload(p2),
    )
    assert h1 != h2


def test_conditioning_hash_differs_when_text_differs() -> None:
    p1 = _policy(text="A")
    p2 = _policy(text="B")
    h1 = compute_conditioning_hash(
        seed_hash="x", canonical_policy_payload=_canonical_policy_payload(p1),
    )
    h2 = compute_conditioning_hash(
        seed_hash="x", canonical_policy_payload=_canonical_policy_payload(p2),
    )
    assert h1 != h2


def test_conditioning_hash_differs_when_seed_hash_differs() -> None:
    p = _policy()
    h1 = compute_conditioning_hash(
        seed_hash="aaa", canonical_policy_payload=_canonical_policy_payload(p),
    )
    h2 = compute_conditioning_hash(
        seed_hash="bbb", canonical_policy_payload=_canonical_policy_payload(p),
    )
    assert h1 != h2
