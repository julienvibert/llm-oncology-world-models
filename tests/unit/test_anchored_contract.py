"""The anchored template must be a diff of the prompt the runs actually used."""

from __future__ import annotations

import pytest

from synthonccases.anchored import contract as C
from synthonccases.anchored import provenance, render_template


def test_parent_is_the_prompt_every_run_recorded():
    """3109486f... is prompt_template_sha256 in every paper-v1 atlas manifest."""
    assert C.verify_parent() == C.FROZEN_PARENT_SHA256


def test_anchored_template_is_only_the_anchor_block_away_from_its_parent():
    assert C.verify_anchored_template()


def test_an_empty_block_returns_the_frozen_prompt_byte_for_byte():
    """The unanchored arm IS the frozen prompt. Exact equality, no tolerance.

    An earlier version of this test allowed up to three differing lines, which let two blank
    lines and an edited Task sentence through while the docstring claimed byte identity.
    A control that is nearly the frozen prompt is not a control.
    """
    assert render_template("") == C.FROZEN_PARENT.read_text()
    assert render_template("   \n  ") == C.FROZEN_PARENT.read_text()


def test_removing_the_anchor_line_reproduces_the_parent():
    """The template differs from the frozen prompt by the anchor line and nothing else."""
    text = C.ANCHORED_TEMPLATE.read_text()
    assert text.replace(C.ANCHOR_LINE, "", 1) == C.FROZEN_PARENT.read_text()


def test_a_template_with_any_other_edit_is_rejected(tmp_path, monkeypatch):
    tampered = tmp_path / "tampered.md"
    tampered.write_text(C.ANCHORED_TEMPLATE.read_text() + "\nAn extra instruction.\n")
    monkeypatch.setattr(C, "ANCHORED_TEMPLATE", tampered)
    with pytest.raises(ValueError, match="not the frozen parent plus the anchor line"):
        C.verify_anchored_template()


def test_composing_fills_every_placeholder_the_runner_knows():
    """Pointing a config at the template without composing leaves {{ANCHOR_BLOCK}} in it."""
    out = C.compose_prompt("RETRIEVED PATIENTS HERE", baseline_seed={},
                           pinned_paths=(), required_patch_leaves=(), policy_text="")
    assert "RETRIEVED PATIENTS HERE" in out
    assert "{{" not in out and "}}" not in out


def test_the_block_is_substituted_and_nothing_else_moves():
    out = render_template("RETRIEVED PATIENTS HERE")
    assert "RETRIEVED PATIENTS HERE" in out
    assert C.ANCHOR_PLACEHOLDER not in out
    # every other placeholder the runner fills must survive untouched
    for ph in ("{{TASK_OPENING}}", "{{KNOWLEDGE_CUTOFF}}", "{{LEDGER_EVENT_SCHEMA}}",
               "{{BASELINE_SCHEMA}}", "{{BASELINE_SEED}}", "{{PINNED_FIELDS}}",
               "{{REQUIRED_PATCH_LEAVES}}", "{{OBSERVED_PREFIX}}", "{{POLICY_TEXT}}"):
        assert ph in out, f"{ph} was lost from the anchored template"


def test_a_changed_parent_fails_loudly(monkeypatch):
    """Silently redefining the baseline is the failure this guard exists for."""
    monkeypatch.setattr(C, "FROZEN_PARENT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="frozen parent prompt has changed"):
        C.verify_parent()


def test_provenance_records_both_hashes():
    p = provenance()
    assert p["frozen_parent_sha256"] == C.FROZEN_PARENT_SHA256
    assert len(p["anchored_template_sha256"]) == 64
    assert "experimental_lean_ledger_v2_rich_neutralized" in p["frozen_parent"]
