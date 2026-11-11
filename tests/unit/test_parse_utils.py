"""Unit tests for ``synthonccases.oncollm.parse_utils``.

Covers the shared parse helpers used by both ``provider.OncoLLMProvider``
(canonical record path) and ``poc_runner.classify_response`` (lean-ledger
path). Centralizing the tests here prevents drift between the two parse
sites.
"""

from __future__ import annotations

from synthonccases.oncollm.parse_utils import strip_code_fence


def test_strip_code_fence_unchanged_on_bare_json():
    assert strip_code_fence('{"foo": 1}') == '{"foo": 1}'


def test_strip_code_fence_strips_surrounding_whitespace_only():
    assert strip_code_fence('  \n {"foo": 1}\n  ') == '{"foo": 1}'


def test_strip_code_fence_handles_json_tagged_fence():
    fenced = '```json\n{"foo": 1, "bar": [2, 3]}\n```'
    assert strip_code_fence(fenced) == '{"foo": 1, "bar": [2, 3]}'


def test_strip_code_fence_handles_untagged_fence():
    fenced = '```\n{"foo": 1}\n```'
    assert strip_code_fence(fenced) == '{"foo": 1}'


def test_strip_code_fence_handles_outer_whitespace_around_fence():
    fenced = '\n\n```json\n{"foo": 1}\n```\n\n'
    assert strip_code_fence(fenced) == '{"foo": 1}'


def test_strip_code_fence_handles_truncated_fence_missing_closer():
    # Defensive: if the closing fence is missing (truncation, max-tokens cap),
    # still strip the opener so the parser can attempt the inner JSON.
    fenced = '```json\n{"foo": 1}'
    assert strip_code_fence(fenced) == '{"foo": 1}'


def test_strip_code_fence_preserves_internal_backticks():
    # A JSON string value containing backticks must NOT be confused with a
    # closing fence — only the OUTER fence is stripped.
    fenced = '```json\n{"snippet": "use `git status`"}\n```'
    assert strip_code_fence(fenced) == '{"snippet": "use `git status`"}'
