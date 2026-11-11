"""Shared parsing helpers for OncoLLM provider responses.

The paper-v1 path has two JSON-parse sites: the OpenAI-shaped record path in
``provider.OncoLLMProvider.generate`` and the lean-ledger path in
``poc_runner.classify_response``. Both must accept the same set of provider
quirks (e.g., Anthropic via OpenRouter wrapping JSON in markdown code
fences). Centralizing the cleanup helpers here ensures the two parse paths
cannot drift apart.

Helpers never modify the on-disk raw response — they only normalize the
string just before ``json.loads``. Audit packs continue to surface the
original model output so quirks like fenced output remain measurable.
"""

from __future__ import annotations


def strip_code_fence(text: str) -> str:
    """Strip an optional markdown code fence around a JSON payload.

    Some providers (notably Anthropic via OpenRouter) wrap structured output
    in ```` ```json ... ``` ```` blocks even when asked for raw JSON, because
    ``response_format: json_object`` is an OpenAI-shaped contract that
    OpenRouter does not translate into Anthropic's tool-use API. This helper
    removes the outer fence if present so the JSON parser can proceed.
    Returns the text unchanged (minus surrounding whitespace) when no leading
    fence is detected.

    The model's original raw text is preserved upstream by callers — only
    the JSON-parse step uses the stripped form. "The model wrapped its JSON
    in a fence" therefore remains a measurable behavior in audit packs even
    though it no longer breaks the parse.
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    newline = s.find("\n")
    if newline == -1:
        s = s[3:]
    else:
        s = s[newline + 1:]
    s = s.rstrip()
    if s.endswith("```"):
        s = s[:-3].rstrip()
    return s
