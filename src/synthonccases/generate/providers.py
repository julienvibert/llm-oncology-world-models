"""LLM provider abstraction for patient generation.

Each provider uses:
- temperature=1.0 when that parameter is sent
- top_p=1.0 when that parameter is sent
- max_tokens varies by provider (see class attributes; recorded per-call on GenerationResult)
- fresh context per call (no conversation history)
- up to 3 provider-level attempts on failure
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REQUEST_TEMPERATURE = 1.0
DEFAULT_REQUEST_TOP_P = 1.0
DEFAULT_REQUEST_SEED: int | None = None
PROVIDER_INTERNAL_MAX_ATTEMPTS = 3


@dataclass
class GenerationResult:
    """Result from a single LLM generation call.

    Fields added for provider-derived provenance:
      - max_tokens_sent: the actual token cap sent on the request (varies by provider).
      - max_tokens_param: "max_tokens" or "max_completion_tokens" — disambiguates
        the OpenAI reasoning family which uses the latter.
      - reasoning_mode: "off" (standard call), "on" (explicit reasoning:high),
        "forced_off" (paired reasoning model with reasoning:none), or
        "native_reasoning_family" (gpt-5/o-series — reasoning is intrinsic).
      - temperature_sent / top_p_sent / seed_sent: exact sampling parameters
        sent to the provider, or null when that parameter was intentionally not
        sent for that provider/model family.
      - finish_reason: provider terminal reason for the final API response
        (OpenAI/LiteLLM choice.finish_reason, Anthropic stop_reason, Gemini
        candidate.finish_reason when available).
      - prompt_cache_usage_reported / cached_input_tokens / cache_write_tokens:
        provider-reported prompt-cache accounting. These describe reused prompt
        computation, never cached model responses.
      - duration_s: wall-clock seconds for the final (accepted) attempt.
      - served_provider: the OpenRouter backend that actually served the call. Recorded
        because on the first GENIE partial 909 attempts were fanned across 15 backends, one
        of which returned reasoning tokens past `forced_off` on 80% of its calls, and no
        saved field named the backend -- response_model and every sampling parameter were
        identical across all of them. Attribution WAS possible, by resolving each saved
        response_id through OpenRouter's generation endpoint, so the partial is auditable;
        it just required a second API pass over every call. Recording the name inline makes
        the run self-describing and lets a pin be verified while it runs.
    """
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    model_id: str = ""
    response_model: str = ""  # actual model returned by API (may differ from requested)
    response_id: str = ""  # API response ID (reveals routing: bdrk=Bedrock, etc.)
    served_provider: str = ""  # OpenRouter backend that actually served the call
    retries: int = 0
    reported_cost: float | None = None  # actual cost reported by proxy (e.g. OpenRouter)
    max_tokens_sent: int = 0
    max_tokens_param: str = ""
    reasoning_mode: str = ""
    temperature_sent: float | None = None
    top_p_sent: float | None = None
    seed_sent: int | None = None
    finish_reason: str = ""
    prompt_cache_usage_reported: bool = False
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    duration_s: float = 0.0


# Approximate pricing per 1M tokens (USD), as of April 2026
_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_1M, output_per_1M)
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-sonnet-4.5": (3.0, 15.0),
    "claude-sonnet-4.6": (3.0, 15.0),
    "claude-opus-4.5": (5.0, 25.0),
    "claude-opus-4.6": (5.0, 25.0),
    "gpt-4o-2025-03-01": (2.5, 10.0),
    "gpt-5": (1.25, 10.0),
    "gpt-5.4": (2.5, 15.0),
    "gemini-2.5-pro-preview-03-25": (1.25, 10.0),
    "gemini-2.5-pro": (1.25, 10.0),
    "google/gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "moonshotai/kimi-k3": (3.0, 15.0),
    "moonshotai/kimi-k3:reasoning-minimal": (3.0, 15.0),
}


@dataclass
class UsageTracker:
    """Tracks cumulative token usage and estimated cost across calls.

    Also records request parameters from the last recorded call so the
    manifest can report provider-derived values instead of a static default.
    """
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    total_retries: int = 0
    total_reported_cost: float = 0.0  # actual cost reported by proxy (e.g. OpenRouter)
    total_cached_input_tokens: int = 0
    total_cache_write_tokens: int = 0
    calls_with_prompt_cache_usage: int = 0
    last_max_tokens_sent: int = 0
    last_max_tokens_param: str = ""
    last_reasoning_mode: str = ""
    last_temperature_sent: float | None = None
    last_top_p_sent: float | None = None
    last_seed_sent: int | None = None
    _model_id: str = ""

    def record(self, result: GenerationResult) -> None:
        self.total_input_tokens += result.input_tokens
        self.total_output_tokens += result.output_tokens
        self.total_calls += 1
        self.total_retries += result.retries
        if result.reported_cost is not None:
            self.total_reported_cost += result.reported_cost
        self.total_cached_input_tokens += result.cached_input_tokens
        self.total_cache_write_tokens += result.cache_write_tokens
        if result.prompt_cache_usage_reported:
            self.calls_with_prompt_cache_usage += 1
        if result.model_id:
            self._model_id = result.model_id
        if result.max_tokens_sent:
            self.last_max_tokens_sent = result.max_tokens_sent
        if result.max_tokens_param:
            self.last_max_tokens_param = result.max_tokens_param
        if result.reasoning_mode:
            self.last_reasoning_mode = result.reasoning_mode
        self.last_temperature_sent = result.temperature_sent
        self.last_top_p_sent = result.top_p_sent
        self.last_seed_sent = result.seed_sent

    @property
    def estimated_cost_usd(self) -> float:
        """Estimate cost based on model pricing. Returns 0.0 if model unknown."""
        pricing = _PRICING.get(self._model_id)
        if pricing is None:
            return 0.0
        input_price, output_price = pricing
        return (
            self.total_input_tokens * input_price / 1_000_000
            + self.total_output_tokens * output_price / 1_000_000
        )


def model_family(model_id: str) -> str:
    """Extract a clean family name from a model ID.

    Strips org prefix (before '/'), ':reasoning' suffix, and takes the
    first segment (up to the first '-' or end of string).

    Examples:
        'claude-sonnet-4-6'                         → 'claude'
        'anthropic/claude-sonnet-4.6'               → 'claude'
        'qwen/qwen3.5-flash-02-23:reasoning'        → 'qwen3.5'
        'google/gemma-4-31b-it'                     → 'gemma'
        'z-ai/glm-5.1'                             → 'glm'
        'openai/gpt-5.4'                           → 'gpt'
    """
    # Strip org prefix
    bare = model_id.split("/")[-1] if "/" in model_id else model_id
    # Strip :reasoning suffix
    bare = bare.split(":")[0]
    # Take first segment before '-'
    return bare.split("-")[0]


def model_id_to_slug(model_id: str) -> str:
    """Convert a model ID to a filesystem-safe slug.

    OpenRouter IDs contain '/' (e.g. 'google/gemini-2.5-flash-lite') which
    cannot appear in directory names. Replace '/' with '__' and ':' with '_'.

    Examples:
        'claude-sonnet-4-6'                 → 'claude-sonnet-4-6'
        'google/gemini-2.5-flash-lite'      → 'google__gemini-2.5-flash-lite'
        'anthropic/claude-sonnet-4.6'       → 'anthropic__claude-sonnet-4.6'
        'openai/gpt-5.4:reasoning'          → 'openai__gpt-5.4_reasoning'
    """
    return model_id.replace("/", "__").replace(":", "_")


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.usage = UsageTracker()

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        """Generate a response from the LLM. Fresh context per call."""
        ...


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    _MAX_TOKENS = 8192

    def __init__(self, model_id: str = "claude-sonnet-4-20250514"):
        super().__init__(model_id)
        import anthropic
        self.client = anthropic.Anthropic()

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        import anthropic

        last_error = None
        started = time.perf_counter()
        for attempt in range(PROVIDER_INTERNAL_MAX_ATTEMPTS):
            try:
                response = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=self._MAX_TOKENS,
                    temperature=DEFAULT_REQUEST_TEMPERATURE,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                result = GenerationResult(
                    text=response.content[0].text,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    model_id=self.model_id,
                    retries=attempt,
                    max_tokens_sent=self._MAX_TOKENS,
                    max_tokens_param="max_tokens",
                    reasoning_mode="off",
                    temperature_sent=DEFAULT_REQUEST_TEMPERATURE,
                    top_p_sent=None,
                    seed_sent=DEFAULT_REQUEST_SEED,
                    finish_reason=str(getattr(response, "stop_reason", "") or ""),
                    duration_s=time.perf_counter() - started,
                )
                self.usage.record(result)
                return result
            except (anthropic.APIError, anthropic.APIConnectionError) as e:
                last_error = e
                if attempt < PROVIDER_INTERNAL_MAX_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"Anthropic API failed after {PROVIDER_INTERNAL_MAX_ATTEMPTS} attempts: "
            f"{last_error}"
        )


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    _MAX_TOKENS = 8192

    def __init__(self, model_id: str = "gpt-4o-2025-03-01"):
        super().__init__(model_id)
        import openai
        self.client = openai.OpenAI()

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        import openai

        last_error = None
        started = time.perf_counter()
        for attempt in range(PROVIDER_INTERNAL_MAX_ATTEMPTS):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    max_tokens=self._MAX_TOKENS,
                    temperature=DEFAULT_REQUEST_TEMPERATURE,
                    top_p=DEFAULT_REQUEST_TOP_P,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                choice = response.choices[0]
                usage = response.usage
                result = GenerationResult(
                    text=choice.message.content or "",
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    model_id=self.model_id,
                    retries=attempt,
                    max_tokens_sent=self._MAX_TOKENS,
                    max_tokens_param="max_tokens",
                    reasoning_mode="off",
                    temperature_sent=DEFAULT_REQUEST_TEMPERATURE,
                    top_p_sent=DEFAULT_REQUEST_TOP_P,
                    seed_sent=DEFAULT_REQUEST_SEED,
                    finish_reason=str(getattr(choice, "finish_reason", "") or ""),
                    duration_s=time.perf_counter() - started,
                )
                self.usage.record(result)
                return result
            except (openai.APIError, openai.APIConnectionError) as e:
                last_error = e
                if attempt < PROVIDER_INTERNAL_MAX_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"OpenAI API failed after {PROVIDER_INTERNAL_MAX_ATTEMPTS} attempts: "
            f"{last_error}"
        )


def _gemini_finish_reason(response) -> str:
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""
    finish_reason = getattr(candidates[0], "finish_reason", "")
    return str(finish_reason or "")


class GoogleProvider(LLMProvider):
    """Google Gemini provider."""

    _MAX_OUTPUT_TOKENS = 8192

    def __init__(self, model_id: str = "gemini-2.5-pro-preview-03-25"):
        super().__init__(model_id)
        import google.generativeai as genai
        self.genai = genai

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        # Create model with system_instruction per call for fresh context
        model = self.genai.GenerativeModel(
            self.model_id,
            system_instruction=system_prompt,
        )
        last_error = None
        started = time.perf_counter()
        for attempt in range(PROVIDER_INTERNAL_MAX_ATTEMPTS):
            try:
                response = model.generate_content(
                    user_prompt,
                    generation_config=self.genai.types.GenerationConfig(
                        temperature=DEFAULT_REQUEST_TEMPERATURE,
                        top_p=DEFAULT_REQUEST_TOP_P,
                        max_output_tokens=self._MAX_OUTPUT_TOKENS,
                    ),
                )
                # Token counting varies by SDK version
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, "usage_metadata"):
                    input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                    output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

                result = GenerationResult(
                    text=response.text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model_id=self.model_id,
                    retries=attempt,
                    max_tokens_sent=self._MAX_OUTPUT_TOKENS,
                    max_tokens_param="max_output_tokens",
                    reasoning_mode="off",
                    temperature_sent=DEFAULT_REQUEST_TEMPERATURE,
                    top_p_sent=DEFAULT_REQUEST_TOP_P,
                    seed_sent=DEFAULT_REQUEST_SEED,
                    finish_reason=_gemini_finish_reason(response),
                    duration_s=time.perf_counter() - started,
                )
                self.usage.record(result)
                return result
            except Exception as e:
                last_error = e
                if attempt < PROVIDER_INTERNAL_MAX_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"Google API failed after {PROVIDER_INTERNAL_MAX_ATTEMPTS} attempts: "
            f"{last_error}"
        )


class LiteLLMProvider(LLMProvider):
    """Provider for LiteLLM proxy (OpenAI-compatible API).

    .. deprecated:: Use OpenRouterProvider for new work. LiteLLM support is
       retained only for reproducibility of smoke runs generated before
       April 2026.

    Uses LITELLM_BASE_URL and LITELLM_API_KEY environment variables.
    Routes all models (Claude, GPT, Gemini, etc.) through the same endpoint.
    """

    def __init__(self, model_id: str):
        super().__init__(model_id)
        import os
        import openai
        base_url = os.environ.get("LITELLM_BASE_URL")
        api_key = os.environ.get("LITELLM_API_KEY")
        if not base_url or not api_key:
            raise RuntimeError(
                "LiteLLM requires LITELLM_BASE_URL and LITELLM_API_KEY environment variables"
            )
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)

    _MAX_TOKENS = 16384
    _MAX_COMPLETION_TOKENS = 16384

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        import openai

        last_error = None
        started = time.perf_counter()
        for attempt in range(PROVIDER_INTERNAL_MAX_ATTEMPTS):
            try:
                # No response_format — some LiteLLM backends don't support it.
                # JSON output is enforced by the system prompt ("Output ONLY valid JSON")
                # and validated by _parse_json_response + schema validation.
                # Reasoning models (GPT-5, o-series) need different params:
                # - max_completion_tokens instead of max_tokens
                # - reasoning_effort="none" to avoid spending budget on thinking
                # - no temperature (reasoning models reject it)
                is_reasoning = any(
                    self.model_id.lower().startswith(p)
                    for p in ("gpt-5", "o1", "o3", "o4")
                )
                if is_reasoning:
                    max_tokens_sent = self._MAX_COMPLETION_TOKENS
                    max_tokens_param = "max_completion_tokens"
                    reasoning_mode = "native_reasoning_family"
                    response = self.client.chat.completions.create(
                        model=self.model_id,
                        max_completion_tokens=max_tokens_sent,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        extra_body={"reasoning_effort": "none"},
                    )
                    temperature_sent = None
                    top_p_sent = None
                else:
                    max_tokens_sent = self._MAX_TOKENS
                    max_tokens_param = "max_tokens"
                    reasoning_mode = "off"
                    response = self.client.chat.completions.create(
                        model=self.model_id,
                        max_tokens=max_tokens_sent,
                        temperature=DEFAULT_REQUEST_TEMPERATURE,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    temperature_sent = DEFAULT_REQUEST_TEMPERATURE
                    top_p_sent = None
                choice = response.choices[0]
                usage = response.usage
                # Extract reasoning tokens if present (reasoning models like GPT-5)
                reasoning_tokens = 0
                if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                    reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
                result = GenerationResult(
                    text=choice.message.content or "",
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    reasoning_tokens=reasoning_tokens,
                    model_id=self.model_id,
                    response_model=getattr(response, "model", ""),
                    response_id=getattr(response, "id", ""),
                    retries=attempt,
                    max_tokens_sent=max_tokens_sent,
                    max_tokens_param=max_tokens_param,
                    reasoning_mode=reasoning_mode,
                    temperature_sent=temperature_sent,
                    top_p_sent=top_p_sent,
                    seed_sent=DEFAULT_REQUEST_SEED,
                    finish_reason=str(getattr(choice, "finish_reason", "") or ""),
                    duration_s=time.perf_counter() - started,
                )
                self.usage.record(result)
                return result
            except (openai.APIError, openai.APIConnectionError) as e:
                last_error = e
                if attempt < PROVIDER_INTERNAL_MAX_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"LiteLLM API failed after {PROVIDER_INTERNAL_MAX_ATTEMPTS} attempts: "
            f"{last_error}"
        )


_LITELLM_ENV_PATH = Path.home() / ".config" / "litellm.env"


def _load_litellm_env() -> None:
    """Load LiteLLM credentials from ~/.config/litellm.env.

    Raises RuntimeError if the file is missing or doesn't set the required vars.
    """
    import os
    from dotenv import load_dotenv

    if not _LITELLM_ENV_PATH.exists():
        raise RuntimeError(
            f"LiteLLM env file not found at {_LITELLM_ENV_PATH}. "
            f"Create it with LITELLM_BASE_URL and LITELLM_API_KEY."
        )
    load_dotenv(_LITELLM_ENV_PATH, override=True)
    if not os.environ.get("LITELLM_BASE_URL") or not os.environ.get("LITELLM_API_KEY"):
        raise RuntimeError(
            f"{_LITELLM_ENV_PATH} must define LITELLM_BASE_URL and LITELLM_API_KEY"
        )


_OPENROUTER_ENV_PATH = Path.home() / ".config" / "openrouter.env"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_DEFAULT_TIMEOUT_SECONDS = 120.0
_KATHER_ENV_PATH = Path.home() / ".config" / "kather.env"
_KATHER_BASE_URL = "https://api.kather.ai/v1"
_KATHER_DEFAULT_TIMEOUT_SECONDS = 600.0
_KATHER_DEFAULT_MAX_TOKENS = 16384
_KATHER_REASONING_OFF_BODY = {
    "reasoning": {"effort": "none"},
    "chat_template_kwargs": {"enable_thinking": False},
}


def _openrouter_timeout_seconds() -> float:
    """Return the OpenRouter request timeout.

    A bounded timeout keeps sequential smoke runs from hanging indefinitely on a
    single stalled provider call. The environment override is intentionally
    narrow and provider-specific.
    """
    import os

    raw = os.environ.get("OPENROUTER_TIMEOUT_SECONDS")
    if raw is None or raw.strip() == "":
        return _OPENROUTER_DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            "OPENROUTER_TIMEOUT_SECONDS must be a positive number of seconds"
        ) from exc
    if value <= 0:
        raise RuntimeError(
            "OPENROUTER_TIMEOUT_SECONDS must be a positive number of seconds"
        )
    return value


def _load_openrouter_env() -> None:
    """Load OpenRouter credentials from ~/.config/openrouter.env.

    Raises RuntimeError if the file is missing or doesn't set OPENROUTER_API_KEY.
    """
    import os
    from dotenv import load_dotenv

    if not _OPENROUTER_ENV_PATH.exists():
        raise RuntimeError(
            f"OpenRouter env file not found at {_OPENROUTER_ENV_PATH}. "
            f"Create it with OPENROUTER_API_KEY=sk-or-..."
        )
    load_dotenv(_OPENROUTER_ENV_PATH, override=True)
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError(
            f"{_OPENROUTER_ENV_PATH} must define OPENROUTER_API_KEY"
        )


def _kather_timeout_seconds() -> float:
    """Return the Kather/OpenAI-compatible request timeout."""
    import os

    raw = os.environ.get("KATHER_TIMEOUT_SECONDS")
    if raw is None or raw.strip() == "":
        return _KATHER_DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            "KATHER_TIMEOUT_SECONDS must be a positive number of seconds"
        ) from exc
    if value <= 0:
        raise RuntimeError(
            "KATHER_TIMEOUT_SECONDS must be a positive number of seconds"
        )
    return value


def _kather_max_tokens() -> int:
    """Return the Kather/OpenAI-compatible output token cap."""
    import os

    raw = os.environ.get("KATHER_MAX_TOKENS")
    if raw is None or raw.strip() == "":
        return _KATHER_DEFAULT_MAX_TOKENS
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("KATHER_MAX_TOKENS must be a positive integer") from exc
    if value <= 0:
        raise RuntimeError("KATHER_MAX_TOKENS must be a positive integer")
    return value


def _load_kather_env() -> None:
    """Load Kather LiteLLM proxy credentials from ~/.config/kather.env.

    Operators can also export KATHER_API_KEY/KATHER_BASE_URL directly. The file
    is loaded only when present so cluster jobs can inject credentials through
    their scheduler environment.
    """
    import os
    from dotenv import load_dotenv

    if _KATHER_ENV_PATH.exists():
        load_dotenv(_KATHER_ENV_PATH, override=False)
    if not os.environ.get("KATHER_API_KEY"):
        raise RuntimeError(
            f"KATHER_API_KEY is not set. Create {_KATHER_ENV_PATH} with "
            "KATHER_API_KEY=<virtual key> and optionally "
            f"KATHER_BASE_URL={_KATHER_BASE_URL}."
        )


class KatherProvider(LLMProvider):
    """Provider for Kather's OpenAI-compatible LiteLLM proxy.

    This is intentionally a standard chat-completions call without
    ``response_format``. JSON is enforced by the paper-v1 system prompt and the
    existing parse/schema/replay gates, matching the historical LiteLLM posture.
    """

    _MAX_TOKENS = _KATHER_DEFAULT_MAX_TOKENS

    def __init__(self, model_id: str = "GLM-5.2-FP8"):
        super().__init__(model_id)
        _load_kather_env()
        import os
        import openai

        self._max_tokens = _kather_max_tokens()
        self.client = openai.OpenAI(
            base_url=os.environ.get("KATHER_BASE_URL", _KATHER_BASE_URL),
            api_key=os.environ["KATHER_API_KEY"],
            timeout=_kather_timeout_seconds(),
        )

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        import openai

        last_error = None
        started = time.perf_counter()
        for attempt in range(PROVIDER_INTERNAL_MAX_ATTEMPTS):
            try:
                max_tokens = getattr(self, "_max_tokens", self._MAX_TOKENS)
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    max_tokens=max_tokens,
                    temperature=DEFAULT_REQUEST_TEMPERATURE,
                    top_p=DEFAULT_REQUEST_TOP_P,
                    extra_body=_KATHER_REASONING_OFF_BODY,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                choice = response.choices[0]
                usage = response.usage
                reported_cost = None
                if usage:
                    reported_cost = getattr(usage, "cost", None)
                    if reported_cost is not None:
                        try:
                            reported_cost = float(reported_cost)
                        except (TypeError, ValueError):
                            reported_cost = None
                result = GenerationResult(
                    text=choice.message.content or "",
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    model_id=self.model_id,
                    response_model=getattr(response, "model", ""),
                    response_id=getattr(response, "id", ""),
                    retries=attempt,
                    reported_cost=reported_cost,
                    max_tokens_sent=max_tokens,
                    max_tokens_param="max_tokens",
                    reasoning_mode="forced_off",
                    temperature_sent=DEFAULT_REQUEST_TEMPERATURE,
                    top_p_sent=DEFAULT_REQUEST_TOP_P,
                    seed_sent=DEFAULT_REQUEST_SEED,
                    finish_reason=str(getattr(choice, "finish_reason", "") or ""),
                    duration_s=time.perf_counter() - started,
                )
                self.usage.record(result)
                return result
            except (openai.APIError, openai.APIConnectionError) as e:
                last_error = e
                if attempt < PROVIDER_INTERNAL_MAX_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"Kather API failed after {PROVIDER_INTERNAL_MAX_ATTEMPTS} attempts: "
            f"{last_error}"
        )


# Models whose reasoning must be explicitly controlled via extra_body.
# No-suffix → reasoning off; :reasoning suffix → reasoning on.
_OPENROUTER_REASONING_MODELS: set[str] = {
    "deepseek/deepseek-v4-pro",
    # Added 2026-08-05. Absent from this set, a model gets NO reasoning body at all, so it
    # reasons by default — and DeepSeek V4 Flash does. Measured in the PRIME smoke: a median
    # of 5,495 and up to 11,775 reasoning tokens against a 16,384 budget shared with an
    # ~8,000-token answer. One attempt spent the entire 16,384 on reasoning and returned
    # {"baseline_patch": {}} with no ledger. That read as a JSON-competence failure and was
    # nothing of the kind: it was token-budget exhaustion caused by reasoning we believed we
    # had switched off. The manifest said reasoning_mode="off" while Gemma, which IS in this
    # set, said "forced_off" — the two labels are the tell.
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-flash-0731",
    "google/gemini-3.1-flash-lite-preview",
    "moonshotai/kimi-k3",
    "z-ai/glm-5.1",
    "qwen/qwen3.5-flash-02-23",
    "google/gemma-4-31b-it",
}

_OPENROUTER_REASONING_SUFFIXES: tuple[tuple[str, str, bool, str], ...] = (
    (":reasoning-minimal", "minimal", True, "on:minimal"),
    (":reasoning-low", "low", True, "on:low"),
    (":reasoning", "high", False, "on"),
)

# OpenRouter strict-mode status codes that must NOT be retried. Semantic /
# request-shape errors won't fix themselves on retry; under
# require_parameters=True a 4xx schema rejection is a real smoke finding
# (it tells us per-cancer schema resolution is required) and should
# surface immediately. 408 (timeout), 429 (rate limit), and all 5xx
# remain retryable — those are recoverable by waiting.
_STRICT_MODE_FATAL_STATUS_CODES: frozenset[int] = frozenset({400, 401, 403, 404, 422})


def _openrouter_prompt_cache_usage(usage: object | None) -> tuple[bool, int, int]:
    """Return whether cache accounting was reported, plus read/write token counts.

    OpenRouter exposes these under ``usage.prompt_tokens_details``. The OpenAI SDK
    normally materialises that object as a typed model, while test doubles and some SDK
    versions may expose a mapping, so both representations are accepted.
    """
    details = getattr(usage, "prompt_tokens_details", None) if usage is not None else None
    if details is None:
        return False, 0, 0

    def value(name: str) -> int:
        raw = details.get(name, 0) if isinstance(details, dict) else getattr(details, name, 0)
        try:
            return max(0, int(raw or 0))
        except (TypeError, ValueError):
            return 0

    return True, value("cached_tokens"), value("cache_write_tokens")


#: Header that asks OpenRouter to return routing metadata alongside the completion, so the
#: serving backend is known without a second call to the generation endpoint.
_OPENROUTER_METADATA_HEADER = {"X-OpenRouter-Metadata": "enabled"}

#: Response fields that can carry the serving backend, in the order they are trusted.
#: `provider_name` is what the generation endpoint and the routing metadata call it; bare
#: `provider` appears on some completions.
_SERVED_PROVIDER_FIELDS = ("provider_name", "provider", "name")

#: Keys marking WHICH of the listed endpoints actually served the call. `endpoints.available`
#: enumerates candidates, so reading the first entry would name a backend that may never have
#: been used. Only an explicit marker, or a single-entry list, is trusted.
_SELECTED_ENDPOINT_FLAGS = ("selected", "is_selected", "chosen", "used")


def _as_dict(obj: object) -> dict:
    """A plain dict view of an SDK model, a mapping, or neither."""
    if isinstance(obj, dict):
        return obj
    extra = getattr(obj, "model_extra", None)
    return extra if isinstance(extra, dict) else {}


def _name_from(source: object) -> str:
    bag = _as_dict(source)
    for field in _SERVED_PROVIDER_FIELDS:
        for value in (getattr(source, field, None), bag.get(field)):
            if isinstance(value, str) and value:
                return value
    return ""


def _openrouter_served_provider(response: object) -> str:
    """Name the backend that served this call, or "" when the API did not say.

    Read in the order OpenRouter documents it. With `X-OpenRouter-Metadata: enabled` the
    routing block arrives as `openrouter_metadata`, whose `endpoints.available[]` list
    enumerates the CANDIDATE endpoints -- so the entry that served the call is identified by
    an explicit selection flag, or, when exactly one candidate exists, by being the only one.
    Guessing the first entry of a multi-entry list would name a backend that may never have
    been used, which is worse than admitting ignorance. Falls back to `provider_name` (the
    generation endpoint's spelling) and bare `provider`.

    Returning "" is a real answer, not a guess. The caller must treat it as a FAILURE: an
    unnamed backend cannot verify a pin, cannot verify an exclusion, and breaks the per-call
    provenance the governance statement commits to.
    """
    meta = getattr(response, "openrouter_metadata", None) or _as_dict(response).get(
        "openrouter_metadata")
    if meta is not None:
        endpoints = getattr(meta, "endpoints", None) or _as_dict(meta).get("endpoints")
        available = getattr(endpoints, "available", None) or _as_dict(endpoints).get(
            "available")
        if isinstance(available, (list, tuple)) and available:
            flagged = [e for e in available
                       if any(bool(getattr(e, f, None) or _as_dict(e).get(f))
                              for f in _SELECTED_ENDPOINT_FLAGS)]
            candidates = flagged or (list(available) if len(available) == 1 else [])
            for entry in candidates:
                name = _name_from(entry)
                if name:
                    return name
        direct = _name_from(meta)
        if direct:
            return direct
    return _name_from(response)


#: OpenRouter routing keys a caller may set through `provider_policy`. Anything else is a typo
#: or a silently ignored parameter, and a routing policy that is silently ignored is worse than
#: none: the manifest would record a guarantee the run never had.
#:
#: `require_parameters` is deliberately EXCLUDED. It has a dedicated argument, and allowing it
#: here would let a policy dict switch off the strict parameter enforcement that exists to stop
#: a backend ignoring `forced_off` -- the exact failure the pinning was introduced to prevent.
_OPENROUTER_PROVIDER_POLICY_KEYS = frozenset({
    "only", "ignore", "order", "allow_fallbacks", "data_collection", "zdr",
    "sort", "quantizations",
})


class OpenRouterProvider(LLMProvider):
    """Provider for OpenRouter (OpenAI-compatible API).

    Uses OPENROUTER_API_KEY from environment. Routes all models through
    https://openrouter.ai/api/v1. Model IDs use OpenRouter format with
    organization prefixes (e.g. anthropic/claude-sonnet-4-6, openai/gpt-5).

    Append ':reasoning' to the model ID to enable reasoning mode for
    models in _OPENROUTER_REASONING_MODELS. The suffix is stripped before
    calling the API, but preserved in model_id for manifests and slugs.
    """

    def __init__(
        self,
        model_id: str,
        *,
        response_format: dict | None = None,
        require_parameters: bool = False,
        disable_response_cache: bool = False,
        session_id: str | None = None,
        provider_policy: dict | None = None,
        max_tokens: int | None = None,
    ):
        # Parse explicit reasoning suffixes. Bare paired models still follow
        # their model-specific default branch below.
        self._reasoning_effort: str | None = None
        self._reasoning_exclude = False
        self._reasoning_mode_label = "off"
        for suffix, effort, exclude, mode_label in _OPENROUTER_REASONING_SUFFIXES:
            if model_id.endswith(suffix):
                self._api_model_id = model_id.removesuffix(suffix)
                self._reasoning_effort = effort
                self._reasoning_exclude = exclude
                self._reasoning_mode_label = mode_label
                break
        else:
            self._api_model_id = model_id

        self._reasoning_on = self._reasoning_effort is not None
        if self._reasoning_on:
            if self._api_model_id not in _OPENROUTER_REASONING_MODELS:
                raise ValueError(
                    f"Model {self._api_model_id} is not in the reasoning allowlist. "
                    f"Supported: {sorted(_OPENROUTER_REASONING_MODELS)}"
                )

        # Strict structured-output config (atlas-v1 plumbing-smoke contract).
        # Both default off so existing callers behave unchanged.
        self._response_format = response_format
        self._require_parameters = require_parameters
        self._disable_response_cache = disable_response_cache
        if session_id is not None and len(session_id) > 256:
            raise ValueError("OpenRouter session_id must be at most 256 characters")
        self._session_id = session_id
        if provider_policy is not None:
            unknown = sorted(set(provider_policy) - _OPENROUTER_PROVIDER_POLICY_KEYS)
            if unknown:
                raise ValueError(
                    f"unknown OpenRouter provider routing keys {unknown}; allowed: "
                    f"{sorted(_OPENROUTER_PROVIDER_POLICY_KEYS)}"
                )
            if provider_policy.get("allow_fallbacks") is False and not provider_policy.get("only"):
                raise ValueError(
                    "allow_fallbacks=False without `only` pins nothing -- it forbids a "
                    "fallback from an unconstrained first choice"
                )
        self._provider_policy = dict(provider_policy) if provider_policy else None
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        # Per-caller override, default None so every existing caller keeps the class value.
        # The GENIE run raises it because truncation is CORRELATED WITH THE OUTCOME: the
        # contract asks for the whole course to death, so a long survivor costs more tokens
        # than a short one and is rejected more often. That is informative censoring on the
        # endpoint being scored, not a cosmetic failure rate.
        self._max_tokens_override = max_tokens

        super().__init__(model_id)  # self.model_id = full ID with suffix
        import os
        import openai
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenRouter requires OPENROUTER_API_KEY environment variable"
            )
        site_url = os.environ.get("OPENROUTER_SITE_URL", "")
        app_title = os.environ.get("OPENROUTER_APP_TITLE", "SynthOncCases")
        headers: dict[str, str] = {"X-Title": app_title}
        if site_url:
            headers["HTTP-Referer"] = site_url
        self.client = openai.OpenAI(
            base_url=_OPENROUTER_BASE_URL,
            api_key=api_key,
            default_headers=headers,
            timeout=_openrouter_timeout_seconds(),
        )

    _MAX_TOKENS = 16384
    _MAX_COMPLETION_TOKENS = 16384

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        import openai

        last_error = None
        started = time.perf_counter()
        for attempt in range(PROVIDER_INTERNAL_MAX_ATTEMPTS):
            try:
                params, reasoning_mode = self._build_request_params(system_prompt, user_prompt)
                # Extract the token cap and its parameter name from the resolved params
                if "max_completion_tokens" in params:
                    max_tokens_sent = params["max_completion_tokens"]
                    max_tokens_param = "max_completion_tokens"
                else:
                    max_tokens_sent = params["max_tokens"]
                    max_tokens_param = "max_tokens"
                response = self.client.chat.completions.create(**params)
                choice = response.choices[0]
                usage = response.usage
                reasoning_tokens = 0
                if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                    reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
                cache_reported, cached_input_tokens, cache_write_tokens = (
                    _openrouter_prompt_cache_usage(usage)
                )
                # OpenRouter reports actual cost in usage.cost (USD)
                reported_cost = None
                if usage:
                    reported_cost = getattr(usage, "cost", None)
                    if reported_cost is not None:
                        try:
                            reported_cost = float(reported_cost)
                        except (TypeError, ValueError):
                            reported_cost = None
                result = GenerationResult(
                    text=choice.message.content or "",
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    reasoning_tokens=reasoning_tokens,
                    model_id=self.model_id,
                    response_model=getattr(response, "model", ""),
                    response_id=getattr(response, "id", ""),
                    served_provider=_openrouter_served_provider(response),
                    retries=attempt,
                    reported_cost=reported_cost,
                    max_tokens_sent=max_tokens_sent,
                    max_tokens_param=max_tokens_param,
                    reasoning_mode=reasoning_mode,
                    temperature_sent=params.get("temperature"),
                    top_p_sent=params.get("top_p"),
                    seed_sent=params.get("seed"),
                    finish_reason=str(getattr(choice, "finish_reason", "") or ""),
                    prompt_cache_usage_reported=cache_reported,
                    cached_input_tokens=cached_input_tokens,
                    cache_write_tokens=cache_write_tokens,
                    duration_s=time.perf_counter() - started,
                )
                self.usage.record(result)
                return result
            except (openai.APIError, openai.APIConnectionError) as e:
                # Strict mode: re-raise immediately on semantic / request-shape
                # errors that won't recover on retry. 408 / 429 / 5xx /
                # connection errors keep retrying as today.
                if (
                    self._require_parameters
                    and getattr(e, "status_code", None) in _STRICT_MODE_FATAL_STATUS_CODES
                ):
                    raise
                last_error = e
                if attempt < PROVIDER_INTERNAL_MAX_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"OpenRouter API failed after {PROVIDER_INTERNAL_MAX_ATTEMPTS} attempts: "
            f"{last_error}"
        )

    def _build_request_params(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[dict, str]:
        """Build chat completion params + a reasoning-mode label.

        Four reasoning cases:
        1. :reasoning suffix → reasoning on (effort=high) — "on"
        2. Paired reasoning model, no suffix → reasoning explicitly off — "forced_off"
        3. OpenAI reasoning family (gpt-5, o1, o3, o4) → reasoning off,
           max_completion_tokens, no temperature — "native_reasoning_family"
        4. Everything else → no reasoning body, standard params — "off"

        Strict-mode injections (when constructor args are set) layer on top
        of all four reasoning cases without disturbing them.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        bare = self._api_model_id.split("/")[-1].lower()
        is_openai_reasoning = any(
            bare.startswith(p) for p in ("gpt-5", "o1", "o3", "o4")
        )

        if self._reasoning_on:
            reasoning_body = {"effort": self._reasoning_effort}
            if self._reasoning_exclude:
                reasoning_body["exclude"] = True
            params: dict = {
                "model": self._api_model_id,
                "max_tokens": self._max_tokens_override or self._MAX_TOKENS,
                "temperature": DEFAULT_REQUEST_TEMPERATURE,
                "messages": messages,
                "extra_body": {"reasoning": reasoning_body},
            }
            mode = self._reasoning_mode_label
        elif self._api_model_id in _OPENROUTER_REASONING_MODELS:
            params = {
                "model": self._api_model_id,
                "max_tokens": self._max_tokens_override or self._MAX_TOKENS,
                "temperature": DEFAULT_REQUEST_TEMPERATURE,
                "messages": messages,
                "extra_body": {"reasoning": {"effort": "none"}},
            }
            mode = "forced_off"
        elif is_openai_reasoning:
            params = {
                "model": self._api_model_id,
                "max_completion_tokens": self._MAX_COMPLETION_TOKENS,
                "messages": messages,
                "extra_body": {"reasoning": {"effort": "none"}},
            }
            mode = "native_reasoning_family"
        else:
            params = {
                "model": self._api_model_id,
                "max_tokens": self._max_tokens_override or self._MAX_TOKENS,
                "temperature": DEFAULT_REQUEST_TEMPERATURE,
                "messages": messages,
            }
            mode = "off"

        # Strict-mode injections — both no-ops when their flag is off.
        # response_format goes top-level (OpenAI chat-completions param);
        # provider.require_parameters goes into extra_body (OpenRouter-only,
        # forces routing to a backend that honours all requested params).
        if self._response_format is not None:
            params["response_format"] = self._response_format
        if self._provider_policy:
            # Routing policy: which backends may serve this call, whether a fallback is
            # allowed, and the retention terms.
            extra_body = params.setdefault("extra_body", {})
            provider_block = extra_body.setdefault("provider", {})
            provider_block.update(self._provider_policy)
            # Ask for routing metadata on the response so the serving backend is known from
            # the call itself rather than from a later lookup.
            params.setdefault("extra_headers", {}).update(_OPENROUTER_METADATA_HEADER)
        if self._require_parameters:
            # LAST, and deliberately so. `require_parameters` is what makes OpenRouter refuse
            # a backend that will not honour the decoding settings, and it must not be
            # overridable by a routing policy -- the key is barred from provider_policy for
            # the same reason. Order here is the second guard, not the only one.
            extra_body = params.setdefault("extra_body", {})
            provider_block = extra_body.setdefault("provider", {})
            provider_block["require_parameters"] = True
        if self._disable_response_cache:
            # OpenRouter response caching returns the same completion verbatim for an
            # identical request. That is useful operationally and invalid for repeated
            # stochastic draws. This header does not disable provider prompt-prefix caching.
            # Merged, not assigned: assigning would silently drop the metadata header above.
            params.setdefault("extra_headers", {})["X-OpenRouter-Cache"] = "false"
        if self._session_id is not None:
            # OpenRouter consumes this routing key and pins subsequent requests to the
            # same provider endpoint. It is not model-visible prompt content.
            extra_body = params.setdefault("extra_body", {})
            extra_body["session_id"] = self._session_id

        return params, mode


_PROVIDER_MAP: dict[str, type[LLMProvider]] = {
    "claude": AnthropicProvider,
    "anthropic": AnthropicProvider,
    "gpt": OpenAIProvider,
    "openai": OpenAIProvider,
    "gemini": GoogleProvider,
    "google": GoogleProvider,
    "glm": KatherProvider,
    "z-ai": KatherProvider,
    "zai": KatherProvider,
    "zhipu": KatherProvider,
}


def get_provider(
    model_string: str,
    *,
    use_litellm: bool = False,
    use_openrouter: bool = False,
    response_format: dict | None = None,
    require_parameters: bool = False,
    disable_response_cache: bool = False,
    session_id: str | None = None,
    provider_policy: dict | None = None,
    max_tokens: int | None = None,
) -> LLMProvider:
    """Get an LLM provider from a model string.

    Args:
        model_string: Model ID like 'claude-sonnet-4-6', 'gpt-4o-2025-03-01',
            or OpenRouter format like 'anthropic/claude-sonnet-4-6'.
        use_litellm: If True, load credentials from ~/.config/litellm.env and
            route through the LiteLLM proxy. Deprecated — use use_openrouter.
        use_openrouter: If True, load credentials from ~/.config/openrouter.env
            and route through OpenRouter. Model IDs should use OpenRouter format
            with organization prefixes (e.g. anthropic/claude-sonnet-4-6).
        response_format: OpenRouter-only. Forwarded as a top-level chat-
            completions parameter; the caller supplies the exact dict (e.g.
            {"type": "json_schema", "json_schema": {...}}).
        require_parameters: OpenRouter-only. When True, sets
            extra_body.provider.require_parameters=True so OpenRouter routes
            only to a backend that honours all requested parameters, AND
            switches the OpenRouter provider into fail-fast mode for
            semantic 4xx errors (no retry on 400/401/403/404/422).
        disable_response_cache: OpenRouter-only. When True, sends
            ``X-OpenRouter-Cache: false`` so identical stochastic requests are
            independently generated. Provider prompt-prefix caching remains enabled.
        session_id: OpenRouter-only sticky-routing key. It is sent as request
            metadata, not model-visible content, and pins repeated requests to one
            provider endpoint so provider prompt caches can be reused.
        provider_policy: OpenRouter-only routing policy, merged into
            extra_body.provider. Keys are validated against the documented routing
            set. Use it to pin a backend (``{"only": ["decart"],
            "allow_fallbacks": False}``) and to declare retention terms
            (``{"data_collection": "deny", "zdr": True}``). Pinning matters
            scientifically, not just operationally: an unpinned run fans across
            backends that differ in whether they honour decoding parameters, and an
            arm-differential backend mixture turns a treatment contrast into a
            partly hardware contrast.
    """
    if use_litellm and use_openrouter:
        raise ValueError("Cannot use both --litellm and --openrouter. Choose one.")

    if (response_format is not None or require_parameters or disable_response_cache
            or session_id is not None or provider_policy is not None
            or max_tokens is not None) \
            and not use_openrouter:
        raise ValueError(
            "response_format, require_parameters, disable_response_cache, session_id and "
            "provider_policy are OpenRouter-only; "
            "set use_openrouter=True."
        )

    if use_openrouter:
        _load_openrouter_env()
        return OpenRouterProvider(
            model_string,
            response_format=response_format,
            require_parameters=require_parameters,
            disable_response_cache=disable_response_cache,
            session_id=session_id,
            provider_policy=provider_policy,
            max_tokens=max_tokens,
        )

    if use_litellm:
        _load_litellm_env()
        known_prefixes = (
            "claude",
            "gpt",
            "gemini",
            "meta-llama",
            "deepseek",
            "nemotron",
            "nova",
            "glm",
            "z-ai",
            "zai",
            "zhipu",
        )
        if not any(model_string.lower().startswith(p) for p in known_prefixes):
            raise ValueError(
                f"Unknown model: {model_string}. "
                f"Known prefixes for LiteLLM: {known_prefixes}"
            )
        return LiteLLMProvider(model_string)

    # Native SDK routing by model prefix
    model_lower = model_string.lower()
    for prefix, provider_cls in _PROVIDER_MAP.items():
        if model_lower.startswith(prefix):
            return provider_cls(model_string)
    raise ValueError(
        f"Unknown model: {model_string}. "
        f"Supported prefixes: {list(_PROVIDER_MAP.keys())}. "
        f"For other models, use --openrouter to route through OpenRouter."
    )
