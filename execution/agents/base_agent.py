"""
Shared base class for all agents.
Handles LLM interaction and common utilities.
"""

import os
import time

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception,
)

try:
    from google import genai
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base exception for LLM call failures."""
    pass


class ProviderError(LLMError):
    """A specific provider failed."""
    def __init__(self, provider: str, message: str, original_error: Exception = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"{provider}: {message}")


class AllProvidersFailedError(LLMError):
    """All providers in the cascade failed."""
    def __init__(self, errors: list):
        self.errors = errors
        messages = "; ".join(str(e) for e in errors)
        super().__init__(f"All providers failed: {messages}")


class LLMNotConfiguredError(LLMError):
    """Raised when no LLM provider is configured."""
    pass


# ---------------------------------------------------------------------------
# Transient error detection (typed exceptions + string fallback)
# ---------------------------------------------------------------------------

_TRANSIENT_TYPES: tuple | None = None  # Lazy-initialised on first call


def _build_transient_types() -> tuple:
    """Collect typed transient exceptions from installed provider SDKs."""
    types = []
    try:
        import openai
        types.extend([openai.RateLimitError, openai.APITimeoutError,
                      openai.APIConnectionError, openai.InternalServerError])
    except (ImportError, AttributeError):
        pass
    try:
        from google.api_core.exceptions import (
            ResourceExhausted, DeadlineExceeded, ServiceUnavailable,
            TooManyRequests, GatewayTimeout,
        )
        types.extend([ResourceExhausted, DeadlineExceeded,
                      ServiceUnavailable, TooManyRequests, GatewayTimeout])
    except (ImportError, AttributeError):
        pass
    try:
        import groq
        types.extend([groq.RateLimitError, groq.APITimeoutError,
                      groq.APIConnectionError, groq.InternalServerError])
    except (ImportError, AttributeError):
        pass
    return tuple(types)


# String-based fallback patterns for providers without typed exceptions
_TRANSIENT_SUBSTRINGS = [
    "rate limit",
    "rate_limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
]


def _is_transient(exc: Exception) -> bool:
    """Check if an exception is transient and worth retrying.

    Prefers typed exception checks from provider SDKs.
    Falls back to string matching for unknown exception types.
    """
    global _TRANSIENT_TYPES
    if _TRANSIENT_TYPES is None:
        _TRANSIENT_TYPES = _build_transient_types()

    # Type-based check (preferred — robust across message format changes)
    if _TRANSIENT_TYPES and isinstance(exc, _TRANSIENT_TYPES):
        return True

    # String-based fallback for unknown exception types
    msg = str(exc).lower()
    return any(s in msg for s in _TRANSIENT_SUBSTRINGS)


class BaseAgent:
    def __init__(self, role, persona, model=None, provider=None):
        if model is None:
            from execution.config import config
            model = config.models.DEFAULT_BASE_MODEL
        self.role = role
        self.persona = persona
        self.model_name = model
        self.provider = "unknown"

        if provider:
            # Per-agent provider override — skip auto-detection
            self.provider = provider
            self.api_key = self._get_key_for_provider(provider)
        else:
            self._get_api_key()

        self._setup_client()

    @staticmethod
    def _get_key_for_provider(provider: str):
        """Return the API key for an explicitly-requested provider."""
        mapping = {
            "groq": "GROQ_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "google": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        env_var = mapping.get(provider)
        return os.getenv(env_var) if env_var else None

    def _get_api_key(self):
        # Priority: Groq -> Vertex (Project) -> Gemini (Key) -> OpenAI
        if os.getenv("GROQ_API_KEY"):
            self.provider = "groq"
            self.api_key = os.getenv("GROQ_API_KEY")
        elif os.getenv("GOOGLE_CLOUD_PROJECT"):
            self.provider = "gemini"
            self.api_key = os.getenv("GOOGLE_CLOUD_API_KEY") or os.getenv("GOOGLE_API_KEY")
            self.project = os.getenv("GOOGLE_CLOUD_PROJECT")
            self.location = os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"
            self.vertexai = True
        elif os.getenv("GOOGLE_API_KEY"):
            self.provider = "gemini"
            self.api_key = os.getenv("GOOGLE_API_KEY")
            self.vertexai = False
        elif os.getenv("OPENAI_API_KEY"):
            self.provider = "openai"
            self.api_key = os.getenv("OPENAI_API_KEY")
        else:
            self.api_key = None

    def _setup_client(self):
        if self.provider == "groq" and OpenAI:
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.api_key
            )
            self.client_type = "groq"
        elif self.provider in ("gemini", "google") and genai:
            if getattr(self, 'vertexai', False) and not self.api_key:
                # Use ADC for Vertex if no key provided
                self.client = genai.Client(
                    vertexai=True,
                    project=self.project,
                    location=self.location
                )
            else:
                # Use API Key (optionally with Vertex)
                self.client = genai.Client(
                    api_key=self.api_key,
                    vertexai=getattr(self, 'vertexai', False)
                )
            self.client_type = "gemini"
        elif self.provider == "openai" and OpenAI:
            self.client = OpenAI(api_key=self.api_key)
            self.client_type = "openai"
        else:
            self.client_type = "unknown"

    # ------------------------------------------------------------------
    # Provider call with retry (Upgrade 2 — tenacity)
    # ------------------------------------------------------------------

    @staticmethod
    def _log_retry(retry_state):
        """Logging callback invoked before each retry sleep."""
        exc = retry_state.outcome.exception()
        attempt = retry_state.attempt_number
        print(f"[retry] Attempt {attempt} failed ({exc}), retrying...")

    def _call_provider_with_retry(self, provider_name, call_fn):
        """
        Execute *call_fn* with up to 2 retry attempts on transient errors
        using tenacity for exponential backoff with jitter.

        Raises ProviderError on permanent failure.
        """
        retrying = retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(0, 2),
            retry=retry_if_exception(_is_transient),
            before_sleep=self._log_retry,
            reraise=True,
        )
        try:
            return retrying(call_fn)()
        except Exception as e:
            raise ProviderError(
                provider_name,
                str(e),
                original_error=e,
            )

    # ------------------------------------------------------------------
    # Response validation (Upgrade 3)
    # ------------------------------------------------------------------

    def _validate_response(self, response: str, provider_name: str) -> str:
        """Validate that the LLM response is real content, not empty/error."""
        if not response or not response.strip():
            raise ProviderError(provider_name, "Empty response from model")
        stripped = response.strip()
        if len(stripped) < 20:
            raise ProviderError(provider_name, f"Response too short ({len(stripped)} chars)")
        return stripped

    # ------------------------------------------------------------------
    # Prompt caching
    # ------------------------------------------------------------------

    def _prepare_cached_messages(self, system_prompt: str, user_prompt: str) -> list:
        """Structure messages with cache-control hints for supported providers.

        Anthropic Claude supports explicit cache_control on message blocks.
        OpenAI auto-caches prompts >1024 tokens — no code changes needed.
        Other providers get the standard messages format.
        """
        if self.provider == "anthropic":
            # Claude supports cache_control on message content blocks
            return [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                },
                {"role": "user", "content": user_prompt}
            ]
        # OpenAI auto-caches; Groq/Gemini use standard format
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def call_llm(self, prompt, system_instruction=None, temperature=0.7,
                 system_prompt=None):
        """Standardized call to the underlying LLM.

        Args:
            prompt: The user/dynamic prompt content.
            system_instruction: Additional instructions appended to the
                default role/persona system prompt.
            temperature: Sampling temperature.
            system_prompt: Optional fully-formed static system prompt.
                When provided, this replaces the auto-generated
                role/persona preamble entirely. Use this for agents with
                long, stable persona text so providers can cache it.

        Raises:
            LLMNotConfiguredError: No provider configured.
            ProviderError: The configured provider failed after retries.
        """
        if self.client_type == "unknown":
            raise LLMNotConfiguredError("No LLM provider is configured.")

        if system_prompt is not None:
            # Caller supplied a complete static system prompt (cacheable).
            full_system_prompt = system_prompt
            if system_instruction:
                full_system_prompt += f"\nSpecific Instructions:\n{system_instruction}"
        else:
            # Default: build from role + persona
            full_system_prompt = f"You are the {self.role}.\nPersona: {self.persona}\n"
            if system_instruction:
                full_system_prompt += f"\nSpecific Instructions:\n{system_instruction}"

        # Build messages — uses cache-control hints when the provider supports it
        messages = self._prepare_cached_messages(full_system_prompt, prompt)

        if self.client_type == "groq":
            def _groq_call():
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature
                )
                return resp.choices[0].message.content

            result = self._call_provider_with_retry("Groq", _groq_call)
            return self._validate_response(result, "Groq")

        elif self.client_type == "gemini":
            def _gemini_call():
                from google.genai import types
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=full_system_prompt,
                        temperature=temperature
                    )
                )
                return resp.text

            result = self._call_provider_with_retry("Gemini", _gemini_call)
            return self._validate_response(result, "Gemini")

        elif self.client_type == "openai":
            def _openai_call():
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature
                )
                return resp.choices[0].message.content

            result = self._call_provider_with_retry("OpenAI", _openai_call)
            return self._validate_response(result, "OpenAI")

        raise ProviderError(self.provider, "Unsupported client type")

    def generate(self, prompt: str, expect_json: bool = False,
                 system_prompt: str = None):
        """Generate LLM response, optionally parsing as JSON.

        Convenience wrapper around call_llm() for agents that need
        structured JSON output.

        Args:
            prompt: The user/dynamic prompt content.
            expect_json: If True, parse the response as JSON.
            system_prompt: Optional static system prompt for caching.
                Passed through to call_llm().
        """
        response = self.call_llm(prompt, system_prompt=system_prompt)
        if expect_json:
            from execution.utils.json_parser import extract_json_from_llm
            return extract_json_from_llm(response, default={})
        return response

    # ------------------------------------------------------------------
    # Async variants for parallel execution
    # ------------------------------------------------------------------

    async def call_llm_async(self, prompt: str, **kwargs) -> str:
        """Async variant of call_llm for parallel execution.

        Uses run_in_executor to run the sync call_llm in a thread pool,
        enabling real parallelism for I/O-bound LLM calls without
        requiring async provider SDK setup.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.call_llm(prompt, **kwargs)
        )

    async def generate_async(self, prompt: str, expect_json: bool = False,
                             system_prompt: str = None):
        """Async variant of generate() for parallel execution."""
        response = await self.call_llm_async(prompt, system_prompt=system_prompt)
        if expect_json:
            from execution.utils.json_parser import extract_json_from_llm
            return extract_json_from_llm(response, default={})
        return response
