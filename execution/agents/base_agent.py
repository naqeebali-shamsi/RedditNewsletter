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


# Transient error types worth retrying (rate limits, timeouts, server errors)
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
    """Return True if the exception looks transient and worth retrying."""
    msg = str(exc).lower()
    return any(s in msg for s in _TRANSIENT_SUBSTRINGS)


class BaseAgent:
    def __init__(self, role, persona, model="gemini-2.0-flash-exp", provider=None):
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
    # Main entry point
    # ------------------------------------------------------------------

    def call_llm(self, prompt, system_instruction=None, temperature=0.7):
        """Standardized call to the underlying LLM.

        Raises:
            LLMNotConfiguredError: No provider configured.
            ProviderError: The configured provider failed after retries.
        """
        if self.client_type == "unknown":
            raise LLMNotConfiguredError("No LLM provider is configured.")

        full_system_prompt = f"You are the {self.role}.\nPersona: {self.persona}\n"
        if system_instruction:
            full_system_prompt += f"\nSpecific Instructions:\n{system_instruction}"

        if self.client_type == "groq":
            def _groq_call():
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": full_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
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
                    messages=[
                        {"role": "system", "content": full_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )
                return resp.choices[0].message.content

            result = self._call_provider_with_retry("OpenAI", _openai_call)
            return self._validate_response(result, "OpenAI")

        raise ProviderError(self.provider, "Unsupported client type")
