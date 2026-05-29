from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GroqProviderError(RuntimeError):
    reason_code: str
    safe_message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.safe_message


_JSON_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)


def _extract_json_text(content: str) -> str:
    text = (content or "").strip()
    if not text:
        raise GroqProviderError("empty_response", "Provider returned an empty response.")

    if text.startswith("```"):
        text = _JSON_FENCE_PATTERN.sub("", text).strip()

    if not text:
        raise GroqProviderError("empty_response", "Provider returned an empty response.")

    return text


class GroqLLMProvider:
    """Groq SDK-backed provider for Stage2 LLMOrchestrator protocol."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model_name: str | None,
        temperature: float = 0.1,
        max_tokens: int = 1200,
        timeout_seconds: float = 20.0,
    ) -> None:
        key = str(api_key or "").strip()
        if not key:
            raise GroqProviderError("client_init_error", "GROQ_API_KEY is missing for provider initialization.")

        self._api_key = key
        self._model_name = model_name or "llama-3.3-70b-versatile"
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._timeout_seconds = float(timeout_seconds)

    def _build_client(self):
        try:
            from groq import Groq
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise GroqProviderError("sdk_missing", "Groq SDK is not available.") from exc

        try:
            return Groq(api_key=self._api_key)
        except Exception as exc:  # pragma: no cover - defensive
            raise GroqProviderError("client_init_error", "Failed to initialize provider client.") from exc

    async def generate_json(self, prompt: str, schema_name: str) -> Mapping[str, Any]:
        client = self._build_client()
        system_instruction = (
            "Return only one valid JSON object. "
            "Do not wrap in markdown. "
            "Do not include any text outside JSON. "
            "Do not include chain_of_thought, reasoning_chain, thought, scratchpad, or internal_reasoning. "
            f"The JSON must match schema: {schema_name}."
        )

        try:
            response = client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            try:
                from groq import APIConnectionError, APIStatusError, AuthenticationError, BadRequestError, RateLimitError
            except Exception:  # pragma: no cover - environment-dependent
                APIConnectionError = APIStatusError = AuthenticationError = BadRequestError = RateLimitError = tuple()  # type: ignore[assignment]

            if AuthenticationError and isinstance(exc, AuthenticationError):
                raise GroqProviderError("auth_error", "Provider authentication failed.") from exc
            if RateLimitError and isinstance(exc, RateLimitError):
                raise GroqProviderError("rate_limit", "Provider rate limit exceeded.") from exc
            if BadRequestError and isinstance(exc, BadRequestError):
                raise GroqProviderError("model_error", "Provider rejected model or request parameters.") from exc
            if APIStatusError and isinstance(exc, APIStatusError):
                status_code = getattr(exc, "status_code", None)
                if status_code in {401, 403}:
                    raise GroqProviderError("auth_error", "Provider authentication failed.") from exc
                if status_code == 429:
                    raise GroqProviderError("rate_limit", "Provider rate limit exceeded.") from exc
                if status_code in {400, 404, 422}:
                    raise GroqProviderError("model_error", "Provider rejected model or request parameters.") from exc
                if status_code == 408:
                    raise GroqProviderError("timeout", "Provider request timed out.") from exc
                raise GroqProviderError("unknown_api_error", "Provider API request failed.") from exc
            if APIConnectionError and isinstance(exc, APIConnectionError):
                raise GroqProviderError("network_call_error", "Provider network call failed.") from exc
            if isinstance(exc, TimeoutError):
                raise GroqProviderError("timeout", "Provider request timed out.") from exc
            raise GroqProviderError("unknown_api_error", "Provider API request failed.") from exc

        content = (
            getattr(response, "choices", [{}])[0]
            .message
            .content
            if getattr(response, "choices", None)
            else ""
        )
        if not isinstance(content, str):
            raise GroqProviderError("schema_unavailable", "Provider response content was unavailable.")

        json_text = _extract_json_text(content)
        try:
            data = json.loads(json_text)
        except Exception as exc:
            raise GroqProviderError("invalid_json", "Provider returned invalid JSON content.") from exc

        if not isinstance(data, dict):
            raise GroqProviderError("invalid_json", "Provider JSON payload was not an object.")
        return data
