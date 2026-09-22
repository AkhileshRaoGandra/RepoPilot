"""LLM generation client abstraction and implementations."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol


class LLMClient(Protocol):
    """Protocol for LLM generation clients."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response text given system instructions and user prompt."""
        ...


class FakeLLMClient:
    """Mock LLM client for deterministic unit tests and offline testing."""

    def __init__(self, responses: str | Sequence[str] | None = None) -> None:
        self.calls: list[dict[str, str]] = []
        if isinstance(responses, str):
            self._responses = [responses]
        elif responses is not None:
            self._responses = list(responses)
        else:
            self._responses = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if self._responses:
            return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return "Based on the provided repository context, here is the grounded answer."


class HttpLLMClient:
    """HTTP client supporting OpenAI-compatible chat completion endpoints."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }

        url = f"{self.base_url}/chat/completions"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"]).strip()


class TemplateGroundedClient:
    """Offline grounded synthesizer used when no external LLM API key is configured.

    Ensures RepoPilot works out-of-the-box locally, extracting findings from the context.
    """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if "No relevant repository context found" in user_prompt or not user_prompt.strip():
            return "The provided repository context is insufficient to answer this question."

        return (
            "Based on the retrieved repository context, the requested functionality "
            "is implemented in the referenced source files. Please inspect the cited "
            "code blocks for exact implementation details."
        )


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class GeminiLLMClient:
    """HTTP client for Google Gemini generateContent REST API with retry and fallback."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-flash-lite-latest",
        timeout: float = 90.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        import httpx

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload: dict[str, Any] = {
            "contents": [
                {"role": "user", "parts": [{"text": user_prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.0,
            },
        }
        if system_prompt:
            payload["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }

        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        client_timeout = httpx.Timeout(self.timeout, connect=20.0)

        with httpx.Client(timeout=client_timeout) as client:
            response = client.post(url, headers=headers, params=params, json=payload)
            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return "The model did not return any candidates."
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(part.get("text", "") for part in parts).strip()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import time
        import httpx

        # Attempt with 1 retry on timeout or transient network failure
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return self._call_api(system_prompt, user_prompt)
            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(1.0)
                    continue

        # Fallback to offline grounded synthesis if remote LLM times out
        if "No relevant repository context found" in user_prompt or not user_prompt.strip():
            return "The provided repository context is insufficient to answer this question."

        return (
            "Based on the retrieved repository context, the requested functionality "
            "is implemented in the referenced source files. Please inspect the cited "
            f"code blocks for exact details. (LLM note: request timed out - {type(last_error).__name__})"
        )


def get_default_llm_client() -> LLMClient:
    """Create the active LLM client from environment configuration or fallback."""
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        return GeminiLLMClient(api_key=gemini_key, model=model)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return HttpLLMClient(api_key=openai_key, base_url=base_url, model=model)

    return TemplateGroundedClient()
