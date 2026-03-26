"""
simulator/providers.py — Provider abstraction + Anthropic/HF implementations.

Defines a common Provider protocol and concrete implementations for
Anthropic and Hugging Face, plus a factory function.
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, runtime_checkable

import anthropic
import requests

from simulator.tracking import UsageTracker


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CallResult:
    text: str
    input_tokens: int
    output_tokens: int


class ProviderError(Exception):
    """Raised when a provider API call fails after retries."""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Provider(Protocol):
    def call(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        max_tokens: int = 1024,
    ) -> CallResult: ...


# ---------------------------------------------------------------------------
# Anthropic implementation
# ---------------------------------------------------------------------------


class AnthropicProvider:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
        tracker: Optional[UsageTracker] = None,
    ) -> None:
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Anthropic API key found. Please either:\n"
                "1. Create a .env file with: ANTHROPIC_API_KEY=your-key-here\n"
                "2. Set environment variable: export ANTHROPIC_API_KEY=your-key-here\n"
                "3. Pass api_key when constructing AnthropicProvider()"
            )
        self._client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model
        self._tracker = tracker

    def call(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        max_tokens: int = 1024,
        call_type: str = "conversation",
        persona_name: str = "",
        question_id: Optional[str] = None,
    ) -> CallResult:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
        )
        text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        if self._tracker is not None:
            self._tracker.record(call_type, persona_name, input_tokens, output_tokens, question_id)

        return CallResult(text=text, input_tokens=input_tokens, output_tokens=output_tokens)


# ---------------------------------------------------------------------------
# Hugging Face implementation
# ---------------------------------------------------------------------------


class HuggingFaceProvider:
    _API_URL = "https://router.huggingface.co/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "Qwen/Qwen3-4B-Instruct-2507:nscale",
        retry_wait: int = 20,
        tracker: Optional[UsageTracker] = None,
    ) -> None:
        resolved_key = api_key or os.getenv("HF_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Hugging Face API key found. Please either:\n"
                "1. Create a .env file with: HF_API_KEY=your-key-here\n"
                "2. Set environment variable: export HF_API_KEY=your-key-here\n"
                "3. Pass api_key when constructing HuggingFaceProvider()"
            )
        self._headers = {
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        }
        self.model = model
        self.retry_wait = retry_wait
        self._tracker = tracker

    def call(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        max_tokens: int = 1024,
        call_type: str = "conversation",
        persona_name: str = "",
        question_id: Optional[str] = None,
    ) -> CallResult:
        chat_messages = [{"role": "system", "content": system_prompt}] + [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        payload = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        response = requests.post(self._API_URL, headers=self._headers, json=payload, timeout=60)

        if response.status_code == 503:
            print(f"⏳ Model is loading, waiting {self.retry_wait} seconds...")
            time.sleep(self.retry_wait)
            response = requests.post(self._API_URL, headers=self._headers, json=payload, timeout=60)

        if response.status_code != 200:
            raise ProviderError(
                f"Hugging Face API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        text = result["choices"][0]["message"]["content"].strip()
        usage = result.get("usage", {})
        input_tokens = usage.get("prompt_tokens", len(str(chat_messages)) // 4)
        output_tokens = usage.get("completion_tokens", len(text) // 4)

        if self._tracker is not None:
            self._tracker.record(call_type, persona_name, input_tokens, output_tokens, question_id)

        return CallResult(text=text, input_tokens=input_tokens, output_tokens=output_tokens)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_VALID_PROVIDERS = ("anthropic", "huggingface", "tinkr")


def get_provider(name: str, **kwargs) -> Provider:
    """Return a Provider instance for *name*.

    Valid names: 'anthropic', 'huggingface', 'tinkr'.
    Raises ValueError with the list of valid names if *name* is unknown.
    """
    if name == "anthropic":
        return AnthropicProvider(**kwargs)
    if name == "huggingface":
        return HuggingFaceProvider(**kwargs)
    if name == "tinkr":
        from train.tinkr_provider import TinkrProvider
        return TinkrProvider(**kwargs)
    raise ValueError(
        f"Unknown provider {name!r}. Valid providers: {', '.join(_VALID_PROVIDERS)}"
    )
