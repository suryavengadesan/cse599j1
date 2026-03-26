"""
simulator/tinkr_provider.py — Tinkr inference provider for fine-tuned models.

Uses the Tinker API's sampling interface to serve completions from
LoRA-adapted Qwen3 models trained via the RLHF pipeline.
"""

import os
from typing import Dict, List, Optional

from simulator.providers import CallResult, ProviderError
from simulator.tracking import UsageTracker

try:
    from tinker import ServiceInterface, TrainingClient
except ImportError:
    ServiceInterface = None  # type: ignore[assignment,misc]
    TrainingClient = None  # type: ignore[assignment,misc]


class TinkrProvider:
    """Provider that samples from a Tinker-hosted fine-tuned model.

    Can operate in two modes:
      1. **Checkpoint mode** — load a specific LoRA checkpoint by path.
      2. **Base mode** — use a base model name without fine-tuned weights.

    Both modes use Tinker's ``sample`` API for generation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "Qwen/Qwen3-4B-Instruct",
        checkpoint_path: Optional[str] = None,
        tracker: Optional[UsageTracker] = None,
    ) -> None:
        if ServiceInterface is None:
            raise ImportError(
                "The 'tinker' package is required for TinkrProvider. "
                "Install it with: pip install tinker"
            )

        resolved_key = api_key or os.getenv("TINKER_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Tinker API key found. Please either:\n"
                "1. Create a .env file with: TINKER_API_KEY=your-key-here\n"
                "2. Set environment variable: export TINKER_API_KEY=your-key-here\n"
                "3. Pass api_key when constructing TinkrProvider()"
            )

        os.environ["TINKER_API_KEY"] = resolved_key
        self._service = ServiceInterface()
        self.model = model
        self._tracker = tracker

        self._client = TrainingClient(model_name=model)
        if checkpoint_path:
            self._client.load_weights(checkpoint_path)

    def call(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        max_tokens: int = 1024,
        call_type: str = "conversation",
        persona_name: str = "",
        question_id: Optional[str] = None,
    ) -> CallResult:
        """Generate a completion via Tinker's sample API."""
        # Format as chat template: system + messages
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        result = self._client.sample(
            prompts=[prompt],
            max_new_tokens=max_tokens,
            temperature=0.7,
        )

        text = result[0].strip() if result else ""

        # Estimate tokens from character counts (Tinker doesn't always
        # return usage stats during sampling)
        input_tokens = len(prompt) // 4
        output_tokens = len(text) // 4

        if self._tracker is not None:
            self._tracker.record(
                call_type, persona_name, input_tokens, output_tokens, question_id
            )

        return CallResult(text=text, input_tokens=input_tokens, output_tokens=output_tokens)
