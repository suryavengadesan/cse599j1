"""
simulator/generator.py — LLM-powered scenario generator.

Uses the existing provider abstraction to generate complete scenario YAML
files from a short topic description.

Usage (standalone):
    python -m simulator.generator "remote work vs office work"
    python -m simulator.generator "cats vs dogs" --provider anthropic --output scenarios/cats-dogs.yaml

Usage (as library):
    from simulator.generator import ScenarioGenerator
    gen = ScenarioGenerator(provider="anthropic")
    gen.generate("remote work vs office work", output_path="scenarios/remote-office.yaml")
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from simulator.providers import get_provider

load_dotenv()

_SCENARIOS_DIR = Path("scenarios")

_GENERATION_PROMPT = """\
You are a scenario designer for a conversation simulation framework.

Given a debate topic, generate a complete scenario YAML structure as a JSON object.
The scenario simulates two personas having a conversation where one tries to
persuade the other.

Requirements:
- "name": a short kebab-case slug derived from the topic (e.g. "remote-vs-office")
- "persona_a": the persona who holds the initial position. Has "name" and "preference".
- "persona_b": the opposing persona. Has "name" and "preference".
- "persona_a_adversarial": same as persona_a (copy name and preference).
- "persona_b_adversarial": has "name" and "strategy" — a detailed persuasion strategy
  (3-5 sentences) for convincing persona_a to change their mind.
- "survey": has "title" and "questions" (q1 through q4). Each question has a
  "question" string and "options" with keys A, B, C, D ranging from strongly
  favoring persona_a's position (A) to strongly favoring persona_b's position (D).
- "initial_message": a 2-3 sentence opening message from persona_a expressing
  their current position enthusiastically.

Respond with ONLY a valid JSON object, no markdown fences, no commentary."""


class ScenarioGenerator:
    """Generate scenario YAML files from a topic description using an LLM."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
    ) -> None:
        kwargs = {}
        if model:
            kwargs["model"] = model
        self._provider = get_provider(provider, **kwargs)

    def generate(
        self,
        topic: str,
        output_path: Optional[str] = None,
        scenario_name: Optional[str] = None,
    ) -> Path:
        """Generate a scenario YAML from a topic and write it to disk.

        Parameters
        ----------
        topic:
            A short description of the debate topic, e.g. "remote work vs office".
        output_path:
            Where to write the YAML. If None, derives from the generated/provided name.
        scenario_name:
            Force a specific scenario name (and filename). If None, the LLM picks one.

        Returns
        -------
        Path to the written YAML file.
        """
        result = self._provider.call(
            messages=[{"role": "user", "content": f"Topic: {topic}"}],
            system_prompt=_GENERATION_PROMPT,
            max_tokens=2048,
        )

        data = self._parse_response(result.text)
        self._validate(data)

        # Override the name if the caller specified one
        if scenario_name:
            data["name"] = scenario_name

        if output_path is None:
            name = data["name"]
            _SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
            output_path = _SCENARIOS_DIR / f"{name}.yaml"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

        return output_path

    @staticmethod
    def _parse_response(text: str) -> dict:
        """Extract JSON from the LLM response, tolerating markdown fences."""
        cleaned = text.strip()
        # Strip markdown code fences if present
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse LLM response as JSON: {exc}\nResponse was:\n{text}"
            ) from exc

    @staticmethod
    def _validate(data: dict) -> None:
        """Ensure the generated data has all required fields."""
        required = ("name", "persona_a", "persona_b", "survey", "initial_message")
        for field in required:
            if field not in data:
                raise ValueError(f"Generated scenario is missing required field: {field!r}")

        survey = data["survey"]
        if "title" not in survey or "questions" not in survey:
            raise ValueError("Survey must contain 'title' and 'questions'")

        questions = survey["questions"]
        if not questions:
            raise ValueError("Survey must have at least one question")

        for qid, q in questions.items():
            if "question" not in q or "options" not in q:
                raise ValueError(f"Survey question {qid!r} missing 'question' or 'options'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a new scenario YAML from a topic description"
    )
    parser.add_argument("topic", help="Debate topic, e.g. 'remote work vs office work'")
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "huggingface"],
        help="LLM provider to use (default: anthropic)",
    )
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument(
        "--output", "-o", default=None, help="Output path (default: scenarios/<name>.yaml)"
    )

    args = parser.parse_args()

    print(f"Generating scenario for: {args.topic}")
    gen = ScenarioGenerator(provider=args.provider, model=args.model)

    try:
        path = gen.generate(args.topic, output_path=args.output)
    except (ValueError, Exception) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Scenario written to: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
