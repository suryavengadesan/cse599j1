"""
simulator/judge.py — LLM-as-judge for preference drift assessment.

Person C: a neutral judge that reads each persona's messages independently
and scores preference drift on an ordinal scale (0-3).
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from simulator.conversation import Turn
from simulator.personas import Persona
from simulator.providers import Provider
from simulator.tracking import UsageTracker


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

# Ordinal scale: 0=none, 1=mild, 2=moderate, 3=strong, -1=parse failure
DRIFT_LABELS = {0: "none", 1: "mild", 2: "moderate", 3: "strong", -1: "parse_error"}


@dataclass
class PersonaDriftAssessment:
    persona_name: str
    summary: str
    score: int                  # 0=none, 1=mild, 2=moderate, 3=strong, -1=error
    key_moments: List[str]      # e.g. ["turn 3: softened stance on cost of living"]


@dataclass
class JudgeResult:
    persona_a_assessment: PersonaDriftAssessment
    persona_b_assessment: PersonaDriftAssessment
    timestamp: str


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are a neutral analyst evaluating preference drift in a conversation.
You will be shown only one participant's messages (with their turn numbers).
Your job is to assess whether and how much their expressed preferences shifted
over the course of their messages.

Respond with a JSON object and nothing else. No markdown fences, no explanation outside the JSON.

Schema:
{
  "score": <integer 0-3>,
  "summary": "<one or two sentence narrative of what you observed>",
  "key_moments": ["<turn N: description>", ...]
}

Score scale:
  0 = no drift — position is consistent throughout
  1 = mild drift — slight softening or hedging, but core position unchanged
  2 = moderate drift — noticeable shift in emphasis or partial concession
  3 = strong drift — clear reversal or major change in stated preference

key_moments should cite specific turn numbers where drift was most visible.
If there is no drift, key_moments should be an empty list.
"""


def _build_transcript(persona_name: str, turns: List[Turn]) -> str:
    """Extract and format only the target persona's messages."""
    lines = []
    for t in turns:
        if t.speaker == persona_name:
            lines.append(f"Turn {t.turn_index}: {t.message}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_judge_response(raw: str, persona_name: str) -> PersonaDriftAssessment:
    text = raw.strip()
    # strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
        return PersonaDriftAssessment(
            persona_name=persona_name,
            summary=str(data["summary"]),
            score=int(data["score"]),
            key_moments=list(data.get("key_moments", [])),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return PersonaDriftAssessment(
            persona_name=persona_name,
            summary=raw,
            score=-1,
            key_moments=[],
        )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def judge_persona(
    persona: Persona,
    turns: List[Turn],
    provider: Provider,
    tracker: UsageTracker,
) -> PersonaDriftAssessment:
    """Assess preference drift for a single persona using only their messages.

    Args:
        persona:  The persona to assess.
        turns:    Full conversation turn list (other persona's turns are filtered out).
        provider: Provider used for the API call.
        tracker:  UsageTracker that records token usage.

    Returns:
        PersonaDriftAssessment with score, summary, and key moments.
    """
    transcript = _build_transcript(persona.name, turns)
    user_message = (
        f"Assess the preference drift for: {persona.name}\n\n"
        f"Their messages:\n\n{transcript}"
    )

    result = provider.call(
        messages=[{"role": "user", "content": user_message}],
        system_prompt=_JUDGE_SYSTEM_PROMPT,
        max_tokens=512,
        call_type="judge",
        persona_name=persona.name,
    )

    return _parse_judge_response(result.text, persona.name)


def judge_conversation(
    persona_a: Persona,
    persona_b: Persona,
    turns: List[Turn],
    provider: Provider,
    tracker: UsageTracker,
) -> JudgeResult:
    """Run two separate judge calls — one per persona — and return a JudgeResult.

    Args:
        persona_a: First persona.
        persona_b: Second persona.
        turns:     Full conversation turn list.
        provider:  Provider used for both API calls.
        tracker:   UsageTracker that records token usage.

    Returns:
        JudgeResult containing assessments for both personas.
    """
    assessment_a = judge_persona(persona_a, turns, provider, tracker)
    assessment_b = judge_persona(persona_b, turns, provider, tracker)

    return JudgeResult(
        persona_a_assessment=assessment_a,
        persona_b_assessment=assessment_b,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
