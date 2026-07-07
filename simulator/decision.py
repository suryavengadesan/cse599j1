"""
simulator/decision.py — End-of-conversation match decisions for interviews.

Unlike the LLM judge (Person C, a neutral outside analyst), the match decision
is made *by each agent about itself*, in character. After an interview
conversation, the worker (persona_a) and the firm interviewer (persona_b) each
independently decide whether they want to MATCH — i.e. move forward together.

A match is mutual and binary: it happens only if BOTH sides choose to match,
mirroring two-sided matching markets (deferred acceptance). Each side decides
privately, without seeing the other's decision.
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
# Defaults (overridable per-scenario via the YAML ``decision:`` block)
# ---------------------------------------------------------------------------

DEFAULT_WORKER_ROLE = "a candidate interviewing for a role at the firm"
DEFAULT_FIRM_ROLE = "an interviewer at the firm evaluating the candidate"
DEFAULT_WORKER_MATCH_MEANING = "you would accept an offer to join the firm in this role"
DEFAULT_FIRM_MATCH_MEANING = "you would extend an offer to hire this candidate"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class MatchDecision:
    persona_name: str
    role: str
    wants_match: Optional[bool]   # True / False, or None if the response failed to parse
    reasoning: str


@dataclass
class MatchResult:
    worker_decision: MatchDecision
    firm_decision: MatchDecision
    mutual_match: Optional[bool]  # True/False; None if either side failed to parse
    timestamp: str


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_DECISION_SYSTEM_PROMPT_TEMPLATE = """\
You are {name}.
{identity}

Your role in this interview: {role}

You have just finished an interview conversation with {counterpart}. Now, privately
and independently, decide whether you want to MATCH with them — that is, whether
{match_meaning}.

Base your decision only on what you learned during this interview together with
your own priorities, requirements, and constraints. Do not assume anything the
other party did not actually convey. The other party is making their own decision
separately and cannot see yours; a match only happens if BOTH of you choose to match.

Respond with a JSON object and nothing else. No markdown fences, no text outside the JSON.

Schema:
{{
  "wants_match": <true or false>,
  "reasoning": "<one to three sentences explaining your decision>"
}}
"""


def _persona_identity(persona: Persona) -> str:
    """Summarise a persona's (possibly private) identity for the decision prompt."""
    parts: List[str] = []
    if persona.background:
        parts.append(f"Background: {persona.background}")
    if persona.personality:
        parts.append(f"Personality: {persona.personality}")
    if persona.goals:
        parts.append(f"What you are looking for: {persona.goals}")
    if persona.preference:
        parts.append(f"Preference: {persona.preference}")
    return "\n".join(parts) if parts else "(no additional details provided)"


def _build_full_transcript(turns: List[Turn]) -> str:
    """Format the entire conversation as readable ``Speaker: message`` lines."""
    return "\n\n".join(f"{t.speaker}: {t.message}" for t in turns)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _coerce_bool(value) -> Optional[bool]:
    """Best-effort conversion of a JSON value to a bool. None if unrecognised."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "match", "y", "1"):
            return True
        if v in ("false", "no", "reject", "pass", "n", "0"):
            return False
    return None


def _parse_decision(raw: str, persona_name: str, role: str) -> MatchDecision:
    text = raw.strip()
    # strip markdown code fences if present (mirrors judge.py)
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
        wants = _coerce_bool(data["wants_match"])
        if wants is None:
            raise ValueError("unrecognised wants_match value")
        return MatchDecision(
            persona_name=persona_name,
            role=role,
            wants_match=wants,
            reasoning=str(data.get("reasoning", "")),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return MatchDecision(
            persona_name=persona_name,
            role=role,
            wants_match=None,
            reasoning=raw,
        )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def decide_match(
    persona: Persona,
    role: str,
    counterpart_name: str,
    match_meaning: str,
    turns: List[Turn],
    provider: Provider,
    tracker: UsageTracker,
) -> MatchDecision:
    """Ask a single persona, in character, whether it wants to match.

    Args:
        persona:          The persona making the decision (its private info is used).
        role:             Human-readable description of this persona's interview role.
        counterpart_name: Name of the other party in the interview.
        match_meaning:    What choosing to match means for this persona.
        turns:            Full conversation transcript.
        provider:         Provider used for the API call.
        tracker:          UsageTracker that records token usage.

    Returns:
        MatchDecision with ``wants_match`` (None on parse failure) and reasoning.
    """
    system = _DECISION_SYSTEM_PROMPT_TEMPLATE.format(
        name=persona.name,
        identity=_persona_identity(persona),
        role=role,
        counterpart=counterpart_name,
        match_meaning=match_meaning,
    )
    transcript = _build_full_transcript(turns)
    user_message = (
        f"Here is the full interview transcript:\n\n{transcript}\n\n"
        f"Now make your private match decision as {persona.name}."
    )

    result = provider.call(
        messages=[{"role": "user", "content": user_message}],
        system_prompt=system,
        max_tokens=512,
        call_type="decision",
        persona_name=persona.name,
    )

    return _parse_decision(result.text, persona.name, role)


def collect_match_decisions(
    persona_a: Persona,
    persona_b: Persona,
    turns: List[Turn],
    provider: Provider,
    tracker: UsageTracker,
    worker_role: str = DEFAULT_WORKER_ROLE,
    firm_role: str = DEFAULT_FIRM_ROLE,
    worker_match_meaning: str = DEFAULT_WORKER_MATCH_MEANING,
    firm_match_meaning: str = DEFAULT_FIRM_MATCH_MEANING,
) -> MatchResult:
    """Collect independent match decisions from the worker and the firm.

    persona_a is treated as the worker/candidate; persona_b as the firm interviewer.
    Each decides privately; ``mutual_match`` is True only if both choose to match.

    Returns:
        MatchResult bundling both decisions and the mutual outcome.
    """
    worker = decide_match(
        persona_a, worker_role, persona_b.name, worker_match_meaning,
        turns, provider, tracker,
    )
    firm = decide_match(
        persona_b, firm_role, persona_a.name, firm_match_meaning,
        turns, provider, tracker,
    )

    if worker.wants_match is None or firm.wants_match is None:
        mutual: Optional[bool] = None
    else:
        mutual = worker.wants_match and firm.wants_match

    return MatchResult(
        worker_decision=worker,
        firm_decision=firm,
        mutual_match=mutual,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
