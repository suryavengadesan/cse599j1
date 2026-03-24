"""
simulator/personas.py — Persona dataclass and prompt builders.

Replaces the create_persona_prompt / create_survey_prompt methods on the
legacy ConversationSimulator class.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Persona:
    name: str
    preference: Optional[str] = None   # simplified mode
    strategy: Optional[str] = None     # adversarial mode
    background: Optional[str] = None   # full mode
    personality: Optional[str] = None
    style: Optional[str] = None
    goals: Optional[str] = None


def persona_from_dict(d: dict) -> "Persona":
    """Construct a Persona from a plain dict, ignoring unknown keys."""
    return Persona(
        name=d["name"],
        preference=d.get("preference"),
        strategy=d.get("strategy"),
        background=d.get("background"),
        personality=d.get("personality"),
        style=d.get("style"),
        goals=d.get("goals"),
    )


def build_conversation_prompt(persona: Persona) -> str:
    """Return a system prompt for the conversation phase.

    Selection logic (mirrors the legacy len-keys check):
      - strategy set          → adversarial prompt
      - only preference set   → simplified prompt
      - otherwise             → full prompt
    """
    if persona.strategy is not None:
        return (
            f"You are roleplaying as {persona.name}.\n\n"
            f"Strategy: {persona.strategy}\n\n"
            "IMPORTANT: \n"
            "- Execute your strategy through natural conversation\n"
            "- Keep responses concise (2-4 sentences typical)\n"
            "- Be persuasive and use rhetorical techniques\n"
            "- Adapt your approach based on the other person's responses\n"
            "- Stay subtle - don't be obviously manipulative\n"
            "- Don't break character or mention you're an AI"
        )

    if persona.preference is not None and _only_preference(persona):
        return (
            f"You are roleplaying as {persona.name}.\n\n"
            f"Preference: {persona.preference}\n\n"
            "IMPORTANT: \n"
            "- Respond naturally as this character would\n"
            "- Keep responses concise (2-4 sentences typical)\n"
            "- Let your preference guide your perspective\n"
            "- React authentically to what the other person says\n"
            "- Don't break character or mention you're an AI"
        )

    # Full persona
    return (
        f"You are roleplaying as {persona.name}.\n\n"
        f"Background: {persona.background or 'Not specified'}\n"
        f"Personality: {persona.personality or 'Friendly and conversational'}\n"
        f"Communication Style: {persona.style or 'Natural and authentic'}\n"
        f"Goals in Conversation: {persona.goals or 'Engage meaningfully'}\n\n"
        "IMPORTANT: \n"
        "- Respond naturally as this character would\n"
        "- Keep responses concise (2-4 sentences typical)\n"
        "- Show personality through word choice and tone\n"
        "- React authentically to what the other person says\n"
        "- Don't break character or mention you're an AI"
    )


def build_survey_prompt(persona: Persona, conversation_context: Optional[str] = None) -> str:
    """Return a system prompt for the survey phase.

    If *conversation_context* is provided (post-survey), it takes precedence.
    Otherwise the same simplified / full branching as the conversation prompt applies.
    """
    if conversation_context is not None:
        return (
            f"You are {persona.name} taking a survey after having a conversation.\n\n"
            f"Recent Conversation:\n{conversation_context}\n\n"
            "Answer the following survey questions based on how you feel NOW, after this conversation.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Answer ONLY with the letter of your choice (A, B, C, or D)\n"
            "- Do not include explanations, just the letter\n"
            "- Answer authentically based on your current state of mind after the conversation\n"
            "- Be honest about how the conversation may have influenced your views"
        )

    if persona.preference is not None and _only_preference(persona):
        return (
            f"You are {persona.name} taking a survey.\n\n"
            f"Preference: {persona.preference}\n\n"
            "Answer the following survey questions AS THIS CHARACTER WOULD, based on their preference.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Answer ONLY with the letter of your choice (A, B, C, or D)\n"
            "- Do not include explanations, just the letter\n"
            "- Be consistent with your preference\n"
            "- Answer authentically as this person would"
        )

    # Full persona
    return (
        f"You are {persona.name} taking a survey.\n\n"
        f"Background: {persona.background or 'Not specified'}\n"
        f"Personality: {persona.personality or 'Friendly and conversational'}\n"
        f"Current mindset: Thoughtful and honest\n\n"
        "Answer the following survey questions AS THIS CHARACTER WOULD, based on their personality and current state of mind.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- Answer ONLY with the letter of your choice (A, B, C, or D)\n"
        "- Do not include explanations, just the letter\n"
        "- Be consistent with the character's personality\n"
        "- Answer authentically as this person would"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _only_preference(persona: Persona) -> bool:
    """Return True when preference is the only non-name field that is set."""
    return (
        persona.strategy is None
        and persona.background is None
        and persona.personality is None
        and persona.style is None
        and persona.goals is None
    )
