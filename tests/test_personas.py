"""
Tests for simulator/personas.py — Persona dataclass and prompt builders.
"""

import pytest
from simulator.personas import Persona, persona_from_dict, build_conversation_prompt, build_survey_prompt


def test_persona_from_dict_simplified():
    p = persona_from_dict({"name": "Alice", "preference": "Loves SF"})
    assert p.name == "Alice"
    assert p.preference == "Loves SF"
    assert p.strategy is None
    assert p.background is None


def test_persona_from_dict_adversarial():
    p = persona_from_dict({"name": "Bob", "strategy": "Convince Alice"})
    assert p.name == "Bob"
    assert p.strategy == "Convince Alice"
    assert p.preference is None


def test_persona_from_dict_full():
    p = persona_from_dict({
        "name": "Carol",
        "background": "Engineer",
        "personality": "Analytical",
        "style": "Direct",
        "goals": "Find best city",
    })
    assert p.background == "Engineer"
    assert p.preference is None
    assert p.strategy is None


def test_build_conversation_prompt_adversarial():
    p = Persona(name="Bob", strategy="Convince Alice")
    prompt = build_conversation_prompt(p)
    assert len(prompt) > 0
    assert "Bob" in prompt
    assert "Strategy" in prompt


def test_build_conversation_prompt_simplified():
    p = Persona(name="Alice", preference="Loves SF")
    prompt = build_conversation_prompt(p)
    assert len(prompt) > 0
    assert "Alice" in prompt
    assert "Preference" in prompt


def test_build_conversation_prompt_full():
    p = Persona(name="Carol", background="Engineer", personality="Analytical",
                style="Direct", goals="Find best city")
    prompt = build_conversation_prompt(p)
    assert len(prompt) > 0
    assert "Carol" in prompt
    assert "Background" in prompt


def test_build_survey_prompt_pre():
    p = Persona(name="Alice", preference="Loves SF")
    prompt = build_survey_prompt(p)
    assert len(prompt) > 0
    assert "Alice" in prompt


def test_build_survey_prompt_post_with_context():
    p = Persona(name="Alice", preference="Loves SF")
    prompt = build_survey_prompt(p, conversation_context="Bob: Seattle is great!")
    assert "Seattle is great" in prompt
    assert "after" in prompt.lower() or "conversation" in prompt.lower()
