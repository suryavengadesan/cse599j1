"""
Tests for simulator/scenarios.py — YAML loader and registry.
"""

import pytest
from simulator.scenarios import load_scenario, list_scenarios, ScenarioNotFoundError, ScenarioValidationError


def test_list_scenarios_includes_seattle_sf():
    scenarios = list_scenarios()
    assert "seattle-sf" in scenarios


def test_load_seattle_sf_basic():
    config = load_scenario("seattle-sf")
    assert config.name == "seattle-sf"
    assert config.persona_a.name == "Alice"
    assert config.persona_b.name == "Bob"
    assert config.initial_message
    assert len(config.survey["questions"]) == 4


def test_load_seattle_sf_adversarial():
    config = load_scenario("seattle-sf", adversarial=True)
    # Bob's adversarial variant has a strategy
    assert config.persona_b.strategy is not None


def test_load_unknown_scenario_raises():
    with pytest.raises(ScenarioNotFoundError) as exc_info:
        load_scenario("nonexistent-scenario")
    assert "nonexistent-scenario" in str(exc_info.value)
    assert "seattle-sf" in str(exc_info.value)


def test_scenario_has_initial_message():
    config = load_scenario("seattle-sf")
    assert isinstance(config.initial_message, str)
    assert len(config.initial_message.strip()) > 0
