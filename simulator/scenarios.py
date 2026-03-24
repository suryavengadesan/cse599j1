"""
simulator/scenarios.py — YAML scenario loader and registry.

Loads scenario definitions from ``scenarios/{name}.yaml`` and exposes
``load_scenario()`` and ``list_scenarios()`` for use by the experiment runner
and CLI.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

from simulator.personas import Persona, persona_from_dict

# Directory where scenario YAML files live (relative to project root)
_SCENARIOS_DIR = Path("scenarios")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ScenarioNotFoundError(FileNotFoundError):
    """Raised when the requested scenario YAML does not exist."""

    def __init__(self, name: str, available: List[str]) -> None:
        available_str = ", ".join(sorted(available)) if available else "(none)"
        super().__init__(
            f"Scenario {name!r} not found. Available scenarios: {available_str}"
        )
        self.name = name
        self.available = available


class ScenarioValidationError(ValueError):
    """Raised when a scenario YAML is missing required fields."""

    def __init__(self, name: str, missing_field: str) -> None:
        super().__init__(
            f"Scenario {name!r} is missing required field: {missing_field!r}"
        )
        self.name = name
        self.missing_field = missing_field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    name: str
    persona_a: Persona
    persona_b: Persona
    survey: dict
    initial_message: str
    persona_a_adversarial: Optional[Persona] = None
    persona_b_adversarial: Optional[Persona] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_scenarios() -> List[str]:
    """Return the names (without extension) of all YAML files in ``scenarios/``."""
    if not _SCENARIOS_DIR.exists():
        return []
    return [p.stem for p in sorted(_SCENARIOS_DIR.glob("*.yaml"))]


def load_scenario(name: str, adversarial: bool = False) -> ScenarioConfig:
    """Load a scenario from ``scenarios/{name}.yaml``.

    Parameters
    ----------
    name:
        Scenario name (without ``.yaml`` extension).
    adversarial:
        When *True*, the returned config's ``persona_a`` / ``persona_b`` are
        replaced with their adversarial variants (if defined in the YAML).

    Raises
    ------
    ScenarioNotFoundError
        If the YAML file does not exist.
    ScenarioValidationError
        If a required top-level field is absent.
    """
    path = _SCENARIOS_DIR / f"{name}.yaml"
    if not path.exists():
        raise ScenarioNotFoundError(name, list_scenarios())

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    # Validate required fields
    for field in ("name", "persona_a", "persona_b", "survey", "initial_message"):
        if field not in data or data[field] is None:
            raise ScenarioValidationError(name, field)

    persona_a = persona_from_dict(data["persona_a"])
    persona_b = persona_from_dict(data["persona_b"])

    persona_a_adv: Optional[Persona] = None
    persona_b_adv: Optional[Persona] = None

    if "persona_a_adversarial" in data and data["persona_a_adversarial"]:
        persona_a_adv = persona_from_dict(data["persona_a_adversarial"])
    if "persona_b_adversarial" in data and data["persona_b_adversarial"]:
        persona_b_adv = persona_from_dict(data["persona_b_adversarial"])

    config = ScenarioConfig(
        name=data["name"],
        persona_a=persona_a,
        persona_b=persona_b,
        survey=data["survey"],
        initial_message=data["initial_message"],
        persona_a_adversarial=persona_a_adv,
        persona_b_adversarial=persona_b_adv,
    )

    if adversarial:
        if persona_a_adv is not None:
            config.persona_a = persona_a_adv
        if persona_b_adv is not None:
            config.persona_b = persona_b_adv

    return config
