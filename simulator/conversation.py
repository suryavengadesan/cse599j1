"""
simulator/conversation.py — Conversation loop and Turn dataclass.

Implements a clean, symmetric turn-taking loop between two personas.
"""

from dataclasses import dataclass
from typing import List, Optional

from simulator.personas import Persona, build_conversation_prompt
from simulator.providers import Provider
from simulator.tracking import UsageTracker


@dataclass
class Turn:
    speaker: str
    message: str
    turn_index: int


def run(
    persona_a: Persona,
    persona_b: Persona,
    initial_message: str,
    num_turns: int,
    provider: Provider,
    tracker: UsageTracker,
    verbose: bool = False,
) -> List[Turn]:
    """Run a conversation between two personas and return the list of Turns.

    Turn 0 is the initial message from persona_a (no API call).
    Then for each of *num_turns* exchanges:
      - persona_b responds  (turn 2i+1)
      - persona_a responds  (turn 2i+2), skipped on the last exchange

    Each persona maintains its own ``messages`` list so the API always sees a
    valid alternating user/assistant sequence.

    Args:
        persona_a:       First persona (sends the initial message).
        persona_b:       Second persona.
        initial_message: Opening message from persona_a (turn 0).
        num_turns:       Number of full exchanges (one response from each persona).
        provider:        Provider used for all API calls.
        tracker:         UsageTracker that records token usage.
        verbose:         If True, print each turn to stdout.

    Returns:
        List of Turn objects with sequential turn_index values starting at 0.
        Total length: 2 * num_turns + 1.
    """
    system_a = build_conversation_prompt(persona_a)
    system_b = build_conversation_prompt(persona_b)

    # Separate message histories for each persona (user/assistant alternation)
    messages_a: List[dict] = []
    messages_b: List[dict] = []

    turns: List[Turn] = []

    # Turn 0 — initial message, no API call
    turns.append(Turn(speaker=persona_a.name, message=initial_message, turn_index=0))
    if verbose:
        print(f"\n{'='*60}")
        print(f"CONVERSATION: {persona_a.name} & {persona_b.name}")
        print(f"{'='*60}\n")
        print(f"{persona_a.name}: {initial_message}\n")

    current_message = initial_message

    for exchange in range(num_turns):
        # --- Persona B responds ---
        messages_b.append({"role": "user", "content": current_message})
        result_b = provider.call(
            messages=messages_b,
            system_prompt=system_b,
            call_type="conversation",
            persona_name=persona_b.name,
        )
        messages_b.append({"role": "assistant", "content": result_b.text})

        turn_index_b = 2 * exchange + 1
        turns.append(Turn(speaker=persona_b.name, message=result_b.text, turn_index=turn_index_b))
        if verbose:
            print(f"{persona_b.name}: {result_b.text}\n")

        # --- Persona A responds ---
        messages_a.append({"role": "user", "content": result_b.text})
        result_a = provider.call(
            messages=messages_a,
            system_prompt=system_a,
            call_type="conversation",
            persona_name=persona_a.name,
        )
        messages_a.append({"role": "assistant", "content": result_a.text})

        turn_index_a = 2 * exchange + 2
        turns.append(Turn(speaker=persona_a.name, message=result_a.text, turn_index=turn_index_a))
        if verbose:
            print(f"{persona_a.name}: {result_a.text}\n")

        current_message = result_a.text

    if verbose:
        print(f"{'='*60}\n")

    return turns
