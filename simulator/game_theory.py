"""
simulator/game_theory.py — Decision-theoretic & two-player game analysis of conversations.

Models a conversation between two personas as a sequential two-player game.
Each turn is treated as a strategic move with payoffs derived from LLM-as-judge
assessments. Provides:

  - Move classification (concede, persuade, anchor, deflect, compromise)
  - Per-turn utility estimation for each player
  - Payoff matrix construction
  - Nash equilibrium approximation
  - Strategy profile detection (dominant, mixed, tit-for-tat)
  - Conversation-level game summary
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from simulator.conversation import Turn
from simulator.personas import Persona
from simulator.providers import Provider
from simulator.tracking import UsageTracker


# ---------------------------------------------------------------------------
# Move taxonomy
# ---------------------------------------------------------------------------

class MoveType(str, Enum):
    """Strategic move classification for a single turn."""
    PERSUADE = "persuade"       # actively trying to shift the other player
    ANCHOR = "anchor"           # reinforcing own position
    CONCEDE = "concede"         # yielding ground to the other player
    COMPROMISE = "compromise"   # proposing middle ground
    DEFLECT = "deflect"         # changing subject or avoiding commitment
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MoveAnalysis:
    """Analysis of a single turn as a strategic move."""
    turn_index: int
    speaker: str
    move_type: str              # MoveType value
    utility_self: float         # how much this move benefits the speaker (0-1)
    utility_opponent: float     # how much this move benefits the other player (0-1)
    reasoning: str              # one-line explanation


@dataclass
class PayoffMatrix:
    """2x2 payoff matrix summarizing the game.

    Rows = Player A strategies, Cols = Player B strategies.
    Each cell is (payoff_a, payoff_b).

    Strategies are simplified to {Cooperate, Compete}:
      - Cooperate = compromise/concede
      - Compete   = persuade/anchor
    """
    # (row_strategy, col_strategy) -> (payoff_a, payoff_b)
    cooperate_cooperate: Tuple[float, float] = (0.0, 0.0)
    cooperate_compete: Tuple[float, float] = (0.0, 0.0)
    compete_cooperate: Tuple[float, float] = (0.0, 0.0)
    compete_compete: Tuple[float, float] = (0.0, 0.0)


@dataclass
class StrategyProfile:
    """Detected strategy profile for a player across the conversation."""
    player: str
    dominant_move: str          # most frequent MoveType
    move_distribution: Dict[str, float]   # MoveType -> frequency (0-1)
    cooperation_rate: float     # fraction of cooperative moves
    strategy_label: str         # e.g. "tit-for-tat", "dominant-compete", "mixed"


@dataclass
class NashEquilibrium:
    """Approximate Nash equilibrium of the conversation game."""
    equilibrium_type: str       # "pure" or "mixed"
    player_a_strategy: str      # "cooperate" or "compete" (or mix description)
    player_b_strategy: str
    payoff_a: float
    payoff_b: float
    is_pareto_optimal: bool
    explanation: str


@dataclass
class GameTheoryResult:
    """Full game-theoretic analysis of a conversation."""
    moves: List[MoveAnalysis]
    payoff_matrix: PayoffMatrix
    strategy_a: StrategyProfile
    strategy_b: StrategyProfile
    nash_equilibrium: NashEquilibrium
    game_type: str              # e.g. "prisoners_dilemma", "stag_hunt", "coordination", "other"
    total_utility_a: float
    total_utility_b: float
    fairness_index: float       # Jain's fairness index (0-1)
    timestamp: str


# ---------------------------------------------------------------------------
# LLM prompt for move classification
# ---------------------------------------------------------------------------

_MOVE_ANALYSIS_SYSTEM_PROMPT = """\
You are a game theory analyst examining a two-person conversation as a sequential game.

Each participant has a stated preference/position. You will analyze ONE turn at a time
and classify the speaker's move strategically.

Context:
- Player A ({player_a}) prefers: {pref_a}
- Player B ({player_b}) prefers: {pref_b}

For the given turn, respond with a JSON object and nothing else. No markdown fences.

Schema:
{{
  "move_type": "<one of: persuade, anchor, concede, compromise, deflect>",
  "utility_self": <float 0.0-1.0>,
  "utility_opponent": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}}

Move types:
  persuade   = actively trying to change the other person's mind
  anchor     = reinforcing or restating own position without engaging the other's
  concede    = yielding ground, acknowledging the other's point has merit
  compromise = proposing middle ground that partially satisfies both
  deflect    = avoiding the topic, changing subject, or non-committal response

Utility scoring (0.0 to 1.0):
  utility_self     = how much this move advances the speaker's original position
  utility_opponent = how much this move (perhaps unintentionally) helps the other player's position

Examples:
  - Strong persuasion attempt: utility_self=0.8, utility_opponent=0.1
  - Full concession: utility_self=0.1, utility_opponent=0.9
  - Mutual compromise: utility_self=0.5, utility_opponent=0.5
  - Anchoring own view: utility_self=0.7, utility_opponent=0.2
"""

_MOVE_USER_TEMPLATE = """\
Conversation so far:
{context}

Analyze this turn:
Turn {turn_index} ({speaker}): {message}
"""



# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_move_response(raw: str, turn_index: int, speaker: str) -> MoveAnalysis:
    """Parse the LLM's JSON response into a MoveAnalysis."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
        move_type = data.get("move_type", "unknown")
        if move_type not in {e.value for e in MoveType}:
            move_type = "unknown"
        return MoveAnalysis(
            turn_index=turn_index,
            speaker=speaker,
            move_type=move_type,
            utility_self=max(0.0, min(1.0, float(data.get("utility_self", 0.5)))),
            utility_opponent=max(0.0, min(1.0, float(data.get("utility_opponent", 0.5)))),
            reasoning=str(data.get("reasoning", "")),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return MoveAnalysis(
            turn_index=turn_index,
            speaker=speaker,
            move_type=MoveType.UNKNOWN.value,
            utility_self=0.5,
            utility_opponent=0.5,
            reasoning=f"Parse error: {raw[:120]}",
        )


def _get_preference_text(persona: Persona) -> str:
    """Extract a short preference description from a persona."""
    if persona.preference:
        return persona.preference
    if persona.strategy:
        return persona.strategy
    if persona.goals:
        return persona.goals
    return persona.name


def _format_context(turns: List[Turn], up_to_index: int) -> str:
    """Format conversation turns up to (but not including) the target turn."""
    lines = []
    for t in turns:
        if t.turn_index < up_to_index:
            lines.append(f"Turn {t.turn_index} ({t.speaker}): {t.message}")
    return "\n".join(lines) if lines else "(conversation start)"


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def analyze_move(
    turn: Turn,
    all_turns: List[Turn],
    persona_a: Persona,
    persona_b: Persona,
    provider: Provider,
    tracker: UsageTracker,
) -> MoveAnalysis:
    """Classify a single turn as a strategic move using the LLM.

    Args:
        turn:       The turn to analyze.
        all_turns:  Full conversation for context.
        persona_a:  First player.
        persona_b:  Second player.
        provider:   LLM provider for the analysis call.
        tracker:    Token usage tracker.

    Returns:
        MoveAnalysis with move type, utilities, and reasoning.
    """
    system_prompt = _MOVE_ANALYSIS_SYSTEM_PROMPT.format(
        player_a=persona_a.name,
        player_b=persona_b.name,
        pref_a=_get_preference_text(persona_a),
        pref_b=_get_preference_text(persona_b),
    )

    context = _format_context(all_turns, turn.turn_index)
    user_message = _MOVE_USER_TEMPLATE.format(
        context=context,
        turn_index=turn.turn_index,
        speaker=turn.speaker,
        message=turn.message,
    )

    result = provider.call(
        messages=[{"role": "user", "content": user_message}],
        system_prompt=system_prompt,
        max_tokens=256,
        call_type="game_theory",
        persona_name=turn.speaker,
    )

    return _parse_move_response(result.text, turn.turn_index, turn.speaker)


def analyze_all_moves(
    turns: List[Turn],
    persona_a: Persona,
    persona_b: Persona,
    provider: Provider,
    tracker: UsageTracker,
) -> List[MoveAnalysis]:
    """Classify every turn in the conversation as a strategic move.

    Makes one LLM call per turn.
    """
    moves = []
    for turn in turns:
        move = analyze_move(turn, turns, persona_a, persona_b, provider, tracker)
        moves.append(move)
    return moves


# ---------------------------------------------------------------------------
# Payoff matrix construction
# ---------------------------------------------------------------------------

def _is_cooperative(move_type: str) -> bool:
    """Return True if the move type is cooperative (concede/compromise)."""
    return move_type in (MoveType.CONCEDE.value, MoveType.COMPROMISE.value)


def build_payoff_matrix(
    moves: List[MoveAnalysis],
    persona_a_name: str,
    persona_b_name: str,
) -> PayoffMatrix:
    """Construct a 2x2 payoff matrix from observed move utilities.

    Groups moves into Cooperate vs Compete for each player, then averages
    the observed utilities for each (strategy_a, strategy_b) combination.

    Cooperate = concede or compromise
    Compete   = persuade, anchor, or deflect
    """
    # Buckets: (a_cooperative, b_cooperative) -> list of (util_a, util_b)
    buckets: Dict[Tuple[bool, bool], List[Tuple[float, float]]] = {
        (True, True): [],
        (True, False): [],
        (False, True): [],
        (False, False): [],
    }

    # Pair consecutive turns (A then B, or B then A)
    a_moves = [m for m in moves if m.speaker == persona_a_name]
    b_moves = [m for m in moves if m.speaker == persona_b_name]

    # Match moves by proximity in turn order
    for a_m in a_moves:
        # Find the closest B move
        closest_b = min(b_moves, key=lambda b: abs(b.turn_index - a_m.turn_index), default=None)
        if closest_b is None:
            continue
        a_coop = _is_cooperative(a_m.move_type)
        b_coop = _is_cooperative(closest_b.move_type)
        # Utility for A in this pair, utility for B in this pair
        util_a = a_m.utility_self
        util_b = closest_b.utility_self
        buckets[(a_coop, b_coop)].append((util_a, util_b))

    def _avg(pairs: List[Tuple[float, float]]) -> Tuple[float, float]:
        if not pairs:
            return (0.0, 0.0)
        avg_a = sum(p[0] for p in pairs) / len(pairs)
        avg_b = sum(p[1] for p in pairs) / len(pairs)
        return (round(avg_a, 3), round(avg_b, 3))

    return PayoffMatrix(
        cooperate_cooperate=_avg(buckets[(True, True)]),
        cooperate_compete=_avg(buckets[(True, False)]),
        compete_cooperate=_avg(buckets[(False, True)]),
        compete_compete=_avg(buckets[(False, False)]),
    )


# ---------------------------------------------------------------------------
# Strategy profile detection
# ---------------------------------------------------------------------------

def detect_strategy_profile(
    moves: List[MoveAnalysis],
    player_name: str,
) -> StrategyProfile:
    """Analyze a player's moves to detect their strategy profile.

    Returns move distribution, cooperation rate, and a strategy label.
    """
    player_moves = [m for m in moves if m.speaker == player_name]
    if not player_moves:
        return StrategyProfile(
            player=player_name,
            dominant_move=MoveType.UNKNOWN.value,
            move_distribution={},
            cooperation_rate=0.0,
            strategy_label="no_moves",
        )

    # Count move types
    counts: Dict[str, int] = {}
    for m in player_moves:
        counts[m.move_type] = counts.get(m.move_type, 0) + 1

    total = len(player_moves)
    distribution = {k: round(v / total, 3) for k, v in counts.items()}
    dominant = max(counts, key=counts.get)  # type: ignore[arg-type]

    coop_count = sum(1 for m in player_moves if _is_cooperative(m.move_type))
    coop_rate = round(coop_count / total, 3)

    # Detect strategy pattern
    label = _classify_strategy(player_moves, coop_rate, dominant)

    return StrategyProfile(
        player=player_name,
        dominant_move=dominant,
        move_distribution=distribution,
        cooperation_rate=coop_rate,
        strategy_label=label,
    )


def _classify_strategy(
    moves: List[MoveAnalysis],
    coop_rate: float,
    dominant: str,
) -> str:
    """Classify the overall strategy pattern."""
    if coop_rate >= 0.8:
        return "dominant_cooperate"
    if coop_rate <= 0.2:
        return "dominant_compete"

    # Check for tit-for-tat pattern (alternating cooperation)
    if len(moves) >= 4:
        coop_sequence = [_is_cooperative(m.move_type) for m in moves]
        alternations = sum(
            1 for i in range(1, len(coop_sequence))
            if coop_sequence[i] != coop_sequence[i - 1]
        )
        if alternations / (len(coop_sequence) - 1) > 0.6:
            return "tit_for_tat"

    # Check for escalation (increasing competition over time)
    if len(moves) >= 3:
        first_half = moves[: len(moves) // 2]
        second_half = moves[len(moves) // 2 :]
        first_coop = sum(1 for m in first_half if _is_cooperative(m.move_type)) / len(first_half)
        second_coop = sum(1 for m in second_half if _is_cooperative(m.move_type)) / len(second_half)
        if first_coop - second_coop > 0.3:
            return "escalating"
        if second_coop - first_coop > 0.3:
            return "de_escalating"

    if 0.4 <= coop_rate <= 0.6:
        return "mixed"

    return f"leaning_{'cooperate' if coop_rate > 0.5 else 'compete'}"


# ---------------------------------------------------------------------------
# Nash equilibrium approximation
# ---------------------------------------------------------------------------

def find_nash_equilibrium(matrix: PayoffMatrix) -> NashEquilibrium:
    """Find the Nash equilibrium of the 2x2 game defined by the payoff matrix.

    Checks for pure strategy Nash equilibria first, then computes the
    mixed strategy equilibrium if no pure NE exists.

    Returns a NashEquilibrium with the equilibrium type, strategies, and payoffs.
    """
    # Extract payoffs: matrix[row][col] = (payoff_a, payoff_b)
    # Row player = A, Col player = B
    # Strategies: 0 = cooperate, 1 = compete
    payoffs = {
        (0, 0): matrix.cooperate_cooperate,
        (0, 1): matrix.cooperate_compete,
        (1, 0): matrix.compete_cooperate,
        (1, 1): matrix.compete_compete,
    }

    strategy_names = {0: "cooperate", 1: "compete"}

    # Check for pure strategy Nash equilibria
    pure_ne: List[Tuple[int, int]] = []

    for a_strat in (0, 1):
        for b_strat in (0, 1):
            a_payoff = payoffs[(a_strat, b_strat)][0]
            # Check if A wants to deviate
            a_alt = 1 - a_strat
            a_alt_payoff = payoffs[(a_alt, b_strat)][0]
            a_best_response = a_payoff >= a_alt_payoff

            b_payoff = payoffs[(a_strat, b_strat)][1]
            # Check if B wants to deviate
            b_alt = 1 - b_strat
            b_alt_payoff = payoffs[(a_strat, b_alt)][1]
            b_best_response = b_payoff >= b_alt_payoff

            if a_best_response and b_best_response:
                pure_ne.append((a_strat, b_strat))

    if pure_ne:
        # Take the first pure NE (or the Pareto-dominant one if multiple)
        best = pure_ne[0]
        if len(pure_ne) > 1:
            # Pick the one with highest total payoff
            best = max(pure_ne, key=lambda s: payoffs[s][0] + payoffs[s][1])

        pa, pb = payoffs[best]
        # Check Pareto optimality: no other outcome makes both better off
        is_pareto = True
        for other in payoffs.values():
            if other[0] > pa and other[1] > pb:
                is_pareto = False
                break

        return NashEquilibrium(
            equilibrium_type="pure",
            player_a_strategy=strategy_names[best[0]],
            player_b_strategy=strategy_names[best[1]],
            payoff_a=pa,
            payoff_b=pb,
            is_pareto_optimal=is_pareto,
            explanation=_explain_pure_ne(best, payoffs, strategy_names, len(pure_ne)),
        )

    # Mixed strategy NE for 2x2 game
    # A mixes to make B indifferent, B mixes to make A indifferent
    # B indifferent when: p * u_B(C,C) + (1-p) * u_B(K,C) = p * u_B(C,K) + (1-p) * u_B(K,K)
    # where p = prob A cooperates
    denom_b = (
        payoffs[(0, 0)][1] - payoffs[(0, 1)][1]
        - payoffs[(1, 0)][1] + payoffs[(1, 1)][1]
    )
    denom_a = (
        payoffs[(0, 0)][0] - payoffs[(1, 0)][0]
        - payoffs[(0, 1)][0] + payoffs[(1, 1)][0]
    )

    if abs(denom_b) < 1e-9 or abs(denom_a) < 1e-9:
        # Degenerate game — default to compete-compete
        pa, pb = payoffs[(1, 1)]
        return NashEquilibrium(
            equilibrium_type="degenerate",
            player_a_strategy="compete",
            player_b_strategy="compete",
            payoff_a=pa,
            payoff_b=pb,
            is_pareto_optimal=False,
            explanation="Degenerate payoff structure; no clear equilibrium.",
        )

    p_a_coop = (payoffs[(1, 1)][1] - payoffs[(1, 0)][1]) / denom_b
    p_b_coop = (payoffs[(1, 1)][0] - payoffs[(0, 1)][0]) / denom_a

    # Clamp to [0, 1]
    p_a_coop = max(0.0, min(1.0, p_a_coop))
    p_b_coop = max(0.0, min(1.0, p_b_coop))

    # Expected payoffs under mixed strategy
    exp_a = (
        p_a_coop * p_b_coop * payoffs[(0, 0)][0]
        + p_a_coop * (1 - p_b_coop) * payoffs[(0, 1)][0]
        + (1 - p_a_coop) * p_b_coop * payoffs[(1, 0)][0]
        + (1 - p_a_coop) * (1 - p_b_coop) * payoffs[(1, 1)][0]
    )
    exp_b = (
        p_a_coop * p_b_coop * payoffs[(0, 0)][1]
        + p_a_coop * (1 - p_b_coop) * payoffs[(0, 1)][1]
        + (1 - p_a_coop) * p_b_coop * payoffs[(1, 0)][1]
        + (1 - p_a_coop) * (1 - p_b_coop) * payoffs[(1, 1)][1]
    )

    return NashEquilibrium(
        equilibrium_type="mixed",
        player_a_strategy=f"cooperate with p={p_a_coop:.2f}",
        player_b_strategy=f"cooperate with p={p_b_coop:.2f}",
        payoff_a=round(exp_a, 3),
        payoff_b=round(exp_b, 3),
        is_pareto_optimal=False,  # mixed NE rarely Pareto optimal
        explanation=(
            f"Mixed strategy NE: A cooperates {p_a_coop:.0%} of the time, "
            f"B cooperates {p_b_coop:.0%} of the time."
        ),
    )


def _explain_pure_ne(
    ne: Tuple[int, int],
    payoffs: Dict[Tuple[int, int], Tuple[float, float]],
    names: Dict[int, str],
    total_ne: int,
) -> str:
    """Generate a human-readable explanation of a pure NE."""
    a_s, b_s = names[ne[0]], names[ne[1]]
    pa, pb = payoffs[ne]
    parts = [f"Pure NE at ({a_s}, {b_s}) with payoffs ({pa:.2f}, {pb:.2f})."]
    if total_ne > 1:
        parts.append(f"{total_ne} pure NE exist; selected the Pareto-dominant one.")
    if ne == (1, 1):
        parts.append("Both players compete — resembles a Prisoner's Dilemma outcome.")
    elif ne == (0, 0):
        parts.append("Both players cooperate — socially optimal outcome.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Game type classification
# ---------------------------------------------------------------------------

def classify_game_type(matrix: PayoffMatrix) -> str:
    """Classify the 2x2 game into a known game type based on payoff structure.

    Returns one of: prisoners_dilemma, stag_hunt, chicken, coordination,
    harmony, deadlock, other.
    """
    cc = matrix.cooperate_cooperate
    cd = matrix.cooperate_compete
    dc = matrix.compete_cooperate
    dd = matrix.compete_compete

    # Check from A's perspective (symmetric check)
    # Prisoner's Dilemma: T > R > P > S (dc[0] > cc[0] > dd[0] > cd[0])
    if dc[0] > cc[0] > dd[0] > cd[0]:
        return "prisoners_dilemma"

    # Stag Hunt: R > T > P > S (cc[0] > dc[0] > dd[0] > cd[0])
    if cc[0] > dc[0] > dd[0] > cd[0]:
        return "stag_hunt"

    # Chicken/Hawk-Dove: T > R > S > P (dc[0] > cc[0] > cd[0] > dd[0])
    if dc[0] > cc[0] > cd[0] > dd[0]:
        return "chicken"

    # Coordination: R > T, R > S, P > T, P > S (both prefer matching)
    if cc[0] > dc[0] and cc[0] > cd[0] and dd[0] > dc[0] and dd[0] > cd[0]:
        return "coordination"

    # Harmony: R > T > S > P but cooperation dominates
    if cc[0] > dc[0] and cc[0] > dd[0] and cc[0] > cd[0]:
        return "harmony"

    # Deadlock: T > P > R > S (both prefer to defect regardless)
    if dc[0] > dd[0] > cc[0] > cd[0]:
        return "deadlock"

    return "other"


# ---------------------------------------------------------------------------
# Fairness index
# ---------------------------------------------------------------------------

def jains_fairness_index(utility_a: float, utility_b: float) -> float:
    """Compute Jain's fairness index for two players.

    Returns a value in [0, 1] where 1 = perfectly fair.
    Formula: (sum)^2 / (n * sum_of_squares)
    """
    total = utility_a + utility_b
    sum_sq = utility_a ** 2 + utility_b ** 2
    if sum_sq == 0:
        return 1.0  # both zero = trivially fair
    return round((total ** 2) / (2 * sum_sq), 3)


# ---------------------------------------------------------------------------
# Top-level analysis function
# ---------------------------------------------------------------------------

def analyze_conversation_as_game(
    turns: List[Turn],
    persona_a: Persona,
    persona_b: Persona,
    provider: Provider,
    tracker: UsageTracker,
) -> GameTheoryResult:
    """Run full game-theoretic analysis on a conversation.

    This is the main entry point. It:
      1. Classifies each turn as a strategic move (LLM calls)
      2. Builds a 2x2 payoff matrix
      3. Detects strategy profiles for each player
      4. Finds the Nash equilibrium
      5. Classifies the game type
      6. Computes fairness metrics

    Args:
        turns:     Full conversation turn list.
        persona_a: First player.
        persona_b: Second player.
        provider:  LLM provider for move classification.
        tracker:   Token usage tracker.

    Returns:
        GameTheoryResult with complete analysis.
    """
    # 1. Classify all moves
    moves = analyze_all_moves(turns, persona_a, persona_b, provider, tracker)

    # 2. Build payoff matrix
    matrix = build_payoff_matrix(moves, persona_a.name, persona_b.name)

    # 3. Strategy profiles
    strategy_a = detect_strategy_profile(moves, persona_a.name)
    strategy_b = detect_strategy_profile(moves, persona_b.name)

    # 4. Nash equilibrium
    nash = find_nash_equilibrium(matrix)

    # 5. Game type
    game_type = classify_game_type(matrix)

    # 6. Aggregate utilities
    a_moves = [m for m in moves if m.speaker == persona_a.name]
    b_moves = [m for m in moves if m.speaker == persona_b.name]
    total_a = sum(m.utility_self for m in a_moves) / max(len(a_moves), 1)
    total_b = sum(m.utility_self for m in b_moves) / max(len(b_moves), 1)

    # 7. Fairness
    fairness = jains_fairness_index(total_a, total_b)

    return GameTheoryResult(
        moves=moves,
        payoff_matrix=matrix,
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        nash_equilibrium=nash,
        game_type=game_type,
        total_utility_a=round(total_a, 3),
        total_utility_b=round(total_b, 3),
        fairness_index=fairness,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
