"""Tests for simulator.game_theory module."""

import pytest

from simulator.conversation import Turn
from simulator.game_theory import (
    GameTheoryResult,
    MoveAnalysis,
    MoveType,
    NashEquilibrium,
    PayoffMatrix,
    StrategyProfile,
    _is_cooperative,
    _parse_move_response,
    build_payoff_matrix,
    classify_game_type,
    detect_strategy_profile,
    find_nash_equilibrium,
    jains_fairness_index,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_move(turn_index, speaker, move_type, util_self=0.5, util_opp=0.5):
    return MoveAnalysis(
        turn_index=turn_index,
        speaker=speaker,
        move_type=move_type,
        utility_self=util_self,
        utility_opponent=util_opp,
        reasoning="test",
    )


# ---------------------------------------------------------------------------
# MoveType / cooperative classification
# ---------------------------------------------------------------------------

class TestMoveClassification:
    def test_concede_is_cooperative(self):
        assert _is_cooperative(MoveType.CONCEDE.value) is True

    def test_compromise_is_cooperative(self):
        assert _is_cooperative(MoveType.COMPROMISE.value) is True

    def test_persuade_is_not_cooperative(self):
        assert _is_cooperative(MoveType.PERSUADE.value) is False

    def test_anchor_is_not_cooperative(self):
        assert _is_cooperative(MoveType.ANCHOR.value) is False

    def test_deflect_is_not_cooperative(self):
        assert _is_cooperative(MoveType.DEFLECT.value) is False


# ---------------------------------------------------------------------------
# Parse move response
# ---------------------------------------------------------------------------

class TestParseMoveResponse:
    def test_valid_json(self):
        raw = '{"move_type": "persuade", "utility_self": 0.8, "utility_opponent": 0.2, "reasoning": "strong argument"}'
        result = _parse_move_response(raw, 1, "Alice")
        assert result.move_type == "persuade"
        assert result.utility_self == 0.8
        assert result.utility_opponent == 0.2
        assert result.speaker == "Alice"

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"move_type": "anchor", "utility_self": 0.7, "utility_opponent": 0.3, "reasoning": "held firm"}\n```'
        result = _parse_move_response(raw, 2, "Bob")
        assert result.move_type == "anchor"
        assert result.utility_self == 0.7

    def test_invalid_json_returns_unknown(self):
        result = _parse_move_response("not json at all", 0, "Alice")
        assert result.move_type == MoveType.UNKNOWN.value
        assert result.utility_self == 0.5

    def test_utility_clamped_to_bounds(self):
        raw = '{"move_type": "concede", "utility_self": 1.5, "utility_opponent": -0.3, "reasoning": "out of bounds"}'
        result = _parse_move_response(raw, 0, "Alice")
        assert result.utility_self == 1.0
        assert result.utility_opponent == 0.0

    def test_unknown_move_type_defaults(self):
        raw = '{"move_type": "attack", "utility_self": 0.5, "utility_opponent": 0.5, "reasoning": "invalid type"}'
        result = _parse_move_response(raw, 0, "Alice")
        assert result.move_type == "unknown"


# ---------------------------------------------------------------------------
# Payoff matrix
# ---------------------------------------------------------------------------

class TestPayoffMatrix:
    def test_all_cooperative_moves(self):
        moves = [
            _make_move(0, "Alice", "compromise", 0.6, 0.6),
            _make_move(1, "Bob", "concede", 0.4, 0.8),
            _make_move(2, "Alice", "compromise", 0.5, 0.5),
            _make_move(3, "Bob", "compromise", 0.5, 0.5),
        ]
        matrix = build_payoff_matrix(moves, "Alice", "Bob")
        # All moves are cooperative, so cooperate-cooperate should have values
        assert matrix.cooperate_cooperate != (0.0, 0.0)

    def test_mixed_moves(self):
        moves = [
            _make_move(0, "Alice", "persuade", 0.8, 0.2),
            _make_move(1, "Bob", "anchor", 0.7, 0.3),
            _make_move(2, "Alice", "concede", 0.3, 0.7),
            _make_move(3, "Bob", "compromise", 0.5, 0.5),
        ]
        matrix = build_payoff_matrix(moves, "Alice", "Bob")
        # Should have entries in multiple cells
        assert isinstance(matrix, PayoffMatrix)

    def test_empty_moves(self):
        matrix = build_payoff_matrix([], "Alice", "Bob")
        assert matrix.cooperate_cooperate == (0.0, 0.0)
        assert matrix.compete_compete == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Strategy profile detection
# ---------------------------------------------------------------------------

class TestStrategyProfile:
    def test_dominant_compete(self):
        moves = [
            _make_move(0, "Alice", "persuade"),
            _make_move(2, "Alice", "anchor"),
            _make_move(4, "Alice", "persuade"),
            _make_move(6, "Alice", "anchor"),
            _make_move(8, "Alice", "persuade"),
        ]
        profile = detect_strategy_profile(moves, "Alice")
        assert profile.cooperation_rate == 0.0
        assert profile.strategy_label == "dominant_compete"

    def test_dominant_cooperate(self):
        moves = [
            _make_move(0, "Bob", "concede"),
            _make_move(2, "Bob", "compromise"),
            _make_move(4, "Bob", "concede"),
            _make_move(6, "Bob", "compromise"),
            _make_move(8, "Bob", "concede"),
        ]
        profile = detect_strategy_profile(moves, "Bob")
        assert profile.cooperation_rate == 1.0
        assert profile.strategy_label == "dominant_cooperate"

    def test_mixed_strategy(self):
        moves = [
            _make_move(0, "Alice", "persuade"),
            _make_move(2, "Alice", "concede"),
            _make_move(4, "Alice", "anchor"),
            _make_move(6, "Alice", "compromise"),
        ]
        profile = detect_strategy_profile(moves, "Alice")
        assert profile.cooperation_rate == 0.5
        assert profile.strategy_label in ("mixed", "tit_for_tat")

    def test_no_moves_returns_no_moves_label(self):
        profile = detect_strategy_profile([], "Alice")
        assert profile.strategy_label == "no_moves"

    def test_move_distribution_sums_to_one(self):
        moves = [
            _make_move(0, "Alice", "persuade"),
            _make_move(2, "Alice", "concede"),
            _make_move(4, "Alice", "anchor"),
            _make_move(6, "Alice", "compromise"),
            _make_move(8, "Alice", "deflect"),
        ]
        profile = detect_strategy_profile(moves, "Alice")
        total = sum(profile.move_distribution.values())
        assert abs(total - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Nash equilibrium
# ---------------------------------------------------------------------------

class TestNashEquilibrium:
    def test_prisoners_dilemma_structure(self):
        """Classic PD: compete-compete is the NE even though cooperate-cooperate is better."""
        matrix = PayoffMatrix(
            cooperate_cooperate=(0.6, 0.6),
            cooperate_compete=(0.1, 0.9),
            compete_cooperate=(0.9, 0.1),
            compete_compete=(0.3, 0.3),
        )
        ne = find_nash_equilibrium(matrix)
        assert ne.equilibrium_type == "pure"
        assert ne.player_a_strategy == "compete"
        assert ne.player_b_strategy == "compete"

    def test_coordination_game(self):
        """Both cooperate and both compete are NE; should pick the better one."""
        matrix = PayoffMatrix(
            cooperate_cooperate=(0.8, 0.8),
            cooperate_compete=(0.1, 0.1),
            compete_cooperate=(0.1, 0.1),
            compete_compete=(0.5, 0.5),
        )
        ne = find_nash_equilibrium(matrix)
        assert ne.equilibrium_type == "pure"
        # Should pick cooperate-cooperate as it's Pareto dominant
        assert ne.player_a_strategy == "cooperate"
        assert ne.player_b_strategy == "cooperate"
        assert ne.is_pareto_optimal is True

    def test_zero_payoffs_degenerate(self):
        matrix = PayoffMatrix(
            cooperate_cooperate=(0.0, 0.0),
            cooperate_compete=(0.0, 0.0),
            compete_cooperate=(0.0, 0.0),
            compete_compete=(0.0, 0.0),
        )
        ne = find_nash_equilibrium(matrix)
        # All strategies are NE when payoffs are equal
        assert ne.equilibrium_type == "pure"


# ---------------------------------------------------------------------------
# Game type classification
# ---------------------------------------------------------------------------

class TestGameTypeClassification:
    def test_prisoners_dilemma(self):
        matrix = PayoffMatrix(
            cooperate_cooperate=(0.6, 0.6),
            cooperate_compete=(0.1, 0.9),
            compete_cooperate=(0.9, 0.1),
            compete_compete=(0.3, 0.3),
        )
        assert classify_game_type(matrix) == "prisoners_dilemma"

    def test_stag_hunt(self):
        matrix = PayoffMatrix(
            cooperate_cooperate=(0.9, 0.9),
            cooperate_compete=(0.1, 0.5),
            compete_cooperate=(0.5, 0.1),
            compete_compete=(0.3, 0.3),
        )
        assert classify_game_type(matrix) == "stag_hunt"

    def test_chicken(self):
        matrix = PayoffMatrix(
            cooperate_cooperate=(0.6, 0.6),
            cooperate_compete=(0.3, 0.8),
            compete_cooperate=(0.8, 0.3),
            compete_compete=(0.1, 0.1),
        )
        assert classify_game_type(matrix) == "chicken"

    def test_harmony(self):
        matrix = PayoffMatrix(
            cooperate_cooperate=(0.9, 0.9),
            cooperate_compete=(0.5, 0.5),
            compete_cooperate=(0.4, 0.4),
            compete_compete=(0.3, 0.3),
        )
        assert classify_game_type(matrix) == "harmony"


# ---------------------------------------------------------------------------
# Fairness index
# ---------------------------------------------------------------------------

class TestFairnessIndex:
    def test_perfectly_fair(self):
        assert jains_fairness_index(0.5, 0.5) == 1.0

    def test_completely_unfair(self):
        result = jains_fairness_index(1.0, 0.0)
        assert result == 0.5  # Jain's index for (1, 0) = 1/2

    def test_both_zero(self):
        assert jains_fairness_index(0.0, 0.0) == 1.0

    def test_moderate_inequality(self):
        result = jains_fairness_index(0.8, 0.4)
        assert 0.5 < result < 1.0
