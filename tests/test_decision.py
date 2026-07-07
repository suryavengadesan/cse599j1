"""
Tests for simulator/decision.py — match-decision parsing and aggregation.

These tests exercise pure parsing / bundling logic and the interview-mode
summary; they do not make API calls.
"""

from simulator.conversation import Turn
from simulator.decision import (
    MatchDecision,
    MatchResult,
    _coerce_bool,
    _parse_decision,
    collect_match_decisions,
)
from simulator.personas import Persona


# ---------------------------------------------------------------------------
# _coerce_bool
# ---------------------------------------------------------------------------

def test_coerce_bool_native():
    assert _coerce_bool(True) is True
    assert _coerce_bool(False) is False


def test_coerce_bool_strings():
    assert _coerce_bool("true") is True
    assert _coerce_bool("Yes") is True
    assert _coerce_bool("no") is False
    assert _coerce_bool("reject") is False


def test_coerce_bool_unrecognised():
    assert _coerce_bool("maybe") is None
    assert _coerce_bool(None) is None


# ---------------------------------------------------------------------------
# _parse_decision
# ---------------------------------------------------------------------------

def test_parse_decision_plain_json():
    d = _parse_decision('{"wants_match": true, "reasoning": "great fit"}', "Jordan", "worker")
    assert d.wants_match is True
    assert d.reasoning == "great fit"
    assert d.persona_name == "Jordan"
    assert d.role == "worker"


def test_parse_decision_false():
    d = _parse_decision('{"wants_match": false, "reasoning": "no remote days"}', "Jordan", "worker")
    assert d.wants_match is False


def test_parse_decision_markdown_fenced():
    raw = '```json\n{"wants_match": true, "reasoning": "ok"}\n```'
    d = _parse_decision(raw, "Riley", "firm")
    assert d.wants_match is True


def test_parse_decision_bad_json_becomes_none():
    d = _parse_decision("I think yes but I'm not sure", "Jordan", "worker")
    assert d.wants_match is None
    assert d.reasoning == "I think yes but I'm not sure"


def test_parse_decision_missing_key_becomes_none():
    d = _parse_decision('{"reasoning": "forgot the field"}', "Jordan", "worker")
    assert d.wants_match is None


# ---------------------------------------------------------------------------
# collect_match_decisions — mutual-match logic (with a stub provider)
# ---------------------------------------------------------------------------

class _StubResult:
    def __init__(self, text):
        self.text = text
        self.input_tokens = 0
        self.output_tokens = 0


class _StubProvider:
    """Returns a queued response per call, keyed by call order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def call(self, messages, system_prompt, max_tokens=512, **kwargs):
        text = self._responses[self.calls]
        self.calls += 1
        return _StubResult(text)


def _turns():
    return [
        Turn(speaker="Jordan", message="hi", turn_index=0),
        Turn(speaker="Riley", message="hello", turn_index=1),
    ]


def _run(worker_text, firm_text):
    provider = _StubProvider([worker_text, firm_text])
    return collect_match_decisions(
        Persona(name="Jordan"),
        Persona(name="Riley"),
        _turns(),
        provider,
        tracker=None,
    )


def test_mutual_match_true_when_both_yes():
    r = _run('{"wants_match": true, "reasoning": "a"}', '{"wants_match": true, "reasoning": "b"}')
    assert isinstance(r, MatchResult)
    assert r.worker_decision.wants_match is True
    assert r.firm_decision.wants_match is True
    assert r.mutual_match is True


def test_mutual_match_false_when_one_says_no():
    r = _run('{"wants_match": true, "reasoning": "a"}', '{"wants_match": false, "reasoning": "b"}')
    assert r.mutual_match is False


def test_mutual_match_none_on_parse_error():
    r = _run('garbage', '{"wants_match": true, "reasoning": "b"}')
    assert r.worker_decision.wants_match is None
    assert r.mutual_match is None
