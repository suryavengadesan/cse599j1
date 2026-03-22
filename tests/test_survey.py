"""
Tests for simulator/survey.py — analyze_changes.
"""

import pytest
from simulator.personas import Persona
from simulator.survey import SurveyResponse, SurveyResult, SurveyChange, analyze_changes
from datetime import datetime, timezone


def _make_result(persona_name, stage, answers: dict) -> SurveyResult:
    """Helper: build a SurveyResult from a {q_id: answer} dict."""
    responses = {
        q_id: SurveyResponse(
            question_id=q_id,
            question=f"Question {q_id}",
            answer=answer,
            answer_text=f"Option {answer}",
        )
        for q_id, answer in answers.items()
    }
    return SurveyResult(
        persona_name=persona_name,
        stage=stage,
        responses=responses,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def test_analyze_changes_no_changes():
    pre = _make_result("Alice", "pre", {"q1": "A", "q2": "B"})
    post = _make_result("Alice", "post", {"q1": "A", "q2": "B"})
    changes = analyze_changes(pre, post)
    assert len(changes) == 2
    assert all(not c.changed for c in changes)


def test_analyze_changes_all_changed():
    pre = _make_result("Alice", "pre", {"q1": "A", "q2": "B"})
    post = _make_result("Alice", "post", {"q1": "C", "q2": "D"})
    changes = analyze_changes(pre, post)
    assert len(changes) == 2
    assert all(c.changed for c in changes)


def test_analyze_changes_partial():
    pre = _make_result("Alice", "pre", {"q1": "A", "q2": "B", "q3": "C"})
    post = _make_result("Alice", "post", {"q1": "A", "q2": "D", "q3": "C"})
    changes = analyze_changes(pre, post)
    changed = [c for c in changes if c.changed]
    unchanged = [c for c in changes if not c.changed]
    assert len(changed) == 1
    assert changed[0].question_id == "q2"
    assert len(unchanged) == 2


def test_analyze_changes_missing_post_question():
    """Questions in pre but not in post are skipped."""
    pre = _make_result("Alice", "pre", {"q1": "A", "q2": "B"})
    post = _make_result("Alice", "post", {"q1": "C"})
    changes = analyze_changes(pre, post)
    assert len(changes) == 1
    assert changes[0].question_id == "q1"
