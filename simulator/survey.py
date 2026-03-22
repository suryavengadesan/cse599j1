"""
simulator/survey.py — Survey administration and change analysis.

Stateless functions that administer a multiple-choice survey to a persona
and diff pre/post results to identify opinion changes.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from simulator.personas import Persona, build_survey_prompt
from simulator.providers import Provider
from simulator.tracking import UsageTracker


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SurveyResponse:
    question_id: str
    question: str
    answer: str          # single letter A-D (or "?" if invalid)
    answer_text: str


@dataclass
class SurveyResult:
    persona_name: str
    stage: str           # "pre" or "post"
    responses: Dict[str, SurveyResponse]
    timestamp: str


@dataclass
class SurveyChange:
    question_id: str
    question: str
    pre_answer: str
    post_answer: str
    changed: bool


# ---------------------------------------------------------------------------
# Survey administration
# ---------------------------------------------------------------------------

_VALID_ANSWERS = {"A", "B", "C", "D"}


def administer_survey(
    persona: Persona,
    survey: dict,
    stage: str,
    provider: Provider,
    tracker: UsageTracker,
    question_ids: Optional[List[str]] = None,
    conversation_context: Optional[str] = None,
    verbose: bool = True,
) -> SurveyResult:
    """Administer a multiple-choice survey to *persona* and return a SurveyResult.

    Args:
        persona:              The persona taking the survey.
        survey:               Dict with a ``questions`` key mapping question IDs to
                              ``{question, options}`` dicts.
        stage:                ``"pre"`` or ``"post"``.
        provider:             Provider used for API calls.
        tracker:              UsageTracker that records token usage.
        question_ids:         Subset of question IDs to ask; None means all.
        conversation_context: Formatted conversation text for post-survey context.
        verbose:              If True, print questions and answers to stdout.

    Returns:
        SurveyResult with one SurveyResponse per question asked.
    """
    system_prompt = build_survey_prompt(persona, conversation_context)
    questions_to_ask = question_ids if question_ids else list(survey["questions"].keys())

    if verbose:
        print(f"\n{'='*60}")
        print(f"{stage.upper()}-CONVERSATION SURVEY: {persona.name}")
        print(f"{'='*60}\n")

    responses: Dict[str, SurveyResponse] = {}

    for q_id in questions_to_ask:
        if q_id not in survey["questions"]:
            print(f"Warning: Question ID '{q_id}' not found in survey. Skipping.")
            continue

        question_data = survey["questions"][q_id]
        question_text = question_data["question"]
        options = question_data["options"]

        # Format question with lettered options
        formatted = f"{question_text}\n\n"
        for opt_key, opt_text in options.items():
            formatted += f"{opt_key}) {opt_text}\n"
        formatted += "\nAnswer with only the letter (A, B, C, or D):"

        result = provider.call(
            messages=[{"role": "user", "content": formatted}],
            system_prompt=system_prompt,
            max_tokens=10,
            call_type="survey",
            persona_name=persona.name,
            question_id=q_id,
        )

        # Normalise answer: take first char, upper-case; fall back to "?"
        raw = result.text.strip().upper()
        answer = raw[0] if raw else "?"
        if answer not in _VALID_ANSWERS:
            print(f"Warning: unexpected survey answer {raw!r} for {q_id}; recording as '?'")
            answer = "?"

        answer_text = options.get(answer, "Invalid response")

        responses[q_id] = SurveyResponse(
            question_id=q_id,
            question=question_text,
            answer=answer,
            answer_text=answer_text,
        )

        if verbose:
            print(f"Q: {question_text}")
            print(f"A: {answer}) {answer_text}\n")

    return SurveyResult(
        persona_name=persona.name,
        stage=stage,
        responses=responses,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Change analysis
# ---------------------------------------------------------------------------


def analyze_changes(pre: SurveyResult, post: SurveyResult) -> List[SurveyChange]:
    """Diff *pre* and *post* SurveyResults and return a list of SurveyChange objects.

    Only questions present in *pre* are compared; questions that appear only in
    *post* are ignored.

    Args:
        pre:  SurveyResult from before the conversation.
        post: SurveyResult from after the conversation.

    Returns:
        List of SurveyChange, one per question in *pre*.
    """
    changes: List[SurveyChange] = []

    for q_id, pre_resp in pre.responses.items():
        post_resp = post.responses.get(q_id)
        if post_resp is None:
            # Question was not asked in post-survey; skip
            continue

        changed = pre_resp.answer != post_resp.answer
        changes.append(
            SurveyChange(
                question_id=q_id,
                question=pre_resp.question,
                pre_answer=pre_resp.answer,
                post_answer=post_resp.answer,
                changed=changed,
            )
        )

    return changes
