"""
simulator/rlhf_data.py — Export experiment results as RLHF training data.

Converts ExperimentResult objects into formats consumable by Tinker's
SFT, DPO, and RL training pipelines.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from simulator.conversation import Turn
from simulator.experiments import ExperimentResult
from simulator.judge import JudgeResult
from simulator.personas import Persona, build_conversation_prompt
from simulator.survey import SurveyChange


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------


@dataclass
class RewardedTrajectory:
    """A single adversarial conversation with a computed reward score."""

    experiment_id: int
    scenario_name: str
    system_prompt: str          # adversarial persona's system prompt
    prompt_turns: List[dict]    # chat messages leading to each response
    completions: List[str]      # adversarial persona's responses
    reward: float               # composite reward signal
    survey_drift: float         # fraction of survey answers that changed
    judge_score: Optional[int]  # 0-3 drift score from judge, or None
    metadata: Dict[str, Any]


def compute_reward(
    changes: List[SurveyChange],
    judge_result: Optional[JudgeResult] = None,
    survey_weight: float = 0.6,
    judge_weight: float = 0.4,
) -> Tuple[float, float, Optional[int]]:
    """Compute a scalar reward from survey drift and optional judge score.

    Returns (reward, survey_drift, judge_score).
    """
    total_questions = len(changes) if changes else 1
    changed_count = sum(1 for c in changes if c.changed)
    survey_drift = changed_count / total_questions

    judge_score: Optional[int] = None
    if judge_result is not None:
        # Use persona_a's drift score (the target of persuasion)
        judge_score = judge_result.persona_a_assessment.score
        if judge_score < 0:
            judge_score = 0  # treat parse errors as no drift
        normalised_judge = judge_score / 3.0
        reward = survey_weight * survey_drift + judge_weight * normalised_judge
    else:
        # No judge — reward is purely survey-based
        reward = survey_drift

    return reward, survey_drift, judge_score


# ---------------------------------------------------------------------------
# Trajectory extraction
# ---------------------------------------------------------------------------


def extract_trajectory(result: ExperimentResult) -> RewardedTrajectory:
    """Extract a RewardedTrajectory from a single ExperimentResult.

    The adversarial persona is persona_b. We extract its system prompt
    and all its conversation turns as completions, paired with the
    preceding context as prompts.
    """
    from simulator.scenarios import load_scenario

    scenario = load_scenario(
        result.config.scenario_name, adversarial=result.config.adversarial
    )
    persona_b = scenario.persona_b
    system_prompt = build_conversation_prompt(persona_b)

    # Build prompt/completion pairs for persona_b's turns
    prompt_turns: List[dict] = []
    completions: List[str] = []

    context: List[dict] = []
    for turn in result.conversation:
        if turn.speaker == persona_b.name:
            # The context so far is the prompt; this turn is the completion
            prompt_turns.append({"messages": list(context)})
            completions.append(turn.message)
        # Add this turn to running context for the next completion
        role = "assistant" if turn.speaker == persona_b.name else "user"
        context.append({"role": role, "content": turn.message})

    reward, survey_drift, judge_score = compute_reward(
        result.changes, result.judge
    )

    return RewardedTrajectory(
        experiment_id=result.experiment_id,
        scenario_name=result.config.scenario_name,
        system_prompt=system_prompt,
        prompt_turns=prompt_turns,
        completions=completions,
        reward=reward,
        survey_drift=survey_drift,
        judge_score=judge_score,
        metadata={
            "num_turns": result.config.num_turns,
            "provider": result.config.provider,
            "changed_answers": sum(1 for c in result.changes if c.changed),
            "total_questions": len(result.changes),
            "timestamp": result.timestamp,
        },
    )


# ---------------------------------------------------------------------------
# SFT data export
# ---------------------------------------------------------------------------


def export_sft_data(
    results: List[ExperimentResult],
    min_reward: float = 0.5,
    output_path: Optional[str] = None,
) -> str:
    """Export high-reward conversations as SFT training data.

    Filters for trajectories with reward >= min_reward and formats them
    as chat-completion examples for Tinker's SFT pipeline.

    Returns the path to the written JSONL file.
    """
    trajectories = [extract_trajectory(r) for r in results]
    good = [t for t in trajectories if t.reward >= min_reward]

    if output_path is None:
        os.makedirs("results/training_data", exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = f"results/training_data/sft_data_{ts}.jsonl"

    with open(output_path, "w", encoding="utf-8") as fh:
        for traj in good:
            for prompt_ctx, completion in zip(traj.prompt_turns, traj.completions):
                messages = (
                    [{"role": "system", "content": traj.system_prompt}]
                    + prompt_ctx["messages"]
                    + [{"role": "assistant", "content": completion}]
                )
                record = {
                    "messages": messages,
                    "reward": traj.reward,
                    "experiment_id": traj.experiment_id,
                    "scenario": traj.scenario_name,
                }
                fh.write(json.dumps(record) + "\n")

    print(f"SFT data: {len(good)}/{len(trajectories)} trajectories "
          f"(reward >= {min_reward}) written to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# DPO pairs export
# ---------------------------------------------------------------------------


def export_dpo_pairs(
    results: List[ExperimentResult],
    output_path: Optional[str] = None,
) -> str:
    """Export preference pairs for DPO training.

    For each scenario, pairs the highest-reward trajectory (chosen) with
    the lowest-reward trajectory (rejected) across experiments.

    Returns the path to the written JSONL file.
    """
    trajectories = [extract_trajectory(r) for r in results]

    # Group by scenario
    by_scenario: Dict[str, List[RewardedTrajectory]] = {}
    for t in trajectories:
        by_scenario.setdefault(t.scenario_name, []).append(t)

    if output_path is None:
        os.makedirs("results/training_data", exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = f"results/training_data/dpo_pairs_{ts}.jsonl"

    pair_count = 0
    with open(output_path, "w", encoding="utf-8") as fh:
        for scenario_name, trajs in by_scenario.items():
            if len(trajs) < 2:
                continue
            sorted_trajs = sorted(trajs, key=lambda t: t.reward, reverse=True)
            # Pair top half with bottom half
            mid = len(sorted_trajs) // 2
            chosen_set = sorted_trajs[:mid]
            rejected_set = sorted_trajs[mid:]

            for chosen, rejected in zip(chosen_set, rejected_set):
                if chosen.reward <= rejected.reward:
                    continue
                # Use the first turn's context as the shared prompt
                if not chosen.prompt_turns or not rejected.prompt_turns:
                    continue
                prompt_messages = (
                    [{"role": "system", "content": chosen.system_prompt}]
                    + chosen.prompt_turns[0]["messages"]
                )
                record = {
                    "prompt": prompt_messages,
                    "chosen": chosen.completions[0] if chosen.completions else "",
                    "rejected": rejected.completions[0] if rejected.completions else "",
                    "chosen_reward": chosen.reward,
                    "rejected_reward": rejected.reward,
                    "scenario": scenario_name,
                }
                fh.write(json.dumps(record) + "\n")
                pair_count += 1

    print(f"DPO pairs: {pair_count} pairs written to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# RL reward data export
# ---------------------------------------------------------------------------


def export_rl_prompts(
    results: List[ExperimentResult],
    output_path: Optional[str] = None,
) -> str:
    """Export prompts and reward signals for the RL training loop.

    Each record contains the scenario prompt context and the reward
    function parameters needed to score new rollouts.

    Returns the path to the written JSONL file.
    """
    trajectories = [extract_trajectory(r) for r in results]

    if output_path is None:
        os.makedirs("results/training_data", exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = f"results/training_data/rl_prompts_{ts}.jsonl"

    with open(output_path, "w", encoding="utf-8") as fh:
        for traj in trajectories:
            if not traj.prompt_turns:
                continue
            record = {
                "system_prompt": traj.system_prompt,
                "initial_context": traj.prompt_turns[0]["messages"],
                "scenario": traj.scenario_name,
                "reference_reward": traj.reward,
                "reference_completions": traj.completions,
            }
            fh.write(json.dumps(record) + "\n")

    print(f"RL prompts: {len(trajectories)} records written to {output_path}")
    return output_path
