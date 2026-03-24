"""
simulator/experiments.py — ExperimentConfig, ExperimentRunner, AblationGrid.

Provides the high-level experiment infrastructure:
  - ExperimentConfig: all parameters needed to reproduce a single run
  - ExperimentResult: self-describing result that embeds its config
  - ExperimentRunner: runs one or many experiments and summarises results
  - AblationGrid: sweeps a cartesian product of config dimensions
"""

import json
import os
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
from typing import Any, Dict, List, Optional

from simulator.conversation import Turn, run as run_conversation
from simulator.personas import Persona
from simulator.providers import get_provider
from simulator.scenarios import load_scenario
from simulator.survey import SurveyChange, SurveyResult, administer_survey, analyze_changes
from simulator.tracking import UsageTracker


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """All parameters needed to reproduce a single experiment."""

    scenario_name: str
    provider: str = "anthropic"
    model: Optional[str] = None          # None → provider default
    num_turns: int = 5
    adversarial: bool = False
    survey_questions: Optional[List[str]] = None
    verbose: bool = False
    debug: bool = False


@dataclass
class ExperimentResult:
    """Self-describing result that embeds its full ExperimentConfig."""

    experiment_id: int
    config: ExperimentConfig             # full config for self-describing results
    conversation: List[Turn]
    pre_survey: SurveyResult
    post_survey: SurveyResult
    changes: List[SurveyChange]
    tokens: Dict                         # input/output/total for this run
    timestamp: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize(obj: Any) -> Any:
    """Recursively convert dataclasses / Turn / SurveyResult etc. to dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def _conversation_to_text(turns: List[Turn]) -> str:
    """Format a list of Turns as a readable string for post-survey context."""
    lines = []
    for t in turns:
        lines.append(f"{t.speaker}: {t.message}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    """Runs one or many experiments for a given ExperimentConfig."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._experiment_counter = 0

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------

    def run_one(self) -> ExperimentResult:
        """Execute one full experiment: pre-survey → conversation → post-survey.

        Returns an ExperimentResult with the full config embedded.
        """
        self._experiment_counter += 1
        exp_id = self._experiment_counter

        cfg = self.config
        tracker = UsageTracker()

        # Build provider kwargs
        provider_kwargs: Dict[str, Any] = {"tracker": tracker}
        if cfg.model is not None:
            provider_kwargs["model"] = cfg.model

        provider = get_provider(cfg.provider, **provider_kwargs)

        # Load scenario (adversarial flag selects persona variants)
        scenario = load_scenario(cfg.scenario_name, adversarial=cfg.adversarial)

        persona_a = scenario.persona_a
        persona_b = scenario.persona_b

        # Pre-survey
        pre_survey = administer_survey(
            persona=persona_a,
            survey=scenario.survey,
            stage="pre",
            provider=provider,
            tracker=tracker,
            question_ids=cfg.survey_questions,
            verbose=cfg.verbose,
        )

        # Conversation
        turns = run_conversation(
            persona_a=persona_a,
            persona_b=persona_b,
            initial_message=scenario.initial_message,
            num_turns=cfg.num_turns,
            provider=provider,
            tracker=tracker,
            verbose=cfg.verbose,
        )

        # Post-survey (with conversation context)
        conversation_context = _conversation_to_text(turns)
        post_survey = administer_survey(
            persona=persona_a,
            survey=scenario.survey,
            stage="post",
            provider=provider,
            tracker=tracker,
            question_ids=cfg.survey_questions,
            conversation_context=conversation_context,
            verbose=cfg.verbose,
        )

        changes = analyze_changes(pre_survey, post_survey)

        token_summary = tracker.summary()
        tokens = {
            "input": token_summary["total_input_tokens"],
            "output": token_summary["total_output_tokens"],
            "total": token_summary["total_tokens"],
        }

        if cfg.debug:
            tracker.export_csv()

        return ExperimentResult(
            experiment_id=exp_id,
            config=cfg,
            conversation=turns,
            pre_survey=pre_survey,
            post_survey=post_survey,
            changes=changes,
            tokens=tokens,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Batch run
    # ------------------------------------------------------------------

    def run_many(self, num_experiments: int) -> List[ExperimentResult]:
        """Run *num_experiments* repetitions, printing progress to stdout.

        Prints ``[Experiment K/N]`` before each run.
        """
        results: List[ExperimentResult] = []
        for k in range(1, num_experiments + 1):
            print(f"[Experiment {k}/{num_experiments}]")
            results.append(self.run_one())
        return results

    # ------------------------------------------------------------------
    # Summarise
    # ------------------------------------------------------------------

    def summarize(self, results: List[ExperimentResult]) -> Dict:
        """Compute per-question change rates and aggregate token statistics.

        Returns a dict with:
          - num_experiments
          - survey_changes: per-question rates + mean/median/stdev of changed count
          - tokens: mean input/output/total across runs
        """
        if not results:
            return {
                "num_experiments": 0,
                "survey_changes": {},
                "tokens": {"mean_input": 0, "mean_output": 0, "mean_total": 0},
            }

        # Per-question change counts
        question_changed: Dict[str, int] = {}
        question_total: Dict[str, int] = {}
        changed_counts_per_run: List[int] = []

        for r in results:
            run_changed = 0
            for ch in r.changes:
                question_total[ch.question_id] = question_total.get(ch.question_id, 0) + 1
                if ch.changed:
                    question_changed[ch.question_id] = question_changed.get(ch.question_id, 0) + 1
                    run_changed += 1
            changed_counts_per_run.append(run_changed)

        per_question = {
            q_id: {
                "changed": question_changed.get(q_id, 0),
                "total": question_total[q_id],
                "change_rate": question_changed.get(q_id, 0) / question_total[q_id],
            }
            for q_id in question_total
        }

        mean_changed = statistics.mean(changed_counts_per_run) if changed_counts_per_run else 0.0
        median_changed = statistics.median(changed_counts_per_run) if changed_counts_per_run else 0.0
        stdev_changed = (
            statistics.stdev(changed_counts_per_run) if len(changed_counts_per_run) > 1 else 0.0
        )

        # Token aggregates
        mean_input = statistics.mean(r.tokens["input"] for r in results)
        mean_output = statistics.mean(r.tokens["output"] for r in results)
        mean_total = statistics.mean(r.tokens["total"] for r in results)

        return {
            "num_experiments": len(results),
            "survey_changes": {
                "per_question": per_question,
                "changed_answers": {
                    "mean": mean_changed,
                    "median": median_changed,
                    "stdev": stdev_changed,
                },
            },
            "tokens": {
                "mean_input": mean_input,
                "mean_output": mean_output,
                "mean_total": mean_total,
            },
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(
        self,
        results: List[ExperimentResult],
        summary: Dict,
        filepath: Optional[str] = None,
    ) -> str:
        """Write results + summary to JSON.

        If *filepath* is None, writes to
        ``results/experiments/{scenario_name}_{timestamp}.json``.

        Returns the path written.
        """
        if filepath is None:
            os.makedirs("results/experiments", exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            scenario = self.config.scenario_name
            filepath = f"results/experiments/{scenario}_{ts}.json"

        payload = {
            "summary": summary,
            "results": [_serialize(r) for r in results],
        }

        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

        return filepath


# ---------------------------------------------------------------------------
# AblationGrid
# ---------------------------------------------------------------------------

_ABLATABLE_FIELDS = frozenset(
    {"num_turns", "adversarial", "provider", "scenario_name", "survey_questions", "model"}
)


class AblationGrid:
    """Sweeps a cartesian product of ExperimentConfig dimensions.

    Parameters
    ----------
    base_config:
        The baseline ExperimentConfig; axis values override individual fields.
    axes:
        Dict mapping field names to lists of values to sweep, e.g.
        ``{"num_turns": [3, 5, 10], "adversarial": [True, False]}``.
    """

    def __init__(
        self,
        base_config: ExperimentConfig,
        axes: Dict[str, List[Any]],
    ) -> None:
        invalid = set(axes) - _ABLATABLE_FIELDS
        if invalid:
            raise ValueError(
                f"Invalid ablation parameter(s): {sorted(invalid)}. "
                f"Valid parameters: {sorted(_ABLATABLE_FIELDS)}"
            )
        self.base_config = base_config
        self.axes = axes

    # ------------------------------------------------------------------
    # Conditions
    # ------------------------------------------------------------------

    def conditions(self) -> List[ExperimentConfig]:
        """Return the full cartesian product of axis values as ExperimentConfigs.

        Each config is a copy of *base_config* with the axis fields overridden.
        """
        if not self.axes:
            return [ExperimentConfig(**asdict(self.base_config))]

        param_names = sorted(self.axes)          # sorted for deterministic labels
        value_lists = [self.axes[p] for p in param_names]

        configs: List[ExperimentConfig] = []
        for combo in product(*value_lists):
            overrides = dict(zip(param_names, combo))
            base_dict = asdict(self.base_config)
            base_dict.update(overrides)
            configs.append(ExperimentConfig(**base_dict))

        return configs

    def _condition_label(self, config: ExperimentConfig) -> str:
        """Build a human-readable label for a condition config.

        Format: ``"param1=val1,param2=val2"`` sorted by param name.
        """
        parts = []
        for param in sorted(self.axes):
            val = getattr(config, param)
            parts.append(f"{param}={val}")
        return ",".join(parts)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, num_experiments: int) -> Dict[str, Any]:
        """Run each condition *num_experiments* times.

        Returns a dict keyed by condition label, each value containing
        ``{"results": [...], "summary": {...}}``.
        """
        all_conditions = self.conditions()
        output: Dict[str, Any] = {}

        for i, cfg in enumerate(all_conditions, start=1):
            label = self._condition_label(cfg)
            print(f"\n[Condition {i}/{len(all_conditions)}] {label}")
            runner = ExperimentRunner(cfg)
            results = runner.run_many(num_experiments)
            summary = runner.summarize(results)
            output[label] = {
                "config": asdict(cfg),
                "summary": summary,
                "results": [_serialize(r) for r in results],
            }

        return output

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, results: Dict, filepath: Optional[str] = None) -> str:
        """Write the full ablation results dict to a single JSON file.

        If *filepath* is None, writes to
        ``results/experiments/ablation_{timestamp}.json``.

        Returns the path written.
        """
        if filepath is None:
            os.makedirs("results/experiments", exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filepath = f"results/experiments/ablation_{ts}.json"

        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)

        return filepath
