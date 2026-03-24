"""
run.py — CLI entry point for the conversation simulator.

Replaces the __main__ block in the legacy simulator.py.

Usage examples:
  python run.py --scenario seattle-sf --num-experiments 10 --num-turns 5
  python run.py --scenario seattle-sf --adversarial --provider huggingface
  python run.py --scenario seattle-sf --ablate '{"num_turns": [3, 5], "adversarial": [true, false]}'
"""

import argparse
import json
import sys

from dotenv import load_dotenv

from simulator.experiments import AblationGrid, ExperimentConfig, ExperimentRunner
from simulator.scenarios import ScenarioNotFoundError, list_scenarios

load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run conversation experiments with surveys",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--num-experiments",
        type=int,
        default=10,
        help="Number of experiments to run (default: 10)",
    )
    parser.add_argument(
        "--num-turns",
        type=int,
        default=5,
        help="Number of conversation turns per experiment (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed conversation output",
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        default=False,
        help="Use adversarial mode where the second persona tries to change the first's preference",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode to export token usage CSV after each run",
    )
    parser.add_argument(
        "--survey-questions",
        type=str,
        default=None,
        help='Comma-separated list of question IDs to ask (e.g. "q1,q3"). If not specified, asks all questions.',
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "huggingface"],
        help='API provider: "anthropic" (Claude) or "huggingface" (Qwen). Default: anthropic.',
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override (uses provider default if not specified)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario name to run (must match a YAML file in scenarios/). Use --show-survey to list available scenarios.",
    )
    parser.add_argument(
        "--show-survey",
        action="store_true",
        default=False,
        help="Print available scenarios and exit",
    )
    parser.add_argument(
        "--ablate",
        type=str,
        default=None,
        help=(
            'JSON string specifying ablation axes, e.g. \'{"num_turns": [3, 5], "adversarial": [true, false]}\'. '
            "Runs a full cartesian product of all specified values."
        ),
    )

    return parser


def print_config(args: argparse.Namespace, survey_questions) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("CONFIGURATION")
    print(sep)
    print(f"Provider:             {args.provider}")
    if args.model:
        print(f"Model:                {args.model}")
    print(f"Scenario:             {args.scenario}")
    print(f"Number of experiments:{args.num_experiments}")
    print(f"Turns per conversation:{args.num_turns}")
    print(f"Verbose mode:         {args.verbose}")
    print(f"Adversarial mode:     {args.adversarial}")
    print(f"Debug mode:           {args.debug}")
    print(f"Survey questions:     {survey_questions if survey_questions else 'All questions'}")
    if args.ablate:
        print(f"Ablation axes:        {args.ablate}")
    print(f"{sep}\n")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    available = list_scenarios()

    # --show-survey: list scenarios and exit
    if args.show_survey:
        if available:
            print("Available scenarios:")
            for name in sorted(available):
                print(f"  {name}")
        else:
            print("No scenarios found in scenarios/ directory.")
        return 0

    # Require --scenario for all other operations
    if args.scenario is None:
        parser.error("--scenario is required. Use --show-survey to list available scenarios.")

    # Validate scenario name early so we can give a helpful error
    if args.scenario not in available:
        print(f"Error: unknown scenario {args.scenario!r}.", file=sys.stderr)
        if available:
            print("Available scenarios:", file=sys.stderr)
            for name in sorted(available):
                print(f"  {name}", file=sys.stderr)
        else:
            print("No scenarios found in scenarios/ directory.", file=sys.stderr)
        return 1

    # Parse survey questions
    survey_questions = None
    if args.survey_questions:
        survey_questions = [q.strip() for q in args.survey_questions.split(",")]

    print_config(args, survey_questions)

    # Build base ExperimentConfig
    base_config = ExperimentConfig(
        scenario_name=args.scenario,
        provider=args.provider,
        model=args.model,
        num_turns=args.num_turns,
        adversarial=args.adversarial,
        survey_questions=survey_questions,
        verbose=args.verbose,
        debug=args.debug,
    )

    # --ablate path
    if args.ablate:
        try:
            axes = json.loads(args.ablate)
        except json.JSONDecodeError as exc:
            print(f"Error: --ablate value is not valid JSON: {exc}", file=sys.stderr)
            return 1

        if not isinstance(axes, dict):
            print("Error: --ablate must be a JSON object mapping parameter names to lists of values.", file=sys.stderr)
            return 1

        try:
            grid = AblationGrid(base_config=base_config, axes=axes)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        results = grid.run(num_experiments=args.num_experiments)
        output_path = grid.export(results)
        print(f"\nAblation results written to: {output_path}")
        return 0

    # Standard experiment batch
    runner = ExperimentRunner(base_config)
    results = runner.run_many(args.num_experiments)
    summary = runner.summarize(results)
    output_path = runner.export(results, summary)
    print(f"\nResults written to: {output_path}")

    # Print a brief summary
    sc = summary.get("survey_changes", {})
    changed = sc.get("changed_answers", {})
    print(f"\nSummary: {summary['num_experiments']} experiments")
    if changed:
        print(f"  Mean changed answers per run: {changed.get('mean', 0):.2f}")
        print(f"  Median: {changed.get('median', 0):.2f}  Stdev: {changed.get('stdev', 0):.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
