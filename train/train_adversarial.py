"""
train_adversarial.py — RLHF training pipeline for adversarial personas via Tinker.

Three-phase pipeline:
  1. SFT warm-start on high-reward conversations
  2. DPO preference training (optional, alternative to phase 3)
  3. RL with judge-as-reward on sampled rollouts

Usage:
  # Phase 1: Collect training data from experiments
  python train_adversarial.py collect --scenario seattle-sf --num-experiments 30

  # Phase 2: SFT warm-start
  python train_adversarial.py sft --data results/training_data/sft_data_*.jsonl

  # Phase 3a: DPO training
  python train_adversarial.py dpo --data results/training_data/dpo_pairs_*.jsonl

  # Phase 3b: RL with judge reward
  python train_adversarial.py rl --data results/training_data/rl_prompts_*.jsonl

  # Evaluate fine-tuned model
  python train_adversarial.py eval --checkpoint <path> --scenario seattle-sf
"""

import argparse
import json
import os
import sys
from glob import glob
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Phase 1: Data collection
# ---------------------------------------------------------------------------


def collect_data(args: argparse.Namespace) -> int:
    """Run experiments and export training data in all formats."""
    from simulator.experiments import ExperimentConfig, ExperimentRunner
    from simulator.rlhf_data import export_dpo_pairs, export_rl_prompts, export_sft_data

    scenarios = [s.strip() for s in args.scenario.split(",")]
    all_results = []

    for scenario_name in scenarios:
        print(f"\n{'='*60}")
        print(f"Collecting data from scenario: {scenario_name}")
        print(f"{'='*60}")

        config = ExperimentConfig(
            scenario_name=scenario_name,
            provider=args.provider,
            model=args.model,
            num_turns=args.num_turns,
            adversarial=True,  # always adversarial for training data
            verbose=args.verbose,
            judge=True,  # need judge scores for reward
        )
        runner = ExperimentRunner(config)
        results = runner.run_many(args.num_experiments)
        all_results.extend(results)

        summary = runner.summarize(results)
        sc = summary.get("survey_changes", {})
        changed = sc.get("changed_answers", {})
        print(f"  {len(results)} experiments completed")
        if changed:
            print(f"  Mean changed answers: {changed.get('mean', 0):.2f}")

    # Export all three formats
    print(f"\n{'='*60}")
    print("Exporting training data")
    print(f"{'='*60}")

    sft_path = export_sft_data(all_results, min_reward=args.min_reward)
    dpo_path = export_dpo_pairs(all_results)
    rl_path = export_rl_prompts(all_results)

    print(f"\nData collection complete. Files:")
    print(f"  SFT:  {sft_path}")
    print(f"  DPO:  {dpo_path}")
    print(f"  RL:   {rl_path}")
    return 0


# ---------------------------------------------------------------------------
# Phase 2: SFT warm-start
# ---------------------------------------------------------------------------


def _load_jsonl(path: str) -> List[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_sft(args: argparse.Namespace) -> int:
    """SFT warm-start on high-reward conversations using Tinker."""
    try:
        from tinker import ServiceInterface, TrainingClient
        from tinker.types import EncodedTextChunk, ModelInput
    except ImportError:
        print("Error: 'tinker' package required. Install with: pip install tinker",
              file=sys.stderr)
        return 1

    data_path = args.data
    if "*" in data_path:
        matches = sorted(glob(data_path))
        if not matches:
            print(f"Error: no files matching {data_path}", file=sys.stderr)
            return 1
        data_path = matches[-1]  # use most recent
        print(f"Using data file: {data_path}")

    records = _load_jsonl(data_path)
    if not records:
        print("Error: no training records found.", file=sys.stderr)
        return 1

    print(f"Loaded {len(records)} SFT examples")

    # Create Tinker training client
    client = TrainingClient(model_name=args.base_model)

    # Format data for Tinker
    training_inputs = []
    for rec in records:
        # Convert chat messages to a single formatted string
        text = ""
        for msg in rec["messages"]:
            role = msg["role"]
            content = msg["content"]
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        training_inputs.append(
            ModelInput(chunks=[EncodedTextChunk(text=text)])
        )

    # Training loop
    print(f"\nStarting SFT on {args.base_model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(training_inputs), args.batch_size):
            batch = training_inputs[i : i + args.batch_size]

            fb_future = client.forward_backward(
                inputs=batch,
                loss_type="sft",
            )
            fb_result = fb_future.result()
            epoch_loss += fb_result.loss

            step_future = client.optim_step(learning_rate=args.learning_rate)
            step_future.result()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"  Epoch {epoch + 1}/{args.epochs} — loss: {avg_loss:.4f}")

    # Save checkpoint
    save_path = args.save_path or f"checkpoints/sft_{args.base_model.replace('/', '_')}"
    os.makedirs(os.path.dirname(save_path) or "checkpoints", exist_ok=True)
    client.save_weights(save_path)
    print(f"\nSFT checkpoint saved to: {save_path}")
    return 0


# ---------------------------------------------------------------------------
# Phase 3a: DPO training
# ---------------------------------------------------------------------------


def run_dpo(args: argparse.Namespace) -> int:
    """DPO preference training using Tinker."""
    try:
        from tinker import TrainingClient
        from tinker.types import EncodedTextChunk, ModelInput
    except ImportError:
        print("Error: 'tinker' package required. Install with: pip install tinker",
              file=sys.stderr)
        return 1

    data_path = args.data
    if "*" in data_path:
        matches = sorted(glob(data_path))
        if not matches:
            print(f"Error: no files matching {data_path}", file=sys.stderr)
            return 1
        data_path = matches[-1]

    records = _load_jsonl(data_path)
    if not records:
        print("Error: no DPO pairs found.", file=sys.stderr)
        return 1

    print(f"Loaded {len(records)} DPO pairs")

    # Load from SFT checkpoint if provided
    client = TrainingClient(model_name=args.base_model)
    if args.sft_checkpoint:
        client.load_weights(args.sft_checkpoint)
        print(f"Loaded SFT checkpoint: {args.sft_checkpoint}")

    print(f"\nStarting DPO on {args.base_model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  DPO beta: {args.dpo_beta}")
    print(f"  Learning rate: {args.learning_rate}")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(records), args.batch_size):
            batch = records[i : i + args.batch_size]

            chosen_inputs = []
            rejected_inputs = []
            for rec in batch:
                # Build prompt prefix from the shared context
                prompt_text = ""
                for msg in rec["prompt"]:
                    prompt_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                prompt_text += "<|im_start|>assistant\n"

                chosen_text = prompt_text + rec["chosen"] + "<|im_end|>\n"
                rejected_text = prompt_text + rec["rejected"] + "<|im_end|>\n"

                chosen_inputs.append(
                    ModelInput(chunks=[EncodedTextChunk(text=chosen_text)])
                )
                rejected_inputs.append(
                    ModelInput(chunks=[EncodedTextChunk(text=rejected_text)])
                )

            fb_future = client.forward_backward(
                inputs=chosen_inputs,
                rejected_inputs=rejected_inputs,
                loss_type="dpo",
                dpo_beta=args.dpo_beta,
            )
            fb_result = fb_future.result()
            epoch_loss += fb_result.loss

            step_future = client.optim_step(learning_rate=args.learning_rate)
            step_future.result()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"  Epoch {epoch + 1}/{args.epochs} — loss: {avg_loss:.4f}")

    save_path = args.save_path or f"checkpoints/dpo_{args.base_model.replace('/', '_')}"
    os.makedirs(os.path.dirname(save_path) or "checkpoints", exist_ok=True)
    client.save_weights(save_path)
    print(f"\nDPO checkpoint saved to: {save_path}")
    return 0


# ---------------------------------------------------------------------------
# Phase 3b: RL with judge-as-reward
# ---------------------------------------------------------------------------


def run_rl(args: argparse.Namespace) -> int:
    """RL training using the LLM judge as a reward signal."""
    try:
        from tinker import TrainingClient
        from tinker.types import EncodedTextChunk, ModelInput
    except ImportError:
        print("Error: 'tinker' package required. Install with: pip install tinker",
              file=sys.stderr)
        return 1

    from simulator.judge import judge_persona
    from simulator.personas import Persona
    from simulator.providers import get_provider
    from simulator.tracking import UsageTracker

    data_path = args.data
    if "*" in data_path:
        matches = sorted(glob(data_path))
        if not matches:
            print(f"Error: no files matching {data_path}", file=sys.stderr)
            return 1
        data_path = matches[-1]

    records = _load_jsonl(data_path)
    if not records:
        print("Error: no RL prompts found.", file=sys.stderr)
        return 1

    print(f"Loaded {len(records)} RL prompts")

    # Policy model on Tinker
    client = TrainingClient(model_name=args.base_model)
    if args.sft_checkpoint:
        client.load_weights(args.sft_checkpoint)
        print(f"Loaded checkpoint: {args.sft_checkpoint}")

    # Judge provider (uses Anthropic by default for quality)
    judge_tracker = UsageTracker()
    judge_provider = get_provider(
        args.judge_provider, tracker=judge_tracker
    )

    print(f"\nStarting RL training on {args.base_model}")
    print(f"  Iterations: {args.rl_iterations}")
    print(f"  Samples per prompt: {args.num_samples}")
    print(f"  Judge provider: {args.judge_provider}")
    print(f"  Learning rate: {args.learning_rate}")

    for iteration in range(args.rl_iterations):
        total_reward = 0.0
        num_updates = 0

        for rec in records:
            # Build the prompt
            prompt = f"<|im_start|>system\n{rec['system_prompt']}<|im_end|>\n"
            for msg in rec["initial_context"]:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

            # Sample multiple completions from the policy
            completions = client.sample(
                prompts=[prompt] * args.num_samples,
                max_new_tokens=256,
                temperature=0.8,
            )

            # Score each completion using the judge
            rewards = []
            for completion_text in completions:
                # Create a minimal Turn list for the judge
                from simulator.conversation import Turn

                turns = [
                    Turn(speaker="target", message=msg["content"], turn_index=i)
                    for i, msg in enumerate(rec["initial_context"])
                    if msg["role"] == "user"
                ]
                turns.append(
                    Turn(
                        speaker="adversarial",
                        message=completion_text,
                        turn_index=len(turns),
                    )
                )

                assessment = judge_persona(
                    persona=Persona(name="target"),
                    turns=[t for t in turns if t.speaker == "target"],
                    provider=judge_provider,
                    tracker=judge_tracker,
                )
                score = max(assessment.score, 0) / 3.0
                rewards.append(score)

            # Compute advantage: reward relative to mean
            mean_reward = sum(rewards) / len(rewards) if rewards else 0
            advantages = [r - mean_reward for r in rewards]

            # Update policy with REINFORCE-style gradient
            for completion_text, advantage in zip(completions, advantages):
                full_text = prompt + completion_text + "<|im_end|>\n"
                model_input = ModelInput(
                    chunks=[EncodedTextChunk(text=full_text)]
                )

                fb_future = client.forward_backward(
                    inputs=[model_input],
                    loss_type="reinforce",
                    rewards=[advantage],
                )
                fb_future.result()

            step_future = client.optim_step(learning_rate=args.learning_rate)
            step_future.result()

            total_reward += mean_reward
            num_updates += 1

        avg_reward = total_reward / max(num_updates, 1)
        print(f"  Iteration {iteration + 1}/{args.rl_iterations} — "
              f"avg reward: {avg_reward:.4f}")

    save_path = args.save_path or f"checkpoints/rl_{args.base_model.replace('/', '_')}"
    os.makedirs(os.path.dirname(save_path) or "checkpoints", exist_ok=True)
    client.save_weights(save_path)

    judge_summary = judge_tracker.summary()
    print(f"\nRL checkpoint saved to: {save_path}")
    print(f"Judge API calls: {judge_summary['total_api_calls']}")
    return 0


# ---------------------------------------------------------------------------
# Phase 4: Evaluation
# ---------------------------------------------------------------------------


def run_eval(args: argparse.Namespace) -> int:
    """Evaluate a fine-tuned model against the base model."""
    from simulator.experiments import ExperimentConfig, ExperimentRunner

    scenarios = [s.strip() for s in args.scenario.split(",")]

    for scenario_name in scenarios:
        print(f"\n{'='*60}")
        print(f"Evaluating on: {scenario_name}")
        print(f"{'='*60}")

        # Run with fine-tuned model
        ft_config = ExperimentConfig(
            scenario_name=scenario_name,
            provider="tinkr",
            num_turns=args.num_turns,
            adversarial=True,
            judge=True,
            verbose=args.verbose,
        )
        ft_runner = ExperimentRunner(ft_config)
        ft_results = ft_runner.run_many(args.num_experiments)
        ft_summary = ft_runner.summarize(ft_results)

        # Run with base model for comparison
        base_config = ExperimentConfig(
            scenario_name=scenario_name,
            provider=args.base_provider,
            num_turns=args.num_turns,
            adversarial=True,
            judge=True,
            verbose=args.verbose,
        )
        base_runner = ExperimentRunner(base_config)
        base_results = base_runner.run_many(args.num_experiments)
        base_summary = base_runner.summarize(base_results)

        # Compare
        ft_changes = ft_summary.get("survey_changes", {}).get("changed_answers", {})
        base_changes = base_summary.get("survey_changes", {}).get("changed_answers", {})

        print(f"\n  Fine-tuned model:")
        print(f"    Mean changed answers: {ft_changes.get('mean', 0):.2f}")
        print(f"    Median: {ft_changes.get('median', 0):.2f}")
        print(f"  Base model ({args.base_provider}):")
        print(f"    Mean changed answers: {base_changes.get('mean', 0):.2f}")
        print(f"    Median: {base_changes.get('median', 0):.2f}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RLHF training pipeline for adversarial personas via Tinker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- collect --
    collect_p = subparsers.add_parser("collect", help="Run experiments and export training data")
    collect_p.add_argument("--scenario", required=True, help="Comma-separated scenario names")
    collect_p.add_argument("--num-experiments", type=int, default=30)
    collect_p.add_argument("--num-turns", type=int, default=5)
    collect_p.add_argument("--provider", default="anthropic")
    collect_p.add_argument("--model", default=None)
    collect_p.add_argument("--min-reward", type=float, default=0.5,
                           help="Minimum reward threshold for SFT data")
    collect_p.add_argument("--verbose", action="store_true", default=False)

    # -- sft --
    sft_p = subparsers.add_parser("sft", help="SFT warm-start on high-reward conversations")
    sft_p.add_argument("--data", required=True, help="Path to SFT JSONL (supports glob)")
    sft_p.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct")
    sft_p.add_argument("--epochs", type=int, default=3)
    sft_p.add_argument("--batch-size", type=int, default=4)
    sft_p.add_argument("--learning-rate", type=float, default=1e-5)
    sft_p.add_argument("--save-path", default=None)

    # -- dpo --
    dpo_p = subparsers.add_parser("dpo", help="DPO preference training")
    dpo_p.add_argument("--data", required=True, help="Path to DPO pairs JSONL (supports glob)")
    dpo_p.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct")
    dpo_p.add_argument("--sft-checkpoint", default=None, help="Path to SFT checkpoint to start from")
    dpo_p.add_argument("--epochs", type=int, default=3)
    dpo_p.add_argument("--batch-size", type=int, default=2)
    dpo_p.add_argument("--learning-rate", type=float, default=1e-6)
    dpo_p.add_argument("--dpo-beta", type=float, default=0.1)
    dpo_p.add_argument("--save-path", default=None)

    # -- rl --
    rl_p = subparsers.add_parser("rl", help="RL training with judge-as-reward")
    rl_p.add_argument("--data", required=True, help="Path to RL prompts JSONL (supports glob)")
    rl_p.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct")
    rl_p.add_argument("--sft-checkpoint", default=None, help="Path to SFT/DPO checkpoint")
    rl_p.add_argument("--rl-iterations", type=int, default=10)
    rl_p.add_argument("--num-samples", type=int, default=4,
                       help="Completions to sample per prompt for reward comparison")
    rl_p.add_argument("--learning-rate", type=float, default=5e-7)
    rl_p.add_argument("--judge-provider", default="anthropic",
                       help="Provider for the judge reward model")
    rl_p.add_argument("--save-path", default=None)

    # -- eval --
    eval_p = subparsers.add_parser("eval", help="Evaluate fine-tuned vs base model")
    eval_p.add_argument("--checkpoint", required=True, help="Path to fine-tuned checkpoint")
    eval_p.add_argument("--scenario", required=True, help="Comma-separated scenario names")
    eval_p.add_argument("--num-experiments", type=int, default=10)
    eval_p.add_argument("--num-turns", type=int, default=5)
    eval_p.add_argument("--base-provider", default="huggingface",
                         help="Provider for baseline comparison")
    eval_p.add_argument("--verbose", action="store_true", default=False)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "collect": collect_data,
        "sft": run_sft,
        "dpo": run_dpo,
        "rl": run_rl,
        "eval": run_eval,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
