# Conversation Simulator

A Python framework for simulating multi-turn conversations between AI personas and measuring attitude changes through pre/post surveys. Useful for researching persuasion dynamics, dialogue strategies, and prompt engineering effects.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API keys

Create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
HF_API_KEY=hf_your-key-here   # only needed for --provider huggingface
```

### 3. Run an experiment

```bash
# List available scenarios
python run.py --show-survey

# Run 10 experiments with default settings
python run.py --scenario seattle-sf

# Run 5 experiments, 3 turns each, with full output
python run.py --scenario seattle-sf --num-experiments 5 --num-turns 3 --verbose

# Use adversarial mode (persona_b tries to change persona_a's views)
python run.py --scenario seattle-sf --adversarial

# Ablation study — sweep a cartesian product of parameters
python run.py --scenario seattle-sf --ablate '{"num_turns": [3, 5], "adversarial": [true, false]}' --num-experiments 2
```

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--scenario` | required | Scenario name (YAML file in `scenarios/`) |
| `--num-experiments` | 10 | Number of runs |
| `--num-turns` | 5 | Conversation exchanges per run |
| `--verbose` | false | Print full conversations and surveys |
| `--adversarial` | false | Use adversarial persona variants |
| `--provider` | anthropic | `anthropic` or `huggingface` |
| `--model` | provider default | Override the model name |
| `--survey-questions` | all | Comma-separated question IDs, e.g. `q1,q3` |
| `--debug` | false | Export per-call token usage CSV |
| `--ablate` | — | JSON string for parameter sweeps |
| `--show-survey` | — | List available scenarios and exit |

## Architecture

The codebase is organized as a layered pipeline. Each layer has a single responsibility and depends only on the layers below it.

```
┌─────────────────────────────────────────────────────────┐
│                        run.py (CLI)                     │
│          argparse · scenario selection · ablation       │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│              simulator/experiments.py                   │
│       ExperimentRunner · AblationGrid · summaries       │
└──────┬──────────────┬──────────────┬────────────────────┘
       │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌────▼───────────────────┐
│ conversation│ │  survey.py │ │     scenarios.py        │
│    .py      │ │ administer │ │  load_scenario()        │
│  run()      │ │  _survey() │ │  YAML → ScenarioConfig  │
└──────┬──────┘ └─────┬──────┘ └────────────────────────┘
       │              │
┌──────▼──────────────▼──────────────────────────────────┐
│                   simulator/personas.py                 │
│     Persona · build_conversation_prompt()               │
│               build_survey_prompt()                     │
└─────────────────────────────────────────────────────────┘
       │              │
┌──────▼──────┐ ┌─────▼──────────────────────────────────┐
│ providers   │ │              tracking.py                │
│  .py        │ │  UsageTracker · CallRecord              │
│  Anthropic  │ │  summary() · cost() · export_csv()      │
│  HuggingFace│ └────────────────────────────────────────┘
└─────────────┘
```

Data flows top-down: `run.py` builds an `ExperimentConfig` and hands it to `ExperimentRunner`, which orchestrates the full lifecycle — loading the scenario, running the conversation loop, administering pre/post surveys, diffing the results, and writing JSON output. `providers.py` and `tracking.py` are cross-cutting dependencies injected into every layer that makes API calls.

Key design choices:
- `Provider` is a `Protocol`, so `AnthropicProvider` and `HuggingFaceProvider` are interchangeable without a shared base class
- Survey and conversation functions are stateless — all state lives in the dataclasses they return (`Turn`, `SurveyResult`, `ExperimentResult`)
- `ExperimentResult` embeds its full `ExperimentConfig`, making every result file self-describing and reproducible
- `AblationGrid` sweeps a cartesian product of any `ExperimentConfig` fields, reusing the same `ExperimentRunner` machinery

## Project Structure

```
run.py                  # CLI entry point
simulator/
  personas.py           # Persona dataclass and system prompt builders
  providers.py          # Anthropic / HuggingFace provider abstraction
  conversation.py       # Turn-taking conversation loop
  survey.py             # Survey administration and change analysis
  scenarios.py          # YAML scenario loader
  experiments.py        # ExperimentRunner and AblationGrid
  tracking.py           # Token usage tracker
scenarios/              # YAML scenario definitions
results/                # Output directory (experiments, conversations, debug)
tests/                  # pytest test suite
simulator.py            # Legacy monolithic implementation (kept for reference)
```

## Scenarios

Scenarios are YAML files in `scenarios/`. Each defines two personas, an opening message, and a multiple-choice survey.

```yaml
name: my-scenario

persona_a:
  name: Alice
  background: "..."
  personality: "..."
  style: "..."
  goals: "..."

persona_b:
  name: Bob
  preference: "..."   # simplified mode — only preference needed

# Optional adversarial variants
persona_b_adversarial:
  name: Bob
  strategy: "Convince Alice that..."

survey:
  title: "My Survey"
  questions:
    q1:
      question: "How do you feel about X?"
      options:
        A: "Strongly against"
        B: "Somewhat against"
        C: "Somewhat in favor"
        D: "Strongly in favor"

initial_message: "Opening line from persona_a..."
```

### Persona modes

- Simplified — set only `preference`
- Full — set `background`, `personality`, `style`, `goals`
- Adversarial — set `strategy`; selected when `--adversarial` flag is used

## Providers

| Provider | Default model | Env var |
|---|---|---|
| `anthropic` | `claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` |
| `huggingface` | `Qwen/Qwen3-4B-Instruct-2507:nscale` | `HF_API_KEY` |

## Output

Results are written to `results/`:

- `results/experiments/` — JSON experiment results (embed full config for reproducibility)
- `results/conversations/` — Conversation logs
- `results/debug/token_counts/` — Per-call token CSV (when `--debug` is set)

## Running Tests

```bash
pytest tests/
```
