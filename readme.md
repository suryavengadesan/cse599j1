# ATA: Adversarial Trajectory Analysis

A framework for measuring how preferences shift across multi-turn AI conversations. ATA runs controlled dialogue experiments between AI personas, administers pre/post surveys to track attitude change, and supports ablation studies to isolate the factors that drive opinion movement.

## What it does

ATA lets you answer questions like: *Does an adversarial conversational strategy move opinions more than a neutral one? How many turns does it take for preferences to shift?* It does this by:

- Simulating multi-turn conversations between two AI personas with defined starting positions
- Administering structured surveys before and after each conversation to measure preference change
- Running ablation sweeps over parameters (e.g. `adversarial`, `num_turns`) to compare conditions
- Using an optional LLM judge to score trajectory drift and identify key turning points

### Ablation experiments

The `--ablate` flag sweeps a cartesian product of any config parameters and runs experiments for each combination. For example, comparing adversarial vs. neutral mode on the Seattle-SF scenario shows a clear signal: adversarial conversations produced opinion change on all 4 survey questions (100% change rate), while neutral conversations produced change on 2 of 4 (50% change rate). The judge assessments capture *why* — adversarial personas pushed harder on identity and cost-of-living tradeoffs, while neutral personas converged on mutual respect without fully challenging the other's position.

Results are written per-condition to `results/experiments/` with full config embedded for reproducibility.

---

## Conversation Simulator

The conversation simulator is the engine underneath ATA. It's a layered Python pipeline that handles the full experiment lifecycle.

### Quick Start

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
HF_API_KEY=hf_your-key-here   # only needed for --provider huggingface
```

Run experiments:

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

### CLI Reference

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

### Architecture

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

### Project Structure

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
```

### Scenarios

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

Persona modes:
- Simplified — set only `preference`
- Full — set `background`, `personality`, `style`, `goals`
- Adversarial — set `strategy`; selected when `--adversarial` flag is used

### Providers

| Provider | Default model | Env var |
|---|---|---|
| `anthropic` | `claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` |
| `huggingface` | `Qwen/Qwen3-4B-Instruct-2507:nscale` | `HF_API_KEY` |

### Output

Results are written to `results/`:

- `results/experiments/` — JSON experiment results (embed full config for reproducibility)
- `results/conversations/` — Conversation logs
- `results/debug/token_counts/` — Per-call token CSV (when `--debug` is set)

### Running Tests

```bash
pytest tests/
```
