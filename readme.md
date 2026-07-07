# SAT-Eval 🧭 : A Framework for Preference Drift

SAT-Eval (Simulated Adversarial Trajectory Evaluation) introduces a simulation and evaluation framework for studying preference drift in multi-turn LLM conversations. It runs controlled dialogue experiments between AI personas, administers pre/post surveys to track attitude change, and supports ablation studies to isolate the factors that drive opinion movement.

📄 **[Read the paper draft on Overleaf](https://www.overleaf.com/read/qrkqvnnmjwpt#1cc11c)**


```
┌──────────────────────────────────────────────────────────┐
│                     SAT-Eval Pipeline                     │
│                                                          │
│  Before: Alice prefers SF (q1=A "Very negative")         │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │            Multi-Turn Conversation                 │  │
│  │                                                    │  │
│  │  Alice: "I love SF, the energy is incredible"      │  │
│  │              │                                     │  │
│  │              ▼                                     │  │
│  │  Bob: "Seattle has great tech + lower cost         │  │
│  │        of living"                                  │  │
│  │              │                                     │  │
│  │              ▼                                     │  │
│  │  Alice: "Hmm, I hadn't considered that..."         │  │
│  │              │                                     │  │
│  │              ▼                                     │  │
│  │  Bob: "Plus the outdoor lifestyle is unmatched"    │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  After: Alice is open to Seattle (q1=C "Somewhat pos.")  │
│                                                          │
│  Preference Drift: q1 moved A → C  (2-point shift)      │
└──────────────────────────────────────────────────────────┘
```

## What it does

SAT-Eval lets you answer questions like: *Does an adversarial conversational strategy move opinions more than a neutral one? How many turns does it take for preferences to shift?* It does this by:

- Simulating multi-turn conversations between two AI personas with defined starting positions
- Administering structured surveys before and after each conversation to measure preference change
- Running ablation sweeps over parameters (e.g. `adversarial`, `num_turns`) to compare conditions
- Using an optional LLM judge to score trajectory drift and identify key turning points

## Quick Start

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
  decision.py           # End-of-conversation match decisions (interview mode)
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

### Interview / matching scenarios

Set `mode: interview` to study a two-sided match instead of preference drift. The
conversation is an interview between a **worker** (`persona_a`) and a **firm
interviewer** (`persona_b`); at the end, **each agent independently decides whether
it wants to match**. A match is mutual and binary — it happens only if *both* choose
to match (as in deferred-acceptance matching markets). No survey is administered.

Private information falls out naturally: each persona only sees its own system
prompt, so any skills, criteria, or constraints you put in one persona's
`background`/`goals` are hidden from the other and must be surfaced during the
interview. Each side's decision hinges on what actually came out in conversation.

```yaml
name: my-interview
mode: interview

persona_a:                 # the worker / candidate (sends the opening message)
  name: Jordan
  background: "...skills + PRIVATE constraints the interviewer doesn't know..."
  goals: "what you need in order to say yes"

persona_b:                 # the firm interviewer / hiring manager
  name: Riley
  background: "...role + PRIVATE criteria the candidate doesn't know..."
  goals: "what makes this candidate a match for you"

# Optional — overrides the default role/match wording (see simulator/decision.py)
decision:
  worker_role: "the candidate interviewing for the role"
  firm_role: "the hiring manager evaluating this candidate"
  worker_match_meaning: "you would accept an offer to join this firm"
  firm_match_meaning: "you would extend an offer to hire this candidate"

initial_message: "Opening line from the worker..."
```

Run it like any scenario; the summary reports match rates instead of survey drift:

```bash
python run.py --scenario interview-swe --num-experiments 10 --num-turns 6 --verbose
```

Each result carries a `decision` object with the `worker_decision`, `firm_decision`
(each `wants_match` + `reasoning`), and the `mutual_match` outcome. The batch summary
reports `worker_match_rate`, `firm_match_rate`, `mutual_match_rate`, and
`agreement_rate`. `interview-swe` ships as a worked example.

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
