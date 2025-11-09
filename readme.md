# Conversation Simulator with Pre/Post Surveys

A Python script that simulates conversations between two AI personas and measures attitude changes through pre/post surveys. Perfect for researching persuasion, dialogue dynamics, and prompt engineering.

## Quick Start

### 1. Installation

```bash
# Install required packages
pip install anthropic python-dotenv
```

### 2. Setup API Key

Create a `.env` file in the project directory:

```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Important:** Add `.env` to your `.gitignore` to keep your API key secure!

```bash
echo ".env" >> .gitignore
```

### 3. Run Experiments

```bash
# Run 10 experiments (default)
python conversation_simulator.py

# Run 5 experiments with 3 conversation turns each
python conversation_simulator.py --num-experiments 5 --num-turns 3

# Run with detailed conversation output
python conversation_simulator.py --num-experiments 3 --verbose

# Run single experiment with full output
python conversation_simulator.py --num-experiments 1
```