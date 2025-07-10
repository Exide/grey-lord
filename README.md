# Grey Lord - MajorMUD AI

A machine learning model project using [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning) with the goal of a fully autonomous MajorMUD player. 

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python grey-lord.py --help
```

## Commands

**Data Management:**
```bash
python grey-lord.py data copy ../telnet-data new_training_data
```
```bash
python grey-lord.py data prune new_training_data --min-size 1MB --pattern *.log
```

## Project Structure

```
grey-lord/
├── README.md        # This document
├── grey-lord.py     # Main CLI interface
├── src/             # RL training
├── data/            # Training datasets
├── models/          # Trained models
├── artifacts/       # Things created along the way
```

## Design Decisions

### Reader Thread Architecture
The environment uses a dedicated reader thread for socket communication rather than synchronous I/O in the main thread. This prevents blocking the agent's decision loop while ensuring continuous data collection from the real-time, multiplayer BBS server.

**Benefits:**
- **Non-blocking actions**: Agent steps don't wait for network responses
- **Real-time state capture**: Server events and other players' actions are buffered continuously
- **Robust recovery**: Connection issues can be handled without halting training
- **Temporal fidelity**: Maintains accurate timing relationships critical for RL temporal reasoning

## Notes

### Pre-tokenization
When the data stream is being parsed within the environment we transform Telnet and ANSI sequences into string tokens. For Telnet sequences it simply uses the 3-byte sequence; `<|telnet#fffb01|>` would be `IAC WILL ECHO`. With the ANSI sequences we actually decompose complex commands into discreet string tokens; `<|ansi#reset>` and `<|ansi#fg_white|>` as an example.

### Tokenization
Currently we're using a custom tokenizer based on [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased).
- **Special tokens**: `<|client|>`, `<|server|>`, `<|delay_short|>`, etc.
- **Byte encoding**: Raw bytes mapped to `byte_0` through `byte_255`
- **Sequence handling**: Automatic padding and truncation

## Neural Network Tuning Notes

### Layer Architecture
- **Rule of thumb**: Start with 3-4 layers for most problems. Add more if the problem is very complex, reduce if you're overfitting.
- **Layer sizing**: Common pattern is to start wide and narrow down (e.g., 256 → 128 → 64 → action_size)
- **Auxiliary heads**: Use 2-layer structure for more learning capacity: `Linear(hidden, hidden//2) → ReLU → Linear(hidden//2, output)`

### Auxiliary Heads
**Think of heads as specialized experts - each becomes highly skilled in one domain:**

**Currently Implemented:**
- **Reward Predictor**: Self-supervised learning of state-reward correlations
- **Next Token Predictor**: Self-supervised learning of game language patterns and dynamics
- **Amygdala Head**: Survival expert - immediate threats, health status, escape routes (uses reward-based heuristics)

**Future Implementation:**
- **Cerebrum Head**: Strategic expert - long-term goals, resource management, planning
- **Cartographer Head**: Navigation expert - spatial relationships, pathfinding, landmarks
- **PvP Head**: Combat expert - player threats, combat tactics, team coordination
- **Economic Head**: Wealth expert - market prices, resource valuation, opportunities
- **Quest Head**: Mission expert - quest objectives, completion requirements, reward evaluation
- **Quartermaster Head**: Equipment expert - optimal gear selection, inventory management, level-appropriate upgrades
