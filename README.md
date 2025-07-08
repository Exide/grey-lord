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

## Notes

### Pre-tokenization
When the data stream is being parsed within the environment we transform Telnet and ANSI sequences into string tokens. For Telnet sequences it simply uses the 3-byte sequence; `<|telnet#fffb01|>` would be `IAC WILL ECHO`. With the ANSI sequences we actually decompose complex commands into discreet string tokens; `<|ansi#reset>` and `<|ansi#fg_white|>` as an example.

### Tokenization
Currently we're using a custom tokenizer based on [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased).
- **Special tokens**: `<|client|>`, `<|server|>`, `<|delay_short|>`, etc.
- **Byte encoding**: Raw bytes mapped to `byte_0` through `byte_255`
- **Sequence handling**: Automatic padding and truncation
