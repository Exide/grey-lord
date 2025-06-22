# Grey Lord - MajorMUD AI

A language model trained on raw MajorMUD bytes sent/received from many play sessions. Includes an agent that can be enabled within a command-line Telnet connection.

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python grey-lord.py --help
```

## Commands

**Agent:**
```bash
python grey-lord.py agent start --config agent_config.json
```

**Model Training:**
```bash
# Basic training
python grey-lord.py model train --epochs 50

# Continue from checkpoint
python grey-lord.py model train --model-path models/your-model --epochs 25

# Custom parameters
python grey-lord.py model train --epochs 100 --batch-size 16 --learning-rate 1e-4
```

**Model Management:**
```bash
python grey-lord.py model list                      # List models
python grey-lord.py model leaderboard               # Performance ranking
python grey-lord.py model compare model1 model2     # Compare models
python grey-lord.py model cleanup --keep 5          # Clean old models
python grey-lord.py model links                     # Create convenience links
python grey-lord.py model export --model-dir models/your-model --output exported/
```

**Model Analysis & Debugging:**
```bash
python grey-lord.py model analyze --model-dir models/your-model
python grey-lord.py model debug vocab
python grey-lord.py model debug model --model-dir models/your-model
python grey-lord.py model debug sequences --data-dir training_data
```

**Model Optimization:**
```bash
python grey-lord.py model optimize --batch-size 4 --memory-gb 8
```

**Data Management:**
```bash
python grey-lord.py data prepare --source ../telnet-data --target training_data
python grey-lord.py data analyze --data-dir training_data
python grey-lord.py data process --session-dir data/agent_sessions
```

**Configuration:**
```bash
python grey-lord.py config show                     # View current config
python grey-lord.py config validate                 # Validate config files
```

## Training Process

```
INITIALIZATION:
  Load telnet binary data files
  Tokenize using custom vocabulary (262 tokens: bytes + special tokens)
  Split data → 80% training, 20% validation
  Initialize GPT-2 model (4 layers, 128 dim, 4 heads)
  Setup AdamW optimizer with weight decay
  Setup learning rate scheduler (linear warmup + decay)

TRAINING LOOP:
  FOR each epoch (1 to max_epochs):
    
    TRAINING PHASE:
      Set model to training mode
      FOR each batch in training_data:
        Forward pass: predict next token given context
        Calculate loss: cross-entropy between prediction and actual next token
        Backward pass: compute gradients
        Clip gradients (prevent exploding gradients)
        Update model parameters
        Update learning rate schedule
      
    VALIDATION PHASE:
      Set model to evaluation mode
      FOR each batch in validation_data:
        Forward pass (no gradient computation)
        Calculate validation loss
      
    EARLY STOPPING CHECK:
      IF validation_loss improved:
        Save as best model
        Reset patience counter
      ELSE:
        Increment patience counter
        IF patience_counter >= patience_limit:
          Stop training (prevent overfitting)
          BREAK

FINALIZATION:
  Save final model (best checkpoint)
  Generate training curves and statistics
  Export model in HuggingFace format
  Copy vocabulary files for reproducibility
```

**Key concepts:**
- **Causal modeling**: Predicts next token given previous context (left-to-right)
- **Teacher forcing**: During training, uses actual next token (not predicted)
- **Early stopping**: Monitors validation loss to prevent overfitting
- **Gradient clipping**: Prevents training instability from large gradients

## Architecture

- **Model**: GPT-2 (4 layers, 128 embedding dim, 4 attention heads)
- **Vocabulary**: 262 tokens (binary telnet data + special tokens)
- **Context**: Up to 8,192 tokens (configurable)
- **Training**: Causal language modeling with early stopping
- **Hardware**: Optimized for NVIDIA GPUs (4GB+ VRAM recommended)

## Project Structure

```
grey-lord/
├── grey-lord.py                   # Main CLI interface
├── training/                      # Model training code
├── agent/                         # AI agent telnet client
├── analysis/                      # Analysis & debugging tools
├── data/                          # Vocabulary and training data
├── models/                        # Trained models
├── agent_config.json              # AI agent configuration
└── model_config.json              # Training configuration
```

## Training Output

Each training run creates a timestamped directory with:
- **Model files**: `model.safetensors`, `config.json`
- **Training data**: `training_history.json`, `training_summary.json`, `training_curves.png`
- **Vocabulary**: `vocab_to_int.json`, `int_to_vocab.json`

**Load a trained model:**
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("models/your-model-name")
```

## Agent Data Collection & Retraining

**The agent automatically collects data while playing that can be used for improving the model:**

### What's Collected
- **Raw telnet data**: Server responses and client commands
- **AI decisions**: Context → Action pairs with timestamps  
- **Experience replay**: State-action-reward sequences
- **Session metadata**: Performance metrics and game outcomes

### Data Processing
```bash
# Process agent session data for training
python grey-lord.py data process --session-dir data/agent_sessions --output-dir data/processed_agent_data
```

### Training with Agent Data
```bash
# Continue training with agent experience
python grey-lord.py model train --data-dir data/processed_agent_data/continued --epochs 10

# Train specialized model from successful behaviors
python grey-lord.py model train --data-dir data/processed_agent_data/behavioral --epochs 25
```

**Use cases:**
- **Continued training**: Improve the model with real gameplay experience
- **Behavioral cloning**: Train on only successful agent actions
- **Domain adaptation**: Adapt to specific MUD server behaviors
- **Reinforcement learning**: Train with reward signals from game outcomes

## Data Format

Processes binary telnet proxy files with custom tokenization:
- **Special tokens**: `<|client|>`, `<|server|>`, `<|delay_short|>`, etc.
- **Byte encoding**: Raw bytes mapped to `byte_0` through `byte_255`
- **Sequence handling**: Automatic padding and truncation
