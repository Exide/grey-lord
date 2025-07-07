# Grey Lord - MajorMUD AI

A language model trained on raw MajorMUD bytes sent/received from many play sessions.

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

**Model Training:**
```bash
# Basic training with default parameters from config.json
python grey-lord.py train --epochs 50

# Continue training from a checkpoint
python grey-lord.py train --model-path models/your-model --epochs 25

# Override configuration parameters
python grey-lord.py train --epochs 100 --batch-size 16 --learning-rate 1e-4
```

**Analysis & Debugging:**
```bash
# Analyze training results for overfitting and performance
python grey-lord.py analyze training --model-dir models/your-model

# Get statistics on a data directory
python grey-lord.py analyze data --data-dir training_data

# Debug the tokenizer and vocabulary
python grey-lord.py debug vocab --vocab-path data

# Debug a trained model's structure and configuration
python grey-lord.py debug model --model-dir models/your-model
```

**Hardware Optimization:**
```bash
# Find optimal batch size and context window for your GPU
python grey-lord.py optimize --memory-gb 8
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
├── analysis/                      # Analysis & debugging tools
├── data/                          # Vocabulary and training data
├── models/                        # Trained models
└── config.json                    # Training configuration
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

## Data Format

Processes binary telnet proxy files with custom tokenization:
- **Special tokens**: `<|client|>`, `<|server|>`, `<|delay_short|>`, etc.
- **Byte encoding**: Raw bytes mapped to `byte_0` through `byte_255`
- **Sequence handling**: Automatic padding and truncation





```
grey-lord data copy ~/Desktop/some_dataset NewData
grey-lord data prune NewData --size 1MB --pattern *.log
```
