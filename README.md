# Grey Lord - MajorMUD Language Model

A GPT-2 language model trained on binary telnet proxy data to understand and predict MajorMUD game interactions.

## Quick Start

```bash
# Create virtual environment
python -m venv .venv

# Activate the environment
(Windows) `.venv\Scripts\activate`
(Linux/Mac) `source .venv/bin/activate`

# Install dependencies
pip install -r requirements.txt

# Train a model
python grey-lord.py train --epochs 50 --batch-size 32

# Analyze results
python grey-lord.py model list
python grey-lord.py analyze --training-dir models/your-model-name
```

## Project Structure

```
grey-lord/
├── grey-lord.py                   # Main CLI interface
├── requirements.txt               # Python dependencies
├── model_config.json              # Configuration
├── src/                           # Core code
│   ├── train.py                   # Training logic
│   ├── model_manager.py           # Model management
│   └── config_utils.py            # Configuration utilities
├── tools/                         # Utility scripts
├── data/                          # Vocabulary and training data
│   ├── vocab_to_int.json          # Vocabulary mappings
│   ├── int_to_vocab.json          # Reverse vocabulary  
│   └── training_data_v1/          # Dataset version 1
└── models/                        # Trained models
```

## Training

**Basic training:**
```bash
python grey-lord.py train --epochs 50
```

**Continue from checkpoint:**
```bash
python grey-lord.py train --model-path models/your-model --epochs 25
```

**Custom parameters:**
```bash
python grey-lord.py train --epochs 100 --batch-size 16 --learning-rate 1e-4 --max-seq-len 2048
```

## Analysis & Management

**List models:**
```bash
python grey-lord.py model list
python grey-lord.py model leaderboard
```

**Analyze training:**
```bash
python grey-lord.py analyze --training-dir models/your-model
```

**Compare models:**
```bash
python grey-lord.py model compare model1 model2
```

**Clean up old models:**
```bash
python grey-lord.py model cleanup --keep 5
```

## Optimization

**Find optimal batch size:**
```bash
python grey-lord.py optimize batch-size
```

**Calculate memory usage:**
```bash
python grey-lord.py optimize memory
```

## Debugging

**Check vocabulary:**
```bash
python grey-lord.py debug vocab
```

**Validate model:**
```bash
python grey-lord.py debug model
```

**View configuration:**
```bash
python grey-lord.py config show
```

## Technical Details

- **Architecture**: GPT-2 (4 layers, 128 embedding dim, 4 attention heads)
- **Vocabulary**: 262 tokens (binary telnet data + special tokens)
- **Context Length**: Up to 8,192 tokens (configurable)
- **Training**: Causal language modeling with early stopping
- **Hardware**: Optimized for NVIDIA GPUs (4GB+ VRAM recommended)

## Training Output

Each training run creates a self-contained directory with:
- **Model files**: `model.safetensors`, `config.json` (for loading with transformers)
- **Training data**: `training_history.json`, `training_summary.json`, `training_curves.png`
- **Vocabulary**: `vocab_to_int.json`, `int_to_vocab.json` (copied for reproducibility)

**Model naming**: `{dataset_version}_{type}_{params}_{timestamp}`
- `v1_new_1733072400` (default parameters)
- `v1_cont_batch-16_learning-rate-1e4_1733072400` (custom batch size and learning rate)
- `v2_new_seq-2k_1733072400` (dataset v2, custom sequence length)

Timestamp is epoch seconds for precise ordering and uniqueness.

**Load a trained model:**
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("models/your-model-name")
```

## Data Format

Grey Lord processes binary telnet proxy files with custom tokenization:
- **Special tokens**: `<|client|>`, `<|server|>`, `<|delay_short|>`, etc.
- **Byte encoding**: Raw bytes mapped to `byte_0` through `byte_255`
- **Sequence handling**: Automatic padding and truncation
