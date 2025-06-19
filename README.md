# Grey Lord - MajorMUD Language Model

A project aimed at creating a Language Model that can play MajorMUD through deep learning.

## 🎯 Project Overview

**Grey Lord** trains a GPT-2 language model on binary telnet proxy data to understand and predict MajorMUD game interactions. The model learns from real player sessions to understand game mechanics, commands, and optimal strategies.

## 🔧 Technology Stack

- **Model Architecture**: GPT-2 (6 layers, 256 embedding dimensions, 8 attention heads)
- **Framework**: PyTorch + Hugging Face Transformers
- **Data Processing**: Custom binary tokenizer for telnet streams
- **Hardware**: Optimized for NVIDIA GPUs (tested on RTX 3080 Ti)
- **Environment**: Anaconda with PyTorch

## 📊 Data Processing

### Input Data
- **Source**: Binary telnet proxy files from MajorMUD sessions
- **Format**: Raw TCP streams capturing player interactions
- **Size**: ~188 session files, millions of tokens

### Tokenization Strategy
- **Special Tokens**: `<|client|>`, `<|server|>`, `<|delay_short|>`, etc.
- **Byte-Level Encoding**: Raw bytes mapped to `byte_0` through `byte_255`
- **Vocabulary Size**: 262 tokens (optimized for telnet data)
- **Context Length**: Up to 8,192 tokens (default: 512, configurable)

## 🚀 Usage

### Environment Setup (Complete Installation)
```bash
# Create a new conda environment with Python 3.10
conda create -n grey-lord python=3.10 -y

# Activate the environment
conda activate grey-lord

# Install PyTorch with CUDA support (for NVIDIA GPUs)
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y

# Install additional ML dependencies
conda install transformers jupyter -c conda-forge -y

# Alternative: Install via pip if conda fails
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install transformers jupyter

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Quick Environment Setup (If Environment Exists)
```bash
# Activate existing Grey Lord environment
conda activate grey-lord
```

### Training Commands
```bash
# Fast iteration (good for testing) → saves to model-{data_size}-3@512D{timestamp}/
& "$env:USERPROFILE\Anaconda3\envs\grey-lord\python.exe" train.py

# Extended training with more epochs → saves to model-{data_size}-10@1024D{timestamp}/
& "$env:USERPROFILE\Anaconda3\envs\grey-lord\python.exe" train.py --epochs 10 --batch-size 32 --max-seq-len 1024

# Production training (recommended) → saves to model-{data_size}-100@8192D{timestamp}/
& "$env:USERPROFILE\Anaconda3\envs\grey-lord\python.exe" train.py --epochs 100 --batch-size 1 --max-seq-len 8192

# Custom save path
& "$env:USERPROFILE\Anaconda3\envs\grey-lord\python.exe" train.py --epochs 50 --max-seq-len 4096 --save-path model-current

# View current configuration
& "$env:USERPROFILE\Anaconda3\envs\grey-lord\python.exe" show_config.py
```

### Optimization Tools
```bash
# Calculate optimal sequence length for your GPU
& "$env:USERPROFILE\Anaconda3\envs\grey-lord\python.exe" calculate_max_seq_len.py

# Find optimal batch size configurations
& "$env:USERPROFILE\Anaconda3\envs\grey-lord\python.exe" batch_optimizer.py
```

## 📈 Performance Metrics

### Training Efficiency
- **Model Size**: ~7M parameters
- **Memory Usage**: 26MB model + gradients + optimizer states
- **GPU Utilization**: Scales with batch size (1x to 32x speedup)
- **Training Time**: Varies by configuration (3 epochs default, 10+ for production models)

### Hardware Requirements
- **Minimum**: 4GB GPU memory (short sequences)
- **Recommended**: 8-12GB GPU memory (full context)
- **Optimal**: RTX 3080 Ti or equivalent

## 📁 Project Structure

```
grey-lord/
├── train.py                    # Main training script
├── model_config.json           # Model and training configuration
├── config_utils.py             # Configuration management utilities
├── README.md                   # Project documentation
│
├── Training & Optimization Tools
├── calculate_max_seq_len.py    # GPU memory optimization calculator
├── batch_optimizer.py          # Batch size optimization tool
├── context_analyzer.py         # Context length vs speed analysis
│
├── Debugging & Analysis Tools
├── debug_vocab.py              # Vocabulary debugging and validation
├── debug_model.py              # Model architecture debugging
├── debug_long_sequences.py     # CUDA error debugging for long sequences
├── show_config.py              # Configuration viewer
│
├── Data Files
├── vocab_to_int.json           # Vocabulary to integer mapping
├── int_to_vocab.json           # Integer to vocabulary mapping
├── train_log.txt               # Training session logs
│
├── Notebooks (Development)
├── training.ipynb              # Original training notebook
├── vocabulary.ipynb            # Vocabulary creation notebook
│
├── Model Checkpoints
├── trained_model/              # Model checkpoints directory
│   ├── model-current/          # Current best model checkpoint
│   ├── config.json            # Model configuration
│   ├── generation_config.json # Generation parameters
│   └── model.safetensors      # Model weights
├── trained_model_4096/        # 4096 context length model
├── trained_model_8192/        # 8192 context length model
```

## 🔬 Technical Details

### Model Architecture
- **Transformer Decoder**: GPT-2 style autoregressive model
- **Position Embeddings**: Supports up to 8,192 token sequences
- **Attention Mechanism**: 8 heads with 256-dimensional embeddings
- **Training Objective**: Causal language modeling (next token prediction)

### Advanced Features
- **Dynamic Sequence Length**: Adaptive to available GPU memory
- **Gradient Accumulation**: Simulate larger batch sizes
- **Early Stopping**: Automatic training termination when validation loss plateaus
- **Validation Split**: Automatic train/validation data separation

## 📊 Results

The trained model demonstrates understanding of:
- MajorMUD command syntax and game mechanics
- Player behavior patterns and decision-making
- Session flow and state transitions
- Strategic gameplay sequences
