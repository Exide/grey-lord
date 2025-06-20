"""train.py
A clear, step-by-step replication of the original training notebook in plain Python.
Each section is small, self-contained, and heavily commented so that you can
reason about what every single line does.

ENVIRONMENT REQUIREMENTS:
This script requires an Anaconda environment with PyTorch installed.
Run with: & "$env:USERPROFILE\\Anaconda3\\envs\\grey-lord\\python.exe" train.py

How to use (quick-start):
  1. Place this file next to `vocab_to_int.json` and `int_to_vocab.json`.
  2. Adjust the DATA_DIR and FILE_GLOB below to point at your dataset files.
  3. Run with the PyTorch environment activated.

All external dependencies are from the PyPI packages:
  - torch (from conda/pip in pytorch environment)
  - transformers (from conda/pip in pytorch environment)

Tip: Open the file in an IDE and follow along; every block is documented.
"""

from __future__ import annotations  # makes type hints forward-compatible

import argparse
from datetime import datetime
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Iterable, Tuple, Union, Any

# PyTorch and ML libraries (installed in Anaconda pytorch environment)
import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore
from transformers import GPT2Config, AutoModelForCausalLM  # type: ignore

# Local configuration utilities
from config_utils import get_model_config, get_training_config, get_data_config, get_vocab_config  # type: ignore

###############################################################################
# 1. Vocabulary helpers
###############################################################################

# Vocabulary file paths will be loaded from config


def load_json(path: Path) -> dict:
    """Load a JSON file and return the parsed Python object."""
    if not path.exists():
        raise FileNotFoundError(f"Required vocabulary file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_vocabulary() -> Tuple[Dict[str, int], Dict[str, str], int]:
    """Load vocabulary files and return vocab_to_int, int_to_vocab, and pad_token_id."""
    try:
        # Load paths from config
        vocab_config = get_vocab_config()
        vocab_file = Path(vocab_config.get("vocab_to_int_file", "vocab_to_int.json"))
        int_vocab_file = Path(vocab_config.get("int_to_vocab_file", "int_to_vocab.json"))
        pad_token = vocab_config.get("pad_token", "<|pad|>")
        
        vocab_to_int: Dict[str, int] = load_json(vocab_file)
        int_to_vocab: Dict[str, str] = load_json(int_vocab_file)
        print(f"[info] Loaded vocabulary with {len(vocab_to_int)} items.")
        
        # Validate PAD_TOKEN exists
        if pad_token not in vocab_to_int:
            raise ValueError(f"PAD_TOKEN '{pad_token}' not found in vocabulary.")
        
        pad_token_id = vocab_to_int[pad_token]
        
        # Validate vocabulary integrity
        max_token_id = max(vocab_to_int.values())
        min_token_id = min(vocab_to_int.values())
        print(f"[info] Token ID range: {min_token_id} to {max_token_id}")
        
        if min_token_id < 0:
            raise ValueError(f"Negative token ID found: {min_token_id}")
        
        # Check for gaps in token IDs
        expected_ids = set(range(max_token_id + 1))
        actual_ids = set(vocab_to_int.values())
        missing_ids = expected_ids - actual_ids
        if missing_ids:
            print(f"[warning] Missing token IDs: {sorted(list(missing_ids))[:10]}{'...' if len(missing_ids) > 10 else ''}")
        
        return vocab_to_int, int_to_vocab, pad_token_id
        
    except FileNotFoundError as e:
        print(f"[error] {e}")
        print("[error] Please ensure vocab_to_int.json and int_to_vocab.json are in the current directory.")
        sys.exit(1)
    except ValueError as e:
        print(f"[error] {e}")
        sys.exit(1)

###############################################################################
# 2. Tokenisation utilities
###############################################################################

def _bin_delay_token(delay_token: str) -> str:
    """Map a raw delay token (e.g. '<|delay#0.85|>') into a coarse bucket.

    The binning rules are intentionally *very* simple – tweak as needed.
    """
    # Extract the numeric part after '#', strip the trailing '|>'
    delay_value = float(delay_token.split("#")[1].strip("|>"))

    if delay_value <= 1.0:
        return "<|delay_short|>"
    if delay_value <= 5.0:
        return "<|delay_medium|>"
    return "<|delay_long|>"


# Pre-compile the main regex pattern once – doing it in the hot loop is wasteful.
_SPECIAL_TOKEN_RE = re.compile(rb"(<\|[^|>]+?\|>)|(.)")


def custom_tokenizer(byte_stream: bytes, vocab: Dict[str, int], max_token_id: int = None) -> List[int]:
    """Convert a byte stream into a list of integer token IDs.

    A token is either:
      • A *special* token, delimited by the pattern <| ... |>
      • A single raw byte (0-255) mapped through the vocabulary

    Any special token that encodes a delay ("<|delay#X.YZ|>") is first mapped to
    a coarser category using `_bin_delay_token` so that the vocabulary stays
    small.
    """
    token_ids: List[int] = []

    for match in _SPECIAL_TOKEN_RE.finditer(byte_stream):
        special_token_bytes, single_byte = match.groups()

        if special_token_bytes:  # We matched something like b'<|client|>'
            token_str = special_token_bytes.decode("utf-8")

            if token_str.startswith("<|delay#"):
                token_str = _bin_delay_token(token_str)

            # Handle unknown special tokens gracefully
            if token_str in vocab:
                token_id = vocab[token_str]
                if max_token_id is not None and token_id >= max_token_id:
                    print(f"[warning] Token ID {token_id} for '{token_str}' exceeds max_token_id {max_token_id}, skipping...")
                    continue
                token_ids.append(token_id)
            else:
                print(f"[warning] Unknown special token: {token_str}, skipping...")
                continue
        elif single_byte:
            # Map byte value through vocabulary: byte_X -> vocab_id
            byte_value = int.from_bytes(single_byte, "big")
            byte_token = f"byte_{byte_value}"
            if byte_token in vocab:
                token_id = vocab[byte_token]
                if max_token_id is not None and token_id >= max_token_id:
                    print(f"[warning] Token ID {token_id} for '{byte_token}' exceeds max_token_id {max_token_id}, skipping...")
                    continue
                token_ids.append(token_id)
            else:
                print(f"[warning] Unknown byte token: {byte_token}, skipping...")
                continue

    return token_ids

###############################################################################
# 3. PyTorch Dataset + DataLoader helpers
###############################################################################

class ByteStreamDataset(Dataset):
    """Dataset that lazily reads binary files and tokenises them on-the-fly."""

    def __init__(self, paths: Iterable[Path], vocab: Dict[str, int], pad_token_id: int, max_token_id: int = None, label: str = ""):
        self.paths = list(paths)
        self.vocab = vocab
        self.pad_token_id = pad_token_id
        self.max_token_id = max_token_id
        dataset_type = f" ({label})" if label else ""
        print(f"[info] Dataset{dataset_type} initialized with {len(self.paths)} files.")

    # ----------------------------------------------------------------------
    # PyTorch mandatory overrides
    # ----------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.paths)

    def __getitem__(self, idx: int):  # type: ignore[override]
        file_path = self.paths[idx]
        try:
            with file_path.open("rb") as f:
                raw = f.read()
        except Exception as e:
            print(f"[error] Failed to read file {file_path}: {e}")
            return torch.tensor([], dtype=torch.long)

        token_ids = custom_tokenizer(raw, self.vocab, self.max_token_id)
        if not token_ids:
            print(f"[warning] No tokens found in file {file_path}")
            return torch.tensor([self.pad_token_id], dtype=torch.long)
        
        return torch.tensor(token_ids, dtype=torch.long)


def collate_fn(batch: List[torch.Tensor], max_seq_len: int, pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences and build an attention mask."""
    # Filter out empty sequences and truncate long sequences
    batch = [seq[:max_seq_len] for seq in batch if seq.numel() > 0]
    if not batch:
        # Return minimal batch if all sequences were empty
        return {
            "input_ids": torch.tensor([[pad_token_id]], dtype=torch.long),
            "attention_mask": torch.tensor([[0]], dtype=torch.long)
        }
    
    max_len = min(max(seq.size(0) for seq in batch), max_seq_len)

    # Pre-allocate and *fill* with pad-value – cheaper than later .fill_()
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)

    for i, seq in enumerate(batch):
        seq_len = min(seq.size(0), max_len)
        input_ids[i, :seq_len] = seq[:seq_len]
        attention_mask[i, :seq_len] = 1  # mark real tokens with 1s

    return {"input_ids": input_ids, "attention_mask": attention_mask}

###############################################################################
# 4. Build the model – a tiny GPT-2 variant from 🤗 Transformers
###############################################################################

def build_model(vocab_size: int, model_path: Union[str, None] = None) -> AutoModelForCausalLM:
    """Create a GPT-2-style model, either fresh or loaded from existing checkpoint."""
    if model_path and Path(model_path).exists():
        print(f"[info] Loading existing model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Check vocabulary size compatibility
        model_vocab_size = model.config.vocab_size
        print(f"[info] Loaded model vocabulary size: {model_vocab_size}")
        print(f"[info] Current vocabulary size: {vocab_size}")
        
        if vocab_size > model_vocab_size:
            print(f"[warning] Current vocabulary ({vocab_size}) is larger than model vocabulary ({model_vocab_size})")
            print("[warning] This will cause indexing errors. Please retrain from scratch or use compatible vocabulary.")
            raise ValueError(f"Vocabulary size mismatch: current={vocab_size}, model={model_vocab_size}")
        elif vocab_size < model_vocab_size:
            print(f"[warning] Current vocabulary ({vocab_size}) is smaller than model vocabulary ({model_vocab_size})")
            print("[warning] Some embeddings will be unused, but training can continue.")
        
        return model
    else:
        print("[info] Creating fresh model with random weights")
        model_config = get_model_config()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=model_config.get("n_positions", 1024),
            n_embd=model_config.get("n_embd", 768),
            n_layer=model_config.get("n_layer", 12),
            n_head=model_config.get("n_head", 12),
            dropout=model_config.get("dropout", 0.1),
            attention_dropout=model_config.get("attention_dropout", 0.1),
            resid_dropout=model_config.get("resid_dropout", 0.1),
        )
        # Explicitly set the loss type to avoid warnings and be clear about our intention
        config.loss_type = "ForCausalLMLoss"
        return AutoModelForCausalLM.from_config(config)

###############################################################################
# 5. Training loop (very small; suitable for debugging)
###############################################################################

def validate(model: AutoModelForCausalLM, val_loader: DataLoader, device: torch.device) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Skip batches with no real tokens
            if attention_mask.sum() == 0:
                continue
                
            outputs = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=input_ids)
            loss = outputs.loss
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else float('inf')


def train(
        data_dir: str,
        file_glob: str,
        num_epochs: int = 3, 
        batch_size: int = 1, 
        learning_rate: float = 5e-5, 
        save_path: Union[str, None] = None, 
        max_seq_len: int = 512, 
        model_path: Union[str, None] = None, 
        val_split: float = 0.2, 
        patience: int = 5,
        early_stopping_threshold: float = 0.01
    ) -> None:

    # Load vocabulary first
    vocab_to_int, int_to_vocab, pad_token_id = load_vocabulary()
    vocab_size = max(vocab_to_int.values()) + 1  # ensure the highest ID fits in the embedding
    
    print(f"[DEBUG] Vocabulary loaded: {len(vocab_to_int)} items")
    print(f"[DEBUG] Max token ID: {max(vocab_to_int.values())}")
    print(f"[DEBUG] Calculated vocab_size: {vocab_size}")
    
    # Set default save path if not provided (will be set after data_size calculation)
    save_path_template = save_path
    
    # ------------------------------------------------------------------
    # DataLoader setup with train/validation split
    # ------------------------------------------------------------------
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    file_paths = sorted(data_path.glob(file_glob))
    if not file_paths:
        raise FileNotFoundError(
            f"No files found in {data_dir} matching pattern '{file_glob}' – "
            "double-check data_dir and file_glob arguments."
        )

    print(f"[info] Found {len(file_paths)} files matching pattern '{file_glob}'")
    
    # Split files into train and validation sets
    num_val_files = max(1, int(len(file_paths) * val_split))
    val_files = file_paths[:num_val_files]
    train_files = file_paths[num_val_files:]
    
    print(f"[info] Split: {len(train_files)} training files, {len(val_files)} validation files")
    
    # Calculate data size (total bytes of training files)
    data_size = 0
    for file_path in train_files:
        try:
            data_size += file_path.stat().st_size
        except OSError:
            print(f"[warning] Could not get size of file: {file_path}")
    
    # Convert to human-readable format
    if data_size >= 1024**3:  # GB
        data_size_str = f"{data_size / (1024**3):.1f}GB"
    elif data_size >= 1024**2:  # MB
        data_size_str = f"{data_size / (1024**2):.1f}MB"
    elif data_size >= 1024:  # KB
        data_size_str = f"{data_size / 1024:.1f}KB"
    else:
        data_size_str = f"{data_size:.1f}B"
    
    print(f"[info] Training data size: {data_size_str} ({data_size:,} bytes)")
    
    # Set default save path if not provided (now that we have data_size)
    if save_path_template is None:
        # Generate very terse model name
        timestamp = str(int(time.time()))
        
        # Extract dataset version from data directory name
        data_dir_path = Path(data_dir)
        data_version = "v1"  # default
        if "_v" in data_dir_path.name:
            data_version = data_dir_path.name.split("_v")[-1]
        
        # Build minimal name components
        name_parts = [data_version]  # Start with dataset version
        
        # Add training type
        if model_path:
            name_parts.append("cont")
        else:
            name_parts.append("new")
        
        # Add key parameters (only non-defaults)
        if batch_size != 32:
            name_parts.append(f"batch-{batch_size}")
        
        if learning_rate != 3e-4:
            lr_str = f"{learning_rate:.0e}".replace("e-0", "e").replace("e-", "e")
            name_parts.append(f"learning-rate-{lr_str}")
        
        if max_seq_len != 1024:
            if max_seq_len >= 1024:
                seq_str = f"{max_seq_len//1024}k" if max_seq_len % 1024 == 0 else f"{max_seq_len}"
            else:
                seq_str = str(max_seq_len)
            name_parts.append(f"seq-{seq_str}")
        
        # Add timestamp
        name_parts.append(timestamp)
        
        save_path = "_".join(name_parts)
    else:
        save_path = save_path_template
    
    # ------------------------------------------------------------------
    # Model setup (needed early to get model vocabulary size)
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(vocab_size, model_path)
    model_vocab_size = model.config.vocab_size
    model = model.to(device)

    # Create separate datasets and dataloaders with vocabulary size validation
    train_dataset = ByteStreamDataset(train_files, vocab_to_int, pad_token_id, model_vocab_size, "training")
    val_dataset = ByteStreamDataset(val_files, vocab_to_int, pad_token_id, model_vocab_size, "validation")
    
    # Create a collate function with max sequence length and pad token
    def limited_collate_fn(batch):
        return collate_fn(batch, max_seq_len, pad_token_id)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=limited_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=limited_collate_fn)

    # ------------------------------------------------------------------
    # Optimiser setup with weight decay and learning rate scheduling
    # ------------------------------------------------------------------
    training_config = get_training_config()
    weight_decay = training_config.get("weight_decay", 0.01)
    warmup_steps = training_config.get("warmup_steps", 100)
    gradient_clip = training_config.get("gradient_clip", 1.0)
    
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler with warmup
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    total_steps = len(train_loader) * num_epochs
    warmup_scheduler = LinearLR(optimiser, start_factor=0.1, total_iters=warmup_steps)
    main_scheduler = LinearLR(optimiser, start_factor=1.0, end_factor=0.1, total_iters=total_steps - warmup_steps)
    scheduler = SequentialLR(optimiser, [warmup_scheduler, main_scheduler], milestones=[warmup_steps])

    print(f"[info] Starting training for {num_epochs} epoch(s) on {device} ...")
    print(f"[info] Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"[info] Max sequence length: {max_seq_len}")
    print(f"[info] Current vocabulary size: {vocab_size}")
    print(f"[info] Model vocabulary size: {model_vocab_size}")
    print(f"[info] Regularization settings:")
    print(f"  - Weight decay: {weight_decay}")
    print(f"  - Gradient clipping: {gradient_clip}")
    print(f"  - Warmup steps: {warmup_steps}")
    print(f"  - Dropout: {model.config.dropout if hasattr(model.config, 'dropout') else 'default'}")
    print(f"  - Early stopping patience: {patience}")
    print(f"  - Validation split: {val_split:.1%}")

    # ------------------------------------------------------------------
    # Training artifacts and logging setup
    # ------------------------------------------------------------------
    save_dir = Path(save_path)
    save_dir.mkdir(exist_ok=True)
    
    # Training history for tracking progress
    training_history = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
        'total_training_time': 0,
        'early_stopped': False
    }
    
    # Save training configuration
    training_run_config = {
        'model_config': get_model_config(),
        'training_config': get_training_config(),
        'data_config': get_data_config(),
        'vocab_config': get_vocab_config(),
        'runtime_config': {
            'data_dir': data_dir,
            'file_glob': file_glob,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_seq_len': max_seq_len,
            'val_split': val_split,
            'patience': patience,
            'model_path': model_path,
            'save_path': save_path,
            'data_size_bytes': data_size,
            'data_size_str': data_size_str,
            'num_train_files': len(train_files),
            'num_val_files': len(val_files),
            'device': str(device),
            'vocab_size': vocab_size,
            'model_vocab_size': model_vocab_size,
            'weight_decay': weight_decay,
            'warmup_steps': warmup_steps,
            'gradient_clip': gradient_clip
        },
        'training_start_time': datetime.now().isoformat()
    }
    
    with open(save_dir / "training_config.json", "w") as f:
        json.dump(training_run_config, f, indent=2)
    
    # Copy vocabulary files to output directory
    vocab_config = get_vocab_config()
    import shutil
    try:
        vocab_to_int_file = vocab_config.get("vocab_to_int_file", "vocab_to_int.json")
        int_to_vocab_file = vocab_config.get("int_to_vocab_file", "int_to_vocab.json")
        shutil.copy2(vocab_to_int_file, save_dir / "vocab_to_int.json")
        shutil.copy2(int_to_vocab_file, save_dir / "int_to_vocab.json")
        print(f"[info] Vocabulary files copied to output directory")
    except Exception as e:
        print(f"[warning] Could not copy vocabulary files: {e}")
    
    # ------------------------------------------------------------------
    # Training loop with validation
    # ------------------------------------------------------------------
    import time
    training_start_time = time.time()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"\n[info] Epoch {epoch+1}/{num_epochs} - Training...")
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Skip batches with no real tokens
            if attention_mask.sum() == 0:
                continue

            # The labels are just the input sequence shifted internally by 🤗
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids)
            loss: torch.Tensor = outputs.loss  # scalar tensor

            optimiser.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimiser.step()
            scheduler.step()

            loss_value = loss.item()
            if not torch.isnan(loss) and not torch.isinf(loss):
                epoch_loss += loss_value
                num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx+1}, Loss: {loss_value:.4f}")

        # Calculate average training loss
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        
        # Validation phase
        print(f"[info] Epoch {epoch+1}/{num_epochs} - Validating...")
        val_loss = validate(model, val_loader, device)
        
        # Record training history
        current_lr = scheduler.get_last_lr()[0] if scheduler else learning_rate
        training_history['epochs'].append(epoch + 1)
        training_history['train_losses'].append(avg_train_loss)
        training_history['val_losses'].append(val_loss)
        training_history['learning_rates'].append(current_lr)
        
        # Print epoch summary
        if num_batches > 0:
            print(f"Epoch {epoch+1}/{num_epochs} – Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} – No valid training batches, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            training_history['best_epoch'] = epoch + 1
            training_history['best_val_loss'] = val_loss
            patience_counter = 0
            print(f"[info] New best validation loss: {val_loss:.4f}")
            # Save the best model and training state
            model.save_pretrained(save_dir)
            print(f"[info] Best model saved to: {save_dir}")
        else:
            patience_counter += 1
            print(f"[info] Validation loss did not improve. Patience: {patience_counter}/{patience}")
        
        # Save training history after each epoch
        with open(save_dir / "training_history.json", "w") as f:
            json.dump(training_history, f, indent=2)
            
        # Check early stopping condition
        if patience_counter >= patience:
            print(f"[info] Early stopping triggered after {epoch+1} epochs")
            print(f"[info] No improvement in validation loss for {patience} consecutive epochs")
            training_history['early_stopped'] = True
            break

    # Training complete - save final artifacts
    training_end_time = time.time()
    training_history['total_training_time'] = training_end_time - training_start_time
    training_history['final_epoch'] = epoch + 1
    
    # Update training config with final results
    training_run_config['training_end_time'] = datetime.now().isoformat()
    training_run_config['final_results'] = {
        'best_epoch': training_history['best_epoch'],
        'best_val_loss': training_history['best_val_loss'],
        'final_train_loss': training_history['train_losses'][-1] if training_history['train_losses'] else None,
        'final_val_loss': training_history['val_losses'][-1] if training_history['val_losses'] else None,
        'total_training_time_minutes': training_history['total_training_time'] / 60,
        'early_stopped': training_history['early_stopped'],
        'epochs_completed': training_history['final_epoch']
    }
    
    # Save final training configuration and history
    with open(save_dir / "training_config.json", "w") as f:
        json.dump(training_run_config, f, indent=2)
    
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)
    
    # Create a training summary
    summary = {
        'model_name': save_dir.name,
        'training_completed': datetime.now().isoformat(),
        'best_epoch': training_history['best_epoch'],
        'best_validation_loss': training_history['best_val_loss'],
        'total_epochs': training_history['final_epoch'],
        'training_time_minutes': round(training_history['total_training_time'] / 60, 2),
        'early_stopped': training_history['early_stopped'],
        'data_size': data_size_str,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'regularization': {
            'dropout': model.config.dropout if hasattr(model.config, 'dropout') else None,
            'weight_decay': weight_decay,
            'gradient_clip': gradient_clip
        }
    }
    
    with open(save_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Generate training plots if matplotlib available
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        epochs_list = training_history['epochs']
        ax1.plot(epochs_list, training_history['train_losses'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs_list, training_history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
        ax1.axvline(x=training_history['best_epoch'], color='g', linestyle='--', 
                   label=f'Best Epoch ({training_history["best_epoch"]})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate schedule
        ax2.plot(epochs_list, training_history['learning_rates'], 'purple', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[info] Training curves saved to: {save_dir / 'training_curves.png'}")
    except ImportError:
        print("[warning] matplotlib not available, skipping training plots")
    except Exception as e:
        print(f"[warning] Could not generate training plots: {e}")

    print("[info] Training complete!")
    print(f"[info] Best validation loss: {best_val_loss:.4f} at epoch {training_history['best_epoch']}")
    print(f"[info] Training time: {training_history['total_training_time']/60:.1f} minutes")
    print(f"[info] All training artifacts saved to: {save_dir}")
    
    # Load and return the best model
    if save_dir.exists():
        print(f"[info] Loading best model from: {save_dir}")
        model = AutoModelForCausalLM.from_pretrained(save_dir)
        return model
    else:
        print(f"[warning] Best model not found at {save_dir}, returning current model")
        return model

###############################################################################
# 6. Command line interface
###############################################################################

def main() -> None:
    # Load default values from config
    training_config = get_training_config()
    data_config = get_data_config()
    
    parser = argparse.ArgumentParser(description="Train the Grey Lord model (GPT-2) on binary data")
    parser.add_argument("--data-dir", type=str, 
                       default=data_config.get("default_data_dir", "./data"),
                       help="Directory containing training data")
    parser.add_argument("--file-glob", type=str, 
                       default=data_config.get("default_file_glob", "*"),
                       help="Glob pattern to match training files")
    parser.add_argument("--epochs", type=int, 
                       default=training_config.get("default_epochs", 10),
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, 
                       default=training_config.get("default_batch_size", 4),
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, 
                       default=training_config.get("default_lr", 5e-5),
                       help="Learning rate")
    parser.add_argument("--save-path", type=str, default=None,
                       help="Directory to save the trained model (default: {data_version}_{type}_{params}_{timestamp})")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to existing model to continue training from")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU training even if CUDA is available")
    parser.add_argument("--max-seq-len", type=int, 
                       default=training_config.get("default_max_seq_len", 512),
                       help="Maximum sequence length for training")
    parser.add_argument("--val-split", type=float, 
                       default=training_config.get("default_val_split", 0.2),
                       help="Fraction of data to use for validation")
    parser.add_argument("--patience", type=int, 
                       default=training_config.get("default_patience", 5),
                       help="Number of epochs to wait for improvement before early stopping")
    
    args = parser.parse_args()
    
    # Force CPU BEFORE importing torch modules
    if args.cpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Force PyTorch to use CPU
        torch.cuda.is_available = lambda: False
    
    try:
        train(data_dir=args.data_dir,
              file_glob=args.file_glob,
              num_epochs=args.epochs,
              batch_size=args.batch_size,
              learning_rate=args.learning_rate,
              save_path=args.save_path,
              max_seq_len=args.max_seq_len,
              model_path=args.model_path,
              val_split=args.val_split,
              patience=args.patience)
    except Exception as e:
        print(f"[error] Training failed: {e}")
        sys.exit(1)

###############################################################################
# 7. Entry-point guard – so the file can be imported without side-effects
###############################################################################

if __name__ == "__main__":
    main()
