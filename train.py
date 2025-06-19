"""train.py
A clear, step-by-step replication of the original training notebook in plain Python.
Each section is small, self-contained, and heavily commented so that you can
reason about what every single line does.

ENVIRONMENT REQUIREMENTS:
This script requires an Anaconda environment with PyTorch installed.
Run with: & "$env:USERPROFILE\Anaconda3\envs\pytorch\python.exe" train.py

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
        vocab_file = Path(vocab_config["vocab_to_int_file"])
        int_vocab_file = Path(vocab_config["int_to_vocab_file"])
        pad_token = vocab_config["pad_token"]
        
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

    def __init__(self, paths: Iterable[Path], vocab: Dict[str, int], pad_token_id: int, max_token_id: int = None):
        self.paths = list(paths)
        self.vocab = vocab
        self.pad_token_id = pad_token_id
        self.max_token_id = max_token_id
        print(f"[info] Dataset initialized with {len(self.paths)} files.")

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
            n_positions=model_config["n_positions"],
            n_embd=model_config["n_embd"],
            n_layer=model_config["n_layer"],
            n_head=model_config["n_head"],
        )
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


def train(data_dir: str, file_glob: str, num_epochs: int = 3, batch_size: int = 1, 
          lr: float = 5e-5, save_path: Union[str, None] = None, max_seq_len: int = 512, 
          model_path: Union[str, None] = None, val_split: float = 0.2) -> None:
    
    # Load vocabulary first
    vocab_to_int, int_to_vocab, pad_token_id = load_vocabulary()
    vocab_size = max(vocab_to_int.values()) + 1  # ensure the highest ID fits in the embedding
    
    print(f"[DEBUG] Vocabulary loaded: {len(vocab_to_int)} items")
    print(f"[DEBUG] Max token ID: {max(vocab_to_int.values())}")
    print(f"[DEBUG] Calculated vocab_size: {vocab_size}")
    
    # Set default save path if not provided
    if save_path is None:
        save_path = f"grey-lord.{datetime.now().strftime('%Y-%m-%dT%H-%M-%S.%f')}.gpt2"
    
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
    
    # ------------------------------------------------------------------
    # Model setup (needed early to get model vocabulary size)
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(vocab_size, model_path)
    model_vocab_size = model.config.vocab_size
    model = model.to(device)

    # Create separate datasets and dataloaders with vocabulary size validation
    train_dataset = ByteStreamDataset(train_files, vocab_to_int, pad_token_id, model_vocab_size)
    val_dataset = ByteStreamDataset(val_files, vocab_to_int, pad_token_id, model_vocab_size)
    
    # Create a collate function with max sequence length and pad token
    def limited_collate_fn(batch):
        return collate_fn(batch, max_seq_len, pad_token_id)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=limited_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=limited_collate_fn)

    # ------------------------------------------------------------------
    # Optimiser setup
    # ------------------------------------------------------------------
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"[info] Starting training for {num_epochs} epoch(s) on {device} ...")
    print(f"[info] Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"[info] Max sequence length: {max_seq_len}")
    print(f"[info] Current vocabulary size: {vocab_size}")
    print(f"[info] Model vocabulary size: {model_vocab_size}")

    # ------------------------------------------------------------------
    # Training loop with validation
    # ------------------------------------------------------------------
    best_val_loss = float('inf')
    
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
            optimiser.step()

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
        
        # Print epoch summary
        if num_batches > 0:
            print(f"Epoch {epoch+1}/{num_epochs} – Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} – No valid training batches, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"[info] New best validation loss: {val_loss:.4f}")
            # Save the best model
            save_dir = Path(save_path)
            save_dir.mkdir(exist_ok=True)
            model.save_pretrained(save_dir)
            print(f"[info] Best model saved to: {save_dir}")

    print("[info] Training complete!")
    print(f"[info] Best validation loss: {best_val_loss:.4f}")

###############################################################################
# 6. Command line interface
###############################################################################

def main() -> None:
    # Load default values from config
    training_config = get_training_config()
    data_config = get_data_config()
    
    parser = argparse.ArgumentParser(description="Train the Grey Lord model (GPT-2) on binary data")
    parser.add_argument("--data-dir", type=str, 
                       default=data_config["default_data_dir"],
                       help="Directory containing training data")
    parser.add_argument("--file-glob", type=str, 
                       default=data_config["default_file_glob"],
                       help="Glob pattern to match training files")
    parser.add_argument("--epochs", type=int, 
                       default=training_config["default_epochs"],
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, 
                       default=training_config["default_batch_size"],
                       help="Training batch size")
    parser.add_argument("--lr", type=float, 
                       default=training_config["default_lr"],
                       help="Learning rate")
    parser.add_argument("--save-path", type=str, default="trained_model",
                       help="Directory to save the trained model")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to existing model to continue training from")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU training even if CUDA is available")
    parser.add_argument("--max-seq-len", type=int, 
                       default=training_config["default_max_seq_len"],
                       help="Maximum sequence length for training")
    parser.add_argument("--val-split", type=float, 
                       default=training_config["default_val_split"],
                       help="Fraction of data to use for validation")
    
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
              lr=args.lr,
              save_path=args.save_path,
              max_seq_len=args.max_seq_len,
              model_path=args.model_path,
              val_split=args.val_split)
    except Exception as e:
        print(f"[error] Training failed: {e}")
        sys.exit(1)

###############################################################################
# 7. Entry-point guard – so the file can be imported without side-effects
###############################################################################

if __name__ == "__main__":
    main()
