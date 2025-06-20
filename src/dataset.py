"""dataset.py

PyTorch Dataset and DataLoader utilities for Grey Lord training.
Handles byte stream datasets and custom collate functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import Dataset

from tokenizer import custom_tokenizer


class ByteStreamDataset(Dataset):
    """Dataset that lazily reads binary files and tokenises them on-the-fly."""

    def __init__(self, paths: Iterable[Path], vocab: Dict[str, int], pad_token_id: int, max_token_id: int = None, label: str = ""):
        """Initialize the dataset.
        
        Args:
            paths: Iterable of file paths to include in dataset
            vocab: Vocabulary mapping strings to integers
            pad_token_id: ID of the padding token
            max_token_id: Optional maximum token ID for filtering
            label: Optional label for logging (e.g., "training", "validation")
        """
        self.paths = list(paths)
        self.vocab = vocab
        self.pad_token_id = pad_token_id
        self.max_token_id = max_token_id
        dataset_type = f" ({label})" if label else ""
        print(f"[info] Dataset{dataset_type} initialized with {len(self.paths)} files.")

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
    """Pad variable-length sequences and build an attention mask.
    
    Args:
        batch: List of token ID tensors
        max_seq_len: Maximum sequence length to pad/truncate to
        pad_token_id: Token ID to use for padding
        
    Returns:
        Dictionary with 'input_ids' and 'attention_mask' tensors
    """
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


def create_collate_fn(max_seq_len: int, pad_token_id: int):
    """Create a collate function with fixed parameters.
    
    Args:
        max_seq_len: Maximum sequence length
        pad_token_id: Padding token ID
        
    Returns:
        Collate function that can be used with DataLoader
    """
    def limited_collate_fn(batch):
        return collate_fn(batch, max_seq_len, pad_token_id)
    return limited_collate_fn


def calculate_data_size(file_paths: List[Path]) -> tuple[int, str]:
    """Calculate total data size and return both bytes and human-readable string.
    
    Args:
        file_paths: List of file paths to calculate size for
        
    Returns:
        Tuple of (size_in_bytes, human_readable_string)
    """
    data_size = 0
    for file_path in file_paths:
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
    
    return data_size, data_size_str


def split_files(file_paths: List[Path], validation_split: float) -> tuple[List[Path], List[Path]]:
    """Split file paths into training and validation sets.
    
    Args:
        file_paths: List of all file paths
        validation_split: Fraction of files to use for validation
        
    Returns:
        Tuple of (training_files, validation_files)
    """
    num_validation_files = max(1, int(len(file_paths) * validation_split))
    validation_files = file_paths[:num_validation_files]
    training_files = file_paths[num_validation_files:]
    
    return training_files, validation_files 