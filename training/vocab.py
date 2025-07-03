"""vocab.py

Vocabulary loading, validation, and management functionality.
Handles the loading of vocabulary files and validates their integrity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

from .config_utils import get_vocab_config


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


def validate_vocabulary_compatibility(vocab_size: int, model_vocab_size: int) -> None:
    """Validate that vocabulary sizes are compatible between current vocab and model."""
    print(f"[info] Current vocabulary size: {vocab_size}")
    print(f"[info] Model vocabulary size: {model_vocab_size}")
    
    if vocab_size > model_vocab_size:
        print(f"[warning] Current vocabulary ({vocab_size}) is larger than model vocabulary ({model_vocab_size})")
        print("[warning] This will cause indexing errors. Please retrain from scratch or use compatible vocabulary.")
        raise ValueError(f"Vocabulary size mismatch: current={vocab_size}, model={model_vocab_size}")
    elif vocab_size < model_vocab_size:
        print(f"[warning] Current vocabulary ({vocab_size}) is smaller than model vocabulary ({model_vocab_size})")
        print("[warning] Some embeddings will be unused, but training can continue.")


def get_vocab_size(vocab_to_int: Dict[str, int]) -> int:
    """Calculate the vocabulary size needed for model configuration."""
    return max(vocab_to_int.values()) + 1 
