"""tokenizer.py

Custom tokenization logic for Grey Lord model.
Handles binary data tokenization with special tokens and delay binning.
"""

from __future__ import annotations

import re
from typing import Dict, List


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
    
    Args:
        byte_stream: Raw bytes to tokenize
        vocab: Vocabulary mapping strings to integer IDs
        max_token_id: Optional maximum token ID to filter out tokens exceeding this value
        
    Returns:
        List of token IDs
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


def tokenize_file(file_path, vocab: Dict[str, int], max_token_id: int = None) -> List[int]:
    """Tokenize a single file and return token IDs.
    
    Args:
        file_path: Path to the file to tokenize
        vocab: Vocabulary mapping
        max_token_id: Optional maximum token ID filter
        
    Returns:
        List of token IDs from the file
    """
    try:
        with open(file_path, "rb") as f:
            raw = f.read()
        return custom_tokenizer(raw, vocab, max_token_id)
    except Exception as e:
        print(f"[error] Failed to read file {file_path}: {e}")
        return [] 