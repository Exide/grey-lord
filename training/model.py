"""model.py

Model building and configuration for Grey Lord GPT-2 style model.
Handles model creation, loading, and configuration management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from transformers import GPT2Config, AutoModelForCausalLM

from .config_utils import get_model_config
from .vocab import validate_vocabulary_compatibility


def build_model(vocab_size: int, model_path: Union[str, None] = None) -> AutoModelForCausalLM:
    """Create a GPT-2-style model, either fresh or loaded from existing checkpoint.
    
    Args:
        vocab_size: Size of the vocabulary for the model
        model_path: Optional path to existing model to load
        
    Returns:
        AutoModelForCausalLM instance
    """
    if model_path and Path(model_path).exists():
        print(f"[info] Loading existing model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Check vocabulary size compatibility
        model_vocab_size = model.config.vocab_size
        validate_vocabulary_compatibility(vocab_size, model_vocab_size)
        
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


def setup_device_and_model(vocab_size: int, model_path: Union[str, None] = None) -> tuple[torch.device, AutoModelForCausalLM]:
    """Set up the device and model for training.
    
    Args:
        vocab_size: Size of the vocabulary
        model_path: Optional path to existing model
        
    Returns:
        Tuple of (device, model)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")
    
    model = build_model(vocab_size, model_path)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[info] Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return device, model


def get_model_vocab_size(model: AutoModelForCausalLM) -> int:
    """Get the vocabulary size from a model's configuration.
    
    Args:
        model: The model to get vocab size from
        
    Returns:
        Vocabulary size from model config
    """
    return model.config.vocab_size


def print_model_info(model: AutoModelForCausalLM) -> None:
    """Print detailed information about the model.
    
    Args:
        model: Model to print information about
    """
    config = model.config
    print(f"[info] Model Configuration:")
    print(f"  • Architecture: {config.model_type}")
    print(f"  • Vocabulary size: {config.vocab_size:,}")
    print(f"  • Hidden size: {config.n_embd}")
    print(f"  • Number of layers: {config.n_layer}")
    print(f"  • Number of heads: {config.n_head}")
    print(f"  • Max position embeddings: {config.n_positions}")
    if hasattr(config, 'dropout'):
        print(f"  • Dropout: {config.dropout}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  • Parameters: {total_params:,} total, {trainable_params:,} trainable") 
