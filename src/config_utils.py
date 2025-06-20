#!/usr/bin/env python3
"""Configuration utilities for the Grey Lord training project"""

import json
from pathlib import Path
from typing import Dict, Any

CONFIG_FILE = Path("model_config.json")

def load_config() -> Dict[str, Any]:
    """Load configuration from the config file"""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")
    
    with CONFIG_FILE.open('r', encoding='utf-8') as f:
        return json.load(f)

def get_model_config() -> Dict[str, Any]:
    """Get model configuration parameters"""
    config = load_config()
    return config.get("model", {})

def get_training_config() -> Dict[str, Any]:
    """Get training configuration parameters"""
    config = load_config()
    return config.get("training", {})

def get_data_config() -> Dict[str, Any]:
    """Get data configuration parameters"""
    config = load_config()
    return config.get("data", {})

def get_vocab_config() -> Dict[str, Any]:
    """Get vocabulary configuration parameters"""
    config = load_config()
    return config.get("vocab", {})

def print_config_summary():
    """Print a summary of the current configuration"""
    config = load_config()
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    
    print("=== Configuration Summary ===")
    print(f"Model: {model_config.get('n_layer', '?')} layers, {model_config.get('n_embd', '?')} embedding dim")
    print(f"Max sequence length: {model_config.get('n_positions', '?')} tokens")
    print(f"Vocabulary size: {model_config.get('vocab_size', '?')}")
    print(f"Default training: {training_config.get('default_epochs', '?')} epochs, batch size {training_config.get('default_batch_size', '?')}")
    
    # Extract dataset version from directory name
    data_dir = data_config.get('default_data_dir', '')
    if "_v" in data_dir:
        dataset_version = data_dir.split("_v")[-1]
    else:
        dataset_version = "unknown"
    print(f"Dataset: {dataset_version}")
    print(f"Data source: {data_config.get('default_data_dir', '?')}")
    print(f"File pattern: {data_config.get('default_file_glob', '?')}") 