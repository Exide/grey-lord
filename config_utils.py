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
    return config["model"]

def get_training_config() -> Dict[str, Any]:
    """Get training configuration parameters"""
    config = load_config()
    return config["training"]

def get_data_config() -> Dict[str, Any]:
    """Get data configuration parameters"""
    config = load_config()
    return config["data"]

def get_vocab_config() -> Dict[str, Any]:
    """Get vocabulary configuration parameters"""
    config = load_config()
    return config["vocab"]

def print_config_summary():
    """Print a summary of the current configuration"""
    config = load_config()
    
    print("=== Configuration Summary ===")
    print(f"Model: {config['model']['n_layer']} layers, {config['model']['n_embd']} embedding dim")
    print(f"Max sequence length: {config['model']['n_positions']} tokens")
    print(f"Vocabulary size: {config['model']['vocab_size']}")
    print(f"Default training: {config['training']['default_epochs']} epochs, batch size {config['training']['default_batch_size']}")
    print(f"Data source: {config['data']['default_data_dir']}")
    print(f"File pattern: {config['data']['default_file_glob']}") 