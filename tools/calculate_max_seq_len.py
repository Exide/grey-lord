#!/usr/bin/env python3
"""Calculate maximum sequence length based on available GPU memory"""

import sys
from pathlib import Path

# Add src directory to path to find config_utils
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import math
from config_utils import get_model_config

def estimate_memory_usage(seq_len, batch_size=1):
    """Estimate GPU memory usage for the model configuration"""
    
    # Load model configuration from config file
    model_config = get_model_config()
    vocab_size = model_config["vocab_size"]
    n_embd = model_config["n_embd"]
    n_layer = model_config["n_layer"]
    n_head = model_config["n_head"]
    n_positions = model_config["n_positions"]
    
    # Calculate model parameters
    # Embedding layers
    token_embed_params = vocab_size * n_embd
    pos_embed_params = n_positions * n_embd
    
    # Transformer blocks
    # Each block has: self-attention (4 * n_embd * n_embd) + MLP (8 * n_embd * n_embd) + layer norms
    per_block_params = (4 * n_embd * n_embd) + (8 * n_embd * n_embd) + (2 * n_embd)
    transformer_params = n_layer * per_block_params
    
    # Output layer (usually tied to token embeddings)
    total_params = token_embed_params + pos_embed_params + transformer_params
    
    # Memory calculations (in bytes, assuming fp32)
    bytes_per_param = 4  # fp32
    
    # 1. Model parameters
    model_memory = total_params * bytes_per_param
    
    # 2. Gradients (same size as parameters)
    gradient_memory = total_params * bytes_per_param
    
    # 3. Optimizer states (AdamW: 2x parameters for momentum and variance)
    optimizer_memory = total_params * bytes_per_param * 2
    
    # 4. Activations (the tricky part that scales with sequence length)
    # Key components:
    # - Hidden states: batch_size * seq_len * n_embd * n_layer
    # - Attention matrices: batch_size * n_head * seq_len * seq_len * n_layer
    # - MLP activations: batch_size * seq_len * (n_embd * 4) * n_layer
    
    hidden_states_memory = batch_size * seq_len * n_embd * n_layer * bytes_per_param
    attention_memory = batch_size * n_head * seq_len * seq_len * n_layer * bytes_per_param
    mlp_memory = batch_size * seq_len * (n_embd * 4) * n_layer * bytes_per_param
    
    # Additional overhead (intermediate computations, etc.)
    activation_overhead = (hidden_states_memory + mlp_memory) * 0.5
    
    total_activation_memory = hidden_states_memory + attention_memory + mlp_memory + activation_overhead
    
    # Total memory
    total_memory = model_memory + gradient_memory + optimizer_memory + total_activation_memory
    
    return {
        'total_memory_mb': total_memory / (1024 * 1024),
        'model_memory_mb': model_memory / (1024 * 1024),
        'gradient_memory_mb': gradient_memory / (1024 * 1024),
        'optimizer_memory_mb': optimizer_memory / (1024 * 1024),
        'activation_memory_mb': total_activation_memory / (1024 * 1024),
        'attention_memory_mb': attention_memory / (1024 * 1024),
        'total_params': total_params
    }

def find_max_seq_len(available_memory_gb, batch_size=1, safety_margin=0.85):
    """Binary search to find maximum sequence length that fits in memory"""
    
    available_memory_mb = available_memory_gb * 1024 * safety_margin
    
    # Binary search
    low, high = 64, 8192  # reasonable range
    best_seq_len = 64
    
    while low <= high:
        mid = (low + high) // 2
        memory_info = estimate_memory_usage(mid, batch_size)
        
        if memory_info['total_memory_mb'] <= available_memory_mb:
            best_seq_len = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return best_seq_len, estimate_memory_usage(best_seq_len, batch_size)

def main():
    print("=== GPU Memory Calculator for Sequence Length ===")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"Detected GPU: {torch.cuda.get_device_name(device)}")
        print(f"Total GPU Memory: {gpu_memory_gb:.1f} GB")
    else:
        print("CUDA not available, using manual input")
        gpu_memory_gb = None
    
    # Get available memory from user
    try:
        if gpu_memory_gb:
            available_memory = input(f"Available GPU memory for training (default: {gpu_memory_gb:.1f} GB): ").strip()
            if not available_memory:
                available_memory_gb = gpu_memory_gb
            else:
                available_memory_gb = float(available_memory)
        else:
            available_memory_gb = float(input("Available GPU memory for training (GB): "))
        
        batch_size = int(input("Batch size (default: 1): ").strip() or "1")
    except EOFError:
        # Use defaults if running non-interactively
        print("Running in non-interactive mode, using defaults...")
        available_memory_gb = gpu_memory_gb if gpu_memory_gb else 10.0
        batch_size = 1
    
    print(f"\nCalculating optimal sequence length for {available_memory_gb:.1f} GB GPU memory...")
    
    # Find maximum sequence length
    max_seq_len, memory_breakdown = find_max_seq_len(available_memory_gb, batch_size)
    
    print(f"\n=== Results ===")
    print(f"Recommended max sequence length: {max_seq_len}")
    print(f"Total estimated memory usage: {memory_breakdown['total_memory_mb']:.1f} MB")
    print(f"Safety margin used: 15% (to account for PyTorch overhead)")
    
    print(f"\n=== Memory Breakdown ===")
    print(f"Model parameters: {memory_breakdown['model_memory_mb']:.1f} MB")
    print(f"Gradients: {memory_breakdown['gradient_memory_mb']:.1f} MB") 
    print(f"Optimizer states: {memory_breakdown['optimizer_memory_mb']:.1f} MB")
    print(f"Activations: {memory_breakdown['activation_memory_mb']:.1f} MB")
    print(f"  - Attention matrices: {memory_breakdown['attention_memory_mb']:.1f} MB")
    print(f"Total parameters: {memory_breakdown['total_params']:,}")
    
    # Test a few other sequence lengths for comparison
    print(f"\n=== Comparison Table ===")
    print("Seq Len | Memory (MB) | Fits in GPU?")
    print("-" * 40)
    
    test_lengths = [512, 1024, 2048, 4096, max_seq_len]
    for seq_len in sorted(set(test_lengths)):
        memory_info = estimate_memory_usage(seq_len, batch_size)
        fits = "✅" if memory_info['total_memory_mb'] <= available_memory_gb * 1024 * 0.85 else "❌"
        print(f"{seq_len:7d} | {memory_info['total_memory_mb']:10.1f} | {fits}")
    
    print(f"\n=== Recommendations ===")
    print(f"1. Use --max-seq-len {max_seq_len} for optimal memory usage")
    print(f"2. If you get OOM errors, try --max-seq-len {max_seq_len // 2}")
    print(f"3. You can increase batch size or use gradient accumulation if needed")
    
    # Get model config
    model_config = get_model_config()
    model_n_positions = model_config["n_positions"]
    
    if max_seq_len > model_n_positions:
        print(f"\n⚠️  WARNING: Your model's position embeddings are limited to {model_n_positions} tokens!")
        print(f"   To use sequences longer than {model_n_positions}, you need to increase n_positions in the model config.")
    else:
        print(f"\n✅ Your model supports up to {model_n_positions} tokens, so {max_seq_len} is fine!")

if __name__ == "__main__":
    main()
