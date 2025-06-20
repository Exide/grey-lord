#!/usr/bin/env python3
"""Find optimal batch size for training"""

import sys
from pathlib import Path

# Add src directory to path to find config_utils
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from config_utils import get_model_config
from calculate_max_seq_len import estimate_memory_usage

def find_optimal_batch_size(available_memory_gb=10.0, target_seq_lens=None):
    """Find optimal batch sizes for different sequence lengths"""
    
    if target_seq_lens is None:
        target_seq_lens = [512, 1024, 2048, 4096, 6000]
    
    available_memory_mb = available_memory_gb * 1024 * 0.85  # 15% safety margin
    
    print(f"=== Batch Size Optimizer for {available_memory_gb}GB GPU ===")
    print(f"Available memory: {available_memory_mb:.0f} MB (with 15% safety margin)")
    
    model_config = get_model_config()
    print(f"Model: {model_config['n_layer']} layers, {model_config['n_embd']} embedding dim")
    print(f"Max position embeddings: {model_config['n_positions']} tokens")
    
    print(f"\n{'Seq Len':<8} | {'Batch=1':<10} | {'Batch=2':<10} | {'Batch=4':<10} | {'Batch=8':<10} | {'Max Batch':<10}")
    print("-" * 70)
    
    recommendations = []
    
    for seq_len in target_seq_lens:
        if seq_len > model_config['n_positions']:
            print(f"{seq_len:<8} | {'❌ Too long for model position embeddings':<58}")
            continue
            
        batch_sizes = [1, 2, 4, 8]
        memory_usages = []
        max_batch = 1
        
        for batch_size in batch_sizes:
            memory_info = estimate_memory_usage(seq_len, batch_size)
            memory_mb = memory_info['total_memory_mb']
            memory_usages.append(memory_mb)
            
            if memory_mb <= available_memory_mb:
                max_batch = batch_size
        
        # Find actual maximum batch size
        test_batch = max_batch
        while test_batch <= 32:  # reasonable upper limit
            memory_info = estimate_memory_usage(seq_len, test_batch)
            if memory_info['total_memory_mb'] <= available_memory_mb:
                max_batch = test_batch
                test_batch += 1
            else:
                break
        
        # Format memory usage for display
        usage_strs = []
        for i, (batch_size, memory_mb) in enumerate(zip(batch_sizes, memory_usages)):
            if memory_mb <= available_memory_mb:
                usage_strs.append(f"✅ {memory_mb:.0f}MB")
            else:
                usage_strs.append(f"❌ {memory_mb:.0f}MB")
        
        print(f"{seq_len:<8} | {usage_strs[0]:<10} | {usage_strs[1]:<10} | {usage_strs[2]:<10} | {usage_strs[3]:<10} | {max_batch:<10}")
        
        # Calculate speedup estimate
        theoretical_speedup = max_batch  # Linear speedup (optimistic)
        actual_speedup = max_batch * 0.8  # More realistic (80% efficiency)
        
        recommendations.append({
            'seq_len': seq_len,
            'max_batch': max_batch,
            'memory_usage': estimate_memory_usage(seq_len, max_batch)['total_memory_mb'],
            'speedup': actual_speedup
        })
    
    print(f"\n=== Recommendations ===")
    for rec in recommendations:
        speedup_text = f"{rec['speedup']:.1f}x faster" if rec['speedup'] > 1 else "baseline"
        print(f"Seq len {rec['seq_len']:<4}: Use --batch-size {rec['max_batch']:<2} ({rec['memory_usage']:.0f}MB, ~{speedup_text})")
    
    print(f"\n=== Key Insights ===")
    print(f"• Your GPU can handle much larger batch sizes!")
    print(f"• Larger batches = better GPU utilization = faster training")
    print(f"• Start with batch size 4-8 for most sequence lengths")
    print(f"• Monitor GPU memory usage during training")
    
    # Find sweet spot recommendation
    best_rec = max(recommendations, key=lambda x: x['speedup'] * (x['seq_len'] / 1000))  # Balance speed and context
    print(f"\n🎯 **Sweet Spot**: --max-seq-len {best_rec['seq_len']} --batch-size {best_rec['max_batch']}")
    print(f"   This gives you good context window + ~{best_rec['speedup']:.1f}x speedup")

def show_gradient_accumulation_option():
    """Explain gradient accumulation as an alternative"""
    print(f"\n=== Alternative: Gradient Accumulation ===")
    print(f"If you want even larger effective batch sizes:")
    print(f"• Keep hardware batch size low (e.g., 4)")
    print(f"• Use gradient accumulation to simulate larger batches")
    print(f"• Example: batch_size=4 + accumulation=4 = effective batch size 16")
    print(f"• This uses less memory but gives similar training dynamics")

def main():
    print("Analyzing optimal batch sizes for your RTX 3080 Ti...")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"Detected: {torch.cuda.get_device_name(device)} ({gpu_memory_gb:.1f}GB)")
        available_memory = min(gpu_memory_gb, 10.0)  # Conservative estimate
    else:
        print("CUDA not detected, using manual estimate")
        available_memory = 10.0
    
    find_optimal_batch_size(available_memory)
    show_gradient_accumulation_option()

if __name__ == "__main__":
    main() 