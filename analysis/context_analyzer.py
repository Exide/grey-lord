#!/usr/bin/env python3
"""Analyze context length vs batch size trade-offs for telnet data"""

import sys
from pathlib import Path

# Add src directory to path to find config_utils
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from pathlib import Path
from config_utils import get_model_config
from calculate_context_window import estimate_memory_usage

def analyze_context_vs_speed():
    """Find the best balance between context length and training speed"""
    
    available_memory_mb = 10.0 * 1024 * 0.85  # 10GB with safety margin
    model_config = get_model_config()
    max_positions = model_config['n_positions']
    
    print(f"=== Context Length vs Speed Analysis ===")
    print(f"Model supports up to {max_positions} tokens")
    print(f"Available GPU memory: {available_memory_mb:.0f}MB")
    
    # Test different context lengths
    context_lengths = [512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
    
    print(f"\n{'Context':<7} | {'Max Batch':<9} | {'Memory':<8} | {'Speed Est.':<11} | {'Capability'}")
    print("-" * 70)
    
    recommendations = []
    
    for ctx_len in context_lengths:
        if ctx_len > max_positions:
            print(f"{ctx_len:<7} | {'❌ Exceeds model limit':<49}")
            continue
        
        # Find max batch size for this context length
        max_batch = 1
        for batch_size in range(1, 33):
            memory_info = estimate_memory_usage(ctx_len, batch_size)
            if memory_info['total_memory_mb'] <= available_memory_mb:
                max_batch = batch_size
            else:
                break
        
        memory_used = estimate_memory_usage(ctx_len, max_batch)['total_memory_mb']
        
        # Estimate relative training speed (batch_size * efficiency_factor)
        efficiency = min(1.0, max_batch / 8)  # Diminishing returns after batch size 8
        speed_score = max_batch * efficiency
        
        # Estimate capability based on context length
        if ctx_len <= 1024:
            capability = "Basic patterns"
        elif ctx_len <= 2048:
            capability = "Command sequences"
        elif ctx_len <= 4096:
            capability = "Session chunks"
        else:
            capability = "Full sessions"
        
        print(f"{ctx_len:<7} | {max_batch:<9} | {memory_used:.0f}MB   | {speed_score:.1f}x        | {capability}")
        
        recommendations.append({
            'context': ctx_len,
            'max_batch': max_batch,
            'memory': memory_used,
            'speed_score': speed_score,
            'capability': capability
        })
    
    return recommendations

def analyze_telnet_data_characteristics():
    """Analyze what context lengths make sense for telnet data"""
    print(f"\n=== Telnet Session Analysis ===")
    print(f"Based on typical telnet proxy patterns:")
    print(f"")
    print(f"🔸 **512-1024 tokens**: Individual commands + responses")
    print(f"   • Good for: Basic command completion")
    print(f"   • Captures: Single interactions")
    print(f"")
    print(f"🔸 **2048-3072 tokens**: Command sequences")
    print(f"   • Good for: Multi-step operations")
    print(f"   • Captures: File transfers, directory listings")
    print(f"")
    print(f"🔸 **4096-6144 tokens**: Session segments")
    print(f"   • Good for: Workflow understanding")
    print(f"   • Captures: Login → work → logout patterns")
    print(f"")
    print(f"🔸 **6144-8192 tokens**: Full sessions")
    print(f"   • Good for: Complete session modeling")
    print(f"   • Captures: Entire user workflows")

def recommend_strategy(recommendations):
    """Provide strategic recommendations"""
    print(f"\n=== Strategic Recommendations ===")
    
    # Find different sweet spots
    speed_optimized = max(recommendations, key=lambda x: x['speed_score'])
    context_optimized = max(recommendations, key=lambda x: x['context'])
    balanced = max(recommendations, key=lambda x: x['speed_score'] * (x['context'] / 1000))
    
    print(f"🚀 **Speed Optimized**: {speed_optimized['context']} tokens, batch {speed_optimized['max_batch']}")
    print(f"   → {speed_optimized['speed_score']:.1f}x faster training")
    print(f"   → Good for: Rapid prototyping and testing")
    print(f"")
    print(f"🎯 **Balanced Approach**: {balanced['context']} tokens, batch {balanced['max_batch']}")
    print(f"   → {balanced['speed_score']:.1f}x training speed")
    print(f"   → Good for: Production training")
    print(f"")
    print(f"🧠 **Context Optimized**: {context_optimized['context']} tokens, batch {context_optimized['max_batch']}")
    print(f"   → {context_optimized['speed_score']:.1f}x training speed")
    print(f"   → Good for: Maximum model capability")
    
    print(f"\n=== Training Strategy ===")
    print(f"I recommend a **progressive approach**:")
    print(f"")
    print(f"1. **Start Fast** ({speed_optimized['context']} tokens): Quick iteration, verify training works")
    print(f"2. **Scale Up** ({balanced['context']} tokens): Better performance, reasonable speed")
    print(f"3. **Final Model** ({context_optimized['context']} tokens): Maximum capability")
    print(f"")
    print(f"This way you get fast feedback early, then scale to maximum capability.")

def main():
    print("Analyzing context length vs training speed trade-offs...")
    
    recommendations = analyze_context_vs_speed()
    analyze_telnet_data_characteristics()
    recommend_strategy(recommendations)
    
    print(f"\n=== The Bottom Line ===")
    print(f"• **Different context lengths = different model capabilities**")
    print(f"• **Longer context ≠ always better** (diminishing returns)")
    print(f"• **Your GPU can handle up to ~6000 tokens efficiently**")
    print(f"• **Progressive training is often the best strategy**")

if __name__ == "__main__":
    main() 
