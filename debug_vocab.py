#!/usr/bin/env python3
"""Debug script to check vocabulary and model compatibility"""

import json
from pathlib import Path
from transformers import AutoModelForCausalLM

def main():
    print("=== Vocabulary Debug ===")
    
    # Load vocabulary files
    vocab_file = Path("vocab_to_int.json")
    int_vocab_file = Path("int_to_vocab.json")
    
    if not vocab_file.exists():
        print(f"ERROR: {vocab_file} not found!")
        return
    
    if not int_vocab_file.exists():
        print(f"ERROR: {int_vocab_file} not found!")
        return
    
    # Load and analyze vocabulary
    with vocab_file.open('r') as f:
        vocab_to_int = json.load(f)
    
    with int_vocab_file.open('r') as f:
        int_to_vocab = json.load(f)
    
    print(f"Vocabulary size: {len(vocab_to_int)}")
    
    # Check token ID range
    token_ids = list(vocab_to_int.values())
    min_id = min(token_ids)
    max_id = max(token_ids)
    
    print(f"Token ID range: {min_id} to {max_id}")
    print(f"Expected vocabulary size from max ID: {max_id + 1}")
    
    # Check for gaps
    expected_ids = set(range(max_id + 1))
    actual_ids = set(token_ids)
    missing_ids = expected_ids - actual_ids
    
    if missing_ids:
        print(f"Missing token IDs: {sorted(list(missing_ids))[:20]}")
        if len(missing_ids) > 20:
            print(f"... and {len(missing_ids) - 20} more")
    else:
        print("No missing token IDs - vocabulary is contiguous")
    
    # Check model if it exists
    model_path = Path("trained_model")
    if model_path.exists():
        print("\n=== Model Debug ===")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            model_vocab_size = model.config.vocab_size
            print(f"Model vocabulary size: {model_vocab_size}")
            
            # Check compatibility
            current_vocab_size = max_id + 1  # This is what the train script calculates
            print(f"Current vocab calculation: max(vocab_to_int.values()) + 1 = {current_vocab_size}")
            
            if current_vocab_size > model_vocab_size:
                print(f"❌ PROBLEM: Current vocab size ({current_vocab_size}) > Model vocab size ({model_vocab_size})")
                print("   This will cause CUDA indexing errors!")
                print("   Solution: Retrain from scratch or fix vocabulary")
            elif current_vocab_size == model_vocab_size:
                print("✅ Vocabulary sizes match")
            else:
                print(f"⚠️  Current vocab size ({current_vocab_size}) < Model vocab size ({model_vocab_size})")
                print("   This should work but some embeddings will be unused")
                
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"\nModel path {model_path} doesn't exist")
    
    # Check for problematic tokens
    print("\n=== High Token ID Analysis ===")
    high_tokens = [(k, v) for k, v in vocab_to_int.items() if v > 260]  # Check near vocabulary boundary
    if high_tokens:
        print("Tokens with high IDs:")
        for token, id in sorted(high_tokens, key=lambda x: x[1]):
            print(f"  {token}: {id}")
    else:
        print("No tokens with unusually high IDs found")

if __name__ == "__main__":
    main() 