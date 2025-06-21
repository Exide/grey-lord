#!/usr/bin/env python3
"""Debug model and vocabulary compatibility"""

import sys
from pathlib import Path

# Add src directory to path to find config_utils
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM

def main():
    print("=== Model Configuration Debug ===")
    
    model_path = Path("trained_model")
    if not model_path.exists():
        print(f"ERROR: {model_path} not found!")
        return
    
    try:
        # Load model and check all configuration
        model = AutoModelForCausalLM.from_pretrained(model_path)
        config = model.config
        
        print(f"Model type: {config.model_type}")
        print(f"Vocabulary size: {config.vocab_size}")
        print(f"Max position embeddings: {config.n_positions if hasattr(config, 'n_positions') else config.max_position_embeddings}")
        print(f"Hidden size: {config.n_embd if hasattr(config, 'n_embd') else config.hidden_size}")
        print(f"Number of layers: {config.n_layer if hasattr(config, 'n_layer') else config.num_layers}")
        print(f"Number of heads: {config.n_head if hasattr(config, 'n_head') else config.num_heads}")
        
        # Check if position embeddings match what we're trying to use
        max_seq_len = 4096  # This is what you're using in training
        max_pos_emb = config.n_positions if hasattr(config, 'n_positions') else config.max_position_embeddings
        
        print(f"\n=== Position Embedding Check ===")
        print(f"Requested max sequence length: {max_seq_len}")
        print(f"Model's max position embeddings: {max_pos_emb}")
        
        if max_seq_len > max_pos_emb:
            print(f"❌ PROBLEM: Requested sequence length ({max_seq_len}) > Model's max position embeddings ({max_pos_emb})")
            print("   This will cause indexing errors!")
            print(f"   Solution: Use --context-window {max_pos_emb} or smaller")
        else:
            print("✅ Sequence length is within model's position embedding range")
        
        # Check embedding layer sizes
        print(f"\n=== Embedding Layer Check ===")
        if hasattr(model, 'transformer'):
            # GPT-2 style model
            wte = model.transformer.wte  # word token embeddings
            wpe = model.transformer.wpe  # word position embeddings
            print(f"Word token embedding shape: {wte.weight.shape}")
            print(f"Word position embedding shape: {wpe.weight.shape}")
            
            # Check if the actual embedding weights match config
            actual_vocab_size = wte.weight.shape[0]
            actual_pos_emb_size = wpe.weight.shape[0]
            
            print(f"Actual vocab size from weights: {actual_vocab_size}")
            print(f"Actual position embedding size from weights: {actual_pos_emb_size}")
            
            if actual_pos_emb_size < max_seq_len:
                print(f"❌ CRITICAL: Position embedding weights only support {actual_pos_emb_size} positions, but you're using {max_seq_len}")
                print(f"   This is likely the cause of the CUDA assertion error!")
                
        # Try a small test forward pass
        print(f"\n=== Test Forward Pass ===")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Test with a small sequence first
        test_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long).to(device)
        print(f"Testing with small sequence (length {test_input.shape[1]})...")
        
        try:
            with torch.no_grad():
                output = model(test_input)
            print("✅ Small sequence test passed")
        except Exception as e:
            print(f"❌ Small sequence test failed: {e}")
            return
        
        # Test with a sequence that might be too long
        if max_pos_emb < max_seq_len:
            problem_length = max_pos_emb + 1
            print(f"Testing with problematic sequence (length {problem_length})...")
            problem_input = torch.ones((1, problem_length), dtype=torch.long).to(device)
            
            try:
                with torch.no_grad():
                    output = model(problem_input)
                print(f"⚠️  Unexpectedly passed with length {problem_length}")
            except Exception as e:
                print(f"❌ Failed as expected with length {problem_length}: {e}")
                print("   This confirms the position embedding size is the issue!")
                
    except Exception as e:
        print(f"Error during model analysis: {e}")

if __name__ == "__main__":
    main() 
