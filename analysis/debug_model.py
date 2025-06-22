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
    
    # Look for model files in current directory (we're cd'd into the model directory)
    current_dir = Path(".")
    
    # Check for various model files
    model_files = list(current_dir.glob("*.safetensors")) + list(current_dir.glob("pytorch_model.bin"))
    config_file = current_dir / "config.json"
    
    if not config_file.exists():
        print(f"ERROR: config.json not found in {current_dir.resolve()}!")
        print("Available files:")
        for f in current_dir.iterdir():
            if f.is_file():
                print(f"  {f.name}")
        return
    
    if not model_files:
        print(f"ERROR: No model files (*.safetensors or pytorch_model.bin) found in {current_dir.resolve()}!")
        print("Available files:")
        for f in current_dir.iterdir():
            if f.is_file():
                print(f"  {f.name}")
        return
    
    try:
        # Load model and check all configuration
        print(f"Loading model from: {current_dir.resolve()}")
        model = AutoModelForCausalLM.from_pretrained(current_dir)
        config = model.config
        
        print(f"Model type: {config.model_type}")
        print(f"Vocabulary size: {config.vocab_size}")
        print(f"Max position embeddings: {config.n_positions if hasattr(config, 'n_positions') else config.max_position_embeddings}")
        print(f"Hidden size: {config.n_embd if hasattr(config, 'n_embd') else config.hidden_size}")
        print(f"Number of layers: {config.n_layer if hasattr(config, 'n_layer') else config.num_layers}")
        print(f"Number of heads: {config.n_head if hasattr(config, 'n_head') else config.num_heads}")
        
        # Check if position embeddings match what we're trying to use
        max_seq_len = 2048  # Based on your model name: seq-2k
        max_pos_emb = config.n_positions if hasattr(config, 'n_positions') else config.max_position_embeddings
        
        print(f"\n=== Position Embedding Check ===")
        print(f"Detected max sequence length from model name: {max_seq_len}")
        print(f"Model's max position embeddings: {max_pos_emb}")
        
        if max_seq_len > max_pos_emb:
            print(f"❌ PROBLEM: Detected sequence length ({max_seq_len}) > Model's max position embeddings ({max_pos_emb})")
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
                print(f"❌ CRITICAL: Position embedding weights only support {actual_pos_emb_size} positions, but model name suggests {max_seq_len}")
                print(f"   This could cause CUDA assertion errors!")
                
        # Check vocabulary compatibility if vocab files exist
        vocab_to_int_file = current_dir / "vocab_to_int.json"
        if vocab_to_int_file.exists():
            print(f"\n=== Vocabulary Compatibility Check ===")
            with open(vocab_to_int_file, 'r') as f:
                vocab_to_int = json.load(f)
            
            calculated_vocab_size = max(vocab_to_int.values()) + 1
            print(f"Vocabulary file vocab size: {calculated_vocab_size}")
            print(f"Model config vocab size: {config.vocab_size}")
            
            if calculated_vocab_size != config.vocab_size:
                print(f"❌ VOCAB MISMATCH: File vocab size ({calculated_vocab_size}) != Model vocab size ({config.vocab_size})")
                print("   This will cause CUDA indexing errors!")
            else:
                print("✅ Vocabulary sizes match")
        
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
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
