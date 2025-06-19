#!/usr/bin/env python3
"""Debug CUDA crashes with long sequences"""

import torch
import json
from pathlib import Path
from transformers import GPT2Config, AutoModelForCausalLM
from config_utils import get_model_config, get_data_config, get_vocab_config
from train import load_vocabulary, custom_tokenizer, ByteStreamDataset, collate_fn
import traceback

def test_synthetic_data(model, vocab_size, max_seq_len):
    """Test model with synthetic data to isolate the issue"""
    print(f"\n=== Testing Synthetic Data (seq_len={max_seq_len}) ===")
    
    # Create synthetic data with valid token IDs
    batch_size = 1
    input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    
    print(f"Synthetic data shape: {input_ids.shape}")
    print(f"Token ID range: {input_ids.min().item()} to {input_ids.max().item()}")
    print(f"Vocab size: {vocab_size}")
    
    # Test on CPU first
    try:
        print("Testing on CPU...")
        model_cpu = model.cpu()
        outputs = model_cpu(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        print(f"✅ CPU test passed. Loss: {outputs.loss.item():.4f}")
    except Exception as e:
        print(f"❌ CPU test failed: {e}")
        return False
    
    # Test on GPU
    try:
        print("Testing on GPU...")
        model_gpu = model.cuda()
        input_ids_gpu = input_ids.cuda()
        attention_mask_gpu = attention_mask.cuda()
        
        outputs = model_gpu(input_ids=input_ids_gpu, attention_mask=attention_mask_gpu, labels=input_ids_gpu)
        print(f"✅ GPU test passed. Loss: {outputs.loss.item():.4f}")
        return True
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        print(f"Full error: {traceback.format_exc()}")
        return False

def analyze_real_data_tokens(data_dir, file_glob, vocab_to_int, max_files=5):
    """Analyze token IDs from real data files"""
    print(f"\n=== Analyzing Real Data Tokens ===")
    
    data_path = Path(data_dir)
    file_paths = sorted(data_path.glob(file_glob))[:max_files]
    
    all_token_ids = []
    vocab_size = max(vocab_to_int.values()) + 1
    
    for i, file_path in enumerate(file_paths):
        print(f"Analyzing file {i+1}/{len(file_paths)}: {file_path.name}")
        
        try:
            with file_path.open("rb") as f:
                raw_data = f.read()
            
            # Tokenize the file
            token_ids = custom_tokenizer(raw_data, vocab_to_int, vocab_size)
            all_token_ids.extend(token_ids)
            
            print(f"  File size: {len(raw_data)} bytes")
            print(f"  Tokens generated: {len(token_ids)}")
            if token_ids:
                print(f"  Token range: {min(token_ids)} to {max(token_ids)}")
                
                # Check for invalid tokens
                invalid_tokens = [t for t in token_ids if t >= vocab_size or t < 0]
                if invalid_tokens:
                    print(f"  ❌ INVALID TOKENS FOUND: {len(invalid_tokens)} tokens")
                    print(f"     Invalid token IDs: {list(set(invalid_tokens))[:10]}...")
                    return False, invalid_tokens
            
        except Exception as e:
            print(f"  ❌ Error processing file: {e}")
            return False, []
    
    if all_token_ids:
        print(f"\nOverall statistics:")
        print(f"  Total tokens: {len(all_token_ids)}")
        print(f"  Token ID range: {min(all_token_ids)} to {max(all_token_ids)}")
        print(f"  Vocab size: {vocab_size}")
        
        # Check for any invalid tokens
        invalid_tokens = [t for t in all_token_ids if t >= vocab_size or t < 0]
        if invalid_tokens:
            print(f"  ❌ INVALID TOKENS: {len(invalid_tokens)} out of {len(all_token_ids)}")
            return False, invalid_tokens
        else:
            print(f"  ✅ All tokens are valid")
            return True, []
    
    return True, []

def test_dataloader_with_different_lengths(data_dir, file_glob, vocab_to_int, pad_token_id, seq_lengths):
    """Test dataloader with different sequence lengths"""
    print(f"\n=== Testing DataLoader with Different Lengths ===")
    
    data_path = Path(data_dir)
    file_paths = sorted(data_path.glob(file_glob))[:5]  # Just test a few files
    vocab_size = max(vocab_to_int.values()) + 1
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        try:
            # Create dataset
            dataset = ByteStreamDataset(file_paths, vocab_to_int, pad_token_id, vocab_size)
            
            # Create collate function
            def limited_collate_fn(batch):
                return collate_fn(batch, seq_len, pad_token_id)
            
            # Create dataloader
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=limited_collate_fn)
            
            # Test a few batches
            for i, batch in enumerate(dataloader):
                if i >= 3:  # Test first 3 batches
                    break
                
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                
                print(f"  Batch {i+1}: shape={input_ids.shape}, range={input_ids.min().item()}-{input_ids.max().item()}")
                
                # Check for invalid token IDs
                invalid_mask = (input_ids >= vocab_size) | (input_ids < 0)
                if invalid_mask.any():
                    invalid_ids = input_ids[invalid_mask].unique().tolist()
                    print(f"    ❌ Invalid token IDs found: {invalid_ids}")
                    return False
                else:
                    print(f"    ✅ All token IDs valid")
            
        except Exception as e:
            print(f"  ❌ Error with seq_len {seq_len}: {e}")
            print(f"     Full error: {traceback.format_exc()}")
            return False
    
    return True

def test_model_with_real_data(model, data_dir, file_glob, vocab_to_int, pad_token_id, seq_len):
    """Test model with real data at specific sequence length"""
    print(f"\n=== Testing Model with Real Data (seq_len={seq_len}) ===")
    
    data_path = Path(data_dir)
    file_paths = sorted(data_path.glob(file_glob))[:2]  # Just test 2 files
    vocab_size = max(vocab_to_int.values()) + 1
    
    try:
        # Create dataset and dataloader
        dataset = ByteStreamDataset(file_paths, vocab_to_int, pad_token_id, vocab_size)
        
        def limited_collate_fn(batch):
            return collate_fn(batch, seq_len, pad_token_id)
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=limited_collate_fn)
        
        # Test model with real data
        model = model.cuda()
        model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 1:  # Just test first batch
                    break
                
                input_ids = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()
                
                print(f"Testing batch: {input_ids.shape}")
                print(f"Token range: {input_ids.min().item()}-{input_ids.max().item()}")
                print(f"Attention mask sum: {attention_mask.sum().item()}")
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                print(f"✅ Forward pass successful. Loss: {outputs.loss.item():.4f}")
                
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        print(f"Full error: {traceback.format_exc()}")
        return False

def debug_cuda_crash():
    """Main debugging function"""
    print("=== CUDA Crash Debug Session ===")
    
    # Load vocabulary and config
    vocab_to_int, int_to_vocab, pad_token_id = load_vocabulary()
    vocab_size = max(vocab_to_int.values()) + 1
    model_config = get_model_config()
    data_config = get_data_config()
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Model max positions: {model_config['n_positions']}")
    print(f"Pad token ID: {pad_token_id}")
    
    # Create a fresh model
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=model_config["n_positions"],
        n_embd=model_config["n_embd"],
        n_layer=model_config["n_layer"],
        n_head=model_config["n_head"],
    )
    model = AutoModelForCausalLM.from_config(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test sequence lengths that we know work vs don't work
    working_seq_len = 512  # We know this works
    failing_seq_len = 4096  # We know this fails
    
    # Step 1: Test with synthetic data
    print(f"\n{'='*60}")
    print("STEP 1: Testing with synthetic data")
    
    print("Testing known working length...")
    if not test_synthetic_data(model, vocab_size, working_seq_len):
        print("❌ Even synthetic data fails - model configuration issue!")
        return
    
    print("Testing known failing length...")
    if not test_synthetic_data(model, vocab_size, failing_seq_len):
        print("❌ Model fails with synthetic data - model size issue!")
        return
    
    print("✅ Synthetic data tests passed - issue is with real data!")
    
    # Step 2: Analyze real data tokens
    print(f"\n{'='*60}")
    print("STEP 2: Analyzing real data tokens")
    
    valid_data, invalid_tokens = analyze_real_data_tokens(
        data_config["default_data_dir"], 
        data_config["default_file_glob"], 
        vocab_to_int
    )
    
    if not valid_data:
        print("❌ Invalid tokens found in data!")
        print(f"Invalid token IDs: {invalid_tokens[:20]}...")
        return
    
    # Step 3: Test dataloader with different lengths
    print(f"\n{'='*60}")
    print("STEP 3: Testing dataloader")
    
    if not test_dataloader_with_different_lengths(
        data_config["default_data_dir"], 
        data_config["default_file_glob"], 
        vocab_to_int, 
        pad_token_id,
        [512, 1024, 2048, 4096]
    ):
        print("❌ Dataloader issues found!")
        return
    
    # Step 4: Test model with real data
    print(f"\n{'='*60}")
    print("STEP 4: Testing model with real data")
    
    print("Testing with working sequence length...")
    if not test_model_with_real_data(model, data_config["default_data_dir"], 
                                   data_config["default_file_glob"], 
                                   vocab_to_int, pad_token_id, working_seq_len):
        print("❌ Model fails with real data at working length!")
        return
    
    print("Testing with failing sequence length...")
    if not test_model_with_real_data(model, data_config["default_data_dir"], 
                                   data_config["default_file_glob"], 
                                   vocab_to_int, pad_token_id, failing_seq_len):
        print("❌ Model fails with real data at failing length!")
        return
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED! The issue might be training-specific.")
    print("Try running with CUDA_LAUNCH_BLOCKING=1 for better error messages.")

if __name__ == "__main__":
    debug_cuda_crash() 