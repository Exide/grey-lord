#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

from tokenizer import GreyLordTokenizer


def test_english_tokenization():
    """Test that English text is tokenized efficiently"""
    print("=== Testing English Tokenization ===")
    
    tokenizer = GreyLordTokenizer()
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Test English sentences
    test_sentences = [
        "Hello world!",
        "The kobold thief lunges at you with their shortsword, but you dodge!",
        "The kobold thief stabs you for 5 damage!",
        "You gain 13 experience."
    ]
    
    for sentence in test_sentences:
        print(f"\nTesting: '{sentence}'")
        tokens = tokenizer.encode(sentence)
        print(f"Tokens: {tokens} (length: {len(tokens)})")
        
        # Test round-trip
        decoded = tokenizer.decode(tokens)
        decoded_str = decoded.decode('utf-8')
        print(f"Decoded: '{decoded_str}'")
        
        assert decoded_str == sentence, f"Round-trip failed: '{sentence}' != '{decoded_str}'"
        print("✓ Round-trip successful")
    
    print("✓ English tokenization tests passed")


def test_special_token_handling():
    """Test that special ANSI and Telnet tokens are handled correctly"""
    print("\n=== Testing Special Token Handling ===")
    
    tokenizer = GreyLordTokenizer()
    
    # Test cases with special tokens
    test_cases = [
        "Hello <|ANSI#BOLD|>World<|ANSI#RESET|>!",
        "<|TELNET#WILL_BINARY|>Login data<|TELNET#DO_ECHO|>",
        "You see a <|ANSI#FG_RED|>red dragon<|ANSI#RESET|> approaching!",
        "Status: <|ANSI#FG_GREEN|>Healthy<|ANSI#RESET|> | Level: 15"
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: '{test_case}'")
        tokens = tokenizer.encode(test_case)
        print(f"Tokens: {tokens} (length: {len(tokens)})")
        
        # Test round-trip
        decoded = tokenizer.decode(tokens)
        decoded_str = decoded.decode('utf-8')
        print(f"Decoded: '{decoded_str}'")
        
        assert decoded_str == test_case, f"Round-trip failed: '{test_case}' != '{decoded_str}'"
        print("✓ Round-trip successful")
    
    print("✓ Special token handling tests passed")


def test_mixed_content():
    """Test mixed content with English and special tokens"""
    print("\n=== Testing Mixed Content ===")
    
    tokenizer = GreyLordTokenizer()
    
    # Realistic game scenarios
    test_scenarios = [
        "[HP=100/100]: <|ANSI#FG_GREEN|>You are in perfect health.<|ANSI#RESET|>",
        "<|ANSI#BOLD|>Combat:<|ANSI#RESET|> The orc warrior attacks you with a rusty sword!",
        "You gained <|ANSI#FG_YELLOW|>250 gold pieces<|ANSI#RESET|> from the treasure chest!"
    ]
    
    for scenario in test_scenarios:
        print(f"\nTesting: '{scenario}'")
        tokens = tokenizer.encode(scenario)
        print(f"Tokens: {tokens} (length: {len(tokens)})")
        
        # Test round-trip
        decoded = tokenizer.decode(tokens)
        decoded_str = decoded.decode('utf-8')
        print(f"Decoded: '{decoded_str}'")
        
        assert decoded_str == scenario, f"Round-trip failed: '{scenario}' != '{decoded_str}'"
        print("✓ Round-trip successful")
    
    print("✓ Mixed content tests passed")


def test_basic_properties():
    """Test basic tokenizer properties"""
    print("\n=== Testing Basic Properties ===")
    
    tokenizer = GreyLordTokenizer()
    
    # Test vocabulary size
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    assert vocab_size > 50000, f"Vocab size should be > 50000 (GPT-2 + special tokens), got {vocab_size}"
    
    # Test token ranges
    ranges = tokenizer.get_token_ranges()
    print(f"Token ranges: {ranges}")
    assert 'gpt2' in ranges, "Should have GPT-2 token range"
    assert 'ansi' in ranges, "Should have ANSI token range"
    assert 'telnet' in ranges, "Should have Telnet token range"
    
    # Test special token IDs
    special_ids = tokenizer.get_special_token_ids()
    print(f"Special token IDs: {special_ids}")
    assert 'pad' in special_ids, "Should have PAD token"
    
    # Test PAD token
    pad_id = tokenizer.pad_token_id
    print(f"PAD token ID: {pad_id}")
    assert pad_id > 50000, f"PAD token should be > 50000, got {pad_id}"
    
    print("✓ Basic properties tests passed")


def test_batch_operations():
    """Test batch encoding/decoding"""
    print("\n=== Testing Batch Operations ===")
    
    tokenizer = GreyLordTokenizer()
    
    # Test batch encoding
    test_data_list = [
        "Hello World",
        "<|ANSI#RESET|>Test",
        "You attack the goblin",
        "<|ANSI#FG_RED|>Warning!<|ANSI#RESET|>"
    ]
    
    print(f"Batch test data: {test_data_list}")
    
    # Test with padding
    encoded_batch = tokenizer.batch_encode(test_data_list, max_length=20)
    print(f"Encoded batch lengths: {[len(seq) for seq in encoded_batch]}")
    
    # All sequences should be padded to length 20
    for seq in encoded_batch:
        assert len(seq) == 20, f"Sequence should be padded to 20, got {len(seq)}"
    
    # Test batch decoding
    decoded_batch = tokenizer.batch_decode(encoded_batch)
    print(f"Decoded batch: {[data.decode('utf-8') for data in decoded_batch]}")
    
    print("✓ Batch operations tests passed")


def main():
    """Run all tests"""
    print("GREYLORD TOKENIZER TESTS")
    print("=" * 50)
    
    try:
        test_english_tokenization()
        test_special_token_handling()
        test_mixed_content()
        test_basic_properties()
        test_batch_operations()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("The GreyLord tokenizer is working correctly.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 