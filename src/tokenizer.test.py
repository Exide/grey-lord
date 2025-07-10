#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

from tokenizer import GreyLordTokenizer


def test_token_ranges():
    """Test that token ranges are correctly organized"""
    print("=== Testing Token Ranges ===")
    
    tokenizer = GreyLordTokenizer()
    ranges = tokenizer.get_token_ranges()
    print(f"Token ranges: {ranges}")
    
    # Verify BYTE_0 through BYTE_255 are token IDs 0-255
    for i in range(256):
        byte_token = f'<|BYTE_{i}|>'
        token_id = tokenizer.get_token_id(byte_token)
        assert token_id == i, f"BYTE_{i} should have token ID {i}, got {token_id}"
    print("✓ Byte tokens correctly mapped to IDs 0-255")
    
    # Verify Telnet tokens start at 300
    telnet_tokens = ['<|TELNET#UNKNOWN|>', '<|TELNET#WILL_BINARY|>', '<|TELNET#DO_ECHO|>']
    for token in telnet_tokens:
        token_id = tokenizer.get_token_id(token)
        assert token_id >= 300, f"Telnet token {token} should have ID >= 300, got {token_id}"
    print("✓ Telnet tokens start at 300")
    
    # Verify ANSI tokens start at 400
    ansi_tokens = ['<|ANSI#RESET|>', '<|ANSI#BOLD|>', '<|ANSI#FG_RED|>']
    for token in ansi_tokens:
        token_id = tokenizer.get_token_id(token)
        assert token_id >= 400, f"ANSI token {token} should have ID >= 400, got {token_id}"
    print("✓ ANSI tokens start at 400")
    
    # Verify special tokens start at 500
    special_tokens = ['<|PAD|>']
    for token in special_tokens:
        token_id = tokenizer.get_token_id(token)
        assert token_id >= 500, f"Special token {token} should have ID >= 500, got {token_id}"
    print("✓ Special tokens start at 500")


def test_basic_functionality():
    """Test basic encode/decode functionality"""
    print("\n=== Testing Basic Functionality ===")
    
    tokenizer = GreyLordTokenizer()
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Test raw bytes
    test_data = b"Hello World!"
    print(f"Original data: {test_data}")
    
    encoded = tokenizer.encode(test_data)
    print(f"Encoded: {encoded}")
    
    # Verify that regular bytes are encoded as their byte values (0-255)
    for i, byte_val in enumerate(test_data):
        assert encoded[i] == byte_val, f"Byte {byte_val} should encode to {byte_val}, got {encoded[i]}"
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    assert test_data == decoded, f"Round-trip failed: {test_data} != {decoded}"
    print("✓ Basic round-trip test passed")
    

def test_byte_token_mapping():
    """Test that byte tokens map directly to byte values"""
    print("\n=== Testing Byte Token Mapping ===")
    
    tokenizer = GreyLordTokenizer()
    
    # Test that get_byte_token_id works correctly
    for i in range(256):
        token_id = tokenizer.get_byte_token_id(i)
        assert token_id == i, f"get_byte_token_id({i}) should return {i}, got {token_id}"
    
    # Test that encoding single bytes gives their byte value as token ID
    test_bytes = [0, 1, 65, 127, 128, 255]  # A, DEL, high bit set, max
    for byte_val in test_bytes:
        single_byte = bytes([byte_val])
        encoded = tokenizer.encode(single_byte)
        assert len(encoded) == 1, f"Single byte should encode to single token"
        assert encoded[0] == byte_val, f"Byte {byte_val} should encode to token {byte_val}, got {encoded[0]}"
    
    print("✓ Byte token mapping test passed")


def test_ansi_tokens():
    """Test ANSI token handling"""
    print("\n=== Testing ANSI Tokens ===")
    
    tokenizer = GreyLordTokenizer()
    
    # Test ANSI tokens
    ansi_tokens = ['<|ANSI#RESET|>', '<|ANSI#BOLD|>', '<|ANSI#FG_RED|>']
    for token in ansi_tokens:
        token_id = tokenizer.get_token_id(token)
        print(f"Token '{token}' -> ID: {token_id}")
        assert token_id is not None, f"ANSI token {token} not found"
        assert token_id >= 400, f"ANSI token {token} should have ID >= 400"
        
        # Test round-trip
        recovered_token = tokenizer.get_token(token_id)
        assert recovered_token == token, f"Token recovery failed: {token} != {recovered_token}"
        
    print("✓ ANSI token tests passed")


def test_telnet_tokens():
    """Test Telnet token handling"""
    print("\n=== Testing Telnet Tokens ===")
    
    tokenizer = GreyLordTokenizer()
    
    # Test Telnet tokens
    telnet_tokens = ['<|TELNET#WILL_BINARY|>', '<|TELNET#DO_ECHO|>', '<|TELNET#UNKNOWN|>']
    for token in telnet_tokens:
        token_id = tokenizer.get_token_id(token)
        print(f"Token '{token}' -> ID: {token_id}")
        assert token_id is not None, f"Telnet token {token} not found"
        assert token_id >= 300, f"Telnet token {token} should have ID >= 300"
        
        # Test round-trip
        recovered_token = tokenizer.get_token(token_id)
        assert recovered_token == token, f"Token recovery failed: {token} != {recovered_token}"
        
    print("✓ Telnet token tests passed")


def test_mixed_content():
    """Test mixed content with bytes and special tokens"""
    print("\n=== Testing Mixed Content ===")
    
    tokenizer = GreyLordTokenizer()
    
    # Create test data that includes both special tokens and regular bytes
    test_data = b"Hello<|ANSI#BOLD|>World<|TELNET#DO_ECHO|>"
    print(f"Mixed test data: {test_data}")
    
    encoded = tokenizer.encode(test_data)
    print(f"Encoded: {encoded}")
    
    # Check that we have the right mix of byte tokens and special tokens
    assert any(token_id >= 400 for token_id in encoded), "Should have ANSI tokens"
    assert any(token_id >= 300 for token_id in encoded), "Should have Telnet tokens"
    assert any(token_id <= 255 for token_id in encoded), "Should have byte tokens"
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    assert test_data == decoded, f"Mixed content round-trip failed: {test_data} != {decoded}"
    print("✓ Mixed content test passed")


def test_special_tokens():
    """Test special control tokens"""
    print("\n=== Testing Special Tokens ===")
    
    tokenizer = GreyLordTokenizer()
    
    special_token_ids = tokenizer.get_special_token_ids()
    print(f"Special token IDs: {special_token_ids}")
    
    # Should only have PAD token now
    assert len(special_token_ids) == 1, f"Should only have 1 special token, got {len(special_token_ids)}"
    assert 'pad' in special_token_ids, "Should have PAD token"
    
    for name, token_id in special_token_ids.items():
        assert token_id >= 500, f"Special token '{name}' should have ID >= 500, got {token_id}"
        token = tokenizer.get_token(token_id)
        print(f"Special token '{name}' -> ID: {token_id} -> Token: {token}")
        assert token is not None, f"Special token ID {token_id} not found"
    
    print("✓ Special token tests passed")


def test_batch_operations():
    """Test batch encoding/decoding"""
    print("\n=== Testing Batch Operations ===")
    
    tokenizer = GreyLordTokenizer()
    
    # Test batch encoding
    test_data_list = [
        b"Hello World",
        b"<|ANSI#RESET|>",
        b"Test\x00\x01\x02",
        b"<|TELNET#WILL_BINARY|>Mixed"
    ]
    
    print(f"Batch test data: {test_data_list}")
    
    # Test with padding
    encoded_batch = tokenizer.batch_encode(test_data_list, max_length=20)
    print(f"Encoded batch (padded to 20): {[len(seq) for seq in encoded_batch]}")
    
    # All sequences should be padded to length 20
    for seq in encoded_batch:
        assert len(seq) == 20, f"Sequence should be padded to 20, got {len(seq)}"
    
    decoded_batch = tokenizer.batch_decode(encoded_batch)
    print(f"Decoded batch length: {len(decoded_batch)}")
    
    print("✓ Batch operations test passed")


if __name__ == "__main__":
    try:
        test_token_ranges()
        test_basic_functionality()
        test_byte_token_mapping()
        test_ansi_tokens()
        test_telnet_tokens()
        test_mixed_content()
        test_special_tokens()
        test_batch_operations()
        
        print("\n🎉 All tests passed! Reorganized tokenizer is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 