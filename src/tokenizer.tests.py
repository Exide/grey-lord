"""tokenizer.tests.py

Unit tests for the tokenizer module.
Co-located with the implementation for better organization.
"""

import pytest
from tokenizer import _bin_delay_token, custom_tokenizer


class TestTokenizerModule:
    """Test cases for tokenizer module functions."""
    
    def test_bin_delay_token_short(self):
        """Test delay token binning for short delays."""
        result = _bin_delay_token("<|delay#0.5|>")
        assert result == "<|delay_short|>"
    
    def test_bin_delay_token_medium(self):
        """Test delay token binning for medium delays."""
        result = _bin_delay_token("<|delay#3.0|>")
        assert result == "<|delay_medium|>"
    
    def test_bin_delay_token_long(self):
        """Test delay token binning for long delays."""
        result = _bin_delay_token("<|delay#10.0|>")
        assert result == "<|delay_long|>"
    
    def test_custom_tokenizer_special_tokens(self):
        """Test tokenization of special tokens."""
        vocab = {
            "<|client|>": 100,
            "<|delay_short|>": 101,
            "byte_65": 102  # 'A'
        }
        
        # Test data with special token and byte
        test_data = b"<|client|>A"
        result = custom_tokenizer(test_data, vocab)
        
        expected = [100, 102]  # <|client|> and byte_65 (A)
        assert result == expected
    
    def test_custom_tokenizer_delay_binning(self):
        """Test that delay tokens are properly binned."""
        vocab = {
            "<|delay_short|>": 200,
            "byte_65": 102
        }
        
        # Original delay token should be binned to delay_short
        test_data = b"<|delay#0.8|>A"
        result = custom_tokenizer(test_data, vocab)
        
        expected = [200, 102]  # binned delay + byte_65
        assert result == expected
    
    def test_custom_tokenizer_unknown_tokens(self):
        """Test handling of unknown tokens."""
        vocab = {"byte_65": 102}
        
        # Unknown special token should be skipped
        test_data = b"<|unknown|>A"
        result = custom_tokenizer(test_data, vocab)
        
        expected = [102]  # Only byte_65, unknown token skipped
        assert result == expected
    
    def test_custom_tokenizer_max_token_id_filter(self):
        """Test max_token_id filtering."""
        vocab = {
            "byte_65": 102,
            "byte_66": 999  # This exceeds max_token_id
        }
        
        test_data = b"AB"
        result = custom_tokenizer(test_data, vocab, max_token_id=500)
        
        expected = [102]  # Only byte_65, byte_66 filtered out
        assert result == expected


# Example of how to run these tests:
# pytest src/tokenizer.tests.py -v 