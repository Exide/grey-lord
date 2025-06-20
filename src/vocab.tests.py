"""vocab.tests.py

Unit tests for the vocab module.
Co-located with the implementation for better organization.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, mock_open

from vocab import load_json, get_vocab_size, validate_vocabulary_compatibility


class TestVocabModule:
    """Test cases for vocab module functions."""
    
    def test_load_json_success(self):
        """Test successful JSON loading."""
        mock_data = {"token1": 0, "token2": 1}
        mock_file = mock_open(read_data='{"token1": 0, "token2": 1}')
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.open", mock_file):
            result = load_json(Path("test.json"))
            assert result == mock_data
    
    def test_load_json_file_not_found(self):
        """Test FileNotFoundError when file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Required vocabulary file not found"):
                load_json(Path("missing.json"))
    
    def test_get_vocab_size(self):
        """Test vocabulary size calculation."""
        vocab = {"token1": 0, "token2": 5, "token3": 10}
        size = get_vocab_size(vocab)
        assert size == 11  # max(values) + 1
    
    def test_validate_vocabulary_compatibility_success(self):
        """Test successful vocabulary validation."""
        # Should not raise any exception
        validate_vocabulary_compatibility(100, 150)  # current < model
    
    def test_validate_vocabulary_compatibility_size_mismatch(self):
        """Test vocabulary size mismatch error."""
        with pytest.raises(ValueError, match="Vocabulary size mismatch"):
            validate_vocabulary_compatibility(150, 100)  # current > model


# Example of how to run these tests:
# pytest src/vocab.tests.py -v 