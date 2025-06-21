"""dataset.tests.py

Unit tests for the dataset module.
Co-located with the implementation for better organization.
"""

import unittest
import torch
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock

from dataset import (
    ByteStreamDataset,
    collate_fn,
    create_collate_fn,
    calculate_data_size,
    split_files
)


class TestByteStreamDataset(unittest.TestCase):
    """Test cases for ByteStreamDataset class."""
    
    def _create_mock_file_read(self, file_content=b"test data", should_raise=None):
        """Helper to create mock file reading setup."""
        if should_raise:
            return Mock(side_effect=should_raise)
        
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = file_content
        return Mock(return_value=mock_file)
    
    def _assert_print_contains(self, mock_print, expected_text):
        """Helper to check if print was called with text containing expected_text."""
        call_args = [str(call[0][0]) for call in mock_print.call_args_list]
        self.assertTrue(any(expected_text in arg for arg in call_args))
    
    @patch("builtins.print")
    def test_initialization(self, mock_print):
        """Test dataset initialization."""
        paths = [Path("file1.bin"), Path("file2.bin")]
        vocab = {"byte_65": 0, "byte_66": 1}
        pad_token_id = 999
        
        dataset = ByteStreamDataset(paths, vocab, pad_token_id, label="test")
        
        self.assertEqual(dataset.paths, paths)
        self.assertEqual(dataset.vocab, vocab)
        self.assertEqual(dataset.pad_token_id, pad_token_id)
        mock_print.assert_called_once_with("[info] Dataset (test) initialized with 2 files.")
    
    @patch("builtins.print")
    @patch("dataset.custom_tokenizer")
    @patch("pathlib.Path.open")
    def test_getitem_success(self, mock_open, mock_tokenizer, mock_print):
        """Test successful item retrieval."""
        # Setup file read mock
        mock_file = mock_open.return_value.__enter__.return_value
        mock_file.read.return_value = b"test data"
        
        mock_tokenizer.return_value = [1, 2, 3]
        
        dataset = ByteStreamDataset([Path("test.bin")], {}, 0)
        result = dataset[0]
        
        self.assertTrue(torch.equal(result, torch.tensor([1, 2, 3], dtype=torch.long)))
        mock_tokenizer.assert_called_once_with(b"test data", {}, None)
    
    @patch("builtins.print")
    @patch("dataset.custom_tokenizer")
    @patch("pathlib.Path.open")
    def test_getitem_file_read_error(self, mock_open, mock_tokenizer, mock_print):
        """Test item retrieval with file read error."""
        # Setup file read to raise error
        mock_open.side_effect = IOError("File not found")
        
        dataset = ByteStreamDataset([Path("missing.bin")], {}, 999)
        result = dataset[0]
        
        # Should return empty tensor on error
        self.assertTrue(torch.equal(result, torch.tensor([], dtype=torch.long)))
        # Check that error was printed (at least 2 calls: init + error)
        self.assertGreaterEqual(mock_print.call_count, 2)
        self._assert_print_contains(mock_print, "Failed to read file")
    
    @patch("builtins.print")
    @patch("dataset.custom_tokenizer")
    @patch("pathlib.Path.open")
    def test_getitem_no_tokens(self, mock_open, mock_tokenizer, mock_print):
        """Test item retrieval when no tokens are generated."""
        # Setup file read mock
        mock_file = mock_open.return_value.__enter__.return_value
        mock_file.read.return_value = b"test data"
        
        mock_tokenizer.return_value = []  # Empty token list
        
        dataset = ByteStreamDataset([Path("test.bin")], {}, 999)
        result = dataset[0]
        
        # Should return pad token when no tokens found
        self.assertTrue(torch.equal(result, torch.tensor([999], dtype=torch.long)))
        # Check that warning was printed
        self.assertGreaterEqual(mock_print.call_count, 2)
        self._assert_print_contains(mock_print, "No tokens found")
    
    def test_len(self):
        """Test dataset length."""
        paths = [Path("file1.bin"), Path("file2.bin"), Path("file3.bin")]
        dataset = ByteStreamDataset(paths, {}, 0)
        self.assertEqual(len(dataset), 3)


class TestCollateFn(unittest.TestCase):
    """Test cases for collate_fn functionality."""
    
    def test_collate_fn_normal_case(self):
        """Test collation with normal sequences."""
        batch = [
            torch.tensor([1, 2, 3], dtype=torch.long),
            torch.tensor([4, 5], dtype=torch.long),
            torch.tensor([6, 7, 8, 9], dtype=torch.long)
        ]
        
        result = collate_fn(batch, max_seq_len=10, pad_token_id=999)
        
        # Check shapes
        self.assertEqual(result["input_ids"].shape, (3, 4))  # batch_size=3, max_len=4
        self.assertEqual(result["attention_mask"].shape, (3, 4))
        
        # Check padding
        expected_input_ids = torch.tensor([
            [1, 2, 3, 999],
            [4, 5, 999, 999],
            [6, 7, 8, 9]
        ], dtype=torch.long)
        
        expected_attention_mask = torch.tensor([
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1]
        ], dtype=torch.long)
        
        self.assertTrue(torch.equal(result["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(result["attention_mask"], expected_attention_mask))
    
    def test_collate_fn_truncation(self):
        """Test collation with sequence truncation."""
        batch = [
            torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long),  # Will be truncated
            torch.tensor([7, 8], dtype=torch.long)
        ]
        
        result = collate_fn(batch, max_seq_len=4, pad_token_id=999)
        
        # Check shapes
        self.assertEqual(result["input_ids"].shape, (2, 4))
        
        # Check truncation
        expected_input_ids = torch.tensor([
            [1, 2, 3, 4],  # Truncated to max_seq_len
            [7, 8, 999, 999]  # Padded
        ], dtype=torch.long)
        
        self.assertTrue(torch.equal(result["input_ids"], expected_input_ids))
    
    def test_collate_fn_empty_sequences(self):
        """Test collation with empty sequences."""
        batch = [
            torch.tensor([], dtype=torch.long),
            torch.tensor([1, 2], dtype=torch.long),
            torch.tensor([], dtype=torch.long)
        ]
        
        result = collate_fn(batch, max_seq_len=10, pad_token_id=999)
        
        # Should filter out empty sequences
        self.assertEqual(result["input_ids"].shape, (1, 2))  # Only one non-empty sequence
        
        expected_input_ids = torch.tensor([[1, 2]], dtype=torch.long)
        expected_attention_mask = torch.tensor([[1, 1]], dtype=torch.long)
        
        self.assertTrue(torch.equal(result["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(result["attention_mask"], expected_attention_mask))
    
    def test_collate_fn_all_empty_sequences(self):
        """Test collation when all sequences are empty."""
        batch = [
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.long)
        ]
        
        result = collate_fn(batch, max_seq_len=10, pad_token_id=999)
        
        # Should return minimal batch
        expected_input_ids = torch.tensor([[999]], dtype=torch.long)
        expected_attention_mask = torch.tensor([[0]], dtype=torch.long)
        
        self.assertTrue(torch.equal(result["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(result["attention_mask"], expected_attention_mask))
    
    def test_collate_fn_single_sequence(self):
        """Test collation with single sequence."""
        batch = [torch.tensor([1, 2, 3], dtype=torch.long)]
        
        result = collate_fn(batch, max_seq_len=5, pad_token_id=999)
        
        expected_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        expected_attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
        
        self.assertTrue(torch.equal(result["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(result["attention_mask"], expected_attention_mask))


class TestCreateCollateFn(unittest.TestCase):
    """Test cases for create_collate_fn factory function."""
    
    def test_create_collate_fn(self):
        """Test collate function factory."""
        collate_func = create_collate_fn(max_seq_len=8, pad_token_id=777)
        
        # Test that it works like regular collate_fn
        batch = [
            torch.tensor([1, 2], dtype=torch.long),
            torch.tensor([3, 4, 5], dtype=torch.long)
        ]
        
        result = collate_func(batch)
        
        self.assertEqual(result["input_ids"].shape, (2, 3))
        self.assertEqual(result["input_ids"][0, 2], 777)  # Should use the specified pad token


class TestCalculateDataSize(unittest.TestCase):
    """Test cases for calculate_data_size function."""
    
    def test_calculate_data_size_success(self):
        """Test successful data size calculation."""
        # Mock file paths with different sizes
        mock_path1 = Mock(spec=Path)
        mock_path1.stat.return_value.st_size = 1024  # 1KB
        
        mock_path2 = Mock(spec=Path)
        mock_path2.stat.return_value.st_size = 2048  # 2KB
        
        paths = [mock_path1, mock_path2]
        
        size_bytes, size_str = calculate_data_size(paths)
        
        self.assertEqual(size_bytes, 3072)  # 1024 + 2048
        self.assertEqual(size_str, "3.0KB")
    
    def test_calculate_data_size_large_files(self):
        """Test data size calculation with large files."""
        mock_path = Mock(spec=Path)
        mock_path.stat.return_value.st_size = 2 * 1024**3  # 2GB
        
        size_bytes, size_str = calculate_data_size([mock_path])
        
        self.assertEqual(size_bytes, 2 * 1024**3)
        self.assertEqual(size_str, "2.0GB")
    
    def test_calculate_data_size_mb_files(self):
        """Test data size calculation with MB files."""
        mock_path = Mock(spec=Path)
        mock_path.stat.return_value.st_size = 10 * 1024**2  # 10MB
        
        size_bytes, size_str = calculate_data_size([mock_path])
        
        self.assertEqual(size_bytes, 10 * 1024**2)
        self.assertEqual(size_str, "10.0MB")
    
    def test_calculate_data_size_bytes(self):
        """Test data size calculation with small files (bytes)."""
        mock_path = Mock(spec=Path)
        mock_path.stat.return_value.st_size = 512  # 512 bytes
        
        size_bytes, size_str = calculate_data_size([mock_path])
        
        self.assertEqual(size_bytes, 512)
        self.assertEqual(size_str, "512.0B")
    
    @patch("builtins.print")
    def test_calculate_data_size_os_error(self, mock_print):
        """Test data size calculation with OS error."""
        mock_path = Mock(spec=Path)
        mock_path.stat.side_effect = OSError("Permission denied")
        
        size_bytes, size_str = calculate_data_size([mock_path])
        
        # Should handle error gracefully and return 0
        self.assertEqual(size_bytes, 0)
        self.assertEqual(size_str, "0.0B")
        mock_print.assert_called_once()
        self.assertIn("Could not get size", str(mock_print.call_args[0][0]))


class TestSplitFiles(unittest.TestCase):
    """Test cases for split_files function."""
    
    def test_split_files_normal_split(self):
        """Test normal file splitting."""
        files = [Path(f"file{i}.txt") for i in range(10)]
        
        train_files, val_files = split_files(files, validation_split=0.2)
        
        # Should have 2 validation files (20% of 10)
        self.assertEqual(len(val_files), 2)
        self.assertEqual(len(train_files), 8)
        
        # Validation files should be first files
        self.assertEqual(val_files, files[:2])
        self.assertEqual(train_files, files[2:])
    
    def test_split_files_small_dataset(self):
        """Test file splitting with small dataset."""
        files = [Path("file1.txt"), Path("file2.txt")]
        
        train_files, val_files = split_files(files, validation_split=0.5)
        
        # Should ensure at least 1 validation file
        self.assertEqual(len(val_files), 1)
        self.assertEqual(len(train_files), 1)
    
    def test_split_files_single_file(self):
        """Test file splitting with single file."""
        files = [Path("file1.txt")]
        
        train_files, val_files = split_files(files, validation_split=0.3)
        
        # Should ensure at least 1 validation file
        self.assertEqual(len(val_files), 1)
        self.assertEqual(len(train_files), 0)
    
    def test_split_files_zero_validation(self):
        """Test file splitting with zero validation split."""
        files = [Path(f"file{i}.txt") for i in range(5)]
        
        train_files, val_files = split_files(files, validation_split=0.0)
        
        # Should still ensure at least 1 validation file
        self.assertEqual(len(val_files), 1)  # max(1, int(5 * 0.0)) = 1
        self.assertEqual(len(train_files), 4)
    
    def test_split_files_large_validation(self):
        """Test file splitting with large validation split."""
        files = [Path(f"file{i}.txt") for i in range(10)]
        
        train_files, val_files = split_files(files, validation_split=0.8)
        
        # Should have 8 validation files (80% of 10)
        self.assertEqual(len(val_files), 8)
        self.assertEqual(len(train_files), 2)
    
    def test_split_files_empty_list(self):
        """Test file splitting with empty file list."""
        files = []
        
        train_files, val_files = split_files(files, validation_split=0.2)
        
        # Both should be empty
        self.assertEqual(len(val_files), 0)
        self.assertEqual(len(train_files), 0)


class TestDatasetIntegration(unittest.TestCase):
    """Integration tests for dataset functionality."""
    
    @patch("dataset.custom_tokenizer")
    @patch("pathlib.Path.open")
    def test_dataset_with_pytorch_dataloader(self, mock_open, mock_tokenizer):
        """Test dataset integration with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        # Mock file reading
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = b"test"
        mock_open.return_value = mock_file
        
        # Mock tokenizer
        mock_tokenizer.side_effect = [[1, 2], [3, 4, 5]]
        
        # Create dataset
        paths = [Path("file1.bin"), Path("file2.bin")]
        dataset = ByteStreamDataset(paths, {}, pad_token_id=999)
        
        # Create dataloader with collate function
        collate_func = create_collate_fn(max_seq_len=5, pad_token_id=999)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_func)
        
        # Test iteration
        batch = next(iter(dataloader))
        
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertEqual(batch["input_ids"].shape[0], 2)  # batch size
        self.assertEqual(batch["input_ids"].dtype, torch.long)
        self.assertEqual(batch["attention_mask"].dtype, torch.long)


# Example of how to run these tests:
# pytest training/dataset.tests.py -v 

if __name__ == "__main__":
    unittest.main()
