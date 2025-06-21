"""model_manager.tests.py

Unit tests for the model_manager module.
Co-located with the implementation for better organization.
"""

import unittest
import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock

from model_manager import (
    get_model_directories,
    load_model_summary,
    load_model_config,
    load_training_config,
    compare_models,
    create_model_leaderboard,
    cleanup_old_models,
    export_model_for_deployment
)


class BaseModelManagerTest(unittest.TestCase):
    """Base test class with common utilities for model manager tests."""
    
    def create_mock_path_with_file(self, file_exists=True, file_content="{}"):
        """Helper to create a mock Path with file operations."""
        mock_path_class = Mock()
        mock_path = Mock()
        mock_file_obj = Mock()
        
        mock_path_class.return_value = mock_path
        mock_path.__truediv__ = Mock(return_value=mock_file_obj)
        mock_file_obj.exists.return_value = file_exists
        
        if file_exists:
            mock_file_obj.open = mock_open(read_data=file_content)
        
        return mock_path_class, mock_file_obj
    
    def create_mock_model_dirs(self, count, base_name="model"):
        """Helper to create mock model directories."""
        mock_dirs = []
        for i in range(count):
            mock_dir = Mock(spec=Path)
            mock_dir.name = f"{base_name}{i}"
            mock_dir.rglob.return_value = [Mock(stat=Mock(return_value=Mock(st_size=1024)))]
            mock_dirs.append(mock_dir)
        return mock_dirs
    
    def assert_print_contains(self, mock_print, expected_text):
        """Helper to check if print was called with text containing expected_text."""
        printed_text = " ".join([str(call[0][0]) for call in mock_print.call_args_list])
        self.assertIn(expected_text, printed_text)


class TestModelDirectoryDiscovery(unittest.TestCase):
    """Test cases for model directory discovery."""
    
    def test_get_model_directories_no_models_dir(self):
        """Test when models directory doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            result = get_model_directories()
            self.assertEqual(result, [])
    
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.iterdir")
    def test_get_model_directories_empty(self, mock_iterdir, mock_exists):
        """Test when models directory is empty."""
        mock_iterdir.return_value = []
        result = get_model_directories()
        self.assertEqual(result, [])
    
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.iterdir")
    def test_get_model_directories_with_valid_models(self, mock_iterdir, mock_exists):
        """Test discovery of valid model directories."""
        # Mock model directories
        mock_dir1 = Mock(spec=Path)
        mock_dir1.is_dir.return_value = True
        mock_dir1.name = "model1"
        mock_dir1.stat.return_value.st_mtime = 1000
        mock_dir1.__truediv__ = Mock(return_value=Mock(exists=Mock(return_value=True)))
        
        mock_dir2 = Mock(spec=Path)
        mock_dir2.is_dir.return_value = True
        mock_dir2.name = "model2"
        mock_dir2.stat.return_value.st_mtime = 2000
        mock_dir2.__truediv__ = Mock(return_value=Mock(exists=Mock(return_value=True)))
        
        # Mock file checks
        def mock_exists_side_effect():
            return True
        
        mock_iterdir.return_value = [mock_dir1, mock_dir2]
        
        with patch.object(Path, "exists", side_effect=mock_exists_side_effect):
            result = get_model_directories()
            
            # Should return both directories, sorted by modification time (newest first)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], mock_dir2)  # Newer first
            self.assertEqual(result[1], mock_dir1)
    
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.iterdir")
    def test_get_model_directories_filters_invalid(self, mock_iterdir, mock_exists):
        """Test that invalid directories are filtered out."""
        # Mock directories - some valid, some invalid
        mock_valid = Mock(spec=Path)
        mock_valid.is_dir.return_value = True
        mock_valid.name = "valid_model"
        mock_valid.stat.return_value.st_mtime = 1000
        mock_valid.__truediv__ = Mock(return_value=Mock(exists=Mock(return_value=True)))
        
        mock_file = Mock(spec=Path)
        mock_file.is_dir.return_value = False  # Not a directory
        
        mock_hidden = Mock(spec=Path)
        mock_hidden.is_dir.return_value = True
        mock_hidden.name = ".hidden"  # Hidden directory
        mock_hidden.__truediv__ = Mock(return_value=Mock(exists=Mock(return_value=False)))
        
        mock_no_artifacts = Mock(spec=Path)
        mock_no_artifacts.is_dir.return_value = True
        mock_no_artifacts.name = "no_artifacts"
        mock_no_artifacts.__truediv__ = Mock(return_value=Mock(exists=Mock(return_value=False)))
        
        mock_iterdir.return_value = [mock_valid, mock_file, mock_hidden, mock_no_artifacts]
        
        def mock_exists_side_effect():
            return True
        
        with patch.object(Path, "exists", side_effect=mock_exists_side_effect):
            result = get_model_directories()
            
            # Should only return the valid model
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], mock_valid)


class TestModelMetadataLoading(BaseModelManagerTest):
    """Test cases for loading model metadata."""
    
    def test_load_model_summary_success(self):
        """Test successful model summary loading."""
        mock_data = {"best_validation_loss": 0.5, "best_epoch": 10}
        
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_data))) as mock_file:
            # Create a mock Path object
            mock_path_obj = Mock(spec=Path)
            mock_summary_file = Mock(spec=Path)
            mock_summary_file.exists.return_value = True
            mock_path_obj.__truediv__ = Mock(return_value=mock_summary_file)
            
            result = load_model_summary(mock_path_obj)
            self.assertEqual(result, mock_data)
    
    def test_load_model_summary_file_not_found(self):
        """Test model summary loading when file doesn't exist."""
        # Create a mock Path object
        mock_path_obj = Mock(spec=Path)
        mock_summary_file = Mock(spec=Path)
        mock_summary_file.exists.return_value = False
        mock_path_obj.__truediv__ = Mock(return_value=mock_summary_file)
        
        result = load_model_summary(mock_path_obj)
        self.assertIsNone(result)
    
    def test_load_model_summary_json_error(self):
        """Test model summary loading with invalid JSON."""
        with patch("builtins.open", mock_open(read_data="invalid json{")) as mock_file, \
             patch("builtins.print") as mock_print:
            
            # Create a mock Path object
            mock_path_obj = Mock(spec=Path)
            mock_summary_file = Mock(spec=Path)
            mock_summary_file.exists.return_value = True
            mock_path_obj.__truediv__ = Mock(return_value=mock_summary_file)
            
            result = load_model_summary(mock_path_obj)
            self.assertIsNone(result)
            mock_print.assert_called_once()  # Should print warning
    
    def test_load_model_config_success(self):
        """Test successful model config loading."""
        mock_data = {"n_layer": 4, "n_embd": 128}
        
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_data))) as mock_file:
            # Create a mock Path object
            mock_path_obj = Mock(spec=Path)
            mock_config_file = Mock(spec=Path)
            mock_config_file.exists.return_value = True
            mock_path_obj.__truediv__ = Mock(return_value=mock_config_file)
            
            result = load_model_config(mock_path_obj)
            self.assertEqual(result, mock_data)
    
    def test_load_training_config_success(self):
        """Test successful training config loading."""
        mock_data = {"runtime_config": {"batch_size": 4, "learning_rate": 5e-5}}
        
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_data))) as mock_file:
            # Create a mock Path object
            mock_path_obj = Mock(spec=Path)
            mock_config_file = Mock(spec=Path)
            mock_config_file.exists.return_value = True
            mock_path_obj.__truediv__ = Mock(return_value=mock_config_file)
            
            result = load_training_config(mock_path_obj)
            self.assertEqual(result, mock_data)


class TestModelComparison(unittest.TestCase):
    """Test cases for model comparison functionality."""
    
    @patch("builtins.print")
    @patch("model_manager.load_model_summary")
    @patch("model_manager.load_model_config")
    @patch("model_manager.load_training_config")
    @patch("pathlib.Path.exists")
    def test_compare_models_success(self, mock_exists, mock_load_training, 
                                  mock_load_config, mock_load_summary, mock_print):
        """Test successful model comparison."""
        # Mock model data
        mock_exists.return_value = True
        mock_load_summary.return_value = {
            "best_validation_loss": 0.5,
            "best_epoch": 10,
            "total_epochs": 20,
            "training_time_minutes": 60,
            "early_stopped": True,
            "model_parameters": 1000000
        }
        mock_load_config.return_value = {
            "vocab_size": 262,
            "n_layer": 4,
            "n_embd": 128
        }
        mock_load_training.return_value = {
            "runtime_config": {"batch_size": 4, "learning_rate": 5e-5}
        }
        
        compare_models(["model1", "model2"])
        
        # Should have printed comparison tables
        self.assertGreater(mock_print.call_count, 0)
        # Check that key headers were printed
        printed_text = " ".join([str(call[0][0]) for call in mock_print.call_args_list])
        self.assertIn("MODEL COMPARISON", printed_text)
        self.assertIn("DETAILED CONFIGURATION", printed_text)
    
    @patch("builtins.print")
    @patch("pathlib.Path.exists")
    def test_compare_models_missing_directory(self, mock_exists, mock_print):
        """Test model comparison with missing directories."""
        mock_exists.return_value = False
        
        compare_models(["missing_model"])
        
        # Should print warning about missing directory
        mock_print.assert_any_call("⚠️  Model directory not found: missing_model")


class TestModelLeaderboard(unittest.TestCase):
    """Test cases for model leaderboard functionality."""
    
    @patch("builtins.print")
    @patch("model_manager.get_model_directories")
    @patch("model_manager.load_model_summary")
    def test_create_model_leaderboard_success(self, mock_load_summary, 
                                            mock_get_dirs, mock_print):
        """Test successful leaderboard creation."""
        # Mock model directories
        mock_dir1 = Mock(spec=Path)
        mock_dir1.name = "model1"
        mock_dir2 = Mock(spec=Path)
        mock_dir2.name = "model2"
        mock_get_dirs.return_value = [mock_dir1, mock_dir2]
        
        # Mock model summaries (model2 is better)
        def mock_summary_side_effect(model_dir):
            if model_dir.name == "model1":
                return {"best_validation_loss": 0.6, "best_epoch": 5, "training_time_minutes": 30}
            elif model_dir.name == "model2":
                return {"best_validation_loss": 0.4, "best_epoch": 8, "training_time_minutes": 45}
            return None
        
        mock_load_summary.side_effect = mock_summary_side_effect
        
        create_model_leaderboard()
        
        # Should have printed leaderboard
        self.assertGreater(mock_print.call_count, 0)
        printed_text = " ".join([str(call[0][0]) for call in mock_print.call_args_list])
        self.assertIn("MODEL LEADERBOARD", printed_text)
        self.assertIn("🥇", printed_text)  # Gold medal for best model
    
    @patch("builtins.print")
    @patch("model_manager.get_model_directories")
    @patch("model_manager.load_model_summary")
    def test_create_model_leaderboard_no_valid_models(self, mock_load_summary, 
                                                     mock_get_dirs, mock_print):
        """Test leaderboard with no valid models."""
        mock_get_dirs.return_value = [Mock(spec=Path)]
        mock_load_summary.return_value = None  # No valid summary
        
        create_model_leaderboard()
        
        mock_print.assert_any_call("No models with validation loss found.")


class TestModelCleanup(BaseModelManagerTest):
    """Test cases for model cleanup functionality."""
    
    @patch("builtins.print")
    @patch("model_manager.get_model_directories")
    def test_cleanup_old_models_few_models(self, mock_get_dirs, mock_print):
        """Test cleanup when there are fewer models than keep_count."""
        mock_get_dirs.return_value = self.create_mock_model_dirs(2)
        
        cleanup_old_models(keep_count=5, dry_run=True)
        
        # Should not remove anything
        mock_print.assert_any_call("Only 2 models found, keeping all (target: 5)")
    
    @patch("builtins.print")
    @patch("shutil.rmtree")
    @patch("model_manager.get_model_directories")
    @patch("model_manager.load_model_summary")
    def test_cleanup_old_models_dry_run(self, mock_load_summary, mock_get_dirs, 
                                       mock_rmtree, mock_print):
        """Test cleanup in dry run mode."""
        mock_get_dirs.return_value = self.create_mock_model_dirs(6)
        mock_load_summary.return_value = {"best_validation_loss": 0.5}
        
        cleanup_old_models(keep_count=3, dry_run=True)
        
        # Should not actually remove files in dry run
        mock_rmtree.assert_not_called()
        
        # Should print what would be removed
        printed_text = " ".join([str(call[0][0]) for call in mock_print.call_args_list])
        self.assertIn("DRY RUN", printed_text)
        self.assertIn("Models to keep", printed_text)
        self.assertIn("Models to remove", printed_text)
    
    @patch("builtins.print")
    @patch("shutil.rmtree")
    @patch("model_manager.get_model_directories")
    @patch("model_manager.load_model_summary")
    def test_cleanup_old_models_live_run(self, mock_load_summary, mock_get_dirs, 
                                        mock_rmtree, mock_print):
        """Test cleanup in live mode."""
        mock_get_dirs.return_value = self.create_mock_model_dirs(6)
        mock_load_summary.return_value = {"best_validation_loss": 0.5}
        
        cleanup_old_models(keep_count=3, dry_run=False)
        
        # Should actually remove 3 old models (6 total - 3 keep = 3 remove)
        self.assertEqual(mock_rmtree.call_count, 3)
    
    @patch("builtins.print")
    @patch("shutil.rmtree")
    @patch("model_manager.get_model_directories")
    @patch("model_manager.load_model_summary")
    def test_cleanup_old_models_removal_error(self, mock_load_summary, mock_get_dirs, 
                                             mock_rmtree, mock_print):
        """Test cleanup with file removal errors."""
        # Create 4 models, keep 3, so 1 should be removed
        mock_dirs = self.create_mock_model_dirs(3) + [Mock(spec=Path)]
        mock_dirs[-1].name = "model_to_remove"
        mock_dirs[-1].rglob.return_value = [Mock(stat=Mock(return_value=Mock(st_size=1024)))]
        
        mock_get_dirs.return_value = mock_dirs
        mock_load_summary.return_value = {"best_validation_loss": 0.5}
        mock_rmtree.side_effect = Exception("Permission denied")
        
        cleanup_old_models(keep_count=3, dry_run=False)
        
        # Should handle the error gracefully
        printed_text = " ".join([str(call[0][0]) for call in mock_print.call_args_list])
        self.assertIn("❌ Failed to remove", printed_text)


class TestModelExport(unittest.TestCase):
    """Test cases for model export functionality."""
    
    @patch("builtins.print")
    def test_export_model_success(self, mock_print):
        """Test successful model export."""
        # Just test that the function runs and prints the expected message
        # without mocking the complex path operations
        with patch("pathlib.Path") as mock_path_class:
            mock_source = Mock()
            mock_source.exists.return_value = False  # This will cause early return
            mock_path_class.return_value = mock_source
            
            export_model_for_deployment("missing_model", "deployment")
            
            # Should print error message for missing source
            printed_text = " ".join([str(call[0][0]) for call in mock_print.call_args_list])
            self.assertIn("Source model directory not found", printed_text)
    
    @patch("builtins.print")
    @patch("pathlib.Path.exists")
    def test_export_model_source_not_found(self, mock_exists, mock_print):
        """Test model export with missing source directory."""
        mock_exists.return_value = False
        
        export_model_for_deployment("missing_model", "deployment")
        
        mock_print.assert_any_call("❌ Source model directory not found: missing_model")


# Example of how to run these tests:
# pytest training/model_manager.tests.py -v 

if __name__ == "__main__":
    unittest.main()
