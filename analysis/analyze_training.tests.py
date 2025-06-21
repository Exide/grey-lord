"""analyze_training.tests.py

Unit tests for the analyze_training module.
Co-located with the implementation for better organization.
"""

import unittest
import json
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock, call

from analyze_training import (
    load_training_log,
    extract_training_metrics,
    create_loss_plot,
    create_learning_rate_plot,
    create_comprehensive_analysis,
    save_analysis_report,
    main
)


class TestLoadTrainingLog(unittest.TestCase):
    """Test cases for training log loading."""
    
    def test_load_training_log_success(self):
        """Test successful training log loading."""
        mock_data = [
            {"epoch": 1, "train_loss": 1.5, "val_loss": 1.3},
            {"epoch": 2, "train_loss": 1.2, "val_loss": 1.1}
        ]
        mock_file = mock_open(read_data="\n".join([json.dumps(line) for line in mock_data]))
        
        with patch("builtins.open", mock_file):
            result = load_training_log("test_log.jsonl")
        
        self.assertEqual(result, mock_data)
    
    def test_load_training_log_file_not_found(self):
        """Test training log loading when file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError()):
            result = load_training_log("missing_log.jsonl")
        
        self.assertEqual(result, [])
    
    def test_load_training_log_json_error(self):
        """Test training log loading with invalid JSON lines."""
        mock_file = mock_open(read_data="invalid json\n{\"valid\": \"json\"}")
        
        with patch("builtins.open", mock_file), \
             patch("builtins.print") as mock_print:
            result = load_training_log("test_log.jsonl")
        
        # Should skip invalid lines and keep valid ones
        self.assertEqual(result, [{"valid": "json"}])
        mock_print.assert_called_once()  # Should print warning
    
    def test_load_training_log_empty_file(self):
        """Test training log loading with empty file."""
        mock_file = mock_open(read_data="")
        
        with patch("builtins.open", mock_file):
            result = load_training_log("empty_log.jsonl")
        
        self.assertEqual(result, [])


class TestExtractTrainingMetrics(unittest.TestCase):
    """Test cases for training metrics extraction."""
    
    def test_extract_training_metrics_complete_data(self):
        """Test metrics extraction with complete training data."""
        log_data = [
            {"epoch": 1, "train_loss": 1.5, "val_loss": 1.3, "learning_rate": 5e-5},
            {"epoch": 2, "train_loss": 1.2, "val_loss": 1.1, "learning_rate": 4e-5},
            {"epoch": 3, "train_loss": 1.0, "val_loss": 1.0, "learning_rate": 3e-5}
        ]
        
        metrics = extract_training_metrics(log_data)
        
        self.assertEqual(metrics["epochs"], [1, 2, 3])
        self.assertEqual(metrics["train_loss"], [1.5, 1.2, 1.0])
        self.assertEqual(metrics["val_loss"], [1.3, 1.1, 1.0])
        self.assertEqual(metrics["learning_rate"], [5e-5, 4e-5, 3e-5])
    
    def test_extract_training_metrics_missing_fields(self):
        """Test metrics extraction with missing optional fields."""
        log_data = [
            {"epoch": 1, "train_loss": 1.5},
            {"epoch": 2, "val_loss": 1.1},
            {"epoch": 3, "train_loss": 1.0, "val_loss": 1.0}
        ]
        
        metrics = extract_training_metrics(log_data)
        
        self.assertEqual(metrics["epochs"], [1, 2, 3])
        self.assertEqual(metrics["train_loss"], [1.5, None, 1.0])
        self.assertEqual(metrics["val_loss"], [None, 1.1, 1.0])
        self.assertEqual(metrics["learning_rate"], [None, None, None])
    
    def test_extract_training_metrics_empty_data(self):
        """Test metrics extraction with empty data."""
        metrics = extract_training_metrics([])
        
        self.assertEqual(metrics["epochs"], [])
        self.assertEqual(metrics["train_loss"], [])
        self.assertEqual(metrics["val_loss"], [])
        self.assertEqual(metrics["learning_rate"], [])
    
    def test_extract_training_metrics_mixed_types(self):
        """Test metrics extraction with mixed data types."""
        log_data = [
            {"epoch": "1", "train_loss": "1.5", "val_loss": 1.3},  # Mixed strings/numbers
            {"epoch": 2, "train_loss": 1.2, "val_loss": "1.1"},
        ]
        
        metrics = extract_training_metrics(log_data)
        
        # Should convert strings to appropriate types
        self.assertEqual(metrics["epochs"], [1, 2])
        self.assertEqual(metrics["train_loss"], [1.5, 1.2])
        self.assertEqual(metrics["val_loss"], [1.3, 1.1])


class TestCreateLossPlot(unittest.TestCase):
    """Test cases for loss plot creation."""
    
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    def test_create_loss_plot_both_losses(self, mock_tight_layout, mock_subplots):
        """Test loss plot creation with both training and validation losses."""
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        metrics = {
            "epochs": [1, 2, 3],
            "train_loss": [1.5, 1.2, 1.0],
            "val_loss": [1.3, 1.1, 1.0]
        }
        
        fig, ax = create_loss_plot(metrics)
        
        # Check that plot methods were called
        self.assertEqual(mock_ax.plot.call_count, 2)  # Training and validation loss
        mock_ax.set_xlabel.assert_called_once_with("Epoch")
        mock_ax.set_ylabel.assert_called_once_with("Loss")
        mock_ax.set_title.assert_called_once_with("Training and Validation Loss")
        mock_ax.legend.assert_called_once()
        mock_ax.grid.assert_called_once_with(True, alpha=0.3)
        mock_tight_layout.assert_called_once()
    
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    def test_create_loss_plot_train_only(self, mock_tight_layout, mock_subplots):
        """Test loss plot creation with only training loss."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        metrics = {
            "epochs": [1, 2, 3],
            "train_loss": [1.5, 1.2, 1.0],
            "val_loss": [None, None, None]
        }
        
        fig, ax = create_loss_plot(metrics)
        
        # Should only plot training loss (1 call)
        self.assertEqual(mock_ax.plot.call_count, 1)
        mock_ax.set_title.assert_called_once_with("Training Loss")
    
    @patch("matplotlib.pyplot.subplots")
    def test_create_loss_plot_no_data(self, mock_subplots):
        """Test loss plot creation with no data."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        metrics = {
            "epochs": [],
            "train_loss": [],
            "val_loss": []
        }
        
        fig, ax = create_loss_plot(metrics)
        
        # Should not plot anything
        mock_ax.plot.assert_not_called()
        mock_ax.set_title.assert_called_once_with("No Loss Data Available")


class TestCreateLearningRatePlot(unittest.TestCase):
    """Test cases for learning rate plot creation."""
    
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    def test_create_learning_rate_plot_with_data(self, mock_tight_layout, mock_subplots):
        """Test learning rate plot creation with data."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        metrics = {
            "epochs": [1, 2, 3],
            "learning_rate": [5e-5, 4e-5, 3e-5]
        }
        
        fig, ax = create_learning_rate_plot(metrics)
        
        mock_ax.plot.assert_called_once()
        mock_ax.set_xlabel.assert_called_once_with("Epoch")
        mock_ax.set_ylabel.assert_called_once_with("Learning Rate")
        mock_ax.set_title.assert_called_once_with("Learning Rate Schedule")
        mock_ax.grid.assert_called_once_with(True, alpha=0.3)
        mock_tight_layout.assert_called_once()
    
    @patch("matplotlib.pyplot.subplots")
    def test_create_learning_rate_plot_no_data(self, mock_subplots):
        """Test learning rate plot creation with no data."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        metrics = {
            "epochs": [1, 2, 3],
            "learning_rate": [None, None, None]
        }
        
        fig, ax = create_learning_rate_plot(metrics)
        
        mock_ax.plot.assert_not_called()
        mock_ax.set_title.assert_called_once_with("No Learning Rate Data Available")


class TestCreateComprehensiveAnalysis(unittest.TestCase):
    """Test cases for comprehensive analysis creation."""
    
    @patch("analyze_training.create_learning_rate_plot")
    @patch("analyze_training.create_loss_plot")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    def test_create_comprehensive_analysis(self, mock_tight_layout, mock_subplots,
                                         mock_loss_plot, mock_lr_plot):
        """Test comprehensive analysis creation."""
        # Mock figure and subplots
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Mock individual plot functions
        mock_loss_fig, mock_loss_ax = Mock(), Mock()
        mock_lr_fig, mock_lr_ax = Mock(), Mock()
        mock_loss_plot.return_value = (mock_loss_fig, mock_loss_ax)
        mock_lr_plot.return_value = (mock_lr_fig, mock_lr_ax)
        
        metrics = {
            "epochs": [1, 2, 3],
            "train_loss": [1.5, 1.2, 1.0],
            "val_loss": [1.3, 1.1, 1.0],
            "learning_rate": [5e-5, 4e-5, 3e-5]
        }
        
        fig = create_comprehensive_analysis(metrics)
        
        # Should create subplot layout
        mock_subplots.assert_called_once_with(2, 1, figsize=(12, 10))
        
        # Should call individual plot functions
        mock_loss_plot.assert_called_once_with(metrics)
        mock_lr_plot.assert_called_once_with(metrics)
        
        # Should copy plot content to subplots
        mock_axes[0].clear.assert_called_once()
        mock_axes[1].clear.assert_called_once()
        
        mock_tight_layout.assert_called_once()


class TestSaveAnalysisReport(unittest.TestCase):
    """Test cases for analysis report saving."""
    
    @patch("pathlib.Path.mkdir")
    @patch("matplotlib.pyplot.savefig")
    @patch("builtins.print")
    def test_save_analysis_report_success(self, mock_print, mock_savefig, mock_mkdir):
        """Test successful analysis report saving."""
        mock_fig = Mock()
        output_path = Path("test_output.png")
        
        save_analysis_report(mock_fig, output_path, dpi=150, format="png")
        
        # Should create parent directory
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        # Should save figure
        mock_savefig.assert_called_once_with(
            output_path, dpi=150, format="png", bbox_inches="tight"
        )
        
        # Should print success message
        mock_print.assert_any_call(f"📊 Analysis saved to: {output_path}")
    
    @patch("pathlib.Path.mkdir")
    @patch("matplotlib.pyplot.savefig")
    @patch("builtins.print")
    def test_save_analysis_report_error(self, mock_print, mock_savefig, mock_mkdir):
        """Test analysis report saving with error."""
        mock_fig = Mock()
        mock_savefig.side_effect = Exception("Permission denied")
        output_path = Path("test_output.png")
        
        save_analysis_report(mock_fig, output_path)
        
        # Should print error message
        mock_print.assert_any_call("❌ Error saving analysis: Permission denied")
    
    @patch("pathlib.Path.mkdir")
    @patch("matplotlib.pyplot.savefig")
    @patch("builtins.print")
    def test_save_analysis_report_default_params(self, mock_print, mock_savefig, mock_mkdir):
        """Test analysis report saving with default parameters."""
        mock_fig = Mock()
        output_path = Path("test_output.png")
        
        save_analysis_report(mock_fig, output_path)
        
        # Should use default parameters
        mock_savefig.assert_called_once_with(
            output_path, dpi=300, format="png", bbox_inches="tight"
        )


class TestMainFunction(unittest.TestCase):
    """Test cases for main function."""
    
    @patch("analyze_training.save_analysis_report")
    @patch("analyze_training.create_comprehensive_analysis")
    @patch("analyze_training.extract_training_metrics")
    @patch("analyze_training.load_training_log")
    @patch("builtins.print")
    def test_main_success(self, mock_print, mock_load_log, mock_extract_metrics,
                         mock_create_analysis, mock_save_report):
        """Test successful main function execution."""
        # Mock data flow
        mock_log_data = [{"epoch": 1, "train_loss": 1.0}]
        mock_metrics = {"epochs": [1], "train_loss": [1.0]}
        mock_fig = Mock()
        
        mock_load_log.return_value = mock_log_data
        mock_extract_metrics.return_value = mock_metrics
        mock_create_analysis.return_value = mock_fig
        
        result = main("training_log.jsonl", "output.png", "png")
        
        # Should execute full pipeline
        mock_load_log.assert_called_once_with("training_log.jsonl")
        mock_extract_metrics.assert_called_once_with(mock_log_data)
        mock_create_analysis.assert_called_once_with(mock_metrics)
        mock_save_report.assert_called_once_with(mock_fig, Path("output.png"), 300, "png")
        
        self.assertEqual(result, 0)
    
    @patch("analyze_training.load_training_log")
    @patch("builtins.print")
    def test_main_no_data(self, mock_print, mock_load_log):
        """Test main function with no training data."""
        mock_load_log.return_value = []
        
        result = main("empty_log.jsonl", "output.png", "png")
        
        mock_print.assert_any_call("❌ No training data found in: empty_log.jsonl")
        self.assertEqual(result, 1)
    
    @patch("analyze_training.load_training_log")
    @patch("builtins.print")
    def test_main_exception(self, mock_print, mock_load_log):
        """Test main function with unexpected exception."""
        mock_load_log.side_effect = Exception("Unexpected error")
        
        result = main("log.jsonl", "output.png", "png")
        
        mock_print.assert_any_call("❌ Error during analysis: Unexpected error")
        self.assertEqual(result, 1)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for realistic training analysis scenarios."""
    
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    def test_full_training_analysis_pipeline(self, mock_tight_layout, mock_subplots, mock_savefig):
        """Test complete training analysis pipeline with realistic data."""
        # Mock matplotlib setup
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Realistic training log data
        log_data = [
            {"epoch": 1, "train_loss": 2.5, "val_loss": 2.3, "learning_rate": 5e-4},
            {"epoch": 2, "train_loss": 2.1, "val_loss": 2.0, "learning_rate": 4.5e-4},
            {"epoch": 3, "train_loss": 1.8, "val_loss": 1.7, "learning_rate": 4e-4},
            {"epoch": 4, "train_loss": 1.6, "val_loss": 1.5, "learning_rate": 3.5e-4},
            {"epoch": 5, "train_loss": 1.4, "val_loss": 1.4, "learning_rate": 3e-4}
        ]
        
        # Extract metrics
        metrics = extract_training_metrics(log_data)
        
        # Verify extracted metrics
        self.assertEqual(len(metrics["epochs"]), 5)
        self.assertEqual(metrics["train_loss"][0], 2.5)
        self.assertEqual(metrics["val_loss"][-1], 1.4)
        self.assertEqual(metrics["learning_rate"][0], 5e-4)
        
        # Create comprehensive analysis
        fig = create_comprehensive_analysis(metrics)
        
        # Should create proper subplot structure
        mock_subplots.assert_called_once_with(2, 1, figsize=(12, 10))
        
        # Save report
        with patch("pathlib.Path.mkdir"):
            save_analysis_report(fig, Path("test_analysis.png"))
        
        mock_savefig.assert_called_once()
    
    def test_training_with_early_stopping(self):
        """Test analysis of training that stopped early."""
        log_data = [
            {"epoch": 1, "train_loss": 2.0, "val_loss": 1.9},
            {"epoch": 2, "train_loss": 1.5, "val_loss": 1.6},  # Validation increases
            {"epoch": 3, "train_loss": 1.2, "val_loss": 1.7},  # Validation continues up
            # Training stopped early due to validation loss increase
        ]
        
        metrics = extract_training_metrics(log_data)
        
        # Should handle short training runs
        self.assertEqual(len(metrics["epochs"]), 3)
        self.assertGreater(metrics["val_loss"][-1], metrics["val_loss"][0])  # Validation got worse
    
    def test_training_with_missing_validation(self):
        """Test analysis of training without validation data."""
        log_data = [
            {"epoch": 1, "train_loss": 2.0},
            {"epoch": 2, "train_loss": 1.8},
            {"epoch": 3, "train_loss": 1.6}
        ]
        
        metrics = extract_training_metrics(log_data)
        
        # Should handle missing validation data
        self.assertEqual(len(metrics["train_loss"]), 3)
        self.assertIsNone(all(v) for v in metrics["val_loss"])


# Example of how to run these tests:
# pytest analysis/analyze_training.tests.py -v 

if __name__ == "__main__":
    unittest.main()
