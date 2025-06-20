"""trainer.tests.py

Unit tests for the trainer module.
Co-located with the implementation for better organization.
"""

import pytest
from unittest.mock import Mock

from trainer import TrainingState


class TestTrainingState:
    """Test cases for TrainingState class."""
    
    def test_initialization(self):
        """Test TrainingState initialization."""
        state = TrainingState()
        
        assert state.history['epochs'] == []
        assert state.history['train_losses'] == []
        assert state.history['val_losses'] == []
        assert state.history['learning_rates'] == []
        assert state.history['best_epoch'] == 1
        assert state.history['best_val_loss'] == float('inf')
        assert state.history['early_stopped'] == False
        assert state.best_val_loss == float('inf')
        assert state.patience_counter == 0
    
    def test_record_epoch(self):
        """Test epoch recording."""
        state = TrainingState()
        
        state.record_epoch(1, 0.5, 0.4, 1e-4)
        
        assert state.history['epochs'] == [1]
        assert state.history['train_losses'] == [0.5]
        assert state.history['val_losses'] == [0.4]
        assert state.history['learning_rates'] == [1e-4]
    
    def test_check_improvement_success(self):
        """Test improvement detection when validation loss improves."""
        state = TrainingState()
        
        # First improvement
        improved = state.check_improvement(0.3, 1)
        
        assert improved == True
        assert state.best_val_loss == 0.3
        assert state.history['best_epoch'] == 1
        assert state.history['best_val_loss'] == 0.3
        assert state.patience_counter == 0
    
    def test_check_improvement_no_improvement(self):
        """Test no improvement detection."""
        state = TrainingState()
        state.best_val_loss = 0.2  # Set a lower baseline
        
        # No improvement
        improved = state.check_improvement(0.3, 2)
        
        assert improved == False
        assert state.best_val_loss == 0.2  # Unchanged
        assert state.patience_counter == 1
    
    def test_should_early_stop(self):
        """Test early stopping condition."""
        state = TrainingState()
        state.patience_counter = 5
        
        # Should stop when patience exceeded
        assert state.should_early_stop(patience=5) == True
        assert state.should_early_stop(patience=6) == False
    
    def test_finalize_training(self):
        """Test training finalization."""
        state = TrainingState()
        state.start_training()
        
        # Simulate some time passing
        import time
        time.sleep(0.01)
        
        state.finalize_training(epoch=10)
        
        assert state.history['final_epoch'] == 10
        assert state.history['total_training_time'] > 0


class TestTrainerUtilities:
    """Test cases for trainer utility functions."""
    
    def test_training_flow_integration(self):
        """Test the overall training state flow."""
        state = TrainingState()
        state.start_training()
        
        # Simulate training epochs
        epochs = [
            (1, 0.8, 0.7, 1e-4),  # Initial improvement
            (2, 0.6, 0.6, 9e-5),  # Improvement
            (3, 0.5, 0.65, 8e-5), # No improvement
            (4, 0.4, 0.7, 7e-5),  # No improvement
        ]
        
        patience = 3
        
        for epoch, train_loss, val_loss, lr in epochs:
            state.record_epoch(epoch, train_loss, val_loss, lr)
            improved = state.check_improvement(val_loss, epoch)
            
            if epoch <= 2:
                assert improved == True
            else:
                assert improved == False
            
            # Check if should early stop
            if state.should_early_stop(patience):
                break
        
        # Should not have triggered early stopping yet (patience=3, counter=2)
        assert state.should_early_stop(patience) == False
        assert state.patience_counter == 2
        assert state.best_val_loss == 0.6  # From epoch 2


# Example of how to run these tests:
# pytest src/trainer.tests.py -v 