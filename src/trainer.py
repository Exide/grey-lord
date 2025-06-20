"""trainer.py

Core training logic, validation, and artifact management for Grey Lord model.
Contains the main training loop and validation functionality.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup

from config_utils import get_vocab_config, get_training_config


def validate(model: AutoModelForCausalLM, validation_loader: DataLoader, device: torch.device) -> float:
    """Run validation and return average loss.
    
    Args:
        model: Model to validate
        validation_loader: Validation data loader
        device: Device to run validation on
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in validation_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Skip batches with no real tokens
            if attention_mask.sum() == 0:
                continue
                
            outputs = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=input_ids)
            loss = outputs.loss
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else float('inf')


class TrainingState:
    """Manages training state and history."""
    
    def __init__(self):
        self.history = {
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'best_epoch': 1,
            'best_val_loss': float('inf'),
            'early_stopped': False,
            'total_training_time': 0.0,
            'final_epoch': 0
        }
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.start_time = 0.0
    
    def start_training(self):
        """Mark the start of training."""
        self.start_time = time.time()
    
    def record_epoch(self, epoch: int, train_loss: float, val_loss: float, learning_rate: float):
        """Record results from an epoch."""
        self.history['epochs'].append(epoch)
        self.history['train_losses'].append(train_loss)
        self.history['val_losses'].append(val_loss)
        self.history['learning_rates'].append(learning_rate)
    
    def check_improvement(self, val_loss: float, epoch: int) -> bool:
        """Check if validation loss improved and update best values."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.history['best_epoch'] = epoch
            self.history['best_val_loss'] = val_loss
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False
    
    def should_early_stop(self, patience: int) -> bool:
        """Check if early stopping should be triggered."""
        return self.patience_counter >= patience
    
    def finalize_training(self, epoch: int):
        """Finalize training state."""
        end_time = time.time()
        self.history['total_training_time'] = end_time - self.start_time
        self.history['final_epoch'] = epoch


def setup_optimizer_and_scheduler(
    model: AutoModelForCausalLM, 
    learning_rate: float, 
    num_training_steps: int,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1
) -> tuple[torch.optim.Optimizer, Any]:
    """Set up optimizer and learning rate scheduler.
    
    Args:
        model: Model to optimize
        learning_rate: Initial learning rate
        num_training_steps: Total number of training steps
        weight_decay: Weight decay for regularization
        warmup_ratio: Fraction of steps to use for warmup
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8
    )
    
    # Linear warmup + decay scheduler
    num_warmup_steps = int(warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler


def save_training_artifacts(
    save_dir: Path,
    training_state: TrainingState,
    training_config: Dict[str, Any],
    data_size_str: str,
    model: AutoModelForCausalLM
):
    """Save all training artifacts to disk.
    
    Args:
        save_dir: Directory to save artifacts to
        training_state: Training state with history
        training_config: Training configuration
        data_size_str: Human-readable data size string
        model: Trained model
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training configuration
    training_config['training_end_time'] = datetime.now().isoformat()
    training_config['final_results'] = {
        'best_epoch': training_state.history['best_epoch'],
        'best_val_loss': training_state.history['best_val_loss'],
        'final_train_loss': training_state.history['train_losses'][-1] if training_state.history['train_losses'] else None,
        'final_val_loss': training_state.history['val_losses'][-1] if training_state.history['val_losses'] else None,
        'total_training_time_minutes': training_state.history['total_training_time'] / 60,
        'early_stopped': training_state.history['early_stopped'],
        'epochs_completed': training_state.history['final_epoch']
    }
    
    with open(save_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    
    # Save training history
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(training_state.history, f, indent=2)
    
    # Create training summary
    summary = {
        'model_name': save_dir.name,
        'training_completed': datetime.now().isoformat(),
        'best_epoch': training_state.history['best_epoch'],
        'best_validation_loss': training_state.history['best_val_loss'],
        'total_epochs': training_state.history['final_epoch'],
        'training_time_minutes': round(training_state.history['total_training_time'] / 60, 2),
        'early_stopped': training_state.history['early_stopped'],
        'data_size': data_size_str,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'regularization': {
            'dropout': model.config.dropout if hasattr(model.config, 'dropout') else None,
            'weight_decay': training_config.get('weight_decay', 0.01),
            'gradient_clip': training_config.get('gradient_clip', 1.0)
        }
    }
    
    with open(save_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Copy vocabulary files
    try:
        vocab_config = get_vocab_config()
        vocab_to_int_file = vocab_config.get("vocab_to_int_file", "vocab_to_int.json")
        int_to_vocab_file = vocab_config.get("int_to_vocab_file", "int_to_vocab.json")
        shutil.copy2(vocab_to_int_file, save_dir / "vocab_to_int.json")
        shutil.copy2(int_to_vocab_file, save_dir / "int_to_vocab.json")
        print(f"[info] Vocabulary files copied to output directory")
    except Exception as e:
        print(f"[warning] Could not copy vocabulary files: {e}")


def generate_training_plots(save_dir: Path, training_state: TrainingState):
    """Generate and save training plots if matplotlib is available.
    
    Args:
        save_dir: Directory to save plots to
        training_state: Training state with history
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        epochs_list = training_state.history['epochs']
        ax1.plot(epochs_list, training_state.history['train_losses'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs_list, training_state.history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
        ax1.axvline(x=training_state.history['best_epoch'], color='g', linestyle='--', 
                   label=f'Best Epoch ({training_state.history["best_epoch"]})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate schedule
        ax2.plot(epochs_list, training_state.history['learning_rates'], 'purple', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[info] Training curves saved to: {save_dir / 'training_curves.png'}")
    except ImportError:
        print("[warning] matplotlib not available, skipping training plots")
    except Exception as e:
        print(f"[warning] Could not generate training plots: {e}")


def run_training_loop(
    model: AutoModelForCausalLM,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    patience: int,
    gradient_clip: float = 1.0
) -> TrainingState:
    """Run the main training loop.
    
    Args:
        model: Model to train
        training_loader: Training data loader
        validation_loader: Validation data loader  
        device: Device to train on
        num_epochs: Number of epochs to train
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        patience: Early stopping patience
        gradient_clip: Gradient clipping threshold
        
    Returns:
        TrainingState with training history
    """
    training_state = TrainingState()
    training_state.start_training()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"\n[info] Epoch {epoch+1}/{num_epochs} - Training...")
        for batch_idx, batch in enumerate(training_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Skip batches with no real tokens
            if attention_mask.sum() == 0:
                continue

            # The labels are just the input sequence shifted internally by 🤗
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids)
            loss: torch.Tensor = outputs.loss  # scalar tensor

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            scheduler.step()

            loss_value = loss.item()
            if not torch.isnan(loss) and not torch.isinf(loss):
                epoch_loss += loss_value
                num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx+1}, Loss: {loss_value:.4f}")

        # Calculate average training loss
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        
        # Validation phase
        print(f"[info] Epoch {epoch+1}/{num_epochs} - Validating...")
        validation_loss = validate(model, validation_loader, device)
        
        # Record training history
        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        training_state.record_epoch(epoch + 1, avg_train_loss, validation_loss, current_lr)
        
        # Print epoch summary
        if num_batches > 0:
            print(f"Epoch {epoch+1}/{num_epochs} – Train Loss: {avg_train_loss:.4f}, Val Loss: {validation_loss:.4f}, LR: {current_lr:.2e}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} – No valid training batches, Val Loss: {validation_loss:.4f}, LR: {current_lr:.2e}")
        
        # Early stopping logic
        improved = training_state.check_improvement(validation_loss, epoch + 1)
        if improved:
            print(f"[info] New best validation loss: {validation_loss:.4f}")
        else:
            print(f"[info] Validation loss did not improve. Patience: {training_state.patience_counter}/{patience}")
        
        # Check early stopping condition
        if training_state.should_early_stop(patience):
            print(f"[info] Early stopping triggered after {epoch+1} epochs")
            print(f"[info] No improvement in validation loss for {patience} consecutive epochs")
            training_state.history['early_stopped'] = True
            break

    training_state.finalize_training(epoch + 1)
    return training_state
