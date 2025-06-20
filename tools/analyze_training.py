#!/usr/bin/env python3
"""
Analyze training output to identify overfitting and optimal stopping points.
"""

import re
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np

def parse_training_log(log_text: str) -> Tuple[List[int], List[float], List[float]]:
    """Parse training log to extract epoch, train loss, and validation loss."""
    epochs = []
    train_losses = []
    val_losses = []
    
    # Pattern to match epoch summary lines
    pattern = r'Epoch (\d+)/\d+ – Train Loss: ([\d.]+), Val Loss: ([\d.]+)'
    
    for match in re.finditer(pattern, log_text):
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        val_loss = float(match.group(3))
        
        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    return epochs, train_losses, val_losses

def find_optimal_stopping_point(epochs: List[int], val_losses: List[float], patience: int = 5) -> int:
    """Find the optimal stopping point based on validation loss."""
    if not val_losses:
        return 1
    
    best_epoch = 1
    best_loss = float('inf')
    patience_counter = 0
    
    for i, (epoch, val_loss) in enumerate(zip(epochs, val_losses)):
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                return best_epoch
    
    return best_epoch

def analyze_overfitting(epochs: List[int], train_losses: List[float], val_losses: List[float]) -> dict:
    """Analyze overfitting patterns in the training data."""
    if len(epochs) < 5:
        return {"overfitting": False, "severity": "none"}
    
    # Find the point where validation loss starts increasing
    min_val_idx = np.argmin(val_losses)
    min_val_epoch = epochs[min_val_idx]
    min_val_loss = val_losses[min_val_idx]
    
    # Check if validation loss increased significantly after the minimum
    final_val_loss = val_losses[-1]
    val_loss_increase = final_val_loss - min_val_loss
    relative_increase = val_loss_increase / min_val_loss
    
    # Check if training loss kept decreasing
    final_train_loss = train_losses[-1]
    train_loss_at_min_val = train_losses[min_val_idx]
    train_loss_decrease = train_loss_at_min_val - final_train_loss
    
    # Determine overfitting severity
    if relative_increase > 0.5:  # 50% increase in validation loss
        severity = "severe"
    elif relative_increase > 0.2:  # 20% increase
        severity = "moderate"
    elif relative_increase > 0.05:  # 5% increase
        severity = "mild"
    else:
        severity = "none"
    
    return {
        "overfitting": relative_increase > 0.05,
        "severity": severity,
        "optimal_epoch": min_val_epoch,
        "min_val_loss": min_val_loss,
        "final_val_loss": final_val_loss,
        "val_loss_increase": val_loss_increase,
        "relative_increase": relative_increase,
        "train_loss_at_optimal": train_loss_at_min_val,
        "final_train_loss": final_train_loss,
        "train_val_gap": final_val_loss - final_train_loss
    }

def plot_training_curves(epochs: List[int], train_losses: List[float], val_losses: List[float], 
                        optimal_epoch: int, save_path: str = "training_analysis.png"):
    """Plot training and validation curves with optimal stopping point."""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.axvline(x=optimal_epoch, color='g', linestyle='--', label=f'Optimal Stop (Epoch {optimal_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoomed view of early epochs
    plt.subplot(2, 1, 2)
    early_epochs = min(30, len(epochs))
    plt.plot(epochs[:early_epochs], train_losses[:early_epochs], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs[:early_epochs], val_losses[:early_epochs], 'r-', label='Validation Loss', linewidth=2)
    if optimal_epoch <= early_epochs:
        plt.axvline(x=optimal_epoch, color='g', linestyle='--', label=f'Optimal Stop (Epoch {optimal_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (First 30 Epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function."""
    print("=== Training Analysis ===")
    print("Paste your training log here (end with 'END' on a new line):")
    
    log_lines = []
    while True:
        line = input()
        if line.strip() == 'END':
            break
        log_lines.append(line)
    
    log_text = '\n'.join(log_lines)
    epochs, train_losses, val_losses = parse_training_log(log_text)
    
    if not epochs:
        print("❌ No training data found in the log. Please check the format.")
        return
    
    print(f"\n📊 **Training Summary**")
    print(f"• Total epochs: {len(epochs)}")
    print(f"• Initial train loss: {train_losses[0]:.4f}")
    print(f"• Final train loss: {train_losses[-1]:.4f}")
    print(f"• Initial val loss: {val_losses[0]:.4f}")
    print(f"• Final val loss: {val_losses[-1]:.4f}")
    
    # Analyze overfitting
    analysis = analyze_overfitting(epochs, train_losses, val_losses)
    
    print(f"\n🔍 **Overfitting Analysis**")
    if analysis["overfitting"]:
        print(f"❌ **Overfitting detected**: {analysis['severity']}")
        print(f"• Optimal stopping point: Epoch {analysis['optimal_epoch']}")
        print(f"• Best validation loss: {analysis['min_val_loss']:.4f}")
        print(f"• Final validation loss: {analysis['final_val_loss']:.4f}")
        print(f"• Validation loss increased by: {analysis['val_loss_increase']:.4f} ({analysis['relative_increase']:.1%})")
        print(f"• Training-validation gap: {analysis['train_val_gap']:.4f}")
        
        print(f"\n💡 **Recommendations**")
        print(f"• Use early stopping with patience=3-5")
        print(f"• Your model was actually best at epoch {analysis['optimal_epoch']}")
        print(f"• Consider reducing model size or adding regularization")
        print(f"• Increase batch size to 4-8 for better training stability")
    else:
        print("✅ No significant overfitting detected")
        print("• Training appears to be progressing normally")
    
    # Find optimal stopping with different patience values
    print(f"\n⏱️ **Early Stopping Analysis**")
    for patience in [3, 5, 10]:
        optimal_stop = find_optimal_stopping_point(epochs, val_losses, patience)
        print(f"• With patience={patience}: Would stop at epoch {optimal_stop}")
    
    # Plot training curves
    plot_training_curves(epochs, train_losses, val_losses, analysis['optimal_epoch'])
    print(f"\n📈 Training curves saved as 'training_analysis.png'")

if __name__ == "__main__":
    main() 