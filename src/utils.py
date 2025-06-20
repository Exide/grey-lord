"""utils.py

Utility functions for Grey Lord training system.
"""

import time
from pathlib import Path
from typing import Union


def generate_save_path(
    data_dir: str,
    batch_size: int,
    learning_rate: float,
    max_seq_len: int,
    model_path: Union[str, None] = None
) -> str:
    """Generate a save path based on training parameters.
    
    Args:
        data_dir: Data directory path
        batch_size: Training batch size
        learning_rate: Learning rate
        max_seq_len: Maximum sequence length
        model_path: Optional existing model path
        
    Returns:
        Generated save path string
    """
    timestamp = str(int(time.time()))
    
    # Extract dataset version from data directory name
    data_dir_path = Path(data_dir)
    data_version = "v1"  # default
    if "_v" in data_dir_path.name:
        data_version = data_dir_path.name.split("_v")[-1]
    
    # Build minimal name components
    name_parts = [data_version]  # Start with dataset version
    
    # Add training type
    if model_path:
        name_parts.append("cont")
    else:
        name_parts.append("new")
    
    # Add key parameters (only non-defaults)
    if batch_size != 32:
        name_parts.append(f"batch-{batch_size}")
    
    if learning_rate != 3e-4:
        lr_str = f"{learning_rate:.0e}".replace("e-0", "e").replace("e-", "e")
        name_parts.append(f"learning-rate-{lr_str}")
    
    if max_seq_len != 1024:
        if max_seq_len >= 1024:
            seq_str = f"{max_seq_len//1024}k" if max_seq_len % 1024 == 0 else f"{max_seq_len}"
        else:
            seq_str = str(max_seq_len)
        name_parts.append(f"seq-{seq_str}")
    
    # Add timestamp
    name_parts.append(timestamp)
    
    return "_".join(name_parts) 