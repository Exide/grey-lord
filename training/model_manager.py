#!/usr/bin/env python3
"""
Model Manager - Utilities for managing, comparing, and organizing trained models
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

def get_model_directories() -> List[Path]:
    """Find all model directories in the models/all/ directory."""
    models_all_dir = Path("models/all")
    if not models_all_dir.exists():
        return []
    
    model_dirs = []
    
    # Look for directories that contain training artifacts
    for item in models_all_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check for training summary or model files
            has_summary = (item / "training_summary.json").exists()
            has_model = (item / "config.json").exists() or (item / "model.safetensors").exists()
            
            if has_summary or has_model:
                model_dirs.append(item)
    
    return sorted(model_dirs, key=lambda x: x.stat().st_mtime, reverse=True)

def load_model_summary(model_dir: Path) -> Optional[Dict]:
    """Load training summary for a model directory."""
    summary_file = model_dir / "training_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load summary for {model_dir}: {e}")
    return None

def load_model_config(model_dir: Path) -> Optional[Dict]:
    """Load model configuration."""
    config_file = model_dir / "config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config for {model_dir}: {e}")
    return None

def load_training_config(model_dir: Path) -> Optional[Dict]:
    """Load training configuration."""
    config_file = model_dir / "training_config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load training config for {model_dir}: {e}")
    return None

def compare_models(model_dirs: List[str]) -> None:
    """Compare multiple models and display a comparison table."""
    print("📊 MODEL COMPARISON")
    print("=" * 80)
    
    model_data = []
    
    for model_dir_str in model_dirs:
        model_dir = Path(model_dir_str)
        if not model_dir.exists():
            print(f"⚠️  Model directory not found: {model_dir}")
            continue
        
        summary = load_model_summary(model_dir)
        config = load_model_config(model_dir)
        training_config = load_training_config(model_dir)
        
        # Extract key metrics
        data = {
            'name': model_dir.name,
            'val_loss': summary.get('best_validation_loss', 'N/A') if summary else 'N/A',
            'best_epoch': summary.get('best_epoch', 'N/A') if summary else 'N/A',
            'total_epochs': summary.get('total_epochs', 'N/A') if summary else 'N/A',
            'training_time': summary.get('training_time_minutes', 'N/A') if summary else 'N/A',
            'early_stopped': summary.get('early_stopped', 'N/A') if summary else 'N/A',
            'parameters': summary.get('model_parameters', 'N/A') if summary else 'N/A',
            'vocab_size': config.get('vocab_size', 'N/A') if config else 'N/A',
            'n_layer': config.get('n_layer', 'N/A') if config else 'N/A',
            'n_embd': config.get('n_embd', 'N/A') if config else 'N/A',
            'batch_size': training_config.get('runtime_config', {}).get('batch_size', 'N/A') if training_config else 'N/A',
            'learning_rate': training_config.get('runtime_config', {}).get('learning_rate', 'N/A') if training_config else 'N/A',
        }
        model_data.append(data)
    
    if not model_data:
        print("No valid models found for comparison.")
        return
    
    # Print comparison table
    print(f"{'Model Name':<30} {'Val Loss':<10} {'Best Epoch':<12} {'Total Epochs':<13} {'Time (min)':<12} {'Early Stop':<11}")
    print("-" * 95)
    
    for data in model_data:
        val_loss = f"{data['val_loss']:.4f}" if isinstance(data['val_loss'], (int, float)) else str(data['val_loss'])
        training_time = f"{data['training_time']:.1f}" if isinstance(data['training_time'], (int, float)) else str(data['training_time'])
        early_stop = "Yes" if data['early_stopped'] is True else "No" if data['early_stopped'] is False else str(data['early_stopped'])
        
        print(f"{data['name']:<30} {val_loss:<10} {str(data['best_epoch']):<12} {str(data['total_epochs']):<13} {training_time:<12} {early_stop:<11}")
    
    # Print detailed configuration comparison
    print(f"\n📋 DETAILED CONFIGURATION")
    print("-" * 80)
    print(f"{'Model Name':<30} {'Layers':<8} {'Emb Dim':<9} {'Vocab':<7} {'Batch':<7} {'LR':<12} {'Params':<10}")
    print("-" * 80)
    
    for data in model_data:
        lr_str = f"{data['learning_rate']:.1e}" if isinstance(data['learning_rate'], (int, float)) else str(data['learning_rate'])
        params_str = f"{data['parameters']:,}" if isinstance(data['parameters'], (int, float)) else str(data['parameters'])
        
        print(f"{data['name']:<30} {str(data['n_layer']):<8} {str(data['n_embd']):<9} {str(data['vocab_size']):<7} {str(data['batch_size']):<7} {lr_str:<12} {params_str:<10}")

def create_model_leaderboard() -> None:
    """Create a leaderboard of all models sorted by validation loss."""
    print("🏆 MODEL LEADERBOARD")
    print("=" * 60)
    
    model_dirs = get_model_directories()
    valid_models = []
    
    for model_dir in model_dirs:
        summary = load_model_summary(model_dir)
        if summary and 'best_validation_loss' in summary:
            valid_models.append({
                'name': model_dir.name,
                'val_loss': summary['best_validation_loss'],
                'best_epoch': summary.get('best_epoch', 'N/A'),
                'training_time': summary.get('training_time_minutes', 'N/A'),
                'early_stopped': summary.get('early_stopped', False)
            })
    
    if not valid_models:
        print("No models with validation loss found.")
        return
    
    # Sort by validation loss (lower is better)
    valid_models.sort(key=lambda x: x['val_loss'])
    
    print(f"{'Rank':<6} {'Model Name':<30} {'Val Loss':<12} {'Best Epoch':<12} {'Time (min)':<12}")
    print("-" * 75)
    
    for i, model in enumerate(valid_models, 1):
        rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}"
        training_time = f"{model['training_time']:.1f}" if isinstance(model['training_time'], (int, float)) else str(model['training_time'])
        
        print(f"{rank_emoji:<6} {model['name']:<30} {model['val_loss']:<12.4f} {str(model['best_epoch']):<12} {training_time:<12}")

def cleanup_old_models(keep_count: int = 5, dry_run: bool = True) -> None:
    """Clean up old model directories, keeping only the most recent ones."""
    print(f"🧹 MODEL CLEANUP ({'DRY RUN' if dry_run else 'LIVE RUN'})")
    print("=" * 50)
    
    model_dirs = get_model_directories()
    
    if len(model_dirs) <= keep_count:
        print(f"Only {len(model_dirs)} models found, keeping all (target: {keep_count})")
        return
    
    to_keep = model_dirs[:keep_count]
    to_remove = model_dirs[keep_count:]
    
    print(f"📂 Models to keep ({len(to_keep)}):")
    for model_dir in to_keep:
        summary = load_model_summary(model_dir)
        val_loss = summary.get('best_validation_loss', 'N/A') if summary else 'N/A'
        val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)
        print(f"  ✅ {model_dir.name} (val_loss: {val_loss_str})")
    
    print(f"\n🗑️  Models to remove ({len(to_remove)}):")
    total_size = 0
    for model_dir in to_remove:
        # Calculate directory size
        dir_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        total_size += dir_size
        size_str = f"{dir_size / (1024**2):.1f}MB" if dir_size > 0 else "0MB"
        
        summary = load_model_summary(model_dir)
        val_loss = summary.get('best_validation_loss', 'N/A') if summary else 'N/A'
        val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)
        
        print(f"  🗑️  {model_dir.name} ({size_str}, val_loss: {val_loss_str})")
        
        if not dry_run:
            try:
                shutil.rmtree(model_dir)
                print(f"    ✅ Removed")
            except Exception as e:
                print(f"    ❌ Failed to remove: {e}")
    
    print(f"\n💾 Total space {'would be' if dry_run else ''} freed: {total_size / (1024**2):.1f}MB")
    
    if dry_run:
        print("\n💡 Run with --no-dry-run to actually remove these directories")

def create_model_links() -> None:
    """Create symbolic links for 'latest' and 'best' models."""
    print("🔗 CREATING MODEL LINKS")
    print("=" * 30)
    
    model_dirs = get_model_directories()
    if not model_dirs:
        print("No models found.")
        return
    
    # Find latest model (most recently modified)
    latest_model = model_dirs[0] if model_dirs else None
    
    # Find best model (lowest validation loss)
    best_model = None
    best_val_loss = float('inf')
    
    for model_dir in model_dirs:
        summary = load_model_summary(model_dir)
        if summary and 'best_validation_loss' in summary:
            val_loss = summary['best_validation_loss']
            # Only compare if val_loss is a number (not "unknown" string)
            if isinstance(val_loss, (int, float)) and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model_dir
    
    # Create links inside models directory
    models_dir = Path("models")
    latest_link = models_dir / "latest"
    best_link = models_dir / "best"
    
    # Remove existing links (including broken symlinks)
    for link in [latest_link, best_link]:
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
                print(f"🗑️  Removed existing {link.name} symlink")
        except FileNotFoundError:
            # Link doesn't exist, which is fine
            pass
        except Exception as e:
            print(f"⚠️  Warning: Could not remove existing {link.name}: {e}")
            # Try to force remove it
            try:
                link.unlink(missing_ok=True)
            except:
                pass
    
    # Create new links (point to all/modelname since models are in models/all/)
    if latest_model:
        try:
            target_path = Path("all") / latest_model.name
            latest_link.symlink_to(target_path, target_is_directory=True)
            print(f"✅ 'models/latest' -> {target_path}")
        except Exception as e:
            print(f"❌ Failed to create 'models/latest' link: {e}")
    
    if best_model:
        try:
            target_path = Path("all") / best_model.name
            best_link.symlink_to(target_path, target_is_directory=True)
            print(f"✅ 'models/best' -> {target_path} (val_loss: {best_val_loss:.4f})")
        except Exception as e:
            print(f"❌ Failed to create 'models/best' link: {e}")

def export_model_for_deployment(model_dir: str, output_dir: str = "deployment") -> None:
    """Export a model with minimal files needed for deployment."""
    print(f"📦 EXPORTING MODEL FOR DEPLOYMENT")
    print("=" * 40)
    
    source_dir = Path(model_dir)
    target_dir = Path(output_dir)
    
    if not source_dir.exists():
        print(f"❌ Source model directory not found: {source_dir}")
        return
    
    # Create deployment directory
    target_dir.mkdir(exist_ok=True)
    
    # Essential files for deployment
    essential_files = [
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "generation_config.json",
        "vocab_to_int.json",
        "int_to_vocab.json",
        "training_summary.json"
    ]
    
    copied_files = []
    for filename in essential_files:
        source_file = source_dir / filename
        target_file = target_dir / filename
        
        if source_file.exists():
            shutil.copy2(source_file, target_file)
            copied_files.append(filename)
            file_size = source_file.stat().st_size
            print(f"✅ {filename} ({file_size / 1024:.1f}KB)")
    
    # Calculate total size
    total_size = sum((target_dir / f).stat().st_size for f in copied_files)
    
    print(f"\n📊 Deployment package created:")
    print(f"  • Location: {target_dir}")
    print(f"  • Files: {len(copied_files)}")
    print(f"  • Total size: {total_size / (1024**2):.1f}MB")
    
    # Create a deployment info file
    deployment_info = {
        "source_model": str(source_dir),
        "export_date": datetime.now().isoformat(),
        "files_included": copied_files,
        "total_size_bytes": total_size
    }
    
    with open(target_dir / "deployment_info.json", "w") as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"  • Deployment info: deployment_info.json")

if __name__ == "__main__":
    # Simple CLI for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_manager.py [command]")
        print("Commands: list, leaderboard, compare <dir1> <dir2> ..., cleanup, links, export <model_dir>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list":
        model_dirs = get_model_directories()
        print(f"Found {len(model_dirs)} model directories:")
        for model_dir in model_dirs:
            print(f"  • {model_dir.name}")
    
    elif command == "leaderboard":
        create_model_leaderboard()
    
    elif command == "compare":
        if len(sys.argv) < 4:
            print("Usage: python model_manager.py compare <dir1> <dir2> [dir3] ...")
            sys.exit(1)
        compare_models(sys.argv[2:])
    
    elif command == "cleanup":
        cleanup_old_models(keep_count=5, dry_run=True)
    
    elif command == "links":
        create_model_links()
    
    elif command == "export":
        if len(sys.argv) < 3:
            print("Usage: python model_manager.py export <model_dir>")
            sys.exit(1)
        export_model_for_deployment(sys.argv[2])
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1) 
