#!/usr/bin/env python3
"""
Grey Lord - Unified CLI Interface

A unified command-line interface for all Grey Lord model training and analysis operations.
This replaces the collection of separate scripts with a clean, organized CLI.

Usage:
    python grey-lord.py train --epochs 50
    python grey-lord.py continue --model-path trained_model
    python grey-lord.py analyze --training-dir model-xyz
    python grey-lord.py debug vocab
    python grey-lord.py optimize batch-size
    python grey-lord.py data prepare --source ../telnet-data
"""

import argparse
import sys
from pathlib import Path

def create_parser():
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Grey Lord - MajorMUD Language Model Training & Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Common Examples:
  # Train a new model
  python grey-lord.py train --epochs 50 --batch-size 32
  
  # Continue training from checkpoint
  python grey-lord.py train --model-path trained_model --epochs 25 --learning-rate 1e-5
  
  # Analyze training results
  python grey-lord.py analyze --training-dir grey-lord-v1-batch32-lr3e4-20241201-143022
  
  # Debug vocabulary issues
  python grey-lord.py debug vocab
  
  # Optimize batch size for your GPU
  python grey-lord.py optimize batch-size
  
  # Prepare training data
  python grey-lord.py data prepare --source ../c-telnet-proxy
        """
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train or continue training a model')
    add_train_arguments(train_parser)
    
    # Analyze subcommand
    analyze_parser = subparsers.add_parser('analyze', help='Analyze training results')
    add_analyze_arguments(analyze_parser)
    
    # Debug subcommand
    debug_parser = subparsers.add_parser('debug', help='Debug model, vocab, or data issues')
    add_debug_arguments(debug_parser)
    
    # Optimize subcommand
    optimize_parser = subparsers.add_parser('optimize', help='Optimization tools')
    add_optimize_arguments(optimize_parser)
    
    # Data subcommand
    data_parser = subparsers.add_parser('data', help='Data management tools')
    add_data_arguments(data_parser)
    
    # Config subcommand
    config_parser = subparsers.add_parser('config', help='Configuration management')
    add_config_arguments(config_parser)
    
    # Model management subcommand
    model_parser = subparsers.add_parser('model', help='Model management and comparison')
    add_model_arguments(model_parser)
    
    return parser

def add_train_arguments(parser):
    """Add training-specific arguments."""
    from src.config_utils import get_training_config, get_data_config
    
    training_config = get_training_config()
    data_config = get_data_config()
    
    # Core training parameters
    parser.add_argument("--epochs", type=int, 
                       default=training_config.get("default_epochs", 10),
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, 
                       default=training_config.get("default_batch_size", 4),
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, 
                       default=training_config.get("default_lr", 5e-5),
                       help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, 
                       default=training_config.get("default_max_seq_len", 512),
                       help="Maximum sequence length")
    
    # Data parameters
    parser.add_argument("--data-dir", type=str, 
                       default=data_config.get("default_data_dir", "./data"),
                       help="Directory containing training data")
    parser.add_argument("--file-glob", type=str, 
                       default=data_config.get("default_file_glob", "*"),
                       help="Glob pattern to match training files")
    
    # Model management
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to existing model to continue training from")
    parser.add_argument("--save-path", type=str, default=None,
                       help="Directory to save the trained model")
    
    # Training behavior
    parser.add_argument("--val-split", type=float, 
                       default=training_config.get("default_val_split", 0.2),
                       help="Fraction of data to use for validation")
    parser.add_argument("--patience", type=int, 
                       default=training_config.get("default_patience", 5),
                       help="Number of epochs to wait for improvement before early stopping")
    
    # Hardware
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU training even if CUDA is available")

def add_analyze_arguments(parser):
    """Add analysis-specific arguments."""
    parser.add_argument("--training-dir", type=str, required=True,
                       help="Directory containing training artifacts to analyze")
    parser.add_argument("--output", type=str, default="analysis_report.png",
                       help="Output file for analysis plots")

def add_debug_arguments(parser):
    """Add debugging-specific arguments."""
    debug_subparsers = parser.add_subparsers(dest='debug_type', help='Debug tools')
    
    # Vocabulary debugging
    vocab_parser = debug_subparsers.add_parser('vocab', help='Debug vocabulary and model compatibility')
    
    # Model debugging
    model_parser = debug_subparsers.add_parser('model', help='Debug model configuration and architecture')
    
    # Sequence debugging
    seq_parser = debug_subparsers.add_parser('sequences', help='Debug long sequence issues')

def add_optimize_arguments(parser):
    """Add optimization-specific arguments."""
    opt_subparsers = parser.add_subparsers(dest='optimize_type', help='Optimization tools')
    
    # Batch size optimization
    batch_parser = opt_subparsers.add_parser('batch-size', help='Find optimal batch size for your GPU')
    batch_parser.add_argument("--memory-gb", type=float, default=10.0,
                             help="Available GPU memory in GB")
    
    # Sequence length optimization
    seq_parser = opt_subparsers.add_parser('seq-len', help='Calculate optimal sequence length')
    seq_parser.add_argument("--batch-size", type=int, default=4,
                           help="Batch size to optimize for")
    
    # Memory analysis
    mem_parser = opt_subparsers.add_parser('memory', help='Analyze memory usage')

def add_data_arguments(parser):
    """Add data management arguments."""
    data_subparsers = parser.add_subparsers(dest='data_type', help='Data management tools')
    
    # Data preparation
    prep_parser = data_subparsers.add_parser('prepare', help='Prepare training data')
    prep_parser.add_argument("--source", type=str, required=True,
                            help="Source directory containing raw training files")
    prep_parser.add_argument("--target", type=str, default="training_data",
                            help="Target directory for prepared training data")
    prep_parser.add_argument("--min-size-kb", type=int, default=1,
                            help="Minimum file size in KB")
    prep_parser.add_argument("--move", action="store_true",
                            help="Move files instead of copying")
    
    # Data validation
    val_parser = data_subparsers.add_parser('validate', help='Validate training data')
    val_parser.add_argument("--data-dir", type=str, default="training_data",
                           help="Directory to validate")
    
    # Data statistics
    stats_parser = data_subparsers.add_parser('stats', help='Show data statistics')
    stats_parser.add_argument("--data-dir", type=str, default="training_data",
                             help="Directory to analyze")

def add_config_arguments(parser):
    """Add configuration management arguments."""
    config_subparsers = parser.add_subparsers(dest='config_type', help='Configuration tools')
    
    # Show configuration
    show_parser = config_subparsers.add_parser('show', help='Show current configuration')
    
    # Validate configuration
    val_parser = config_subparsers.add_parser('validate', help='Validate configuration')

def add_model_arguments(parser):
    """Add model management arguments."""
    model_subparsers = parser.add_subparsers(dest='model_type', help='Model management tools')
    
    # List models
    list_parser = model_subparsers.add_parser('list', help='List all available models')
    
    # Model leaderboard
    leaderboard_parser = model_subparsers.add_parser('leaderboard', help='Show model leaderboard by validation loss')
    
    # Compare models
    compare_parser = model_subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('models', nargs='+', help='Model directories to compare')
    
    # Cleanup old models
    cleanup_parser = model_subparsers.add_parser('cleanup', help='Clean up old model directories')
    cleanup_parser.add_argument('--keep', type=int, default=5, help='Number of models to keep')
    cleanup_parser.add_argument('--no-dry-run', action='store_true', help='Actually delete files (default is dry run)')
    
    # Create model links
    links_parser = model_subparsers.add_parser('links', help='Create latest/best model symlinks')
    
    # Export model for deployment
    export_parser = model_subparsers.add_parser('export', help='Export model for deployment')
    export_parser.add_argument('model_dir', help='Model directory to export')
    export_parser.add_argument('--output', default='deployment', help='Output directory')

def handle_train_command(args):
    """Handle the train command."""
    from src.train import train
    
    # Force CPU if requested
    if args.cpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        torch.cuda.is_available = lambda: False
    
    print("🚀 TRAINING WITH GREY LORD")
    print("=" * 50)
    
    try:
        train(
            data_dir=args.data_dir,
            file_glob=args.file_glob,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_path=args.save_path,
            max_seq_len=args.max_seq_len,
            model_path=args.model_path,
            val_split=args.val_split,
            patience=args.patience
        )
        print("\n🎉 Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)

def handle_analyze_command(args):
    """Handle the analyze command."""
    training_dir = Path(args.training_dir)
    
    if not training_dir.exists():
        print(f"❌ Training directory not found: {training_dir}")
        sys.exit(1)
    
    # Check for training artifacts
    history_file = training_dir / "training_history.json"
    summary_file = training_dir / "training_summary.json"
    
    if not history_file.exists():
        print(f"❌ Training history not found: {history_file}")
        print("This directory may not contain Grey Lord training artifacts.")
        sys.exit(1)
    
    print(f"📊 ANALYZING TRAINING: {training_dir.name}")
    print("=" * 50)
    
    # Load and display training summary
    import json
    
    try:
        with open(summary_file) as f:
            summary = json.load(f)
        
        print(f"📋 Training Summary:")
        print(f"  • Model: {summary['model_name']}")
        print(f"  • Best epoch: {summary['best_epoch']}")
        print(f"  • Best validation loss: {summary['best_validation_loss']:.4f}")
        print(f"  • Total epochs: {summary['total_epochs']}")
        print(f"  • Training time: {summary['training_time_minutes']:.1f} minutes")
        print(f"  • Early stopped: {'Yes' if summary['early_stopped'] else 'No'}")
        print(f"  • Model parameters: {summary['model_parameters']:,}")
        
        # Generate comparison plots if we have training history
        with open(history_file) as f:
            history = json.load(f)
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss curves
            epochs = history['epochs']
            ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
            ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
            ax1.axvline(x=history['best_epoch'], color='g', linestyle='--', 
                       label=f'Best Epoch ({history["best_epoch"]})')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'Training Progress: {training_dir.name}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Learning rate
            if 'learning_rates' in history:
                ax2.plot(epochs, history['learning_rates'], 'purple', linewidth=2)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Learning Rate')
                ax2.set_title('Learning Rate Schedule')
                ax2.set_yscale('log')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(args.output, dpi=300, bbox_inches='tight')
            print(f"\n📈 Analysis plot saved: {args.output}")
            
        except ImportError:
            print("⚠️  matplotlib not available, skipping plots")
            
    except Exception as e:
        print(f"❌ Error analyzing training data: {e}")
        sys.exit(1)

def handle_debug_command(args):
    """Handle the debug command."""
    if args.debug_type == 'vocab':
        print("🔍 DEBUGGING VOCABULARY")
        print("=" * 30)
        from tools.debug_vocab import main as debug_vocab_main
        debug_vocab_main()
    
    elif args.debug_type == 'model':
        print("🔍 DEBUGGING MODEL")
        print("=" * 30)
        from tools.debug_model import main as debug_model_main
        debug_model_main()
    
    elif args.debug_type == 'sequences':
        print("🔍 DEBUGGING SEQUENCES")
        print("=" * 30)
        # For now, just give instructions
        print("Long sequence debugging tools:")
        print("1. Check model's max position embeddings")
        print("2. Verify CUDA memory availability")
        print("3. Test with smaller sequence lengths")
        print("Use: python debug_long_sequences.py for detailed analysis")
    
    else:
        print("❌ Unknown debug type. Available: vocab, model, sequences")
        sys.exit(1)

def handle_optimize_command(args):
    """Handle the optimize command."""
    if args.optimize_type == 'batch-size':
        print("⚡ OPTIMIZING BATCH SIZE")
        print("=" * 30)
        from tools.batch_optimizer import find_optimal_batch_size
        find_optimal_batch_size(args.memory_gb)
    
    elif args.optimize_type == 'seq-len':
        print("⚡ OPTIMIZING SEQUENCE LENGTH")
        print("=" * 30)
        from tools.calculate_max_seq_len import main as calc_main
        calc_main()
    
    elif args.optimize_type == 'memory':
        print("⚡ ANALYZING MEMORY USAGE")
        print("=" * 30)
        from tools.calculate_max_seq_len import estimate_memory_usage, get_model_config
        
        model_config = get_model_config()
        seq_lengths = [512, 1024, 2048, 4096]
        
        print("Memory usage for different sequence lengths:")
        for seq_len in seq_lengths:
            mem_info = estimate_memory_usage(seq_len, 1)
            print(f"  {seq_len:4d} tokens: {mem_info['total_memory_mb']:.1f} MB")
    
    else:
        print("❌ Unknown optimization type. Available: batch-size, seq-len, memory")
        sys.exit(1)

def handle_data_command(args):
    """Handle the data command."""
    if args.data_type == 'prepare':
        print("📁 PREPARING TRAINING DATA")
        print("=" * 30)
        from tools.prepare_training_data import prepare_training_data
        prepare_training_data(
            source_dir=args.source,
            target_dir=args.target,
            min_size_kb=args.min_size_kb,
            move_instead_of_copy=args.move
        )
    
    elif args.data_type == 'validate':
        print("✅ VALIDATING TRAINING DATA")
        print("=" * 30)
        data_dir = Path(args.data_dir)
        
        if not data_dir.exists():
            print(f"❌ Data directory not found: {data_dir}")
            sys.exit(1)
        
        files = list(data_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        
        print(f"📊 Data Directory: {data_dir}")
        print(f"  • Files: {len(files)}")
        print(f"  • Total size: {total_size / (1024**2):.1f} MB")
        print(f"  • Average file size: {total_size / len(files) / 1024:.1f} KB" if files else "  • No files")
    
    elif args.data_type == 'stats':
        print("📊 DATA STATISTICS")
        print("=" * 30)
        # Similar to validate but with more detail
        data_dir = Path(args.data_dir)
        
        if not data_dir.exists():
            print(f"❌ Data directory not found: {data_dir}")
            sys.exit(1)
        
        files = [f for f in data_dir.glob("*") if f.is_file()]
        if not files:
            print("No files found.")
            return
        
        sizes = [f.stat().st_size for f in files]
        total_size = sum(sizes)
        
        print(f"📊 Statistics for: {data_dir}")
        print(f"  • Total files: {len(files)}")
        print(f"  • Total size: {total_size / (1024**2):.1f} MB")
        print(f"  • Average size: {total_size / len(files) / 1024:.1f} KB")
        print(f"  • Largest file: {max(sizes) / 1024:.1f} KB")
        print(f"  • Smallest file: {min(sizes) / 1024:.1f} KB")
    
    else:
        print("❌ Unknown data command. Available: prepare, validate, stats")
        sys.exit(1)

def handle_config_command(args):
    """Handle the config command."""
    if args.config_type == 'show':
        print("⚙️  CURRENT CONFIGURATION")
        print("=" * 30)
        from src.config_utils import print_config_summary
        print_config_summary()
    
    elif args.config_type == 'validate':
        print("✅ VALIDATING CONFIGURATION")
        print("=" * 30)
        try:
            from src.config_utils import load_config
            config = load_config()
            print("✅ Configuration file is valid JSON")
            
            # Check required sections
            required_sections = ['model', 'training', 'data', 'vocab']
            for section in required_sections:
                if section in config:
                    print(f"✅ Section '{section}' found")
                else:
                    print(f"❌ Section '{section}' missing")
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            sys.exit(1)
    
    else:
        print("❌ Unknown config command. Available: show, validate")
        sys.exit(1)

def handle_model_command(args):
    """Handle the model command."""
    from src.model_manager import (
        get_model_directories, create_model_leaderboard, 
        compare_models, cleanup_old_models, create_model_links, 
        export_model_for_deployment
    )
    
    if args.model_type == 'list':
        print("📂 AVAILABLE MODELS")
        print("=" * 30)
        model_dirs = get_model_directories()
        if not model_dirs:
            print("No trained models found.")
            return
        
        for i, model_dir in enumerate(model_dirs, 1):
            from src.model_manager import load_model_summary
            summary = load_model_summary(model_dir)
            val_loss = summary.get('best_validation_loss', 'N/A') if summary else 'N/A'
            val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)
            
            # Get directory size
            try:
                dir_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                size_str = f"{dir_size / (1024**2):.1f}MB"
            except:
                size_str = "?MB"
            
            print(f"  {i:2d}. {model_dir.name} (val_loss: {val_loss_str}, size: {size_str})")
    
    elif args.model_type == 'leaderboard':
        create_model_leaderboard()
    
    elif args.model_type == 'compare':
        compare_models(args.models)
    
    elif args.model_type == 'cleanup':
        cleanup_old_models(keep_count=args.keep, dry_run=not args.no_dry_run)
    
    elif args.model_type == 'links':
        create_model_links()
    
    elif args.model_type == 'export':
        export_model_for_deployment(args.model_dir, args.output)
    
    else:
        print("❌ Unknown model command. Available: list, leaderboard, compare, cleanup, links, export")
        sys.exit(1)

def check_virtual_environment():
    """Check if running in a virtual environment and warn if not."""
    import os
    import sys
    
    # Check for virtual environment indicators
    in_venv = (
        hasattr(sys, 'real_prefix') or  # virtualenv
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or  # venv
        'VIRTUAL_ENV' in os.environ  # environment variable
    )
    
    if not in_venv:
        print("⚠️  WARNING: Not running in a virtual environment!")
        print("   Recommended: Create and activate a virtual environment first:")
        print("   python -m venv .venv")
        if os.name == 'nt':  # Windows
            print("   .venv\\Scripts\\activate")
        else:  # Linux/Mac
            print("   source .venv/bin/activate")
        print("   pip install -r requirements.txt")
        print()

def main():
    """Main entry point."""
    check_virtual_environment()
    
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Handle commands
    try:
        if args.command == 'train':
            handle_train_command(args)
        elif args.command == 'analyze':
            handle_analyze_command(args)
        elif args.command == 'debug':
            handle_debug_command(args)
        elif args.command == 'optimize':
            handle_optimize_command(args)
        elif args.command == 'data':
            handle_data_command(args)
        elif args.command == 'config':
            handle_config_command(args)
        elif args.command == 'model':
            handle_model_command(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 