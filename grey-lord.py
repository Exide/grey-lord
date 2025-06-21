#!/usr/bin/env python3
"""
Grey Lord - MajorMUD AI

Unified command-line interface for all operations:

USAGE:
Simply call with your preferred python binary:
  python grey-lord.py [command]

COMMANDS:
- AI agent telnet client with trained model
- Model training and management
- Analysis, debugging, and optimization tools
- Data preparation and management

EXAMPLES:
    python grey-lord.py agent --config agent_config.json
    python grey-lord.py train --epochs 50 --batch-size 32
    python grey-lord.py analyze --training-dir models/your-model
    python grey-lord.py debug vocab
    python grey-lord.py optimize batch-size
    python grey-lord.py data prepare --source ../telnet-data
"""

import argparse
import sys
import subprocess
import logging
from pathlib import Path


def setup_logging():
    """Set up basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('grey-lord.log'),
            logging.StreamHandler()
        ]
    )


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
    
    # Check for conda environment indicators
    in_conda = (
        'CONDA_DEFAULT_ENV' in os.environ or  # conda environment variable
        'conda' in sys.executable.lower() or  # conda in python path
        'anaconda' in sys.executable.lower() or  # anaconda in python path
        'miniconda' in sys.executable.lower()  # miniconda in python path
    )
    
    if not (in_venv or in_conda):
        print("⚠️  WARNING: Not running in a virtual environment!")
        print("   Recommended: Create and activate a virtual environment first:")
        print("   python -m venv .venv")
        if os.name == 'nt':  # Windows
            print("   .venv\\Scripts\\activate")
        else:  # Linux/Mac
            print("   source .venv/bin/activate")
        print("   pip install -r requirements.txt")
        print()


def start_agent_client(config_path: str = 'agent_config.json') -> int:
    """Start the AI agent telnet client.
    
    Args:
        config_path: Path to the agent configuration file
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        print("🤖 Starting Grey Lord...")
        print("Ctrl+T: Toggle AI | Ctrl+C: Quit")
        print()
        
        # Import and start the client
        from agent.telnet_client import TelnetClient
        
        client = TelnetClient(config_path)
        client.start()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nExiting...")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def handle_train_command(args):
    """Handle the train command."""
    sys.path.insert(0, str(Path(__file__).parent / "training"))
    
    print("🚀 TRAINING MODEL")
    print("=" * 30)
    
    try:
        from trainer import ModelTrainer
        from config_utils import get_training_config, get_data_config
        
        # Initialize trainer with arguments
        trainer = ModelTrainer(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_len=args.context_window,
            data_dir=args.data_dir,
            file_glob=args.file_glob,
            model_path=args.model_path,
            save_path=args.save_path,
            val_split=args.val_split,
            patience=args.patience,
            force_cpu=args.cpu
        )
        
        trainer.train()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


def handle_analyze_command(args):
    """Handle the analyze command."""
    sys.path.insert(0, str(Path(__file__).parent / "analysis"))
    
    print("📊 ANALYZING TRAINING RESULTS")
    print("=" * 30)
    print(f"Training directory: {args.training_dir}")
    print(f"Output: {args.output} ({args.format})")
    
    try:
        from analyze_training import parse_training_log, analyze_overfitting, plot_training_curves, find_optimal_stopping_point
        
        # Look for training log file
        training_dir = Path(args.training_dir)
        log_file = training_dir / "training.log"
        
        if not log_file.exists():
            # Try to find any .log file
            log_files = list(training_dir.glob("*.log"))
            if log_files:
                log_file = log_files[0]
            else:
                print(f"❌ No training log found in {training_dir}")
                print("   Expected: training.log or any .log file")
                sys.exit(1)
        
        print(f"📊 Reading training log: {log_file}")
        
        # Parse the training log
        with open(log_file, 'r') as f:
            log_text = f.read()
        
        epochs, train_losses, val_losses = parse_training_log(log_text)
        
        if not epochs:
            print("❌ No training data found in the log. Please check the format.")
            sys.exit(1)
        
        print(f"📈 Found {len(epochs)} epochs of training data")
        print(f"   Train loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
        print(f"   Val loss: {val_losses[0]:.4f} → {val_losses[-1]:.4f}")
        
        # Analyze overfitting
        analysis = analyze_overfitting(epochs, train_losses, val_losses)
        
        print(f"\n🔍 Overfitting Analysis:")
        if analysis["overfitting"]:
            print(f"   ❌ Overfitting detected: {analysis['severity']}")
            print(f"   • Optimal stopping point: Epoch {analysis['optimal_epoch']}")
            print(f"   • Best validation loss: {analysis['min_val_loss']:.4f}")
            print(f"   • Val loss increased by: {analysis['relative_increase']:.1%}")
        else:
            print("   ✅ No significant overfitting detected")
        
        # Generate plots
        output_path = args.output
        if not output_path.endswith(('.png', '.pdf', '.svg')):
            output_path = f"{output_path}.{args.format}"
        
        plot_training_curves(epochs, train_losses, val_losses, analysis['optimal_epoch'], output_path)
        print(f"📊 Analysis plots saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def handle_debug_command(args):
    """Handle the debug command."""
    sys.path.insert(0, str(Path(__file__).parent / "analysis"))
    
    if not args.debug_type:
        print("❌ Please specify a debug type: vocab, model, or sequences")
        sys.exit(1)
    
    print(f"🔍 DEBUG: {args.debug_type.upper()}")
    print("=" * 30)
    
    try:
        if args.debug_type == 'vocab':
            # Change to vocab directory for debugging
            original_dir = Path.cwd()
            vocab_path = Path(args.vocab_path)
            if vocab_path.exists():
                import os
                os.chdir(vocab_path)
            
            try:
                # Import and run vocab debugging
                from debug_vocab import main as debug_vocab_main
                debug_vocab_main()
            finally:
                os.chdir(original_dir)
                
        elif args.debug_type == 'model':
            # Change to model directory for debugging
            original_dir = Path.cwd()
            model_path = Path(args.model_path)
            if model_path.exists():
                import os
                os.chdir(model_path.parent if model_path.is_file() else model_path)
            
            try:
                from debug_model import main as debug_model_main
                debug_model_main()
            finally:
                os.chdir(original_dir)
                
        elif args.debug_type == 'sequences':
            from debug_long_sequences import debug_cuda_crash
            debug_cuda_crash()
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def handle_optimize_command(args):
    """Handle the optimize command."""
    sys.path.insert(0, str(Path(__file__).parent / "analysis"))
    
    if not args.optimize_type:
        print("❌ Please specify optimization type: batch-size, context-window, or memory")
        sys.exit(1)
    
    print(f"⚡ OPTIMIZE: {args.optimize_type.upper()}")
    print("=" * 30)
    
    try:
        if args.optimize_type == 'batch-size':
            from batch_optimizer import find_optimal_batch_size
            find_optimal_batch_size(args.memory_gb)
            
        elif args.optimize_type == 'context-window':
            from calculate_context_window import find_max_seq_len, estimate_memory_usage
            import torch
            
            print(f"🔍 Calculating optimal context window for batch size {args.batch_size}...")
            
            # Use provided memory or detect GPU memory
            available_memory_gb = args.memory_gb
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                print(f"🎮 Detected GPU: {torch.cuda.get_device_name(device)} ({gpu_memory_gb:.1f}GB)")
                if args.memory_gb == 10.0:  # Default value, use detected
                    available_memory_gb = min(gpu_memory_gb, 10.0)  # Conservative estimate
            
            # Find maximum sequence length
            max_seq_len, memory_breakdown = find_max_seq_len(available_memory_gb, args.batch_size)
            
            print(f"✅ Recommended context window: {max_seq_len} tokens")
            print(f"📊 Memory usage: {memory_breakdown['total_memory_mb']:.1f} MB")
            print(f"⚙️  Use: --context-window {max_seq_len} --batch-size {args.batch_size}")
            
        elif args.optimize_type == 'memory':
            from calculate_context_window import estimate_memory_usage
            
            # Use model path if provided, otherwise use default settings
            seq_len = 2048  # Default sequence length for analysis
            batch_size = 4  # Default batch size
            
            print(f"🔍 Analyzing memory usage (seq_len={seq_len}, batch_size={batch_size})...")
            
            memory_info = estimate_memory_usage(seq_len, batch_size)
            
            print(f"\n📊 Memory Breakdown:")
            print(f"   Model parameters: {memory_info['model_memory_mb']:.1f} MB")
            print(f"   Gradients: {memory_info['gradient_memory_mb']:.1f} MB")
            print(f"   Optimizer states: {memory_info['optimizer_memory_mb']:.1f} MB")
            print(f"   Activations: {memory_info['activation_memory_mb']:.1f} MB")
            print(f"   Total: {memory_info['total_memory_mb']:.1f} MB")
            print(f"   Parameters: {memory_info['total_params']:,}")
            
            # Show different sequence lengths
            print(f"\n📈 Memory usage by sequence length:")
            for test_seq_len in [512, 1024, 2048, 4096]:
                mem = estimate_memory_usage(test_seq_len, batch_size)
                print(f"   {test_seq_len:4d} tokens: {mem['total_memory_mb']:.1f} MB")
                
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def handle_data_command(args):
    """Handle the data command."""
    if not args.data_type:
        print("❌ Please specify data command: prepare, validate, or stats")
        sys.exit(1)
    
    if args.data_type == 'prepare':
        print("📁 PREPARING TRAINING DATA")
        print("=" * 30)
        
        source_dir = Path(args.source)
        target_dir = Path(args.target)
        
        if not source_dir.exists():
            print(f"❌ Source directory not found: {source_dir}")
            sys.exit(1)
        
        target_dir.mkdir(exist_ok=True)
        
        import shutil
        
        files_processed = 0
        total_size = 0
        
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                file_size_kb = file_path.stat().st_size / 1024
                
                if file_size_kb >= args.min_size_kb:
                    dest_path = target_dir / file_path.name
                    
                    if args.move:
                        shutil.move(str(file_path), str(dest_path))
                    else:
                        shutil.copy2(str(file_path), str(dest_path))
                    
                    files_processed += 1
                    total_size += file_path.stat().st_size
        
        print(f"✅ Processed {files_processed} files")
        print(f"📊 Total size: {total_size / (1024**2):.1f} MB")
        print(f"📂 Target directory: {target_dir}")
    
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


def handle_config_command(args):
    """Handle the config command."""
    sys.path.insert(0, str(Path(__file__).parent / "training"))
    
    if args.config_type == 'show':
        print("⚙️  CURRENT CONFIGURATION")
        print("=" * 30)
        try:
            from config_utils import print_config_summary
            print_config_summary()
        except ImportError:
            print("Configuration utilities not available")
    
    elif args.config_type == 'validate':
        print("✅ VALIDATING CONFIGURATION")
        print("=" * 30)
        try:
            from config_utils import load_config
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


def handle_model_command(args):
    """Handle the model command."""
    sys.path.insert(0, str(Path(__file__).parent / "training"))
    
    try:
        from model_manager import (
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
                try:
                    from model_manager import load_model_summary
                    summary = load_model_summary(model_dir)
                    val_loss = summary.get('best_validation_loss', 'N/A') if summary else 'N/A'
                    val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)
                    
                    # Get directory size
                    dir_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                    size_str = f"{dir_size / (1024**2):.1f}MB"
                    
                    print(f"  {i:2d}. {model_dir.name} (val_loss: {val_loss_str}, size: {size_str})")
                except:
                    print(f"  {i:2d}. {model_dir.name} (unable to read details)")
        
        elif args.model_type == 'leaderboard':
            create_model_leaderboard()
        
        elif args.model_type == 'compare':
            compare_models(args.models)
        
        elif args.model_type == 'cleanup':
            cleanup_old_models(keep_count=args.keep, dry_run=args.dry_run)
        
        elif args.model_type == 'links':
            create_model_links()
        
        elif args.model_type == 'export':
            export_model_for_deployment(args.model_dir, args.output)
        
    except ImportError as e:
        print(f"❌ Model management utilities not available: {e}")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="Grey Lord - MajorMUD AI Training & Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Commands:
  agent     Start AI agent telnet client
  train     Model training operations
  analyze   Analyze training results
  debug     Debug model, vocab, or data issues
  optimize  Performance optimization tools
  data      Data preparation and management
  config    Configuration management
  model     Model management and comparison

Examples:
  # Start AI Agent
  python grey-lord.py agent --config agent_config.json
  
  # Train Model
  python grey-lord.py train --epochs 50 --batch-size 32
  
  # Analyze Results
  python grey-lord.py analyze --training-dir models/your-model
  
  # Debug Issues
  python grey-lord.py debug vocab
  
  # Optimize Performance
  python grey-lord.py optimize batch-size --memory-gb 8
  
  # Manage Data
  python grey-lord.py data prepare --source ../telnet-data
  
  # Model Management
  python grey-lord.py model list

For detailed help on each command:
  python grey-lord.py <command> --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Agent command
    agent_parser = subparsers.add_parser('agent', help='Start AI agent telnet client')
    agent_parser.add_argument('--config', default='agent_config.json', 
                             help='Configuration file path (default: agent_config.json)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train or continue training a model')
    
    # Add training configuration dynamically
    try:
        sys.path.insert(0, str(Path(__file__).parent / "training"))
        from config_utils import get_training_config, get_data_config
        training_config = get_training_config()
        data_config = get_data_config()
    except ImportError:
        training_config = {}
        data_config = {}
    
    # Core training parameters
    train_parser.add_argument("--epochs", type=int, 
                           default=training_config.get("default_epochs", 10),
                           help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, 
                           default=training_config.get("default_batch_size", 4),
                           help="Training batch size")
    train_parser.add_argument("--learning-rate", type=float, 
                           default=training_config.get("default_lr", 5e-5),
                           help="Learning rate")
    train_parser.add_argument("--context-window", type=int, 
                           default=training_config.get("default_max_seq_len", 512),
                           help="Context window length (in tokens)")
    
    # Data parameters
    train_parser.add_argument("--data-dir", type=str, 
                           default=data_config.get("default_data_dir", "./data"),
                           help="Directory containing training data")
    train_parser.add_argument("--file-glob", type=str, 
                           default=data_config.get("default_file_glob", "*"),
                           help="Glob pattern to match training files")
    
    # Model management
    train_parser.add_argument("--model-path", type=str, default=None,
                           help="Path to existing model to continue training from")
    train_parser.add_argument("--save-path", type=str, default=None,
                           help="Directory to save the trained model")
    
    # Training behavior
    train_parser.add_argument("--val-split", type=float, 
                           default=training_config.get("default_val_split", 0.2),
                           help="Fraction of data to use for validation")
    train_parser.add_argument("--patience", type=int, 
                           default=training_config.get("default_patience", 5),
                           help="Number of epochs to wait for improvement before early stopping")
    
    # Hardware
    train_parser.add_argument("--cpu", action="store_true",
                           help="Force CPU training even if CUDA is available")
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze training results')
    analyze_parser.add_argument("--training-dir", type=str, required=True,
                               help="Directory containing training artifacts to analyze")
    analyze_parser.add_argument("--output", type=str, default="analysis_report.png",
                               help="Output file for analysis plots")
    analyze_parser.add_argument("--format", type=str, default="png",
                               choices=["png", "pdf", "svg", "html"],
                               help="Output format")
    
    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Debug model, vocab, or data issues')
    debug_subparsers = debug_parser.add_subparsers(dest='debug_type', help='Debug tools')
    
    # Vocabulary debugging
    vocab_parser = debug_subparsers.add_parser('vocab', help='Debug vocabulary and model compatibility')
    vocab_parser.add_argument("--vocab-path", type=str, default="data/",
                             help="Path to vocabulary files")
    
    # Model debugging
    model_debug_parser = debug_subparsers.add_parser('model', help='Debug model configuration and architecture')
    model_debug_parser.add_argument("--model-path", type=str, required=True,
                                   help="Path to model to debug")
    
    # Sequence debugging
    seq_parser = debug_subparsers.add_parser('sequences', help='Debug long sequence issues')
    seq_parser.add_argument("--data-dir", type=str, default="training_data",
                           help="Training data directory")
    seq_parser.add_argument("--max-length", type=int, default=8192,
                           help="Maximum sequence length to analyze")
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Performance optimization tools')
    opt_subparsers = optimize_parser.add_subparsers(dest='optimize_type', help='Optimization tools')
    
    # Batch size optimization
    batch_parser = opt_subparsers.add_parser('batch-size', help='Find optimal batch size for your GPU')
    batch_parser.add_argument("--memory-gb", type=float, default=10.0,
                             help="Available GPU memory in GB")
    batch_parser.add_argument("--model-config", type=str, default="model_config.json",
                             help="Model configuration file")
    
    # Context window optimization
    context_parser = opt_subparsers.add_parser('context-window', help='Calculate optimal context window length')
    context_parser.add_argument("--batch-size", type=int, default=4,
                               help="Batch size to optimize for")
    context_parser.add_argument("--memory-gb", type=float, default=10.0,
                               help="Available GPU memory in GB")
    
    # Memory analysis
    mem_parser = opt_subparsers.add_parser('memory', help='Analyze memory usage')
    mem_parser.add_argument("--model-path", type=str, required=True,
                           help="Path to model for memory analysis")
    
    # Data command
    data_parser = subparsers.add_parser('data', help='Data preparation and management')
    data_subparsers = data_parser.add_subparsers(dest='data_type', help='Data management tools')
    
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
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_type', help='Configuration tools')
    
    # Show configuration
    show_parser = config_subparsers.add_parser('show', help='Show current configuration')
    
    # Validate configuration
    validate_parser = config_subparsers.add_parser('validate', help='Validate configuration file')
    
    # Model command
    model_parser = subparsers.add_parser('model', help='Model management and comparison')
    model_subparsers = model_parser.add_subparsers(dest='model_type', help='Model management tools')
    
    # List models
    list_parser = model_subparsers.add_parser('list', help='List available trained models')
    
    # Leaderboard
    leaderboard_parser = model_subparsers.add_parser('leaderboard', help='Show model performance leaderboard')
    
    # Compare models
    compare_parser = model_subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument("models", nargs="+", help="Model directories to compare")
    
    # Cleanup old models
    cleanup_parser = model_subparsers.add_parser('cleanup', help='Clean up old model checkpoints')
    cleanup_parser.add_argument("--keep", type=int, default=3,
                               help="Number of models to keep")
    cleanup_parser.add_argument("--dry-run", action="store_true",
                               help="Preview changes without actually deleting files")
    
    # Create model links
    links_parser = model_subparsers.add_parser('links', help='Create convenience links to models')
    
    # Export model
    export_parser = model_subparsers.add_parser('export', help='Export model for deployment')
    export_parser.add_argument("--model-dir", type=str, required=True,
                              help="Directory containing model to export")
    export_parser.add_argument("--output", type=str, required=True,
                              help="Output directory for exported model")
    
    return parser


def main() -> int:
    """Main entry point and command router.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Handle the case where no arguments are provided
    if len(sys.argv) == 1:
        parser = create_parser()
        parser.print_help()
        return 1
    
    parser = create_parser()
    
    # Special case: if first arg is --help, show main help
    if sys.argv[1] in ['-h', '--help']:
        parser.print_help()
        return 0
    
    try:
        args = parser.parse_args()
    except SystemExit as e:
        return e.code if e.code is not None else 1
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Check virtual environment for all commands except agent
    if args.command != 'agent':
        check_virtual_environment()
    
    # Set up logging for all commands
    setup_logging()
    
    # Route to appropriate handler
    try:
        if args.command == 'agent':
            return start_agent_client(args.config)
        elif args.command == 'train':
            handle_train_command(args)
            return 0
        elif args.command == 'analyze':
            handle_analyze_command(args)
            return 0
        elif args.command == 'debug':
            handle_debug_command(args)
            return 0
        elif args.command == 'optimize':
            handle_optimize_command(args)
            return 0
        elif args.command == 'data':
            handle_data_command(args)
            return 0
        elif args.command == 'config':
            handle_config_command(args)
            return 0
        elif args.command == 'model':
            handle_model_command(args)
            return 0
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
