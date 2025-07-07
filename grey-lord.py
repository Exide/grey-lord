#!/usr/bin/env python3
import argparse
import sys
import logging
from pathlib import Path
import json
from datetime import datetime
from data.api import copy_data_files, prune_data_files


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


def handle_analyze_command(args):
    """Unified analysis command for training results and data."""
    if args.analyze_type == 'training':
        print("📊 ANALYZING TRAINING RESULTS")
        print("=" * 40)
        print(f"Model directory: {args.model_dir}")
        
        try:
            from analysis.analyze_training import analyze_overfitting, plot_training_curves
            
            training_dir = Path(args.model_dir)
            history_file = training_dir / "training_history.json"
            
            if not history_file.exists():
                log_file = training_dir / "training.log"
                if not log_file.exists():
                    log_files = list(training_dir.glob("*.log"))
                    if log_files:
                        log_file = log_files[0]
                        print(f"❌ No training_history.json found in {training_dir}")
                        print("   Found legacy .log file, but this analysis expects JSON format")
                        print("   Please re-train your model to generate proper training history")
                        sys.exit(1)
                    else:
                        print(f"❌ No training history found in {training_dir}")
                        print("   Expected: training_history.json (created during training)")
                        print("   This appears to be an incomplete or corrupted model directory")
                        sys.exit(1)
            
            analysis_results_file = training_dir / "analysis_results.json"
            analysis_plots_file = training_dir / "analysis_plots.png"
            
            if analysis_results_file.exists() and analysis_plots_file.exists():
                print(f"✅ Found existing analysis artifacts:")
                print(f"   📄 Analysis results: {analysis_results_file}")
                print(f"   📊 Analysis plots: {analysis_plots_file}")
                
                with open(analysis_results_file, 'r') as f:
                    analysis = json.load(f)
                
                epochs = analysis['epochs']
                train_losses = analysis['train_losses'] 
                val_losses = analysis['val_losses']
                overfitting_analysis = analysis['overfitting_analysis']
                
                print(f"📈 Training data: {len(epochs)} epochs")
                print(f"   Train loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
                print(f"   Val loss: {val_losses[0]:.4f} → {val_losses[-1]:.4f}")
                
                print(f"\n🔍 Overfitting Analysis:")
                if overfitting_analysis["overfitting"]:
                    print(f"   ❌ Overfitting detected: {overfitting_analysis['severity']}")
                    print(f"   • Optimal stopping point: Epoch {overfitting_analysis['optimal_epoch']}")
                    print(f"   • Best validation loss: {overfitting_analysis['min_val_loss']:.4f}")
                    print(f"   • Val loss increased by: {overfitting_analysis['relative_increase']:.1%}")
                else:
                    print("   ✅ No significant overfitting detected")
                
                print(f"\n📊 Using existing analysis plots: {analysis_plots_file}")
                
                if hasattr(args, 'output') and args.output != "analysis_report.png":
                    output_path = args.output
                    if not output_path.endswith(('.png', '.pdf', '.svg')):
                        output_path = f"{output_path}.png"
                    
                    import shutil
                    shutil.copy2(analysis_plots_file, output_path)
                    print(f"📋 Analysis plots copied to: {output_path}")
                
                return
            
            print(f"🔄 Generating analysis artifacts...")
            print(f"📊 Reading training history: {history_file}")
            
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            epochs = history_data.get('epochs', [])
            train_losses = history_data.get('train_losses', [])
            val_losses = history_data.get('val_losses', [])
            
            if not epochs or not train_losses or not val_losses:
                print("❌ Incomplete training data found in the history file.")
                print(f"   Epochs: {len(epochs)}, Train losses: {len(train_losses)}, Val losses: {len(val_losses)}")
                sys.exit(1)
            
            print(f"📈 Found {len(epochs)} epochs of training data")
            print(f"   Train loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
            print(f"   Val loss: {val_losses[0]:.4f} → {val_losses[-1]:.4f}")
            
            overfitting_analysis = analyze_overfitting(epochs, train_losses, val_losses)
            
            print(f"\n🔍 Overfitting Analysis:")
            if overfitting_analysis["overfitting"]:
                print(f"   ❌ Overfitting detected: {overfitting_analysis['severity']}")
                print(f"   • Optimal stopping point: Epoch {overfitting_analysis['optimal_epoch']}")
                print(f"   • Best validation loss: {overfitting_analysis['min_val_loss']:.4f}")
                print(f"   • Val loss increased by: {overfitting_analysis['relative_increase']:.1%}")
            else:
                print("   ✅ No significant overfitting detected")
            
            plots_output_path = training_dir / "analysis_plots.png"
            plot_training_curves(epochs, train_losses, val_losses, overfitting_analysis['optimal_epoch'], str(plots_output_path))
            print(f"📊 Analysis plots saved to: {plots_output_path}")
            
            analysis_results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'model_directory': str(training_dir),
                'epochs': epochs,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'overfitting_analysis': overfitting_analysis,
                'plots_file': 'analysis_plots.png'
            }
            
            with open(analysis_results_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            print(f"📄 Analysis results saved to: {analysis_results_file}")
            
            if hasattr(args, 'output') and args.output != "analysis_report.png":
                output_path = args.output
                if not output_path.endswith(('.png', '.pdf', '.svg')):
                    output_path = f"{output_path}.png"
                
                import shutil
                shutil.copy2(plots_output_path, output_path)
                print(f"📋 Analysis plots also copied to: {output_path}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            sys.exit(1)
    
    elif args.analyze_type == 'data':
        print("📊 ANALYZING DATA DIRECTORY")
        print("=" * 40)
        
        data_dir = Path(args.data_dir)
        
        if not data_dir.exists():
            print(f"❌ Data directory not found: {data_dir}")
            sys.exit(1)
        
        files = [f for f in data_dir.glob("*") if f.is_file()]
        if not files:
            print("❌ No files found in directory")
            return
        
        sizes = [f.stat().st_size for f in files]
        total_size = sum(sizes)
        
        print(f"📊 DIRECTORY ANALYSIS: {data_dir}")
        print(f"   Total files: {len(files)}")
        print(f"   Total size: {total_size / (1024**2):.1f}MB")
        print(f"   Average size: {total_size / len(files) / 1024:.1f}KB")
        print(f"   Largest file: {max(sizes) / 1024:.1f}KB")
        print(f"   Smallest file: {min(sizes) / 1024:.1f}KB")
        
        print(f"\n📈 SIZE DISTRIBUTION:")
        size_ranges = [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, float('inf'))]
        for min_kb, max_kb in size_ranges:
            count = sum(1 for s in sizes if min_kb*1024 <= s < max_kb*1024)
            if count > 0:
                range_str = f"{min_kb}KB-{max_kb}KB" if max_kb != float('inf') else f"{min_kb}KB+"
                print(f"   {range_str:>10}: {count:4d} files ({count/len(files)*100:.1f}%)")
        
        total_size_mb = total_size / (1024**2)
        if total_size_mb < 10:
            print(f"\n⚠️  WARNING: Dataset might be too small ({total_size_mb:.1f}MB)")
            print("   Recommended: At least 50-100MB for meaningful training")
        elif total_size_mb > 1000:
            print(f"\n✅ Large dataset detected ({total_size_mb:.0f}MB)")
            print("   Consider using data streaming or smaller context windows")
        else:
            print(f"\n✅ Good dataset size ({total_size_mb:.0f}MB)")


def handle_debug_command(args):
    """Handle the debug command."""
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
                from analysis.debug_vocab import main as debug_vocab_main
                debug_vocab_main()
            finally:
                os.chdir(original_dir)
                
        elif args.debug_type == 'model':
            # Change to model directory for debugging
            original_dir = Path.cwd()
            model_dir = Path(args.model_dir)
            if model_dir.exists():
                import os
                os.chdir(model_dir)
            
            try:
                from analysis.debug_model import main as debug_model_main
                debug_model_main()
            finally:
                os.chdir(original_dir)
                
        elif args.debug_type == 'sequences':
            from analysis.debug_long_sequences import debug_cuda_crash
            debug_cuda_crash()
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def handle_optimize_command(args):
    """Find optimal training parameters based on available GPU memory."""
    print(f"⚡ OPTIMIZE: TRAINING PARAMETERS")
    print("=" * 40)
    
    try:
        from analysis.calculate_context_window import find_max_seq_len, estimate_memory_usage
        import torch
        
        available_memory_gb = args.memory_gb
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(device)
            print(f"🎮 Detected GPU: {gpu_name} ({gpu_memory_gb:.1f}GB)")
            if args.memory_gb == 10.0:  # Default value, use detected
                available_memory_gb = min(gpu_memory_gb * 0.85, 10.0)  # Conservative 85% usage
        else:
            print("💻 Using CPU mode")
        
        print(f"🧠 Available memory: {available_memory_gb:.1f}GB")
        print(f"🎯 Target batch size: {args.batch_size}")
        
        max_context_window, memory_breakdown = find_max_seq_len(available_memory_gb, args.batch_size)
        
        print(f"\n✅ RECOMMENDATIONS:")
        print(f"   Context window: {max_context_window} tokens")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Memory usage: {memory_breakdown['total_memory_mb']:.0f}MB ({memory_breakdown['total_memory_mb']/1024:.1f}GB)")
        
        print(f"\n📊 MEMORY BREAKDOWN:")
        print(f"   Model parameters: {memory_breakdown['model_memory_mb']:.0f}MB")
        print(f"   Gradients: {memory_breakdown['gradient_memory_mb']:.0f}MB") 
        print(f"   Optimizer states: {memory_breakdown['optimizer_memory_mb']:.0f}MB")
        print(f"   Activations: {memory_breakdown['activation_memory_mb']:.0f}MB")
        print(f"   Total parameters: {memory_breakdown['total_params']:,}")
        
        print(f"\n⚙️  RECOMMENDED TRAINING COMMAND:")
        print(f"   python grey-lord.py train --batch-size {args.batch_size} --context-window {max_context_window}")
        
        if args.show_alternatives:
            print(f"\n📈 ALTERNATIVE CONFIGURATIONS:")
            for alt_batch in [1, 2, 4, 8, 16, 32]:
                if alt_batch == args.batch_size:
                    continue
                try:
                    alt_context, alt_memory = find_max_seq_len(available_memory_gb, alt_batch)
                    print(f"   Batch {alt_batch:2d}: context {alt_context:4d} tokens ({alt_memory['total_memory_mb']:.0f}MB)")
                except:
                    print(f"   Batch {alt_batch:2d}: too large for available memory")
                    break
                
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def handle_data_command(args):
    """Unified data processing command."""
    match args.data_sub_command:
        case "process":
            try:
                from training.process_agent_data import process_sessions
                process_sessions(args.session_dir, args.output_dir)
            except Exception as e:
                print(f"❌ Data processing failed: {e}")
                sys.exit(1)
        case "export":
            try:
                from training.export_model import export_model_to_gguf
                export_model_to_gguf(args.model_dir, args.output_file, args.llama_cpp_dir)
            except Exception as e:
                print(f"❌ Model export failed: {e}")
                sys.exit(1)
        case "copy":
            try:
                copy_data_files(args.source_dir, args.dataset_name)
            except Exception as e:
                print(f"❌ Data copy failed: {e}")
                sys.exit(1)
        case "prune":
            try:
                prune_data_files(args.dataset_name, args.min_size, getattr(args, 'pattern', None))
            except Exception as e:
                print(f"❌ Data prune failed: {e}")
                sys.exit(1)
        case _:
            print(f"❌ Unknown data command: {args.data_sub_command}")
            sys.exit(1)


class GreyLordArgParser(argparse.ArgumentParser):
    """Custom ArgumentParser to make things a little prettier."""

    usage_prefix = "📝"

    def format_usage(self):
        usage = super().format_usage().strip()
        return f"{self.usage_prefix} {usage}"

    def format_help(self):
        help_text = super().format_help()
        help_text = help_text.replace("usage:", f"{self.usage_prefix} usage:")
        return f"{help_text}\n"

    def error(self, message):
        sys.stdout.write(f"{self.format_usage()}\n")
        sys.stderr.write(f"\n❌ {message}\n\n")
        self.exit(2)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser for the application with branded output."""
    # Load config to provide defaults for arguments
    try:
        from training.config import load_config
        config = load_config()
        training_config = config.get("training", {})
        data_config = config.get("data", {})
    except (FileNotFoundError, ImportError):
        training_config = {}
        data_config = {}

    parser = GreyLordArgParser()
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    create_train_parser(subparsers, training_config)
    create_analyze_parser(subparsers)
    create_debug_parser(subparsers)
    create_optimize_parser(subparsers)
    create_data_parser(subparsers)

    return parser

def create_train_parser(subparsers, config):
    train_parser = subparsers.add_parser("train", help="Train or continue training a model")
    train_parser.add_argument("--dataset", type=str, default="./data", help="Name of the dataset to train on")
    train_parser.add_argument("--epochs", type=int, default=config.get("training.default_epochs", 10), help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=config.get("default_batch_size", 4), help="Training batch size")
    train_parser.add_argument("--learning-rate", type=float, default=config.get("default_lr", 5e-5), help="Learning rate")
    train_parser.add_argument("--context-window", type=int, default=config.get("default_max_seq_len", 512), help="Context window length (in tokens)")
    train_parser.add_argument("--model-path", type=str, default=None, help="Path to existing model to continue training from")
    train_parser.add_argument("--save-path", type=str, default=None, help="Directory to save the trained model")
    train_parser.add_argument("--val-split", type=float, default=config.get("default_val_split", 0.2), help="Fraction of data to use for validation")
    train_parser.add_argument("--patience", type=int, default=config.get("default_patience", 5), help="Number of epochs to wait for improvement before early stopping")
    train_parser.add_argument("--cpu", action="store_true", help="Force CPU training even if CUDA is available")


def create_analyze_parser(subparsers):
    analyze_parser = subparsers.add_parser("analyze", help="Analyze model performance or data")
    analyze_subparsers = analyze_parser.add_subparsers(dest='analyze_type', required=True)

    # Analyze Training
    analyze_training_parser = analyze_subparsers.add_parser('training', help='Analyze training history')
    analyze_training_parser.add_argument("--model-dir", type=str, required=True, help="Directory containing training_history.json")
    analyze_training_parser.add_argument("--output", type=str, default="analysis_report.png", help="Output file for analysis plots")

    # Analyze Data
    analyze_data_parser = analyze_subparsers.add_parser('data', help='Analyze data directory')
    analyze_data_parser.add_argument("--data-dir", type=str, required=True, help="Directory containing training data files")


def create_debug_parser(subparsers):
    debug_parser = subparsers.add_parser("debug", help="Debug parts of the system (vocabulary, model, sequences)")
    debug_subparsers = debug_parser.add_subparsers(dest='debug_type', required=True)

    # Debug Vocab
    debug_vocab_parser = debug_subparsers.add_parser('vocab', help='Debug vocabulary and tokenization')
    debug_vocab_parser.add_argument("--vocab-path", type=str, default="data", help="Path to vocabulary files (vocab.json, merges.txt)")

    # Debug Model
    debug_model_parser = debug_subparsers.add_parser('model', help='Debug a trained model')
    debug_model_parser.add_argument("--model-dir", type=str, required=True, help="Directory of the model to debug")

    # Debug Sequences
    debug_subparsers.add_parser('sequences', help='Debug long sequences for potential CUDA errors')


def create_optimize_parser(subparsers):
    optimize_parser = subparsers.add_parser("optimize", help="Find optimal training parameters for your hardware")
    optimize_parser.add_argument("--memory-gb", type=float, default=10.0, help="Available GPU memory in GB")
    optimize_parser.add_argument("--batch-size", type=int, default=4, help="Target batch size to optimize for")
    optimize_parser.add_argument("--show-alternatives", action="store_true", help="Show alternative batch sizes and context windows")


def create_data_parser(subparsers):
    data_parser = subparsers.add_parser("data", help="Manage data (process, copy, prune, export)")
    data_subparsers = data_parser.add_subparsers(dest='data_sub_command', required=True)

    copy_parser = data_subparsers.add_parser("copy", help="Copy files from source directory to data/ with flattened structure")
    copy_parser.add_argument("source_dir", type=str, help="Source directory (supports ~, globs, and various path formats)")
    copy_parser.add_argument("dataset_name", type=str, help="Name of the dataset to create (e.g., 'new_training_data')")

    prune_parser = data_subparsers.add_parser("prune", help="Remove files from dataset based on size and/or pattern criteria")
    prune_parser.add_argument("dataset_name", type=str, help="Name of dataset to use (e.g., 'new_training_data')")
    prune_parser.add_argument("--min-size", type=str, default="1", help="Minimum file size (e.g., '1024', '10KB', '5MB', '1GB')")
    prune_parser.add_argument("--pattern", type=str, help="Keep only files matching this pattern (e.g., '*.log', '*session*')")


def main() -> int:
    """Main entry point for the application."""
    setup_logging()
    check_virtual_environment()
    
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == 'train':
        # Lazy import to avoid heavy dependencies for simple commands
        from training.trainer import train_model
        return train_model(args)
    elif args.command == 'analyze':
        return handle_analyze_command(args)
    elif args.command == 'debug':
        return handle_debug_command(args)
    elif args.command == 'optimize':
        return handle_optimize_command(args)
    elif args.command == 'data':
        return handle_data_command(args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

