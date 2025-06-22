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
import json
from datetime import datetime


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





def handle_analyze_command(args):
    """Unified analysis command for training results and data."""
    sys.path.insert(0, str(Path(__file__).parent / "analysis"))
    
    if args.analyze_type == 'training':
        print("📊 ANALYZING TRAINING RESULTS")
        print("=" * 40)
        print(f"Training directory: {args.training_dir}")
        
        try:
            from analyze_training import analyze_overfitting, plot_training_curves
            
            training_dir = Path(args.training_dir)
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
                
                if args.output != "analysis_report.png":
                    output_path = args.output
                    if not output_path.endswith(('.png', '.pdf', '.svg')):
                        output_path = f"{output_path}.{args.format}"
                    
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
            
            if args.output != "analysis_report.png":
                output_path = args.output
                if not output_path.endswith(('.png', '.pdf', '.svg')):
                    output_path = f"{output_path}.{args.format}"
                
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
    """Find optimal training parameters based on available GPU memory."""
    sys.path.insert(0, str(Path(__file__).parent / "analysis"))
    
    print(f"⚡ OPTIMIZE: TRAINING PARAMETERS")
    print("=" * 40)
    
    try:
        from calculate_context_window import find_max_seq_len, estimate_memory_usage
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
        
        print(f"\n⚙️  COMMAND TO USE:")
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
    """Unified data management: prepare, analyze, and process training data."""
    print(f"📁 DATA: {args.data_type.upper()}")
    print("=" * 40)
    
    if args.data_type == 'prepare':
        source_dir = Path(args.source)
        target_dir = Path(args.target)
        
        if not source_dir.exists():
            print(f"❌ Source directory not found: {source_dir}")
            sys.exit(1)
        
        target_dir.mkdir(exist_ok=True)
        
        import shutil
        files_processed = 0
        total_size = 0
        
        print(f"🔄 Processing files from {source_dir}")
        print(f"   Min size: {args.min_size_kb}KB")
        print(f"   Action: {'Move' if args.move else 'Copy'}")
        
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
        
        print(f"\n✅ RESULTS:")
        print(f"   Files processed: {files_processed}")
        print(f"   Total size: {total_size / (1024**2):.1f}MB")
        print(f"   Target directory: {target_dir}")
    
    elif args.data_type == 'analyze':
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
    
    elif args.data_type == 'process':
        print("🤖 PROCESSING AGENT SESSIONS")
        
        try:
            sys.path.insert(0, str(Path(__file__).parent / "training"))
            from process_agent_data import SessionDataProcessor
            
            session_dir = Path(args.session_dir)
            output_dir = Path(args.output_dir)
            
            if not session_dir.exists():
                print(f"❌ Session directory not found: {session_dir}")
                sys.exit(1)
            
            processor = SessionDataProcessor(session_dir)
            num_sessions = processor.load_sessions()
            
            if num_sessions == 0:
                print("❌ No session data found")
                sys.exit(1)
            
            report = processor.generate_training_report()
            print(f"\n📊 SESSION ANALYSIS:")
            print(f"   Sessions: {report['sessions']}")
            print(f"   Total interactions: {report['total_interactions']}")
            print(f"   AI commands: {report['ai_commands']}")
            print(f"   Server responses: {report['server_responses']}")
            print(f"   Avg interactions/session: {report['data_quality']['avg_interactions_per_session']:.1f}")
            
            print(f"\n🚀 GENERATING TRAINING DATA:")
            processor.create_continued_training_data(output_dir / "continued")
            processor.create_behavioral_cloning_data(output_dir / "behavioral") 
            processor.create_rl_experience_replay(output_dir / "rl")
            
            print(f"\n✅ Agent data processed successfully")
            print(f"   Output directory: {output_dir}")
            
        except Exception as e:
            print(f"❌ Agent processing failed: {e}")
            sys.exit(1)


def handle_agent_command(args):
    """Handle agent operations."""
    if args.agent_type == 'start':
        return start_agent_client(args.config)
    else:
        print(f"❌ Unknown agent command: {args.agent_type}")
        sys.exit(1)


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
    """Handle all model-related operations: train, analyze, debug, optimize, manage."""
    sys.path.insert(0, str(Path(__file__).parent / "training"))
    sys.path.insert(0, str(Path(__file__).parent / "analysis"))
    
    if args.model_type == 'train':
        print("🚀 TRAINING MODEL")
        print("=" * 30)
        
        try:
            from trainer import ModelTrainer
            from config_utils import get_training_config, get_data_config
            
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
    
    elif args.model_type == 'analyze':
        print("📊 ANALYZING TRAINING RESULTS")
        print("=" * 40)
        print(f"Training directory: {args.training_dir}")
        
        try:
            from analyze_training import analyze_overfitting, plot_training_curves
            
            training_dir = Path(args.training_dir)
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
                
                if args.output != "analysis_report.png":
                    output_path = args.output
                    if not output_path.endswith(('.png', '.pdf', '.svg')):
                        output_path = f"{output_path}.{args.format}"
                    
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
            
            if args.output != "analysis_report.png":
                output_path = args.output
                if not output_path.endswith(('.png', '.pdf', '.svg')):
                    output_path = f"{output_path}.{args.format}"
                
                import shutil
                shutil.copy2(plots_output_path, output_path)
                print(f"📋 Analysis plots also copied to: {output_path}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            sys.exit(1)
    
    elif args.model_type == 'debug':
        if not args.debug_type:
            print("❌ Please specify a debug type: vocab, model, or sequences")
            sys.exit(1)
        
        print(f"🔍 DEBUG: {args.debug_type.upper()}")
        print("=" * 30)
        
        try:
            if args.debug_type == 'vocab':
                original_dir = Path.cwd()
                vocab_path = Path(args.vocab_path)
                if vocab_path.exists():
                    import os
                    os.chdir(vocab_path)
                
                try:
                    from debug_vocab import main as debug_vocab_main
                    debug_vocab_main()
                finally:
                    os.chdir(original_dir)
                    
            elif args.debug_type == 'model':
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
    
    elif args.model_type == 'optimize':
        print("⚙️  OPTIMIZING TRAINING PARAMETERS")
        print("=" * 40)
        
        try:
            from memory_optimizer import find_max_seq_len, get_gpu_memory
            
            available_memory_gb = args.memory_gb
            if available_memory_gb <= 0:
                available_memory_gb = get_gpu_memory()
                if available_memory_gb <= 0:
                    available_memory_gb = 10.0
                    print(f"⚠️  Could not detect GPU memory, using default: {available_memory_gb}GB")
                else:
                    print(f"🔍 Detected GPU memory: {available_memory_gb:.1f}GB")
            else:
                print(f"🎯 Using specified memory: {available_memory_gb}GB")
            
            max_context_window, memory_breakdown = find_max_seq_len(available_memory_gb, args.batch_size)
            
            print(f"\n✅ OPTIMAL CONFIGURATION:")
            print(f"   Context window: {max_context_window} tokens")
            print(f"   Batch size: {args.batch_size}")
            print(f"   Memory usage: {memory_breakdown['total_memory_mb']:.0f}MB ({memory_breakdown['total_memory_mb']/1024:.1f}GB)")
            
            print(f"\n📊 MEMORY BREAKDOWN:")
            print(f"   Model parameters: {memory_breakdown['model_memory_mb']:.0f}MB")
            print(f"   Gradients: {memory_breakdown['gradient_memory_mb']:.0f}MB") 
            print(f"   Optimizer states: {memory_breakdown['optimizer_memory_mb']:.0f}MB")
            print(f"   Activations: {memory_breakdown['activation_memory_mb']:.0f}MB")
            print(f"   Total parameters: {memory_breakdown['total_params']:,}")
            
            print(f"\n⚙️  COMMAND TO USE:")
            print(f"   python grey-lord.py model train --batch-size {args.batch_size} --context-window {max_context_window}")
            
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
    
    elif args.model_type == 'list':
        print("📂 AVAILABLE MODELS")
        print("=" * 30)
        
        try:
            from model_manager import get_model_directories
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
                    
                    dir_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                    size_str = f"{dir_size / (1024**2):.1f}MB"
                    
                    print(f"  {i:2d}. {model_dir.name} (val_loss: {val_loss_str}, size: {size_str})")
                except:
                    print(f"  {i:2d}. {model_dir.name} (unable to read details)")
        except ImportError as e:
            print(f"❌ Model management utilities not available: {e}")
            sys.exit(1)
    
    elif args.model_type == 'leaderboard':
        try:
            from model_manager import create_model_leaderboard
            create_model_leaderboard()
        except ImportError as e:
            print(f"❌ Model management utilities not available: {e}")
            sys.exit(1)
    
    elif args.model_type == 'compare':
        try:
            from model_manager import compare_models
            compare_models(args.models)
        except ImportError as e:
            print(f"❌ Model management utilities not available: {e}")
            sys.exit(1)
    
    elif args.model_type == 'cleanup':
        try:
            from model_manager import cleanup_old_models
            cleanup_old_models(keep_count=args.keep, dry_run=args.dry_run)
        except ImportError as e:
            print(f"❌ Model management utilities not available: {e}")
            sys.exit(1)
    
    elif args.model_type == 'links':
        try:
            from model_manager import create_model_links
            create_model_links()
        except ImportError as e:
            print(f"❌ Model management utilities not available: {e}")
            sys.exit(1)
    
    elif args.model_type == 'export':
        try:
            from model_manager import export_model_for_deployment
            export_model_for_deployment(args.model_dir, args.output)
        except ImportError as e:
            print(f"❌ Model management utilities not available: {e}")
            sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="Grey Lord - MajorMUD AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Commands:
  data      Data preparation, analysis, and processing
  model     Model training, analysis, debugging, and management  
  agent     AI agent operations and session analysis
  config    Configuration management

For detailed help on each command:
  python grey-lord.py <command> --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add training configuration dynamically for model commands
    try:
        sys.path.insert(0, str(Path(__file__).parent / "training"))
        from config_utils import get_training_config, get_data_config
        training_config = get_training_config()
        data_config = get_data_config()
    except ImportError:
        training_config = {}
        data_config = {}
    
    # Data command - unified data management
    data_parser = subparsers.add_parser('data', help='Data preparation, analysis, and processing')
    data_subparsers = data_parser.add_subparsers(dest='data_type', help='Data management tools')
    
    # Data preparation
    prep_parser = data_subparsers.add_parser('prepare', help='Prepare and filter training data files')
    prep_parser.add_argument("--source", type=str, required=True,
                            help="Source directory containing raw files")
    prep_parser.add_argument("--target", type=str, default="training_data",
                            help="Target directory for prepared data")
    prep_parser.add_argument("--min-size-kb", type=int, default=1,
                            help="Minimum file size in KB")
    prep_parser.add_argument("--move", action="store_true",
                            help="Move files instead of copying")
    
    # Data analysis
    data_analyze_parser = data_subparsers.add_parser('analyze', help='Analyze data directory')
    data_analyze_parser.add_argument("--data-dir", type=str, required=True,
                                   help="Directory containing data files to analyze")
    
    # Process agent sessions  
    process_parser = data_subparsers.add_parser('process', help='Process agent session data for training')
    process_parser.add_argument("--session-dir", type=str, default="data/agent_sessions",
                               help="Directory containing agent session data")
    process_parser.add_argument("--output-dir", type=str, default="data/processed_agent_data",
                               help="Output directory for processed training data")
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_type', help='Configuration tools')
    
    # Show configuration
    show_parser = config_subparsers.add_parser('show', help='Show current configuration')
    
    # Validate configuration
    validate_parser = config_subparsers.add_parser('validate', help='Validate configuration file')
    
    # Model command - unified model operations
    model_parser = subparsers.add_parser('model', help='Model training, analysis, debugging, and management')
    model_subparsers = model_parser.add_subparsers(dest='model_type', help='Model operations')
    
    # Model training
    train_parser = model_subparsers.add_parser('train', help='Train or continue training a model')
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
    train_parser.add_argument("--data-dir", type=str, 
                           default=data_config.get("default_data_dir", "./data"),
                           help="Directory containing training data")
    train_parser.add_argument("--file-glob", type=str, 
                           default=data_config.get("default_file_glob", "*"),
                           help="Glob pattern to match training files")
    train_parser.add_argument("--model-path", type=str, default=None,
                           help="Path to existing model to continue training from")
    train_parser.add_argument("--save-path", type=str, default=None,
                           help="Directory to save the trained model")
    train_parser.add_argument("--val-split", type=float, 
                           default=training_config.get("default_val_split", 0.2),
                           help="Fraction of data to use for validation")
    train_parser.add_argument("--patience", type=int, 
                           default=training_config.get("default_patience", 5),
                           help="Number of epochs to wait for improvement before early stopping")
    train_parser.add_argument("--cpu", action="store_true",
                           help="Force CPU training even if CUDA is available")
    
    # Model analysis
    analyze_parser = model_subparsers.add_parser('analyze', help='Analyze training results')
    analyze_parser.add_argument("--training-dir", type=str, required=True,
                               help="Directory containing training artifacts to analyze")
    analyze_parser.add_argument("--output", type=str, default="analysis_report.png",
                               help="Output file for analysis plots")
    analyze_parser.add_argument("--format", type=str, default="png",
                               choices=["png", "pdf", "svg", "html"],
                               help="Output format")
    
    # Model debugging
    debug_parser = model_subparsers.add_parser('debug', help='Debug model, vocab, or data issues')
    debug_subparsers = debug_parser.add_subparsers(dest='debug_type', help='Debug tools')
    
    vocab_parser = debug_subparsers.add_parser('vocab', help='Debug vocabulary and model compatibility')
    vocab_parser.add_argument("--vocab-path", type=str, default="data/",
                             help="Path to vocabulary files")
    
    model_debug_parser = debug_subparsers.add_parser('model', help='Debug model configuration and architecture')
    model_debug_parser.add_argument("--model-path", type=str, required=True,
                                   help="Path to model to debug")
    
    seq_parser = debug_subparsers.add_parser('sequences', help='Debug long sequence issues')
    seq_parser.add_argument("--data-dir", type=str, default="training_data",
                           help="Training data directory")
    seq_parser.add_argument("--max-length", type=int, default=8192,
                           help="Maximum sequence length to analyze")
    
    # Model optimization
    optimize_parser = model_subparsers.add_parser('optimize', help='Find optimal training parameters')
    optimize_parser.add_argument("--batch-size", type=int, default=4,
                               help="Target batch size to optimize for")
    optimize_parser.add_argument("--memory-gb", type=float, default=10.0,
                               help="Available GPU memory in GB (auto-detected if available)")
    optimize_parser.add_argument("--show-alternatives", action="store_true",
                               help="Show alternative batch size configurations")
    
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
    
    # Agent command - AI agent operations
    agent_parser = subparsers.add_parser('agent', help='AI agent operations and session analysis')
    agent_subparsers = agent_parser.add_subparsers(dest='agent_type', help='Agent operations')
    
    # Start agent
    start_parser = agent_subparsers.add_parser('start', help='Start AI agent telnet client')
    start_parser.add_argument('--config', default='agent_config.json', 
                             help='Configuration file path (default: agent_config.json)')
    
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
        if args.command == 'data':
            handle_data_command(args)
            return 0
        elif args.command == 'model':
            handle_model_command(args)
            return 0
        elif args.command == 'agent':
            return handle_agent_command(args)
        elif args.command == 'config':
            handle_config_command(args)
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

