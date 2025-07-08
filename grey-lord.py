#!/usr/bin/env python3
import argparse
import sys
import logging
from pathlib import Path
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


def handle_help_command(args):
    """Handle the help command."""
    parser = create_parser()
    parser.print_help()
    return 0


def handle_data_command(args):
    """Unified data processing command."""
    match args.data_sub_command:
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
    parser = GreyLordArgParser()
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    create_help_parser(subparsers)
    create_data_parser(subparsers)

    return parser


def create_help_parser(subparsers):
    """Create the help subcommand parser."""
    help_parser = subparsers.add_parser("help", help="Show help information")


def create_data_parser(subparsers):
    """Create the data subcommand parser."""
    data_parser = subparsers.add_parser("data", help="Manage data (copy, prune)")
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
    
    if args.command == 'help':
        return handle_help_command(args)
    elif args.command == 'data':
        return handle_data_command(args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

