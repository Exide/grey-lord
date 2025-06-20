#!/usr/bin/env python3
"""
Prepare training data from raw telnet log files.
This script copies/moves files that meet size requirements to a training directory.
"""

import sys
from pathlib import Path

# Add src directory to path to find config_utils
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import shutil
import sys
from pathlib import Path
from typing import List
from config_utils import get_data_config

def get_file_size_str(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    if size_bytes >= 1024**3:  # GB
        return f"{size_bytes / (1024**3):.1f}GB"
    elif size_bytes >= 1024**2:  # MB
        return f"{size_bytes / (1024**2):.1f}MB"
    elif size_bytes >= 1024:  # KB
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes}B"

def find_training_files(source_dir: Path, file_glob: str, min_size_kb: int = 1) -> tuple[List[Path], List[Path]]:
    """Find training files and separate them into qualifying and too-small categories."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    print(f"[info] Searching for files matching pattern '{file_glob}' in {source_dir}")
    all_files = sorted(source_dir.glob(file_glob))
    
    if not all_files:
        raise FileNotFoundError(f"No files found matching pattern '{file_glob}' in {source_dir}")
    
    print(f"[info] Found {len(all_files)} files matching pattern")
    
    min_size_bytes = min_size_kb * 1024
    qualifying_files = []
    small_files = []
    
    for file_path in all_files:
        try:
            file_size = file_path.stat().st_size
            if file_size > min_size_bytes:
                qualifying_files.append(file_path)
            else:
                small_files.append(file_path)
        except OSError as e:
            print(f"[warning] Could not get size of {file_path}: {e}")
            continue
    
    return qualifying_files, small_files

def prepare_training_data(source_dir: str, target_dir: str = "training_data", 
                         file_glob: str = None, min_size_kb: int = 1, 
                         move_instead_of_copy: bool = False) -> None:
    """Main function to prepare training data."""
    
    # Use default glob pattern if not provided
    if file_glob is None:
        data_config = get_data_config()
        file_glob = data_config["default_file_glob"]
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    print("=" * 60)
    print("🧹 PREPARING TRAINING DATA")
    print("=" * 60)
    print(f"📁 Source directory: {source_path}")
    print(f"📂 Target directory: {target_path}")
    print(f"🔍 File pattern: {file_glob}")
    print(f"📏 Minimum file size: {min_size_kb}KB")
    print(f"🔄 Operation: {'Move' if move_instead_of_copy else 'Copy'} (source files will {'be removed' if move_instead_of_copy else 'remain intact'})")
    print("=" * 60)
    
    # Create target directory
    target_path.mkdir(exist_ok=True)
    print(f"[info] Created/verified target directory: {target_path}")
    
    # Find files
    try:
        qualifying_files, small_files = find_training_files(source_path, file_glob, min_size_kb)
    except FileNotFoundError as e:
        print(f"[error] {e}")
        sys.exit(1)
    
    # Report findings
    print(f"\n📊 **File Analysis**")
    print(f"• Total files found: {len(qualifying_files) + len(small_files)}")
    print(f"• Qualifying files (>{min_size_kb}KB): {len(qualifying_files)}")
    print(f"• Too small files (≤{min_size_kb}KB): {len(small_files)}")
    
    if small_files:
        print(f"\n⚠️  **Files too small (will be skipped):**")
        for file_path in small_files[:10]:  # Show first 10
            size = file_path.stat().st_size
            print(f"   • {file_path.name} ({get_file_size_str(size)})")
        if len(small_files) > 10:
            print(f"   ... and {len(small_files) - 10} more")
    
    if not qualifying_files:
        print(f"[error] No files meet the minimum size requirement!")
        sys.exit(1)
    
    # Calculate total data size
    total_size = 0
    for file_path in qualifying_files:
        total_size += file_path.stat().st_size
    
    print(f"\n📈 **Qualifying Data Summary**")
    print(f"• Files to process: {len(qualifying_files)}")
    print(f"• Total data size: {get_file_size_str(total_size)}")
    
    # Confirm operation
    operation = "move" if move_instead_of_copy else "copy"
    print(f"\n❓ Ready to {operation} {len(qualifying_files)} files to {target_path}")
    response = input("Continue? (y/N): ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("[info] Operation cancelled by user")
        return
    
    # Process files
    print(f"\n🚀 **Processing Files**")
    processed = 0
    failed = 0
    
    for i, file_path in enumerate(qualifying_files, 1):
        try:
            target_file = target_path / file_path.name
            
            # Handle file name conflicts
            counter = 1
            original_name = target_file.stem
            suffix = target_file.suffix
            while target_file.exists():
                target_file = target_path / f"{original_name}_{counter}{suffix}"
                counter += 1
            
            if move_instead_of_copy:
                shutil.move(str(file_path), str(target_file))
            else:
                shutil.copy2(file_path, target_file)
            
            processed += 1
            
            if processed % 10 == 0 or processed == len(qualifying_files):
                print(f"   Processed {processed}/{len(qualifying_files)} files...")
                
        except Exception as e:
            print(f"[error] Failed to process {file_path.name}: {e}")
            failed += 1
    
    # Final report
    print(f"\n✅ **Operation Complete**")
    print(f"• Successfully processed: {processed} files")
    if failed > 0:
        print(f"• Failed: {failed} files")
    print(f"• Training data ready in: {target_path}")
    
    # Show some example files in target directory
    target_files = list(target_path.glob("*"))[:5]
    if target_files:
        print(f"\n📁 **Sample files in training_data:**")
        for file_path in target_files:
            size = file_path.stat().st_size
            print(f"   • {file_path.name} ({get_file_size_str(size)})")
        if len(target_files) == 5 and len(list(target_path.glob("*"))) > 5:
            print(f"   ... and more")

def main():
    parser = argparse.ArgumentParser(description="Prepare training data by moving/copying files")
    parser.add_argument("source_dir", type=str,
                       help="Source directory containing training files")
    parser.add_argument("--target-dir", type=str, default="training_data",
                       help="Target directory for cleaned training data")
    parser.add_argument("--file-glob", type=str, default=None,
                       help="File pattern to match (default: from config)")
    parser.add_argument("--min-size-kb", type=int, default=1,
                       help="Minimum file size in KB (default: 1)")
    parser.add_argument("--move", action="store_true",
                       help="Move files instead of copying them")
    
    args = parser.parse_args()
    
    try:
        prepare_training_data(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            file_glob=args.file_glob,
            min_size_kb=args.min_size_kb,
            move_instead_of_copy=args.move
        )
    except Exception as e:
        print(f"[error] Failed to prepare training data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
 