"""
Utility functions that support Grey Lord's data sub-commands.
Extracted into a dedicated module (data/api.py).
"""

from __future__ import annotations

import sys
import re
import os
import glob
import shutil
import fnmatch
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

__all__ = [
    "parse_size_to_kb",
    "copy_data_files",
    "prune_data_files",
    "log_dataset_change",
]


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_dataset_change(dataset_path: Path, action: str, details: str) -> None:
    """Log a change to the dataset's change.log file.
    
    Args:
        dataset_path: Path to the dataset directory
        action: Type of action (e.g., 'COPY', 'PRUNE')
        details: Description of what was done
    """
    log_file = dataset_path / "change.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {action}: {details}\n"
    
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
            f.flush()
        print(f"📝 Logged {action} to {log_file}")
    except Exception as exc:
        print(f"⚠️  Failed to write to change.log: {exc}")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_size_to_kb(size_str: str) -> int:
    """Convert a human-friendly size string to kilobytes.

    Examples
    --------
    >>> parse_size_to_kb("10KB")
    10
    >>> parse_size_to_kb("5mb")
    5120
    >>> parse_size_to_kb("2048")  # bytes assumed
    2
    """
    size_str = size_str.strip().lower()
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([kmg]?b?)?$", size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")

    number_str, unit = match.groups()
    number = float(number_str)

    if not unit or unit == "b":
        return max(1, int(number / 1024))
    if unit in {"k", "kb"}:
        return max(1, int(number))
    if unit in {"m", "mb"}:
        return int(number * 1024)
    if unit in {"g", "gb"}:
        return int(number * 1024 * 1024)
    raise ValueError(f"Unsupported unit: {unit}")


# ---------------------------------------------------------------------------
# Copy helpers
# ---------------------------------------------------------------------------

def copy_data_files(source_dir: str, target_name: str) -> None:
    """Copy files from *source_dir* into *data/target_name* (flattened)."""
    print()  # new line for clarity
    print("📁 COPYING DATA FILES")
    print("=" * 30)

    expanded = os.path.expanduser(os.path.expandvars(source_dir))

    # Accept glob patterns
    if any(ch in expanded for ch in "*?[]"):
        matches = glob.glob(expanded)
        if not matches:
            print(f"❌ No paths found matching pattern: {source_dir}")
            sys.exit(1)
        dirs = [p for p in matches if os.path.isdir(p)]
        if not dirs:
            print(f"❌ No directories found matching pattern: {source_dir}")
            sys.exit(1)
        if len(dirs) > 1:
            print(f"📂 Found {len(dirs)} directories matching pattern:")
            for idx, d in enumerate(dirs, 1):
                print(f"   {idx}. {d}")
            print(f"   Using first directory: {dirs[0]}")
        source_path = Path(dirs[0])
    else:
        source_path = Path(expanded)

    source_path = source_path.resolve()
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_path}")
        sys.exit(1)
    if not source_path.is_dir():
        print(f"❌ Source path is not a directory: {source_path}")
        sys.exit(1)

    target_path = Path("data") / target_name
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"📂 Source: {source_path}")
    print(f"📂 Target: {target_path}")

    all_files = [Path(root) / f for root, _, files in os.walk(source_path) for f in files]
    if not all_files:
        print("❌ No files found in source directory")
        sys.exit(1)

    print(f"📄 Found {len(all_files)} files to copy")

    copied = skipped = 0
    for fp in all_files:
        try:
            dest = target_path / fp.name
            if dest.exists():
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest = target_path / f"{fp.stem}_{ts}{fp.suffix}"
            shutil.copy2(fp, dest)
            copied += 1
            if copied % 100 == 0:
                print(f"   📋 Copied {copied}/{len(all_files)} files…")
        except Exception as exc:
            print(f"⚠️  Skipped {fp.name}: {exc}")
            skipped += 1

    print("\n✅ Copy completed!")
    print(f"   📄 Files copied: {copied}")
    if skipped:
        print(f"   ⚠️  Files skipped: {skipped}")

    total_size = sum(p.stat().st_size for p in target_path.glob("*") if p.is_file())
    print(f"   📊 Target directory: {len(list(target_path.glob('*')))} files, {total_size / (1024**2):.1f}MB")
    print(f"   📂 Location: {target_path}")

    # Log the copy operation
    log_details = f"Copied {copied} files from {source_path} to {target_path}"
    if skipped:
        log_details += f" (skipped {skipped} files)"
    log_details += f" - Total size: {total_size / (1024**2):.1f}MB"
    log_dataset_change(target_path, "COPY", log_details)


# ---------------------------------------------------------------------------
# Prune helpers
# ---------------------------------------------------------------------------

def prune_data_files(dataset_name: str, min_size_str: str, pattern: Optional[str] = None) -> None:
    """Remove files from *data/dataset_name* that don't match criteria."""
    print()
    print("🧹 PRUNING DATA FILES")
    print("=" * 30)

    try:
        min_kb = parse_size_to_kb(min_size_str)
    except ValueError as exc:
        print(f"❌ Invalid size format: {exc}")
        sys.exit(1)
    min_bytes = min_kb * 1024

    dataset_path = Path("data") / dataset_name
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)
    if not dataset_path.is_dir():
        print(f"❌ Dataset path is not a directory: {dataset_path}")
        sys.exit(1)

    size_disp = (
        f"{min_bytes / (1024 ** 3):.1f}GB" if min_bytes >= 1024 ** 3 else
        f"{min_bytes / (1024 ** 2):.1f}MB" if min_bytes >= 1024 ** 2 else
        f"{min_bytes / 1024:.1f}KB" if min_bytes >= 1024 else
        f"{min_bytes}B"
    )

    print(f"📂 Dataset: {dataset_path}")
    print(f"📏 Minimum size: {size_disp}")
    if pattern:
        print(f"🔍 Pattern: {pattern} (keep only matching files)")

    all_files = [p for p in dataset_path.glob("*") if p.is_file() and p.name != "change.log"]
    if not all_files:
        print("❌ No files found in dataset")
        sys.exit(1)

    print(f"📄 Found {len(all_files)} files to analyze")

    to_prune: list[Tuple[Path, int]] = []
    to_keep: list[Tuple[Path, int]] = []

    for fp in all_files:
        size = fp.stat().st_size
        keep = size >= min_bytes and (not pattern or fnmatch.fnmatch(fp.name, pattern))
        (to_keep if keep else to_prune).append((fp, size))

    if not to_prune:
        print("✅ No files to prune! All files meet the criteria.")
        return

    total_remove = sum(s for _, s in to_prune)
    total_keep = sum(s for _, s in to_keep)

    print("\n📊 PRUNING ANALYSIS:")
    print(f"   🗑️  Files to remove: {len(to_prune)} ({total_remove / (1024**2):.1f}MB)")
    print(f"   ✅ Files to keep: {len(to_keep)} ({total_keep / (1024**2):.1f}MB)")

    if pattern:
        size_only = sum(1 for fp, s in to_prune if s < min_bytes and fnmatch.fnmatch(fp.name, pattern))
        pattern_only = sum(1 for fp, s in to_prune if s >= min_bytes and not fnmatch.fnmatch(fp.name, pattern))
        both = len(to_prune) - size_only - pattern_only
        print("\n📋 REMOVAL REASONS:")
        if size_only:
            print(f"   📏 Size too small: {size_only} files")
        if pattern_only:
            print(f"   🔍 Pattern mismatch: {pattern_only} files")
        if both:
            print(f"   📏🔍 Both size & pattern: {both} files")

    print("\n📋 FILES TO BE REMOVED:")
    for fp, s in to_prune[:5]:
        reason = []
        if s < min_bytes:
            reason.append(f"size={s / 1024:.1f}KB")
        if pattern and not fnmatch.fnmatch(fp.name, pattern):
            reason.append("pattern mismatch")
        print(f"   • {fp.name} ({', '.join(reason)})")
    if len(to_prune) > 5:
        print(f"   • … and {len(to_prune) - 5} more files")

    print(f"\n⚠️  This will permanently delete {len(to_prune)} files!")
    if input("Continue? (y/N): ").strip().lower() not in {"y", "yes"}:
        print("❌ Operation cancelled")
        sys.exit(0)

    removed = failed = 0
    for fp, _ in to_prune:
        try:
            os.remove(fp)
            removed += 1
            if removed % 50 == 0:
                print(f"   🗑️  Removed {removed}/{len(to_prune)} files…")
        except Exception as exc:
            print(f"⚠️  Failed to remove {fp.name}: {exc}")
            failed += 1

    print("\n✅ Pruning completed!")
    print(f"   🗑️  Files removed: {removed}")
    if failed:
        print(f"   ❌ Failed to remove: {failed}")
    print(f"   💾 Space freed: {total_remove / (1024**2):.1f}MB")

    remaining = [p for p in dataset_path.glob("*") if p.is_file()]
    remaining_size = sum(p.stat().st_size for p in remaining)
    print(f"   📊 Final dataset: {len(remaining)} files, {remaining_size / (1024**2):.1f}MB")

    # Log the pruning operation
    log_details = f"Pruned {removed} files (min_size={size_disp}"
    if pattern:
        log_details += f", pattern={pattern}"
    log_details += f") - Space freed: {total_remove / (1024**2):.1f}MB"
    if failed:
        log_details += f" (failed to remove {failed} files)"
    log_details += f" - Final dataset: {len(remaining)} files, {remaining_size / (1024**2):.1f}MB"
    log_dataset_change(dataset_path, "PRUNE", log_details)
