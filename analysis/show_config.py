#!/usr/bin/env python3
"""Show current configuration"""

import sys
from pathlib import Path

# Add src directory to path to find config_utils
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config_utils import print_config_summary, load_config
import json

def main():
    try:
        config = load_config()
        print_config_summary()
        
        print("\n=== Full Configuration ===")
        print(json.dumps(config, indent=2))
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure model_config.json exists in the current directory")
    except Exception as e:
        print(f"Error loading configuration: {e}")

if __name__ == "__main__":
    main()
