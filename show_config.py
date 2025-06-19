#!/usr/bin/env python3
"""Display the current configuration"""

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
