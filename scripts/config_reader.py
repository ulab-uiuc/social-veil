#!/usr/bin/env python3
"""
Utility script to read configuration values from config.yaml for shell scripts.
Usage: python config_reader.py <key_path>
Example: python config_reader.py models.model_a
"""

import os
import sys
import yaml
from pathlib import Path

def get_config_value(key_path: str):
    """Get a configuration value using dot notation (e.g., 'models.model_a')"""
    
    # Get the config file path relative to this script
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "configs" / "config.yaml"
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}", file=sys.stderr)
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Navigate through nested keys using dot notation
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                print(f"Error: Key '{key_path}' not found in config", file=sys.stderr)
                return None

        if isinstance(value, str) and not os.path.isabs(value) and (
            '/' in value or 
            value.endswith(('.jinja', '.json', '.yaml', '.yml', '.txt', '.py'))
        ):
            project_root = script_dir.parent
            value = str(project_root / value)
            
        return value
    
    except Exception as e:
        print(f"Error reading config: {e}", file=sys.stderr)
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python config_reader.py <key_path>", file=sys.stderr)
        print("Example: python config_reader.py models.model_a", file=sys.stderr)
        sys.exit(1)
    
    key_path = sys.argv[1]
    value = get_config_value(key_path)
    
    if value is not None:
        print(value)
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()