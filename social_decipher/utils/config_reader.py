#!/usr/bin/env python3
import os
import sys
import json

try:
    import yaml
except Exception as e:
    print(f"ERROR: Missing dependency 'pyyaml': {e}", file=sys.stderr)
    sys.exit(1)


def load_config() -> dict:
    # This file is at social_decipher/utils/, project root is two levels up
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    cfg_path = os.path.join(project_root, "configs", "config.yaml")
    if not os.path.isfile(cfg_path):
        print(f"ERROR: Config not found at {cfg_path}", file=sys.stderr)
        sys.exit(2)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_by_path(d: dict, path: str):
    cur = d
    for part in path.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m social_decipher.utils.config_reader <dotted.key.path>", file=sys.stderr)
        sys.exit(3)
    key_path = sys.argv[1]
    cfg = load_config()
    val = get_by_path(cfg, key_path)
    if val is None:
        print("", end="")
        print(f"WARN: Key '{key_path}' not found", file=sys.stderr)
        return
    if isinstance(val, (str, int, float)):
        print(val)
    else:
        print(json.dumps(val))


if __name__ == "__main__":
    main()

