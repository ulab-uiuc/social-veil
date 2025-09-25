import json
import argparse
from typing import List, Dict, Any, Set

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Reads a JSONL file and returns a list of dictionaries."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line: {e}")
    return rows

def main():
    """Main function to extract and print unique scenarios."""
    parser = argparse.ArgumentParser(description="Print all unique scenarios from a JSONL file.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input JSONL file (e.g., data/episode_all_neutralized.jsonl)"
    )
    args = parser.parse_args()

    episodes = read_jsonl(args.input_file)
    unique_scenarios: Set[str] = set()
    unique_code_names: Set[str] = set()

    for episode in episodes:
        scenario = episode.get("scenario")
        agent_goal = episode.get("agent_goals")
        if scenario and isinstance(scenario, str):
            unique_scenarios.add(scenario.strip() + " " + "Agent A Goal: " + agent_goal[0].strip() + " " + "Agent B Goal: " + agent_goal[1].strip())
    # for episode in episodes:
    #     code_name = episode.get("codename")
    #     print(code_name)
        # unique_code_names.add(code_name)

    print("--- Unique Scenarios ---")
    for i, scenario in enumerate(sorted(list(unique_scenarios)), 1):
        print(f"{i}. {scenario}")
    print(f"\nFound {len(unique_code_names)} unique scenarios.")
    
    # save the unique scenarios to a file
    with open("unique_scenarios.txt", "w") as f:
        for i, scenario in enumerate(sorted(list(unique_scenarios)), 1):
            f.write(f"{i}. {scenario}\n")


if __name__ == "__main__":
    main()