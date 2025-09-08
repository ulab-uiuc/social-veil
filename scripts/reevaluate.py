import argparse
import os
import json
import sys
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from social_decipher.evaluate import ConversationEvaluator

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs/config.yaml")
with open(CONFIG_PATH, "r") as f:
    _config = yaml.safe_load(f)
os.environ["OPENAI_API_KEY"] = _config.get("OPENAI_API_KEY") 

def reevaluate_scenario(scenario_dir: str, evaluator: ConversationEvaluator, mode_name: str):
    """Loads a scenario's data, re-runs evaluation, and overwrites the result."""
    convo_log_path = os.path.join(scenario_dir, "conversation_log.json")
    if not os.path.exists(convo_log_path):
        print(f"  - Skipping, conversation_log.json not found.")
        return

    try:
        with open(convo_log_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)

        # Extract necessary data from the log
        context = log_data.get("experimental_context", {})
        agents_context = context.get("agents", {})
        agent_a_context = agents_context.get("agent_a", {})
        agent_b_context = agents_context.get("agent_b", {})
        
        conversation = log_data.get("conversation_log", [])
        agent_goals = [agent_a_context.get("goal"), agent_b_context.get("goal")]
        agent_reasons = [agent_a_context.get("reason"), agent_b_context.get("reason")]
        scenario = context.get("scenario", {}).get("description", "")
        mcq_logs = log_data.get("mcq_logs")
        
        # First, try to get barrier_type from the log (for newer logs)
        barrier_type = context.get("scenario", {}).get("barrier_type")

        # If not found, infer from the directory name for backwards compatibility
        if barrier_type is None:
            if "semantic" in mode_name:
                barrier_type = "semantic_structure"
            elif "cultural" in mode_name:
                barrier_type = "cultural_style"
            elif "emotional" in mode_name:
                barrier_type = "emotional_influence"

        # Re-run evaluation. Note: The `scenario` text is not passed here
        # because the original `evaluate_conversation` does not use it, but
        # the prompt in `evaluation.yaml` *does*. The evaluator loads the yaml
        # but only formats the fields it has, so this will run without error.

        new_evaluation_result = evaluator.evaluate_conversation(
            conversation,
            agent_goals=agent_goals,
            agent_reasons=agent_reasons,
            mcq_logs=mcq_logs,
            barrier_type=barrier_type,
        )

        # Detect if the evaluation failed (e.g., bad API key returns null scores)
        # and do not overwrite the existing file if it did.
        is_failed_result = (
            new_evaluation_result is None or
            new_evaluation_result.get("aggregated_scores", {})
            .get("episode_level", {})
            .get("unresolved_confusion") is None
        )

        if is_failed_result:
            print(f"  - Evaluation failed for {os.path.basename(scenario_dir)}. Skipping file write.")
            return

        # Overwrite the old evaluation file
        result_path = os.path.join(scenario_dir, "eval_result.json")
        with open(result_path, "w") as f:
            json.dump(new_evaluation_result, f, indent=4)
            
    except Exception as e:
        print(f"  - FAILED to re-evaluate: {e}")
        pass


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate conversation logs without re-running simulation.")
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Path to the parent experiment results directory (e.g., 'results/exp_...')."
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="Model to use for conversation evaluation."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Error: Directory not found at {args.results_dir}")
        return

    print(f"🔎 Scanning experiment directory: {args.results_dir}")
    print(f"🤖 Using evaluator model: {args.model}")

    evaluator = ConversationEvaluator(args.model)

    # Find all 'mode_*' directories within the results directory
    mode_dirs = [d.path for d in os.scandir(args.results_dir) if d.is_dir() and d.name.startswith("mode_")]

    # If no mode directories are found, assume the user passed a single mode directory
    if not mode_dirs:
        print("No 'mode_*' subdirectories found. Assuming current directory is a single mode.")
        mode_dirs.append(args.results_dir)

    total_scenarios_found = 0
    for mode_dir in sorted(mode_dirs):
        mode_name = os.path.basename(mode_dir)
        print(f"\n--- Processing Mode: {mode_name} ---")
        
        scenario_dirs = [d.path for d in os.scandir(mode_dir) if d.is_dir() and d.name.startswith("scenario_")]
        
        if not scenario_dirs:
            print(f"No 'scenario_*' directories found in {mode_name}.")
            continue

        total_scenarios_found += len(scenario_dirs)
        print(f"📊 Found {len(scenario_dirs)} scenarios to re-evaluate in this mode.")
        
        for scenario_dir in tqdm(sorted(scenario_dirs), desc=f"Re-evaluating {mode_name}"):
            reevaluate_scenario(scenario_dir, evaluator, mode_name)

    if total_scenarios_found == 0:
        print("\nNo scenarios were found to re-evaluate in any mode.")
    else:
        print(f"\n✅ Re-evaluation complete. Processed a total of {total_scenarios_found} scenarios across all modes.")


if __name__ == "__main__":
    main()