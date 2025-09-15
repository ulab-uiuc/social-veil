import argparse
import json
import os
import glob
import random
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import pandas as pd


def find_conversation_files(base_dir: str, mode: str, scenario_idx: int) -> Tuple[str, str]:
    """Finds the conversation log and eval result file for a given scenario."""
    scenario_dir = os.path.join(base_dir, f"mode_{mode}", f"scenario_{scenario_idx}")
    conv_log_path = os.path.join(scenario_dir, "conversation_log.txt")
    eval_result_path = os.path.join(scenario_dir, "eval_result.json")

    if not os.path.exists(conv_log_path) or not os.path.exists(eval_result_path):
        raise FileNotFoundError(f"Files for {mode}/scenario_{scenario_idx} not found in {scenario_dir}")

    return conv_log_path, eval_result_path


def load_episodes(episodes_file: str) -> List[Dict[str, Any]]:
    """Loads the original episodes file to get context."""
    if not os.path.exists(episodes_file):
        raise FileNotFoundError(f"Episodes file not found: {episodes_file}")

    with open(episodes_file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_task_data(
    mode: str, scenario_idx: int, base_dir: str, episodes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Extracts all necessary data for a single annotation task."""
    conv_log_path, eval_result_path = find_conversation_files(base_dir, mode, scenario_idx)

    with open(conv_log_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    with open(eval_result_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # The episode data is 0-indexed, while scenario folders are 1-indexed
    episode_data = episodes[scenario_idx - 1]

    ep_level = eval_data.get("aggregated_scores", {}).get("episode_level", {})
    unresolved_confusion = ep_level.get("unresolved_confusion", {}).get("score", "")
    mutual_understanding = ep_level.get("mutual_understanding", {}).get("score", "")

    return {
        "conversation_id": f"{mode}/scenario_{scenario_idx}",
        "barrier_type": mode,
        "scenario": episode_data.get("scenario", "N/A"),
        "agent_a_profile": episode_data.get("agent1_profile", "N/A"),
        "agent_b_profile": episode_data.get("agent2_profile", "N/A"),
        "agent_a_goal": episode_data.get("agent_goals", ["N/A"])[0],
        "agent_b_goal": episode_data.get("agent_goals", ["N/A", "N/A"])[1],
        "transcript": transcript,
        "ai_unresolved_confusion": unresolved_confusion,
        "ai_mutual_understanding": mutual_understanding,
    }


def assign_tasks(
    tasks: List[Dict[str, Any]], num_annotators: int, annotations_per_convo: int
) -> Dict[str, List[Dict[str, Any]]]:
    """Assigns tasks to annotators ensuring balanced workloads."""
    assignments = defaultdict(list)
    annotator_pool = [f"human_{i+1}" for i in range(num_annotators)]

    # Create a list of all required annotations
    all_annotation_slots = []
    for task in tasks:
        all_annotation_slots.extend([task] * annotations_per_convo)
    
    random.shuffle(all_annotation_slots)

    # Distribute shuffled slots to annotators in a round-robin fashion
    for i, task in enumerate(all_annotation_slots):
        annotator_id = annotator_pool[i % num_annotators]
        # Avoid assigning the same task twice to the same annotator if possible
        # This check isn't perfect for all combos of annotators/reps but works for 6/3
        if task not in assignments[annotator_id]:
            assignments[annotator_id].append(task)
        else:
            # If a duplicate occurs, find the next available annotator
            for j in range(1, num_annotators):
                next_annotator_id = annotator_pool[(i + j) % num_annotators]
                if task not in assignments[next_annotator_id]:
                    assignments[next_annotator_id].append(task)
                    break

    return assignments


def main():
    parser = argparse.ArgumentParser(description="Prepare data for human evaluation.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base results directory containing mode_* subfolders.")
    parser.add_argument("--episodes_file", type=str, required=True, help="The .jsonl file used to generate the conversations, for context.")
    parser.add_argument("--output_dir", type=str, default="./human_evaluation", help="Directory to save the annotation files.")
    parser.add_argument("--samples_per_mode", type=int, default=30, help="Number of conversations to sample from each mode.")
    parser.add_argument("--num_annotators", type=int, default=6, help="Total number of human annotators.")
    parser.add_argument("--annotations_per_convo", type=int, default=3, help="Number of annotators required for each conversation.")
    parser.add_argument("--score_threshold", type=float, default=3.0, help="For barrier modes, only sample conversations where ai_mutual_understanding is at or below this value.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading episodes from {args.episodes_file}...")
    episodes = load_episodes(args.episodes_file)
    print(f"Loaded {len(episodes)} episodes.")

    all_tasks = []
    modes = ["baseline", "semantic", "cultural", "emotional"]

    for mode in modes:
        print(f"Sampling from mode: {mode}...")
        scenario_dirs = glob.glob(os.path.join(args.base_dir, f"mode_{mode}", "scenario_*"))
        
        if not scenario_dirs:
            print(f"  WARNING: No scenario directories found for mode '{mode}'. Skipping.")
            continue

        candidate_indices = []
        if mode == "baseline":
            # For baseline, no score filtering is needed
            candidate_indices = [int(os.path.basename(d).split("_")[1]) for d in scenario_dirs]
        else:
            # For barrier modes, filter by score threshold
            print(f"  Filtering scenarios with mutual_understanding <= {args.score_threshold}...")
            for scenario_dir in scenario_dirs:
                try:
                    idx = int(os.path.basename(scenario_dir).split("_")[1])
                    _, eval_path = find_conversation_files(args.base_dir, mode, idx)
                    with open(eval_path, "r", encoding="utf-8") as f:
                        eval_data = json.load(f)
                    
                    mu_score = eval_data.get("aggregated_scores", {}).get("episode_level", {}).get("mutual_understanding", {}).get("score")
                    
                    if mu_score is not None and float(mu_score) <= args.score_threshold:
                        candidate_indices.append(idx)
                except (FileNotFoundError, ValueError, KeyError) as e:
                    print(f"    - Could not process {scenario_dir} for filtering. Skipping. Error: {e}")
                    continue
            print(f"  Found {len(candidate_indices)} valid candidates for sampling.")

        
        if len(candidate_indices) < args.samples_per_mode:
            print(f"  WARNING: Found only {len(candidate_indices)} valid scenarios, less than the requested {args.samples_per_mode}. Using all available.")
            sampled_indices = candidate_indices
        else:
            sampled_indices = random.sample(candidate_indices, args.samples_per_mode)
        
        print(f"  Selected {len(sampled_indices)} scenarios.")

        for idx in sampled_indices:
            try:
                task_data = extract_task_data(mode, idx, args.base_dir, episodes)
                all_tasks.append(task_data)
            except FileNotFoundError as e:
                print(f"  ERROR: Could not process {mode}/scenario_{idx}. Details: {e}")

    print(f"\nTotal tasks prepared: {len(all_tasks)}")
    
    if not all_tasks:
        print("No tasks were prepared. Exiting.")
        return

    print("Assigning tasks to annotators...")
    assignments = assign_tasks(all_tasks, args.num_annotators, args.annotations_per_convo)

    output_columns = [
        "conversation_id",
        "annotator_id",
        "barrier_type",
        "scenario",
        "agent_a_profile",
        "agent_b_profile",
        "agent_a_goal",
        "agent_b_goal",
        "transcript",
        "ai_unresolved_confusion",
        "ai_mutual_understanding",
        "human_unresolved_confusion (1-5)",
        "human_mutual_understanding (1-5)",
        "human_barrier_assessment (semantic/cultural/emotional/none/unclear)",
        "human_barrier_assessment_reasoning",
        "human_confidence (1-5)",
        "notes",
    ]

    print("Generating annotation files...")
    for annotator_id, tasks in assignments.items():
        df = pd.DataFrame(tasks)
        df["annotator_id"] = annotator_id
        
        # Add empty columns for human input
        for col in output_columns:
            if col not in df.columns:
                df[col] = ""
        
        # Reorder columns
        df = df[output_columns]

        output_path = os.path.join(args.output_dir, f"{annotator_id}.csv")
        df.to_csv(output_path, index=False)
        print(f"  - Saved worksheet for {annotator_id} with {len(df)} tasks to {output_path}")

    print("\nHuman evaluation data preparation complete.")


if __name__ == "__main__":
    main()