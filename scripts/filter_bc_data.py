import json
import argparse
import os

def get_score(score_value):
    """Handles cases where score is a dict like {'score': 5} or a direct value."""
    if isinstance(score_value, dict) and 'score' in score_value:
        return score_value['score']
    if isinstance(score_value, (int, float)):
        return score_value
    return 0  # Default to 0 if score is missing or in an unexpected format

def filter_conversations(input_path, output_path, min_confusion, min_understanding):
    """
    Filters conversations based on 'unresolved_confusion' and 'mutual_understanding' scores.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_path}")
        return

    filtered_convos = []
    for convo in conversations:
        eval_data = convo.get('eval_result', {})
        episode_scores = eval_data.get('aggregated_scores', {}).get('episode_level', {})
        
        confusion_score = get_score(episode_scores.get('unresolved_confusion'))
        understanding_score = get_score(episode_scores.get('mutual_understanding'))
        
        if confusion_score > min_confusion and understanding_score > min_understanding:
            filtered_convos.append(convo)
            
    print(f"Original conversations: {len(conversations)}")
    print(f"Filtered conversations: {len(filtered_convos)}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_convos, f, indent=2)
        
    print(f"✅ Filtered data saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Filter bc_data.json based on evaluation scores.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="training_data/bc_data.json",
        help="Path to the input bc_data.json file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="training_data/bc_data_filtered.json",
        help="Path to save the filtered output JSON file."
    )
    parser.add_argument(
        "--min_confusion",
        type=float,
        default=3.0,
        help="Minimum score for 'unresolved_confusion'."
    )
    parser.add_argument(
        "--min_understanding",
        type=float,
        default=3.0,
        help="Minimum score for 'mutual_understanding'."
    )
    args = parser.parse_args()

    filter_conversations(args.input_file, args.output_file, args.min_confusion, args.min_understanding)

if __name__ == "__main__":
    main()