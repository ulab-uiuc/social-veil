import argparse
import glob
import os
import json
import pandas as pd
import simpledorff
from typing import Dict, Any, Optional


def load_human_annotations(input_dir: str) -> Optional[pd.DataFrame]:
    """Loads all annotation JSONs and combines them into a single DataFrame."""
    json_files = glob.glob(os.path.join(input_dir, "human_*_annotations.json"))
    if not json_files:
        print(f"Error: No 'human_*_annotations.json' files found in directory: {input_dir}")
        return None
    
    all_dfs = [pd.read_json(f) for f in json_files]
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Coerce score columns to numeric, setting errors to NaN
    for col in ["human_unresolved_confusion (1-5)", "human_mutual_understanding (1-5)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.rename(columns={
        "human_unresolved_confusion (1-5)": "human_unresolved_confusion",
        "human_mutual_understanding (1-5)": "human_mutual_understanding",
        "human_barrier_assessment (semantic/cultural/emotional/none/unclear)": "human_barrier_assessment"
    }, inplace=True)
    
    return df

def get_automated_score(base_dir: str, conversation_id: str) -> Optional[Dict[str, float]]:
    """Loads the automated eval_result.json for a given conversation_id."""
    try:
        mode, scenario_folder = conversation_id.split('/')
        file_path = os.path.join(base_dir, f"mode_{mode}", scenario_folder, 'eval_result.json')
        if not os.path.exists(file_path): return None

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        ep_level = data.get("aggregated_scores", {}).get("episode_level", {})
        if not isinstance(ep_level, dict): return None

        scores = {}
        for metric in ["unresolved_confusion", "mutual_understanding"]:
            val = ep_level.get(metric)
            if isinstance(val, (int, float)):
                scores[f"auto_{metric}"] = float(val)
            elif isinstance(val, dict) and "score" in val:
                scores[f"auto_{metric}"] = float(val["score"])
        return scores if len(scores) == 2 else None
    except Exception:
        return None

# --- Main Calculation Functions ---

def calculate_accuracy(df: pd.DataFrame) -> Dict[str, float]:
    """Calculates barrier identification accuracy."""
    accuracy_results = {}
    # Treat 'none' as the correct assessment for 'baseline'
    df['is_correct'] = (
        (df['barrier_type'] == 'baseline') & (df['human_barrier_assessment'] == 'none')
    ) | (df['barrier_type'] == df['human_barrier_assessment'])
    
    grouped = df.groupby('barrier_type')['is_correct'].agg(['sum', 'count'])
    for barrier_type, row in grouped.iterrows():
        # Skip baseline as it doesn't fit the ACC model of the table
        if barrier_type == "baseline":
            continue
        accuracy = (row['sum'] / row['count']) * 100 if row['count'] > 0 else 0
        accuracy_results[barrier_type] = round(accuracy, 2)
        
    return accuracy_results

def calculate_alignment(human_df: pd.DataFrame, auto_dir: str) -> Dict[str, Dict[str, float]]:
    """Calculates Pearson correlation between human and auto scores per barrier type."""
    # Average human scores per conversation
    human_avg_scores = human_df.groupby('conversation_id').agg({
        'human_unresolved_confusion': 'mean',
        'human_mutual_understanding': 'mean',
        'barrier_type': 'first'
    }).reset_index()

    # Get automated scores
    auto_scores_list = [get_automated_score(auto_dir, cid) for cid in human_avg_scores['conversation_id']]
    auto_scores_df = pd.DataFrame([s for s in auto_scores_list if s is not None], 
                                  index=human_avg_scores[pd.notna(auto_scores_list)].index)

    # Merge and calculate correlation
    merged_df = pd.concat([human_avg_scores, auto_scores_df], axis=1).dropna()
    
    alignment_results = {}
    for barrier_type, group in merged_df.groupby('barrier_type'):
        if barrier_type == 'baseline' or len(group) < 2:
            continue
        
        corr_confusion = group['human_unresolved_confusion'].corr(group['auto_unresolved_confusion'])
        corr_understanding = group['human_mutual_understanding'].corr(group['auto_mutual_understanding'])
        
        # Taking the average of the two correlations for the final Align. ρ̄ value
        avg_corr = (corr_confusion + corr_understanding) / 2
        alignment_results[barrier_type] = round(avg_corr, 2)
        
    return alignment_results

def calculate_irr(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates Krippendorff's Alpha for the barrier assessment task."""
    try:
        alpha_result = simpledorff.calculate_krippendorffs_alpha_for_df(
            df,
            experiment_col="conversation_id",
            annotator_col="annotator_id",
            class_col="human_barrier_assessment",
        )
        # simpledorff can return a float or a DataFrame depending on version/context
        if isinstance(alpha_result, pd.DataFrame):
            kappa = alpha_result.iloc[0]["krippendorffs_alpha"]
        else:
            kappa = alpha_result # Assume it's the float value

        return {"kappa": round(kappa, 2)}
    except Exception as e:
        return {"kappa": f"ERROR ({e})"}


def main():
    parser = argparse.ArgumentParser(description="Generate a summary table of human evaluation metrics.")
    parser.add_argument("--human_dir", type=str, required=True, help="Directory with 'human_*_annotations.json' files.")
    parser.add_argument("--auto_dir", type=str, required=True, help="Base results directory with 'mode_*' subfolders.")
    args = parser.parse_args()

    df = load_human_annotations(args.human_dir)
    if df is None:
        return

    # Calculate all metrics
    accuracies = calculate_accuracy(df)
    alignments = calculate_alignment(df, args.auto_dir)
    irr = calculate_irr(df)

    # Prepare data for the table
    barrier_types = ['semantic', 'cultural', 'emotional']
    table_data = []

    for bt in barrier_types:
        # Use 'cultural' for 'Sociocultural' mapping
        table_name = 'Sociocultural' if bt == 'cultural' else bt.capitalize()
        
        acc = accuracies.get(bt, 'N/A')
        align = alignments.get(bt, 'N/A')
        
        # IRR is calculated overall, but in the example table it's on one line.
        # We'll place it on the 'Sociocultural' line to match the example.
        irr_val = irr['kappa'] if bt == 'cultural' else ''
        
        table_data.append({
            "Barrier Type": table_name,
            "Acc. (%)": acc,
            "Align. ρ̄": align,
            "IRR κ": irr_val
        })

    # Print the table
    summary_df = pd.DataFrame(table_data)
    print("\n--- Human Evaluation Summary (Table 3) ---")
    print(summary_df.to_string(index=False))
    
    # Calculate and print overall Pearson correlation for Table 4
    merged_df = pd.concat([human_avg_scores, auto_scores_df], axis=1).dropna()
    if not merged_df.empty and len(merged_df) > 1:
        corr_confusion_overall = merged_df['human_unresolved_confusion'].corr(merged_df['auto_unresolved_confusion'])
        corr_understanding_overall = merged_df['human_mutual_understanding'].corr(merged_df['auto_mutual_understanding'])
        n_overall = len(merged_df)
        
        print("\n--- Human vs. Automated Evaluation (Table 4 Pearson) ---")
        print(f"Conf. r={corr_confusion_overall:.3f}, Mutual r={corr_understanding_overall:.3f} (n={n_overall})")

    print("\nNotes:")
    print(" - 'Acc. (%)' is the barrier identification agreement rate.")
    print(" - 'Align. ρ̄' is the avg. Pearson correlation between human & automated scores (confusion and understanding) *within each barrier type*.")
    print(" - 'IRR κ' is the overall Krippendorff's Alpha for the barrier assessment task.")


if __name__ == "__main__":
    main()