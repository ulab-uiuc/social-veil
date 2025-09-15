import argparse
import glob
import os
import json
from typing import Dict, Any

import numpy as np
import pandas as pd
import simpledorff


def load_and_validate_data(input_dir: str) -> pd.DataFrame:
    """Loads all annotation JSONs, validates them, and combines them into a single DataFrame."""
    json_files = glob.glob(os.path.join(input_dir, "human_*_annotations.json"))
    if not json_files:
        raise FileNotFoundError(f"No 'human_*_annotations.json' files found in directory: {input_dir}")

    all_dfs = []
    for f in json_files:
        try:
            df = pd.read_json(f)
            # Basic validation
            if "conversation_id" not in df.columns or "annotator_id" not in df.columns:
                print(f"WARNING: Skipping file {f} due to missing required columns.")
                continue
            all_dfs.append(df)
        except Exception as e:
            print(f"WARNING: Could not read or process file {f}. Error: {e}")

    if not all_dfs:
        raise ValueError("No valid annotation data could be loaded.")

    return pd.concat(all_dfs, ignore_index=True)


def calculate_metric_averages(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates the average scores for the two primary metrics, grouped by barrier type."""
    metrics = ["human_unresolved_confusion (1-5)", "human_mutual_understanding (1-5)"]
    results = {}

    # Ensure metric columns are numeric, coercing errors to NaN
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    grouped = df.groupby("barrier_type")[metrics].mean()

    for barrier_type, row in grouped.iterrows():
        results[barrier_type] = {
            "avg_unresolved_confusion": round(row[metrics[0]], 2),
            "avg_mutual_understanding": round(row[metrics[1]], 2),
            "num_annotations": df[df["barrier_type"] == barrier_type][metrics[0]].notna().sum(),
        }
    return results


def calculate_barrier_agreement(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates the proportion of times annotators correctly identified the intended barrier."""
    results = {}
    # Filter out baseline as it has no "correct" barrier
    barrier_df = df[df["barrier_type"] != "baseline"].copy()
    
    # Normalize the assessment column for easier matching
    barrier_df["assessment_normalized"] = barrier_df["human_barrier_assessment (semantic/cultural/emotional/none/unclear)"].str.lower().str.strip()

    # Agreement is when the intended barrier type matches the human assessment
    correct_identifications = barrier_df[barrier_df["assessment_normalized"] == barrier_df["barrier_type"]]
    
    agreement_by_type = correct_identifications.groupby("barrier_type").size()
    total_by_type = barrier_df.groupby("barrier_type").size()

    for barrier_type, total_count in total_by_type.items():
        correct_count = agreement_by_type.get(barrier_type, 0)
        agreement_rate = (correct_count / total_count) * 100 if total_count > 0 else 0
        results[barrier_type] = {
            "agreement_rate_percent": round(agreement_rate, 2),
            "correct_identifications": correct_count,
            "total_annotations": total_count,
        }
    
    # Also show what baseline conversations were classified as
    baseline_df = df[df["barrier_type"] == "baseline"]
    baseline_counts = baseline_df["human_barrier_assessment (semantic/cultural/emotional/none/unclear)"].value_counts(normalize=True).mul(100).round(2)
    results["baseline_assessment_distribution"] = baseline_counts.to_dict()

    return results


def calculate_inter_annotator_agreement(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates Krippendorff's Alpha for all human-rated columns."""
    results = {}
    annotation_columns = [
        "human_unresolved_confusion (1-5)",
        "human_mutual_understanding (1-5)",
        "human_barrier_assessment (semantic/cultural/emotional/none/unclear)",
    ]

    for col in annotation_columns:
        # simpledorff requires numeric data for interval metrics
        if "(1-5)" in col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            level_of_measurement = "interval"
        else:
            level_of_measurement = "nominal"

        try:
            alpha_df = simpledorff.calculate_krippendorffs_alpha_for_df(
                df,
                experiment_col="conversation_id",
                annotator_col="annotator_id",
                class_col=col,
                level_of_measurement=level_of_measurement
            )
            # Extract the alpha value and round it
            alpha_value = round(alpha_df.iloc[0]["krippendorffs_alpha"], 3)
            results[f"krippendorff_alpha_{col.split(' ')[0]}"] = alpha_value
        except Exception as e:
            results[f"krippendorff_alpha_{col.split(' ')[0]}"] = f"ERROR: Could not calculate ({e})"

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze and aggregate results from human evaluation.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the completed 'human_*.csv' annotation files.")
    args = parser.parse_args()

    print("Loading and validating annotation data...")
    try:
        df = load_and_validate_data(args.input_dir)
        print(f"Successfully loaded {df.shape[0]} total annotations from {df['annotator_id'].nunique()} files.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    print("\n--- 1. Average Scores per Metric ---")
    metric_averages = calculate_metric_averages(df)
    for mode, data in metric_averages.items():
        print(f"  Mode: {mode.capitalize()}")
        print(f"    - Avg. Unresolved Confusion: {data['avg_unresolved_confusion']}")
        print(f"    - Avg. Mutual Understanding: {data['avg_mutual_understanding']}")

    print("\n--- 2. Barrier Identification Agreement ---")
    barrier_agreement = calculate_barrier_agreement(df)
    for mode, data in barrier_agreement.items():
        if mode == "baseline_assessment_distribution":
            continue
        print(f"  Mode: {mode.capitalize()}")
        print(f"    - Agreement Rate: {data['agreement_rate_percent']}% ({data['correct_identifications']}/{data['total_annotations']})")
    print("\n  Baseline Mode Annotation Distribution:")
    for assessment, pct in barrier_agreement.get("baseline_assessment_distribution", {}).items():
        print(f"    - Assessed as '{assessment}': {pct}%")
        
    print("\n--- 3. Inter-Annotator Agreement (Krippendorff's Alpha) ---")
    iaa_results = calculate_inter_annotator_agreement(df)
    for metric, alpha in iaa_results.items():
        print(f"  - {metric}: {alpha}")
    print("(Note: Alpha > 0.67 is considered acceptable, > 0.8 is good)")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()