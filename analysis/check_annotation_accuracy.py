import argparse
import pandas as pd


def main():
    """Main function to run the annotation accuracy check."""
    parser = argparse.ArgumentParser(
        description="Check the accuracy of a single human annotation file."
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        required=True,
        help="Path to the single 'human_*_annotations.json' file to check.",
    )
    args = parser.parse_args()

    try:
        df = pd.read_json(args.annotation_file)
    except Exception as e:
        print(
            f"Error: Could not read or process file {args.annotation_file}. Details: {e}"
        )
        return

    # --- Basic Validation ---
    assessment_col = (
        "human_barrier_assessment (semantic/cultural/emotional/none/unclear)"
    )
    required_cols = ["barrier_type", assessment_col]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: The file is missing one of the required columns: {required_cols}")
        return

    # --- Accuracy Calculation ---
    df["assessment_normalized"] = df[assessment_col].str.lower().str.strip()

    def is_correct(row: pd.Series) -> bool:
        """Determines if an annotation is correct."""
        if row["barrier_type"] == "baseline":
            return row["assessment_normalized"] == "none"
        return row["assessment_normalized"] == row["barrier_type"]

    df["is_correct"] = df.apply(is_correct, axis=1)

    print(f"Analyzing accuracy for: {args.annotation_file}\n")
    print("--- Accuracy per Barrier Type ---")

    barrier_types = sorted(df["barrier_type"].unique())
    total_correct = 0
    total_annotations = 0

    for b_type in barrier_types:
        type_df = df[df["barrier_type"] == b_type]
        if type_df.empty:
            continue

        correct_count = type_df["is_correct"].sum()
        total_count = len(type_df)
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

        print(f"  Mode: {b_type.capitalize()}")
        print(f"    - Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")

        total_correct += correct_count
        total_annotations += total_count

    # --- Overall & Baseline Distribution ---
    overall_accuracy = (
        (total_correct / total_annotations) * 100 if total_annotations > 0 else 0
    )
    print("\n--- Overall ---")
    print(f"  - Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_annotations})")

    baseline_df = df[df["barrier_type"] == "baseline"]
    if not baseline_df.empty:
        print("\n--- Baseline Assessment Distribution ---")
        baseline_counts = (
            baseline_df["assessment_normalized"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )
        for assessment, pct in baseline_counts.items():
            print(f"    - Assessed as '{assessment}': {pct}%")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()