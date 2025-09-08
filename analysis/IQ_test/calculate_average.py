import json
import os
import numpy as np
from collections import defaultdict

def calculate_averages(file_path: str):
    """
    Calculates and prints the average answer accuracy from the math test
    results file, grouped by source and barrier type.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file was not found at '{file_path}'")
        print("Please make sure you have run the math evaluation script first.")
        return

    # Use defaultdict for easier initialization
    scores = defaultdict(lambda: defaultdict(list))

    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                source = data.get('source')
                barrier = data.get('barrier_type') or 'baseline'
                accuracy = data.get('answer_accuracy')

                if source and barrier and accuracy is not None:
                    scores[source][barrier].append(float(accuracy))

            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"Warning: Skipping malformed line. Error: {e}. Line: {line.strip()}")
                continue

    # Prepare data for a single table
    header = ["Benchmark", "Barrier Type", "Average Accuracy", "Samples"]
    rows = []

    for source, barrier_data in sorted(scores.items()):
        for barrier_type, accuracies in sorted(barrier_data.items()):
            num_samples = len(accuracies)
            if num_samples > 0:
                avg_accuracy = np.mean(accuracies)
                barrier_name = barrier_type.replace('_', ' ').title()
                rows.append([
                    source.upper(),
                    barrier_name,
                    f"{avg_accuracy:.2%}",
                    num_samples
                ])

    if not rows:
        print("No valid data found to calculate averages.")
        return

    # Calculate column widths for formatting
    col_widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Print the formatted table
    print("=" * (sum(col_widths) + len(col_widths) * 3 + 1))
    # Print header
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(header, col_widths))
    print(f"| {header_line} |")
    # Print separator
    separator = "-+-".join("-" * w for w in col_widths)
    print(f"|-{separator}-|")
    # Print rows
    for row in rows:
        row_line = " | ".join(f"{str(c):<{w}}" for c, w in zip(row, col_widths))
        print(f"| {row_line} |")
    print("=" * (sum(col_widths) + len(col_widths) * 3 + 1))


if __name__ == "__main__":
    default_file_path = 'results/incremental_results.jsonl'
    calculate_averages(default_file_path)

    
