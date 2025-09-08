import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

def calculate_and_plot_averages(file_path: str, output_dir: str):
    """
    Calculates and prints averages and also plots the distribution of scores.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file was not found at '{file_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Data Aggregation (for both tables and plots) ---
    aggregated_scores = defaultdict(lambda: defaultdict(list))
    profile_scores = defaultdict(lambda: defaultdict(list))

    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                source = data.get('source')
                barrier = data.get('barrier_type') or 'baseline'
                accuracy = data.get('answer_accuracy')
                profile_id = data.get('problem_id', '').split('_')[0]

                if source and barrier and accuracy is not None:
                    aggregated_scores[source][barrier].append(float(accuracy))
                    if profile_id:
                        profile_key = (profile_id, barrier)
                        profile_scores[source][profile_key].append(float(accuracy))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

    # --- 2. Print Aggregated Table ---
    print_aggregated_table(aggregated_scores)

    # --- 3. Print Per-Profile Table ---
    print_per_profile_table(profile_scores)
    
    # --- 4. Generate and Save Plots ---
    plot_distributions(profile_scores, output_dir)


def print_aggregated_table(scores):
    # ... (code to print the main summary table)
    header = ["Benchmark", "Barrier Type", "Average Accuracy", "Samples"]
    rows = []
    for source, barrier_data in sorted(scores.items()):
        for barrier_type, accuracies in sorted(barrier_data.items()):
            num_samples = len(accuracies)
            if num_samples > 0:
                avg_accuracy = np.mean(accuracies)
                barrier_name = barrier_type.replace('_', ' ').title()
                rows.append([source.upper(), barrier_name, f"{avg_accuracy:.2%}", num_samples])
    
    print("\n" + "=" * 80)
    print("Aggregated Performance by Barrier Type")
    print("=" * 80)
    print_formatted_table(rows, header)

def print_per_profile_table(scores):
    # ... (code to print the per-profile table)
    print("\n\n" + "=" * 80)
    print("Average Performance by Individual Agent Profile")
    print("=" * 80)
    profile_rows = []
    for source, profiles_data in sorted(scores.items()):
        for (profile_id, barrier_type), accuracies in sorted(profiles_data.items()):
            num_samples = len(accuracies)
            if num_samples > 0:
                avg_accuracy = np.mean(accuracies)
                barrier_name = barrier_type.replace('_', ' ').title()
                profile_rows.append([source.upper(), profile_id, barrier_name, f"{avg_accuracy:.2%}", num_samples])
    
    if profile_rows:
        profile_header = ["Benchmark", "Profile ID", "Barrier Type", "Average Accuracy", "Samples"]
        print_formatted_table(profile_rows, header=profile_header)
    else:
        print("No per-profile data found.")

def plot_distributions(profile_scores, output_dir):
    # ... (code for plotting)
    plot_data = []
    for source, profiles_data in profile_scores.items():
        for (profile_id, barrier_type), accuracies in profiles_data.items():
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                plot_data.append({
                    "benchmark": source.upper(),
                    "barrier_type": barrier_type.replace('_', ' ').title(),
                    "average_accuracy": avg_accuracy
                })

    if not plot_data:
        print("\nNo data available to plot.")
        return

    df = pd.DataFrame(plot_data)
    benchmarks = df["benchmark"].unique()
    
    print("\n\n" + "=" * 80)
    print("Generating Performance Distribution Plots...")
    print("=" * 80)

    for benchmark in benchmarks:
        plt.figure(figsize=(12, 8))
        subset_df = df[df["benchmark"] == benchmark]
        sns.violinplot(
            data=subset_df, x="average_accuracy", y="barrier_type",
            orient="h", order=sorted(subset_df["barrier_type"].unique()),
            palette="viridis", inner="quartiles", scale="width"
        )
        plt.title(f"Distribution of Agent Profile Performance\nBenchmark: {benchmark}", fontsize=16, fontweight='bold')
        plt.xlabel("Average Answer Accuracy", fontsize=12)
        plt.ylabel("Barrier Type", fontsize=12)
        plt.xlim(0, 1)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'performance_distribution_{benchmark.lower()}.png')
        plt.savefig(output_path, dpi=300)
        print(f"✅ Saved plot to: {output_path}")
        plt.close()

def print_formatted_table(rows: list, header: list):
    # ... (existing helper function to print tables)
    if not rows:
        print("No data to display.")
        return
    
    col_widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    table_width = sum(col_widths) + len(col_widths) * 3 + 1
    print("=" * table_width)
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(header, col_widths))
    print(f"| {header_line} |")
    separator = "-+-".join("-" * w for w in col_widths)
    print(f"|-{separator}-|")
    for row in rows:
        row_line = " | ".join(f"{str(c):<{w}}" for c, w in zip(row, col_widths))
        print(f"| {row_line} |")
    print("=" * table_width)

if __name__ == "__main__":
    default_file_path = 'results/incremental_results.jsonl'
    default_output_dir = 'results/'
    calculate_and_plot_averages(default_file_path, default_output_dir)

    
