import json
import os
from collections import defaultdict
from typing import Dict, List

# Define experiment folders and tags
folders = [
    './exp_no_encryption_no_action',
    './exp_mapping_encryption_no_action',
    './exp_mapping_encryption_action',
    './exp_language_barrier_no_action',
    './exp_language_barrier_action',
]

experiment_tags = ['normal', 'construct_enc', 'construct_act', 'nl_enc', 'nl_act']

# Initialize accumulators for final summary
goal_accuracy = {tag: [] for tag in experiment_tags}
goal_confidence = {tag: [] for tag in experiment_tags}
reason_accuracy = {tag: [] for tag in experiment_tags}
reason_confidence = {tag: [] for tag in experiment_tags}

# Initialize per-round performance tracker
round_stats = {
    tag: defaultdict(lambda: {
        "goal_acc": [],
        "goal_conf": [],
        "reason_acc": [],
        "reason_conf": []
    }) for tag in experiment_tags
}

# Process each experiment folder
for folder, tag in zip(folders, experiment_tags):
    for i in range(1, 5):  # scenario_1 to scenario_4
        path = os.path.join(folder, f'scenario_{i}', 'mcq_logs.json')
        if not os.path.exists(path):
            continue

        with open(path, 'r') as f:
            logs = json.load(f)

        for entry in logs:
            round_number = entry.get("round", -1)
            for agent in ['Alex', 'Jamie']:
                # Goal MCQ
                goal_field = f'{agent}_goal_mcq'
                if goal_field in entry:
                    acc = entry[goal_field]['correct']
                    conf = entry[goal_field]['confidence']
                    goal_accuracy[tag].append(acc)
                    goal_confidence[tag].append(conf)
                    round_stats[tag][round_number]["goal_acc"].append(acc)
                    round_stats[tag][round_number]["goal_conf"].append(conf)

                # Reason MCQ
                reason_field = f'{agent}_reason_mcq'
                if reason_field in entry:
                    acc = entry[reason_field]['correct']
                    conf = entry[reason_field]['confidence']
                    reason_accuracy[tag].append(acc)
                    reason_confidence[tag].append(conf)
                    round_stats[tag][round_number]["reason_acc"].append(acc)
                    round_stats[tag][round_number]["reason_conf"].append(conf)

# Print final overall MCQA performance
print("=== MCQ Evaluation Summary (Overall) ===")
print(f"{'Setting':<22} | {'Goal Acc':>9} | {'Goal Conf':>10} | {'Reason Acc':>11} | {'Reason Conf':>12}")
print("-" * 70)
for tag in experiment_tags:
    g_acc = sum(goal_accuracy[tag]) / len(goal_accuracy[tag]) if goal_accuracy[tag] else 0.0
    g_conf = sum(goal_confidence[tag]) / len(goal_confidence[tag]) if goal_confidence[tag] else 0.0
    r_acc = sum(reason_accuracy[tag]) / len(reason_accuracy[tag]) if reason_accuracy[tag] else 0.0
    r_conf = sum(reason_confidence[tag]) / len(reason_confidence[tag]) if reason_confidence[tag] else 0.0

    print(f"{tag:<22} | {g_acc:>9.3f} | {g_conf:>10.3f} | {r_acc:>11.3f} | {r_conf:>12.3f}")

# Print round-level MCQA trends
print("\n=== MCQ Evaluation Trend by Round (Average Across Scenarios) ===")
for tag in experiment_tags:
    print(f"\n--- {tag.upper()} ---")
    print(f"{'Round':<5} | {'Goal Acc':>9} | {'Goal Conf':>10} | {'Reason Acc':>11} | {'Reason Conf':>12}")
    print("-" * 60)
    for r in sorted(round_stats[tag].keys()):
        g_accs = round_stats[tag][r]["goal_acc"]
        g_confs = round_stats[tag][r]["goal_conf"]
        r_accs = round_stats[tag][r]["reason_acc"]
        r_confs = round_stats[tag][r]["reason_conf"]

        g_acc_avg = sum(g_accs) / len(g_accs) if g_accs else 0.0
        g_conf_avg = sum(g_confs) / len(g_confs) if g_confs else 0.0
        r_acc_avg = sum(r_accs) / len(r_accs) if r_accs else 0.0
        r_conf_avg = sum(r_confs) / len(r_confs) if r_confs else 0.0

        print(f"{r:<5} | {g_acc_avg:>9.3f} | {g_conf_avg:>10.3f} | {r_acc_avg:>11.3f} | {r_conf_avg:>12.3f}")
