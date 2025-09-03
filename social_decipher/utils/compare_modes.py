import argparse
import json
import os
import glob
import csv
from typing import Dict, List

import numpy as np


MODES = ["baseline", "semantic", "cultural", "emotional"]
DIMS = [
    "goal_completion",
    "believability",
    "relationship",
    "knowledge",
    "social_rules",
    "financial_benefits",
]


def collect_mode_stats(base_dir: str, mode: str) -> Dict[str, float]:
    pattern = os.path.join(base_dir, f"mode_{mode}", "scenario_*", "eval_result.json")
    a1_over: List[float] = []
    a2_over: List[float] = []
    dim_acc: Dict[str, List[float]] = {f"a1_{d}": [] for d in DIMS}
    dim_acc.update({f"a2_{d}": [] for d in DIMS})
    dim_acc["iq"] = []
    # Episode-level barrier metrics (only present in barrier runs)
    epi_metrics: Dict[str, List[float]] = {
        "episode_unresolved_confusion": [],
        "episode_mutual_understanding": [],
    }

    mcq_keys = [
        ("goal", "accuracy"), ("goal", "avg_confidence"),
        ("reason", "accuracy"), ("reason", "avg_confidence"),
    ]
    mcq_acc: Dict[str, List[float]] = {}
    for who in ("a1", "a2"):
        for t, m in mcq_keys:
            mcq_acc[f"{who}_{t}_{m}"] = []

    for fp in glob.glob(pattern):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            ag = data.get("aggregated_scores", {})
            a1 = ag.get("agent_1", {})
            a2 = ag.get("agent_2", {})
            iq = ag.get("interaction_quality", 0)
            if isinstance(iq, dict):
                iq = iq.get("score", 0)

            a1_over.append(a1.get("overall", 0))
            a2_over.append(a2.get("overall", 0))
            for d in DIMS:
                dim_acc[f"a1_{d}"].append(a1.get(d, 0))
                dim_acc[f"a2_{d}"].append(a2.get(d, 0))
            dim_acc["iq"].append(iq)

            # Episode-level barrier evaluation (if present)
            ep = ag.get("episode_level")
        
            if isinstance(ep, dict):
                uc = ep.get("unresolved_confusion")
                mu = ep.get("mutual_understanding")
                # Accept either flattened numeric or nested {"score": x}
                if isinstance(uc, (int, float)):
                    epi_metrics["episode_unresolved_confusion"].append(float(uc))
                elif isinstance(uc, dict) and isinstance(uc.get("score"), (int, float)):
                    epi_metrics["episode_unresolved_confusion"].append(float(uc["score"]))
                if isinstance(mu, (int, float)):
                    epi_metrics["episode_mutual_understanding"].append(float(mu))
                elif isinstance(mu, dict) and isinstance(mu.get("score"), (int, float)):
                    epi_metrics["episode_mutual_understanding"].append(float(mu["score"]))

            # MCQ metrics
            mm = data.get("mcq_metrics", {})
            mm_a1 = mm.get("agent_1", {})
            mm_a2 = mm.get("agent_2", {})
            for t, m in mcq_keys:
                k = f"{t}_{m}"
                v1 = mm_a1.get(k)
                v2 = mm_a2.get(k)
                if isinstance(v1, (int, float)):
                    mcq_acc[f"a1_{t}_{m}"].append(float(v1))
                if isinstance(v2, (int, float)):
                    mcq_acc[f"a2_{t}_{m}"].append(float(v2))
        except Exception:
            # Skip unreadable files
            continue

    out: Dict[str, float] = {}

    # per-dimension means and counts
    for k, v in dim_acc.items():
        if k == "iq":
            continue
        out[k] = float(np.mean(v)) if v else None
    out["num_scenarios"] = int(max(len(a1_over), len(a2_over)))

    # episode-level barrier means (if any collected)
    for k, v in epi_metrics.items():
        if v:
            out[k] = float(np.mean(v))

    # mcq means
    for k, v in mcq_acc.items():
        out[k] = float(np.mean(v)) if v else None
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Compare average evaluation results across four modes (baseline/semantic/cultural/emotional)."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="../../results/exp_qwen2.5-7b-instruct_episodes_original",
        help="Base results directory that contains mode_* subfolders",
    )
    parser.add_argument("--out_json", type=str, default="../../results/results.json", help="Optional path to save the summary JSON")
    parser.add_argument("--out_csv", type=str, default="../../results/results.csv", help="Optional path to save the summary CSV")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    if not os.path.isdir(base_dir):
        raise SystemExit(f"Base directory not found: {base_dir}")

    summary = {mode: collect_mode_stats(base_dir, mode) for mode in MODES}

    # Pretty print
    print(json.dumps(summary, indent=2))

    if args.out_json:
        out_path = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved summary to {out_path}")

    if args.out_csv:
        csv_path = os.path.abspath(args.out_csv)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        # Determine a stable header union across modes
        header_keys = set()
        for mode, stats in summary.items():
            header_keys.update(stats.keys())
        # Preferred ordering
        preferred = [
            "num_scenarios",
        ]
        # Sotopia dims
        preferred += [f"a1_{d}" for d in DIMS] + [f"a2_{d}" for d in DIMS]
        # Episode-level barrier metrics
        preferred += [
            "episode_unresolved_confusion",
            "episode_mutual_understanding",
        ]
        # MCQ
        for who in ("a1","a2"):
            for t in ("goal","reason"):
                preferred += [f"{who}_{t}_accuracy", f"{who}_{t}_avg_confidence"]
        # Final header
        header = preferred + [k for k in sorted(header_keys) if k not in preferred]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mode"] + header)
            for mode in MODES:
                stats = summary.get(mode, {})
                row = [mode] + [stats.get(k, "") for k in header]
                w.writerow(row)
        print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()

