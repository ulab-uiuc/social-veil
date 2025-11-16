import argparse
import json
import os
import glob
import csv
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats
import re
import sys


MODES = ["baseline", "semantic", "cultural", "emotional"]
DIMS = [
    "goal_completion",
    "believability",
    "relationship",
    "knowledge",
    "social_rules",
    "financial_benefits",
]


def _safe_import_hard_ids(project_root: str) -> Optional[List[str]]:
    """Try to import SOTOPIA_HARD_ENVS from data/data_check.py.
    Returns list of IDs or None on failure.
    """
    try:
        sys.path.insert(0, project_root)
        from data.data_check import SOTOPIA_HARD_ENVS  # type: ignore
        if isinstance(SOTOPIA_HARD_ENVS, list):
            return [str(x) for x in SOTOPIA_HARD_ENVS]
    except Exception:
        pass
    return None


def _load_episodes(path: str) -> List[dict]:
    if not path:
        return []
    if path.lower().endswith(".jsonl"):
        out: List[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    try:
                        out.append(json.loads(s))
                    except Exception:
                        continue
        return out
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _guess_episode_id(ep: dict) -> Optional[str]:
    # Try common keys used for environment identifiers (prefer environment_id, then episode_id)
    for k in ("environment_id", "episode_id"):
        v = ep.get(k)
        if isinstance(v, (str, int)):
            return str(v)
    # Sometimes under nested keys
    meta = ep.get("meta") if isinstance(ep.get("meta"), dict) else None
    if meta:
        for k in ("environment_id", "episode_id"):
            v = meta.get(k)
            if isinstance(v, (str, int)):
                return str(v)
    return None


def build_hard_index_set(episodes_file: str, hard_ids: List[str], debug: bool = False) -> Set[int]:
    """Map episode indices (1-based) whose IDs are in hard_ids."""
    episodes = _load_episodes(episodes_file)
    hard_set: Set[str] = set(str(x) for x in (hard_ids or []))
    idxs: Set[int] = set()
    found_ids: Set[str] = set()
    for i, ep in enumerate(episodes, start=1):
        eid = _guess_episode_id(ep)
        if eid and eid in hard_set:
            idxs.add(i)
            found_ids.add(eid)

    if debug:
        print(f"🕵️  Scanned '{os.path.basename(episodes_file)}', found {len(found_ids)} matching hard IDs out of {len(hard_set)}.")
        if found_ids:
            print(f"   - Found IDs: {list(found_ids)[:5]}")
    return idxs


def collect_mode_stats(base_dir: str, mode: str, allowed_indices: Optional[Set[int]] = None, debug: bool = False) -> Dict[str, List[float]]:
    """
    Collects raw metric scores for a given mode.
    Returns a dictionary mapping metric names to a list of float scores.
    """
    pattern = os.path.join(base_dir, f"mode_{mode}", "scenario_*", "eval_result.json")
    all_files = glob.glob(pattern)
    files_to_process = []
    
    if allowed_indices is not None:
        for fp in all_files:
            try:
                m = re.search(r"scenario_(\d+)", fp)
                scen_idx = int(m.group(1)) if m else None
                if scen_idx is not None and scen_idx in allowed_indices:
                    files_to_process.append(fp)
            except (ValueError, IndexError):
                continue
    else:
        files_to_process = all_files
        
    if debug:
        print(f"🕵️  [mode_{mode}] Found {len(all_files)} total scenarios. After filtering for hard indices, processing {len(files_to_process)} scenarios.")

    # Store lists of scores for statistical analysis
    raw_scores: Dict[str, List[float]] = {}
    
    # Initialize keys to ensure they exist even if no data is found
    metric_keys = [f"a1_{d}" for d in DIMS] + [f"a2_{d}" for d in DIMS] + ["iq"]
    mcq_metric_keys = [
        ("goal", "accuracy"), ("goal", "avg_confidence"),
        ("reason", "accuracy"), ("reason", "avg_confidence"),
    ]
    for who in ("a1", "a2"):
        for t, m in mcq_metric_keys:
            metric_keys.append(f"{who}_{t}_{m}")
            
    for k in metric_keys:
        raw_scores[k] = []

    # Episode-level barrier metrics (only for barrier modes)
    for k in ["episode_unresolved_confusion", "episode_mutual_understanding"]:
        raw_scores[k] = []

    for fp in files_to_process:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            ag = data.get("aggregated_scores", {})
            a1 = ag.get("agent_1", {})
            a2 = ag.get("agent_2", {})
            iq = ag.get("interaction_quality", 0)
            if isinstance(iq, dict):
                iq = iq.get("score", 0)

            for d in DIMS:
                raw_scores[f"a1_{d}"].append(a1.get(d, 0))
                raw_scores[f"a2_{d}"].append(a2.get(d, 0))
            raw_scores["iq"].append(iq)

            # Episode-level barrier evaluation
            ep = ag.get("episode_level")
            if isinstance(ep, dict):
                for metric, key in [("episode_unresolved_confusion", "unresolved_confusion"), ("episode_mutual_understanding", "mutual_understanding")]:
                    val = ep.get(key)
                    score = None
                    if isinstance(val, (int, float)):
                        score = float(val)
                    elif isinstance(val, dict) and isinstance(val.get("score"), (int, float)):
                        score = float(val["score"])
                    
                    if score is not None:
                        raw_scores[metric].append(score)

            # MCQ metrics
            mm = data.get("mcq_metrics", {})
            mm_a1 = mm.get("agent_1", {})
            mm_a2 = mm.get("agent_2", {})
            for t, m in mcq_metric_keys:
                k = f"{t}_{m}"
                if isinstance(mm_a1.get(k), (int, float)):
                    raw_scores[f"a1_{k}"].append(float(mm_a1[k]))
                if isinstance(mm_a2.get(k), (int, float)):
                    raw_scores[f"a2_{k}"].append(float(mm_a2[k]))

        except Exception:
            continue

    return raw_scores


def main():
    parser = argparse.ArgumentParser(
        description="Compare average evaluation results across four modes (baseline/semantic/cultural/emotional)."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="../results/exp_qwen2.5-7b-instruct_episodes_original",
        help="Base results directory that contains mode_* subfolders",
    )
    parser.add_argument("--out_json", type=str, default="", help="Optional path to save the summary JSON")
    parser.add_argument("--out_csv", type=str, default="", help="Optional path to save the summary CSV")
    parser.add_argument("--subset", type=str, default="all", choices=["all", "sotopia_hard"], help="Subset of scenarios to aggregate")
    parser.add_argument("--episodes_file", type=str, default="", help="Episodes file (json/jsonl) to map scenario indices to env IDs when using --subset sotopia_hard")
    parser.add_argument("--debug", action="store_true", help="Enable debug printing to trace ID matching.")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    if not os.path.isdir(base_dir):
        raise SystemExit(f"Base directory not found: {base_dir}")

    allowed_indices: Optional[Set[int]] = None
    if args.subset == "sotopia_hard":
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        hard_ids = _safe_import_hard_ids(project_root) or []
        if args.debug:
            print(f"🕵️  Loaded {len(hard_ids)} hard environment IDs from data_check.py.")
            print(f"   - Sample IDs: {hard_ids[:5]}")

        if not args.episodes_file:
            raise SystemExit("--episodes_file is required when --subset sotopia_hard")
        allowed_indices = build_hard_index_set(os.path.abspath(args.episodes_file), hard_ids, debug=args.debug)
        if not allowed_indices:
            print("WARNING: No hard indices were resolved from the provided episodes file; results will be empty.")
        elif args.debug:
            print(f"✅ Mapped to {len(allowed_indices)} scenario indices: {sorted(list(allowed_indices))[:10]}...")


    all_raw_scores = {mode: collect_mode_stats(base_dir, mode, allowed_indices=allowed_indices, debug=args.debug) for mode in MODES}

    summary = {}
    for mode, raw_scores in all_raw_scores.items():
        mode_summary = {}
        # Find a representative metric to count scenarios
        representative_metric = next((scores for scores in raw_scores.values() if scores), [])
        mode_summary["num_scenarios"] = len(representative_metric)

        for metric, scores in raw_scores.items():
            if scores:
                mean = np.mean(scores)
                # Confidence interval
                ci = (0.0, 0.0)
                if len(scores) > 1:
                    ci = stats.t.interval(0.95, len(scores)-1, loc=mean, scale=stats.sem(scores))
                
                mode_summary[f"{metric}_mean"] = mean
                mode_summary[f"{metric}_ci_low"] = ci[0]
                mode_summary[f"{metric}_ci_high"] = ci[1]
        summary[mode] = mode_summary

    # Perform paired t-tests between baseline and barrier conditions
    baseline_scores = all_raw_scores.get("baseline", {})
    if baseline_scores:
        for mode in ["semantic", "cultural", "emotional"]:
            mode_scores = all_raw_scores.get(mode)
            if not mode_scores:
                continue
            
            # Use baseline keys as the reference for metrics to test
            for metric in baseline_scores.keys():
                base_data = baseline_scores.get(metric, [])
                mode_data = mode_scores.get(metric, [])
                
                # Ensure data is paired correctly and has enough samples for a t-test
                min_len = min(len(base_data), len(mode_data))
                if min_len > 1:
                    # Perform paired t-test
                    _, p_val = stats.ttest_rel(base_data[:min_len], mode_data[:min_len])
                    summary[mode][f"{metric}_pval_vs_baseline"] = p_val


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
        
        base_summary = summary.get("baseline", {})
        metrics_base = sorted([k.replace("_mean", "") for k in base_summary if k.endswith("_mean")])
        
        header = ["mode", "num_scenarios"]
        for metric in metrics_base:
            header.extend([f"{metric}_mean", f"{metric}_ci"])

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)

            for mode in MODES:
                stats = summary.get(mode, {})
                if not stats: continue
                
                row = [mode, stats.get("num_scenarios", "")]
                for metric in metrics_base:
                    mean = stats.get(f"{metric}_mean")
                    ci_low = stats.get(f"{metric}_ci_low")
                    ci_high = stats.get(f"{metric}_ci_high")
                    
                    formatted_mean = f"{mean:.3f}" if isinstance(mean, float) else ""
                    
                    # Append significance stars for non-baseline modes
                    if mode != "baseline":
                        pval = stats.get(f"{metric}_pval_vs_baseline")
                        if isinstance(pval, float):
                            if pval < 0.001:
                                formatted_mean += '***'
                            elif pval < 0.01:
                                formatted_mean += '**'
                            elif pval < 0.05:
                                formatted_mean += '*'
                    
                    row.append(formatted_mean)
                    row.append(f"({ci_low:.3f}, {ci_high:.3f})" if isinstance(ci_low, float) else "")
                
                w.writerow(row)
        print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()

