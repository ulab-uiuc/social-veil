#!/usr/bin/env python3
"""
Compare correlation between two different evaluator models on the same conversation logs.

This script:
1. Re-evaluates conversation logs using two different evaluator models
2. Computes Pearson and Spearman correlation for each evaluation dimension
3. Generates a comparison report

Usage:
    python analysis/compare_evaluators.py \
        --results_dir results/exp_qwen2.5-7b-instruct_episode_all_neutralized \
        --evaluator1 gpt-4o \
        --evaluator2 qwen2.5-7b-instruct \
        --use_vllm_for_evaluator2 \
        --output comparison_evaluators.csv
"""

import argparse
import json
import os
import sys
import warnings
import time
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy import stats
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from social_decipher.evaluate import ConversationEvaluator

# Suppress constant input warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='An input array is constant')

# Dimensions to analyze (all metrics from aggregated_scores)
AGENT_DIMENSIONS = [
    "goal_completion",
    "believability",
    "relationship",
    "knowledge",
    "social_rules",
    "financial_benefits",
]

EPISODE_DIMENSIONS = [
    "unresolved_confusion",
    "mutual_understanding",
]


def load_conversation_log(scenario_dir: str) -> Dict:
    """Load conversation log data from a scenario directory."""
    convo_log_path = os.path.join(scenario_dir, "conversation_log.json")
    
    if not os.path.exists(convo_log_path):
        return None
    
    try:
        with open(convo_log_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {convo_log_path}: {e}")
        return None


def evaluate_conversation_log(log_data: Dict, evaluator: ConversationEvaluator, mode_name: str) -> Dict:
    """Evaluate a single conversation log using the given evaluator."""
    try:
        # Extract necessary data from the log
        context = log_data.get("experimental_context", {})
        agents_context = context.get("agents", {})
        agent_a_context = agents_context.get("agent_a", {})
        agent_b_context = agents_context.get("agent_b", {})
        
        conversation = log_data.get("conversation_log", [])
        agent_goals = [agent_a_context.get("goal"), agent_b_context.get("goal")]
        agent_reasons = [agent_a_context.get("reason"), agent_b_context.get("reason")]
        scenario = context.get("scenario", {}).get("description", "")
        mcq_logs = log_data.get("mcq_logs")
        
        # Get barrier_type from the log or infer from mode name
        barrier_type = context.get("scenario", {}).get("barrier_type")
        if barrier_type is None:
            if "semantic" in mode_name:
                barrier_type = "semantic_structure"
            elif "cultural" in mode_name:
                barrier_type = "cultural_style"
            elif "emotional" in mode_name:
                barrier_type = "emotional_influence"
        
        # Run evaluation
        evaluation_result = evaluator.evaluate_conversation(
            conversation,
            agent_goals=agent_goals,
            agent_reasons=agent_reasons,
            mcq_logs=mcq_logs,
            barrier_type=barrier_type,
        )
        
        return evaluation_result
    
    except Exception as e:
        print(f"Warning: Evaluation failed: {e}")
        return None


def evaluate_scenario_pair(
    scenario_dir: str,
    evaluator1_config: Dict,
    evaluator2_config: Dict,
    mode_name: str,
    agent_num: int
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluate a single scenario with both evaluators.
    Returns (scores1, scores2) or (None, None) on failure.
    
    Args:
        evaluator1_config: Dict with keys: model, use_vllm, vllm_url
        evaluator2_config: Dict with keys: model, use_vllm, vllm_url
    """
    try:
        # Create fresh evaluators for this thread to avoid client sharing issues
        eval1 = ConversationEvaluator(
            evaluator1_config["model"],
            use_vllm=evaluator1_config["use_vllm"],
            vllm_url=evaluator1_config["vllm_url"]
        )
        eval2 = ConversationEvaluator(
            evaluator2_config["model"],
            use_vllm=evaluator2_config["use_vllm"],
            vllm_url=evaluator2_config["vllm_url"]
        )
        
        log_data = load_conversation_log(scenario_dir)
        if log_data is None:
            return (None, None)
        
        # Evaluate with both evaluators (no delay needed for high-tier API keys)
        result1 = evaluate_conversation_log(log_data, eval1, mode_name)
        result2 = evaluate_conversation_log(log_data, eval2, mode_name)
        
        if result1 is None or result2 is None:
            return (None, None)
        
        # Extract scores
        scores1 = extract_scores(result1, agent_num)
        scores2 = extract_scores(result2, agent_num)
        
        return (scores1, scores2)
    
    except Exception as e:
        print(f"\n⚠️  Error evaluating {os.path.basename(scenario_dir)}: {e}")
        return (None, None)


def extract_scores(eval_result: Dict, agent_num: int = 2) -> Dict[str, float]:
    """Extract all dimension scores from an evaluation result."""
    if eval_result is None:
        return None
    
    scores = {}
    agg_scores = eval_result.get("aggregated_scores", {})
    
    # Agent-specific dimensions
    agent_key = f"agent_{agent_num}"
    agent_scores = agg_scores.get(agent_key, {})
    for dim in AGENT_DIMENSIONS:
        scores[dim] = agent_scores.get(dim, 0.0)
    
    # Episode-level dimensions
    episode_scores = agg_scores.get("episode_level", {})
    for dim in EPISODE_DIMENSIONS:
        scores[dim] = episode_scores.get(dim, 0.0)
    
    return scores


def compute_correlation(scores1: List[float], scores2: List[float]) -> Tuple[float, float, float, float]:
    """
    Compute Pearson and Spearman correlation between two score lists.
    
    Returns:
        (pearson_r, pearson_p, spearman_rho, spearman_p)
    """
    # Check if input is constant
    if len(set(scores1)) == 1 or len(set(scores2)) == 1:
        return np.nan, np.nan, np.nan, np.nan
    
    try:
        pearson_r, pearson_p = stats.pearsonr(scores1, scores2)
        spearman_rho, spearman_p = stats.spearmanr(scores1, scores2)
        return pearson_r, pearson_p, spearman_rho, spearman_p
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def main():
    parser = argparse.ArgumentParser(
        description="Compare correlation between two evaluator models on the same conversations."
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Path to the experiment results directory containing conversation logs."
    )
    parser.add_argument(
        "--evaluator1", type=str, default="gpt-4o",
        help="First evaluator model (e.g., 'gpt-4o')."
    )
    parser.add_argument(
        "--evaluator2", type=str, default="gpt-4o-mini",
        help="Second evaluator model (e.g., 'gpt-4o-mini' or 'qwen2.5-7b-instruct')."
    )
    parser.add_argument(
        "--use_vllm_for_evaluator1", action="store_true",
        help="Use vLLM for evaluator1."
    )
    parser.add_argument(
        "--use_vllm_for_evaluator2", action="store_true",
        help="Use vLLM for evaluator2."
    )
    parser.add_argument(
        "--vllm_url", type=str, default=None,
        help="Base URL for vLLM server (if using vLLM)."
    )
    parser.add_argument(
        "--agent", type=int, default=2, choices=[1, 2],
        help="Which agent to analyze (1 or 2). Default: 2 (partner agent)."
    )
    parser.add_argument(
        "--modes", type=str, nargs="+", 
        default=["baseline", "semantic", "cultural", "emotional"],
        help="Modes to analyze. Default: all modes."
    )
    parser.add_argument(
        "--output", type=str, default="results/evaluator_comparison.csv",
        help="Output CSV file path for correlation results."
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print summary table to console."
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="Number of concurrent evaluations. Default: read from CONCURRENCY env var or 16."
    )
    
    args = parser.parse_args()
    
    # Get concurrency from args, env var, or default to 16 (optimized for high-tier API keys)
    if args.concurrency is not None:
        concurrency = args.concurrency
    else:
        concurrency = int(os.environ.get("CONCURRENCY", 16))
    
    print(f"🚀 Using concurrency: {concurrency}")
    
    if not os.path.isdir(args.results_dir):
        print(f"Error: Directory not found at {args.results_dir}")
        return

    print("="*80)
    print(f"📊 Evaluator Correlation Analysis")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Evaluator 1: {args.evaluator1} {'(vLLM)' if args.use_vllm_for_evaluator1 else '(OpenAI)'}")
    print(f"Evaluator 2: {args.evaluator2} {'(vLLM)' if args.use_vllm_for_evaluator2 else '(OpenAI)'}")
    print(f"Analyzing Agent {args.agent}")
    print(f"Modes: {', '.join(args.modes)}")
    print("="*80)
    
    # Prepare evaluator configs (will be used to create fresh evaluators per thread)
    evaluator1_config = {
        "model": args.evaluator1,
        "use_vllm": args.use_vllm_for_evaluator1,
        "vllm_url": args.vllm_url
    }
    evaluator2_config = {
        "model": args.evaluator2,
        "use_vllm": args.use_vllm_for_evaluator2,
        "vllm_url": args.vllm_url
    }
    
    # Collect all dimension scores across all modes
    all_dimensions = AGENT_DIMENSIONS + EPISODE_DIMENSIONS
    correlation_results = {}
    
    for mode in args.modes:
        print(f"\n{'='*80}")
        print(f"Mode: {mode}")
        print(f"{'='*80}")
        
        mode_dir = os.path.join(args.results_dir, f"mode_{mode}")
        if not os.path.isdir(mode_dir):
            print(f"Warning: Mode directory not found: {mode_dir}")
            continue
        
        # Find all scenario directories
        scenario_dirs = sorted([
            d.path for d in os.scandir(mode_dir) 
            if d.is_dir() and d.name.startswith("scenario_")
        ])
        
        if not scenario_dirs:
            print(f"No scenarios found in {mode}")
            continue
        
        print(f"Found {len(scenario_dirs)} scenarios")
        
        # Collect scores from both evaluators for each dimension
        dimension_scores = {dim: {"eval1": [], "eval2": []} for dim in all_dimensions}
        
        # Use ThreadPoolExecutor for concurrent evaluation
        if concurrency > 1:
            print(f"Using {concurrency} concurrent workers")
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(
                        evaluate_scenario_pair,
                        scenario_dir,
                        evaluator1_config,
                        evaluator2_config,
                        mode,
                        args.agent
                    ): scenario_dir
                    for scenario_dir in scenario_dirs
                }
                
                # Collect results with progress bar
                completed = 0
                failed = 0
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {mode}"):
                    try:
                        scores1, scores2 = future.result()
                        
                        if scores1 is None or scores2 is None:
                            failed += 1
                            continue
                        
                        # Collect scores for each dimension
                        for dim in all_dimensions:
                            dimension_scores[dim]["eval1"].append(scores1[dim])
                            dimension_scores[dim]["eval2"].append(scores2[dim])
                        
                        completed += 1
                    except Exception as e:
                        failed += 1
                        print(f"\n⚠️  Task failed with error: {e}")
                
                print(f"✅ Completed: {completed}, ❌ Failed: {failed}")
        else:
            # Sequential evaluation
            completed = 0
            failed = 0
            for scenario_dir in tqdm(scenario_dirs, desc=f"Evaluating {mode}"):
                scores1, scores2 = evaluate_scenario_pair(
                    scenario_dir, evaluator1_config, evaluator2_config, mode, args.agent
                )
                
                if scores1 is None or scores2 is None:
                    failed += 1
                    continue
                
                # Collect scores for each dimension
                for dim in all_dimensions:
                    dimension_scores[dim]["eval1"].append(scores1[dim])
                    dimension_scores[dim]["eval2"].append(scores2[dim])
                
                completed += 1
            
            print(f"✅ Completed: {completed}, ❌ Failed: {failed}")
        
        # Compute correlations for this mode
        mode_correlations = {}
        print(f"\nCorrelation results for {mode}:")
        print(f"{'Dimension':<25} | {'N':<5} | {'Pearson r':<10} | {'p-value':<10} | {'Spearman ρ':<10} | {'p-value':<10}")
        print("-" * 90)
        
        for dim in all_dimensions:
            scores1 = dimension_scores[dim]["eval1"]
            scores2 = dimension_scores[dim]["eval2"]
            
            if len(scores1) < 2:
                continue
            
            pearson_r, pearson_p, spearman_rho, spearman_p = compute_correlation(scores1, scores2)
            
            mode_correlations[dim] = {
                "n": len(scores1),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "mean_eval1": np.mean(scores1),
                "mean_eval2": np.mean(scores2),
            }
            
            # Format output
            if np.isnan(pearson_r):
                print(f"{dim:<25} | {len(scores1):<5} | {'CONSTANT':<10} | {'':<10} | {'CONSTANT':<10} | {'':<10}")
            else:
                # Add significance stars
                p_stars = ""
                if pearson_p < 0.001:
                    p_stars = "***"
                elif pearson_p < 0.01:
                    p_stars = "**"
                elif pearson_p < 0.05:
                    p_stars = "*"
                
                print(f"{dim:<25} | {len(scores1):<5} | {pearson_r:.3f}{p_stars:<7} | {pearson_p:<10.4f} | {spearman_rho:.3f}{' ':<7} | {spearman_p:<10.4f}")
        
        correlation_results[mode] = mode_correlations
    
    # Save results to CSV
    print(f"\n💾 Saving results to {args.output}")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        # Write header
        header = ["mode", "dimension", "n", "pearson_r", "pearson_p", "spearman_rho", "spearman_p", 
                  f"mean_{args.evaluator1}", f"mean_{args.evaluator2}", "mean_diff"]
        f.write(",".join(header) + "\n")
        
        # Write data
        for mode in args.modes:
            if mode not in correlation_results:
                continue
            
            for dim in all_dimensions:
                if dim not in correlation_results[mode]:
                    continue
                
                stats_dict = correlation_results[mode][dim]
                mean_diff = stats_dict["mean_eval2"] - stats_dict["mean_eval1"]
                
                row = [
                    mode,
                    dim,
                    str(stats_dict["n"]),
                    f"{stats_dict['pearson_r']:.4f}" if not np.isnan(stats_dict['pearson_r']) else "NA",
                    f"{stats_dict['pearson_p']:.6f}" if not np.isnan(stats_dict['pearson_p']) else "NA",
                    f"{stats_dict['spearman_rho']:.4f}" if not np.isnan(stats_dict['spearman_rho']) else "NA",
                    f"{stats_dict['spearman_p']:.6f}" if not np.isnan(stats_dict['spearman_p']) else "NA",
                    f"{stats_dict['mean_eval1']:.4f}",
                    f"{stats_dict['mean_eval2']:.4f}",
                    f"{mean_diff:+.4f}",
                ]
                f.write(",".join(row) + "\n")
    
    print(f"✅ Results saved to {args.output}")
    
    # Print summary if requested
    if args.summary:
        print("\n" + "="*80)
        print("📈 SUMMARY: Average Correlation Across All Modes")
        print("="*80)
        
        # Compute average correlation for each dimension across all modes
        dimension_avg_corr = {dim: [] for dim in all_dimensions}
        
        for mode in correlation_results:
            for dim in all_dimensions:
                if dim in correlation_results[mode]:
                    r = correlation_results[mode][dim]["pearson_r"]
                    if not np.isnan(r):
                        dimension_avg_corr[dim].append(r)
        
        print(f"{'Dimension':<25} | {'Avg Pearson r':<15} | {'Modes':<10}")
        print("-" * 60)
        for dim in all_dimensions:
            if dimension_avg_corr[dim]:
                avg_r = np.mean(dimension_avg_corr[dim])
                n_modes = len(dimension_avg_corr[dim])
                print(f"{dim:<25} | {avg_r:.3f}{' ':<11} | {n_modes}/{len(args.modes)}")
            else:
                print(f"{dim:<25} | {'N/A':<15} | 0/{len(args.modes)}")
        print("="*80)


if __name__ == "__main__":
    main()

