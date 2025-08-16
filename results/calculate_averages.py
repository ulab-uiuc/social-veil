#!/usr/bin/env python3
"""
Calculate average performance metrics across all experiment results.
This script analyzes the updated result format with aggregated eval and MCQ files.
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import re
import math

def load_experiment_data(results_dir: str) -> Dict[str, Any]:
    """Load all experiment data from the results directory."""
    experiment_data = {}
    
    for exp_dir in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_dir)
        if not os.path.isdir(exp_path) or exp_dir.startswith('.') or exp_dir == 'analysis_output':
            continue
            
        print(f"Processing experiment: {exp_dir}")
        experiment_data[exp_dir] = {
            'eval_results': None,
            'mcq_data': None,
            'scenario_count': 0
        }
        
        # Auto-detect evaluation file patterns based on experiment name
        # Remove 'exp_' prefix and try different suffixes
        exp_name_clean = exp_dir.replace('exp_', '')
        
        eval_patterns = [
            f"{exp_name_clean}_eval.json",
            "eval_results.json",
            "aggregated_eval.json",
            "eval.json"
        ]
        
        # Also try the exact filenames we found in the directory
        try:
            available_files = [f for f in os.listdir(exp_path) if f.endswith('.json')]
            eval_files_in_dir = [f for f in available_files if 'eval' in f.lower()]
            eval_patterns.extend(eval_files_in_dir)
        except:
            pass
        
        eval_file = None
        for pattern in eval_patterns:
            potential_file = os.path.join(exp_path, pattern)
            if os.path.exists(potential_file):
                eval_file = potential_file
                break
        
        if eval_file:
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                    experiment_data[exp_dir]['eval_results'] = eval_data
                    experiment_data[exp_dir]['scenario_count'] = len(eval_data) if isinstance(eval_data, list) else 1
                    print(f"  ✅ Loaded evaluation data: {len(eval_data) if isinstance(eval_data, list) else 1} scenarios")
            except Exception as e:
                print(f"  ⚠️  Error loading {eval_file}: {e}")
        
        # Auto-detect MCQ file patterns
        mcq_patterns = [
            f"{exp_name_clean}_mcq.json",
            "mcq_results.json",
            "aggregated_mcq.json",
            "mcq.json"
        ]
        
        # Also try the exact MCQ filenames we found in the directory
        try:
            mcq_files_in_dir = [f for f in available_files if 'mcq' in f.lower()]
            mcq_patterns.extend(mcq_files_in_dir)
        except:
            pass
        
        mcq_file = None
        for pattern in mcq_patterns:
            potential_file = os.path.join(exp_path, pattern)
            if os.path.exists(potential_file):
                mcq_file = potential_file
                break
        
        if mcq_file:
            try:
                with open(mcq_file, 'r') as f:
                    mcq_data = json.load(f)
                    experiment_data[exp_dir]['mcq_data'] = mcq_data
                    print(f"  ✅ Loaded MCQ data: {len(mcq_data) if isinstance(mcq_data, list) else 1} scenarios")
            except Exception as e:
                print(f"  ⚠️  Error loading {mcq_file}: {e}")
        
        if not eval_file and not mcq_file:
            print(f"  ⚠️  No aggregated data files found for {exp_dir}")
            # List available files for debugging
            try:
                available_files = [f for f in os.listdir(exp_path) if f.endswith('.json')]
                if available_files:
                    print(f"      Available JSON files: {available_files}")
            except:
                pass
    
    return experiment_data

def calculate_social_performance_averages(experiment_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Calculate averages for social performance metrics from the new format."""
    results = {}
    
    for exp_name, exp_data in experiment_data.items():
        if not exp_data['eval_results']:
            continue
            
        eval_results = exp_data['eval_results']
        if not isinstance(eval_results, list):
            eval_results = [eval_results]
        
        # Collect all scores by metric
        metrics = defaultdict(list)
        
        for scenario_result in eval_results:
            if 'aggregated_scores' in scenario_result:
                agg_scores = scenario_result['aggregated_scores']
                
                # Agent metrics
                for agent_key in ['agent_1', 'agent_2']:
                    if agent_key in agg_scores:
                        agent_scores = agg_scores[agent_key]
                        for metric_name, score in agent_scores.items():
                            if isinstance(score, (int, float)):
                                metrics[f'{agent_key}_{metric_name}'].append(score)
                
                # Interaction quality
                if 'interaction_quality' in agg_scores:
                    metrics['interaction_quality'].append(agg_scores['interaction_quality'])
        
        # Calculate statistics
        results[exp_name] = {}
        for metric_name, scores in metrics.items():
            if scores:
                results[exp_name][metric_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'median': np.median(scores),
                    'count': len(scores),
                    'scores': scores
                }
    
    return results

def calculate_mcq_averages(experiment_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Calculate averages for MCQ performance metrics from the new format."""
    results = {}
    
    for exp_name, exp_data in experiment_data.items():
        if not exp_data['mcq_data']:
            continue
            
        mcq_data = exp_data['mcq_data']
        if not isinstance(mcq_data, list):
            mcq_data = [mcq_data]
        
        # Collect MCQ metrics
        mcq_metrics = defaultdict(list)
        
        for scenario_mcq in mcq_data:
            # Each scenario_mcq should be a list of rounds
            if isinstance(scenario_mcq, list):
                for round_data in scenario_mcq:
                    if isinstance(round_data, dict):
                        # Process each agent's MCQ results in this round
                        for key, mcq_result in round_data.items():
                            if key in ['round', 'scenario']:
                                continue
                            
                            if isinstance(mcq_result, dict) and mcq_result is not None:
                                # Extract confidence and correctness
                                if 'confidence' in mcq_result:
                                    mcq_metrics[f'{key}_confidence'].append(mcq_result['confidence'])
                                if 'correct' in mcq_result:
                                    mcq_metrics[f'{key}_accuracy'].append(1.0 if mcq_result['correct'] else 0.0)
                                elif 'is_correct' in mcq_result:
                                    mcq_metrics[f'{key}_accuracy'].append(1.0 if mcq_result['is_correct'] else 0.0)
        
        # Calculate statistics
        results[exp_name] = {}
        for metric_name, scores in mcq_metrics.items():
            if scores:
                results[exp_name][metric_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'median': np.median(scores),
                    'count': len(scores),
                    'scores': scores
                }
    
    return results

def calculate_mcq_detailed_averages(experiment_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Calculate detailed MCQ averages by agent and question type."""
    results = {}
    
    for exp_name, exp_data in experiment_data.items():
        if not exp_data['eval_results']:
            continue
            
        eval_results = exp_data['eval_results']
        if not isinstance(eval_results, list):
            eval_results = [eval_results]
        
        # Collect MCQ metrics from eval results
        mcq_metrics = defaultdict(list)
        
        for scenario_result in eval_results:
            if 'mcq_metrics' in scenario_result:
                mcq_data = scenario_result['mcq_metrics']
                
                for agent_key in ['agent_1', 'agent_2']:
                    if agent_key in mcq_data:
                        agent_mcq = mcq_data[agent_key]
                        
                        # Process different question types
                        for question_type in ['goal_pure_list', 'reason_pure_list', 'knowledge_pure_list']:
                            if question_type in agent_mcq and isinstance(agent_mcq[question_type], list):
                                for item in agent_mcq[question_type]:
                                    if isinstance(item, dict):
                                        if 'confidence' in item:
                                            mcq_metrics[f'{agent_key}_{question_type}_confidence'].append(item['confidence'])
                                        if 'correct' in item:
                                            mcq_metrics[f'{agent_key}_{question_type}_accuracy'].append(1.0 if item['correct'] else 0.0)
        
        # Calculate statistics
        results[exp_name] = {}
        for metric_name, scores in mcq_metrics.items():
            if scores:
                results[exp_name][metric_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'median': np.median(scores),
                    'count': len(scores),
                    'scores': scores
                }
    
    return results

def create_summary_report(social_results: Dict, mcq_results: Dict, mcq_detailed_results: Dict, output_dir: str) -> pd.DataFrame:
    """Create comprehensive summary reports."""
    
    summary_data = []
    
    all_experiments = set(social_results.keys()) | set(mcq_results.keys()) | set(mcq_detailed_results.keys())
    
    for exp_name in sorted(all_experiments):
        row = {'experiment': exp_name}
        
        # Social performance metrics
        if exp_name in social_results:
            social = social_results[exp_name]
            
            # Overall scores
            for agent in ['agent_1', 'agent_2']:
                if f'{agent}_overall' in social:
                    row[f'{agent}_overall'] = round(social[f'{agent}_overall']['mean'], 2)
                if f'{agent}_goal_completion' in social:
                    row[f'{agent}_goal_completion'] = round(social[f'{agent}_goal_completion']['mean'], 2)
                if f'{agent}_believability' in social:
                    row[f'{agent}_believability'] = round(social[f'{agent}_believability']['mean'], 2)
                if f'{agent}_relationship' in social:
                    row[f'{agent}_relationship'] = round(social[f'{agent}_relationship']['mean'], 2)
            
            if 'interaction_quality' in social:
                row['interaction_quality'] = round(social['interaction_quality']['mean'], 2)
        
        # MCQ performance (from aggregated MCQ file)
        if exp_name in mcq_results:
            mcq = mcq_results[exp_name]
            
            # Calculate overall accuracy and confidence by agent
            for agent in ['agent_1', 'agent_2']:
                accuracy_metrics = [k for k in mcq.keys() if k.startswith(f'{agent}_') and k.endswith('_accuracy')]
                confidence_metrics = [k for k in mcq.keys() if k.startswith(f'{agent}_') and k.endswith('_confidence')]
                
                if accuracy_metrics:
                    accuracies = [mcq[k]['mean'] for k in accuracy_metrics]
                    row[f'{agent}_mcq_accuracy'] = round(np.mean(accuracies), 3)
                
                if confidence_metrics:
                    confidences = [mcq[k]['mean'] for k in confidence_metrics]
                    row[f'{agent}_mcq_confidence'] = round(np.mean(confidences), 3)
        
        # MCQ detailed performance (from eval results)
        if exp_name in mcq_detailed_results:
            mcq_detailed = mcq_detailed_results[exp_name]
            
            for agent in ['agent_1', 'agent_2']:
                # Goal, reason, knowledge accuracy
                for question_type in ['goal_pure_list', 'reason_pure_list', 'knowledge_pure_list']:
                    acc_key = f'{agent}_{question_type}_accuracy'
                    conf_key = f'{agent}_{question_type}_confidence'
                    
                    if acc_key in mcq_detailed:
                        row[f'{agent}_{question_type.replace("_pure_list", "")}_acc'] = round(mcq_detailed[acc_key]['mean'], 3)
                    if conf_key in mcq_detailed:
                        row[f'{agent}_{question_type.replace("_pure_list", "")}_conf'] = round(mcq_detailed[conf_key]['mean'], 3)
        
        summary_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    df = df.fillna('N/A')
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'experiment_averages_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"📊 Summary saved to: {csv_path}")
    
    # Print summary table
    print("\n" + "="*120)
    print("EXPERIMENT PERFORMANCE SUMMARY")
    print("="*120)
    
    # Display the dataframe with proper formatting
    with pd.option_context('display.max_columns', None, 'display.width', None, 'display.max_colwidth', 15):
        print(df.to_string(index=False))
    
    return df

def save_detailed_results(social_results: Dict, mcq_results: Dict, mcq_detailed_results: Dict, output_dir: str):
    """Save detailed results to JSON files."""
    
    detailed_results = {
        'social_performance': social_results,
        'mcq_performance': mcq_results,
        'mcq_detailed_performance': mcq_detailed_results,
        'metadata': {
            'description': 'Averaged performance metrics across all scenarios in each experiment',
            'social_metrics': [
                'overall', 'goal_completion', 'believability', 'relationship', 
                'knowledge', 'social_rules', 'financial_benefits', 'interaction_quality'
            ],
            'mcq_metrics': [
                'accuracy', 'confidence'
            ],
            'mcq_question_types': [
                'goal', 'reason', 'knowledge'
            ]
        }
    }
    
    detailed_path = os.path.join(output_dir, 'detailed_averages.json')
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results, f, indent=4, default=str)
    
    print(f"📋 Detailed results saved to: {detailed_path}")

def save_individual_experiment_summaries(social_results: Dict, mcq_results: Dict, mcq_detailed_results: Dict, output_dir: str):
    """Save individual summary for each experiment."""
    
    for exp_name in social_results.keys():
        exp_summary = {
            'experiment_name': exp_name,
            'social_performance': social_results.get(exp_name, {}),
            'mcq_performance': mcq_results.get(exp_name, {}),
            'mcq_detailed_performance': mcq_detailed_results.get(exp_name, {})
        }
        
        # Create a simplified summary for this experiment
        simple_summary = {'experiment': exp_name}
        
        # Social metrics
        if exp_name in social_results:
            social = social_results[exp_name]
            for metric_name, stats in social.items():
                simple_summary[metric_name] = {
                    'mean': round(stats['mean'], 3),
                    'std': round(stats['std'], 3),
                    'count': stats['count']
                }
        
        # MCQ metrics
        if exp_name in mcq_results:
            mcq = mcq_results[exp_name]
            for metric_name, stats in mcq.items():
                simple_summary[f'mcq_{metric_name}'] = {
                    'mean': round(stats['mean'], 3),
                    'std': round(stats['std'], 3),
                    'count': stats['count']
                }
        
        # MCQ detailed metrics
        if exp_name in mcq_detailed_results:
            mcq_detailed = mcq_detailed_results[exp_name]
            for metric_name, stats in mcq_detailed.items():
                simple_summary[f'detailed_{metric_name}'] = {
                    'mean': round(stats['mean'], 3),
                    'std': round(stats['std'], 3),
                    'count': stats['count']
                }
        
        # Save full detailed summary
        exp_detailed_path = os.path.join(output_dir, f'{exp_name}_detailed_summary.json')
        with open(exp_detailed_path, 'w') as f:
            json.dump(exp_summary, f, indent=4, default=str)
        
        # Save simple summary
        exp_simple_path = os.path.join(output_dir, f'{exp_name}_summary.json')
        with open(exp_simple_path, 'w') as f:
            json.dump(simple_summary, f, indent=4, default=str)
        
        # Save human-readable text summary
        exp_text_path = os.path.join(output_dir, f'{exp_name}_summary.txt')
        with open(exp_text_path, 'w') as f:
            f.write(f"EXPERIMENT SUMMARY: {exp_name}\n")
            f.write("=" * 60 + "\n\n")
            
            # Social Performance
            if exp_name in social_results:
                f.write("SOCIAL PERFORMANCE METRICS:\n")
                f.write("-" * 30 + "\n")
                social = social_results[exp_name]
                
                for agent in ['agent_1', 'agent_2']:
                    agent_metrics = {k: v for k, v in social.items() if k.startswith(agent)}
                    if agent_metrics:
                        f.write(f"\n{agent.upper()}:\n")
                        for metric, stats in agent_metrics.items():
                            metric_clean = metric.replace(f'{agent}_', '').replace('_', ' ').title()
                            f.write(f"  {metric_clean:20}: {stats['mean']:.2f} ± {stats['std']:.2f} (n={stats['count']})\n")
                
                # Interaction quality
                if 'interaction_quality' in social:
                    iq = social['interaction_quality']
                    f.write(f"\nINTERACTION QUALITY: {iq['mean']:.2f} ± {iq['std']:.2f} (n={iq['count']})\n")
            
            # MCQ Performance
            if exp_name in mcq_results:
                f.write(f"\n\nMCQ PERFORMANCE:\n")
                f.write("-" * 20 + "\n")
                mcq = mcq_results[exp_name]
                
                for agent in ['agent_1', 'agent_2']:
                    agent_metrics = {k: v for k, v in mcq.items() if k.startswith(agent)}
                    if agent_metrics:
                        f.write(f"\n{agent.upper()}:\n")
                        for metric, stats in agent_metrics.items():
                            metric_clean = metric.replace(f'{agent}_', '').replace('_', ' ').title()
                            if 'accuracy' in metric:
                                f.write(f"  {metric_clean:20}: {stats['mean']:.1%} (n={stats['count']})\n")
                            else:
                                f.write(f"  {metric_clean:20}: {stats['mean']:.3f} (n={stats['count']})\n")
            
            f.write(f"\n\nGenerated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"📄 Experiment summary saved: {exp_simple_path}")
        print(f"📋 Experiment detailed summary saved: {exp_detailed_path}")
        print(f"📝 Experiment text summary saved: {exp_text_path}")

def create_visualizations(social_results: Dict, mcq_results: Dict, output_dir: str):
    """Create visualization plots for the results."""
    
    # Set up matplotlib
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Social Performance Visualization
    if social_results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Social Performance Metrics Across Experiments', fontsize=16, fontweight='bold')
        
        # Overall scores
        experiments = list(social_results.keys())
        agent1_overall = [social_results[exp].get('agent_1_overall', {}).get('mean', 0) for exp in experiments]
        agent2_overall = [social_results[exp].get('agent_2_overall', {}).get('mean', 0) for exp in experiments]
        
        x = np.arange(len(experiments))
        width = 0.35
        
        axes[0,0].bar(x - width/2, agent1_overall, width, label='Agent 1', alpha=0.8)
        axes[0,0].bar(x + width/2, agent2_overall, width, label='Agent 2', alpha=0.8)
        axes[0,0].set_title('Overall Scores')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels([exp.replace('exp_', '').replace('_', '\n') for exp in experiments], rotation=45, ha='right')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Goal completion
        agent1_goal = [social_results[exp].get('agent_1_goal_completion', {}).get('mean', 0) for exp in experiments]
        agent2_goal = [social_results[exp].get('agent_2_goal_completion', {}).get('mean', 0) for exp in experiments]
        
        axes[0,1].bar(x - width/2, agent1_goal, width, label='Agent 1', alpha=0.8)
        axes[0,1].bar(x + width/2, agent2_goal, width, label='Agent 2', alpha=0.8)
        axes[0,1].set_title('Goal Completion')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels([exp.replace('exp_', '').replace('_', '\n') for exp in experiments], rotation=45, ha='right')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Believability
        agent1_belief = [social_results[exp].get('agent_1_believability', {}).get('mean', 0) for exp in experiments]
        agent2_belief = [social_results[exp].get('agent_2_believability', {}).get('mean', 0) for exp in experiments]
        
        axes[1,0].bar(x - width/2, agent1_belief, width, label='Agent 1', alpha=0.8)
        axes[1,0].bar(x + width/2, agent2_belief, width, label='Agent 2', alpha=0.8)
        axes[1,0].set_title('Believability')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels([exp.replace('exp_', '').replace('_', '\n') for exp in experiments], rotation=45, ha='right')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Interaction Quality
        interaction_quality = [social_results[exp].get('interaction_quality', {}).get('mean', 0) for exp in experiments]
        
        axes[1,1].bar(x, interaction_quality, alpha=0.8, color='purple')
        axes[1,1].set_title('Interaction Quality')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels([exp.replace('exp_', '').replace('_', '\n') for exp in experiments], rotation=45, ha='right')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        social_plot_path = os.path.join(output_dir, 'social_performance_comparison.png')
        plt.savefig(social_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Social performance plot saved to: {social_plot_path}")

def _extract_ratio_from_name(exp_name: str) -> float:
    """Extract barrier ratio from experiment folder name like '..._ratio50_...'. Returns 0..1 or None."""
    m = re.search(r"ratio(\d+)", exp_name)
    if not m:
        return None
    try:
        val = int(m.group(1))
        return max(0.0, min(1.0, val / 100.0))
    except Exception:
        return None

def _load_conversation_lines(scenario_dir: str) -> List[str]:
    path = os.path.join(scenario_dir, "conversation_log.txt")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r") as f:
            return [line.rstrip("\n") for line in f if line.strip()]
    except Exception:
        return []

def _get_speakers(conv_lines: List[str]) -> tuple[str, str]:
    """Infer personA and personB names from the first two lines."""
    if not conv_lines:
        return ("A", "B")
    try:
        first = conv_lines[0].split(":", 1)[0].strip()
        # Find first line with a different speaker for B
        second = next((l.split(":", 1)[0].strip() for l in conv_lines[1:] if ":" in l and l.split(":", 1)[0].strip() != first), "B")
        return (first, second)
    except Exception:
        return ("A", "B")

def _split_barrier_segments(conv_lines: List[str], name_a: str, ratio: float) -> dict:
    """
    Determine barrier block for Agent A: first ceil(total_A_turns * ratio) A-turns are under barrier.
    Returns indices and segments for analysis of Agent B responses.
    """
    if not conv_lines:
        return {
            "barrier_a_turns": 0,
            "total_a_turns": 0,
            "barrier_b_lines": [],
            "post_b_lines": []
        }
    # Collect indices of lines by speaker
    a_line_indices = [i for i, l in enumerate(conv_lines) if l.startswith(f"{name_a}:")]
    total_a_turns = len(a_line_indices)
    if total_a_turns == 0:
        return {
            "barrier_a_turns": 0,
            "total_a_turns": 0,
            "barrier_b_lines": [],
            "post_b_lines": []
        }
    target = int(math.ceil(max(0.0, min(1.0, ratio)) * total_a_turns))
    barrier_a_indices = set(a_line_indices[:target])
    # For each A barrier line, take the immediate next B line (if exists)
    barrier_b_lines = []
    post_b_lines = []
    # Identify B name by difference from A on early lines
    name_b = None
    for l in conv_lines:
        if ":" not in l:
            continue
        spk = l.split(":", 1)[0].strip()
        if spk != name_a:
            name_b = spk
            break
    if name_b is None:
        name_b = "B"
    # Build mapping from A line index to next B line content
    for idx in a_line_indices:
        # find next line that starts with B
        next_b = None
        for j in range(idx + 1, len(conv_lines)):
            if conv_lines[j].startswith(f"{name_b}:"):
                next_b = conv_lines[j]
                break
            # if we encounter next A before B, stop
            if conv_lines[j].startswith(f"{name_a}:"):
                break
        if next_b is None:
            continue
        if idx in barrier_a_indices:
            barrier_b_lines.append(next_b)
        else:
            post_b_lines.append(next_b)
    return {
        "barrier_a_turns": target,
        "total_a_turns": total_a_turns,
        "barrier_b_lines": barrier_b_lines,
        "post_b_lines": post_b_lines,
    }

def _compute_simple_b_metrics(b_lines: List[str]) -> dict:
    if not b_lines:
        return {"count": 0, "question_rate": None, "avg_len": None}
    msgs = [l.split(":", 1)[1].strip() if ":" in l else l for l in b_lines]
    q_rate = sum(1 for m in msgs if "?" in m) / len(msgs) if msgs else None
    avg_len = float(np.mean([len(m) for m in msgs])) if msgs else None
    return {"count": len(msgs), "question_rate": q_rate, "avg_len": avg_len}

def perform_language_barrier_analysis(experiment_data: Dict[str, Any], base_results_dir: str, output_dir: str) -> pd.DataFrame:
    """Analyze conversation logs for language_barrier experiments and relate metrics to ratio."""
    rows = []
    for exp_name in sorted(experiment_data.keys()):
        if "language_barrier" not in exp_name:
            # Include non-language-barrier experiments with empty cells for alignment
            rows.append({
                "experiment": exp_name,
                "ratio": None,
                "scenarios": experiment_data.get(exp_name, {}).get("scenario_count", 0),
                "barrier_a_turns_mean": None,
                "b_bar_question_rate_mean": None,
                "b_post_question_rate_mean": None,
                "b_bar_avg_len_mean": None,
                "b_post_avg_len_mean": None,
            })
            continue
        ratio = _extract_ratio_from_name(exp_name)
        exp_path = os.path.join(base_results_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue
        scenario_dirs = [d for d in os.listdir(exp_path) if d.startswith("scenario_")]
        per_scn = []
        for s in scenario_dirs:
            sdir = os.path.join(exp_path, s)
            conv = _load_conversation_lines(sdir)
            if not conv:
                continue
            name_a, name_b = _get_speakers(conv)
            seg = _split_barrier_segments(conv, name_a, ratio if ratio is not None else 1.0)
            b_bar = _compute_simple_b_metrics(seg["barrier_b_lines"])
            b_post = _compute_simple_b_metrics(seg["post_b_lines"])
            per_scn.append({
                "scenario": s,
                "barrier_a_turns": seg["barrier_a_turns"],
                "total_a_turns": seg["total_a_turns"],
                "b_bar_count": b_bar["count"],
                "b_bar_question_rate": b_bar["question_rate"],
                "b_bar_avg_len": b_bar["avg_len"],
                "b_post_count": b_post["count"],
                "b_post_question_rate": b_post["question_rate"],
                "b_post_avg_len": b_post["avg_len"],
            })
        if per_scn:
            df = pd.DataFrame(per_scn)
            row = {
                "experiment": exp_name,
                "ratio": ratio,
                "scenarios": len(per_scn),
                "barrier_a_turns_mean": float(df["barrier_a_turns"].mean()),
                "b_bar_question_rate_mean": float(df["b_bar_question_rate"].dropna().mean()) if df["b_bar_question_rate"].notna().any() else None,
                "b_post_question_rate_mean": float(df["b_post_question_rate"].dropna().mean()) if df["b_post_question_rate"].notna().any() else None,
                "b_bar_avg_len_mean": float(df["b_bar_avg_len"].dropna().mean()) if df["b_bar_avg_len"].notna().any() else None,
                "b_post_avg_len_mean": float(df["b_post_avg_len"].dropna().mean()) if df["b_post_avg_len"].notna().any() else None,
            }
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    out_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "language_barrier_summary.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"🧪 Language barrier summary saved: {csv_path}")
    # Plot ratio vs question rate (barrier phase)
    try:
        fig, ax = plt.subplots(figsize=(7,5))
        subset = out_df.dropna(subset=["ratio", "b_bar_question_rate_mean"]) if "b_bar_question_rate_mean" in out_df else out_df
        if not subset.empty:
            ax.scatter(subset["ratio"], subset["b_bar_question_rate_mean"], c="tab:blue")
            ax.set_xlabel("Barrier Ratio")
            ax.set_ylabel("Agent B Question Rate (barrier phase)")
            ax.set_title("Barrier ratio vs question rate")
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "language_barrier_ratio_vs_question_rate.png")
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"📈 Saved: {plot_path}")
    except Exception as e:
        print(f"Plotting error: {e}")
    return out_df

def main():
    """Main function to run the analysis."""
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = current_dir
    output_dir = os.path.join(current_dir, 'analysis_output')
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔍 Loading experiment data...")
    experiment_data = load_experiment_data(results_dir)
    
    if not experiment_data:
        print("❌ No experiment data found!")
        return
    
    print(f"\n📊 Found {len(experiment_data)} experiments")
    for exp_name, exp_data in experiment_data.items():
        print(f"  - {exp_name}: {exp_data['scenario_count']} scenarios")
    
    print("\n🧮 Calculating social performance averages...")
    social_results = calculate_social_performance_averages(experiment_data)
    
    print("🧮 Calculating MCQ performance averages...")
    mcq_results = calculate_mcq_averages(experiment_data)
    
    print("🧮 Calculating detailed MCQ performance averages...")
    mcq_detailed_results = calculate_mcq_detailed_averages(experiment_data)
    
    print("\n📋 Creating summary report...")
    summary_df = create_summary_report(social_results, mcq_results, mcq_detailed_results, output_dir)
    
    print("💾 Saving detailed results...")
    save_detailed_results(social_results, mcq_results, mcq_detailed_results, output_dir)
    
    print("📄 Saving individual experiment summaries...")
    save_individual_experiment_summaries(social_results, mcq_results, mcq_detailed_results, output_dir)
    
    print("📈 Creating visualizations...")
    create_visualizations(social_results, mcq_results, output_dir)

    # New: Automatic language barrier evaluation across all experiments
    print("\n🧪 Running language barrier analysis across experiments...")
    lb_df = perform_language_barrier_analysis(experiment_data, results_dir, output_dir)
    if lb_df is not None and not lb_df.empty:
        print(lb_df.head())
    
    print(f"\n✅ Analysis complete! Results saved in: {output_dir}")
    print("\nFiles created:")
    print("  - experiment_averages_summary.csv (summary table)")
    print("  - detailed_averages.json (detailed statistics)")
    print("  - social_performance_comparison.png (visualization)")
    print("  - [experiment_name]_summary.json (individual experiment summaries)")
    print("  - [experiment_name]_detailed_summary.json (detailed individual summaries)")
    print("  - [experiment_name]_summary.txt (human-readable individual summaries)")
    print(f"\n📊 Processed {len(experiment_data)} experiments:")
    for exp_name, exp_data in experiment_data.items():
        print(f"  - {exp_name}: {exp_data['scenario_count']} scenarios")

if __name__ == "__main__":
    main()