import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

def load_eval_results(base_dir="."):
    """
    Load evaluation results from different experimental conditions.
    Returns a dictionary with experiment names as keys and evaluation data as values.
    """
    experiments = {
        "No Encryption, No Action": "exp_no_encryption_no_action",
        "Mapping Encryption, No Action": "exp_mapping_encryption_no_action",
        "Mapping Encryption, Action": "exp_mapping_encryption_action"
    }
    
    results = {}
    
    for exp_name, exp_dir in experiments.items():
        eval_file = os.path.join(base_dir, exp_dir, "scenario_1", "eval_result.json")
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                results[exp_name] = json.load(f)
                print(f"Loaded {exp_name} evaluation data")
        else:
            print(f"Warning: Could not find evaluation data for {exp_name} at {eval_file}")
    
    return results

def plot_goal_completion(eval_results, output_dir="./results"):
    """
    Create a bar chart comparing goal completion scores across experiments.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    experiments = list(eval_results.keys())
    agent_scores = []
    
    # Extract agent goal scores
    for exp_name in experiments:
        scores = [
            eval_results[exp_name]["aggregated_scores"]["agent_1"]["goal_completion"],
            eval_results[exp_name]["aggregated_scores"]["agent_2"]["goal_completion"]
        ]
        agent_scores.append(scores)
    
    # Set up bar chart
    x = np.arange(len(experiments))
    width = 0.35
    
    # Plot bars
    ax.bar(x - width/2, [scores[0] for scores in agent_scores], width, label='Agent 1 (Alex)')
    ax.bar(x + width/2, [scores[1] for scores in agent_scores], width, label='Agent 2 (Jamie)')
    
    # Customize chart
    ax.set_title('Goal Completion Comparison Across Experiments', fontsize=16)
    ax.set_xlabel('Experimental Condition', fontsize=14)
    ax.set_ylabel('Goal Completion Score (0-10)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=15, ha='right')
    ax.legend()
    
    # Set y-axis to use integers only
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, 10)
    
    # Add score values on top of bars
    for i, scores in enumerate(agent_scores):
        ax.text(i - width/2, scores[0] + 0.2, f"{scores[0]}", ha='center', fontsize=10)
        ax.text(i + width/2, scores[1] + 0.2, f"{scores[1]}", ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, "goal_completion_comparison.png")
    plt.savefig(output_path)
    print(f"Saved goal completion chart to {output_path}")
    
    return fig

def plot_overall_scores(eval_results, output_dir="./results"):
    """
    Create multiple subplots comparing overall scores across experiments.
    Each dimension gets its own subplot for better readability.
    """
    # Extract data
    experiments = list(eval_results.keys())
    metrics = ["believability", "relationship", "overall"]
    
    # Create a dictionary to hold all metrics
    data = {
        "Agent 1": {metric: [] for metric in metrics},
        "Agent 2": {metric: [] for metric in metrics}
    }
    
    # Extract metrics
    for exp_name in experiments:
        for metric in metrics:
            data["Agent 1"][metric].append(eval_results[exp_name]["aggregated_scores"]["agent_1"][metric])
            data["Agent 2"][metric].append(eval_results[exp_name]["aggregated_scores"]["agent_2"][metric])
    
    # Also add interaction quality
    interaction_quality = [eval_results[exp_name]["aggregated_scores"]["interaction_quality"] 
                          for exp_name in experiments]
    
    # Set up the plot with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Define consistent colors
    agent1_color = "#1f77b4"  # Blue
    agent2_color = "#ff7f0e"  # Orange
    interaction_color = "#17becf"  # Cyan
    
    # Bar width
    width = 0.35
    x = np.arange(len(experiments))
    
    # Plot 1: Believability
    axes[0].bar(x - width/2, data["Agent 1"]["believability"], width, label="Agent 1", color=agent1_color)
    axes[0].bar(x + width/2, data["Agent 2"]["believability"], width, label="Agent 2", color=agent2_color)
    axes[0].set_title('Believability', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(experiments, rotation=15, ha='right')
    axes[0].set_ylabel('Score (0-10)', fontsize=12)
    axes[0].set_ylim(0, 10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].legend()
    
    # Add value labels
    for i, v in enumerate(data["Agent 1"]["believability"]):
        axes[0].text(i - width/2, v + 0.1, str(v), ha='center')
    for i, v in enumerate(data["Agent 2"]["believability"]):
        axes[0].text(i + width/2, v + 0.1, str(v), ha='center')
    
    # Plot 2: Relationship
    axes[1].bar(x - width/2, data["Agent 1"]["relationship"], width, label="Agent 1", color=agent1_color)
    axes[1].bar(x + width/2, data["Agent 2"]["relationship"], width, label="Agent 2", color=agent2_color)
    axes[1].set_title('Relationship', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(experiments, rotation=15, ha='right')
    axes[1].set_ylabel('Score (0-10)', fontsize=12)
    axes[1].set_ylim(0, 10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].legend()
    
    # Add value labels
    for i, v in enumerate(data["Agent 1"]["relationship"]):
        axes[1].text(i - width/2, v + 0.1, str(v), ha='center')
    for i, v in enumerate(data["Agent 2"]["relationship"]):
        axes[1].text(i + width/2, v + 0.1, str(v), ha='center')
    
    # Plot 3: Overall score
    axes[2].bar(x - width/2, data["Agent 1"]["overall"], width, label="Agent 1", color=agent1_color)
    axes[2].bar(x + width/2, data["Agent 2"]["overall"], width, label="Agent 2", color=agent2_color)
    axes[2].set_title('Overall Performance', fontsize=14)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(experiments, rotation=15, ha='right')
    axes[2].set_ylabel('Score (0-10)', fontsize=12)
    axes[2].set_ylim(0, 10)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    axes[2].legend()
    
    # Add value labels
    for i, v in enumerate(data["Agent 1"]["overall"]):
        axes[2].text(i - width/2, v + 0.1, str(v), ha='center')
    for i, v in enumerate(data["Agent 2"]["overall"]):
        axes[2].text(i + width/2, v + 0.1, str(v), ha='center')
    
    # Plot 4: Interaction Quality
    axes[3].bar(x, interaction_quality, width, label="Interaction Quality", color=interaction_color)
    axes[3].set_title('Interaction Quality', fontsize=14)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(experiments, rotation=15, ha='right')
    axes[3].set_ylabel('Score (0-10)', fontsize=12)
    axes[3].set_ylim(0, 10)
    axes[3].grid(axis='y', linestyle='--', alpha=0.7)
    axes[3].legend()
    
    # Add value labels
    for i, v in enumerate(interaction_quality):
        axes[3].text(i, v + 0.1, str(v), ha='center')
    
    # Add main title
    fig.suptitle('Performance Metrics Comparison Across Experimental Conditions', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust to make room for suptitle
    
    # Save figure
    output_path = os.path.join(output_dir, "overall_metrics_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved overall metrics chart to {output_path}")
    
    return fig

def plot_mcq_performance(base_dir=".", output_dir="./results"):
    """
    Create visualizations of MCQ performance over conversation rounds
    and compare across different experimental conditions.
    """
    experiments = {
        "No Encryption, No Action": "exp_no_encryption_no_action",
        "Mapping Encryption, No Action": "exp_mapping_encryption_no_action",
        "Mapping Encryption, Action": "exp_mapping_encryption_action"
    }
    
    # Load MCQ data for each experiment
    mcq_data = {}
    for exp_name, exp_dir in experiments.items():
        mcq_file = os.path.join(base_dir, exp_dir, "scenario_1", "mcq_logs.json")
        if os.path.exists(mcq_file):
            with open(mcq_file, 'r') as f:
                mcq_logs = json.load(f)
                mcq_data[exp_name] = mcq_logs
                print(f"Loaded {exp_name} MCQ data")
        else:
            print(f"Warning: Could not find MCQ data for {exp_name} at {mcq_file}")
    
    if not mcq_data:
        print("No MCQ data found, skipping MCQ performance plots")
        return None
    
    # Create a figure with multiple subplots for different aspects of MCQ performance
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Define colors for experiments
    colors = {
        "No Encryption, No Action": "#1f77b4",  # Blue
        "Mapping Encryption, No Action": "#ff7f0e",  # Orange
        "Mapping Encryption, Action": "#2ca02c"  # Green
    }
    
    # Plot 1: Overall MCQ accuracy comparison (as a bar chart)
    # Calculate overall accuracy for each experiment
    accuracy_data = {}
    for exp_name, logs in mcq_data.items():
        total_mcqs = len(logs) * 4  # 4 MCQs per round (2 agents × 2 aspects)
        total_correct = 0
        
        alex_goal_correct = 0
        alex_reason_correct = 0
        jamie_goal_correct = 0
        jamie_reason_correct = 0
        
        for log in logs:
            if log.get("Alex_goal_mcq", {}).get("correct", False):
                total_correct += 1
                alex_goal_correct += 1
            if log.get("Alex_reason_mcq", {}).get("correct", False):
                total_correct += 1
                alex_reason_correct += 1
            if log.get("Jamie_goal_mcq", {}).get("correct", False):
                total_correct += 1
                jamie_goal_correct += 1
            if log.get("Jamie_reason_mcq", {}).get("correct", False):
                total_correct += 1
                jamie_reason_correct += 1
        
        total_rounds = len(logs)
        if total_rounds == 0:
            continue
        
        accuracy_data[exp_name] = {
            "overall": (total_correct / total_mcqs) * 100,
            "alex_goal": (alex_goal_correct / total_rounds) * 100,
            "alex_reason": (alex_reason_correct / total_rounds) * 100,
            "jamie_goal": (jamie_goal_correct / total_rounds) * 100,
            "jamie_reason": (jamie_reason_correct / total_rounds) * 100
        }
    
    # Create the overall accuracy bar chart
    x = np.arange(len(accuracy_data))
    width = 0.15
    
    # Plot bars for each MCQ type
    axes[0].bar(x - width*2, [accuracy_data[exp]["alex_goal"] for exp in accuracy_data], 
                width, label="Alex Goal", color="#1f77b4")
    axes[0].bar(x - width, [accuracy_data[exp]["alex_reason"] for exp in accuracy_data], 
                width, label="Alex Reason", color="#aec7e8")
    axes[0].bar(x, [accuracy_data[exp]["jamie_goal"] for exp in accuracy_data], 
                width, label="Jamie Goal", color="#ff7f0e")
    axes[0].bar(x + width, [accuracy_data[exp]["jamie_reason"] for exp in accuracy_data], 
                width, label="Jamie Reason", color="#ffbb78")
    axes[0].bar(x + width*2, [accuracy_data[exp]["overall"] for exp in accuracy_data], 
                width, label="Overall", color="#2ca02c")
    
    # Customize chart
    axes[0].set_title('MCQ Accuracy by Experimental Condition', fontsize=14)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(list(accuracy_data.keys()), rotation=15, ha='right')
    axes[0].set_ylim(0, 105)  # Allow a little space above 100%
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].legend()
    
    # Add value labels
    for i, exp in enumerate(accuracy_data):
        axes[0].text(i - width*2, accuracy_data[exp]["alex_goal"] + 1, 
                     f"{accuracy_data[exp]['alex_goal']:.0f}%", ha='center', va='bottom', fontsize=8)
        axes[0].text(i - width, accuracy_data[exp]["alex_reason"] + 1, 
                     f"{accuracy_data[exp]['alex_reason']:.0f}%", ha='center', va='bottom', fontsize=8)
        axes[0].text(i, accuracy_data[exp]["jamie_goal"] + 1, 
                     f"{accuracy_data[exp]['jamie_goal']:.0f}%", ha='center', va='bottom', fontsize=8)
        axes[0].text(i + width, accuracy_data[exp]["jamie_reason"] + 1, 
                     f"{accuracy_data[exp]['jamie_reason']:.0f}%", ha='center', va='bottom', fontsize=8)
        axes[0].text(i + width*2, accuracy_data[exp]["overall"] + 1, 
                     f"{accuracy_data[exp]['overall']:.0f}%", ha='center', va='bottom', fontsize=8)
    
    # Plot 2: MCQ confidence over rounds (Alex)
    # Track average confidence by round
    for exp_name, logs in mcq_data.items():
        rounds = [log["round"] for log in logs]
        goal_confidence = [log["Alex_goal_mcq"]["confidence"] * 100 for log in logs]
        reason_confidence = [log["Alex_reason_mcq"]["confidence"] * 100 for log in logs]
        
        axes[1].plot(rounds, goal_confidence, 'o-', label=f"{exp_name} - Goal", color=colors[exp_name])
        axes[1].plot(rounds, reason_confidence, 's--', label=f"{exp_name} - Reason", 
                     color=colors[exp_name], alpha=0.7)
    
    axes[1].set_title('Alex MCQ Confidence Over Rounds', fontsize=14)
    axes[1].set_xlabel('Conversation Round', fontsize=12)
    axes[1].set_ylabel('Confidence (%)', fontsize=12)
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()
    
    # Plot 3: MCQ confidence over rounds (Jamie)
    for exp_name, logs in mcq_data.items():
        rounds = [log["round"] for log in logs]
        goal_confidence = [log["Jamie_goal_mcq"]["confidence"] * 100 for log in logs]
        reason_confidence = [log["Jamie_reason_mcq"]["confidence"] * 100 for log in logs]
        
        axes[2].plot(rounds, goal_confidence, 'o-', label=f"{exp_name} - Goal", color=colors[exp_name])
        axes[2].plot(rounds, reason_confidence, 's--', label=f"{exp_name} - Reason", 
                     color=colors[exp_name], alpha=0.7)
    
    axes[2].set_title('Jamie MCQ Confidence Over Rounds', fontsize=14)
    axes[2].set_xlabel('Conversation Round', fontsize=12)
    axes[2].set_ylabel('Confidence (%)', fontsize=12)
    axes[2].set_ylim(0, 105)
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].legend()
    
    # Plot 4: Correct answers over time
    for exp_name, logs in mcq_data.items():
        rounds = [log["round"] for log in logs]
        
        # Calculate cumulative correct answers by round
        alex_goal_correct = [1 if log["Alex_goal_mcq"]["correct"] else 0 for log in logs]
        alex_reason_correct = [1 if log["Alex_reason_mcq"]["correct"] else 0 for log in logs]
        jamie_goal_correct = [1 if log["Jamie_goal_mcq"]["correct"] else 0 for log in logs]
        jamie_reason_correct = [1 if log["Jamie_reason_mcq"]["correct"] else 0 for log in logs]
        
        alex_goal_cumulative = np.cumsum(alex_goal_correct)
        alex_reason_cumulative = np.cumsum(alex_reason_correct)
        jamie_goal_cumulative = np.cumsum(jamie_goal_correct)
        jamie_reason_cumulative = np.cumsum(jamie_reason_correct)
        
        # Plot lines for each type
        axes[3].plot(rounds, alex_goal_cumulative, 'o-', 
                     label=f"{exp_name} - Alex Goal", color=colors[exp_name])
        axes[3].plot(rounds, jamie_goal_cumulative, 's--', 
                     label=f"{exp_name} - Jamie Goal", color=colors[exp_name], alpha=0.7)
    
    axes[3].set_title('Cumulative Correct Goal Understanding', fontsize=14)
    axes[3].set_xlabel('Conversation Round', fontsize=12)
    axes[3].set_ylabel('Cumulative Correct Answers', fontsize=12)
    axes[3].grid(True, linestyle='--', alpha=0.7)
    axes[3].legend()
    
    # Add main title
    fig.suptitle('MCQ Performance Analysis Across Experimental Conditions', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust to make room for suptitle
    
    # Save figure
    output_path = os.path.join(output_dir, "mcq_performance_analysis.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved MCQ performance analysis to {output_path}")
    
    return fig

def generate_comparative_analysis(eval_results, output_dir="./results"):
    """
    Generate a summary of the comparative analysis and save it to a file.
    """
    summary = "# Comparative Analysis of Social Agents Across Experimental Conditions\n\n"
    
    # Compare goal completion
    summary += "## Goal Completion Comparison\n\n"
    for exp_name, data in eval_results.items():
        agent1_score = data["aggregated_scores"]["agent_1"]["goal_completion"]
        agent2_score = data["aggregated_scores"]["agent_2"]["goal_completion"]
        summary += f"### {exp_name}\n"
        summary += f"- Alex (Agent 1): {agent1_score}/10\n"
        summary += f"- Jamie (Agent 2): {agent2_score}/10\n"
        summary += f"- Goal Achievement: Alex: {'Yes' if data.get('agent0_goal_achieved', False) else 'No'}, "
        summary += f"Jamie: {'Yes' if data.get('agent1_goal_achieved', False) else 'No'}\n\n"
    
    # Compare believability
    summary += "## Believability Comparison\n\n"
    for exp_name, data in eval_results.items():
        agent1_score = data["aggregated_scores"]["agent_1"]["believability"]
        agent2_score = data["aggregated_scores"]["agent_2"]["believability"]
        summary += f"### {exp_name}\n"
        summary += f"- Alex (Agent 1): {agent1_score}/10\n"
        summary += f"- Jamie (Agent 2): {agent2_score}/10\n\n"
    
    # Compare relationship scores
    summary += "## Relationship Quality Comparison\n\n"
    for exp_name, data in eval_results.items():
        agent1_score = data["aggregated_scores"]["agent_1"]["relationship"]
        agent2_score = data["aggregated_scores"]["agent_2"]["relationship"]
        summary += f"### {exp_name}\n"
        summary += f"- Alex (Agent 1): {agent1_score}/10\n"
        summary += f"- Jamie (Agent 2): {agent2_score}/10\n\n"
    
    # Compare interaction quality
    summary += "## Interaction Quality Comparison\n\n"
    for exp_name, data in eval_results.items():
        score = data["aggregated_scores"]["interaction_quality"]
        summary += f"### {exp_name}\n"
        summary += f"- Overall Interaction Quality: {score}/10\n\n"
    
    # Save summary to file
    output_path = os.path.join(output_dir, "comparative_analysis.md")
    with open(output_path, 'w') as f:
        f.write(summary)
    print(f"Saved comparative analysis to {output_path}")
    
    return summary

def main():
    print("Starting analysis of social agent experiments...")
    
    # Create output directory if it doesn't exist
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load evaluation results
    eval_results = load_eval_results()
    
    if not eval_results:
        print("No evaluation data found. Make sure the paths are correct.")
        return
    
    # Generate plots
    plot_goal_completion(eval_results, output_dir)
    plot_overall_scores(eval_results, output_dir)
    plot_mcq_performance(output_dir=output_dir)  # Updated function
    
    # Generate and save comparative analysis
    generate_comparative_analysis(eval_results, output_dir)
    
    print("Analysis complete! Check the results directory for output files.")

if __name__ == "__main__":
    main()