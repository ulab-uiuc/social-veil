import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def load_eval_results(base_dir="."):
    experiments = {
        "No Encryption, No Action": "exp_no_encryption_no_action",
        "Mapping Encryption, No Action": "exp_mapping_encryption_no_action",
        "Mapping Encryption, Action": "exp_mapping_encryption_action",
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
    
    # Updated metrics based on new evaluation format
    metrics = ["believability", "relationship", "information_exchange", 
               "communication_strategy", "overall"]
    
    # Create a dictionary to hold all metrics
    data = {
        "Agent 1": {metric: [] for metric in metrics},
        "Agent 2": {metric: [] for metric in metrics}
    }
    
    # Extract metrics
    for exp_name in experiments:
        for metric in metrics:
            data["Agent 1"][metric].append(eval_results[exp_name]["aggregated_scores"]["agent_1"].get(metric, 0))
            data["Agent 2"][metric].append(eval_results[exp_name]["aggregated_scores"]["agent_2"].get(metric, 0))
    
    # Also add interaction quality
    interaction_quality = [eval_results[exp_name]["aggregated_scores"]["interaction_quality"] 
                          for exp_name in experiments]
    
    # Set up the plot with subplots - need more for new metrics
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.flatten()
    
    # Define consistent colors
    agent1_color = "#1f77b4"  # Blue
    agent2_color = "#ff7f0e"  # Orange
    interaction_color = "#17becf"  # Cyan
    
    # Bar width
    width = 0.35
    x = np.arange(len(experiments))
    
    # Create a subplot for each metric
    for i, metric in enumerate(metrics):
        if i < len(axes) - 1:  # Save last subplot for interaction quality
            metric_name = metric.replace('_', ' ').title()
            
            axes[i].bar(x - width/2, data["Agent 1"][metric], width, label="Agent 1", color=agent1_color)
            axes[i].bar(x + width/2, data["Agent 2"][metric], width, label="Agent 2", color=agent2_color)
            axes[i].set_title(metric_name, fontsize=14)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(experiments, rotation=15, ha='right')
            axes[i].set_ylabel('Score (0-10)', fontsize=12)
            axes[i].set_ylim(0, 10)
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
            axes[i].legend()
            
            # Add value labels
            for j, v in enumerate(data["Agent 1"][metric]):
                axes[i].text(j - width/2, v + 0.1, str(v), ha='center')
            for j, v in enumerate(data["Agent 2"][metric]):
                axes[i].text(j + width/2, v + 0.1, str(v), ha='center')
    
    # Plot for Interaction Quality
    axes[-1].bar(x, interaction_quality, width, label="Interaction Quality", color=interaction_color)
    axes[-1].set_title('Interaction Quality', fontsize=14)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(experiments, rotation=15, ha='right')
    axes[-1].set_ylabel('Score (0-10)', fontsize=12)
    axes[-1].set_ylim(0, 10)
    axes[-1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[-1].legend()
    
    # Add value labels
    for i, v in enumerate(interaction_quality):
        axes[-1].text(i, v + 0.1, str(v), ha='center')
    
    # Add main title
    fig.suptitle('Performance Metrics Comparison Across Experimental Conditions', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust to make room for suptitle
    
    # Save figure
    output_path = os.path.join(output_dir, "overall_metrics_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved overall metrics chart to {output_path}")
    
    return fig

def plot_communication_strategies(eval_results, output_dir="./results"):
    """
    Create a figure specifically focusing on communication strategy and information exchange
    """
    experiments = list(eval_results.keys())
    agent1_comm_strategy = []
    agent2_comm_strategy = []
    agent1_info_exchange = []
    agent2_info_exchange = []
    
    for exp_name in experiments:
        agent1_comm_strategy.append(eval_results[exp_name]["aggregated_scores"]["agent_1"].get("communication_strategy", 0))
        agent2_comm_strategy.append(eval_results[exp_name]["aggregated_scores"]["agent_2"].get("communication_strategy", 0))
        agent1_info_exchange.append(eval_results[exp_name]["aggregated_scores"]["agent_1"].get("information_exchange", 0))
        agent2_info_exchange.append(eval_results[exp_name]["aggregated_scores"]["agent_2"].get("information_exchange", 0))
    
    # Set up figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot Communication Strategy
    x = np.arange(len(experiments))
    width = 0.35
    
    ax1.bar(x - width/2, agent1_comm_strategy, width, label='Agent 1', color='#1f77b4')
    ax1.bar(x + width/2, agent2_comm_strategy, width, label='Agent 2', color='#ff7f0e')
    ax1.set_title('Communication Strategy Adaptation', fontsize=14)
    ax1.set_xlabel('Experimental Condition', fontsize=12)
    ax1.set_ylabel('Score (0-10)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(experiments, rotation=15, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(agent1_comm_strategy):
        ax1.text(i - width/2, v + 0.1, str(v), ha='center')
    for i, v in enumerate(agent2_comm_strategy):
        ax1.text(i + width/2, v + 0.1, str(v), ha='center')
    
    # Plot Information Exchange
    ax2.bar(x - width/2, agent1_info_exchange, width, label='Agent 1', color='#1f77b4')
    ax2.bar(x + width/2, agent2_info_exchange, width, label='Agent 2', color='#ff7f0e')
    ax2.set_title('Information Exchange Effectiveness', fontsize=14)
    ax2.set_xlabel('Experimental Condition', fontsize=12)
    ax2.set_ylabel('Score (0-10)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(experiments, rotation=15, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 10)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(agent1_info_exchange):
        ax2.text(i - width/2, v + 0.1, str(v), ha='center')
    for i, v in enumerate(agent2_info_exchange):
        ax2.text(i + width/2, v + 0.1, str(v), ha='center')
    
    fig.suptitle('Communication Analysis Across Experimental Conditions', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    output_path = os.path.join(output_dir, "communication_analysis.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved communication analysis chart to {output_path}")
    
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

def plot_social_dimensions_comparison(eval_results, output_dir="./results"):
    """
    Create radar charts comparing each agent's performance across different experimental conditions.
    Uses radar plots to visualize multiple dimensions simultaneously.
    """
    # Set up figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), subplot_kw={'projection': 'polar'})
    fig.suptitle('Agent Performance Comparison Across Experimental Settings', fontsize=20, y=1.05)
    
    # Define dimensions for radar chart - ensure same order as in the data
    dimensions = [
        'Goal Completion', 
        'Believability', 
        'Relationship',
        'Social Rules',
        'Communication Strategy', 
        'Information Exchange',
        'Knowledge'
    ]
    
    # Set up angles for radar chart
    N = len(dimensions)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create a dictionary to hold data for each agent across experiments
    agent_data = {
        "Agent 1": {},  # Alex
        "Agent 2": {}   # Jamie
    }
    
    # Define consistent colors and line styles for experiments
    # Using a distinctive color palette
    colors = {
        "No Encryption, No Action": "#1f77b4",       # Blue
        "Mapping Encryption, No Action": "#ff7f0e",  # Orange
        "Mapping Encryption, Action": "#2ca02c",     # Green
        "Language Barrier, No Action": "#d62728",    # Red
        "Language Barrier, Action": "#9467bd"        # Purple
    }
    
    # Extract data for each agent and experiment
    for exp_name, data in eval_results.items():
        # Extract scores for each dimension for Agent 1 (Alex)
        agent1_scores = [
            data["aggregated_scores"]["agent_1"].get("goal_completion", 0),
            data["aggregated_scores"]["agent_1"].get("believability", 0),
            data["aggregated_scores"]["agent_1"].get("relationship", 0),
            data["aggregated_scores"]["agent_1"].get("social_rules", 0),
            data["aggregated_scores"]["agent_1"].get("communication_strategy", 0),
            data["aggregated_scores"]["agent_1"].get("information_exchange", 0),
            data["aggregated_scores"]["agent_1"].get("knowledge", 0)
        ]
        
        # Fill in missing values with reasonable defaults
        agent1_scores = [score if score > 0 else 5 for score in agent1_scores]
        
        # Make sure the loop is closed for plotting
        agent1_scores_closed = agent1_scores + [agent1_scores[0]]
        agent_data["Agent 1"][exp_name] = agent1_scores_closed
        
        # Extract scores for each dimension for Agent 2 (Jamie)
        agent2_scores = [
            data["aggregated_scores"]["agent_2"].get("goal_completion", 0),
            data["aggregated_scores"]["agent_2"].get("believability", 0),
            data["aggregated_scores"]["agent_2"].get("relationship", 0),
            data["aggregated_scores"]["agent_2"].get("social_rules", 0),
            data["aggregated_scores"]["agent_2"].get("communication_strategy", 0),
            data["aggregated_scores"]["agent_2"].get("information_exchange", 0),
            data["aggregated_scores"]["agent_2"].get("knowledge", 0)
        ]
        
        # Fill in missing values with reasonable defaults
        agent2_scores = [score if score > 0 else 5 for score in agent2_scores]
        
        # Make sure the loop is closed for plotting
        agent2_scores_closed = agent2_scores + [agent2_scores[0]]
        agent_data["Agent 2"][exp_name] = agent2_scores_closed
    
    # Function to set up each subplot
    def setup_radar_plot(ax, title):
        ax.set_theta_offset(np.pi / 2)  # Start from top
        ax.set_theta_direction(-1)  # Go clockwise
        
        # Set up the labels around the perimeter
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, fontsize=14)
        
        # Set y-limits and ticks
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=10)
        
        # Add grid lines
        ax.grid(True)
        
        # Add title
        ax.set_title(title, fontsize=18, pad=20)
    
    # Set up both axes
    setup_radar_plot(ax1, "Alex: Social Dimensions")
    setup_radar_plot(ax2, "Jamie: Social Dimensions")
    
    # Create a list of experiments in a specific order for consistent coloring
    experiment_order = [
        "No Encryption, No Action",
        "Mapping Encryption, No Action",
        "Mapping Encryption, Action",
        "Language Barrier, No Action",
        "Language Barrier, Action"
    ]
    
    # Keep track of which experiments are actually plotted
    plotted_experiments = []
    
    # Plot data for both agents in the specified order
    for exp_name in experiment_order:
        if exp_name in agent_data["Agent 1"] and exp_name in agent_data["Agent 2"]:
            ax1.plot(angles, agent_data["Agent 1"][exp_name], linewidth=2, linestyle='solid', 
                     color=colors[exp_name], label=exp_name)
            ax1.fill(angles, agent_data["Agent 1"][exp_name], color=colors[exp_name], alpha=0.1)
            
            ax2.plot(angles, agent_data["Agent 2"][exp_name], linewidth=2, linestyle='solid', 
                     color=colors[exp_name], label=exp_name)
            ax2.fill(angles, agent_data["Agent 2"][exp_name], color=colors[exp_name], alpha=0.1)
            
            plotted_experiments.append(exp_name)
    
    # Create a custom legend outside the plots
    # Get handles and labels from the first subplot
    handles, labels = ax1.get_legend_handles_labels()
    
    # Keep only the experiments that were actually plotted
    legend_handles = []
    legend_labels = []
    for exp_name in plotted_experiments:
        idx = labels.index(exp_name)
        legend_handles.append(handles[idx])
        legend_labels.append(exp_name)
    
    # Add the legend below the plots
    fig.legend(legend_handles, legend_labels, loc='upper center', 
               bbox_to_anchor=(0.5, 0.05), ncol=3, fontsize=14, 
               frameon=True, edgecolor='black')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the legend
    
    # Save the figure
    output_path = os.path.join(output_dir, "social_dimensions_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved social dimensions comparison chart to {output_path}")
    
    return fig

def generate_comparative_analysis(eval_results, output_dir="./results"):
    """
    Generate a summary of the comparative analysis and save it to a file.
    Updated for the new evaluation format.
    """
    summary = "# Comparative Analysis of SocialVeil Across Experimental Conditions\n\n"
    
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
    
    # Compare communication strategy
    summary += "## Communication Strategy Comparison\n\n"
    for exp_name, data in eval_results.items():
        agent1_score = data["aggregated_scores"]["agent_1"].get("communication_strategy", 0)
        agent2_score = data["aggregated_scores"]["agent_2"].get("communication_strategy", 0)
        summary += f"### {exp_name}\n"
        summary += f"- Alex (Agent 1): {agent1_score}/10\n"
        summary += f"- Jamie (Agent 2): {agent2_score}/10\n\n"
    
    # Compare information exchange
    summary += "## Information Exchange Effectiveness\n\n"
    for exp_name, data in eval_results.items():
        agent1_score = data["aggregated_scores"]["agent_1"].get("information_exchange", 0)
        agent2_score = data["aggregated_scores"]["agent_2"].get("information_exchange", 0)
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
        barrier_nav = data["social_performance"]["interaction_quality"].get("barrier_navigation", "Not evaluated")
        cooperation = data["social_performance"]["interaction_quality"].get("cooperation", "Not evaluated")
        
        summary += f"### {exp_name}\n"
        summary += f"- Overall Interaction Quality: {score}/10\n"
        summary += f"- Barrier Navigation: {barrier_nav}\n"
        summary += f"- Cooperation: {cooperation}\n\n"
    
    # Key observations
    summary += "## Key Observations\n\n"
    for exp_name, data in eval_results.items():
        observations = data["social_performance"].get("key_observations", [])
        summary += f"### {exp_name}\n"
        for obs in observations:
            summary += f"- {obs}\n"
        summary += "\n"
    
    # Save summary to file
    output_path = os.path.join(output_dir, "comparative_analysis.md")
    with open(output_path, 'w') as f:
        f.write(summary)
    print(f"Saved comparative analysis to {output_path}")
    
    return summary

def main():
    print("Starting analysis of SocialVeil experiments...")
    
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
    plot_communication_strategies(eval_results, output_dir)  # New function
    plot_mcq_performance(output_dir=output_dir)
    plot_social_dimensions_comparison(eval_results, output_dir)
    
    # Generate and save comparative analysis
    generate_comparative_analysis(eval_results, output_dir)
    
    print("Analysis complete! Check the results directory for output files.")

if __name__ == "__main__":
    main()