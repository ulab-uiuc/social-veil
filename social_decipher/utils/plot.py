import matplotlib.pyplot as plt
import os
import numpy as np
    
from typing import List, Dict, Any

def plot_reasoning_scores(
    tom_scores: list[dict], agent_names: list[str], save_path: str = None
):
    metrics = ["bleu", "rouge", "bertscore", "llmscore"]
    metric_titles = {
        "bleu": "BLEU",
        "rouge": "ROUGE-L",
        "bertscore": "BERTScore",
        "llmscore": "LLM-Based ToM Score",
    }

    rounds = [score["round"] for score in tom_scores]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for agent in agent_names:
            values = [score.get(agent, {}).get(metric, 0.0) for score in tom_scores]
            ax.plot(rounds, values, label=agent, marker="o", linewidth=2)
        ax.set_title(metric_titles[metric], fontsize=14)
        ax.set_xlabel("Conversation Round", fontsize=12)
        ax.set_ylabel("Score (0–10)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=10)

    plt.suptitle("Theory of Mind Metric Trends Over Conversation", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✅ Saved reasoning score plot to {save_path}")
    else:
        plt.show()


def plot_mcq_scores(mcq_scores: List[Dict], agent_names: List[str], save_path: str = None):
    rounds = [entry["round"] for entry in mcq_scores]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    for i, agent in enumerate(agent_names):
        for j, mcq_type in enumerate(["goal_mcq", "reason_mcq"]):
            confidences = [
                entry.get(f"{agent}_{mcq_type}", {}).get("confidence", 0.0)
                for entry in mcq_scores
            ]
            corrects = [
                entry.get(f"{agent}_{mcq_type}", {}).get("correct", False)
                for entry in mcq_scores
            ]

            ax = axes[i][j]
            ax.plot(rounds, confidences, label="Confidence", color="blue", linewidth=2)

            for r, conf, correct in zip(rounds, confidences, corrects):
                marker_style = 'o' if correct else 'x'
                ax.scatter(
                    r,
                    conf,
                    marker=marker_style,
                    color="black",
                    s=100,
                    alpha=0.8
                )

            ax.set_title(f"{agent} - {mcq_type.replace('_', ' ').title()}")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Confidence")
            ax.grid(True)

            # Add custom legend for correctness
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', label='Correct',
                           markerfacecolor='black', markersize=10),
                plt.Line2D([0], [0], marker='x', color='w', label='Incorrect',
                           markeredgecolor='black', markersize=10)
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=9)

    axes[1][0].set_xlabel("Conversation Round")
    axes[1][1].set_xlabel("Conversation Round")
    plt.suptitle("MCQ Prediction Confidence and Correctness Over Rounds", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✅ Saved MCQ score plot to {save_path}")
    else:
        plt.show()

def plot_social_goal(
    eval_result: Dict[str, Any],
    agent_names: List[str],
    save_dir: str = "../social_decipher/results/"
) -> None:
    """
    Create visualizations for a single social goal evaluation result
    
    Args:
        eval_result: Evaluation result dictionary
        agent_names: Names of the two agents
        save_dir: Directory to save the visualization files
    """
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Goal Completion Scores
    plot_goal_completion(eval_result, agent_names, save_dir)
    
    # 2. SOTOPIA Dimensions Radar Charts
    plot_sotopia_dimensions(eval_result, agent_names, save_dir)
    
    # 3. Overall Performance Bar Chart
    plot_overall_performance(eval_result, agent_names, save_dir)
    
    # 4. Interaction Quality
    plot_interaction_quality(eval_result, save_dir)
    
    print(f"✅ Social goal visualizations saved to {save_dir}")


def plot_goal_completion(
    eval_result: Dict[str, Any],
    agent_names: List[str],
    save_dir: str
) -> None:
    """Plot goal completion scores for both agents"""
    # Extract scores
    agent1_score = eval_result["social_performance"]["agent_1"]["goal_completion"]["score"]
    agent2_score = eval_result["social_performance"]["agent_2"]["goal_completion"]["score"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.bar(agent_names, [agent1_score, agent2_score], color=['#3498db', '#e74c3c'], width=0.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12)
    
    # Add goal text
    if "goal_focus" in eval_result and "agent_1" in eval_result["goal_focus"]:
        # Use data from goal_focus if available
        agent1_goal = eval_result["goal_focus"]["agent_1"].get("goal_restated", "").split(".")[0]
        agent2_goal = eval_result["goal_focus"]["agent_2"].get("goal_restated", "").split(".")[0]
    else:
        # Use reasoning from social_performance as fallback
        agent1_goal = eval_result["social_performance"]["agent_1"]["goal_completion"]["reasoning"]
        agent2_goal = eval_result["social_performance"]["agent_2"]["goal_completion"]["reasoning"]
    
    # Wrap text for better display
    def wrap_text(text, max_len=40):
        words = text.split()
        lines = []
        current_line = []
        current_len = 0
        
        for word in words:
            if current_len + len(word) + len(current_line) <= max_len:
                current_line.append(word)
                current_len += len(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_len = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    # Add goal text annotations
    plt.figtext(0.25, 0.02, f"{agent_names[0]}'s Goal:\n{wrap_text(agent1_goal)}", 
                wrap=True, fontsize=9, ha='center')
    plt.figtext(0.75, 0.02, f"{agent_names[1]}'s Goal:\n{wrap_text(agent2_goal)}", 
                wrap=True, fontsize=9, ha='center')
    
    # Customize chart
    ax.set_ylabel('Goal Completion Score (0-10)', fontsize=12)
    ax.set_title('Goal Completion Assessment', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 10.5)  # Max score is 10
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Make room for text at bottom
    plt.savefig(os.path.join(save_dir, "goal_completion.png"), dpi=300)
    plt.close()


def plot_sotopia_dimensions(
    eval_result: Dict[str, Any],
    agent_names: List[str],
    save_dir: str
) -> None:
    """Create radar charts for SOTOPIA dimensions"""
    # Define dimensions and normalization functions
    dimensions = [
        "goal_completion", 
        "believability",
        "relationship", 
        "knowledge", 
        "secret", 
        "social_rules",
        "financial_benefits"
    ]
    
    # Function to normalize scores to 0-10 scale for the radar chart
    def normalize_score(dimension: str, score: float) -> float:
        if dimension == "relationship":
            # -5 to 5 scale → 0 to 10 scale
            return (score + 5) * 1.0
        elif dimension in ["secret", "social_rules"]:
            # -10 to 0 scale → 0 to 10 scale
            return (score + 10) * 1.0
        else:
            # Already on 0-10 scale
            return score
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    N = len(dimensions)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)  # Fixed angular positions
    theta = np.append(theta, theta[0])  # Close the loop
    
    formatted_labels = [dim.replace('_', ' ').title() for dim in dimensions]
    
    colors = ['#3498db', '#e74c3c']  # Blue and red
    
    for i, agent_name in enumerate(agent_names):
        agent_key = f"agent_{i+1}"
        
        agent_data = eval_result["social_performance"][agent_key]
        values = [normalize_score(dim, agent_data[dim]["score"]) for dim in dimensions]
        values = np.append(values, values[0])  # Close the loop
        
        ax.plot(theta, values, 'o-', linewidth=2, label=agent_name, color=colors[i])
        ax.fill(theta, values, alpha=0.25, color=colors[i])

    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(formatted_labels, fontsize=12)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=10)
    ax.grid(True)
    
    plt.title("SOTOPIA Dimensions Evaluation", size=16, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sotopia_dimensions.png"), dpi=300)
    plt.close()
    
    for i, agent_name in enumerate(agent_names):
        agent_key = f"agent_{i+1}"
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        agent_data = eval_result["social_performance"][agent_key]
        values = [normalize_score(dim, agent_data[dim]["score"]) for dim in dimensions]
        values = np.append(values, values[0])  # Close the loop
        
        ax.plot(theta, values, 'o-', linewidth=2, color=colors[i])
        ax.fill(theta, values, alpha=0.3, color=colors[i])
        
        ax.set_xticks(theta[:-1])
        ax.set_xticklabels(formatted_labels, fontsize=12)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=10)
        ax.grid(True)
        
        plt.title(f"{agent_name}: SOTOPIA Dimensions", size=14, fontweight='bold', y=1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{agent_name}_dimensions.png"), dpi=300)
        plt.close()


def plot_overall_performance(
    eval_result: Dict[str, Any],
    agent_names: List[str],
    save_dir: str
) -> None:
    """Plot overall performance scores for both agents"""
    # Extract scores
    agent1_score = eval_result["social_performance"]["agent_1"]["overall_score"]
    agent2_score = eval_result["social_performance"]["agent_2"]["overall_score"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(agent_names))
    bars = ax.barh(y_pos, [agent1_score, agent2_score], color=['#3498db', '#e74c3c'], height=0.5)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width}',
                   xy=(width, bar.get_y() + bar.get_height()/2),
                   xytext=(3, 0),  # 3 points horizontal offset
                   textcoords="offset points",
                   ha='left', va='center', fontsize=12)
    
    # Customize chart
    ax.set_yticks(y_pos)
    ax.set_yticklabels(agent_names, fontsize=12)
    ax.set_xlabel('Overall Score', fontsize=12)
    ax.set_title('Overall Agent Performance', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 10.5)  # Assuming max score is 10
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "overall_performance.png"), dpi=300)
    plt.close()


def plot_interaction_quality(
    eval_result: Dict[str, Any],
    save_dir: str
) -> None:
    """Plot interaction quality gauge chart"""
    # Extract score and reasoning
    score = eval_result["social_performance"]["interaction_quality"]["score"]
    reasoning = eval_result["social_performance"]["interaction_quality"]["reasoning"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
    
    # Define gauge properties
    angles = np.linspace(0, 2*np.pi, 100)
    
    # Convert score to angle (assuming score range 0-10)
    max_score = 10
    angle = (score / max_score) * np.pi
    
    # Create gauge
    ax.plot(angles, [1]*len(angles), color='lightgray', linewidth=10)
    ax.plot(np.linspace(0, angle, 10), [1]*10, color='#2ecc71', linewidth=10)  # Green
    
    # Add score text in center
    ax.text(0, 0, f"{score}", fontsize=36, ha='center', va='center', fontweight='bold')
    ax.text(0, -0.3, "Interaction Quality", fontsize=14, ha='center', va='center')
    
    # Remove ticks and spines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    
    ax.set_ylim(0, 1.2)
    
    plt.figtext(0.5, 0.1, f"Assessment: {reasoning}", wrap=True, ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "interaction_quality.png"), dpi=300)
    plt.close()