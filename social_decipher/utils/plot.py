import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_mcq_scores(
    mcq_scores: list[dict], agent_names: list[str], save_path: str = None
):
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

            for r, conf, correct in zip(rounds, confidences, corrects, strict=False):
                marker_style = "o" if correct else "x"
                ax.scatter(
                    r, conf, marker=marker_style, color="black", s=100, alpha=0.8
                )

            ax.set_title(f"{agent} - {mcq_type.replace('_', ' ').title()}")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Confidence")
            ax.grid(True)

            # Add custom legend for correctness
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Correct",
                    markerfacecolor="black",
                    markersize=10,
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="x",
                    color="w",
                    label="Incorrect",
                    markeredgecolor="black",
                    markersize=10,
                ),
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
    eval_result: dict[str, Any],
    agent_names: list[str],
    save_dir: str = "../social_decipher/results/",
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
    eval_result: dict[str, Any], agent_names: list[str], save_dir: str
) -> None:
    """Plot goal completion scores for both agents"""
    # Extract scores
    agent1_score = eval_result["social_performance"]["agent_1"]["goal_completion"][
        "score"
    ]
    agent2_score = eval_result["social_performance"]["agent_2"]["goal_completion"][
        "score"
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar chart
    bars = ax.bar(
        agent_names,
        [agent1_score, agent2_score],
        color=["#3498db", "#e74c3c"],
        width=0.5,
    )

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # Use reasoning from social_performance as fallback
    agent1_goal = eval_result["social_performance"]["agent_1"]["goal_completion"][
        "reasoning"
    ]
    agent2_goal = eval_result["social_performance"]["agent_2"]["goal_completion"][
        "reasoning"
    ]

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
                lines.append(" ".join(current_line))
                current_line = [word]
                current_len = len(word)

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)

    # Add goal text annotations
    plt.figtext(
        0.25,
        0.02,
        f"{agent_names[0]}'s Goal:\n{wrap_text(agent1_goal)}",
        wrap=True,
        fontsize=9,
        ha="center",
    )
    plt.figtext(
        0.75,
        0.02,
        f"{agent_names[1]}'s Goal:\n{wrap_text(agent2_goal)}",
        wrap=True,
        fontsize=9,
        ha="center",
    )

    # Customize chart
    ax.set_ylabel("Goal Completion Score (0-10)", fontsize=12)
    ax.set_title("Goal Completion Assessment", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 10.5)  # Max score is 10
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Save figure
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Make room for text at bottom
    plt.savefig(os.path.join(save_dir, "goal_completion.png"), dpi=300)
    plt.close()

def plot_sotopia_dimensions(
    eval_result: dict[str, Any], agent_names: list[str], save_dir: str
) -> None:
    """Create radar charts for social dimensions matching the evaluation metrics"""
    # Update dimensions to match Social_Goal_Evaluation prompt
    dimensions = [
        "goal_completion",
        "believability",
        "relationship",
        "knowledge",
        "secret",  # Changed from information_exchange
        "social_rules",
        "financial_benefits",  # Changed from communication_strategy
    ]

    # Function to normalize scores to 0-10 scale for the radar chart
    def normalize_score(dimension: str, score: float) -> float:
        if dimension == "relationship":
            # -5 to 5 scale → 0 to 10 scale
            return (score + 5) * 1.0
        elif dimension in ["social_rules", "secret"]:
            # -10 to 0 scale → 0 to 10 scale
            return (score + 10) * 1.0
        elif dimension == "financial_benefits":
            # -5 to 5 scale → 0 to 10 scale
            return (score + 5) * 1.0
        else:
            # Already on 0-10 scale (goal_completion, believability, knowledge)
            return score

    # Handle missing dimensions gracefully
    def get_score(agent_data, dimension):
        if dimension in agent_data:
            return agent_data[dimension].get("score", 0)
        return 0  # Default value if dimension doesn't exist

    # IMPROVED COMBINED RADAR CHART
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)

    N = len(dimensions)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)  # Fixed angular positions
    theta = np.append(theta, theta[0])  # Close the loop

    # Prettier dimension labels
    formatted_labels = [
        "Goal Completion",
        "Believability",
        "Relationship",
        "Knowledge",
        "Secret Keeping",
        "Social Rules",
        "Financial Benefits"
    ]

    colors = ["#3498db", "#e74c3c"]  # Blue and red
    markers = ["o", "s"]  # Circle for agent 1, square for agent 2
    linestyles = ["-", "--"]  # Solid for agent 1, dashed for agent 2

    # Draw background grid with better visibility
    ax.grid(True, color='gray', alpha=0.3, linestyle='--')
    
    # Draw circular gridlines with labels
    for r in [2, 4, 6, 8]:
        circle = plt.Circle((0, 0), r, transform=ax.transData._b, 
                            fill=False, color='gray', alpha=0.1, linestyle='-')
        ax.add_artist(circle)
    
    # Add score range text along one radius
    ax.text(0, 5.2, "5", fontsize=8, ha='center', va='center', color='gray')
    ax.text(0, 10.2, "10", fontsize=8, ha='center', va='center', color='gray')

    # Plot each agent with enhanced visibility
    for i, agent_name in enumerate(agent_names):
        agent_key = f"agent_{i+1}"

        agent_data = eval_result["social_performance"][agent_key]
        values = [normalize_score(dim, get_score(agent_data, dim)) for dim in dimensions]
        values = np.append(values, values[0])  # Close the loop

        # Plot with enhanced styling
        line = ax.plot(theta, values, markers[i], linewidth=2.5, 
                       label=agent_name, color=colors[i], 
                       linestyle=linestyles[i], markersize=7)
        ax.fill(theta, values, alpha=0.15, color=colors[i])
        
        # Add score labels at each point
        for j, value in enumerate(values[:-1]):  # Skip the last duplicated point
            angle = theta[j]
            # Adjust label position based on value to avoid overlapping
            offset = 0.5
            x = (value + offset) * np.cos(angle)
            y = (value + offset) * np.sin(angle)
            ax.text(x, y, f"{agent_data.get(dimensions[j], {}).get('score', 0):.1f}", 
                    color=colors[i], fontsize=9, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Enhance the dimension labels
    for i, label in enumerate(formatted_labels):
        angle = theta[i]
        # Position labels slightly outside the plot
        x = 1.2 * np.cos(angle)
        y = 1.2 * np.sin(angle)
        # Adjust text alignment based on position
        ha = 'center'
        if angle < np.pi/2 or angle > 3*np.pi/2:
            ha = 'left'
        elif angle > np.pi/2 and angle < 3*np.pi/2:
            ha = 'right'
            
        ax.text(x, y, formatted_labels[i], fontsize=11, fontweight='bold', 
                ha=ha, va='center')

    # Remove default tick labels since we have our custom ones
    ax.set_xticklabels([])
    ax.set_ylim(0, 10)
    ax.set_yticks([])  # Remove radial ticks
    
    # Better title and legend
    plt.title("Social Dimensions Comparison", size=16, fontweight="bold", y=1.05)
    legend = plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1), 
                        fontsize=12, framealpha=0.9)
    legend.get_frame().set_edgecolor('gray')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sotopia_dimensions.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Individual agent radar charts - keep this as in original
    for i, agent_name in enumerate(agent_names):
        agent_key = f"agent_{i+1}"

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)

        agent_data = eval_result["social_performance"][agent_key]
        values = [normalize_score(dim, get_score(agent_data, dim)) for dim in dimensions]
        values = np.append(values, values[0])  # Close the loop

        ax.plot(theta, values, "o-", linewidth=2, color=colors[i])
        ax.fill(theta, values, alpha=0.3, color=colors[i])

        ax.set_xticks(theta[:-1])
        ax.set_xticklabels(formatted_labels, fontsize=12)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=10)
        ax.grid(True)

        plt.title(
            f"{agent_name}: Social Dimensions", size=14, fontweight="bold", y=1.1
        )

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{agent_name}_dimensions.png"), dpi=300)
        plt.close()

def plot_overall_performance(
    eval_result: dict[str, Any], agent_names: list[str], save_dir: str
) -> None:
    """Plot overall performance scores for both agents"""
    # Extract scores
    agent1_score = eval_result["social_performance"]["agent_1"]["overall_score"]
    agent2_score = eval_result["social_performance"]["agent_2"]["overall_score"]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create horizontal bar chart
    y_pos = np.arange(len(agent_names))
    bars = ax.barh(
        y_pos, [agent1_score, agent2_score], color=["#3498db", "#e74c3c"], height=0.5
    )

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),  # 3 points horizontal offset
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=12,
        )

    # Customize chart
    ax.set_yticks(y_pos)
    ax.set_yticklabels(agent_names, fontsize=12)
    ax.set_xlabel("Overall Score", fontsize=12)
    ax.set_title("Overall Agent Performance", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 10.5)  # Assuming max score is 10
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "overall_performance.png"), dpi=300)
    plt.close()


def plot_interaction_quality(eval_result: dict[str, Any], save_dir: str) -> None:
    quality_data = eval_result["social_performance"]["interaction_quality"]
    score = quality_data["score"]
    reasoning = quality_data["reasoning"]
    
    barrier_navigation = quality_data.get("barrier_navigation", "")
    cooperation = quality_data.get("cooperation", "")

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    # Define gauge properties
    angles = np.linspace(0, 2 * np.pi, 100)

    # Convert score to angle (assuming score range 0-10)
    max_score = 10
    angle = (score / max_score) * np.pi

    # Create gauge
    ax.plot(angles, [1] * len(angles), color="lightgray", linewidth=10)
    ax.plot(np.linspace(0, angle, 10), [1] * 10, color="#2ecc71", linewidth=10)  # Green

    # Add score text in center
    ax.text(0, 0, f"{score}", fontsize=36, ha="center", va="center", fontweight="bold")
    ax.text(0, -0.3, "Interaction Quality", fontsize=14, ha="center", va="center")

    # Remove ticks and spines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    ax.set_ylim(0, 1.2)

    # Add detailed assessment
    assessment_text = f"Assessment: {reasoning}"
    
    # Add barrier navigation details if available
    if barrier_navigation:
        assessment_text += f"\n\nBarrier Navigation: {barrier_navigation}"
    if cooperation:
        assessment_text += f"\n\nCooperation: {cooperation}"
        
    plt.figtext(
        0.5, 0.1, assessment_text, wrap=True, ha="center", fontsize=10
    )

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "interaction_quality.png"), dpi=300)
    plt.close()


def plot_cross_scenario_performance(metrics, agent_names, save_dir):
    """
    Plot performance metrics across scenarios
    
    Args:
        metrics: Dictionary containing cross-scenario metrics
        agent_names: List of agent names [agent_a_name, agent_b_name]
        save_dir: Directory to save the plot
        
    Returns:
        None (saves the plot to disk)
    """
    plt.figure(figsize=(12, 8))

    # Plot goal achievement scores
    plt.subplot(2, 2, 1)
    plt.plot(
        metrics["scenario_idx"],
        metrics[f"{agent_names[0]}_goal_score"],
        "b-",
        label=f"{agent_names[0]} Goal",
    )
    plt.plot(
        metrics["scenario_idx"],
        metrics[f"{agent_names[1]}_goal_score"],
        "r-",
        label=f"{agent_names[1]} Goal",
    )
    plt.xlabel("Scenario")
    plt.ylabel("Goal Achievement Score")
    plt.title("Goal Achievement Across Scenarios")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True)

    # Plot reason understanding scores
    plt.subplot(2, 2, 2)
    plt.plot(
        metrics["scenario_idx"],
        metrics[f"{agent_names[0]}_reason_understanding"],
        "b-",
        label=f"{agent_names[0]} Reason",
    )
    plt.plot(
        metrics["scenario_idx"],
        metrics[f"{agent_names[1]}_reason_understanding"],
        "r-",
        label=f"{agent_names[1]} Reason",
    )
    plt.xlabel("Scenario")
    plt.ylabel("Reason Understanding Score")
    plt.title("Reason Understanding Across Scenarios")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True)

    # Plot MCQ goal accuracy
    plt.subplot(2, 2, 3)
    plt.plot(
        metrics["scenario_idx"],
        metrics[f"{agent_names[0]}_mcq_goal_accuracy"],
        "b-",
        label=f"{agent_names[0]} Goal MCQ",
    )
    plt.plot(
        metrics["scenario_idx"],
        metrics[f"{agent_names[1]}_mcq_goal_accuracy"],
        "r-",
        label=f"{agent_names[1]} Goal MCQ",
    )
    plt.xlabel("Scenario")
    plt.ylabel("MCQ Goal Accuracy")
    plt.title("Goal Detection Accuracy Across Scenarios")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True)

    # Plot MCQ reason accuracy
    plt.subplot(2, 2, 4)
    plt.plot(
        metrics["scenario_idx"],
        metrics[f"{agent_names[0]}_mcq_reason_accuracy"],
        "b-",
        label=f"{agent_names[0]} Reason MCQ",
    )
    plt.plot(
        metrics["scenario_idx"],
        metrics[f"{agent_names[1]}_mcq_reason_accuracy"],
        "r-",
        label=f"{agent_names[1]} Reason MCQ",
    )
    plt.xlabel("Scenario")
    plt.ylabel("MCQ Reason Accuracy")
    plt.title("Reason Detection Accuracy Across Scenarios")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cross_scenario_performance.png"))
    plt.close()