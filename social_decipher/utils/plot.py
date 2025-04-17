import matplotlib.pyplot as plt

from typing import List, Dict

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