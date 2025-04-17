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
    """
    Plot the confidence scores for MCQ goal and reason predictions over conversation rounds.
    Each subplot shows the confidence trend for one MCQ type per agent.
    """
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
            axes[i][j].plot(rounds, confidences, label="Confidence", marker="o", color="blue")
            axes[i][j].scatter(
                rounds,
                confidences,
                c=["green" if c else "red" for c in corrects],
                s=100,
                label="Correctness",
                alpha=0.6,
                edgecolors="black"
            )
            axes[i][j].set_title(f"{agent} - {mcq_type.replace('_', ' ').title()}")
            axes[i][j].set_ylim(0, 1.05)
            axes[i][j].set_ylabel("Confidence")
            axes[i][j].grid(True)
            axes[i][j].legend()

    axes[1][0].set_xlabel("Conversation Round")
    axes[1][1].set_xlabel("Conversation Round")
    plt.suptitle("MCQ Prediction Confidence and Correctness Over Rounds", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✅ Saved MCQ score plot to {save_path}")
    else:
        plt.show()