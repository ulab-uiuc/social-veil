import matplotlib.pyplot as plt
from typing import List, Dict

def plot_reasoning_scores(tom_scores: List[Dict], agent_names: List[str], save_path: str = None):
    metrics = ['bleu', 'rouge', 'bertscore']
    rounds = [score["round"] for score in tom_scores]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for agent in agent_names:
            values = [score.get(agent, {}).get(metric, 0.0) for score in tom_scores]
            ax.plot(
                rounds,
                values,
                label=agent,
                marker='o',
                linewidth=2
            )
        ax.set_title(metric.capitalize(), fontsize=14)
        ax.set_xlabel("Conversation Round", fontsize=12)
        ax.set_ylabel("Score (0–10)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10)

    plt.suptitle("Theory of Mind Metric Trends Over Conversation", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Saved reasoning score plot to {save_path}")
    else:
        plt.show()