import matplotlib.pyplot as plt
from typing import List, Dict

def plot_reasoning_scores(tom_scores: List[Dict], agent_names: List[str], save_path: str = None):

    metrics = ['bleu', 'rouge', 'bertscore']
    rounds = [score["round"] for score in tom_scores]

    plt.figure(figsize=(12, 6))

    for agent in agent_names:
        for metric in metrics:
            values = [score.get(agent, {}).get(metric, 0.0) * 10.0 for score in tom_scores]
            plt.plot(
                rounds,
                values,
                label=f"{agent} {metric}",
                marker='o',
                linewidth=2
            )

    plt.xlabel("Conversation Round", fontsize=12)
    plt.ylabel("Evaluation Metric Score", fontsize=12)
    plt.title("Theory of Mind Metric Trends Over Conversation", fontsize=14)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Saved reasoning score plot to {save_path}")
    else:
        plt.show()
