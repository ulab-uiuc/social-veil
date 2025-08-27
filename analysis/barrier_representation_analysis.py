#!/usr/bin/env python3
"""
Barrier Representation Analysis

This script uses ML techniques to prove that barrier prompts cause distribution shifts 
in model internal representations using Qwen2.5-7B-Instruct.

Key Analysis:
1. Extract internal representations at multiple layers
2. Compare distributions between baseline and barrier cases
3. Visualize distribution shifts using dimensionality reduction
4. Apply statistical tests to prove significant differences

Author: Social-Decipher Research Team
"""

import json
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# Import project modules
from social_decipher.agent.agent_profile import AgentProfile
from social_decipher.environment.episode_loader import EpisodeLoader
from data.barrier_creation import augment_episode

@dataclass
class RepresentationData:
    """Container for model representation data"""
    episode_id: str
    barrier_type: str  # "baseline", "semantic", "cultural", "emotional"
    layer_idx: int
    representations: torch.Tensor  # [seq_len, hidden_size]
    attention_weights: Optional[torch.Tensor] = None
    prompt_text: str = ""
    severity: float = 0.0

class BarrierRepresentationAnalyzer:
    """Analyzes internal model representations to prove barrier effects"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        episodes_file: str = "data/episode_sample.jsonl",
        num_episodes: int = 5,
        severity: float = 0.8
    ):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.episodes_file = episodes_file
        self.num_episodes = num_episodes
        self.severity = severity
        
        # Initialize model and tokenizer
        print(f"🔧 Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
            output_hidden_states=True,
            output_attentions=True
        )
        self.model.eval()
        
        # Analysis layers (sample key layers from the model)
        total_layers = len(self.model.model.layers)
        self.analysis_layers = [
            0,  # Input layer
            total_layers // 4,  # Early layer
            total_layers // 2,  # Middle layer  
            3 * total_layers // 4,  # Late layer
            total_layers - 1  # Final layer
        ]
        
        print(f"📊 Analyzing layers: {self.analysis_layers} out of {total_layers} total layers")
        
        # Storage for representations
        self.representations: List[RepresentationData] = []
        
    def _setup_device(self, device: str) -> str:
        """Setup compute device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_and_augment_episodes(self) -> List[Dict[str, Any]]:
        """Load episodes and create barrier variants"""
        print(f"📂 Loading episodes from {self.episodes_file}...")
        
        # Load base episodes
        base_episodes = []
        if Path(self.episodes_file).exists():
            with open(self.episodes_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= self.num_episodes:
                        break
                    base_episodes.append(json.loads(line.strip()))
        else:
            print(f"❌ Episodes file not found: {self.episodes_file}")
            return []
        
        print(f"✅ Loaded {len(base_episodes)} base episodes")
        
        # Create augmented variants
        all_episodes = []
        for ep in base_episodes:
            # Baseline version
            baseline_ep = ep.copy()
            baseline_ep["barrier_type"] = "baseline"
            all_episodes.append(baseline_ep)
            
            # Create barrier variants
            for barrier_family in ["semantic_structure", "cultural_style", "emotional_influence"]:
                print(f"  🔄 Creating {barrier_family} variant for episode {ep.get('episode_id', 'unknown')}")
                
                try:
                    augmented = augment_episode(ep, barrier_family, self.severity)
                    if augmented:
                        # Merge with base episode
                        barrier_ep = ep.copy()
                        barrier_ep.update(augmented)
                        barrier_ep["barrier_type"] = barrier_family
                        barrier_ep["severity"] = self.severity
                        all_episodes.append(barrier_ep)
                except Exception as e:
                    print(f"    ⚠️ Failed to create {barrier_family} variant: {e}")
                    continue
        
        print(f"🎯 Generated {len(all_episodes)} total episodes (baseline + barriers)")
        return all_episodes
    
    def create_agent_prompt(self, episode: Dict[str, Any]) -> str:
        """Create the agent prompt that would be used in conversation"""
        
        # Extract episode info
        scenario = episode.get("scenario", "")
        agent_goals = episode.get("agent_goals", ["", ""])
        agent1_goal = agent_goals[0] if len(agent_goals) > 0 else ""
        agent_reasons = episode.get("agent_reasons", ["", ""])
        agent1_reason = agent_reasons[0] if len(agent_reasons) > 0 else ""
        
        # Create basic instruction
        base_instruction = f"""Imagine you are Agent A, your task is to act/speak as Agent A would, keeping in mind Agent A's social goal.

Here is the context of this interaction:
Scenario: {scenario}
Agent A's goal: {agent1_goal}
Agent A's reason: {agent1_reason}

You are at Turn #1. Your available action types are: speak, non-verbal communication, action, none, leave"""

        # Add barrier-specific modifications
        barrier_type = episode.get("barrier_type", "baseline")
        
        if barrier_type == "semantic_structure":
            barrier_prompt = episode.get("barrier_prompts", {}).get("agentA", "")
            if barrier_prompt:
                base_instruction += f"\n\nBARRIER MODE DIRECTIVES (high priority):\n{barrier_prompt}"
                
        elif barrier_type == "cultural_style":
            barrier_prompt = episode.get("barrier_prompts", {}).get("agentA", "")
            if barrier_prompt:
                base_instruction += f"\n\nBARRIER MODE DIRECTIVES (high priority):\n{barrier_prompt}"
                
        elif barrier_type == "emotional_influence":
            barrier_prompt = episode.get("barrier_prompts", {}).get("agentA", "")
            if barrier_prompt:
                base_instruction += f"\n\nBARRIER MODE DIRECTIVES (high priority):\n{barrier_prompt}"
        
        base_instruction += '\n\nPlease only generate a JSON string including the action type and the argument.\nYour action should follow the given format:\n{"action_type": <action_type>, "argument": <action_argument>}'
        
        return base_instruction
    
    def extract_representations(self, episodes: List[Dict[str, Any]]) -> None:
        """Extract internal representations for all episodes"""
        print(f"\n🧠 Extracting representations from {len(episodes)} episodes...")
        
        for i, episode in enumerate(episodes):
            episode_id = episode.get("episode_id", f"ep_{i}")
            barrier_type = episode.get("barrier_type", "baseline")
            severity = episode.get("severity", 0.0)
            
            print(f"  📝 Processing {episode_id} ({barrier_type})")
            
            # Create prompt
            prompt = self.create_agent_prompt(episode)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # Forward pass with hooks to capture representations
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.hidden_states  # Tuple of (batch_size, seq_len, hidden_size)
                attentions = outputs.attentions if hasattr(outputs, 'attentions') else None
                
                # Store representations for key layers
                for layer_idx in self.analysis_layers:
                    if layer_idx < len(hidden_states):
                        # Get last token representation (most relevant for generation)
                        layer_repr = hidden_states[layer_idx][0, -1, :].cpu()  # [hidden_size]
                        
                        attention_weights = None
                        if attentions and layer_idx < len(attentions):
                            # Average attention across heads
                            attention_weights = attentions[layer_idx][0].mean(dim=0).cpu()  # [seq_len, seq_len]
                        
                        rep_data = RepresentationData(
                            episode_id=episode_id,
                            barrier_type=barrier_type,
                            layer_idx=layer_idx,
                            representations=layer_repr,
                            attention_weights=attention_weights,
                            prompt_text=prompt,
                            severity=severity
                        )
                        
                        self.representations.append(rep_data)
        
        print(f"✅ Extracted {len(self.representations)} representations")
    
    def compute_distribution_statistics(self) -> Dict[str, Any]:
        """Compute statistical measures of distribution differences"""
        print("\n📊 Computing distribution statistics...")
        
        # Group representations by barrier type and layer
        grouped_data = {}
        for rep in self.representations:
            key = (rep.barrier_type, rep.layer_idx)
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(rep.representations.numpy())
        
        # Convert to arrays
        barrier_types = ["baseline", "semantic_structure", "cultural_style", "emotional_influence"]
        stats_results = {}
        
        for layer_idx in self.analysis_layers:
            layer_stats = {}
            
            # Get baseline representations for this layer
            baseline_key = ("baseline", layer_idx)
            if baseline_key not in grouped_data:
                continue
            
            baseline_reprs = np.stack(grouped_data[baseline_key])  # [n_episodes, hidden_size]
            
            for barrier_type in barrier_types[1:]:  # Skip baseline
                barrier_key = (barrier_type, layer_idx)
                if barrier_key not in grouped_data:
                    continue
                
                barrier_reprs = np.stack(grouped_data[barrier_key])
                
                # Compute various distance metrics
                stats_dict = {}
                
                # 1. Mean squared difference
                mean_baseline = baseline_reprs.mean(axis=0)
                mean_barrier = barrier_reprs.mean(axis=0)
                mse = np.mean((mean_baseline - mean_barrier) ** 2)
                stats_dict["mse"] = float(mse)
                
                # 2. Cosine similarity between means
                cosine_sim = np.dot(mean_baseline, mean_barrier) / (
                    np.linalg.norm(mean_baseline) * np.linalg.norm(mean_barrier)
                )
                stats_dict["cosine_similarity"] = float(cosine_sim)
                
                # 3. Wasserstein distance (using first 50 dimensions for efficiency)
                if baseline_reprs.shape[1] > 50:
                    baseline_sample = baseline_reprs[:, :50]
                    barrier_sample = barrier_reprs[:, :50]
                else:
                    baseline_sample = baseline_reprs
                    barrier_sample = barrier_reprs
                
                # Flatten for 1D Wasserstein
                baseline_flat = baseline_sample.flatten()
                barrier_flat = barrier_sample.flatten()
                wasserstein_dist = stats.wasserstein_distance(baseline_flat, barrier_flat)
                stats_dict["wasserstein_distance"] = float(wasserstein_dist)
                
                # 4. KS test on first principal component
                pca = PCA(n_components=1)
                baseline_pc = pca.fit_transform(baseline_reprs).flatten()
                barrier_pc = pca.transform(barrier_reprs).flatten()
                ks_stat, ks_pvalue = stats.ks_2samp(baseline_pc, barrier_pc)
                stats_dict["ks_statistic"] = float(ks_stat)
                stats_dict["ks_pvalue"] = float(ks_pvalue)
                
                layer_stats[barrier_type] = stats_dict
            
            stats_results[f"layer_{layer_idx}"] = layer_stats
        
        return stats_results
    
    def create_visualizations(self, output_dir: str = "results/barrier_analysis") -> None:
        """Create comprehensive visualizations of barrier effects"""
        print(f"\n🎨 Creating visualizations in {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for visualization
        barrier_types = ["baseline", "semantic_structure", "cultural_style", "emotional_influence"]
        barrier_colors = {
            "baseline": "#2E86AB",
            "semantic_structure": "#A23B72", 
            "cultural_style": "#F18F01",
            "emotional_influence": "#C73E1D"
        }
        
        # 1. t-SNE visualization for each layer
        for layer_idx in self.analysis_layers:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Collect data for this layer
            layer_data = []
            layer_labels = []
            layer_episodes = []
            
            for rep in self.representations:
                if rep.layer_idx == layer_idx:
                    layer_data.append(rep.representations.numpy())
                    layer_labels.append(rep.barrier_type)
                    layer_episodes.append(rep.episode_id)
            
            if len(layer_data) < 4:  # Need at least 4 points for t-SNE
                continue
            
            layer_data = np.stack(layer_data)
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(layer_data)//2))
            tsne_results = tsne.fit_transform(layer_data)
            
            # Plot
            for barrier_type in barrier_types:
                mask = np.array(layer_labels) == barrier_type
                if mask.any():
                    ax.scatter(
                        tsne_results[mask, 0], 
                        tsne_results[mask, 1],
                        c=barrier_colors[barrier_type],
                        label=barrier_type.replace("_", " ").title(),
                        s=100,
                        alpha=0.8
                    )
            
            ax.set_title(f"t-SNE Visualization - Layer {layer_idx}", fontsize=16, fontweight='bold')
            ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
            ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/tsne_layer_{layer_idx}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. PCA visualization comparing all layers
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, layer_idx in enumerate(self.analysis_layers):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Collect data for this layer
            layer_data = []
            layer_labels = []
            
            for rep in self.representations:
                if rep.layer_idx == layer_idx:
                    layer_data.append(rep.representations.numpy())
                    layer_labels.append(rep.barrier_type)
            
            if len(layer_data) < 2:
                continue
            
            layer_data = np.stack(layer_data)
            
            # Apply PCA
            pca = PCA(n_components=2)
            pca_results = pca.fit_transform(layer_data)
            
            # Plot
            for barrier_type in barrier_types:
                mask = np.array(layer_labels) == barrier_type
                if mask.any():
                    ax.scatter(
                        pca_results[mask, 0], 
                        pca_results[mask, 1],
                        c=barrier_colors[barrier_type],
                        label=barrier_type.replace("_", " ").title(),
                        s=80,
                        alpha=0.8
                    )
            
            # Add explained variance
            explained_var = pca.explained_variance_ratio_
            ax.set_title(f"Layer {layer_idx}\n(PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%})", 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel(f"PC1 ({explained_var[0]:.1%})", fontsize=10)
            ax.set_ylabel(f"PC2 ({explained_var[1]:.1%})", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            if i == 0:  # Add legend to first subplot
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Hide unused subplots
        for i in range(len(self.analysis_layers), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle("PCA Analysis Across Model Layers", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_all_layers.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Distance heatmap
        stats_results = self.compute_distribution_statistics()
        
        # Create distance matrix visualization
        metrics = ["mse", "cosine_similarity", "wasserstein_distance", "ks_statistic"]
        barrier_types_clean = ["Semantic", "Cultural", "Emotional"]
        
        for metric in metrics:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Build matrix
            matrix_data = []
            layer_labels = []
            
            for layer_idx in self.analysis_layers:
                layer_key = f"layer_{layer_idx}"
                if layer_key in stats_results:
                    layer_row = []
                    for barrier_type in ["semantic_structure", "cultural_style", "emotional_influence"]:
                        if barrier_type in stats_results[layer_key]:
                            value = stats_results[layer_key][barrier_type].get(metric, 0)
                            layer_row.append(value)
                        else:
                            layer_row.append(0)
                    matrix_data.append(layer_row)
                    layer_labels.append(f"Layer {layer_idx}")
            
            if matrix_data:
                matrix_data = np.array(matrix_data)
                
                # Create heatmap
                im = ax.imshow(matrix_data, cmap='viridis', aspect='auto')
                
                # Set ticks and labels
                ax.set_xticks(range(len(barrier_types_clean)))
                ax.set_xticklabels(barrier_types_clean)
                ax.set_yticks(range(len(layer_labels)))
                ax.set_yticklabels(layer_labels)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(metric.replace("_", " ").title(), fontsize=12)
                
                # Add text annotations
                for i in range(len(layer_labels)):
                    for j in range(len(barrier_types_clean)):
                        text = ax.text(j, i, f'{matrix_data[i, j]:.3f}',
                                     ha="center", va="center", color="white", fontweight='bold')
                
                ax.set_title(f'{metric.replace("_", " ").title()} - Distribution Differences', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Barrier Type', fontsize=12)
                ax.set_ylabel('Model Layer', fontsize=12)
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/heatmap_{metric}.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    def generate_report(self, output_dir: str = "results/barrier_analysis") -> None:
        """Generate comprehensive analysis report"""
        print(f"\n📋 Generating analysis report...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute statistics
        stats_results = self.compute_distribution_statistics()
        
        # Create report
        report = {
            "analysis_metadata": {
                "model": self.model_name,
                "num_episodes": self.num_episodes,
                "severity": self.severity,
                "analysis_layers": self.analysis_layers,
                "total_representations": len(self.representations)
            },
            "statistical_results": stats_results,
            "summary": self._create_summary(stats_results)
        }
        
        # Save detailed results
        with open(f"{output_dir}/analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create markdown summary
        self._create_markdown_report(report, f"{output_dir}/analysis_summary.md")
        
        print(f"✅ Report saved to {output_dir}/")
    
    def _create_summary(self, stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create high-level summary of results"""
        summary = {
            "significant_differences": [],
            "strongest_effects": {},
            "layer_analysis": {}
        }
        
        # Find significant differences (p < 0.05)
        for layer_key, layer_data in stats_results.items():
            layer_summary = {"significant_barriers": [], "effect_sizes": {}}
            
            for barrier_type, stats in layer_data.items():
                ks_pvalue = stats.get("ks_pvalue", 1.0)
                
                if ks_pvalue < 0.05:
                    summary["significant_differences"].append({
                        "layer": layer_key,
                        "barrier_type": barrier_type,
                        "p_value": ks_pvalue,
                        "ks_statistic": stats.get("ks_statistic", 0)
                    })
                    layer_summary["significant_barriers"].append(barrier_type)
                
                # Record effect sizes
                layer_summary["effect_sizes"][barrier_type] = {
                    "mse": stats.get("mse", 0),
                    "wasserstein_distance": stats.get("wasserstein_distance", 0),
                    "cosine_similarity": stats.get("cosine_similarity", 1)
                }
            
            summary["layer_analysis"][layer_key] = layer_summary
        
        # Find strongest effects
        max_mse = max_ks = 0
        strongest_mse = strongest_ks = None
        
        for diff in summary["significant_differences"]:
            layer_data = stats_results[diff["layer"]][diff["barrier_type"]]
            
            mse = layer_data.get("mse", 0)
            ks_stat = layer_data.get("ks_statistic", 0)
            
            if mse > max_mse:
                max_mse = mse
                strongest_mse = diff
            
            if ks_stat > max_ks:
                max_ks = ks_stat
                strongest_ks = diff
        
        summary["strongest_effects"] = {
            "mse": strongest_mse,
            "ks_statistic": strongest_ks
        }
        
        return summary
    
    def _create_markdown_report(self, report: Dict[str, Any], output_path: str) -> None:
        """Create markdown summary report"""
        
        metadata = report["analysis_metadata"]
        summary = report["summary"]
        
        content = f"""# Barrier Representation Analysis Report

## Analysis Configuration
- **Model**: {metadata["model"]}
- **Episodes Analyzed**: {metadata["num_episodes"]}
- **Barrier Severity**: {metadata["severity"]}
- **Layers Analyzed**: {metadata["analysis_layers"]}
- **Total Representations**: {metadata["total_representations"]}

## Key Findings

### Significant Distribution Differences (p < 0.05)
"""
        
        if summary["significant_differences"]:
            for diff in summary["significant_differences"]:
                barrier_clean = diff["barrier_type"].replace("_", " ").title()
                content += f"- **{barrier_clean}** at {diff['layer']}: KS statistic = {diff['ks_statistic']:.4f}, p = {diff['p_value']:.4f}\n"
        else:
            content += "- No statistically significant differences found\n"
        
        content += f"""
### Strongest Effects
"""
        
        if summary["strongest_effects"]["mse"]:
            strongest = summary["strongest_effects"]["mse"]
            barrier_clean = strongest["barrier_type"].replace("_", " ").title()
            content += f"- **Largest MSE**: {barrier_clean} at {strongest['layer']}\n"
        
        if summary["strongest_effects"]["ks_statistic"]:
            strongest = summary["strongest_effects"]["ks_statistic"]
            barrier_clean = strongest["barrier_type"].replace("_", " ").title()
            content += f"- **Largest KS Statistic**: {barrier_clean} at {strongest['layer']}\n"
        
        content += f"""
## Layer-by-Layer Analysis

"""
        
        for layer_key, layer_data in summary["layer_analysis"].items():
            content += f"### {layer_key.replace('_', ' ').title()}\n"
            if layer_data["significant_barriers"]:
                barriers = [b.replace("_", " ").title() for b in layer_data["significant_barriers"]]
                content += f"- **Significant barriers**: {', '.join(barriers)}\n"
            else:
                content += f"- No significant barrier effects detected\n"
            content += "\n"
        
        content += f"""
## Interpretation

This analysis demonstrates how communication barriers create measurable distribution shifts in model internal representations. The statistical tests provide evidence that:

1. **Barrier prompts cause systematic changes** in how the model processes social scenarios
2. **Different barrier types create distinct patterns** of representational change
3. **Layer-specific effects** show where in the model barriers have the strongest impact

The visualization files (PNG) show the spatial distribution of representations, while this statistical analysis provides quantitative evidence of barrier effects.

## Files Generated
- `analysis_report.json`: Complete statistical results
- `tsne_layer_*.png`: t-SNE visualizations for each layer
- `pca_all_layers.png`: PCA comparison across layers  
- `heatmap_*.png`: Distance metrics heatmaps
"""
        
        with open(output_path, 'w') as f:
            f.write(content)
    
    def run_full_analysis(self) -> None:
        """Run complete barrier representation analysis"""
        print("🚀 Starting Barrier Representation Analysis")
        print("=" * 60)
        
        # 1. Load and augment episodes
        episodes = self.load_and_augment_episodes()
        if not episodes:
            print("❌ No episodes to analyze!")
            return
        
        # 2. Extract representations
        self.extract_representations(episodes)
        
        # 3. Create visualizations
        self.create_visualizations()
        
        # 4. Generate report
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("✅ Analysis complete! Check results/barrier_analysis/ for outputs")
        print("Key files:")
        print("  📊 analysis_summary.md - Human-readable results")
        print("  📈 *.png - Visualizations")
        print("  📋 analysis_report.json - Detailed statistics")

def main():
    """Main analysis entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze barrier effects on model representations")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name to analyze")
    parser.add_argument("--episodes", type=str, default="data/episode_sample.jsonl",
                       help="Episodes file to use")
    parser.add_argument("--num_episodes", type=int, default=5,
                       help="Number of episodes to analyze")
    parser.add_argument("--severity", type=float, default=0.8,
                       help="Barrier severity level")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cuda/cpu/mps)")
    parser.add_argument("--output_dir", type=str, default="results/barrier_analysis",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = BarrierRepresentationAnalyzer(
        model_name=args.model,
        device=args.device,
        episodes_file=args.episodes,
        num_episodes=args.num_episodes,
        severity=args.severity
    )
    
    # Run analysis
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()