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
from .load_existing_episodes import load_all_episodes
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import (
    pairwise_distances,
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import warnings
import yaml
warnings.filterwarnings('ignore')

# Ensure repository root is on sys.path for absolute imports like `social_decipher.*`
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# Import project modules (absolute imports to avoid relative-import issues)
from social_decipher.agent.agent_profile import AgentProfile
from social_decipher.environment.episode_loader import EpisodeLoader
from social_decipher.agent.social_agent import SocialAgent
from social_decipher.environment.env_profile import EnvironmentProfile
from social_decipher.utils.state import build_dynamic_rules_from_state

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
        episodes_file: str = "data/episode_all_neutralized.jsonl",
        severity: float = 0.8,
        output_dir: str = "preliminary_results/barrier_analysis"
    ):
        self.model_name = model_name
        self.device = self._setup_device(device)
        print(f"Using device: {self.device}")
        self.episodes_file = episodes_file
        self.severity = severity
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
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
            total_layers // 4,  # Early layer
            total_layers // 2,  # Middle layer  
            3 * total_layers // 4,  # Late layer
            total_layers - 1  # Final layer
        ]
        
        print(f"📊 Analyzing layers: {self.analysis_layers} out of {total_layers} total layers")
        
        # --- Centralized Barrier Configuration ---
        self.BARRIER_MAPPING = {
            "baseline": {"name": "Baseline", "color": "#0072B2"},
            "semantic_structure": {"name": "Semantic", "color": "#E69F00"},
            "cultural_style": {"name": "Cultural", "color": "#009E73"},
            "emotional_influence": {"name": "Emotional", "color": "#D55E00"},
        }
        self.BARRIER_ORDER = ["baseline", "semantic_structure", "cultural_style", "emotional_influence"]

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
        print(f"📂 Loading episodes from existing barrier files...")
        
        # Use the dedicated episode loader
        # Load all episodes (no limit)
        all_episodes = load_all_episodes(
            baseline_file=self.episodes_file,
            max_episodes=None
        )
        
        return all_episodes
    
    def create_agent_prompt(self, episode: Dict[str, Any]) -> str:
        # Build environment and prompts using SocialAgent to match live embedding
        scenario = episode.get("scenario", "")
        agent_goals = episode.get("agent_goals", ["", ""]) or ["", ""]
        agent_reasons = episode.get("agent_reasons", ["", ""]) or ["", ""]

        # Build agent profiles
        agent_profiles = episode.get("agent_profiles", [{}, {}])
        a_dict = agent_profiles[0] if len(agent_profiles) > 0 else {}
        b_dict = agent_profiles[1] if len(agent_profiles) > 1 else {}

        # Reconstruct agent profile string to match main pipeline
        agent_a_info = (
            f'{a_dict.get("first_name", "Agent")} {a_dict.get("last_name", "A")}, '
            f'a {a_dict.get("age", "N/A")}-year-old {a_dict.get("occupation", "person")}. '
            f'Public info: {a_dict.get("public_info", "")}'
        ).strip()

        environment = EnvironmentProfile(
            scenario=scenario,
            agent_goals=agent_goals,
            agent_reasons=agent_reasons,
            agent_goals_mcqas=episode.get("agent_goals_mcqas", []),
            agent_reasons_mcqas=episode.get("agent_reasons_mcqas", []),
            agent_knowledge_mcqas=episode.get("agent_knowledge_mcqas", []),
            agent_relationship=episode.get("agent_relationship", "friend"),
            agent1_private_knowledge=episode.get("agent1_private_knowledge", ""),
            agent2_private_knowledge=episode.get("agent2_private_knowledge", ""),
            agent1_profile=agent_a_info, # Pass reconstructed profile
        )

        # Attach barrier fields
        environment.env["barrier_type"] = episode.get("barrier_type", None)
        environment.env["barrier_prompts"] = episode.get("barrier_prompts", {})
        # Seed severity for analysis so banded rules reflect analyzer setting
        environment.env["barrier_state"] = {"severity": float(self.severity)}

        # Create agent profile objects
        agentA = AgentProfile.from_dict(a_dict, model_id=self.model_name)
        agentB = AgentProfile.from_dict(b_dict, model_id=self.model_name)

        # Use SocialAgent to construct the exact instruction as in runtime
        agent = SocialAgent(
            name=f"{agentA.first_name}",
            profile=agentA,
            partner_profile=agentB,
            env=environment,
            role_num=0,
        )

        prompt = agent.build_instruction(transcript="", turn_number=0)

        # The prompt from build_instruction is now the complete and final prompt.
        # The previous logic for injecting extra "extreme" rules has been removed
        # to ensure this analysis script perfectly matches the simulation.
        return prompt
    
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
            ).to(self.device)
            
            # Forward pass with hooks to capture representations
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.hidden_states  # Tuple of (batch_size, seq_len, hidden_size)
            
                # Store representations for key layers
                for layer_idx in self.analysis_layers:
                    if layer_idx < len(hidden_states):
                        layer_repr = hidden_states[layer_idx][0, -1, :].cpu()  # [hidden_size]
                        
                        rep_data = RepresentationData(
                            episode_id=episode_id,
                            barrier_type=barrier_type,
                            layer_idx=layer_idx,
                            representations=layer_repr,
                            prompt_text=prompt,
                            severity=severity
                        )
                        
                        self.representations.append(rep_data)
        
        print(f"✅ Extracted {len(self.representations)} representations")
    
    def _create_preliminary_visualization(self, output_dir: str, barrier_types: List[str], barrier_colors: Dict[str, str]) -> None:

        # Focus on middle layer for preliminary analysis (most informative)
        middle_layer = self.analysis_layers[len(self.analysis_layers) // 2]
        
        # Collect data for the middle layer
        layer_data = []
        layer_labels = []
        layer_episodes = []
        
        for rep in self.representations:
            if rep.layer_idx == middle_layer:
                layer_data.append(rep.representations.numpy())
                layer_labels.append(rep.barrier_type)
                layer_episodes.append(rep.episode_id)
        
        if len(layer_data) < 4:
            print("    ⚠️ Not enough data points for preliminary visualization")
            return
        
        layer_data = np.stack(layer_data)
        print(f"    📈 Analyzing {len(layer_data)} data points from layer {middle_layer}")
        
        # Apply PCA (2D like SafeSwitch paper)
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(layer_data)
        
        # Create the preliminary plot (matching SafeSwitch style)
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot each barrier type with clear separation
        for barrier_type in barrier_types:
            mask = np.array(layer_labels) == barrier_type
            if mask.any():
                x_coords = pca_results[mask, 0]
                y_coords = pca_results[mask, 1]
                
                ax.scatter(
                    x_coords, y_coords,
                    c=barrier_colors[barrier_type],
                    label=f"{barrier_type.replace('_', ' ').title()}",
                    s=120,
                    alpha=0.8,
                    edgecolors='white',
                    linewidth=1.5
                )
                
                # Add text annotations for some points to show episode IDs
                for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                    if i < 2:  # Annotate first 2 points per type
                        episode_ids = [ep for ep, label in zip(layer_episodes, layer_labels) if label == barrier_type]
                        if i < len(episode_ids):
                            ax.annotate(
                                episode_ids[i][:10] + "...",  # Truncate long IDs
                                (x, y),
                                xytext=(5, 5),
                                textcoords='offset points',
                                fontsize=8,
                                alpha=0.7
                            )
        
        # Formatting to match SafeSwitch paper style
        explained_var = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=14, fontweight='bold')
        ax.set_title(
            f'Visualization of {self.model_name.split("/")[-1]}\'s Hidden States\n'
            f'Using 2-Dimensional PCA (Layer {middle_layer})',
            fontsize=16, 
            fontweight='bold',
            pad=20
        )
        
        # Enhanced legend
        legend = ax.legend(
            title="Barrier Type",
            title_fontsize=12,
            fontsize=11,
            loc='upper right',
            frameon=True,
            fancybox=True,
            shadow=True
        )
        legend.get_title().set_fontweight('bold')
        
        # Grid and styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        # Add explanation text
        textstr = (
            f'Total variance explained: {sum(explained_var):.1%}\n'
            f'Data points: {len(layer_data)} episodes\n'
            f'Model layer: {middle_layer}/{len(self.model.model.layers)-1}'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/preliminary_internal_states_pca.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a summary statistics table like in SafeSwitch
        self._create_preliminary_stats_table(layer_data, layer_labels, barrier_types, output_dir)
        
        print(f"    ✅ Preliminary visualization saved to preliminary_internal_states_pca.png")
    
    def _create_preliminary_stats_table(self, data: np.ndarray, labels: List[str], barrier_types: List[str], output_dir: str) -> None:
        """Create summary statistics table for preliminary analysis"""
        
        # Compute centroid distances (like SafeSwitch paper analysis)
        centroids = {}
        for barrier_type in barrier_types:
            mask = np.array(labels) == barrier_type
            if mask.any():
                centroids[barrier_type] = data[mask].mean(axis=0)
        
        # Compute pairwise distances between centroids
        distance_matrix = {}
        for bt1 in barrier_types:
            if bt1 in centroids:
                distance_matrix[bt1] = {}
                for bt2 in barrier_types:
                    if bt2 in centroids:
                        dist = np.linalg.norm(centroids[bt1] - centroids[bt2])
                        distance_matrix[bt1][bt2] = float(dist)
        
        # Save as JSON for reference
        stats_summary = {
            "model": self.model_name,
            "analysis_type": "preliminary_internal_states",
            "centroid_distances": distance_matrix,
            "data_points_per_type": {bt: sum(1 for l in labels if l == bt) for bt in barrier_types}
        }
        
        with open(f"{output_dir}/preliminary_stats.json", 'w') as f:
            json.dump(stats_summary, f, indent=2)
        
        print(f"    📊 Preliminary statistics saved to preliminary_stats.json")
    
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
        barrier_types = self.BARRIER_ORDER
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
    
    def create_combined_publication_plot(self, output_dir: str) -> None:
        """
        Generate a combined, publication-quality plot with PCA and t-SNE side-by-side.
        """
        print(f"\n🎨 Creating combined publication plot in {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        # Use centralized config for colors and labels
        display_names = [self.BARRIER_MAPPING[bt]['name'] for bt in self.BARRIER_ORDER]
        colors = [self.BARRIER_MAPPING[bt]['color'] for bt in self.BARRIER_ORDER]
        name_to_color = {name: color for name, color in zip(display_names, colors)}
        label_to_name_multi = {i: name for i, name in enumerate(display_names)}

        layer_data_map = self._collect_layer_features()

        for layer_idx in self.analysis_layers:
            if layer_idx not in layer_data_map:
                continue

            X, y_bin, y_multi = layer_data_map[layer_idx]
            
            # --- Create Figure ---
            sns.set_theme(style="ticks", context="paper") # Use 'ticks' for a cleaner look
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

            # --- (a) PCA Plot ---
            scaler = StandardScaler(with_mean=True, with_std=True)
            Xs = scaler.fit_transform(X)
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(Xs)

            for lbl, name in label_to_name_multi.items():
                mask = (y_multi == lbl)
                if np.any(mask):
                    ax1.scatter(
                        X_pca[mask, 0], X_pca[mask, 1],
                        c=name_to_color[name], label=name,
                        s=60, alpha=0.85, edgecolors='black', linewidths=0.5
                    )
            
            # Fit and plot linear boundary
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X_pca, y_bin)
            w, b = clf.coef_[0], clf.intercept_[0]
            if abs(w[1]) > 1e-6:
                xs = np.linspace(X_pca[:, 0].min() - 5, X_pca[:, 0].max() + 5, 100)
                ys = (-w[0] / w[1]) * xs - b / w[1]
                ax1.plot(xs, ys, linestyle='--', color='#444444', linewidth=2.0, label='Baseline-Barrier Boundary')

            # PCA styling
            explained_var = pca.explained_variance_ratio_
            ax1.set_title(f"(a) PCA with Linear Boundary (Layer {layer_idx})", fontsize=18, fontweight='bold', pad=20)
            ax1.set_xlabel(f'Principal Component 1 ({explained_var[0]:.1%} Variance)', fontsize=14)
            ax1.set_ylabel(f'Principal Component 2 ({explained_var[1]:.1%} Variance)', fontsize=14)
            pca_legend = ax1.legend(title="Barrier Type", title_fontsize=13, fontsize=12)
            pca_legend.get_title().set_fontweight('bold')
            ax1.tick_params(axis='both', which='major', labelsize=12)

            # --- (b) t-SNE Plot ---
            perplexity = max(5, min(30, len(X) // 4))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, learning_rate='auto', init='pca')
            X_tsne = tsne.fit_transform(X)
            
            for lbl, name in label_to_name_multi.items():
                mask = (y_multi == lbl) # Use y_multi directly
                if mask.any():
                    x, y = X_tsne[mask, 0], X_tsne[mask, 1]
                    color = name_to_color[name]
                    ax2.scatter(
                        x, y, c=color, label=name, s=60,
                        alpha=0.85, edgecolors='black', linewidths=0.5
                    )
                    try: # Add convex hull
                        if x.size >= 3:
                            points = np.c_[x, y]
                            hull = ConvexHull(points)
                            patch = plt.Polygon(
                                points[hull.vertices], facecolor=color, alpha=0.15,
                                edgecolor=color, linewidth=1.5
                            )
                            ax2.add_patch(patch)
                    except Exception:
                        pass
            
            # t-SNE styling
            ax2.set_title(f"(b) t-SNE of Barrier Representations (Layer {layer_idx})", fontsize=18, fontweight='bold', pad=20)
            ax2.set_xlabel("t-SNE Dimension 1", fontsize=14)
            ax2.set_ylabel("t-SNE Dimension 2", fontsize=14)
            tsne_legend = ax2.legend(title="Barrier Type", title_fontsize=13, fontsize=12)
            tsne_legend.get_title().set_fontweight('bold')
            ax2.tick_params(axis='both', which='major', labelsize=12)

            # Despine for a cleaner look with 'ticks' style
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            
            # --- Finalize and Save ---
            plt.tight_layout(pad=2.0)
            plt.savefig(f"{output_dir}/combined_plot_layer_{layer_idx}.png", bbox_inches='tight')
            plt.close()

    def create_visualizations(self, output_dir: str = "preliminary_results/barrier_analysis") -> None:
        """
        DEPRECATED: This method is now replaced by create_combined_publication_plot.
        Kept for potential standalone t-SNE generation if needed.
        """
        print("\n🎨 Skipping standalone t-SNE plot generation. Use `create_combined_publication_plot` instead.")
        return

    def generate_report(self, output_dir: str = "preliminary_results/barrier_analysis") -> None:
        """Generate comprehensive analysis report"""
        print(f"\n📋 Generating analysis report...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute statistics
        stats_results = self.compute_distribution_statistics()
        svm_results = self._run_linear_probes(output_dir)
        
        # Create report
        report = {
            "analysis_metadata": {
                "model": self.model_name,
                "severity": self.severity,
                "analysis_layers": self.analysis_layers,
                "total_representations": len(self.representations)
            },
            "statistical_results": stats_results,
            "svm_probe_results": svm_results,
            "summary": self._create_summary(stats_results)
        }
        
        # Save detailed results
        with open(f"{output_dir}/analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ Report saved to {output_dir}/")

    def _collect_layer_features(self) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Collect features per layer with both binary and multiclass labels.
        Returns dict[layer] -> (X, y_binary, y_multiclass)
        y_binary: 0=baseline, 1=barrier (semantic|cultural|emotional)
        y_multiclass: 0=baseline, 1=semantic, 2=cultural, 3=emotional
        """
        bin_map = {bt: (0 if bt == "baseline" else 1) for bt in self.BARRIER_ORDER}
        multi_map = {bt: i for i, bt in enumerate(self.BARRIER_ORDER)}

        per_layer: Dict[int, List[Tuple[np.ndarray, int, int]]] = {}
        for rep in self.representations:
            # Skip if barrier type is unknown
            if rep.barrier_type not in self.BARRIER_MAPPING:
                continue
            yb = bin_map.get(rep.barrier_type)
            ym = multi_map.get(rep.barrier_type)
            x = rep.representations.numpy()
            per_layer.setdefault(rep.layer_idx, []).append((x, yb, ym))
        out: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for layer_idx, rows in per_layer.items():
            if not rows: continue
            X = np.stack([r[0] for r in rows])
            yb = np.array([r[1] for r in rows])
            ym = np.array([r[2] for r in rows])
            out[layer_idx] = (X, yb, ym)
        return out

    def _run_linear_probes(self, output_dir: str) -> Dict[str, Any]:
        """Train/evaluate SafeSwitch-style binary probers (baseline vs barrier) per layer.
        - Standardized features
        - Logistic regression (L2)
        - Stratified k-fold CV
        - Metrics: ACC, F1, ROC-AUC, PR-AUC
        - Plots: ROC, PR, PCA-2 colored by proba, summary bar chart
        """
        print("\n🧪 Running binary probers (logistic regression) on hidden states...")
        layer_data = self._collect_layer_features()
        results: Dict[str, Any] = {}
        os.makedirs(output_dir, exist_ok=True)

        # Use centralized config
        display_names = [self.BARRIER_MAPPING[bt]['name'] for bt in self.BARRIER_ORDER]
        label_to_name_multi = {i: name for i, name in enumerate(display_names)}
        name_to_color_multi = {name: self.BARRIER_MAPPING[bt]['color'] for bt, name in zip(self.BARRIER_ORDER, display_names)}
        
        # Binary labels
        label_to_name_bin = {0: "Baseline", 1: "Barrier"}
        # Using a consistent color scheme
        name_to_color_bin = {"Baseline": self.BARRIER_MAPPING["baseline"]["color"], "Barrier": "#A23B72"} # Keep barrier color distinct

        for layer_idx, (X, y_bin, y_multi) in layer_data.items():
            if len(np.unique(y_bin)) < 2:
                continue
            # Pipeline: standardize + logistic regression (binary)
            clf = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                LogisticRegression(max_iter=2000)
            )
            # 5-fold CV (or limited by class counts)
            n_splits = max(2, min(5, int(np.bincount(y_bin).min())))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            accs = []
            f1s = []
            rocs = []
            prs = []
            # For averaged curves
            mean_fpr = np.linspace(0, 1, 200)
            tprs = []
            precs_list = []
            recalls_list = []

            for tr_idx, te_idx in cv.split(X, y_bin):
                Xtr, Xte = X[tr_idx], X[te_idx]
                ytr, yte = y_bin[tr_idx], y_bin[te_idx]
                clf.fit(Xtr, ytr)
                ypred = clf.predict(Xte)
                yproba = clf.predict_proba(Xte)[:, 1]
                accs.append(accuracy_score(yte, ypred))
                f1s.append(f1_score(yte, ypred))
                rocs.append(roc_auc_score(yte, yproba))
                prs.append(average_precision_score(yte, yproba))
                fpr, tpr, _ = roc_curve(yte, yproba)
                # Interpolate TPR for mean ROC
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                prec, rec, _ = precision_recall_curve(yte, yproba)
                precs_list.append(prec)
                recalls_list.append(rec)

            # Aggregate
            acc_m, acc_s = float(np.mean(accs)), float(np.std(accs))
            f1_m, f1_s = float(np.mean(f1s)), float(np.std(f1s))
            roc_m, roc_s = float(np.mean(rocs)), float(np.std(rocs))
            pr_m, pr_s = float(np.mean(prs)), float(np.std(prs))
            results[str(layer_idx)] = {
                "accuracy_mean": acc_m,
                "accuracy_std": acc_s,
                "f1_mean": f1_m,
                "f1_std": f1_s,
                "roc_auc_mean": roc_m,
                "roc_auc_std": roc_s,
                "pr_auc_mean": pr_m,
                "pr_auc_std": pr_s,
            }

        with open(f"{output_dir}/svm_probe_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        print("✅ Linear probe results saved to svm_probe_results.json")
        # Summary bar chart across layers
 
        return results
    
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
        
        # 3. Create the combined, publication-quality plot
        self.create_combined_publication_plot(output_dir=self.output_dir)

        # 4. Generate the full report with statistics (prober metrics are computed here)
        self.generate_report(output_dir=self.output_dir)

        print("\n" + "=" * 60)
        print("Key files:")
        print(f"  📈 Combined visualizations saved in: {self.output_dir}/")
        print(f"  📋 Detailed statistics saved in: {self.output_dir}/analysis_report.json")

def main():
    """Main analysis entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze barrier effects on model representations")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name to analyze")
    parser.add_argument("--episodes", type=str, default="data/episode_all_neutralized.jsonl",
                       help="Episodes file to use")
    parser.add_argument("--severity", type=float, default=0.8,
                       help="Barrier severity level")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cuda/cpu/mps)")
    parser.add_argument("--output_dir", type=str, default="preliminary_results/barrier_analysis",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = BarrierRepresentationAnalyzer(
        model_name=args.model,
        device=args.device,
        episodes_file=args.episodes,
        severity=args.severity,
        output_dir=args.output_dir
    )
    
    # Run analysis
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()