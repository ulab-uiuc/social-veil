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
from sklearn.metrics import pairwise_distances, accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
import yaml
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# Import project modules
from social_decipher.agent.agent_profile import AgentProfile
from social_decipher.environment.episode_loader import EpisodeLoader

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
        episodes_file: str = "data/episode_all.jsonl",
        severity: float = 0.8
    ):
        self.model_name = model_name
        self.device = self._setup_device(device)
        print(f"Using device: {self.device}")
        self.episodes_file = episodes_file
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
        """Load episodes from existing barrier files"""
        from analysis.load_existing_episodes import load_all_episodes
        
        print(f"📂 Loading episodes from existing barrier files...")
        
        # Use the dedicated episode loader
        # Load all episodes (no limit)
        all_episodes = load_all_episodes(
            baseline_file=self.episodes_file,
            max_episodes=None
        )
        
        return all_episodes
    
    def create_agent_prompt(self, episode: Dict[str, Any]) -> str:
        config_path = Path(__file__).parent.parent / "configs" / "social_task.yaml"
        with open(config_path, 'r') as f:
            templates = yaml.safe_load(f)
        
        # Extract episode info
        scenario = episode.get("scenario", "")
        agent_goals = episode.get("agent_goals", ["", ""])
        agent1_goal = agent_goals[0] if len(agent_goals) > 0 else ""
        agent_reasons = episode.get("agent_reasons", ["", ""])
        agent1_reason = agent_reasons[0] if len(agent_reasons) > 0 else ""
        
        # Extract agent profiles if available
        agent_profiles = episode.get("agent_profiles", [{}, {}])
        agent1_profile = agent_profiles[0] if len(agent_profiles) > 0 else {}
        agent2_profile = agent_profiles[1] if len(agent_profiles) > 1 else {}
        
        # Prepare template data
        template_data = {
            "agent_name": agent1_profile.get("first_name", "Agent") + " " + agent1_profile.get("last_name", "A"),
            "partner_name": agent2_profile.get("first_name", "Agent") + " " + agent2_profile.get("last_name", "B"),
            "scenario": scenario,
            "agent_age": agent1_profile.get("age", "30"),
            "agent_gender": agent1_profile.get("gender", "person"),
            "agent_occupation": agent1_profile.get("occupation", "professional"),
            "agent_public_info": agent1_profile.get("public_info", "friendly person"),
            "partner_age": agent2_profile.get("age", "30"),
            "partner_gender": agent2_profile.get("gender", "person"),
            "partner_occupation": agent2_profile.get("occupation", "professional"),
            "partner_public_info": agent2_profile.get("public_info", "friendly person"),
            "agent_goal": agent1_goal,
            "agent_reason": agent1_reason,
            "agent_private_knowledge": episode.get("agent1_private_knowledge", ""),
            "turn_number": 1,
            "history": "",
            "action_list": templates.get("action_list", "")
        }
        
        barrier_type = episode.get("barrier_type", "baseline")
        
        if barrier_type == "baseline":
            prompt = templates["social_task_instructions"].format(**template_data)
            
        elif barrier_type == "semantic_structure":
            barrier_data = template_data.copy()
            barrier_data.update({
                "barrier_private_note": episode.get("barrier_cues", {}).get("profile_note_A", "Use vague, ambiguous language"),
                "barrier_prompt": episode.get("barrier_prompts", {}).get("agentA", ""),
                "barrier_dynamic_rules": self._format_barrier_dynamic_rules(episode.get("barrier_cues", {}))
            })
            prompt = templates["social_task_instructions_barrier_semantic"].format(**barrier_data)
            
        elif barrier_type == "cultural_style":
            barrier_data = template_data.copy()
            barrier_data.update({
                "barrier_private_note": episode.get("barrier_cues", {}).get("profile_note_A", "Use indirect, high-context communication"),
                "barrier_prompt": episode.get("barrier_prompts", {}).get("agentA", ""),
                "barrier_dynamic_rules": self._format_barrier_dynamic_rules(episode.get("barrier_cues", {}))
            })
            prompt = templates["social_task_instructions_barrier_cultural"].format(**barrier_data)
            
        elif barrier_type == "emotional_influence":
            barrier_data = template_data.copy()
            barrier_data.update({
                "barrier_private_note": episode.get("barrier_cues", {}).get("profile_note_A", "Maintain negative emotional tone"),
                "barrier_prompt": episode.get("barrier_prompts", {}).get("agentA", ""),
                "barrier_dynamic_rules": self._format_barrier_dynamic_rules(episode.get("barrier_cues", {}))
            })
            prompt = templates["social_task_instructions_barrier_emotional"].format(**barrier_data)
        
        return prompt
    
    def _format_barrier_dynamic_rules(self, barrier_cues: Dict[str, Any]) -> str:
        if not isinstance(barrier_cues, dict):
            return ""
        
        lines: List[str] = []
        
        def _fmt_list(key: str, label: str):
            vals = barrier_cues.get(key)
            if isinstance(vals, list) and vals:
                filtered = [str(v).strip() for v in vals if isinstance(v, str) and v.strip()]
                if filtered:
                    lines.append(f"- {label}: " + ", ".join(filtered[:8]))

        def _fmt_scalar(key: str, label: str):
            val = barrier_cues.get(key)
            if isinstance(val, (int, float, str)) and str(val).strip():
                lines.append(f"- {label}: {val}")

        # Use the exact same order and formatting as social_agent.py
        _fmt_list("lexical_prefer", "Use phrases like")
        _fmt_list("lexical_avoid", "Avoid phrases")
        _fmt_scalar("sentence_length_bias", "Sentence length bias")
        _fmt_list("ambiguity_devices", "Use ambiguity devices")
        _fmt_scalar("question_rate_hint", "Target question rate")
        _fmt_scalar("imperative_rate_hint", "Target imperative rate")
        _fmt_list("affect_lexicon", "Affect lexicon")
        _fmt_scalar("exclamation_bias", "Exclamation bias")
        _fmt_scalar("turn_length_max", "Max sentences per turn")

        if lines:
            return "\n".join(lines)
        else:
            return ""
    
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
    
    def _create_preliminary_visualization(self, output_dir: str, barrier_types: List[str], barrier_colors: Dict[str, str]) -> None:
        """
        Create preliminary visualization inspired by SafeSwitch paper.
        Visualizes model internal states for different barrier prompts using 2D PCA.
        Similar to Figures 15-17 in SafeSwitch paper showing clustering of different query types.
        """
        print("  📊 Creating preliminary visualization (SafeSwitch-style)...")
        
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
    
    def create_visualizations(self, output_dir: str = "preliminary_results/barrier_analysis") -> None:
        """Generate only t-SNE visualizations per layer with improved styling."""
        print(f"\n🎨 Creating visualizations in {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        barrier_types = ["baseline", "semantic_structure", "cultural_style", "emotional_influence"]
        barrier_colors = {
            "baseline": "#2E86AB",
            "semantic_structure": "#A23B72", 
            "cultural_style": "#F18F01",
            "emotional_influence": "#C73E1D"
        }
        
        for layer_idx in self.analysis_layers:
            try:
                sns.set_theme(style="whitegrid")
            except Exception:
                pass
            fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=150)
            
            layer_data = []
            layer_labels = []
            layer_episodes = []
            
            for rep in self.representations:
                if rep.layer_idx == layer_idx:
                    layer_data.append(rep.representations.numpy())
                    layer_labels.append(rep.barrier_type)
                    layer_episodes.append(rep.episode_id)
            
            if len(layer_data) < 4:
                plt.close()
                continue
            
            layer_data = np.stack(layer_data)
            # Use a slightly higher perplexity for nicer separation when possible
            perplexity = max(5, min(30, len(layer_data)//3))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, learning_rate='auto', init='pca')
            tsne_results = tsne.fit_transform(layer_data)
            
            for barrier_type in barrier_types:
                mask = np.array(layer_labels) == barrier_type
                if mask.any():
                    ax.scatter(
                        tsne_results[mask, 0],
                        tsne_results[mask, 1],
                        c=barrier_colors[barrier_type],
                        label=barrier_type.replace("_", " ").title(),
                        s=60,
                        alpha=0.85,
                        edgecolors='white',
                        linewidths=0.6
                    )
            
            ax.set_title(f"t-SNE Visualization - Layer {layer_idx}", fontsize=16, fontweight='bold')
            ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
            ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
            legend = ax.legend(frameon=True, title="Barrier Type")
            legend.get_title().set_fontweight('bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/tsne_layer_{layer_idx}.png", bbox_inches='tight')
            plt.close()
    
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
        
        # Create markdown summary
        self._create_markdown_report(report, f"{output_dir}/analysis_summary.md")
        
        print(f"✅ Report saved to {output_dir}/")

    def _collect_layer_features(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Collect features (X) and labels (y) per layer from representations.
        Labels: 0=baseline, 1=semantic, 2=cultural, 3=emotional
        """
        label_map = {
            "baseline": 0,
            "semantic_structure": 1,
            "cultural_style": 2,
            "emotional_influence": 3,
        }
        per_layer: Dict[int, List[Tuple[np.ndarray, int]]] = {}
        for rep in self.representations:
            y = label_map.get(rep.barrier_type, 0)
            x = rep.representations.numpy()
            per_layer.setdefault(rep.layer_idx, []).append((x, y))
        out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for layer_idx, pairs in per_layer.items():
            X = np.stack([p[0] for p in pairs])
            y = np.array([p[1] for p in pairs])
            out[layer_idx] = (X, y)
        return out

    def _run_linear_probes(self, output_dir: str) -> Dict[str, Any]:
        """Train/evaluate linear probes (logistic regression) per layer. Save results as JSON.
        Returns a dict of metrics per layer.
        """
        print("\n🧪 Running linear probes (logistic regression) on hidden states...")
        layer_data = self._collect_layer_features()
        results: Dict[str, Any] = {}
        os.makedirs(output_dir, exist_ok=True)

        # Consistent color/label mapping
        label_to_name = {0: "Baseline", 1: "Semantic", 2: "Cultural", 3: "Emotional"}
        name_to_color = {"Baseline": "#2E86AB", "Semantic": "#A23B72", "Cultural": "#F18F01", "Emotional": "#C73E1D"}

        # Accumulators for summary plot
        summary_layers: List[int] = []
        summary_acc: List[float] = []
        summary_f1m: List[float] = []

        for layer_idx, (X, y) in layer_data.items():
            if len(np.unique(y)) < 2:
                continue
            # Pipeline: standardize + multinomial logistic regression
            clf = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                LogisticRegression(max_iter=2000, multi_class='multinomial')
            )
            cv = StratifiedKFold(n_splits=min(5, np.bincount(y).min()), shuffle=True, random_state=42)
            # Cross-validated accuracy and macro-F1
            acc = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
            f1m = cross_val_score(clf, X, y, cv=cv, scoring='f1_macro')
            # Confusion matrix via cross-val predictions
            y_pred = cross_val_predict(clf, X, y, cv=cv)
            cm = confusion_matrix(y, y_pred).tolist()
            report = classification_report(y, y_pred, output_dict=True)
            results[str(layer_idx)] = {
                "accuracy_mean": float(np.mean(acc)),
                "accuracy_std": float(np.std(acc)),
                "f1_macro_mean": float(np.mean(f1m)),
                "f1_macro_std": float(np.std(f1m)),
                "confusion_matrix": cm,
                "classification_report": report,
            }

            # Save confusion matrix heatmap
            try:
                fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=150)
                cm_arr = np.array(cm)
                sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                            xticklabels=[label_to_name[i] for i in range(cm_arr.shape[1])],
                            yticklabels=[label_to_name[i] for i in range(cm_arr.shape[0])])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title(f'Linear Probe Confusion Matrix (Layer {layer_idx})')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/svm_cm_layer_{layer_idx}.png", bbox_inches='tight')
                plt.close()
            except Exception:
                pass

            # 2D PCA visualization with decision regions (for illustration only)
            try:
                scaler = StandardScaler(with_mean=True, with_std=True)
                Xs = scaler.fit_transform(X)
                pca2 = PCA(n_components=2, random_state=42)
                X2 = pca2.fit_transform(Xs)
                vis_clf = LogisticRegression(max_iter=2000, multi_class='multinomial')
                vis_clf.fit(X2, y)

                # Meshgrid for decision regions
                x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
                y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
                Z = vis_clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

                fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=150)
                # Background decision regions
                ax.contourf(xx, yy, Z, alpha=0.15, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], colors=[name_to_color[n] for n in ["Baseline","Semantic","Cultural","Emotional"]])
                # Scatter points
                for lbl, name in label_to_name.items():
                    mask = (y == lbl)
                    ax.scatter(
                        X2[mask, 0], X2[mask, 1],
                        c=name_to_color[name], label=name,
                        s=35, alpha=0.9, edgecolors='white', linewidths=0.5
                    )
                ax.set_title(f'Linear Probe (PCA-2D) - Layer {layer_idx}')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.legend(frameon=True)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/svm_pca2_layer_{layer_idx}.png", bbox_inches='tight')
                plt.close()
            except Exception:
                pass

            summary_layers.append(layer_idx)
            summary_acc.append(float(np.mean(acc)))
            summary_f1m.append(float(np.mean(f1m)))

        with open(f"{output_dir}/svm_probe_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        print("✅ Linear probe results saved to svm_probe_results.json")
        # Summary bar chart across layers
        try:
            if summary_layers:
                x = np.arange(len(summary_layers))
                width = 0.35
                fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
                ax.bar(x - width/2, summary_acc, width, label='Accuracy')
                ax.bar(x + width/2, summary_f1m, width, label='F1-macro')
                ax.set_xticks(x)
                ax.set_xticklabels([str(l) for l in summary_layers])
                ax.set_ylim(0, 1.0)
                ax.set_xlabel('Layer index')
                ax.set_ylabel('Score')
                ax.set_title('Linear Probe Performance by Layer')
                ax.legend()
                plt.tight_layout()
                plt.savefig(f"{output_dir}/svm_metrics_summary.png", bbox_inches='tight')
                plt.close()
        except Exception:
            pass

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
    
    def _create_markdown_report(self, report: Dict[str, Any], output_path: str) -> None:
        """Create markdown summary report"""
        
        metadata = report["analysis_metadata"]
        summary = report["summary"]
        
        content = f"""# Barrier Representation Analysis Report

## Analysis Configuration
- **Model**: {metadata["model"]}
# (All episodes in the provided files are analyzed)
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
    parser.add_argument("--episodes", type=str, default="data/episode_original.jsonl",
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
        severity=args.severity
    )
    
    # Run analysis
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()