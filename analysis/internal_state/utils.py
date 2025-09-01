"""
Utility functions for barrier representation analysis.

This module provides helper functions for:
- Data loading and preprocessing
- Statistical analysis
- Visualization utilities
- Model management
"""

import json
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_episodes_from_file(file_path: str, max_episodes: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load episodes from JSONL or JSON file.
    
    Args:
        file_path: Path to episodes file
        max_episodes: Maximum number of episodes to load
        
    Returns:
        List of episode dictionaries
    """
    episodes = []
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ Episodes file not found: {file_path}")
        return episodes
    
    try:
        if file_path.suffix == '.json':
            # JSON format
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    episodes = data
                else:
                    episodes = [data]
        else:
            # JSONL format
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if max_episodes and i >= max_episodes:
                        break
                    line = line.strip()
                    if line:
                        episodes.append(json.loads(line))
        
        if max_episodes:
            episodes = episodes[:max_episodes]
            
        print(f"✅ Loaded {len(episodes)} episodes from {file_path}")
        
    except Exception as e:
        print(f"❌ Error loading episodes: {e}")
        episodes = []
    
    return episodes

def compute_representation_statistics(
    baseline_reprs: np.ndarray, 
    barrier_reprs: np.ndarray
) -> Dict[str, float]:
    """
    Compute statistical measures between baseline and barrier representations.
    
    Args:
        baseline_reprs: Baseline representations [n_samples, hidden_size]
        barrier_reprs: Barrier representations [n_samples, hidden_size]
        
    Returns:
        Dictionary of statistical measures
    """
    stats_dict = {}
    
    # 1. Mean squared error between means
    mean_baseline = baseline_reprs.mean(axis=0)
    mean_barrier = barrier_reprs.mean(axis=0)
    mse = np.mean((mean_baseline - mean_barrier) ** 2)
    stats_dict["mse"] = float(mse)
    
    # 2. Cosine similarity between means
    cosine_sim = np.dot(mean_baseline, mean_barrier) / (
        np.linalg.norm(mean_baseline) * np.linalg.norm(mean_barrier)
    )
    stats_dict["cosine_similarity"] = float(cosine_sim)
    
    # 3. Euclidean distance between means
    euclidean_dist = np.linalg.norm(mean_baseline - mean_barrier)
    stats_dict["euclidean_distance"] = float(euclidean_dist)
    
    # 4. Wasserstein distance (using subset for efficiency)
    if baseline_reprs.shape[1] > 100:
        # Use first 100 dimensions for efficiency
        baseline_subset = baseline_reprs[:, :100]
        barrier_subset = barrier_reprs[:, :100]
    else:
        baseline_subset = baseline_reprs
        barrier_subset = barrier_reprs
    
    # Flatten for 1D Wasserstein
    baseline_flat = baseline_subset.flatten()
    barrier_flat = barrier_subset.flatten()
    try:
        wasserstein_dist = stats.wasserstein_distance(baseline_flat, barrier_flat)
        stats_dict["wasserstein_distance"] = float(wasserstein_dist)
    except Exception:
        stats_dict["wasserstein_distance"] = float('nan')
    
    # 5. Kolmogorov-Smirnov test on first principal component
    try:
        from sklearn.decomposition import PCA
        
        # Combine data for PCA fitting
        combined_data = np.vstack([baseline_reprs, barrier_reprs])
        pca = PCA(n_components=1)
        pca.fit(combined_data)
        
        # Transform each set
        baseline_pc = pca.transform(baseline_reprs).flatten()
        barrier_pc = pca.transform(barrier_reprs).flatten()
        
        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(baseline_pc, barrier_pc)
        stats_dict["ks_statistic"] = float(ks_stat)
        stats_dict["ks_pvalue"] = float(ks_pvalue)
        
    except Exception:
        stats_dict["ks_statistic"] = float('nan')
        stats_dict["ks_pvalue"] = float('nan')
    
    # 6. Mann-Whitney U test (non-parametric)
    try:
        # Use mean of each representation as the test statistic
        baseline_means = baseline_reprs.mean(axis=1)
        barrier_means = barrier_reprs.mean(axis=1)
        
        u_stat, u_pvalue = stats.mannwhitneyu(baseline_means, barrier_means, alternative='two-sided')
        stats_dict["mannwhitney_u"] = float(u_stat)
        stats_dict["mannwhitney_p"] = float(u_pvalue)
        
    except Exception:
        stats_dict["mannwhitney_u"] = float('nan')
        stats_dict["mannwhitney_p"] = float('nan')
    
    return stats_dict

def create_comparison_plot(
    representations_dict: Dict[str, np.ndarray],
    method: str = "pca",
    figsize: Tuple[int, int] = (10, 8),
    output_path: Optional[str] = None
) -> None:
    """
    Create comparison plot of representations using dimensionality reduction.
    
    Args:
        representations_dict: Dict mapping barrier types to representations
        method: Dimensionality reduction method ("pca" or "tsne")
        figsize: Figure size
        output_path: Path to save plot (optional)
    """
    
    # Prepare data
    all_data = []
    all_labels = []
    
    barrier_types = ["baseline", "semantic_structure", "cultural_style", "emotional_influence"]
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
    color_map = dict(zip(barrier_types, colors))
    
    for barrier_type, reprs in representations_dict.items():
        if len(reprs.shape) == 1:
            reprs = reprs.reshape(1, -1)
        
        all_data.append(reprs)
        all_labels.extend([barrier_type] * reprs.shape[0])
    
    all_data = np.vstack(all_data)
    
    # Apply dimensionality reduction
    if method.lower() == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        reduced_data = reducer.fit_transform(all_data)
        explained_var = reducer.explained_variance_ratio_
        xlabel = f"PC1 ({explained_var[0]:.1%} variance)"
        ylabel = f"PC2 ({explained_var[1]:.1%} variance)"
        title = "PCA Visualization of Barrier Effects"
        
    elif method.lower() == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, len(all_data) // 4)
        reducer = TSNE(n_components=2, random_state=42, perplexity=max(5, perplexity))
        reduced_data = reducer.fit_transform(all_data)
        xlabel = "t-SNE Dimension 1"
        ylabel = "t-SNE Dimension 2" 
        title = "t-SNE Visualization of Barrier Effects"
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'")
    
    # Create plot
    plt.figure(figsize=figsize)
    
    for barrier_type in barrier_types:
        if barrier_type in representations_dict:
            mask = np.array(all_labels) == barrier_type
            if mask.any():
                plt.scatter(
                    reduced_data[mask, 0],
                    reduced_data[mask, 1],
                    c=color_map[barrier_type],
                    label=barrier_type.replace("_", " ").title(),
                    s=100,
                    alpha=0.8
                )
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📈 Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def create_distance_heatmap(
    representations_dict: Dict[str, np.ndarray],
    metric: str = "euclidean",
    figsize: Tuple[int, int] = (8, 6),
    output_path: Optional[str] = None
) -> None:
    """
    Create distance heatmap between different barrier types.
    
    Args:
        representations_dict: Dict mapping barrier types to representations
        metric: Distance metric to use
        figsize: Figure size
        output_path: Path to save plot (optional)
    """
    
    # Prepare data - use mean representation for each barrier type
    barrier_types = []
    mean_reprs = []
    
    for barrier_type, reprs in representations_dict.items():
        if len(reprs.shape) == 1:
            mean_repr = reprs
        else:
            mean_repr = reprs.mean(axis=0)
        
        barrier_types.append(barrier_type.replace("_", " ").title())
        mean_reprs.append(mean_repr)
    
    mean_reprs = np.stack(mean_reprs)
    
    # Compute distance matrix
    distance_matrix = pairwise_distances(mean_reprs, metric=metric)
    
    # Create heatmap
    plt.figure(figsize=figsize)
    
    # Use seaborn for better heatmap
    sns.heatmap(
        distance_matrix,
        xticklabels=barrier_types,
        yticklabels=barrier_types,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        square=True
    )
    
    plt.title(f'Pairwise {metric.capitalize()} Distances', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📈 Heatmap saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def setup_model_device(device: str = "auto") -> str:
    """
    Setup the appropriate device for model inference.
    
    Args:
        device: Device specification ("auto", "cuda", "cpu", "mps")
        
    Returns:
        Device string to use
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"🔧 Using CUDA GPU: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("🔧 Using Apple MPS")
        else:
            device = "cpu"
            print("🔧 Using CPU")
    else:
        print(f"🔧 Using specified device: {device}")
    
    return device

def safe_model_loading(model_name: str, device: str) -> Tuple[Any, Any]:
    """
    Safely load model and tokenizer with error handling.
    
    Args:
        model_name: Name/path of the model
        device: Device to load model on
        
    Returns:
        Tuple of (tokenizer, model) or (None, None) if failed
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"🔧 Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"🔧 Loading model {model_name} on {device}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device != "auto" else None,
            trust_remote_code=True,
            output_hidden_states=True,
            low_cpu_mem_usage=True
        )
        model.eval()
        
        print("✅ Model loaded successfully")
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def check_dependencies() -> List[str]:
    """
    Check if required dependencies are installed.
    
    Returns:
        List of missing packages
    """
    required_packages = [
        "transformers",
        "torch", 
        "sklearn",
        "matplotlib",
        "seaborn",
        "scipy",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
    else:
        print("✅ All required packages are installed")
    
    return missing_packages

def save_analysis_results(
    results: Dict[str, Any], 
    output_path: str
) -> None:
    """
    Save analysis results to JSON file.
    
    Args:
        results: Analysis results dictionary
        output_path: Path to save results
    """
    try:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(results)
        
        os.makedirs(Path(output_path).parent, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"💾 Results saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def print_analysis_summary(stats_results: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of analysis results.
    
    Args:
        stats_results: Statistical analysis results
    """
    print("\n" + "="*60)
    print("📊 BARRIER ANALYSIS SUMMARY")
    print("="*60)
    
    significant_effects = []
    
    for layer_key, layer_data in stats_results.items():
        print(f"\n🔍 {layer_key.replace('_', ' ').title()}:")
        
        for barrier_type, stats in layer_data.items():
            barrier_clean = barrier_type.replace("_", " ").title()
            
            cosine_sim = stats.get("cosine_similarity", 1.0)
            ks_pvalue = stats.get("ks_pvalue", 1.0)
            mse = stats.get("mse", 0.0)
            
            print(f"  {barrier_clean}:")
            print(f"    Cosine similarity: {cosine_sim:.4f}")
            print(f"    MSE: {mse:.6f}")
            
            if ks_pvalue < 0.05:
                print(f"    ⭐ Significant difference (p={ks_pvalue:.4f})")
                significant_effects.append(f"{barrier_clean} at {layer_key}")
            else:
                print(f"    KS test p-value: {ks_pvalue:.4f}")
    
    print(f"\n🎯 SIGNIFICANT EFFECTS FOUND: {len(significant_effects)}")
    for effect in significant_effects:
        print(f"  ✅ {effect}")
    
    if not significant_effects:
        print("  ❌ No statistically significant barrier effects detected")
        print("  💡 Try: increasing severity, using more episodes, or different layers")
    
    print("="*60)