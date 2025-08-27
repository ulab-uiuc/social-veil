# Barrier Representation Analysis

This module provides tools for analyzing how communication barriers affect model internal representations. The analysis uses ML techniques to prove that barrier prompts cause measurable distribution shifts in language models.

## Overview

The analysis framework tests the hypothesis that communication barriers create distinct patterns in model representations by:

1. **Extracting internal representations** from Qwen2.5-7B-Instruct at multiple layers
2. **Comparing distributions** between baseline and barrier conditions
3. **Visualizing differences** using dimensionality reduction (PCA, t-SNE)
4. **Applying statistical tests** to prove significant differences

## Quick Start

### 1. Install Dependencies

```bash
# Install analysis requirements
pip install -r analysis/requirements.txt
```

### 2. Run Simple Test

For a quick proof-of-concept test without requiring episode files:

```bash
python analysis/run_analysis.py --mode simple
```

This creates test prompts with different barrier types and analyzes representation differences.

### 3. Run Full Analysis

For comprehensive analysis using your episode data:

```bash
python analysis/run_analysis.py --mode full --episodes data/episode_sample.jsonl --num_episodes 5
```

## Components

### Core Analysis Scripts

- **`barrier_representation_analysis.py`** - Comprehensive analysis framework
  - Loads episodes and creates barrier variants
  - Extracts representations at multiple model layers
  - Computes statistical measures and creates visualizations
  
- **`simple_barrier_test.py`** - Quick test without episode dependencies
  - Creates test prompts with different barrier types
  - Extracts and compares representations
  - Generates basic visualizations

- **`utils.py`** - Helper functions and utilities
  - Data loading and preprocessing
  - Statistical analysis functions
  - Visualization utilities
  - Model management helpers

- **`run_analysis.py`** - Main entry point script
  - Unified interface for running analyses
  - Handles configuration and dependencies

### Configuration Files

- **`requirements.txt`** - Python dependencies for analysis
- **`README.md`** - This documentation

## Analysis Methods

### 1. Representation Extraction

The analysis extracts hidden states from key model layers:
- Input layer (0)
- Early layer (25% through model)
- Middle layer (50% through model)
- Late layer (75% through model)  
- Final layer (last layer)

For each layer, we capture the last token representation as it's most relevant for generation.

### 2. Statistical Measures

The following metrics quantify differences between baseline and barrier representations:

- **Mean Squared Error (MSE)** - L2 distance between mean representations
- **Cosine Similarity** - Angular similarity between mean representations
- **Wasserstein Distance** - Optimal transport distance between distributions
- **Kolmogorov-Smirnov Test** - Distribution difference significance test
- **Mann-Whitney U Test** - Non-parametric difference test

### 3. Visualizations

- **PCA Plots** - 2D projections showing cluster separation
- **t-SNE Plots** - Non-linear dimensionality reduction per layer
- **Distance Heatmaps** - Pairwise distances between barrier types
- **Statistical Heatmaps** - Significance across layers and barriers

## Usage Examples

### Basic Usage

```python
from analysis import BarrierRepresentationAnalyzer

# Create analyzer
analyzer = BarrierRepresentationAnalyzer(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    episodes_file="data/episode_sample.jsonl",
    num_episodes=5,
    severity=0.8
)

# Run complete analysis
analyzer.run_full_analysis()
```

### Simple Test

```python
from analysis import SimpleBarrierTest

# Run quick test
tester = SimpleBarrierTest()
results = tester.run_test()
```

### Custom Analysis

```python
from analysis.utils import compute_representation_statistics, create_comparison_plot

# Load your representations
baseline_reprs = ...  # [n_samples, hidden_size]
barrier_reprs = ...   # [n_samples, hidden_size]

# Compute statistics
stats = compute_representation_statistics(baseline_reprs, barrier_reprs)

# Create visualization
representations_dict = {
    "baseline": baseline_reprs,
    "semantic_structure": barrier_reprs
}
create_comparison_plot(representations_dict, method="pca")
```

## Command Line Options

```bash
python analysis/run_analysis.py [OPTIONS]

Options:
  --mode {simple,full}           Analysis mode (default: simple)
  --model MODEL_NAME             Model to analyze (default: Qwen/Qwen2.5-7B-Instruct)
  --episodes EPISODES_FILE       Episodes file path (default: data/episode_sample.jsonl)
  --num_episodes N               Number of episodes (default: 3)
  --severity FLOAT               Barrier severity 0-1 (default: 0.8)
```

## Output Structure

Results are saved to `results/` directory:

```
results/
├── simple_barrier_test/           # Simple test outputs
│   ├── barrier_pca.png           # PCA visualization
│   └── distance_heatmap.png      # Distance heatmap
└── barrier_analysis/              # Full analysis outputs
    ├── analysis_summary.md       # Human-readable results
    ├── analysis_report.json      # Detailed statistics
    ├── tsne_layer_*.png          # t-SNE per layer
    ├── pca_all_layers.png        # PCA comparison
    └── heatmap_*.png             # Statistical heatmaps
```

## Interpreting Results

### Statistical Significance

- **p < 0.05** indicates significant distributional differences
- **Lower cosine similarity** indicates more different representations
- **Higher MSE/Wasserstein distance** indicates larger differences

### Visual Indicators

- **PCA/t-SNE plots**: Clear cluster separation indicates barrier effects
- **Distance heatmaps**: Different patterns show distinct barrier impacts
- **Layer analysis**: Shows where in the model barriers have strongest effects

### Expected Results

If barriers are effective, you should see:

- Cosine similarities < 0.95 between baseline and barrier conditions
- Statistically significant p-values (< 0.05) in KS tests
- Clear visual separation in PCA/t-SNE plots
- Distinct distance patterns in heatmaps

## Troubleshooting

### No Significant Effects

If analysis shows no significant barrier effects:

1. **Increase severity** (try 0.9 or higher)
2. **Use more episodes** (5-10 episodes)
3. **Check different model layers** (try early vs late layers)
4. **Verify barrier prompts** are being applied correctly

### Memory Issues

For large models:

1. **Reduce num_episodes** (try 2-3 episodes)
2. **Use CPU** if GPU memory is insufficient
3. **Reduce max_length** in tokenization

### Missing Dependencies

Install missing packages:

```bash
pip install transformers torch scikit-learn matplotlib seaborn scipy numpy
```

## Research Applications

This analysis framework enables research into:

- **Barrier robustness** - How well do models handle communication challenges?
- **Model interpretability** - Where do barriers have the strongest effects?
- **Training data quality** - Do barrier-augmented datasets improve robustness?
- **Architecture comparison** - Which models are most resilient to barriers?

## Contributing

To extend the analysis:

1. Add new barrier types in `barrier_representation_analysis.py`
2. Implement new statistical measures in `utils.py`
3. Create additional visualizations
4. Add support for different model architectures

## References

- Social-Decipher: [Project Repository](https://github.com/your-org/social-decipher)
- Qwen Models: [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
- Sotopia Framework: [Sotopia Paper](https://arxiv.org/abs/2310.11667)