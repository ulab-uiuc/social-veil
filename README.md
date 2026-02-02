# Social Decipher

Social Decipher is a research project focused on developing and evaluating socially intelligent agents. This repository contains code for simulating and evaluating agent interactions under various social communication barriers.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to evaluate the social reasoning capabilities of large language models (LLMs) by simulating social interactions with various communication barriers. The framework tests how well agents can navigate challenges like semantic ambiguity, cultural differences, and emotional influences.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ulab-uiuc/social-decipher.git
cd social-decipher
```

### 2. Set Up Python Environment

We recommend using Conda to manage dependencies:

```bash
conda create -n social-decipher python=3.11
conda activate social-decipher
```

### 3. Install Dependencies

This project uses Poetry for dependency management:

```bash
# Install Poetry
pip install poetry

# Install all dependencies
poetry install
```

Alternatively, you can use pip with requirements.txt:

```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
pip install -r requirements.txt
```

### 4. Install Additional Tools

For running experiments, you'll need `yq` (a YAML processor):

```bash
# On macOS
brew install yq

# On Linux
pip install yq

# Or using conda
conda install -c conda-forge yq
```

## Configuration

### 1. Configure API Keys and Models

Edit `configs/config.yaml` to set up your environment:

```yaml
# Data configuration
data_dir: 'data/episode_all_neutralized.jsonl'

# Model configuration
models:
  model_a: "gpt-4o-mini"                    # Agent A model (e.g., OpenAI model)
  model_b: "Qwen/Qwen2.5-7B-Instruct"       # Agent B model (local or HuggingFace model)
  gpu: "0,1,2,3"                             # GPU devices for vLLM server
  vllm_port: 7900                            # Port for vLLM server
  chat_template: "configs/qwen2.5-7b.jinja" # Chat template for local model
  served_model_name: "qwen2.5-7b-instruct"  # Model name for vLLM API
  max_model_len: 4096                        # Maximum context length
  tensor_parallel_size: 2                    # Tensor parallelism for vLLM

# API Keys
AGENT_OPENAI_API_KEY: "your-openai-api-key-here"
EVALUATOR_OPENAI_API_KEY: "your-evaluator-api-key-here"
HF_API_TOKEN: ""  # Optional: for HuggingFace models
```

### 2. Model Configuration Options

**Agent A (Barrier Agent)**: Typically uses an OpenAI model (e.g., `gpt-4o-mini`, `gpt-4o`)

**Agent B (Partner Agent)**: Can use either:
- **OpenAI models**: `gpt-4o`, `gpt-4o-mini`, etc.
- **Local models via vLLM**: Any HuggingFace model or fine-tuned model path
  - Example: `"Qwen/Qwen2.5-7B-Instruct"`
  - Example: `"/path/to/your/fine-tuned-model"`

## Running Experiments

### Step 1: Start vLLM Server (for Local Models)

If using a local model for Agent B, start the vLLM server first:

```bash
bash scripts/start_vllm_server.sh
```

This will:
- Read configuration from `configs/config.yaml`
- Start a vLLM server on the specified port (default: 7900)
- Load the model specified in `models.model_b`

**Verify the server is running:**

```bash
curl http://localhost:7900/v1/models
```

### Step 2: Run Evaluation

Run the main evaluation script:

```bash
bash scripts/run.sh
```

This will:
- Automatically test the agent across **4 different modes**:
  - `baseline`: Standard interactions without barriers
  - `semantic`: Semantic ambiguity barriers
  - `cultural`: Cultural style barriers
  - `emotional`: Emotional influence barriers
- Save results to `results/exp_<model_name>_<data_tag>/`
- Generate conversation logs and evaluation metrics for each scenario

### Step 3: Customize Experiment Settings

You can customize the experiment using environment variables:

```bash
# Set concurrency (number of parallel scenarios)
CONCURRENCY=16 bash scripts/run.sh

# Use repair prompting for Agent B
PARTNER_REPAIR_MODE=true bash scripts/run.sh

# Use Chain-of-Thought prompting for Agent B
PARTNER_COT_MODE=true bash scripts/run.sh

# Custom results directory
RESULTS_DIR="results/my_experiment" bash scripts/run.sh

# Combine multiple settings
CONCURRENCY=32 PARTNER_REPAIR_MODE=true bash scripts/run.sh
```

### Step 4: Analyze Results

After running experiments, analyze the results:

```bash
python results/compare_modes.py \
    --base_dir results/exp_qwen2.5-7b-instruct_episode_all_neutralized \
    --out_csv results/comparison.csv
```

This generates a CSV file with:
- Performance metrics for each mode (baseline, semantic, cultural, emotional)
- Statistical significance tests
- Confidence intervals

## Project Structure

```
social-decipher/
├── configs/                  # Configuration files
│   ├── config.yaml          # Main configuration (API keys, models)
│   ├── evaluation.yaml      # Evaluation prompts
│   ├── social_task.yaml     # Agent instruction templates
│   └── *.jinja              # Chat templates for different models
├── data/                     # Episode data and processing scripts
│   ├── episode_all_neutralized.jsonl  # Main episode dataset
│   └── data_process.py      # Data processing utilities
├── scripts/                  # Executable scripts
│   ├── run.sh               # Main experiment runner
│   ├── run.py               # Python evaluation script
│   └── start_vllm_server.sh # vLLM server startup
├── social_decipher/         # Core Python package
│   ├── agent/               # Agent implementations
│   ├── environment/         # Environment and scenario management
│   ├── evaluate.py          # Evaluation logic
│   └── communication.py     # Conversation simulation
├── results/                  # Experiment outputs
│   └── compare_modes.py     # Results analysis script
└── analysis/                 # Analysis tools
    ├── compare_evaluators.py # Compare different evaluators
    └── check_correlation.py  # Correlation analysis
```

## Testing Your Setup

To verify your installation and configuration:

### 1. Check Dependencies

```bash
# Verify Python environment
python --version  # Should be 3.11

# Verify key packages
python -c "import openai; import yaml; import numpy; print('✅ Dependencies OK')"

# Verify yq is installed
yq --version
```

### 2. Test Configuration

```bash
# Verify config.yaml is valid
yq '.' configs/config.yaml
```

### 3. Quick Test Run

Run a quick test with limited episodes:

```bash
# Edit scripts/run.py and set episode limit to 5 (line 316-320)
# Then run:
CONCURRENCY=1 bash scripts/run.sh
```

This will process only the first 5 episodes to verify everything works.

## Common Issues

### vLLM Server Not Starting

If the vLLM server fails to start:
1. Check GPU availability: `nvidia-smi`
2. Verify model path in `config.yaml`
3. Check port is not in use: `lsof -i :7900`

### API Connection Errors

If you see API connection errors:
1. Verify OpenAI API keys in `config.yaml`
2. Check internet connection
3. Ensure API keys have sufficient quota

### Memory Issues

If you encounter OOM (Out of Memory) errors:
1. Reduce `max_model_len` in `config.yaml`
2. Increase `tensor_parallel_size` for larger models
3. Use a smaller model for Agent B

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{social-decipher-2026,
  title={Social Decipher: Training Socially Intelligent Agents through Communication Barriers},
  author={Your Name et al.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
