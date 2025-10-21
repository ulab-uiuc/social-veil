# Social Decipher

Social Decipher is a research project focused on developing and evaluating socially intelligent agents. This repository contains the code for data generation, model training, and evaluation pipelines.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Generation](#data-generation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to enhance the social reasoning capabilities of large language models (LLMs) by training them to overcome various social barriers. The core idea is to simulate social interactions where one agent (the "Barrier Agent") presents challenges, and another agent (the "Partner Agent") learns to navigate these barriers effectively.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/social-decipher.git
    cd social-decipher
    ```

2.  **Set up the environment:**
    We recommend using Conda to manage dependencies.
    ```bash
    conda create -n social-decipher python=3.10
    conda activate social-decipher
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

-   `configs/`: Configuration files for models, training, and prompts.
-   `data/`: Scripts and data related to episode generation and processing.
-   `scripts/`: High-level scripts for running experiments, training, and evaluation.
-   `social_decipher/`: Core Python package containing the simulation environment, agent definitions, and training logic.
-   `results/`: Directory for storing experiment outputs.
-   `training_output/`: Directory for storing trained model checkpoints and SFT data.

## Data Generation

The data generation process involves two main stages: Behavior Cloning (BC) and Self-Reinforcement (SR).

### Behavior Cloning (BC)

BC data is collected by having an "expert" model (e.g., GPT-4o) act as the Partner Agent to generate high-quality conversation trajectories.

To collect BC data:
```bash
python -m social_decipher.training.prepare_data \
    --data_collection_mode "bc_only" \
    --output_file training_output/sft_data.json \
    --episode_limit 10 \
    --bc_concurrency 8
```

### Filtering and SFT Data Preparation

Once raw data (`bc_data.json`, `sr_data.json`) is collected, you can filter it and format it for Supervised Fine-Tuning (SFT).

```bash
python scripts/summarize_data.py \
    --input-file training_output/bc_data.json \
    --output-file training_output/sft_data_filtered.json \
    --goal-threshold 5.5 \
    --understanding-threshold 3.0 \
    --confusion-threshold 2.0
```

## Training

We use Supervised Fine-Tuning (SFT) to train the Partner Agent.

### Running SFT

The `scripts/train_sft.sh` script handles the SFT process using `accelerate` for multi-GPU training.

```bash
bash scripts/train_sft.sh \
    /path/to/base/model \
    training_output/sft_data_filtered.json \
    training_output/sft_checkpoints
```

### Merging LoRA Weights

After training, merge the LoRA adapter with the base model to create a full model.

```bash
python scripts/merge_lora_weights.py \
    --base_model_path /path/to/base/model \
    --lora_checkpoint_path training_output/sft_checkpoints/checkpoint-XXX \
    --output_path training_output/merged_model
```

## Evaluation

Evaluation is performed by running the trained model in simulated social scenarios.

1.  **Start the vLLM Server:**
    First, serve the merged model using a vLLM server. Update `configs/config.yaml` with the correct model path and start the server.
    ```bash
    bash scripts/start_vllm_server.sh
    ```

2.  **Run the Evaluation Script:**
    Execute the main run script to collect conversation data and evaluate the model's performance across different barrier types.
    ```bash
    bash scripts/run.sh
    ```

3.  **Analyze Results:**
    Use the `compare_modes.py` script to analyze the evaluation results.
    ```bash
    python results/compare_modes.py --results_dir results/ --output_file results/comparison.csv
    ```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the [MIT License](LICENSE).
