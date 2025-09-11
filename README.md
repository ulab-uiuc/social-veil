# SocialVeil: Robustness of social agent to communication barriers

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3109/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![bear-ified](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)
[![Github Action](https://github.com/lwaekfjlk/python-project-template/actions/workflows/pytest.yml/badge.svg?branch=main)]()

> [!NOTE]
> This repo is continuously updating with more tools. Any contribution is welcome.

## 🧰 Environment Setup
```bash
conda create -n socialveil python=3.10
conda activate socialveil
pip install poetry
poetry install
# optional (math IQ tests)
pip install -r analysis/IQ_test/requirements.txt
```

### 🔑 Variables & Model Setup
```bash
# 1) Edit configs/config.yaml
#    Comment/uncomment ONE model block below (OpenAI / Anthropic / Local vLLM)
#    and fill in the credentials/paths for that provider.
#    Only one `models:` block should be active at a time.
# - models.model_a / models.model_b: API or local HF path (used by scripts)
# - models.vllm_port, models.gpu, models.served_model_name, chat_template
# - OPENAI_API_KEY / ANTHROPIC_API_KEY / HF_API_TOKEN / MISTRAL_API_KEY (if needed)

# 2) Export keys (if not stored in config.yaml)
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export HF_API_TOKEN=...
export MISTRAL_API_KEY=...

# 3) Start vLLM (reads all models.* from configs/config.yaml; no args)
bash scripts/start_vllm_server.sh
```

### 🧩 Config Examples
> Enable exactly one of the following by uncommenting it in `configs/config.yaml`.
```yaml
# 1) OpenAI (GPT)
models:
  model_a: "gpt-4o-mini"
  model_b: "gpt-4o-mini"  
  gpu: "4,5"
  vllm_port: 6600
  chat_template: "configs/qwen2.5-7b.jinja"
  served_model_name: "qwen2.5-7b-instruct"
  max_model_len: 4096
  tensor_parallel_size: 0
  enable_repair_and_state: false

# 2) Local model (Mistral via vLLM)
models:
  model_a: "gpt-4o-mini"                     
  model_b: "./models/Ministral-8B-Instruct-2410"
  gpu: "4,5"                            
  vllm_port: 6600
  chat_template: "configs/mistral-8b.jinja"
  served_model_name: "ministral-8b-instruct"
  max_model_len: 4096                    
  tensor_parallel_size: 0
  enable_repair_and_state: false

# 3) Local model (Qwen2.5 via vLLM)
models:
  model_a: "gpt-4o-mini"
  model_b: "models/Qwen2.5-7B-Instruct"  
  gpu: "4,5"
  vllm_port: 6900
  chat_template: "configs/qwen2.5-7b.jinja"
  served_model_name: "qwen2.5-7b-instruct"
  max_model_len: 4096
  tensor_parallel_size: 0
  enable_repair_and_state: false
```


## 🚀 Simulation (barrier modes)
```bash
# If using a local/HF model_b, ensure vLLM is running in another terminal
bash scripts/start_vllm_server.sh   # reads configs/config.yaml

# Run simulations (reads configs/config.yaml and episodes)
bash scripts/run.sh --episodes_file data/episode_all.jsonl
```

## 📐 Math Analysis (GSM8K + AQuA)

The math evaluator tests whether barriers harm communication rather than raw IQ by running GSM8K (numeric) and AQuA-RAT (MCQ) with the same social instruction used for simulation. 
```bash
bash scripts/start_vllm_server.sh
bash scripts/run_single_agent_math_eval.sh --by-profiles --num-profiles 50 --per-profile-questions 200
# or
bash scripts/run_single_agent_math_eval.sh --problems 50
```

Outputs:
- Incremental: `analysis/IQ_test/results/incremental_results.jsonl`
- Final: `analysis/IQ_test/results/detailed_results.json`, `results_by_problem.csv`, `evaluation_by_source.json`
- Profile mode: `profile_averages.json`, `barrier_type_averages.json`, source-split JSONs, and `profile_scores.csv`

Scoring:
- GSM8K: strict numeric match from an explicit final line `Answer: <NUMBER>`
- AQuA: strict letter match from `Answer: <LETTER>`; partial/early numbers are ignored

## 🧠 Internal State Verification (Representation Analysis)
```bash
bash scripts/internal_state_analysis.sh
```
