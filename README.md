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
# - models.model_a / models.model_b: API or local HF path (used by scripts)
# - models.vllm_port, models.gpu, models.served_model_name, chat_template
# - OPENAI_API_KEY / HF_API_TOKEN / MISTRAL_API_KEY (if needed)

# 2) Start vLLM for local/HF models (reads model_b, port, gpu from config)
bash scripts/start_vllm_server.sh
```


## 🚀 Simulation (barrier modes)
```bash
bash scripts/start_vllm_server.sh           # optional: start vLLM (GPU)
bash scripts/run.sh                         # run simulation (reads configs/config.yaml)
```

## 📐 Math Analysis (GSM8K + AQuA)

The math evaluator tests whether barriers harm communication rather than raw IQ by running GSM8K (numeric) and AQuA-RAT (MCQ) with the same social instruction used for simulation. Only the JSON-output requirement is removed, and a minimal final-answer line is requested.

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
bash scripts/start_vllm_server.sh
python analysis/internal_state/barrier_representation_analysis.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --episodes data/episode_all.jsonl \
  --severity 0.8
```
