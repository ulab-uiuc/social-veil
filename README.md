# SocialVeil: Robustness of social agent to communication barriers

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3109/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![bear-ified](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)
[![Github Action](https://github.com/lwaekfjlk/python-project-template/actions/workflows/pytest.yml/badge.svg?branch=main)]()

> [!NOTE]
> This repo is continuously updating with more tools. Any contribution is welcome.


## 🧪 Social-Decipher Simulation (Quick Start)

This project includes a two-agent social simulation with barrier modes (semantic, cultural, emotional).

### 1) Prerequisites
- Python 3.10+
- Create and activate a Conda environment (recommended):
```bash
conda create -n socailveil python=3.10
conda activate socailveil
```

- Install Poetry (choose one method):

**Method 1 (Recommended for macOS/SSL issues):**
```bash
pip install poetry
```

**Method 2 (Alternative if Method 1 doesn't work):**
```bash
curl -sSL https://install.python-poetry.org | python3
```

- Dependencies installed (recommended via Poetry):
```bash
poetry install
```

- Verify installation:
```bash
python -c "import social_decipher; print('✅ Installation successful!')"
```

**Troubleshooting:**
- If you encounter SSL certificate errors during Poetry installation, use Method 1 above
- If `poetry install` fails, ensure you're in the project directory and the conda environment is activated

- API keys in `configs/config.yaml` as needed:
  - `OPENAI_API_KEY` (for OpenAI models)
  - `MISTRAL_API_KEY` (for Mistral API models)
  - `HF_API_TOKEN` (if using local/HF models via vLLM)

### 2) Configure models and data
Edit `configs/config.yaml`:
- `models.model_a`, `models.model_b` (e.g., `gpt-4o`)
- `models.served_model_name`, `models.gpu`, `models.vllm_port` (for local vLLM)
- `data_dir` (e.g., `data/episode_all.jsonl`, `data/episode_hard.jsonl`)

### 3) Run the experiment (recommended)
```bash
./scripts/run.sh
```
What happens:
- Reads config, sets `RESULTS_DIR=results/exp_<modality>_mem<strategy>_<model>_<data_tag>` (no timestamp)
- Runs `scripts/run.py --resume` so reruns skip completed scenarios
- Checks vLLM health automatically if Agent B looks like a local/HF model path

Results layout:
- `results/exp_<...>/mode_semantic|mode_cultural|mode_emotional/` per mode
  - `scenario_<n>/conversation_log.txt`
  - `scenario_<n>/eval_result.json`

Resume behavior:
- Re-run the same command; completed scenarios are skipped when both `eval_result.json` and `conversation_log.txt` exist.

### 4) Advanced: run directly
```bash
python scripts/run.py \
  --model_a gpt-4o \
  --model_b gpt-4o \
  --episodes_file data/episode_all.jsonl \
  --communication_modality text_only \

  --results_dir results/exp_text_only_gpt-4o_episode_all \
  --resume
```

### 5) Barrier episode generation (optional)
`scripts/run.py` auto-generates barrier episodes if missing. To generate manually:
```bash
python data/barrier_creation.py --mode augment \
  --input_episodes data/episode_all.jsonl \
  --seed 42 \
  --out_semantic data/episodes_semantic.json \
  --out_cultural data/episodes_cultural.json \
  --out_emotional data/episodes_emotional.json
```
Notes:
- For reproducibility, pass `--seed` and keep models/prompts constant.
- Barrier templates and cue definitions: `configs/barrier_creation.yaml`.

### 6) Compare modes (CSV/JSON)
```bash
python scripts/compare_modes.py \
  --base_dir results/exp_text_only_memoff_gpt-4o_episode_all \
  --out_json scripts/compare_summary.json \
  --out_csv scripts/compare_summary.csv
```
Outputs:
- Per-mode Sotopia dimension means for Agent 1/2
- MCQ (goal, reason) accuracy and avg confidence (knowledge MCQ omitted)
- `num_scenarios`

### 7) Troubleshooting

#### Installation Issues
- **SSL Certificate Error during Poetry installation:**
  ```
  ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
  ```
  **Solution:** Use `pip install poetry` instead of the curl installer.

- **Poetry install fails with dependency conflicts:**
  - Ensure you're in the correct conda environment: `conda activate socailveil`
  - Try clearing Poetry cache: `poetry cache clear pypi --all`
  - Update Poetry: `pip install --upgrade poetry`

#### Runtime Issues
- **vLLM server issues:** If Agent B is local/HF, start the server first (see `scripts/start_vllm_server.sh`).
- **API key errors:** Ensure your API keys are correctly set in `configs/config.yaml`.
- **Barrier creation parse errors:** The generator retries once and logs failing episode ID + scenario snippet; run again to continue.
- **Resume not working:** Ensure `RESULTS_DIR` matches the previous run and `--resume` is present; a scenario is "complete" only if both required files exist.

#### Model Issues
- **Model not found:** Check that model names in `configs/config.yaml` match available models.
- **Out of memory errors:** Reduce batch size or use smaller models.
- **Slow inference:** Consider using local models with vLLM for faster inference.
