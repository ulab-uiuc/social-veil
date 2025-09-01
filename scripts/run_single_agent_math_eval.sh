#!/usr/bin/env bash
set -euo pipefail

# Run single-agent math evaluation (GSM8K + AQuA) using vLLM-served model
# Defaults align with repository configuration and evaluator behavior.

# bash scripts/run_single_agent_math_eval.sh --by-profiles --num-profiles 50 --per-profile-questions 200

MODEL="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="analysis/IQ_test/results"
SEVERITY="0.8"
BY_PROFILES="false"
NUM_PROFILES="0"           # 0 = use all per barrier type
PER_PROFILE_QUESTIONS="200" # per dataset per profile when --by_profiles
PROBLEMS="10"               # per dataset in non-profile mode; 0 = full

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --model MODEL                   Model id (default: ${MODEL})
  --output-dir DIR                Output directory (default: ${OUTPUT_DIR})
  --severity FLOAT                Barrier severity (default: ${SEVERITY})
  --by-profiles                   Enable profile-driven evaluation
  --num-profiles N                Max profiles per barrier type (default: ${NUM_PROFILES}; 0 = all)
  --per-profile-questions N       Questions per dataset per profile (default: ${PER_PROFILE_QUESTIONS})
  --problems N                    Problems per dataset in non-profile mode (default: ${PROBLEMS}; 0 = full)
  -h, --help                      Show this help and exit

Notes:
  - vLLM server should be running for GPU inference (start with scripts/start_vllm_server.sh).
  - VLLM_PORT is read from configs/config.yaml automatically by the evaluator; you can override with env VLLM_PORT.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --severity) SEVERITY="$2"; shift 2;;
    --by-profiles) BY_PROFILES="true"; shift 1;;
    --num-profiles) NUM_PROFILES="$2"; shift 2;;
    --per-profile-questions) PER_PROFILE_QUESTIONS="$2"; shift 2;;
    --problems) PROBLEMS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

CMD=(python analysis/IQ_test/single_agent_math_eval.py --model "$MODEL" --output_dir "$OUTPUT_DIR" --severity "$SEVERITY")

if [[ "$BY_PROFILES" == "true" ]]; then
  CMD+=(--by_profiles --num_profiles "$NUM_PROFILES" --per_profile_questions "$PER_PROFILE_QUESTIONS")
else
  CMD+=(--problems "$PROBLEMS")
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"

