#!/usr/bin/env bash
set -euo pipefail

# Run single-agent math evaluation (GSM8K + AQuA) using vLLM-served model
# Defaults align with repository configuration and evaluator behavior.
# This script runs in profile-driven mode only.

MODEL="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="analysis/IQ_test/results"
SEVERITY="0.8"
NUM_PROFILES="0"           # 0 = use all per barrier type
PER_PROFILE_QUESTIONS="200" # per dataset per profile
CONCURRENCY="16" # Number of parallel requests
ANSWER_MODE="final_only" # "final_only" or "steps_json"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --model MODEL                   Model id (default: ${MODEL})
  --output-dir DIR                Output directory (default: ${OUTPUT_DIR})
  --severity FLOAT                Barrier severity (default: ${SEVERITY})
  --num-profiles N                Max profiles per barrier type (default: ${NUM_PROFILES}; 0 = all)
  --per-profile-questions N       Questions per profile (default: ${PER_PROFILE_QUESTIONS})
  --concurrency N                 Number of parallel requests (default: ${CONCURRENCY})
  --answer-mode MODE              Answer format (default: ${ANSWER_MODE}; options: final_only, steps_json)
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
    --num-profiles) NUM_PROFILES="$2"; shift 2;;
    --per-profile-questions) PER_PROFILE_QUESTIONS="$2"; shift 2;;
    --concurrency) CONCURRENCY="$2"; shift 2;;
    --answer-mode) ANSWER_MODE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

CMD=(
    python analysis/IQ_test/single_agent_math_eval.py 
    --model "$MODEL" 
    --output_dir "$OUTPUT_DIR" 
    --severity "$SEVERITY"
    --num_profiles "$NUM_PROFILES" 
    --per_profile_questions "$PER_PROFILE_QUESTIONS"
    --concurrency "$CONCURRENCY"
    --answer_mode "$ANSWER_MODE"
)

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"

