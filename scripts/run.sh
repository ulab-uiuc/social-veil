GLOBAL_MODEL_A="gpt-4o-mini"
GLOBAL_MODEL_B="Qwen/Qwen2.5-7B-Instruct"

echo "===================================="
echo "🤖 Global Models: Agent A = $GLOBAL_MODEL_A, Agent B = $GLOBAL_MODEL_B"
echo ""

python run.py --model_a $GLOBAL_MODEL_A --model_b $GLOBAL_MODEL_B --episode_limit 3 --start_vllm