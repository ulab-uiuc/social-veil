import argparse
import json
import random
import os
import yaml
from typing import List, Dict, Any, Optional
from openai import OpenAI
from tqdm import tqdm
import time

# Set up OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load project configuration."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def get_response_from_vllm(client: OpenAI, model_name: str, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    """
    Get response from the configured vLLM server.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying vLLM: {e}")
        return ""

def get_response_from_openai(model: str, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    """
    Get response from OpenAI API (for Partner/Judge).
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        time.sleep(2)
        return ""

def evaluate_goal_completion(dialogue_history: str, background: str, agent_name: str, goal: str, judge_question: str) -> bool:
    """
    Use GPT-4o as a judge to evaluate goal completion.
    """
    prompt = f"""
You are an impartial judge evaluating a social interaction.

**Background:**
{background}

**Interaction:**
{dialogue_history}

**Evaluation Task:**
The character '{agent_name}' had the following goal: "{goal}"

Please answer the following question based on the interaction above:
{judge_question}

Answer with only "Yes" or "No".
"""
    response = get_response_from_openai("gpt-4o", [{"role": "user", "content": prompt}], temperature=0.0)
    return "yes" in response.strip().lower()

def run_simulation(scenario: Dict[str, Any], vllm_client: OpenAI, agent_model_name: str, partner_model: str = "gpt-4o-mini") -> Dict[str, float]:
    """
    Runs a single simulation.
    """
    agent_char = scenario['characters'][0]
    partner_char = scenario['characters'][1]
    background = scenario['background']
    description = scenario['description']
    
    agent_system_prompt = f"""
You are roleplaying as {agent_char['name']}.
Profile: {agent_char['profile']}
Background: {background}
Context: {description}

Your Goals:
{chr(10).join(['- ' + g['goal'] for g in agent_char['goals']])}

Interact with {partner_char['name']} to achieve your goals. Be natural and stay in character.
"""

    partner_system_prompt = f"""
You are roleplaying as {partner_char['name']}.
Profile: {partner_char['profile']}
Background: {background}
Context: {description}

Your Goals:
{chr(10).join(['- ' + g['goal'] for g in partner_char['goals']])}

Interact with {agent_char['name']}. Be natural and stay in character.
"""

    messages_agent = [{"role": "system", "content": agent_system_prompt}]
    messages_partner = [{"role": "system", "content": partner_system_prompt}]
    
    # 6 turns (3 each)
    for _ in range(3):
        # Agent turn
        agent_response = get_response_from_vllm(vllm_client, agent_model_name, messages_agent)
        if not agent_response: agent_response = "..."
        
        messages_agent.append({"role": "assistant", "content": agent_response})
        messages_partner.append({"role": "user", "content": agent_response})
        
        # Partner turn
        partner_response = get_response_from_openai(partner_model, messages_partner)
        if not partner_response: partner_response = "..."
        
        messages_partner.append({"role": "assistant", "content": partner_response})
        messages_agent.append({"role": "user", "content": partner_response})

    # Evaluation
    scores = {}
    for goal_obj in agent_char['goals']:
        judge_q = next((q['question'] for q in goal_obj['eval_questions']['judge']), None)
        if judge_q:
            # Reconstruct dialogue text only when needed
            dialogue_text = ""
            for i in range(1, len(messages_agent)):
                role = messages_agent[i]['role']
                content = messages_agent[i]['content']
                speaker = agent_char['name'] if role == 'assistant' else partner_char['name']
                dialogue_text += f"{speaker}: {content}\n"

            is_success = evaluate_goal_completion(dialogue_text, background, agent_char['name'], goal_obj['goal'], judge_q)
            scores[goal_obj['goal']] = 1.0 if is_success else 0.0
            
    return scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on AgentSense benchmark.")
    parser.add_argument("--data_file", type=str, default="data/final_data.jsonl", help="Path to AgentSense data")
    parser.add_argument("--num_scenarios", type=int, default=30, help="Number of scenarios to evaluate")
    
    # Configuration overrides
    parser.add_argument("--vllm_port", type=int, help="Port of the running vLLM server (overrides config.yaml)")
    parser.add_argument("--model_name", type=str, help="Name of the model served by vLLM (overrides config.yaml)")
    
    parser.add_argument("--save_result", type=str, help="Path to save the evaluation results (JSON)")
    parser.add_argument("--compare_with", type=str, help="Path to a previous result file to compare against")
    
    args = parser.parse_args()

    # 1. Load Config & Apply Overrides
    config = load_config()
    vllm_port = args.vllm_port or config.get("models", {}).get("vllm_port", 8000)
    served_model_name = args.model_name or config.get("models", {}).get("served_model_name", "qwen2.5-7b-instruct")
    
    print(f"Connecting to vLLM server at port {vllm_port}, model: {served_model_name}")
    vllm_client = OpenAI(base_url=f"http://localhost:{vllm_port}/v1", api_key="EMPTY")

    # 2. Load Data
    print(f"Loading data from {args.data_file}...")
    scenarios = []
    with open(args.data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if len(data['characters']) == 2:
                    scenarios.append(data)
    
    random.seed(42)
    eval_set = random.sample(scenarios, min(args.num_scenarios, len(scenarios)))
    print(f"Selected {len(eval_set)} scenarios for evaluation.")

    # 3. Evaluate Current Model
    print(f"\n--- Evaluating Model: {served_model_name} ---")
    current_scores_list = []
    
    for scen in tqdm(eval_set, desc="Simulating"):
        scores = run_simulation(scen, vllm_client, served_model_name)
        current_scores_list.extend(scores.values())

    current_avg = sum(current_scores_list) / len(current_scores_list) if current_scores_list else 0.0
    print(f"\nModel Goal Completion Rate: {current_avg:.2%}")

    # 4. Save Results
    if args.save_result:
        with open(args.save_result, 'w') as f:
            json.dump({"model": served_model_name, "avg_score": current_avg, "raw_scores": current_scores_list}, f)
        print(f"Results saved to {args.save_result}")

    # 5. Compare (if requested)
    if args.compare_with:
        try:
            with open(args.compare_with, 'r') as f:
                prev_result = json.load(f)
            
            prev_avg = prev_result["avg_score"]
            print("\n" + "="*40)
            print("COMPARISON RESULT")
            print("="*40)
            print(f"Baseline Model ({prev_result['model']}): {prev_avg:.2%}")
            print(f"Current Model  ({served_model_name}): {current_avg:.2%}")
            print(f"Improvement:      {current_avg - prev_avg:+.2%}")
            print("="*40)
        except Exception as e:
            print(f"Error comparing results: {e}")

if __name__ == "__main__":
    main()
