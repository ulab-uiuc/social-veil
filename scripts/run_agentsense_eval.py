import argparse
import json
import random
import os
import yaml
import re
from typing import List, Dict, Any, Optional, Tuple
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

def get_response_from_vllm(client: OpenAI, model_name: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 512) -> str:
    """
    Get response from the configured vLLM server.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
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

def evaluate_implicit_reasoning(vllm_client: OpenAI, model_name: str, dialogue_history: str, questions: List[Dict[str, Any]]) -> List[float]:
    """
    Evaluate the agent's ability to reason about implicit information using MCQs.
    Returns a list of scores (1.0 for correct, 0.0 for incorrect).
    """
    scores = []
    for q in questions:
        options_text = "\n".join([f"{i}. {opt}" for i, opt in enumerate(q['options'])])
        
        prompt = f"""
Based on the conversation history below, please answer the multiple-choice question.

**Conversation History:**
{dialogue_history}

**Question:**
{q['question']}

**Options:**
{options_text}

Please respond with ONLY the number of the correct option (e.g., 0, 1, 2, or 3). Do not include any other text.
"""
        # Use a lower temperature for reasoning tasks to be more deterministic
        response = get_response_from_vllm(vllm_client, model_name, [{"role": "user", "content": prompt}], temperature=0.1, max_tokens=10)
        
        # Extract the number from the response
        match = re.search(r'\d+', response)
        if match:
            predicted_idx = int(match.group())
            if predicted_idx == q['answer_label']:
                scores.append(1.0)
            else:
                scores.append(0.0)
        else:
            # Failed to follow instructions format
            scores.append(0.0)
            
    return scores

def run_simulation(scenario: Dict[str, Any], vllm_client: OpenAI, agent_model_name: str, partner_model: str = "gpt-4o-mini") -> Tuple[Dict[str, float], List[float]]:
    """
    Runs a single simulation.
    Returns:
        - A dictionary of goal completion scores.
        - A list of implicit reasoning scores.
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

    # Reconstruct dialogue text for evaluation
    dialogue_text = ""
    for i in range(1, len(messages_agent)):
        role = messages_agent[i]['role']
        content = messages_agent[i]['content']
        speaker = agent_char['name'] if role == 'assistant' else partner_char['name']
        dialogue_text += f"{speaker}: {content}\n"

    # 1. Evaluation: Goal Completion
    goal_scores = {}
    for goal_obj in agent_char['goals']:
        judge_q = next((q['question'] for q in goal_obj['eval_questions']['judge']), None)
        if judge_q:
            is_success = evaluate_goal_completion(dialogue_text, background, agent_char['name'], goal_obj['goal'], judge_q)
            goal_scores[goal_obj['goal']] = 1.0 if is_success else 0.0
    
    # 2. Evaluation: Implicit Reasoning
    # Check if the agent character has specific info reasoning questions about the partner
    reasoning_scores = []
    if "info_reason_questions" in agent_char and agent_char["info_reason_questions"]:
        reasoning_scores = evaluate_implicit_reasoning(vllm_client, agent_model_name, dialogue_text, agent_char["info_reason_questions"])
            
    return goal_scores, reasoning_scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on AgentSense benchmark.")
    parser.add_argument("--data_file", type=str, default="data/final_data.jsonl", help="Path to AgentSense data")
    parser.add_argument("--num_scenarios", type=int, default=50, help="Number of scenarios to evaluate (default: first 50)")
    parser.add_argument("--save_result", type=str, help="Path to save the evaluation results (JSON)")
    parser.add_argument("--compare_with", type=str, help="Path to a previous result file to compare against")
    
    args = parser.parse_args()

    # 1. Load Config
    config = load_config()
    vllm_port = config.get("models", {}).get("vllm_port", 8000)
    served_model_name = config.get("models", {}).get("served_model_name", "qwen2.5-7b-instruct")
    
    print(f"Connecting to vLLM server at port {vllm_port}, model: {served_model_name}")
    vllm_client = OpenAI(base_url=f"http://localhost:{vllm_port}/v1", api_key="EMPTY")

    # 2. Load Data (Prioritize scenarios with Reasoning Questions)
    print(f"Loading and filtering data from {args.data_file}...")
    scenarios_with_reasoning = []
    scenarios_without_reasoning = []
    
    with open(args.data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    # Filter for 2-character scenarios
                    if len(data['characters']) == 2:
                        # Check if the first character (the Agent) has reasoning questions
                        if "info_reason_questions" in data['characters'][0] and data['characters'][0]["info_reason_questions"]:
                            scenarios_with_reasoning.append(data)
                        else:
                            scenarios_without_reasoning.append(data)
                except json.JSONDecodeError:
                    continue
    
    # Combine lists: Prioritize those with reasoning questions
    eval_set = scenarios_with_reasoning + scenarios_without_reasoning
    # Take the top N
    eval_set = eval_set[:args.num_scenarios]
    
    print(f"Selected {len(eval_set)} scenarios for evaluation.")
    print(f"  - {len([s for s in eval_set if s in scenarios_with_reasoning])} scenarios include Implicit Reasoning tasks.")

    # 3. Evaluate Current Model
    print(f"\n--- Evaluating Model: {served_model_name} ---")
    current_goal_scores = []
    current_reasoning_scores = []
    
    for scen in tqdm(eval_set, desc="Simulating"):
        goal_s, reason_s = run_simulation(scen, vllm_client, served_model_name)
        current_goal_scores.extend(goal_s.values())
        current_reasoning_scores.extend(reason_s)

    avg_goal = sum(current_goal_scores) / len(current_goal_scores) if current_goal_scores else 0.0
    avg_reasoning = sum(current_reasoning_scores) / len(current_reasoning_scores) if current_reasoning_scores else 0.0
    
    print(f"\nResults for {served_model_name}:")
    print(f"  - Goal Completion Rate: {avg_goal:.2%}")
    if current_reasoning_scores:
        print(f"  - Reasoning Accuracy:   {avg_reasoning:.2%}")
    else:
        print(f"  - Reasoning Accuracy:   N/A (No reasoning questions in subset)")

    # 4. Save Results
    if args.save_result:
        result_data = {
            "model": served_model_name,
            "goal_completion": {"avg": avg_goal, "raw": current_goal_scores},
            "reasoning": {"avg": avg_reasoning, "raw": current_reasoning_scores}
        }
        with open(args.save_result, 'w') as f:
            json.dump(result_data, f)
        print(f"Results saved to {args.save_result}")

    # 5. Compare (if requested)
    if args.compare_with:
        try:
            with open(args.compare_with, 'r') as f:
                prev_result = json.load(f)
            
            prev_avg_goal = prev_result["goal_completion"]["avg"]
            prev_avg_reason = prev_result["reasoning"]["avg"]
            
            print("\n" + "="*50)
            print("COMPARISON RESULT")
            print("="*50)
            print(f"{'Metric':<25} | {'Baseline':<10} | {'Current (SOCIALVEIL)':<10} | {'Diff':<10}")
            print("-" * 65)
            print(f"{'Goal Completion':<25} | {prev_avg_goal:>9.2%} | {avg_goal:>20.2%} | {avg_goal - prev_avg_goal:>+9.2%}")
            
            if current_reasoning_scores and prev_result["reasoning"]["raw"]:
                print(f"{'Implicit Reasoning':<25} | {prev_avg_reason:>9.2%} | {avg_reasoning:>20.2%} | {avg_reasoning - prev_avg_reason:>+9.2%}")
            else:
                print(f"{'Implicit Reasoning':<25} | {'N/A':>9} | {'N/A':>20} | {'N/A':>9}")
            print("="*50)
            
        except Exception as e:
            print(f"Error comparing results: {e}")

if __name__ == "__main__":
    main()
