import json
import os
import random
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from rich import print

from social_decipher.agent.social_agent import SocialAgent
 
from social_decipher.environment.env_profile import EnvironmentProfile
from social_decipher.evaluate import ConversationEvaluator
from social_decipher.utils.repair import judge_repair_with_llm
from social_decipher.utils.state import init_barrier_state, update_barrier_state
from social_decipher.utils.config_reader import load_config


def _format_action_output(message: Union[str, Dict[str, Any]]) -> str:
    
    if isinstance(message, str):
        return message

    # Dict-based formats
    if isinstance(message, dict):
        # Newer strict action schema
        if "action_type" in message:
            action_type = str(message.get("action_type", "")).strip().lower()
            argument = str(message.get("argument", "")).strip()
            if action_type == "speak":
                return argument
            if action_type in ["non-verbal communication", "nonverbal", "non verbal communication"]:
                return f"[nonverbal] {argument}" if argument else ""
            if action_type == "action":
                return f"[action] {argument}" if argument else ""
            if action_type == "leave":
                return "left the conversation"
            if action_type == "none":
                return ""
            # Unknown action type → show raw
            return argument or str(message)

        # Mixed schema with parallel fields
        speak = str(message.get("speak", "")).strip() if message.get("speak") is not None else ""
        nonverbal = str(message.get("nonverbal", "")).strip() if message.get("nonverbal") is not None else ""
        act = str(message.get("action", "")).strip() if message.get("action") is not None else ""

        parts: List[str] = []
        if speak:
            parts.append(f'says: "{speak}"')
        if nonverbal:
            parts.append(f"[nonverbal] {nonverbal}")
        if act:
            parts.append(f"[action] {act}")
        return " ".join(parts)

    # Fallback
    return str(message)

def simulate_conversation(
    personA: SocialAgent,
    personB: SocialAgent,
    max_rounds: int,
    evaluator: ConversationEvaluator,
    scenario_index: int = 0,
    pair: Any = 0,
    environment = None,
    result = None,
    root_dir = None,
    run_mcq_tests: bool = True,
) -> Union[Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]], Tuple[List[Dict[str, Any]], Dict[str, List[Any]]]]:

    output_dir = f"{root_dir}"
    os.makedirs(output_dir, exist_ok=True)
 
    return run_single_scenario_simulation(
        personA=personA,
        personB=personB,
        environment=environment,
        num_turns=max_rounds,
        evaluator=evaluator,
        pair=pair,
        scenario_idx=scenario_index,
        output_dir=output_dir,
        run_mcq_tests=run_mcq_tests,
    )

def run_single_scenario_simulation(
    personA: SocialAgent,
    personB: SocialAgent,
    environment: EnvironmentProfile,
    evaluator: ConversationEvaluator,
    num_turns: int = 20,
    pair: Any = 0,
    scenario_idx: int = 0,
    output_dir: Optional[str] = None,
    run_mcq_tests: bool = True,
) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
  
    personA.env = environment
    personB.env = environment

    # Extract environment details
    agent_goals = environment.env["agent_goals"]
    agent_reasons = environment.env["agent_reasons"]
    agent_goals_mcqas = environment.env["agent_goals_mcqas"]
    agent_reasons_mcqas = environment.env["agent_reasons_mcqas"]
    agent_knowledge_mcqas = environment.env.get("agent_knowledge_mcqas", [])

    # Initialize conversation logs
    conversation_log = []
    mcq_logs = []

    # Config flag to disable repair/state even in barrier runs
    cfg = load_config()
    enable_repair_state = True
    try:
        flag = ((cfg or {}).get("models", {}) or {}).get("enable_repair_and_state")
        if isinstance(flag, bool):
            enable_repair_state = flag
    except Exception:
        pass

    # Initialize dynamic barrier state for Agent A (single-sided) only for barrier modes
    if enable_repair_state and environment and isinstance(environment.env, dict) and environment.env.get("barrier_type"):
        init_barrier_state(environment.env)

    print(f"🌐 Using agent profile models: {personA.name}({personA.profile.model_id}) ↔ {personB.name}({personB.profile.model_id})")

    # First message from agent A - using the agent's act method directly
    personA_message = personA.act(
        initial=True
    )
 
    conversation_log.append(f"{personA.name}: {_format_action_output(personA_message)}")
 
    for turn_num in range(num_turns):
        print(f"\n--- Round {turn_num+1} ---")

        personB.update_instruction(
            transcript=conversation_log,
            turn_number=turn_num,
        )

        personB_message = personB.act(
            personA_message
        )

        formatted_b = _format_action_output(personB_message)
        conversation_log.append(f"{personB.name}: {formatted_b}")

        # Sotopia-style leave detection
        b_left = False
        if isinstance(personB_message, dict) and personB_message.get("action_type") == "leave":
            b_left = True
            conversation_log.append(f"{personB.name} left the conversation")
            print(f"❌ {personB.name} left the conversation")
        elif isinstance(personB_message, str):
            if "left the conversation" in personB_message.lower():
                b_left = True
                print(f"❌ {personB.name} left the conversation")
            elif any(pattern in personB_message.lower() for pattern in ["goodbye", "bye", "i have to go", "leaving now"]):
                if turn_num >= 3:  
                    b_left = True
                    print(f"👋 {personB.name} indicated goodbye")

        barrier_type = environment.env.get("barrier_type") if environment and environment.env else None

        judge_json = None
        if enable_repair_state and barrier_type:
            judge_json = judge_repair_with_llm(formatted_b, conversation_log, barrier_type)
            print(judge_json)
            repair_score = float(judge_json.get("score", 0.0) or 0.0)
            update_barrier_state(environment.env, repair_score)

        if b_left:
            print(f"🚪 Conversation ended: explicit leave (Turn {turn_num})")
            break
            
        if turn_num >= num_turns:
            print(f"⏰ Conversation ended: maximum turns reached ({num_turns})")
            break
        
        if run_mcq_tests:
            goal_mcq_A = personB.predict_mcq_answer(
                agent_name=personB.name,
                partner_name=personA.name,
                transcript=conversation_log,
                mcqa=agent_goals_mcqas[1],
                test_prompt=evaluator.evaluation_template,
                task_type="goal",
            )
            
            reason_mcq_A = personB.predict_mcq_answer(
                agent_name=personB.name,
                partner_name=personA.name,
                transcript=conversation_log,
                mcqa=agent_reasons_mcqas[1],
                test_prompt=evaluator.evaluation_template,
                task_type="reason",
            )

        # Before Agent A's next turn, add barrier preface if still in barrier window
        # Update agent A's instructions first (this rebuilds the system prompt)
        personA.update_instruction(
            transcript=conversation_log,
            turn_number=turn_num,
        )

        personA_message = personA.act(
            personB_message
        )

        conversation_log.append(f"{personA.name}: {_format_action_output(personA_message)}")
        
        # Check if A decided to leave
        a_left = False
        if isinstance(personA_message, dict) and personA_message.get("action_type") == "leave":
            a_left = True
            conversation_log.append(f"{personA.name} left the conversation")
            print(f"❌ {personA.name} left the conversation")
        elif isinstance(personA_message, str):
            if "left the conversation" in personA_message.lower():
                a_left = True
                print(f"❌ {personA.name} left the conversation")
            elif any(pattern in personA_message.lower() for pattern in ["goodbye", "bye", "i have to go", "leaving now"]):
                if turn_num >= 3:  
                    a_left = True
                    print(f"👋 {personA.name} indicated goodbye")
        if a_left:
            break 

        if run_mcq_tests:
            # MCQ evaluations for agent B's goal and reason
            goal_mcq_B = personA.predict_mcq_answer(
                agent_name=personA.name,
                partner_name=personB.name,
                transcript=conversation_log,
                mcqa=agent_goals_mcqas[0],
                test_prompt=evaluator.evaluation_template,
                task_type="goal",
            )
            
            reason_mcq_B = personA.predict_mcq_answer(
                agent_name=personA.name,
                partner_name=personB.name,
                transcript=conversation_log,
                mcqa=agent_reasons_mcqas[0],
                test_prompt=evaluator.evaluation_template,
                task_type="reason",
            )

            # Log MCQ results and LLM repair judgment
            mcq_logs.append(
                {
                    "round": turn_num + 1,
                    "scenario": scenario_idx + 1,
                    f"agent_1_goal_mcq": goal_mcq_A,
                    f"agent_1_reason_mcq": reason_mcq_A,
                    f"agent_1_knowledge_mcq": None,
                    f"agent_2_goal_mcq": goal_mcq_B,
                    f"agent_2_reason_mcq": reason_mcq_B,
                    f"agent_2_knowledge_mcq": None,
                    "agent_b_repair_eval_llm": judge_json if enable_repair_state else None,
                    "barrier_state": environment.env.get("barrier_state") if (enable_repair_state and environment.env.get("barrier_type")) else None,
                }
            )
    
    # Evaluate conversation
    print("\n===== Evaluating Social Interaction =====")
    barrier_type_for_eval = environment.env.get("barrier_type") if environment and environment.env else None
    eval_result = evaluator.evaluate_conversation(
        conversation_log, agent_goals, agent_reasons, mcq_logs, barrier_type=barrier_type_for_eval
    )

    if output_dir:
        print(f"💾 Saving results to {output_dir}")
        scenario_output_dir = os.path.join(output_dir, f"scenario_{scenario_idx+1}")
        os.makedirs(scenario_output_dir, exist_ok=True)

        # Save evaluation results
        with open(os.path.join(scenario_output_dir, "eval_result.json"), "w") as f:
            json.dump(eval_result, f, indent=4)

        # Prepare comprehensive log data structure
        log_data = {
            "experimental_context": {
                "scenario": {
                    "description": environment.env['scenario'],
                    "agent_relationship": environment.env.get('agent_relationship', 'Unknown'),
                    "barrier_type": environment.env.get("barrier_type"),
                },
                "agents": {
                    "agent_a": {
                        "name": personA.name,
                        "profile": {
                            "first_name": personA.profile.first_name,
                            "last_name": personA.profile.last_name,
                            "age": personA.profile.age,
                            "occupation": personA.profile.occupation,
                            "personality_and_values": personA.profile.personality_and_values,
                            "public_info": personA.profile.public_info,
                            "model_id": personA.profile.model_id
                        },
                        "goal": agent_goals[0],
                        "reason": agent_reasons[0],
                        "private_knowledge": environment.env.get("agent1_private_knowledge", "").strip()
                    },
                    "agent_b": {
                        "name": personB.name,
                        "profile": {
                            "first_name": personB.profile.first_name,
                            "last_name": personB.profile.last_name,
                            "age": personB.profile.age,
                            "occupation": personB.profile.occupation,
                            "personality_and_values": personB.profile.personality_and_values,
                            "public_info": personB.profile.public_info,
                            "model_id": personB.profile.model_id
                        },
                        "goal": agent_goals[1],
                        "reason": agent_reasons[1],
                        "private_knowledge": environment.env.get("agent2_private_knowledge", "").strip()
                    }
                },
                "experimental_configuration": {
                    "max_rounds": num_turns,
                }
            },
            "conversation_log": conversation_log,
            "mcq_logs": mcq_logs
        }

        # Save comprehensive conversation log as JSON
        with open(os.path.join(scenario_output_dir, "conversation_log.json"), "w") as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)

        # Save human-readable conversation log as TXT
        with open(os.path.join(scenario_output_dir, "conversation_log.txt"), "w") as f:
            for line in conversation_log:
                f.write(line + "\n")

        # Save MCQ logs in both formats
        with open(os.path.join(scenario_output_dir, "mcq_logs.json"), "w") as f:
            json.dump(mcq_logs, f, indent=4, ensure_ascii=False)

        with open(os.path.join(scenario_output_dir, "mcq_logs.txt"), "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MCQ EVALUATION LOGS\n")
            f.write("=" * 80 + "\n\n")
            
            for mcq_entry in mcq_logs:
                f.write(f"=== Round {mcq_entry['round']} ===\n")
                
                # Agent A MCQs
                if mcq_entry.get('agent_1_goal_mcq'):
                    goal_mcq = mcq_entry['agent_1_goal_mcq']
                    f.write(f"{personA.name} Goal MCQ: {goal_mcq.get('answer', 'N/A')} (confidence: {goal_mcq.get('confidence', 0.0):.2f})\n")
                
                if mcq_entry.get('agent_1_reason_mcq'):
                    reason_mcq = mcq_entry['agent_1_reason_mcq']
                    f.write(f"{personA.name} Reason MCQ: {reason_mcq.get('answer', 'N/A')} (confidence: {reason_mcq.get('confidence', 0.0):.2f})\n")
                
                if mcq_entry.get('agent_1_knowledge_mcq'):
                    knowledge_mcq = mcq_entry['agent_1_knowledge_mcq']
                    f.write(f"{personA.name} Knowledge MCQ: {knowledge_mcq.get('answer', 'N/A')} (confidence: {knowledge_mcq.get('confidence', 0.0):.2f})\n")
                
                # Agent B MCQs
                if mcq_entry.get('agent_2_goal_mcq'):
                    goal_mcq = mcq_entry['agent_2_goal_mcq']
                    f.write(f"{personB.name} Goal MCQ: {goal_mcq.get('answer', 'N/A')} (confidence: {goal_mcq.get('confidence', 0.0):.2f})\n")
                
                if mcq_entry.get('agent_2_reason_mcq'):
                    reason_mcq = mcq_entry['agent_2_reason_mcq']
                    f.write(f"{personB.name} Reason MCQ: {reason_mcq.get('answer', 'N/A')} (confidence: {reason_mcq.get('confidence', 0.0):.2f})\n")
                
                if mcq_entry.get('agent_2_knowledge_mcq'):
                    knowledge_mcq = mcq_entry['agent_2_knowledge_mcq']
                    f.write(f"{personB.name} Knowledge MCQ: {knowledge_mcq.get('answer', 'N/A')} (confidence: {knowledge_mcq.get('confidence', 0.0):.2f})\n")
                
                f.write("\n")
    return conversation_log, eval_result, mcq_logs


    
