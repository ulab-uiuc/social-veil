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

def simulate_conversation(
    personA: SocialAgent,
    personB: SocialAgent,
    max_rounds: int,
    evaluator: ConversationEvaluator,
    encryption_enabled: bool = False,
    action_enabled: bool = False,
    nature_language: bool = False,
    output_suffix: str = "default",
    scenario_index: int = 0,
    pair: Any = 0,
    environment = None,
    result = None,
    root_dir = None,
    memory_enabled: bool = False,
) -> Union[Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]], Tuple[List[Dict[str, Any]], Dict[str, List[Any]]]]:

    output_dir = f"{root_dir}"
    os.makedirs(output_dir, exist_ok=True)
 
    return run_single_scenario_simulation(
        personA=personA,
        personB=personB,
        environment=environment,
        num_turns=max_rounds,
        evaluator=evaluator,
        encryption_enabled=encryption_enabled,
        action_enabled=action_enabled,
        nature_language=nature_language,
        pair=pair,
        scenario_idx=scenario_index,
        output_dir=output_dir,
    )

def run_single_scenario_simulation(
    personA: SocialAgent,
    personB: SocialAgent,
    environment: EnvironmentProfile,
    evaluator: ConversationEvaluator,
    num_turns: int = 20,
    encryption_enabled: bool = False,
    action_enabled: bool = False,
    nature_language: bool = False,
    pair: Any = 0,
    scenario_idx: int = 0,
    output_dir: Optional[str] = None,
) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
  
    # Set environment for agents
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

    # Inject new barrier prompts from episode, if present
    barrier_cues = environment.env.get("barrier_cues") if environment and environment.env else None

    # Apply barrier_cues to early transcript priming only (no scenario or profile changes)
    if isinstance(barrier_cues, dict):
        # Opening seed: pre-seed the transcript to bias first turns
        opening_seed = barrier_cues.get("opening_seed")
        preseed_lines = []
        if isinstance(opening_seed, list):
            for item in opening_seed[:2]:
                if isinstance(item, dict):
                    spk = item.get("speaker")
                    txt = item.get("text")
                    if isinstance(spk, str) and isinstance(txt, str) and txt.strip():
                        name = personA.name if spk.upper() == "A" else personB.name
                        preseed_lines.append(f"{name}: {txt.strip()}")
        if preseed_lines:
            conversation_log.extend(preseed_lines)

    print(f"🌐 Using agent profile models: {personA.name}({personA.profile.model_id}) ↔ {personB.name}({personB.profile.model_id})")

    # First message from agent A - using the agent's act method directly
    personA_message = personA.act(
        initial=True
    )
 
    conversation_log.append(f"{personA.name}: {personA_message}")
 
    for turn_num in range(num_turns):
        print(f"\n--- Round {turn_num+1} ---")

        personB.update_instruction(
            transcript=conversation_log,
            turn_number=turn_num,
        )

        personB_message = personB.act(
            personA_message
        )
        if isinstance(personB_message, dict):
            response_text = ""
            if personB_message.get("speak"):
                response_text += f'says: "{personB_message["speak"]}" '
            if personB_message.get("nonverbal"):
                response_text += f'[nonverbal] {personB_message["nonverbal"]} '
            if personB_message.get("action"):
                response_text += f'[action] {personB_message["action"]} '
            
            conversation_log.append(f"{personB.name}: {response_text.strip()}")
        else:
            conversation_log.append(f"{personB.name}: {personB_message}")

        # Sotopia-style leave detection
        b_left = False
        
        # Method 1: Action-based (like Sotopia's ActionType)
        if action_enabled and isinstance(personB_message, dict) and personB_message.get("action_type") == "leave":
            b_left = True
            conversation_log.append(f"{personB.name} left the conversation")
            print(f"❌ {personB.name} left the conversation")
            
        # Method 2: Natural language parsing (like Sotopia's parse_single_dialogue)
        elif isinstance(personB_message, str):
            if "left the conversation" in personB_message.lower():
                b_left = True
                print(f"❌ {personB.name} left the conversation")
            # Additional Sotopia-style leave patterns
            elif any(pattern in personB_message.lower() for pattern in ["goodbye", "bye", "i have to go", "leaving now"]):
                if turn_num >= 3:  # Only after some conversation
                    b_left = True
                    print(f"👋 {personB.name} indicated goodbye")

        # Check termination conditions (Sotopia-style)
        if b_left:
            print(f"🚪 Conversation ended: explicit leave (Turn {turn_num})")
            break
            
        # Turn limit check (Sotopia uses ~20 turns max)
        if turn_num >= num_turns:
            print(f"⏰ Conversation ended: maximum turns reached ({num_turns})")
            break

        # MCQ evaluations for agent A's goal and reason
        goal_mcq_A = personB.predict_mcq_answer(
            agent_name=personB.name,
            partner_name=personA.name,
            transcript=conversation_log,
            mcqa=agent_goals_mcqas[0],
            test_prompt=evaluator.evaluation_template,
            task_type="goal",
        )
        
        reason_mcq_A = personB.predict_mcq_answer(
            agent_name=personB.name,
            partner_name=personA.name,
            transcript=conversation_log,
            mcqa=agent_reasons_mcqas[0],
            test_prompt=evaluator.evaluation_template,
            task_type="reason",
        )

        # Knowledge barrier MCQ testing for agent A's private knowledge
        knowledge_mcq_A = None
        agent1_private_knowledge = environment.env.get("agent1_private_knowledge", "").strip()
        if (agent_knowledge_mcqas and len(agent_knowledge_mcqas) >= 1 and agent1_private_knowledge):
            knowledge_mcq_A = personB.predict_mcq_answer(
                agent_name=personB.name,
                partner_name=personA.name,
                transcript=conversation_log,
                mcqa=agent_knowledge_mcqas[0],
                test_prompt=evaluator.evaluation_template,
                task_type="knowledge",
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

        conversation_log.append(f"{personA.name}: {personA_message}")
        
        # Check if A decided to leave
        a_left = False
        if action_enabled and isinstance(personA_message, dict) and personA_message.get("action_type") == "leave":
            a_left = True
            conversation_log.append(f"{personA.name} left the conversation")
            print(f"❌ {personA.name} left the conversation")
        elif not action_enabled and isinstance(personA_message, str) and "left the conversation" in personA_message.lower():
            a_left = True
            print(f"❌ {personA.name} left the conversation")

        if a_left:
            break 

        # MCQ evaluations for agent B's goal and reason
        goal_mcq_B = personA.predict_mcq_answer(
            agent_name=personA.name,
            partner_name=personB.name,
            transcript=conversation_log,
            mcqa=agent_goals_mcqas[1],
            test_prompt=evaluator.evaluation_template,
            task_type="goal",
        )
        
        reason_mcq_B = personA.predict_mcq_answer(
            agent_name=personA.name,
            partner_name=personB.name,
            transcript=conversation_log,
            mcqa=agent_reasons_mcqas[1],
            test_prompt=evaluator.evaluation_template,
            task_type="reason",
        )

        # Knowledge barrier MCQ testing for agent B's private knowledge
        knowledge_mcq_B = None
        agent2_private_knowledge = environment.env.get("agent2_private_knowledge", "").strip()
        if (agent_knowledge_mcqas and len(agent_knowledge_mcqas) >= 2 and agent2_private_knowledge):
            knowledge_mcq_B = personA.predict_mcq_answer(
                agent_name=personA.name,
                partner_name=personB.name,
                transcript=conversation_log,
                mcqa=agent_knowledge_mcqas[1],
                test_prompt=evaluator.evaluation_template,
                task_type="knowledge",
            )

        # Log MCQ results
        mcq_logs.append(
            {
                "round": turn_num + 1,
                "scenario": scenario_idx + 1,
                f"agent_1_goal_mcq": goal_mcq_A,
                f"agent_1_reason_mcq": reason_mcq_A,
                f"agent_1_knowledge_mcq": knowledge_mcq_A,
                f"agent_2_goal_mcq": goal_mcq_B,
                f"agent_2_reason_mcq": reason_mcq_B,
                f"agent_2_knowledge_mcq": knowledge_mcq_B,
            }
        )
    
    # Evaluate conversation
    print("\n===== Evaluating Social Interaction =====")
    eval_result = evaluator.evaluate_conversation(
        conversation_log, agent_goals, agent_reasons, mcq_logs
    )

    if output_dir:
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
                    "agent_relationship": environment.env.get('agent_relationship', 'Unknown')
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
                    "encryption_enabled": encryption_enabled,
                    "action_enabled": action_enabled,
                    "nature_language": nature_language,
                    "max_rounds": num_turns,
                    "barrier_language": barrier_language if (encryption_enabled and nature_language and 'barrier_language' in locals()) else None
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


    
