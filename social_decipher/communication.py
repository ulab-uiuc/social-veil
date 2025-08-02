import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from agency_swarm import Agency
from rich import print

from social_decipher.agent.social_agent import SocialAgent
from social_decipher.encryption import (LanguageModelEncryption)
from social_decipher.environment.env_generator import EnvironmentGenerator
from social_decipher.environment.env_profile import EnvironmentProfile
from social_decipher.evaluate import ConversationEvaluator
from social_decipher.utils.model import ModelManager
from social_decipher.utils.plot import (plot_cross_scenario_performance,
                                        plot_mcq_scores, plot_social_goal)

def simulate_conversation(
    personA: SocialAgent,
    personB: SocialAgent,
    max_rounds: int,
    evaluator: ConversationEvaluator,
    encryption_enabled: bool = False,
    action_enabled: bool = False,
    nature_language: bool = False,
    output_suffix: str = "default",
    pair: Any = 0,
    mix: bool = False,
    client = None,
    environment = None,
    result = None,
    root_dir = None
) -> Union[Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]], Tuple[List[Dict[str, Any]], Dict[str, List[Any]]]]:

    output_dir = f"{root_dir}/exp_{output_suffix}"
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
        scenario_idx=0,
        mix=mix,
        save_results=True,
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
    mix: bool = False,
    save_results: bool = True,
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

    # DEBUG: Print MCQ data to identify the issue
    print(f"\n🔍 MCQ Data Check:")
    print(f"📖 Scenario: {environment.env['scenario']}")
    print(f"🎯 Goals: {personA.name}({agent_goals[0]}), {personB.name}({agent_goals[1]})")

    def print_mcq(label, name, mcq):
        print(f"{label} {name}: {mcq['question']}")
        for k, v in mcq['options'].items():
            print(f"    {k}: {v}")

    if len(agent_goals_mcqas) > 0:
        print_mcq('❓ Goal MCQ', personA.name, agent_goals_mcqas[0])
    if len(agent_goals_mcqas) > 1:
        print_mcq('❓ Goal MCQ', personB.name, agent_goals_mcqas[1])
    
    if len(agent_reasons_mcqas) > 0:
        print_mcq('🤔 Reason MCQ', personA.name, agent_reasons_mcqas[0])
    if len(agent_reasons_mcqas) > 1:
        print_mcq('🤔 Reason MCQ', personB.name, agent_reasons_mcqas[1])

    # Initialize conversation logs
    conversation_log = []
    encrypted_conversation_log = []
    mcq_logs = []

    use_direct_api = encryption_enabled and nature_language
    print(f"Using direct API: {use_direct_api}")
    
    if use_direct_api:
        # Check if models are already set in agent profiles
        if personA.profile.model_id and personB.profile.model_id:
            # Use models from agent profiles
            strong_model = personA.profile.model_id
            weak_model = personB.profile.model_id
            barrier_language = "Chinese"  # Default barrier language, can be customized
            print(f"🌐 Using agent profile models: {personA.name}({strong_model}) ↔ {personB.name}({weak_model})")
        else:
            # Fall back to language barrier pair if models not set
            strong_model, weak_model, barrier_language = ModelManager.language_barrier_pair(pair)
            personA.profile.model_id = strong_model
            personB.profile.model_id = weak_model
            print(f"🌐 Language Barrier (fallback): {personA.name}({strong_model}) ↔ {personB.name}({weak_model}) - {barrier_language}")

        strong_provider = ModelManager.MODEL_PROVIDERS.get(strong_model, {}).get(
            "provider"
        )
        weak_provider = ModelManager.MODEL_PROVIDERS.get(weak_model, {}).get(
            "provider"
        )
    else:
        print(f"🔄 Standard Mode: {personA.name} ↔ {personB.name}")
        barrier_language = None

    # Set up encryption
    if encryption_enabled and nature_language:
        # Use models from agent profiles or fall back to language barrier pair
        if personA.profile.model_id and personB.profile.model_id:
            strong_model = personA.profile.model_id
            weak_model = personB.profile.model_id
            barrier_language = "Chinese"  # Default barrier language
        else:
            strong_model, weak_model, barrier_language = ModelManager.language_barrier_pair(pair)
        
        encryption1 = LanguageModelEncryption(
            target_language=barrier_language, model_id=strong_model
        )
        encryption2 = None
        personA.set_encryption(encryption1)
        personB.set_encryption(encryption2)

    # First message from agent A - using the agent's act method directly
    personA_message = personA.act(
        message=None, initial=True, use_action=action_enabled
    )

    if mix and isinstance(personA_message, dict):
        # Format mixed action response for logs
        response_text = ""
        if personA_message.get("speak"):
            response_text += f'says: "{personA_message["speak"]}" '
        if personA_message.get("nonverbal"):
            response_text += f'[nonverbal] {personA_message["nonverbal"]} '
        if personA_message.get("action"):
            response_text += f'[action] {personA_message["action"]} '
        
        conversation_log.append(f"{personA.name}: {response_text.strip()}")
        encrypted_conversation_log.append(f"{personA.name}: {response_text.strip()}")
    else:
        # Keep your existing logging for non-mix formats
        conversation_log.append(f"{personA.name}: {personA.log[-1]['response_raw']}")
        encrypted_conversation_log.append(f"{personA.name}: {personA.log[-1]['response_encrypted']}")

    for turn_num in range(num_turns):
        print(f"\n--- Round {turn_num+1} ---")

        personB.update_instruction(
            transcript=encrypted_conversation_log,
            turn_number=turn_num,
            use_action=action_enabled,
            mix = mix
        )

        personB_message = personB.act(
            personA_message, use_action=action_enabled
        )
        if mix and isinstance(personB_message, dict):
            response_text = ""
            if personB_message.get("speak"):
                response_text += f'says: "{personB_message["speak"]}" '
            if personB_message.get("nonverbal"):
                response_text += f'[nonverbal] {personB_message["nonverbal"]} '
            if personB_message.get("action"):
                response_text += f'[action] {personB_message["action"]} '
            
            conversation_log.append(f"{personB.name}: {response_text.strip()}")
            encrypted_conversation_log.append(f"{personB.name}: {response_text.strip()}")
        else:
            conversation_log.append(f"{personB.name}: {personB.log[-1]['response_raw']}")
            encrypted_conversation_log.append(
                f"{personB.name}: {personB.log[-1]['response_encrypted']}"
            )
  
        b_left = False
        if action_enabled and isinstance(personB_message, dict) and personB_message.get("action_type") == "leave":
            b_left = True
            conversation_log.append(f"{personB.name} left the conversation")
            encrypted_conversation_log.append(f"{personB.name} left the conversation")
            print(f"❌ {personB.name} left the conversation")
        elif not action_enabled and isinstance(personB_message, str) and "left the conversation" in personB_message.lower():
            b_left = True
            print(f"❌ {personB.name} left the conversation")

        if b_left:
            break

        # MCQ evaluations for agent A's goal and reason
        goal_mcq_A = personB.predict_mcq_answer(
            agent_name=personB.name,
            partner_name=personA.name,
            transcript=encrypted_conversation_log,
            mcqa=agent_goals_mcqas[0],
            test_prompt=evaluator.evaluation_template,
            task_type="goal",
        )
        
        reason_mcq_A = personB.predict_mcq_answer(
            agent_name=personB.name,
            partner_name=personA.name,
            transcript=encrypted_conversation_log,
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
                transcript=encrypted_conversation_log,
                mcqa=agent_knowledge_mcqas[0],
                test_prompt=evaluator.evaluation_template,
                task_type="knowledge",
            )

        # Update agent A's instructions
        personA.update_instruction(
            transcript=encrypted_conversation_log,
            turn_number=turn_num,
            use_action=action_enabled,
            mix = mix
        )

        personA_message = personA.act(
            personB_message, use_action=action_enabled
        )

        if mix and isinstance(personA_message, dict):
            # Format mixed action response for logs
            response_text = ""
            if personA_message.get("speak"):
                response_text += f'says: "{personA_message["speak"]}" '
            if personA_message.get("nonverbal"):
                response_text += f'[nonverbal] {personA_message["nonverbal"]} '
            if personA_message.get("action"):
                response_text += f'[action] {personA_message["action"]} '
            
            conversation_log.append(f"{personA.name}: {response_text.strip()}")
            encrypted_conversation_log.append(f"{personA.name}: {response_text.strip()}")
        else:
            # Keep your existing logging for non-mix formats
            conversation_log.append(f"{personA.name}: {personA.log[-1]['response_raw']}")
            encrypted_conversation_log.append(f"{personA.name}: {personA.log[-1]['response_encrypted']}")
        
        # Check if A decided to leave
        a_left = False
        if action_enabled and isinstance(personA_message, dict) and personA_message.get("action_type") == "leave":
            a_left = True
            conversation_log.append(f"{personA.name} left the conversation")
            encrypted_conversation_log.append(f"{personA.name} left the conversation")
            print(f"❌ {personA.name} left the conversation")
        elif not action_enabled and isinstance(personA_message, str) and "left the conversation" in personA_message.lower():
            a_left = True
            print(f"❌ {personA.name} left the conversation")

        if a_left:
            break  # End conversation

        # MCQ evaluations for agent B's goal and reason
        goal_mcq_B = personA.predict_mcq_answer(
            agent_name=personA.name,
            partner_name=personB.name,
            transcript=encrypted_conversation_log,
            mcqa=agent_goals_mcqas[1],
            test_prompt=evaluator.evaluation_template,
            task_type="goal",
        )
        
        reason_mcq_B = personA.predict_mcq_answer(
            agent_name=personA.name,
            partner_name=personB.name,
            transcript=encrypted_conversation_log,
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
                transcript=encrypted_conversation_log,
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
        encrypted_conversation_log, agent_goals, agent_reasons, mcq_logs
    )

    print(eval_result)
    # print(output_dir)
    print("###############################")
    print(conversation_log)
    print("###############################")


    # Save results if requested
    if save_results and output_dir:
        scenario_output_dir = os.path.join(output_dir, f"scenario_{scenario_idx+1}")
        os.makedirs(scenario_output_dir, exist_ok=True)

        # Save evaluation results
        with open(os.path.join(scenario_output_dir, "eval_result.json"), "w") as f:
            json.dump(eval_result, f, indent=4)

        # Save conversation logs
        with open(os.path.join(scenario_output_dir, "conversation_log.txt"), "w") as f:
            for line in conversation_log:
                f.write(line + "\n")

        # Save encrypted conversation logs
        with open(os.path.join(scenario_output_dir, "encrypted_conversation_log.txt"), "w") as f:
            for line in encrypted_conversation_log:
                f.write(line + "\n")

    return conversation_log, eval_result, mcq_logs
    
