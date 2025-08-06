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
    root_dir = None,
    memory_enabled: bool = False
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
        scenario_idx=0,
        mix=mix,
        output_dir=output_dir,
        memory_enabled=memory_enabled,
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
    output_dir: Optional[str] = None,
    memory_enabled: bool = False,
) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
  
    # Set environment for agents
    personA.env = environment
    personB.env = environment
    
    # Reset memory for each independent scenario simulation
    if memory_enabled:
        personA.reset_memory_for_scenario(memory_enabled=True)
        personB.reset_memory_for_scenario(memory_enabled=True)

    # Extract environment details
    agent_goals = environment.env["agent_goals"]
    agent_reasons = environment.env["agent_reasons"]
    agent_goals_mcqas = environment.env["agent_goals_mcqas"]
    agent_reasons_mcqas = environment.env["agent_reasons_mcqas"]
    agent_knowledge_mcqas = environment.env.get("agent_knowledge_mcqas", [])

    # Initialize conversation logs
    conversation_log = []
    encrypted_conversation_log = []
    mcq_logs = []

    barrier = encryption_enabled and nature_language

    if barrier:
        barrier_language = "Chinese" 
        encryption1 = LanguageModelEncryption(
            target_language=barrier_language, model_id=personA.profile.model_id
        )
        encryption2 = None
        personA.set_encryption(encryption1)
        personB.set_encryption(encryption2)

    else:
        barrier_language = None

    print(f"🌐 Using agent profile models: {personA.name}({personA.profile.model_id}) ↔ {personB.name}({personB.profile.model_id})")

    # First message from agent A - using the agent's act method directly
    personA_message = personA.act(
        initial=True, use_action=action_enabled
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

        # Real-time memory updates if memory is enabled
        if memory_enabled and turn_num > 0:  # Skip first turn since no prior exchange
            # Get the last exchange for memory update
            if len(conversation_log) >= 2:
                # Extract clean messages for memory analysis
                last_personA_msg = conversation_log[-2].split(": ", 1)[1] if ": " in conversation_log[-2] else ""
                last_personB_msg = conversation_log[-1].split(": ", 1)[1] if ": " in conversation_log[-1] else ""
                
                # Update memory for both agents based on the exchange
                personA.update_memory_from_exchange(
                    agent_message=last_personA_msg,
                    partner_response=last_personB_msg,
                    turn_number=turn_num + 1,
                    memory_enabled=memory_enabled
                )
                
                # For personB, we need to get their message and personA's response
                if len(conversation_log) >= 4:  # Make sure we have enough history
                    prev_personB_msg = conversation_log[-4].split(": ", 1)[1] if ": " in conversation_log[-4] else ""
                    prev_personA_response = conversation_log[-3].split(": ", 1)[1] if ": " in conversation_log[-3] else ""
                    
                    personB.update_memory_from_exchange(
                        agent_message=prev_personB_msg,
                        partner_response=prev_personA_response,
                        turn_number=turn_num,
                        memory_enabled=memory_enabled
                    )

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
                    "mix_mode": mix,
                    "max_rounds": num_turns,
                    "barrier_language": barrier_language if (encryption_enabled and nature_language and 'barrier_language' in locals()) else None
                }
            },
            "conversation_log": {
                "raw_messages": conversation_log,
                "encrypted_messages": encrypted_conversation_log
            },
            "mcq_logs": mcq_logs
        }

        # Save comprehensive conversation log as JSON
        with open(os.path.join(scenario_output_dir, "conversation_log.json"), "w") as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)

        # Save human-readable conversation log as TXT
        with open(os.path.join(scenario_output_dir, "conversation_log.txt"), "w") as f:
            # Write experimental context header
            f.write("=" * 80 + "\n")
            f.write("EXPERIMENTAL CONTEXT\n")
            f.write("=" * 80 + "\n\n")
            
            # Environment information
            f.write("📝 SCENARIO:\n")
            f.write(f"{environment.env['scenario']}\n\n")
            
            # Agent profiles
            f.write("👥 AGENT PROFILES:\n")
            f.write(f"Agent A: {personA.name}\n")
            f.write(f"  - Profile: {personA.profile.first_name} {personA.profile.last_name}, {personA.profile.age} years old\n")
            f.write(f"  - Occupation: {personA.profile.occupation}\n")
            f.write(f"  - Personality: {personA.profile.personality_and_values}\n")
            f.write(f"  - Public Info: {personA.profile.public_info}\n")
            f.write(f"  - Model: {personA.profile.model_id}\n\n")
            
            f.write(f"Agent B: {personB.name}\n")
            f.write(f"  - Profile: {personB.profile.first_name} {personB.profile.last_name}, {personB.profile.age} years old\n")
            f.write(f"  - Occupation: {personB.profile.occupation}\n")
            f.write(f"  - Personality: {personB.profile.personality_and_values}\n")
            f.write(f"  - Public Info: {personB.profile.public_info}\n")
            f.write(f"  - Model: {personB.profile.model_id}\n\n")
            
            # Agent goals and reasons
            f.write("🎯 AGENT GOALS:\n")
            f.write(f"{personA.name}'s Goal: {agent_goals[0]}\n")
            f.write(f"{personA.name}'s Reason: {agent_reasons[0]}\n\n")
            f.write(f"{personB.name}'s Goal: {agent_goals[1]}\n")
            f.write(f"{personB.name}'s Reason: {agent_reasons[1]}\n\n")
            
            # Private knowledge (if any)
            agent1_private = environment.env.get("agent1_private_knowledge", "").strip()
            agent2_private = environment.env.get("agent2_private_knowledge", "").strip()
            if agent1_private or agent2_private:
                f.write("🔒 PRIVATE KNOWLEDGE:\n")
                if agent1_private:
                    f.write(f"{personA.name}'s Private Knowledge: {agent1_private}\n")
                if agent2_private:
                    f.write(f"{personB.name}'s Private Knowledge: {agent2_private}\n")
                f.write("\n")
            
            # Experimental configuration
            f.write("⚙️ EXPERIMENTAL CONFIGURATION:\n")
            f.write(f"Encryption Enabled: {encryption_enabled}\n")
            f.write(f"Action Enabled: {action_enabled}\n")
            f.write(f"Nature Language: {nature_language}\n")
            f.write(f"Mix Mode: {mix}\n")
            f.write(f"Max Rounds: {num_turns}\n")
            if encryption_enabled and nature_language and 'barrier_language' in locals():
                f.write(f"Barrier Language: {barrier_language}\n")
            f.write(f"Agent Relationship: {environment.env.get('agent_relationship', 'Unknown')}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("CONVERSATION LOG\n")
            f.write("=" * 80 + "\n\n")
            
            # Write actual conversation
            for line in conversation_log:
                f.write(line + "\n")

        # Save encrypted conversation logs in both formats
        with open(os.path.join(scenario_output_dir, "encrypted_conversation_log.json"), "w") as f:
            encrypted_log_data = {
                "experimental_context": log_data["experimental_context"],
                "encrypted_conversation": encrypted_conversation_log
            }
            json.dump(encrypted_log_data, f, indent=4, ensure_ascii=False)

        with open(os.path.join(scenario_output_dir, "encrypted_conversation_log.txt"), "w") as f:
            f.write("=" * 80 + "\n")
            f.write("ENCRYPTED CONVERSATION LOG\n")
            f.write("=" * 80 + "\n\n")
            for line in encrypted_conversation_log:
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
    
