
import json
import os
import random
from typing import Any, Dict, List

from agency_swarm import Agency, Agent

from social_decipher.encryption import (LanguageModelEncryption,
                                        MappingEncryption)
from social_decipher.evaluate import ConversationEvaluator
from social_decipher.utils.model import ModelManager
from social_decipher.utils.plot import plot_mcq_scores, plot_social_goal


def simulate_conversation(
    personA: Agent,
    personB: Agent,
    num_turns: int,
    agent_goals: List[str],
    agent_reasons: List[str],
    agent_goals_mcqas: Dict[str, Any],
    agent_reasons_mcqas: Dict[str, Any],
    evaluator: ConversationEvaluator,
    encryption_enabled: bool = False,
    action_enabled: bool = False,
    nature_language: bool = False, 
    output_suffix: str = "default",
) -> None:
    agency = Agency(
        [
            personA,
            [personA, personB],
            [personB, personA],
            [personA, personB],
            [personB, personA],
        ],  # Define the conversation participants.
        temperature=0.3,
        max_prompt_tokens=10000,
    )

    conversation_log = []
    encrypted_conversation_log = []
    mcq_logs = []

    personA.set_agency(agency)
    personB.set_agency(agency)

    if encryption_enabled:
        if nature_language:
            strong_model, weak_model, barrier_language = ModelManager.language_barrier_pair(0)
            
            encryption1 = LanguageModelEncryption(
                target_language=barrier_language,
                model_id=strong_model
            )
            
            encryption2 = LanguageModelEncryption(
                target_language=barrier_language,
                model_id=weak_model
            )
            
            strong_provider = ModelManager.MODEL_PROVIDERS.get(strong_model, {}).get("provider", "unknown")
            weak_provider = ModelManager.MODEL_PROVIDERS.get(weak_model, {}).get("provider", "unknown")
            
            print(f"🌐 Language barrier mode enabled:")
            print(f"  - Agent 1 ({personA.name}) using {strong_model} ({strong_provider}) for {barrier_language}")
            print(f"  - Agent 2 ({personB.name}) using {weak_model} ({weak_provider}) for {barrier_language}")
        else:
            encryption1 = MappingEncryption(key=random.randint(1, 100))
            encryption2 = MappingEncryption(key=random.randint(1, 100))

        personA.set_encryption(encryption1)
        personB.set_encryption(encryption2)

    personA_message = personA.act(message=None, initial=True, use_action=action_enabled)
    conversation_log.append(f"{personA.name}: {personA.log[-1]['response_raw']}")

    for num in range(num_turns):
        print("\n")
        print(f"################# ROUND{num+1} #################")

        personB.update_instruction(
            transcript=encrypted_conversation_log,
            turn_number=num,
            use_action=action_enabled
        )
        
        personB_message = personB.act(personA_message, use_action=action_enabled)
        conversation_log.append(f"Agent 2: {personB.log[-1]['response_raw']}")
        encrypted_conversation_log.append(f"Agent 2: {personB.log[-1]['response_encrypted']}")

        goal_mcq_A = personB.predict_mcq_answer(
            transcript=encrypted_conversation_log,
            mcqa=agent_goals_mcqas[0],
            test_prompt = evaluator.evaluation_template,
            task_type="goal"
        )
        reason_mcq_A = personB.predict_mcq_answer(
            transcript=encrypted_conversation_log,
            mcqa=agent_reasons_mcqas[0],
            test_prompt = evaluator.evaluation_template,
            task_type="reason"
        )

        personA.update_instruction(
            transcript=encrypted_conversation_log,
            turn_number=num,
            use_action=action_enabled
        )

        personA_message = personA.act(personB_message, use_action=action_enabled)
        conversation_log.append(f"Agent 1: {personA.log[-1]['response_raw']}")
        encrypted_conversation_log.append(f"Agent 1: {personA.log[-1]['response_encrypted']}")

        goal_mcq_B = personA.predict_mcq_answer(
            transcript=encrypted_conversation_log,
            mcqa=agent_goals_mcqas[1],
            test_prompt = evaluator.evaluation_template,
            task_type="goal"
        )
        reason_mcq_B = personA.predict_mcq_answer(
            transcript=encrypted_conversation_log,
            mcqa=agent_reasons_mcqas[1],
            test_prompt = evaluator.evaluation_template,
            task_type="reason"
        )

        mcq_logs.append({
            "round": num + 1,
            f"{personA.name}_goal_mcq": goal_mcq_A,
            f"{personA.name}_reason_mcq": reason_mcq_A,
            f"{personB.name}_goal_mcq": goal_mcq_B,
            f"{personB.name}_reason_mcq": reason_mcq_B,
        })

    print("\n===== Evaluating Social Interaction =====")
    eval_result = evaluator.evaluate_conversation(
        conversation_log, 
        agent_goals, 
        agent_reasons
    )

    output_dir = f"../social_decipher/results/exp_{output_suffix}"
    os.makedirs(output_dir, exist_ok=True)

    # Save files into output directory
    with open(os.path.join(output_dir, "eval_result.json"), "w") as f:
        json.dump(eval_result, f, indent=4)

    with open(os.path.join(output_dir, "conversation_log.txt"), "w") as f:
        for line in conversation_log:
            f.write(line + "\n")

    with open(os.path.join(output_dir, "mcq_logs.json"), "w") as f:
        json.dump(mcq_logs, f, indent=4)

    plot_mcq_scores(
        mcq_scores=mcq_logs,
        agent_names=[personA.name, personB.name],
        save_path=os.path.join(output_dir, "mcq_trends.png")
    )

    plot_social_goal(
        eval_result,
        [personA.name, personB.name],
        save_dir=output_dir
    )