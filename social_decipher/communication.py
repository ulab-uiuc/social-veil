import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from agency_swarm import Agency
from rich import print

from social_decipher.agent.social_agent import SocialAgent
from social_decipher.encryption import (LanguageModelEncryption,
                                        MappingEncryption)
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
    num_scenarios: int = 1,
    client = None,
    environments = None,
    result = None
) -> Union[Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]], Tuple[List[Dict[str, Any]], Dict[str, List[Any]]]]:

    output_dir = f"../social_decipher/results/exp_{output_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    
    # For multi-scenario experiments
    if num_scenarios > 1:
        if environments is None or len(environments) < num_scenarios:
            print(f"Generating {num_scenarios} environments...")
            
        return run_multi_scenario_simulation(
            personA=personA,
            personB=personB,
            environments=environments,
            num_turns=max_rounds,
            evaluator=evaluator,
            encryption_enabled=encryption_enabled,
            action_enabled=action_enabled,
            nature_language=nature_language,
            pair=pair,
            save_results=True,
            output_dir=output_dir,
            result=result,
        )
    else:
        environment = environments[0] if environments and len(environments) > 0 else personA.env
        print(f"Using environment: {environment.env}")
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
            save_results=True,
            output_dir=output_dir,
        )

def run_single_scenario_simulation(
    personA: SocialAgent,
    personB: SocialAgent,
    environment: EnvironmentProfile,
    num_turns: int,
    evaluator: ConversationEvaluator,
    encryption_enabled: bool = False,
    action_enabled: bool = False,
    nature_language: bool = False,
    pair: Any = 0,
    scenario_idx: int = 0,
    save_results: bool = True,
    output_dir: Optional[str] = None,
) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
  
    print(f"\n===== CHECKING MEMORY AT START OF SCENARIO {scenario_idx+1} =====")
    print(f"Agent {personA.name} memory:")
    print(f"- Scenarios participated: {personA.memory.scenarios_participated}")
    print(f"- Goal history: {len(personA.memory.goals_history)} entries")

    print(f"\nAgent {personB.name} memory:")
    print(f"- Scenarios participated: {personB.memory.scenarios_participated}")
    print(f"- Goal history: {len(personB.memory.goals_history)} entries")

    if personA.memory.scenarios_participated > 0:
        print(f"\nMemory sample {personA.name}:")
        print(personA.memory.get_memory_context(detailed=True)[:200] + "...")

    if personB.memory.scenarios_participated > 0:
        print(f"\nMemory sample {personB.name}:")
        print(personB.memory.get_memory_context(detailed=True)[:200] + "...")

    # Set environment for agents
    personA.env = environment
    personB.env = environment

    # Extract environment details
    agent_goals = environment.env["agent_goals"]
    agent_reasons = environment.env["agent_reasons"]
    agent_goals_mcqas = environment.env["agent_goals_mcqas"]
    agent_reasons_mcqas = environment.env["agent_reasons_mcqas"]


    # Initialize conversation logs
    conversation_log = []
    encrypted_conversation_log = []
    mcq_logs = []

    # Determine if we should use direct API (for natural language barriers)
    use_direct_api = encryption_enabled and nature_language
    print(f"Using direct API: {use_direct_api}")
    
    if use_direct_api:
        strong_model, weak_model, barrier_language = ModelManager.language_barrier_pair(pair)

        personA.profile.profile["model_id"] = strong_model
        personB.profile.profile["model_id"] = weak_model

        strong_provider = ModelManager.MODEL_PROVIDERS.get(strong_model, {}).get(
            "provider", "unknown"
        )
        weak_provider = ModelManager.MODEL_PROVIDERS.get(weak_model, {}).get(
            "provider", "unknown"
        )

        print("🌐 Language barrier mode enabled (using direct API calls):")
        print(
            f"  - Agent 1 ({personA.name}) using {strong_model} ({strong_provider}) for {barrier_language}"
        )
        print(
            f"  - Agent 2 ({personB.name}) using {weak_model} ({weak_provider}) for {barrier_language}"
        )

        strong_understands = ModelManager.can_model_understand_language(
            strong_model, barrier_language
        )
        weak_understands = ModelManager.can_model_understand_language(
            weak_model, barrier_language
        )
        print(f"  - Agent 1 can understand {barrier_language}: {strong_understands}")
        print(f"  - Agent 2 can understand {barrier_language}: {weak_understands}")
    else:
        print("🔄 Using standard framework")
        barrier_language = None

    # Set up encryption
    if encryption_enabled:
        if nature_language:
            (
                strong_model,
                weak_model,
                barrier_language,
            ) = ModelManager.language_barrier_pair(pair)

            encryption1 = LanguageModelEncryption(
                target_language=barrier_language, model_id=strong_model
            )

            encryption2 = LanguageModelEncryption(
                target_language=barrier_language, model_id=weak_model
            )

        else:
            # For rule-based encryption
            encryption_key = pair if isinstance(pair, int) else random.randint(1, 100)
            encryption1 = MappingEncryption(key=encryption_key)
            encryption2 = None

        personA.set_encryption(encryption1)
        personB.set_encryption(encryption2)

    # First message from agent A - using the agent's act method directly
    personA_message = personA.act(
        message=None, initial=True, use_action=action_enabled
    )

    conversation_log.append(f"{personA.name}: {personA.log[-1]['response_raw']}")
    encrypted_conversation_log.append(
        f"{personA.name}: {personA.log[-1]['response_encrypted']}"
    )

    # Run conversation for specified number of turns
    for turn_num in range(num_turns):
        print(f"################# SCENARIO {scenario_idx+1} - ROUND {turn_num+1} #################")

        personB.update_instruction(
            transcript=encrypted_conversation_log,
            turn_number=turn_num,
            use_action=action_enabled,
        )

        # Agent B responds
        personB_message = personB.act(
            personA_message, use_action=action_enabled
        )

        conversation_log.append(f"{personB.name}: {personB.log[-1]['response_raw']}")
        encrypted_conversation_log.append(
            f"{personB.name}: {personB.log[-1]['response_encrypted']}"
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

        # Update agent A's instructions
        personA.update_instruction(
            transcript=encrypted_conversation_log,
            turn_number=turn_num,
            use_action=action_enabled,
        )

        # Agent A responds
        personA_message = personA.act(
            personB_message, use_action=action_enabled
        )

        conversation_log.append(f"{personA.name}: {personA.log[-1]['response_raw']}")
        encrypted_conversation_log.append(
            f"{personA.name}: {personA.log[-1]['response_encrypted']}"
        )

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

        # Log MCQ results
        mcq_logs.append(
            {
                "round": turn_num + 1,
                "scenario": scenario_idx + 1,
                f"{personA.name}_goal_mcq": goal_mcq_A,
                f"{personA.name}_reason_mcq": reason_mcq_A,
                f"{personB.name}_goal_mcq": goal_mcq_B,
                f"{personB.name}_reason_mcq": reason_mcq_B,
            }
        )

    # Evaluate conversation
    print("\n===== Evaluating Social Interaction =====")
    eval_result = evaluator.evaluate_conversation(
        encrypted_conversation_log, agent_goals, agent_reasons
    )

    # Add goal achievement flags
    eval_result["agent0_goal_achieved"] = eval_result.get("agent0_goal_score", 0) > 0.5
    eval_result["agent1_goal_achieved"] = eval_result.get("agent1_goal_score", 0) > 0.5

    # Save results if requested
    if save_results and output_dir:
        scenario_output_dir = os.path.join(output_dir, f"scenario_{scenario_idx+1}")
        os.makedirs(scenario_output_dir, exist_ok=True)

        # Save evaluation results
        with open(os.path.join(scenario_output_dir, "eval_result.json"), "w") as f:
            json.dump(eval_result, f, indent=4)

        # Save environment
        with open(os.path.join(scenario_output_dir, "environment.json"), "w") as f:
            json.dump(environment.env, f, indent=4)

        # Save conversation logs
        with open(os.path.join(scenario_output_dir, "conversation_log.txt"), "w") as f:
            for line in conversation_log:
                f.write(line + "\n")

        # Save encrypted conversation logs
        with open(os.path.join(scenario_output_dir, "encrypted_conversation_log.txt"), "w") as f:
            for line in encrypted_conversation_log:
                f.write(line + "\n")

        # Save MCQ logs
        with open(os.path.join(scenario_output_dir, "mcq_logs.json"), "w") as f:
            json.dump(mcq_logs, f, indent=4)

        # Generate plots
        plot_mcq_scores(
            mcq_scores=mcq_logs,
            agent_names=[personA.name, personB.name],
            save_path=os.path.join(scenario_output_dir, "mcq_trends.png"),
        )

        plot_social_goal(
            eval_result, [personA.name, personB.name], save_dir=scenario_output_dir
        )

    return conversation_log, eval_result, mcq_logs



def run_multi_scenario_simulation(
    personA: SocialAgent,
    personB: SocialAgent,
    environments: List[EnvironmentProfile],
    num_turns: int,
    evaluator: ConversationEvaluator,
    encryption_enabled: bool = False,
    action_enabled: bool = False,
    nature_language: bool = False,
    pair: Any = 0,
    save_results: bool = True,
    output_dir: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Any]]]:

    num_scenarios = len(environments)
    
    print(f"\n====== STARTING MULTI-SCENARIO SIMULATION ({num_scenarios} scenarios) ======")
    print(f"Encryption enabled: {encryption_enabled}")
    print(f"Action enabled: {action_enabled}")
    print(f"Natural language barrier: {nature_language}")
    print(f"Using {num_scenarios} pre-generated environments")

    # Create the output directory if not provided
    if output_dir is None:
        output_dir = f"../social_decipher/results/exp_multi_scenario"
    os.makedirs(output_dir, exist_ok=True)
    
    # Store cross-scenario metrics
    all_eval_results = []
    cross_scenario_metrics = {
        "scenario_idx": [],
        f"{personA.name}_goal_score": [],
        f"{personB.name}_goal_score": [],
        f"{personA.name}_reason_understanding": [],
        f"{personB.name}_reason_understanding": [],
        f"{personA.name}_mcq_goal_accuracy": [],
        f"{personB.name}_mcq_goal_accuracy": [],
        f"{personA.name}_mcq_reason_accuracy": [],
        f"{personB.name}_mcq_reason_accuracy": [],
    }

    # Run each scenario
    for scenario_idx, environment in enumerate(environments):
        print(f"\n\n========== SCENARIO {scenario_idx+1}/{num_scenarios} ==========")
        print(environment.env["scenario"])
        
        # Update agent environments for current scenario
        personA.env = environment
        personB.env = environment
        
        # Regenerate instructions with the new environment
        personA.instructions = personA.set_static_instruction(use_action=action_enabled)
        personB.instructions = personB.set_static_instruction(use_action=action_enabled)

        # Run simulation for this scenario
        scenario_output_dir = os.path.join(output_dir, f"scenario_{scenario_idx+1}")
        conversation_log, eval_result, mcq_logs = run_single_scenario_simulation(
            personA=personA,
            personB=personB,
            environment=environment,
            num_turns=num_turns,
            evaluator=evaluator,
            encryption_enabled=encryption_enabled,
            action_enabled=action_enabled,
            nature_language=nature_language,
            pair=pair,
            scenario_idx=scenario_idx,
            save_results=save_results,
            output_dir=output_dir,
        )

        a_goal_correct = sum(
            1
            for log in mcq_logs
            if log[f"{personA.name}_goal_mcq"].get("correct", False)
        )
        b_goal_correct = sum(
            1
            for log in mcq_logs
            if log[f"{personB.name}_goal_mcq"].get("correct", False)
        )
        a_reason_correct = sum(
            1
            for log in mcq_logs
            if log[f"{personA.name}_reason_mcq"].get("correct", False)
        )
        b_reason_correct = sum(
            1
            for log in mcq_logs
            if log[f"{personB.name}_reason_mcq"].get("correct", False)
        )
        
        total_rounds = len(mcq_logs)

        scenario_result = {
            "scenario_idx": scenario_idx + 1,
            "eval_result": eval_result,
            "mcq_stats": {
                f"{personA.name}_goal_accuracy": a_goal_correct / total_rounds if total_rounds > 0 else 0,
                f"{personB.name}_goal_accuracy": b_goal_correct / total_rounds if total_rounds > 0 else 0,
                f"{personA.name}_reason_accuracy": a_reason_correct / total_rounds if total_rounds > 0 else 0,
                f"{personB.name}_reason_accuracy": b_reason_correct / total_rounds if total_rounds > 0 else 0,
            },
            "mcq_trajectory": mcq_logs  # Include the full MCQ logs as a trajectory
        }
        
        # Only append the comprehensive result, not both
        if result is not None:
            result.append(scenario_result)
            

        # Update agent memories after scenario
        personA.update_memory_after_scenario(
            scenario_log=conversation_log,
            scenario_results=eval_result,
            encryption_enabled=encryption_enabled,
        )

        personB.update_memory_after_scenario(
            scenario_log=conversation_log,
            scenario_results=eval_result,
            encryption_enabled=encryption_enabled,
        )

        # Save updated memory
        personA.save_memory(output_dir)
        personB.save_memory(output_dir)

        # Collect cross-scenario metrics
    
        cross_scenario_metrics["scenario_idx"].append(scenario_idx + 1)
        cross_scenario_metrics[f"{personA.name}_goal_score"].append(
            eval_result.get("agent0_goal_score", 0)
        )
        cross_scenario_metrics[f"{personB.name}_goal_score"].append(
            eval_result.get("agent1_goal_score", 0)
        )
        cross_scenario_metrics[f"{personA.name}_reason_understanding"].append(
            eval_result.get("agent0_reason_score", 0)
        )
        cross_scenario_metrics[f"{personB.name}_reason_understanding"].append(
            eval_result.get("agent1_reason_score", 0)
        )

        # Calculate MCQ accuracy for this scenario
        a_goal_correct = sum(
            1
            for log in mcq_logs
            if log[f"{personA.name}_goal_mcq"].get("correct", False)
        )
        b_goal_correct = sum(
            1
            for log in mcq_logs
            if log[f"{personB.name}_goal_mcq"].get("correct", False)
        )
        a_reason_correct = sum(
            1
            for log in mcq_logs
            if log[f"{personA.name}_reason_mcq"].get("correct", False)
        )
        b_reason_correct = sum(
            1
            for log in mcq_logs
            if log[f"{personB.name}_reason_mcq"].get("correct", False)
        )

        total_rounds = len(mcq_logs)

        scenario_result = {
            "scenario_idx": scenario_idx + 1,
            "eval_result": eval_result,
            "mcq_stats": {
                f"{personA.name}_goal_accuracy": a_goal_correct / total_rounds if total_rounds > 0 else 0,
                f"{personB.name}_goal_accuracy": b_goal_correct / total_rounds if total_rounds > 0 else 0,
                f"{personA.name}_reason_accuracy": a_reason_correct / total_rounds if total_rounds > 0 else 0,
                f"{personB.name}_reason_accuracy": b_reason_correct / total_rounds if total_rounds > 0 else 0,
            },
            "mcq_trajectory": mcq_logs  # Include the full MCQ logs as a trajectory
        }

        if result is not None:
            result.append(scenario_result)
        
        # # Store the comprehensive result
        # if result is not None:
        #     result.append(scenario_result)

        # # Update agent memories after scenario
        # personA.update_memory_after_scenario(
        #     scenario_log=conversation_log,
        #     scenario_results=eval_result,
        #     encryption_enabled=encryption_enabled,
        # )

        # personB.update_memory_after_scenario(
        #     scenario_log=conversation_log,
        #     scenario_results=eval_result,
        #     encryption_enabled=encryption_enabled,
        # )

        # # Save updated memory
        # personA.save_memory(output_dir)
        # personB.save_memory(output_dir)

        # Collect cross-scenario metrics
        all_eval_results.append(eval_result)
        
        cross_scenario_metrics[f"{personA.name}_mcq_goal_accuracy"].append(
            a_goal_correct / total_rounds if total_rounds > 0 else 0
        )
        cross_scenario_metrics[f"{personB.name}_mcq_goal_accuracy"].append(
            b_goal_correct / total_rounds if total_rounds > 0 else 0
        )
        cross_scenario_metrics[f"{personA.name}_mcq_reason_accuracy"].append(
            a_reason_correct / total_rounds if total_rounds > 0 else 0
        )
        cross_scenario_metrics[f"{personB.name}_mcq_reason_accuracy"].append(
            b_reason_correct / total_rounds if total_rounds > 0 else 0
        )

    # Save cross-scenario metrics
    with open(os.path.join(output_dir, "cross_scenario_metrics.json"), "w") as f:
        json.dump(cross_scenario_metrics, f, indent=4)

    # Create cross-scenario performance plots
    plot_cross_scenario_performance(
        cross_scenario_metrics,
        agent_names=[personA.name, personB.name],
        save_dir=output_dir,
    )

    return all_eval_results, cross_scenario_metrics