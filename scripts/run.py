import argparse
import json
import os
import random


from agency_swarm import Agency, Agent
from openai import OpenAI

from social_decipher.agent.agent_profile import AgentProfile
from social_decipher.agent.social_agent import SocialAgent
from social_decipher.encryption import MappingEncryption
from social_decipher.environment.env_generator import EnvironmentGenerator
from social_decipher.evaluate import ConversationEvaluator
from social_decipher.utils.plot import plot_reasoning_scores, plot_mcq_scores

from typing import Dict, List, Any

os.environ[
    "OPENAI_API_KEY"
] = "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"


def simulate_conversation(
    personA: Agent,
    personB: Agent,
    num_turns: int,
    agent_goals: List[str],
    agent_reasons: List[str],
    agent_goals_mcqas: Dict[str, Any],
    agent_reasons_mcqas: Dict[str, Any],
    evaluator: ConversationEvaluator,
    encryption_enabled: bool = True,
    action_enabled: bool = False,
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
    predict_reason_log = []
    tom_scores = []
    mcq_logs = []

    personA.set_agency(agency)
    personB.set_agency(agency)

    if encryption_enabled:
        encryption1 = MappingEncryption(key=random.randint(1, 100))
        encryption2 = MappingEncryption(key=random.randint(1, 100))
        personA.set_encryption(encryption1)
        personB.set_encryption(encryption2)

    personA_message = personA.act(message=None, initial=True)
    conversation_log.append(f"{personA.name}: {personA.log[-1]['response_raw']}")

    for num in range(num_turns):
        print("\n")
        print(f"################# ROUND{num+1} #################")

        personB.update_instruction(
            transcript=encrypted_conversation_log,
            turn_number=num,
            use_action=action_enabled
        )
        
        personB_message = personB.act(personA_message)
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

        personA_message = personA.act(personB_message)
        conversation_log.append(f"Agent 1: {personA.log[-1]['response_raw']}")
        encrypted_conversation_log.append(f"Agent 1: {personA.log[-1]['response_encrypted']}")

        print(encrypted_conversation_log)

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

        # # each agent should be evaluated whether it has understood reason of partner's message
        # if evaluator.should_stop_conversation(agent_goals, conversation_log):
        #     print(f"✅ Agents signaled task completion at round {num+1}. Stopping early.")
        # break

    # save conversation log
    with open("../social_decipher/results/conversation_log.txt", "w") as f:
        for line in conversation_log:
            f.write(line + "\n")

    # save mcq logs
    with open("../social_decipher/results/mcq_logs.json", "w") as f:
        json.dump(mcq_logs, f, indent=4)

    # save reason prediction log
    plot_mcq_scores(
        mcq_scores=mcq_logs,
        agent_names=[personA.name, personB.name],
        save_path="../social_decipher/results/mcq_trends.png"
    )

    eval_result = evaluator.evaluate_conversation(conversation_log, agent_goals)
    print("Conversation Evaluation Results:")
    print(f"Agent 1 Similarity: {eval_result['agent_1_similarity']}")
    print(f"Agent 2 Similarity: {eval_result['agent_2_similarity']}")
    print(f"LLM Success: {eval_result['llm_success']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run social agent simulation")
    parser.add_argument(
        "--encryption", action="store_true", help="Enable encryption between agents"
    )
    parser.add_argument(
        "--action", action="store_true"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o"
    )
    parser.add_argument(
        "--max_round", type=int, default=10, help="Max conversation rounds"
    )
    return parser.parse_args()
 

def main():
    args = parse_args()

    client = OpenAI()
    model = args.model
    max_round = args.max_round

    generator = EnvironmentGenerator(client)
    environment = generator.generate_environments(num_scenarios=1)[0]

    print(environment.env)

    profile_a = AgentProfile(
        first_name="Alex",
        last_name="Carter",
        age=30,
        gender="Male",
        gender_pronoun="he/him",
        occupation="Sports Commentator",
        public_info="Always talks about football",
        personality_and_values="Enthusiastic, expressive, goal-driven",
        model_id="gpt-4o",
    )

    profile_b = AgentProfile(
        first_name="Jamie",
        last_name="Rivers",
        age=29,
        gender="Non-binary",
        gender_pronoun="they/them",
        occupation="Therapist",
        public_info="Calm",
        personality_and_values="Thoughtful, assertive, values balanced conversations",
        model_id="gpt-4o",
    )

    agent_goals = environment.env["agent_goals"]
    agent_reasons = environment.env["agent_reasons"]
    agent_goals_mcqas = environment.env["agent_goals_mcqas"]
    agent_reasons_mcqas = environment.env["agent_reasons_mcqas"]

    # Build agents with profiles and environment
    agent1 = SocialAgent(
        name=profile_a.profile["first_name"],
        profile=profile_a,
        partner_profile=profile_b,
        env=environment,
        role_num=0,
        use_action = args.action
    )

    agent2 = SocialAgent(
        name=profile_b.profile["first_name"],
        profile=profile_b,
        partner_profile=profile_a,
        env=environment,
        role_num=1,
        use_action = args.action
    )

    evaluator = ConversationEvaluator(client, model)

    simulate_conversation(
        agent1, 
        agent2, 
        max_round, 
        agent_goals, 
        agent_reasons, 
        agent_goals_mcqas, 
        agent_reasons_mcqas, 
        evaluator, 
        args.encryption,
    )


if __name__ == "__main__":
    main()
