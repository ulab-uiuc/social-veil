import json
from typing import List, Dict, Any
from openai import OpenAI
from agent.social_agent import SocialAgent
from agency_swarm import Agent, set_openai_key, BaseTool, Agency
from agent.profile import Agent_Profile, Environment_Profile
from encryption import MappingEncryption
from evaluate import ConversationEvaluator
#from social_task.social_env import SocialTaskEnvironment

def simulate_conversation(personA: Agent, 
                          personB: Agent, 
                          num_turns: int, 
                          agent_goals: List[str],
                          evaluator: ConversationEvaluator) -> None:
    
    agency = Agency([personA, [personA,personB], [personB, personA], [personA,personB], [personB, personA]],  # Define the conversation participants.
                    temperature=0.3,
                    max_prompt_tokens=3000,
    )

    conversation_log = []

    personA.set_agency(agency)
    personB.set_agency(agency)

    encryption = MappingEncryption(key=42)

    personA.set_encryption(encryption)
    personB.set_encryption(encryption)

    personA_message = personA.act(message=None, initial=True)
    conversation_log.append(f"{personA.name}: {personA.log[-1]['response_raw']}")


    for num in range(num_turns):
        print('\n')
        print(f"################# ROUND{num+1} #################")

        personB_response = personB.act(personA_message)
        conversation_log.append(f"{personB.name}: {personB.log[-1]['response_raw']}")

        personA_message = personA.act(personB_response)
        conversation_log.append(f"{personA.name}: {personA.log[-1]['response_raw']}")

        if evaluator.should_stop_conversation(agent_goals, conversation_log):
            print(f"✅ Agents signaled task completion at round {num+1}. Stopping early.")
        break

    eval_result = evaluator.evaluate_conversation(conversation_log, agent_goals)
    print("Conversation Evaluation Results:")
    print(f"Number of turns: {eval_result['num_turns']}")
    print(f"Agent 1 Similarity: {eval_result['agent_1_similarity']}")
    print(f"Agent 2 Similarity: {eval_result['agent_2_similarity']}")
    print(f"LLM Success: {eval_result['llm_success']}")
    
def main():
    # Create agent profiles
    client = OpenAI()
    model = "gpt-4"

    profile_a = Agent_Profile(
        first_name="Alex",
        last_name="Carter",
        age=30,
        gender="Male",
        gender_pronoun="he/him",
        occupation="Sports Commentator",
        public_info="Always talks about football",
        personality_and_values="Enthusiastic, expressive, goal-driven",
        model_id="gpt-4"
    )

    profile_b = Agent_Profile(
        first_name="Jamie",
        last_name="Rivers",
        age=29,
        gender="Non-binary",
        gender_pronoun="they/them",
        occupation="Therapist",
        public_info="Very empathetic and calm",
        personality_and_values="Empathetic, listener, emotionally intelligent",
        model_id="gpt-4"
    )

    # Create environment profile
    environment = Environment_Profile(
        scenario="Two people with different interests meet at a café.",
        agent_goals=[
            "Discuss sports news with your conversation partner.",
            "Avoid talking about sports and steer the conversation toward emotions or personal values."
        ]
    )

    agent_goals = environment.env['agent_goals']

    # Build agents with profiles and environment
    agent1 = SocialAgent(name=profile_a.profile["first_name"], 
                         profile=profile_a, 
                         env=environment,
                         role_num=0)
    
    agent2 = SocialAgent(name=profile_b.profile["first_name"], 
                         profile=profile_b,
                         env=environment,
                         role_num=1)
    
    evaluator = ConversationEvaluator(client, model)


    simulate_conversation(agent1, agent2, 10, agent_goals, evaluator)

if __name__ == "__main__":
    main()
