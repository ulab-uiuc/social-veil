import json
from agent.social_agent import SocialAgent
from agency_swarm import Agent, set_openai_key, BaseTool, Agency
from agent.profile import Agent_Profile, Environment_Profile
#from agent.llm_agent import MultiLingualAgent
#from social_task.social_env import SocialTaskEnvironment
'''
def main():
    environment = SocialTaskEnvironment()
    agent1 = MultiLingualAgent("AgentA", "English", "ConlangA")
    agent2 = MultiLingualAgent("AgentB", "French", "ConlangB")

    swarm.run([agent1, agent2], environment)

    for _ in range(environment.turns):
        print(f"Environment state: {environment.state}")
        msg1 = agent1.act()
        msg2 = agent2.act()
        
        if msg1:
            print(f"{agent1.name} communicates: {json.dumps(msg1)}")
        if msg2:
            print(f"{agent2.name} communicates: {json.dumps(msg2)}")
        
        if environment.check_completion():
            print("Task Completed!")
            break
'''

def simulate_conversation(personA, personB, num_turns):
    agency = Agency([personA, [personA,personB], [personB, personA], [personA,personB], [personB, personA]],  # Define the conversation participants.
                    temperature=0.3,
                    max_prompt_tokens=3000,
    )
    personA_message = agency.get_completion("Hello.",recipient_agent=personA)
    print(personA_message)
    for _ in range(num_turns):
        personB_response = agency.get_completion(personA_message, recipient_agent=personB)
        print(personB_response)
        personA_message = agency.get_completion(personB_response, recipient_agent=personA)
        print(personA_message)

# def main():
#     #environment = SocialTaskEnvironment()
#     speaker = SocialAgent(name="Speaker", description="A talkative agent who initiates and drives the conversation with expressiveness. You want to talk about sports. ")
#     listener = SocialAgent(name="Listener", description="A receptive agent that listens, analyzes emotional cues, and provides supportive feedback. You don't want to talk about sports.  Begin your response with 'Wow Wow.'")
#     simulate_conversation(speaker, listener, 3)

def main():
    # Create agent profiles
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

    # Build agents with profiles and environment
    agent1 = SocialAgent(name=profile_a.profile["first_name"], 
                         profile=profile_a, 
                         env=environment,
                         role_num=0)
    
    agent2 = SocialAgent(name=profile_b.profile["first_name"], 
                         profile=profile_b,
                         env=environment,
                         role_num=1)


    simulate_conversation(agent1, agent2, 3)

if __name__ == "__main__":
    main()
