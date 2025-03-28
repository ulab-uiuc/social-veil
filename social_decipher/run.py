import json
from agent.social_agent import SocialAgent
from agency_swarm import Agent, set_openai_key, BaseTool, Agency
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
def main():
    #environment = SocialTaskEnvironment()
    speaker = SocialAgent(name="Speaker", description="A talkative agent who initiates and drives the conversation with expressiveness. You want to talk about sports. ")
    listener = SocialAgent(name="Listener", description="A receptive agent that listens, analyzes emotional cues, and provides supportive feedback. You don't want to talk about sports.  Begin your response with 'Wow Wow.'")
    simulate_conversation(speaker, listener, 3)

if __name__ == "__main__":
    main()