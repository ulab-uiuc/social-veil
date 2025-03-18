import json
from openai import swarm
from agent.llm_agent import MultiLingualAgent
from social_task.social_env import SocialTaskEnvironment

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

if __name__ == "__main__":
    main()