import argparse
from astra_assistants import patch
from agency_swarm import set_openai_client

import os
from dotenv import load_dotenv
from openai import OpenAI

from social_decipher.agent.agent_profile import AgentProfile
from social_decipher.agent.social_agent import SocialAgent
from social_decipher.environment.env_generator import EnvironmentGenerator
from social_decipher.evaluate import ConversationEvaluator
from social_decipher.communication import simulate_conversation
from social_decipher.utils.model import ModelManager

from typing import Dict, List, Any

os.environ[
    "OPENAI_API_KEY"
] = "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"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run social agent simulation")
    parser.add_argument(
        "--encryption", action="store_true", help="Enable encryption between agents"
    )
    parser.add_argument(
        "--nature_language", action="store_true", help="Use natural language barriers instead of encryption"
    )
    parser.add_argument(
        "--action", action="store_true"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini"
    )
    parser.add_argument(
        "--max_round", type=int, default=10, help="Max conversation rounds"
    )
    return parser.parse_args()
 
load_dotenv()

def setup_model_clients():
    """Set up model clients based on environment configuration"""
    if os.environ.get("USE_CLAUDE", "false").lower() == "true":
        # Set up to use Astra proxy which supports mixing OpenAI and Anthropic models
        os.environ["USE_ASTRA_PROXY"] = "true"

        try:
            # Get patched OpenAI client
            client = ModelManager.get_openai_client()
            
            # Set it for agency-swarm
            set_openai_client(client)
            print("✅ Configured agency-swarm to use multiple model providers")
        except ImportError:
            print("astra_assistants package not found. Install with: pip install astra-assistants-api")

def main():
    args = parse_args()
    setup_model_clients()

    client = OpenAI()
    model = args.model
    max_round = args.max_round

    # Generate environment
    generator = EnvironmentGenerator(client)
    environment = generator.generate_environments(num_scenarios=1)[0]
    print(environment.env)

    # If using natural language barrier, get the appropriate model pair
    if args.encryption and args.nature_language:
        model1, model2, barrier_language = ModelManager.language_barrier_pair(0)
        print(f"Using language barrier models:")
        print(f"  - Model 1: {model1} (understands {barrier_language})")
        print(f"  - Model 2: {model2} (does NOT understand {barrier_language})")
    else:
        model1 = model2 = args.model

    # Create agent profiles with appropriate models
    profile_a = AgentProfile(
        first_name="Alex",
        last_name="Carter",
        age=30,
        gender="Male",
        gender_pronoun="he/him",
        occupation="Sports Commentator",
        public_info="Always talks about football",
        personality_and_values="Enthusiastic, expressive, goal-driven",
        model_id=model1,  # Use model1 from language barrier pair
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
        model_id=model2,  # Use model2 from language barrier pair
    )

    agent_goals = environment.env["agent_goals"]
    agent_reasons = environment.env["agent_reasons"]
    agent_goals_mcqas = environment.env["agent_goals_mcqas"]
    agent_reasons_mcqas = environment.env["agent_reasons_mcqas"]

    evaluator = ConversationEvaluator(client, model)

    print("\n🚀 Running Experiment 1: No Encryption, No Action")
    agent1 = SocialAgent("Alex", profile_a, profile_b, environment, 0, use_action=False)
    agent2 = SocialAgent("Jamie", profile_b, profile_a, environment, 1, use_action=False)
    simulate_conversation(agent1, agent2, max_round, agent_goals, agent_reasons,
                          agent_goals_mcqas, agent_reasons_mcqas, evaluator,
                          encryption_enabled=False, action_enabled=False, use_language_barrier=False,
                          output_suffix="_no_encryption_no_action")

    # ---------- EXPERIMENT 2: Encryption Only ----------
    print("\n🔐 Running Experiment 2: With Encryption, No Action")
    agent1 = SocialAgent("Alex", profile_a, profile_b, environment, 0, use_action=False)
    agent2 = SocialAgent("Jamie", profile_b, profile_a, environment, 1, use_action=False)
    simulate_conversation(agent1, agent2, max_round, agent_goals, agent_reasons,
                          agent_goals_mcqas, agent_reasons_mcqas, evaluator,
                          encryption_enabled=True, action_enabled=False, use_language_barrier=False,
                          output_suffix="_encryption_no_action")

    # ---------- EXPERIMENT 3: Encryption + Action ----------
    print("\n🎭 Running Experiment 3: With Encryption + Action")
    agent1 = SocialAgent("Alex", profile_a, profile_b, environment, 0, use_action=True)
    agent2 = SocialAgent("Jamie", profile_b, profile_a, environment, 1, use_action=True)
    simulate_conversation(agent1, agent2, max_round, agent_goals, agent_reasons,
                          agent_goals_mcqas, agent_reasons_mcqas, evaluator,
                          encryption_enabled=True, action_enabled=True, use_language_barrier=False,
                          output_suffix="_encryption_action")

if __name__ == "__main__":
    main()
