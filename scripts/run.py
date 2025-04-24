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

os.environ[
    "OPENAI_API_KEY"
] = "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"

os.environ[
    "HF_API_TOKEN"
] = "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"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run social agent simulation")
    parser.add_argument(
        "--encryption_enabled", action="store_true", help="Enable encryption between agents"
    )
    parser.add_argument(
        "--nature_language", action="store_true", help="Use natural language barriers instead of encryption"
    )
    parser.add_argument(
        "--action", action="store_true", help="Enable action-based communication"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="Model to use when not using language barriers"
    )
    parser.add_argument(
        "--max_round", type=int, default=10, help="Max conversation rounds"
    )
    parser.add_argument(
        "--pair", type=str, default="0", 
        help="Language barrier pair to use (index number or named pair like 'gpt-claude-chinese')"
    )
    parser.add_argument(
        "--list_pairs", action="store_true", help="List available language barrier pairs and exit"
    )
    parser.add_argument(
        "--run_all", action="store_true", help="Run all experiment conditions with the selected model pair"
    )
    return parser.parse_args()
 

 
load_dotenv()

def setup_model_clients():
    """Set up model clients based on environment configuration"""
    if os.environ.get("USE_CLAUDE", "false").lower() == "true" or os.environ.get("USE_ASTRA_PROXY", "false").lower() == "true":
        # Set up to use Astra proxy which supports mixing OpenAI and Anthropic models
        os.environ["USE_ASTRA_PROXY"] = "true"
        try:
            # Get patched OpenAI client
            client = ModelManager.get_openai_client()
            
            set_openai_client(client)
            print("✅ Configured agency-swarm to use multiple model providers")
        except ImportError:
            print("astra_assistants package not found. Install with: pip install astra-assistants-api")
    else:
        print("Using direct API connections (no proxy)")

def main():
    args = parse_args()

    if args.list_pairs:
        ModelManager.list_available_pairs()
        return
    
    setup_model_clients()
    client = OpenAI()

    model = args.model
    max_round = args.max_round

    # Generate environment
    generator = EnvironmentGenerator(client)
    environment = generator.generate_environments(num_scenarios=1)[0]
    print(environment.env)

    if args.encryption_enabled and args.nature_language:
        model1, model2, barrier_language = ModelManager.language_barrier_pair(args.pair)
        print(f"Using language barrier pair:")
        print(f"  - Model 1: {model1} (understands {barrier_language})")
        print(f"  - Model 2: {model2} (does NOT understand {barrier_language})")
        print(f"  - Barrier language: {barrier_language}")
        
        # Check environment variables for API keys
        if ModelManager.MODEL_PROVIDERS.get(model1, {}).get("provider") == "anthropic" or \
           ModelManager.MODEL_PROVIDERS.get(model2, {}).get("provider") == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                print("Warning: ANTHROPIC_API_KEY not set. Required for Claude models.")
                
        if ModelManager.MODEL_PROVIDERS.get(model1, {}).get("provider") == "huggingface" or \
           ModelManager.MODEL_PROVIDERS.get(model2, {}).get("provider") == "huggingface":
            if not os.environ.get("HF_API_TOKEN"):
                print("Warning: HF_API_TOKEN not set. Required for Hugging Face models.")
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

    if args.run_all:
        # Run all experiment conditions
        
        # EXPERIMENT 1: No Encryption, No Action
        print("\n🚀 Running Experiment 1: No Encryption, No Action")
        agent1 = SocialAgent("Alex", profile_a, profile_b, environment, 0, use_action=False)
        agent2 = SocialAgent("Jamie", profile_b, profile_a, environment, 1, use_action=False)
        simulate_conversation(agent1, agent2, max_round, agent_goals, agent_reasons,
                             agent_goals_mcqas, agent_reasons_mcqas, evaluator,
                             encryption_enabled=False, action_enabled=False, nature_language=False,
                             output_suffix="no_encryption_no_action")

        # EXPERIMENT 2: Encryption Only (Mapping)
        print("\n🔐 Running Experiment 2: With Encryption (Mapping), No Action")
        agent1 = SocialAgent("Alex", profile_a, profile_b, environment, 0, use_action=False)
        agent2 = SocialAgent("Jamie", profile_b, profile_a, environment, 1, use_action=False)
        simulate_conversation(agent1, agent2, max_round, agent_goals, agent_reasons,
                             agent_goals_mcqas, agent_reasons_mcqas, evaluator,
                             encryption_enabled=True, action_enabled=False, nature_language=False,
                             output_suffix="mapping_encryption_no_action")
        
        # EXPERIMENT 3: Encryption + Action (Mapping)
        print("\n🔐 Running Experiment 2: With Encryption (Mapping), No Action")
        agent1 = SocialAgent("Alex", profile_a, profile_b, environment, 0, use_action=False)
        agent2 = SocialAgent("Jamie", profile_b, profile_a, environment, 1, use_action=False)
        simulate_conversation(agent1, agent2, max_round, agent_goals, agent_reasons,
                             agent_goals_mcqas, agent_reasons_mcqas, evaluator,
                             encryption_enabled=True, action_enabled=True, nature_language=False,
                             output_suffix="mapping_encryption_no_action")

        # EXPERIMENT 4: Natural Language Barrier
        print("\n🌐 Running Experiment 3: With Natural Language Barrier, No Action")
        agent1 = SocialAgent("Alex", profile_a, profile_b, environment, 0, use_action=False)
        agent2 = SocialAgent("Jamie", profile_b, profile_a, environment, 1, use_action=False)
        simulate_conversation(agent1, agent2, max_round, agent_goals, agent_reasons,
                             agent_goals_mcqas, agent_reasons_mcqas, evaluator,
                             encryption_enabled=True, action_enabled=False, nature_language=True,
                             output_suffix="language_barrier_no_action")

        # EXPERIMENT 5: Natural Language Barrier + Action
        print("\n🎭 Running Experiment 4: With Natural Language Barrier + Action")
        agent1 = SocialAgent("Alex", profile_a, profile_b, environment, 0, use_action=True)
        agent2 = SocialAgent("Jamie", profile_b, profile_a, environment, 1, use_action=True)
        simulate_conversation(agent1, agent2, max_round, agent_goals, agent_reasons,
                             agent_goals_mcqas, agent_reasons_mcqas, evaluator,
                             encryption_enabled=True, action_enabled=True, nature_language=True,
                             output_suffix="language_barrier_action")
    else:
        # Run a single experiment based on command line arguments
        # Determine experiment suffix
        suffix = ""
        if args.encryption_enabled:
            if args.nature_language:
                suffix += "language_barrier"
            else:
                suffix += "mapping_encryption"
        else:
            suffix += "no_encryption"
            
        if args.action:
            suffix += "_action"
        else:
            suffix += "_no_action"
        
        # Add pair info if using language barrier
        if args.encryption_enabled and args.nature_language:
            suffix += f"_pair_{args.pair}"
        
        # Create agents
        agent1 = SocialAgent("Alex", profile_a, profile_b, environment, 0, use_action=args.action)
        agent2 = SocialAgent("Jamie", profile_b, profile_a, environment, 1, use_action=args.action)
        
        # Run simulation
        print(f"\n🚀 Running experiment with settings: {suffix}")
        simulate_conversation(agent1, agent2, max_round, agent_goals, agent_reasons,
                             agent_goals_mcqas, agent_reasons_mcqas, evaluator,
                             encryption_enabled=args.encryption_enabled, action_enabled=args.action, 
                             nature_language=args.nature_language,
                             output_suffix=suffix)


if __name__ == "__main__":
    main()