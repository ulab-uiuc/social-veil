import argparse
import os
import json

from openai import OpenAI

from social_decipher.agent.agent_profile import AgentProfile
from social_decipher.agent.social_agent import SocialAgent
from social_decipher.communication import simulate_conversation
from social_decipher.environment.env_generator import EnvironmentGenerator
from social_decipher.evaluate import ConversationEvaluator, calculate_experiment_averages, analyze_mcq_trajectories
from social_decipher.utils.model import ModelManager

os.environ[
    "OPENAI_API_KEY"
] = "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"
os.environ["HF_API_TOKEN"] = "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"
os.environ["MISTRAL_API_KEY"] = "yQ9wm2nsnrlujAjTbBDuRwnjTV2bA1oT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run social agent simulation")
    parser.add_argument(
        "--encryption_enabled",
        action="store_true",
        help="Enable encryption between agents",
    )
    parser.add_argument(
        "--nature_language",
        action="store_true",
        help="Use natural language barriers instead of encryption",
    )
    parser.add_argument(
        "--action", action="store_true", help="Enable action-based communication"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use when not using language barriers",
    )
    parser.add_argument(
        "--max_round", type=int, default=10, help="Max conversation rounds per scenario"
    )
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=1,
        help="Number of scenarios to run in sequence",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default="0",
        help="Language barrier pair to use (index number or named pair like 'gpt-claude-chinese')",
    )
    parser.add_argument(
        "--list_pairs",
        action="store_true",
        help="List available language barrier pairs and exit",
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run all experiment conditions with the selected model pair",
    )
    parser.add_argument(
        "--memory_path",
        type=str,
        default="",
        help="Path to load agent memories from (optional)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    experiment_tag = ['normal', 'construct_enc', 'construct_act', 'nl_enc', 'nl_act']

    result = {
        experiment_tag[0]: [],
        experiment_tag[1]: [],
        experiment_tag[2]: [],
        experiment_tag[3]: [],
        experiment_tag[4]: [],
    }
    
    if args.list_pairs:
        ModelManager.list_available_pairs()
        return
    
    client = OpenAI()
    model = args.model
    max_round = args.max_round
    num_scenarios = args.num_scenarios
    
    # Generate all environments once
    print(f"Generating {num_scenarios} environments...")
    generator = EnvironmentGenerator(client)
    environments = generator.generate_environments(num_scenarios=num_scenarios)
    
    # Create agent profiles
    profile_a = AgentProfile(
        first_name="Alex",
        last_name="Carter",
        age=30,
        gender="Male",
        gender_pronoun="he/him",
        occupation="Sports Commentator",
        public_info="Always talks about football",
        personality_and_values="Enthusiastic, expressive, goal-driven",
        model_id=args.model,
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
        model_id=args.model,
    )
    
    evaluator = ConversationEvaluator(client, model)
    
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

    # Add number of scenarios to output suffix
    if args.num_scenarios > 1:
        suffix += f"_{args.num_scenarios}_scenarios"
    
    if args.run_all:
        # EXPERIMENT 1: No Encryption, No Action
        print("\n🚀 Running Experiment 1: No Encryption, No Action")
        agent1 = SocialAgent(
            "Alex", profile_a, profile_b, environments[0], 0, use_action=False
        )
        agent2 = SocialAgent(
            "Jamie", profile_b, profile_a, environments[0], 1, use_action=False
        )
        
        # Load memory if path provided
        if args.memory_path:
            memory_path_a = os.path.join(args.memory_path, "Alex_memory.json")
            memory_path_b = os.path.join(args.memory_path, "Jamie_memory.json")
            if os.path.exists(memory_path_a):
                print(f"Loading memory for Alex from {memory_path_a}")
                agent1.load_memory(memory_path_a)
            if os.path.exists(memory_path_b):
                print(f"Loading memory for Jamie from {memory_path_b}")
                agent2.load_memory(memory_path_b)
                
        result[experiment_tag[0]], _ = simulate_conversation(
            personA=agent1,
            personB=agent2,
            max_rounds=max_round,
            evaluator=evaluator,
            encryption_enabled=False,
            action_enabled=False,
            nature_language=False,
            output_suffix="no_encryption_no_action",
            num_scenarios=num_scenarios,
            client=client,
            environments=environments,
            result=result[experiment_tag[0]]
        )

        # EXPERIMENT 2: Encryption Only (Mapping)
        print("\n🔐 Running Experiment 2: With Encryption (Mapping), No Action")
        agent1 = SocialAgent(
            "Alex", profile_a, profile_b, environments[0], 0, use_action=False
        )
        agent2 = SocialAgent(
            "Jamie", profile_b, profile_a, environments[0], 1, use_action=False
        )
        result[experiment_tag[1]], _ = simulate_conversation(
            personA=agent1,
            personB=agent2,
            max_rounds=max_round,
            evaluator=evaluator,
            encryption_enabled=True,
            action_enabled=False,
            nature_language=False,
            output_suffix="mapping_encryption_no_action",
            num_scenarios=num_scenarios,
            client=client,
            environments=environments,
            result=result[experiment_tag[1]]
        )

        # EXPERIMENT 3: Encryption + Action (Mapping)
        print("\n🔐🎭 Running Experiment 3: With Encryption (Mapping) + Action")
        agent1 = SocialAgent(
            "Alex", profile_a, profile_b, environments[0], 0, use_action=True
        )
        agent2 = SocialAgent(
            "Jamie", profile_b, profile_a, environments[0], 1, use_action=True
        )
        result[experiment_tag[2]], _ = simulate_conversation(
            personA=agent1,
            personB=agent2,
            max_rounds=max_round,
            evaluator=evaluator,
            encryption_enabled=True,
            action_enabled=True,
            nature_language=False,
            output_suffix="mapping_encryption_action",
            num_scenarios=num_scenarios,
            client=client,
            environments=environments,
            result=result[experiment_tag[2]]
        )

        # EXPERIMENT 4: Natural Language Barrier
        print("\n🌐 Running Experiment 4: With Natural Language Barrier, No Action")
        agent1 = SocialAgent(
            "Alex", profile_a, profile_b, environments[0], 0, use_action=False
        )
        agent2 = SocialAgent(
            "Jamie", profile_b, profile_a, environments[0], 1, use_action=False
        )
        result[experiment_tag[3]], _ = simulate_conversation(
            personA=agent1,
            personB=agent2,
            max_rounds=max_round,
            evaluator=evaluator,
            encryption_enabled=True,
            action_enabled=False,
            nature_language=True,
            output_suffix="language_barrier_no_action",
            pair=args.pair,
            num_scenarios=num_scenarios,
            client=client,
            environments=environments,
            result=result[experiment_tag[3]]
        )

        # EXPERIMENT 5: Natural Language Barrier + Action
        print("\n🌐🎭 Running Experiment 5: With Natural Language Barrier + Action")
        agent1 = SocialAgent(
            "Alex", profile_a, profile_b, environments[0], 0, use_action=True
        )
        agent2 = SocialAgent(
            "Jamie", profile_b, profile_a, environments[0], 1, use_action=True
        )
        result[experiment_tag[4]], _ = simulate_conversation(
            personA=agent1,
            personB=agent2,
            max_rounds=max_round,
            evaluator=evaluator,
            encryption_enabled=True,
            action_enabled=True,
            nature_language=True,
            output_suffix="language_barrier_action",
            pair=args.pair,
            num_scenarios=num_scenarios,
            client=client,
            environments=environments,
            result=result[experiment_tag[4]]
        )

        # After all experiments have run
        results_dir = "../social_decipher/results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save consolidated results
        with open(os.path.join(results_dir, f"consolidated_results_{num_scenarios}_scenarios.json"), "w") as f:
            json.dump(result, f, indent=4)
        
        # Calculate and save average scores
        experiment_averages = calculate_experiment_averages(result, experiment_tag)
        with open(os.path.join(results_dir, f"experiment_averages_{num_scenarios}_scenarios.json"), "w") as f:
            json.dump(experiment_averages, f, indent=4)
        
        # Process MCQ trajectories separately
        print("Analyzing MCQ trajectories...")
        mcq_analysis = analyze_mcq_trajectories(experiment_tag, num_scenarios)
        
        # Create a combined analysis for easier comparison
        combined_analysis = {
            "performance_comparison": {},
            "mcq_comparison": {},
            "understanding_improvement_ranking": []
        }
        
        # Add performance metrics for comparison
        for tag in experiment_tag:
            if tag in experiment_averages and "no_data" not in experiment_averages[tag]:
                combined_analysis["performance_comparison"][tag] = {
                    "goal_completion": (
                        experiment_averages[tag]["agent1_goal_completion"] + 
                        experiment_averages[tag]["agent2_goal_completion"]
                    ) / 2,
                    "believability": (
                        experiment_averages[tag]["agent1_believability"] + 
                        experiment_averages[tag]["agent2_believability"]
                    ) / 2,
                    "overall_score": (
                        experiment_averages[tag]["agent1_overall"] + 
                        experiment_averages[tag]["agent2_overall"]
                    ) / 2,
                    "interaction_quality": experiment_averages[tag]["interaction_quality"]
                }
        
        # Add MCQ metrics for comparison
        for tag in experiment_tag:
            if tag in mcq_analysis:
                combined_analysis["mcq_comparison"][tag] = {
                    "average_accuracy": (
                        mcq_analysis[tag]["average_accuracy"]["Alex_goal"] +
                        mcq_analysis[tag]["average_accuracy"]["Jamie_goal"] +
                        mcq_analysis[tag]["average_accuracy"]["Alex_reason"] +
                        mcq_analysis[tag]["average_accuracy"]["Jamie_reason"]
                    ) / 4,
                    "understanding_improvement": mcq_analysis[tag].get("understanding_improvement", 0)
                }
        
        # Rank experiments by understanding improvement
        combined_analysis["understanding_improvement_ranking"] = sorted(
            [tag for tag in experiment_tag if tag in mcq_analysis and "understanding_improvement" in mcq_analysis[tag]],
            key=lambda x: mcq_analysis[x]["understanding_improvement"],
            reverse=True
        )
        
        # Save combined analysis
        with open(os.path.join(results_dir, f"combined_analysis_{num_scenarios}_scenarios.json"), "w") as f:
            json.dump(combined_analysis, f, indent=4)
        
        print(f"\n✅ All experiments completed and results saved with full trajectory analysis!")

    else:
        # Create agents with first environment
        agent1 = SocialAgent("Alex", profile_a, profile_b, environments[0], 0, use_action=args.action)
        agent2 = SocialAgent("Jamie", profile_b, profile_a, environments[0], 1, use_action=args.action)
        
        # Load memory if path provided
        if args.memory_path:
            memory_path_a = os.path.join(args.memory_path, "Alex_memory.json")
            memory_path_b = os.path.join(args.memory_path, "Jamie_memory.json")
            
            if os.path.exists(memory_path_a):
                print(f"Loading memory for Alex from {memory_path_a}")
                agent1.load_memory(memory_path_a)
            
            if os.path.exists(memory_path_b):
                print(f"Loading memory for Jamie from {memory_path_b}")
                agent2.load_memory(memory_path_b)
                
        simulate_conversation(
            personA=agent1,
            personB=agent2,
            max_rounds=max_round,
            evaluator=evaluator, 
            encryption_enabled=args.encryption_enabled,
            action_enabled=args.action,
            nature_language=args.nature_language,
            output_suffix=suffix,
            pair=args.pair,
            num_scenarios=num_scenarios,
            client=client,
            environments=environments,
            result=result,
        )

if __name__ == "__main__":
    main()
