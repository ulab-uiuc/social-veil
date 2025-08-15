import argparse
import os
import json
import sys
import yaml
import requests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI

from social_decipher.agent.agent_profile import AgentProfile
from social_decipher.agent.social_agent import SocialAgent
from social_decipher.communication import simulate_conversation
from social_decipher.environment.env_profile import EnvironmentProfile
from social_decipher.evaluate import ConversationEvaluator, calculate_experiment_averages
from social_decipher.utils.model import ModelManager
from social_decipher.utils.utils import load_env

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs/config.yaml")
with open(CONFIG_PATH, "r") as f:
    _config = yaml.safe_load(f)

sotopia_env = _config.get("sotopia_env")
sotopia_hard_env = _config.get("sotopia_hard_env")

os.environ["OPENAI_API_KEY"] = _config.get("OPENAI_API_KEY") 
os.environ["HF_API_TOKEN"] = _config.get("HF_API_TOKEN")
os.environ["MISTRAL_API_KEY"] = _config.get("MISTRAL_API_KEY")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run social agent simulation with 3×3×2 factorial design")
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="Model to use for conversation evaluation",
    )
    parser.add_argument(
        "--model_a", type=str, help="Model to use for agent A (overrides --model)",
    )
    parser.add_argument(
        "--model_b", type=str, help="Model to use for agent B (overrides --model). For local models, use 'Qwen/Qwen2.5-7B-Instruct' and set HF_API_TOKEN",
    )
    parser.add_argument(
        "--max_round", type=int, default=20, help="Max conversation rounds per scenario"
    )
    parser.add_argument(
        "--episode_limit", type=int, default=None, help="Limit number of episodes to process (default: all episodes)",
    )
    parser.add_argument(
        "--pair", type=str, default="0", help="Language barrier pair to use (index number or named pair) - fallback if models not specified",
    )
    parser.add_argument(
        "--list_pairs", action="store_true", help="List available language barrier pairs and exit",
    )
    parser.add_argument(
        "--list_models", action="store_true", help="List available models for agent configuration and exit",
    )
    parser.add_argument(
        "--memory_path", type=str, default="", help="Path to load agent memories from (optional)",
    )
    parser.add_argument(
        "--episodes_file", type=str, default="data/episode_sample.jsonl", 
        help="Path to the pre-processed episode JSONL file",
    )

    parser.add_argument(
        "--scenario_type", type=str, choices=["normal", "language_barrier", "knowledge_barrier"], 
        default="normal", help="Social scenario type to test",
    )
    parser.add_argument(
        "--communication_modality", type=str, choices=["text_only", "action_enabled", "text_action_mix"], 
        default="text_only", help="Communication modality to test",
    )
    parser.add_argument(
        "--memory_strategy", type=str, choices=["off", "on"], 
        default="off", help="Memory strategy to test",
    )
    parser.add_argument(
        "--results_dir", type=str, default="social_decipher/results", 
        help="Base directory for experiment results",
    )
    parser.add_argument(
        "--resume", action="store_true", 
        help="Resume an unfinished run by skipping scenarios that already have results in --results_dir",
    )

    return parser.parse_args()


def load_episode_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

def build_profile_from_episode_data(episode_data, agent_idx, model_id, scenario_type):
    agent_profile_data = episode_data["agent_profiles"][agent_idx]
    
    # Handle private knowledge based on scenario type
    if scenario_type == "knowledge_barrier":
        if agent_idx == 0:
            private_knowledge = episode_data.get("agent1_private_knowledge", "")
        else:
            private_knowledge = episode_data.get("agent2_private_knowledge", "")
        # Add private_knowledge to the agent profile data
        agent_profile_data = agent_profile_data.copy()
        agent_profile_data["private_knowledge"] = private_knowledge
    else:
        # No private knowledge for normal and language barrier scenarios
        agent_profile_data = agent_profile_data.copy()
        agent_profile_data["private_knowledge"] = ""
    
    return AgentProfile.from_dict(agent_profile_data, model_id)

def build_profiles_and_env(episode_data, model_id, model_a=None, model_b=None, scenario_type="normal"):
    """Build agent profiles with custom model configuration."""
    # Use custom models if specified, otherwise use the default model
    agent_a_model = model_a if model_a else model_id
    agent_b_model = model_b if model_b else model_id
    
    profile_a = build_profile_from_episode_data(episode_data, 0, agent_a_model, scenario_type)
    profile_b = build_profile_from_episode_data(episode_data, 1, agent_b_model, scenario_type)
    env = create_environment_from_episode(episode_data, scenario_type)

    agent1_name = profile_a.first_name
    agent2_name = profile_b.first_name
    agent_reasons = [episode_data.get("agent1_reason", ""), episode_data.get("agent2_reason", "")]
    
    print(f"🤖 Agent Models:")
    print(f"   {agent1_name}: {agent_a_model}")
    print(f"   {agent2_name}: {agent_b_model}")
    
    return profile_a, profile_b, env, agent1_name, agent2_name, agent_reasons

def create_environment_from_episode(episode_data, scenario_type):
    return EnvironmentProfile(
        scenario=episode_data["scenario"],
        agent_goals=episode_data["agent_goals"],
        agent_reasons=[episode_data.get("agent1_reason", ""), episode_data.get("agent2_reason", "")],
        agent_goals_mcqas=episode_data.get("agent_goals_mcqas", []),
        agent_reasons_mcqas=episode_data.get("agent_reasons_mcqas", []),
        agent_knowledge_mcqas=episode_data.get("agent_knowledge_mcqas", []),
        agent_relationship=episode_data.get("agent_relationship", "friend"),
        agent1_private_knowledge=episode_data.get("agent1_private_knowledge", "") if scenario_type == "knowledge_barrier" else "",
        agent2_private_knowledge=episode_data.get("agent2_private_knowledge", "") if scenario_type == "knowledge_barrier" else ""
    )

def create_agents(profile_a, profile_b, env, agent1_name, agent2_name, communication_modality):
    # Map communication modality to agent parameters
    if communication_modality == "text_only":
        use_action = False
        mix = False
    elif communication_modality == "action_enabled":
        use_action = True
        mix = False
    else:  # text_action_mix
        use_action = True
        mix = True
    
    agent1 = SocialAgent(agent1_name, profile_a, profile_b, env, 0, use_action=use_action, mix=mix)
    agent2 = SocialAgent(agent2_name, profile_b, profile_a, env, 1, use_action=use_action, mix=mix)
    return agent1, agent2

def get_experiment_config(scenario_type, communication_modality, memory_strategy, results_dir):
    tag_parts = []
    tag_parts.append(scenario_type)
    tag_parts.append(communication_modality)
    tag_parts.append(memory_strategy)
    tag = "_".join(tag_parts)
    
    # Map scenario type to encryption/nature_language settings
    if scenario_type == "normal":
        encryption_enabled = False
        nature_language = False
    elif scenario_type == "language_barrier":
        encryption_enabled = True
        nature_language = True
    else:  # knowledge_barrier
        encryption_enabled = False
        nature_language = False
    
    # Map communication modality to action settings
    if communication_modality == "text_only":
        use_action = False
        mix = False
    elif communication_modality == "action_enabled":
        use_action = True
        mix = False
    else:  # text_action_mix
        use_action = True
        mix = True
    
    return {
        "tag": tag,
        "results_dir": results_dir,
        "scenario_type": scenario_type,
        "communication_modality": communication_modality,
        "memory_strategy": memory_strategy,
        "encryption_enabled": encryption_enabled,
        "use_action": use_action,
        "nature_language": nature_language,
        "mix": mix
    }

def run_experiment(episodes, experiment_config, evaluator, args):
    results_dir = experiment_config["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    
    def _get_completed_scenarios(base_dir: str) -> set[int]:
        completed = set()
        try:
            for name in os.listdir(base_dir):
                if not name.startswith("scenario_"):
                    continue
                try:
                    idx = int(name.split("_")[1])
                except Exception:
                    continue
                scenario_dir = os.path.join(base_dir, name)
                eval_path = os.path.join(scenario_dir, "eval_result.json")
                convo_path = os.path.join(scenario_dir, "conversation_log.txt")
                # Consider a scenario completed only if key outputs exist
                if os.path.isfile(eval_path) and os.path.isfile(convo_path):
                    completed.add(idx)
        except FileNotFoundError:
            pass
        return completed
    
    def _load_all_existing_results(base_dir: str):
        eval_results_all, mcq_logs_all = [], []
        try:
            # Aggregate in numeric order if possible
            scenario_dirs = [d for d in os.listdir(base_dir) if d.startswith("scenario_")]
            def _num(d):
                try:
                    return int(d.split("_")[1])
                except Exception:
                    return 10**9
            for name in sorted(scenario_dirs, key=_num):
                scenario_dir = os.path.join(base_dir, name)
                eval_path = os.path.join(scenario_dir, "eval_result.json")
                mcq_path = os.path.join(scenario_dir, "mcq_log.json")
                if os.path.isfile(eval_path):
                    try:
                        with open(eval_path, "r") as f:
                            eval_results_all.append(json.load(f))
                    except Exception:
                        pass
                if os.path.isfile(mcq_path):
                    try:
                        with open(mcq_path, "r") as f:
                            mcq_logs_all.append(json.load(f))
                    except Exception:
                        pass
        except FileNotFoundError:
            pass
        return eval_results_all, mcq_logs_all
    
    print(f"\n🧪 Running experiment: {experiment_config['tag']}")
    print(f"   Scenario Type: {experiment_config['scenario_type']}")
    print(f"   Communication: {experiment_config['communication_modality']}")
    print(f"   Memory: {experiment_config['memory_strategy']}")
    print(f"   Results: {results_dir}")
    
    eval_results, mcq_logs = [], []
    completed = _get_completed_scenarios(results_dir) if getattr(args, "resume", False) else set()
    if completed:
        print(f"   Resume enabled: detected {len(completed)} completed scenario(s) in {results_dir} → will skip them")
    
    for scenario_idx, episode_data in enumerate(episodes):
        scenario_num = scenario_idx + 1
        if scenario_num in completed:
            print(f"⏭️  Skipping scenario {scenario_num} (already completed)")
            continue
        print(f"📝 Scenario {scenario_num}/{len(episodes)}")
        
        profile_a, profile_b, env, agent1_name, agent2_name, agent_reasons = build_profiles_and_env(
            episode_data, args.model, args.model_a, args.model_b, experiment_config["scenario_type"]
        )
        
        agent1, agent2 = create_agents(
            profile_a, profile_b, env, agent1_name, agent2_name, 
            experiment_config["communication_modality"]
        )
        
        conversation_log, eval_result, mcq_log = simulate_conversation(
            personA=agent1,
            personB=agent2,
            evaluator=evaluator,
            max_rounds=args.max_round,
            encryption_enabled=experiment_config["encryption_enabled"],
            action_enabled=experiment_config["use_action"],
            nature_language=experiment_config["nature_language"],
            output_suffix=f"{experiment_config['tag']}_scenario_{scenario_idx+1}",
            scenario_index=scenario_idx,
            pair=args.pair,
            environment=env,
            result=None,
            root_dir=results_dir,
            mix=experiment_config.get("mix", False),
            memory_enabled=(experiment_config["memory_strategy"] == "on")
        )
        
        eval_results.append(eval_result)
        mcq_logs.append(mcq_log)
        
        # Save individual scenario results
        scenario_dir = os.path.join(results_dir, f"scenario_{scenario_num}")
        os.makedirs(scenario_dir, exist_ok=True)
        
        with open(os.path.join(scenario_dir, "conversation_log.txt"), "w") as f:
            f.write("\n".join(conversation_log))
        
        with open(os.path.join(scenario_dir, "eval_result.json"), "w") as f:
            json.dump(eval_result, f, indent=4)
        
        if mcq_log:
            with open(os.path.join(scenario_dir, "mcq_log.json"), "w") as f:
                json.dump(mcq_log, f, indent=4)
    
    # Save aggregated results
    if getattr(args, "resume", False):
        # When resuming, aggregate from disk to include both previous and newly generated results
        all_eval, all_mcq = _load_all_existing_results(results_dir)
        with open(os.path.join(results_dir, f"{experiment_config['tag']}_eval.json"), "w") as f:
            json.dump(all_eval, f, indent=4)
        with open(os.path.join(results_dir, f"{experiment_config['tag']}_mcq.json"), "w") as f:
            json.dump(all_mcq, f, indent=4)
    else:
        with open(os.path.join(results_dir, f"{experiment_config['tag']}_eval.json"), "w") as f:
            json.dump(eval_results, f, indent=4)
        with open(os.path.join(results_dir, f"{experiment_config['tag']}_mcq.json"), "w") as f:
            json.dump(mcq_logs, f, indent=4)
    
    print(f"   ✅ {experiment_config['tag']} completed for all scenarios!")

def main():
    args = parse_args()
    
    if args.list_pairs:
        ModelManager.list_available_pairs()
        return
    
    if args.list_models:
        ModelManager.list_available_models()
        return

    # Show model configuration
    print("🤖 Model Configuration:")
    print(f"   Agent A: {args.model_a}")
    print(f"   Agent B: {args.model_b}")
    print()
    
    episodes = load_episode_jsonl(args.episodes_file)
    print(f"Loaded {len(episodes)} episodes from {args.episodes_file}")
    
    # Apply episode limit if specified
    if args.episode_limit:
        episodes = episodes[:args.episode_limit]
        print(f"Using first {len(episodes)} episodes (limited by --episode_limit)")
    
    evaluator = ConversationEvaluator(args.model)

    # Run single experiment based on specified parameters
    experiment_config = get_experiment_config(
        args.scenario_type, 
        args.communication_modality, 
        args.memory_strategy, 
        args.results_dir
    )
    
    run_experiment(episodes, experiment_config, evaluator, args)

if __name__ == "__main__":
    main()