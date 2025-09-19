import argparse
import os
import json
import sys
import yaml
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI

from social_decipher.agent.agent_profile import AgentProfile
from social_decipher.agent.social_agent import SocialAgent
from social_decipher.communication import simulate_conversation
from social_decipher.environment.env_profile import EnvironmentProfile
from social_decipher.evaluate import ConversationEvaluator
from social_decipher.utils.model import ModelManager
from social_decipher.utils.utils import load_json

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs/config.yaml")
with open(CONFIG_PATH, "r") as f:
    _config = yaml.safe_load(f)

sotopia_env = _config.get("sotopia_env")
sotopia_hard_env = _config.get("sotopia_hard_env")

os.environ["OPENAI_API_KEY"] = _config.get("AGENT_OPENAI_API_KEY") 
os.environ["HF_API_TOKEN"] = _config.get("HF_API_TOKEN")
os.environ["MISTRAL_API_KEY"] = _config.get("MISTRAL_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = _config.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run social agent simulation: baseline + three barrier variants (semantic/cultural/emotional)")
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
        "--max_rounds", type=int, default=20, help="Max conversation rounds per scenario"
    )
    parser.add_argument(
        "--episode_limit", type=int, default=None, help="Limit number of episodes to process (default: all episodes)",
    )
    parser.add_argument(
        "--list_models", action="store_true", help="List available models for agent configuration and exit",
    )

    parser.add_argument(
        "--episodes_file", type=str, default="data/episode_original.json", 
        help="Path to the pre-processed episode JSONL file",
    )
    parser.add_argument(
        "--results_dir", type=str, default="social_decipher/results", 
        help="Base directory for experiment results",
    )
    parser.add_argument(
        "--resume", action="store_true", 
        help="Resume an unfinished run by skipping scenarios that already have results in --results_dir",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of scenarios to run in parallel (>=1). For local GPU servers via vLLM, 4-16 is typical."
    )
    parser.add_argument(
        "--disable-mcq", action="store_true",
        help="Disable MCQ tests during the simulation."
    )
    parser.add_argument(
        "--partner-repair-prompt", action="store_true",
        help="If set, Agent B (the partner) will receive a special prompt with communication repair guidance."
    )

    return parser.parse_args()

def load_episode_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def load_episodes(path):
    """Load episodes from either JSONL (one JSON per line) or JSON array file."""
    if path.lower().endswith('.jsonl'):
        return load_episode_jsonl(path)
    # JSON array fallback
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError(f"Unsupported JSON structure in {path}; expected a list of episodes.")

def build_profile_from_episode_data(episode_data, agent_idx, model_id, scenario_type=None):
    agent_profile_data = episode_data["agent_profiles"][agent_idx]
    
    agent_profile_data = agent_profile_data.copy()
    agent_profile_data["private_knowledge"] = ""
    
    return AgentProfile.from_dict(agent_profile_data, model_id)

def build_profiles_and_env(episode_data, model_id, model_a=None, model_b=None, scenario_type=None):
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

def create_environment_from_episode(episode_data, scenario_type=None):
    env = EnvironmentProfile(
        scenario=episode_data["scenario"],
        agent_goals=episode_data["agent_goals"],
        agent_reasons=[episode_data.get("agent1_reason", ""), episode_data.get("agent2_reason", "")],
        agent_goals_mcqas=episode_data.get("agent_goals_mcqas", []),
        agent_reasons_mcqas=episode_data.get("agent_reasons_mcqas", []),
        agent_knowledge_mcqas=episode_data.get("agent_knowledge_mcqas", []),
        agent_relationship=episode_data.get("agent_relationship", "friend"),
        agent1_private_knowledge=episode_data.get("agent1_private_knowledge", "") if scenario_type == "knowledge_barrier" else "",
        agent2_private_knowledge=episode_data.get("agent2_private_knowledge", "") if scenario_type == "knowledge_barrier" else "",
        agent1_profile=episode_data.get("agent1_profile"),
        agent2_profile=episode_data.get("agent2_profile"),
    )
    # Attach barrier prompts from augmented episodes so communication layer can inject them
    barrier_prompts = episode_data.get("barrier_prompts")
    if isinstance(barrier_prompts, dict):
        env.env["barrier_prompts"] = barrier_prompts
    # Attach barrier cues for runtime conditioning (scene addendum, profile notes, opening seed)
    barrier_cues = episode_data.get("barrier_cues")
    if isinstance(barrier_cues, dict):
        env.env["barrier_cues"] = barrier_cues
    # Preserve barrier type metadata if present (not used at runtime but useful for logs)
    if "barrier_type" in episode_data:
        env.env["barrier_type"] = episode_data["barrier_type"]
    return env

def create_agents(profile_a, profile_b, env, agent1_name, agent2_name, use_repair_prompt_for_b: bool = False):    
    agent1 = SocialAgent(agent1_name, profile_a, profile_b, env, 0)
    agent2 = SocialAgent(agent2_name, profile_b, profile_a, env, 1, use_repair_prompt=use_repair_prompt_for_b)
    return agent1, agent2

def get_experiment_config(results_dir):
    tag_parts = []
    tag = "_".join(tag_parts)
    
    return {
        "tag": tag,
        "results_dir": results_dir,
    }

def run_experiment(episodes, experiment_config, evaluator, args, mode_tag: str):
    results_dir = os.path.join(experiment_config["results_dir"], f"mode_{mode_tag}")
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
    
    print(f"\n🧪 Running experiment: {experiment_config['tag']}")
    print(f"   Mode: {mode_tag}")
    print(f"   Results: {results_dir}")
    
    eval_results, mcq_logs = [], []
    completed = _get_completed_scenarios(results_dir) if getattr(args, "resume", False) else set()

    if completed:
        print(f"   Resume enabled: detected {len(completed)} completed scenario(s) in {results_dir} → will skip them")
    
    def _run_one(scenario_idx: int, episode_data: dict):
        scenario_num = scenario_idx + 1
        scenario_dir = os.path.join(results_dir, f"scenario_{scenario_num}")
        os.makedirs(scenario_dir, exist_ok=True)

        try:
            print(f"📝 Scenario {scenario_num}/{len(episodes)}")
            # Build everything per task
            profile_a, profile_b, env, agent1_name, agent2_name, agent_reasons = build_profiles_and_env(
                episode_data, args.model, args.model_a, args.model_b, None
            )
            agent1, agent2 = create_agents(
                profile_a, profile_b, env, agent1_name, agent2_name, 
                use_repair_prompt_for_b=args.partner_repair_prompt
            )
            # Create a fresh evaluator per task to avoid client sharing across threads
            local_evaluator = ConversationEvaluator(args.model)
            simulate_conversation(
                personA=agent1,
                personB=agent2,
                evaluator=local_evaluator,
                max_rounds=args.max_rounds,
                scenario_index=scenario_idx,
                pair="0",
                environment=env,
                result=None,
                root_dir=results_dir,
                run_mcq_tests=not args.disable_mcq,
            )
        except Exception as e:
            # Log the exception to a file in the scenario directory
            failure_log_path = os.path.join(scenario_dir, "failure_log.txt")
            with open(failure_log_path, "w") as f:
                f.write(f"Scenario {scenario_num} failed with the following error:\n")
                f.write(str(e))
            print(f"❌ Scenario {scenario_num} failed. Log saved to {failure_log_path}")
            # Re-raise the exception to be caught by the main loop's error handler
            raise

    # Submit tasks
    to_run = [(idx, ep) for idx, ep in enumerate(episodes) if (idx + 1) not in completed]
    if not to_run:
        print("Nothing to do. All scenarios appear completed.")
        return
    workers = max(1, int(getattr(args, "concurrency", 1) or 1))
    if workers == 1:
        for idx, ep in to_run:
            try:
                _run_one(idx, ep)
            except Exception as e:
                print(f"❌ Scenario {idx+1} failed with error: {e}")
    else:
        print(f"🚀 Concurrency: {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one, idx, ep): idx for idx, ep in to_run}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"❌ Scenario {idx+1} failed with error: {e}")

def main():
    args = parse_args()
    
    if args.list_models:
        ModelManager.list_available_models()
        return

    # Show model configuration
    print("🤖 Model Configuration:")
    print(f"   Agent A: {args.model_a}")
    print(f"   Agent B: {args.model_b}")
    print()
    
    episodes = load_episodes(args.episodes_file)
    print(f"Loaded {len(episodes)} episodes from {args.episodes_file}")
    
    # Apply episode limit if specified
    if args.episode_limit:
        episodes = episodes[:args.episode_limit]
        print(f"Using first {len(episodes)} episodes (limited by --episode_limit)")
    
    evaluator = ConversationEvaluator(args.model)

    # Run single experiment based on specified parameters
    experiment_config = get_experiment_config(
        args.results_dir
    )
    
    semantic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "episodes_all_semantic.json"))
    cultural_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "episodes_all_cultural.json"))
    emotional_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "episodes_all_emotional.json"))

    need_generate = not (os.path.isfile(semantic_path) and os.path.isfile(cultural_path) and os.path.isfile(emotional_path))
    if need_generate:
        print("🛠️ Generating augmented barrier episodes (semantic/cultural/emotional)...")
        bc = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "barrier_creation.py"))
        os.system(f"python {bc} --mode augment --input_episodes {args.episodes_file} --out_semantic {semantic_path} --out_cultural {cultural_path} --out_emotional {emotional_path}")

    episodes_semantic = load_json(semantic_path)
    episodes_cultural = load_json(cultural_path)
    episodes_emotional = load_json(emotional_path)

    # Run baseline then each barrier set
    print("\n▶️ Running baseline (original episodes)...")
    run_experiment(episodes, experiment_config, evaluator, args, mode_tag="baseline")

    print("\n▶️ Running semantic barrier episodes...")
    run_experiment(episodes_semantic, experiment_config, evaluator, args, mode_tag="semantic")

    print("\n▶️ Running cultural barrier episodes...")
    run_experiment(episodes_cultural, experiment_config, evaluator, args, mode_tag="cultural")

    print("\n▶️ Running emotional barrier episodes...")
    run_experiment(episodes_emotional, experiment_config, evaluator, args, mode_tag="emotional")

if __name__ == "__main__":
    main()