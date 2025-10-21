import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load config and set API keys from config.yaml
CONFIG_PATH = project_root / "configs" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    _config = yaml.safe_load(f)

# Set API keys from config
os.environ["OPENAI_API_KEY"] = _config.get("AGENT_OPENAI_API_KEY", "")
os.environ["HF_API_TOKEN"] = _config.get("HF_API_TOKEN", "")
os.environ["MISTRAL_API_KEY"] = _config.get("MISTRAL_API_KEY", "")
os.environ["ANTHROPIC_API_KEY"] = _config.get("ANTHROPIC_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")

from social_decipher.training.data_collector import BarrierDataCollector, load_barrier_episode_sets, TrainingConversation
from social_decipher.training.conversation_rater import ConversationRater
from social_decipher.training.policy_updater import SocialPolicyUpdater
from social_decipher.training.scoring_strategy import ScoringManager, get_custom_barrier_focused_config


def _check_conversation_quality(conversation: TrainingConversation, min_goal: float, min_understanding: float, min_confusion: float) -> bool:
    if not conversation or not hasattr(conversation, 'eval_result'):
        return False
    
    eval_result = conversation.eval_result
    if not eval_result:
        return False
    
    # Extract scores from the nested structure
    try:
        aggregated = eval_result.get('aggregated_scores', {})
        
        # Get goal completion from agent_2
        agent_2_data = aggregated.get('agent_2', {})
        if isinstance(agent_2_data, dict):
            goal_score = agent_2_data.get('goal_completion', 0)
        else:
            goal_score = 0
        
        # Fallback to agent_1 if agent_2 goal is not available
        if goal_score == 0:
            agent_1_data = aggregated.get('agent_1', {})
            if isinstance(agent_1_data, dict):
                goal_score = agent_1_data.get('goal_completion', 0)
        
        # Get interaction quality scores from episode_level
        episode_level = aggregated.get('episode_level', {})
        if isinstance(episode_level, dict):
            understanding = episode_level.get('mutual_understanding', 0)
            confusion = episode_level.get('unresolved_confusion', 0)
            
            # Handle nested structure (score dict with 'score' key)
            if isinstance(understanding, dict):
                understanding = understanding.get('score', 0)
            if isinstance(confusion, dict):
                confusion = confusion.get('score', 0)
        else:
            understanding = 0
            confusion = 0
        
        # Check if meets criteria
        meets_criteria = (
            float(goal_score) > min_goal and 
            float(understanding) >= min_understanding and 
            float(confusion) >= min_confusion
        )
        
        if not meets_criteria:
            print(f"   Scores: goal={goal_score}, understanding={understanding}, confusion={confusion}")
        
        return meets_criteria
    except (AttributeError, TypeError, ValueError) as e:
        print(f"⚠️  Error extracting scores: {e}")
        import traceback
        traceback.print_exc()
        return False


def _collect_bc_with_quality_guarantee(
    data_collector: BarrierDataCollector,
    episodes: List,
    args
) -> List[TrainingConversation]:

    bc_filepath = os.path.join(data_collector.output_dir, "bc_data.json")
    all_conversations = []
    
    # Load existing conversations
    existing_episode_ids = set()
    if os.path.exists(bc_filepath):
        try:
            with open(bc_filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                all_conversations = [TrainingConversation(**conv) for conv in existing_data]
                existing_episode_ids = {
                    conv.conversation_id.split('_ep')[1].split('_')[0] 
                    for conv in all_conversations 
                    if '_ep' in conv.conversation_id
                }
                print(f"   Loaded {len(all_conversations)} existing conversations covering {len(existing_episode_ids)} episodes.")
        except Exception as e:
            print(f"⚠️  Could not load existing BC data: {e}")
    
    success_count = 0
    fail_count = 0
    
    for episode_idx, episode_data in enumerate(episodes):
        episode_id = str(episode_idx)
        episode_type = data_collector._get_episode_type(episode_data)
        
        # Skip if we already have a quality conversation for this episode
        if episode_id in existing_episode_ids:
            print(f"✓ Episode {episode_idx + 1}/{len(episodes)} ({episode_type}): Already processed")
            continue
        
        print(f"\n📝 Episode {episode_idx + 1}/{len(episodes)} ({episode_type})")
        
        # Try to get a high-quality conversation
        found_quality = False
        for attempt in range(args.bc_max_retries):
            try:
                print(f"   Attempt {attempt + 1}/{args.bc_max_retries}...", end=" ")
                
                # Run one conversation
                conversation = data_collector._run_expert_conversation(
                    episode_data, episode_idx, attempt, args.max_rounds, episode_type
                )
                
                if conversation and _check_conversation_quality(
                    conversation, 
                    args.bc_quality_goal,
                    args.bc_quality_understanding, 
                    args.bc_quality_confusion
                ):
                    print(f"✅ High quality!")
                    all_conversations.append(conversation)
                    
                    # Save immediately
                    with open(bc_filepath, 'w', encoding='utf-8') as f:
                        json.dump([vars(c) if hasattr(c, '__dict__') else c for c in all_conversations], 
                                f, indent=2, ensure_ascii=False, default=str)
                    
                    found_quality = True
                    success_count += 1
                    break
                else:
                    print(f"❌ Quality check failed")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        if not found_quality:
            print(f"   ⚠️  Could not get quality conversation after {args.bc_max_retries} attempts")
            fail_count += 1
    
    print(f"\n📊 BC Collection Summary:")
    print(f"   ✅ Success: {success_count} episodes")
    print(f"   ⚠️  Failed: {fail_count} episodes")
    print(f"   📦 Total conversations: {len(all_conversations)}")
    
    data_collector.bc_conversations = all_conversations
    return all_conversations


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data from conversation simulations.")
    parser.add_argument("--episodes_file", type=str, default="data/episode_all_neutralized.jsonl", help="Path to base episodes JSONL file.")
    parser.add_argument("--use_barrier_episodes", action="store_true", help="Include barrier-specific episode sets.")
    parser.add_argument("--barrier_types", nargs="+", default=["semantic", "cultural", "emotional"], help="Barrier types to include.")
    parser.add_argument("--episode_limit", type=int, default=10, help="Limit the number of episodes to process for faster runs.")
    parser.add_argument("--output_file", type=str, default="training_data/sft_data.json", help="Path to save the final SFT JSON dataset.")
    parser.add_argument("--expert_model", type=str, default="gpt-4o")
    parser.add_argument("--agent_model", type=str, default="/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--partner_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--evaluator_model", type=str, default="gpt-4o")
    parser.add_argument("--conversations_per_episode", type=int, default=2)
    parser.add_argument("--max_rounds", type=int, default=20)
    parser.add_argument("--load_existing_data", action="store_true", help="Load existing BC data instead of regenerating it.")
    parser.add_argument("--quality_threshold", type=float, default=6.0)
    parser.add_argument("--filter_top_k", type=int, default=5)
    parser.add_argument("--scoring_strategy", type=str, default="custom_barrier_focused")
    parser.add_argument("--data_collection_mode", type=str, default="bc_and_sr", choices=["bc_and_sr", "sr_only", "bc_only"], help="Data collection mode: 'bc_and_sr' for step 0, 'sr_only' for subsequent steps, 'bc_only' for only BC data.")
    parser.add_argument("--barrier_only", action="store_true", help="If set, only use barrier-type episodes.")
    parser.add_argument("--bc_quality_goal", type=float, default=5.0, help="Minimum goal completion for BC data quality check.")
    parser.add_argument("--bc_quality_understanding", type=float, default=3.0, help="Minimum mutual understanding for BC data quality check.")
    parser.add_argument("--bc_quality_confusion", type=float, default=3.0, help="Minimum unresolved confusion for BC data quality check.")
    parser.add_argument("--bc_max_retries", type=int, default=5, help="Maximum retries per episode to get high-quality BC data.")
    
    # New arguments for filtering thresholds
    parser.add_argument("--goal_threshold", type=float, default=7.0, help="Minimum goal completion score.")
    parser.add_argument("--understanding_threshold", type=float, default=5.0, help="Minimum mutual understanding score.")
    parser.add_argument("--confusion_threshold", type=float, default=5.0, help="Minimum unresolved confusion score.")
    
    args = parser.parse_args()

    # 1. Initialize components
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    data_collector = BarrierDataCollector(
        expert_model=args.expert_model,
        agent_model=args.agent_model,
        partner_model=args.partner_model,
        evaluator_model=args.evaluator_model,
        output_dir=output_dir
    )
    rater = ConversationRater()
    policy_updater = SocialPolicyUpdater(output_dir=os.path.join(output_dir, "policy_updates"))
    
    scoring_config = get_custom_barrier_focused_config()
    scoring_config.quality_threshold = args.quality_threshold
    scoring_config.filter_top_k = args.filter_top_k
    scoring_manager = ScoringManager(strategy_name=args.scoring_strategy, config=scoring_config)

    # 2. Load episodes
    print("--- Loading Episodes ---")
    all_episodes = []

    if not args.barrier_only:
        base_episodes = []
        with open(args.episodes_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    base_episodes.append(json.loads(line))

        if args.episode_limit and len(base_episodes) > args.episode_limit:
            print(f"Loaded {len(base_episodes)} base episodes, sampling {args.episode_limit}.")
            base_episodes = base_episodes[:args.episode_limit]
        all_episodes.extend(base_episodes)
    
    if args.use_barrier_episodes:
        barrier_episodes = load_barrier_episode_sets()
        for cat, eps in barrier_episodes.items():
            print(f"Loaded {len(eps)} episodes for barrier type: {cat}")
            if args.episode_limit and len(eps) > args.episode_limit:
                print(f"  -> Sampling {args.episode_limit} episodes for '{cat}'.")
                eps = eps[:args.episode_limit]
            all_episodes.extend(eps)
    
    print(f"Processing a total of {len(all_episodes)} episodes.")

    # 3. Collect data based on the specified mode
    print(f"\n--- Collecting Conversation Data (Mode: {args.data_collection_mode}) ---")
    
    # Always load existing BC data if available, as it's the expert foundation
    bc_convos = []
    bc_filepath = os.path.join(output_dir, "bc_data.json")
    if os.path.exists(bc_filepath):
        print(f"🔄 Loading existing BC data from {bc_filepath}")
        with open(bc_filepath, 'r', encoding='utf-8') as f:
            bc_convos = json.load(f)
        print(f"   Loaded {len(bc_convos)} existing BC conversations.")

    sr_convos = []
    if args.data_collection_mode == "bc_only":
        # BC only mode with quality guarantee
        print("Generating BC data with quality guarantee...")
        new_bc_convos = _collect_bc_with_quality_guarantee(
            data_collector, all_episodes, args
        )
        bc_convos.extend(new_bc_convos)
    
    elif args.data_collection_mode == "bc_and_sr":
        # Step 0: Generate and save BC data, then generate the first round of SR data.
        print("Generating new BC data...")
        new_bc_convos = data_collector.collect_behavior_cloning_data(
            all_episodes, args.conversations_per_episode, args.max_rounds
        )
        # Combine existing and new BC data
        bc_convos.extend(new_bc_convos)
        
        print("Generating initial SR data...")
        sr_convos = data_collector.collect_self_reinforcement_data(
            all_episodes, args.conversations_per_episode, args.max_rounds
        )

    elif args.data_collection_mode == "sr_only":
        # Steps > 0: Only generate new SR data with the updated agent.
        print("Generating new SR data...")
        sr_convos = data_collector.collect_self_reinforcement_data(
            all_episodes, args.conversations_per_episode, args.max_rounds
        )

    collected_convos = bc_convos + sr_convos
    print(f"Processing {len(collected_convos)} total conversations for this step ({len(bc_convos)} BC, {len(sr_convos)} SR).")

    # 4. Rate and filter data
    print("\n--- Rating and Filtering Conversations ---")
    ratings = rater.rate_conversations(collected_convos)
    
    # Pass the thresholds to the filtering manager
    filtering_context = {
        "goal_threshold": args.goal_threshold,
        "understanding_threshold": args.understanding_threshold,
        "confusion_threshold": args.confusion_threshold,
    }
    filtered_convos = scoring_manager.filter_conversations(collected_convos, ratings, context=filtering_context)
    
    top_k_convos = scoring_manager.apply_top_k_filtering(filtered_convos, ratings)
    print(f"Filtered down to {len(top_k_convos)} high-quality conversations.")

    # 5. Prepare and format data for SFT
    print("\n--- Preparing SFT Dataset ---")
    training_examples = policy_updater.prepare_training_data(
        top_k_convos, ratings, min_quality_score=0 # Filtering already done
    )
    sft_data = policy_updater.format_for_sotopia_sft(training_examples)

    # 6. Save data
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, indent=2)
    print(f"\n✅ Successfully prepared SFT data and saved to {args.output_file}")

if __name__ == "__main__":
    main()