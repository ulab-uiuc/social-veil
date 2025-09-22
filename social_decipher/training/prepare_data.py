import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from social_decipher.training.data_collector import BarrierDataCollector, load_barrier_episode_sets
from social_decipher.training.conversation_rater import ConversationRater
from social_decipher.training.policy_updater import SocialPolicyUpdater
from social_decipher.training.scoring_strategy import ScoringManager, get_custom_barrier_focused_config

def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data from conversation simulations.")
    parser.add_argument("--episodes_file", type=str, default="data/episode_sample.jsonl", help="Path to base episodes JSONL file.")
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
    parser.add_argument("--data_collection_mode", type=str, default="bc_and_sr", choices=["bc_and_sr", "sr_only"], help="Data collection mode: 'bc_and_sr' for step 0, 'sr_only' for subsequent steps.")
    parser.add_argument("--barrier_only", action="store_true", help="If set, only use barrier-type episodes.")
    
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
    if args.data_collection_mode == "bc_and_sr":
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