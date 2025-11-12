import json
import argparse
from collections import defaultdict
from typing import Optional

# New imports to reconstruct agent and environment
from social_decipher.agent.agent_profile import AgentProfile
from social_decipher.agent.social_agent import SocialAgent
from social_decipher.environment.env_profile import EnvironmentProfile

from social_decipher.training.policy_updater import SocialPolicyUpdater

def filter_and_combine_data(
    bc_data_path: str,
    sr_data_path: Optional[str],
    ratings_path: Optional[str],
    goal_threshold: float,
    understanding_threshold: float,
    confusion_threshold: float,
):
    """
    Loads conversations and ratings, then filters conversations based on specific
    episode-level rating thresholds.
    """
    print("--- Starting Data Filtering based on Rating Thresholds ---")

    # 1. Load all conversations from both files
    all_conversations = []
    bc_conversations = []
    sr_conversations = []
    
    try:
        with open(bc_data_path, 'r', encoding='utf-8') as f:
            bc_conversations = json.load(f)
            all_conversations.extend(bc_conversations)
        print(f"✅ Loaded {len(bc_conversations)} conversations from {bc_data_path}")
    except Exception as e:
        print(f"⚠️ Could not load BC data from {bc_data_path}: {e}")

    if sr_data_path:
        try:
            with open(sr_data_path, 'r', encoding='utf-8') as f:
                sr_conversations = json.load(f)
                all_conversations.extend(sr_conversations)
                print(f"✅ Loaded {len(sr_conversations)} conversations from {sr_data_path}")
        except Exception as e:
            print(f"⚠️ Could not load SR data from {sr_data_path}: {e}")
        
    print(f"Total conversations loaded: {len(all_conversations)}")

    # 2. Load or construct ratings data
    rating_map = {}
    print("Extracting scores from conversation data directly.")
    for conv in all_conversations:
        conv_id = conv.get('conversation_id')
        eval_result = conv.get('eval_result')
        if conv_id and eval_result:
            # Based on the user-provided sample, scores are nested.
            # Construct a rating object that matches the new ConversationRating dataclass.
            agg_scores = eval_result.get('aggregated_scores', {})
            if agg_scores:
                rating_map[conv_id] = {
                    'conversation_id': conv_id,
                    'agent_1': agg_scores.get('agent_1', {}),
                    'agent_2': agg_scores.get('agent_2', {}),
                    'interaction_quality': agg_scores.get('interaction_quality', 0.0),
                    'episode_level': agg_scores.get('episode_level', {})
                }
    print(f"  Extracted ratings for {len(rating_map)} conversations.")

    # 3. Filter conversations based on the specified thresholds
    print("\n--- Applying Episode-Level Rating Filters ---")
    print(f"  - Goal Completion > {goal_threshold}")
    print(f"  - Mutual Understanding >= {understanding_threshold}")
    print(f"  - Unresolved Confusion >= {confusion_threshold}")

    final_conversations = []
    for conv in all_conversations:
        conv_id = conv.get('conversation_id')
        if not conv_id or conv_id not in rating_map:
            continue

        rating = rating_map[conv_id]
        # Adjust path for filtering based on the new consistent rating_map structure
        episode_scores = rating.get('episode_level', {})
        agent2_scores = rating.get('agent_2', {})
        if not episode_scores or not agent2_scores:
            continue

        # Extract scores, with a default of 0.0 if not present
        goal_score = float(agent2_scores.get("goal_completion", 0.0))
        understanding_score = float(episode_scores.get("mutual_understanding", 0.0))
        confusion_score = float(episode_scores.get("unresolved_confusion", 0.0))

        # Apply the filtering logic
        if (goal_score > goal_threshold and
            understanding_score >= understanding_threshold and
            confusion_score >= confusion_threshold):
            final_conversations.append(conv)

    print(f"  Selected {len(final_conversations)} conversations after filtering.")
    return final_conversations, rating_map

def format_for_sft_with_template(conversations: list, rating_map: dict, output_path: str, train_agent_b: bool):
    """
    Converts filtered conversation data into the SFT format, including the full
    prompt template used during inference.
    """
    print("\n--- Formatting data for SFT with full prompt templates ---")
    
    sft_examples = []
    
    for conv_data in conversations:

        env_profile = EnvironmentProfile(
            scenario=conv_data.get("scenario"),
            agent_goals=[
                conv_data.get("agent_a_goal"),
                conv_data.get("agent_b_goal")
            ],
            agent_reasons=[
                conv_data.get("agent_a_reason", ""),
                conv_data.get("agent_b_reason", "")
            ]
        )

        agent_names = []
        for line in conv_data["conversation_log"]:
            if ":" in line:
                name = line.split(":", 1)[0]
                if name not in agent_names:
                    agent_names.append(name)
                if len(agent_names) == 2:
                    break
        
        if len(agent_names) < 2:
            print(f"⚠️  Skipping conversation {conv_data.get('conversation_id')} due to missing agent names.")
            continue

        agent_a_name, agent_b_name = agent_names[0], agent_names[1]

        # Create profiles with placeholder data for missing fields, as bc_data is simplified.
        profile_a = AgentProfile(
            first_name=agent_a_name, 
            last_name="",
            age=30,
            gender="neutral",
            gender_pronoun="they/them",
            occupation="researcher",
            public_info="A participant in a study.",
            pk=f"{agent_a_name}_placeholder_pk",
            model_id=conv_data["agent_a_model"]
        )
        profile_b = AgentProfile(
            first_name=agent_b_name, 
            last_name="",
            age=30,
            gender="neutral",
            gender_pronoun="they/them",
            occupation="researcher",
            public_info="A participant in a study.",
            pk=f"{agent_b_name}_placeholder_pk",
            model_id=conv_data["agent_b_model"]
        )

        agent_a = SocialAgent(name=agent_a_name, profile=profile_a, partner_profile=profile_b, env=env_profile)
        agent_b = SocialAgent(name=agent_b_name, profile=profile_b, partner_profile=profile_a, env=env_profile)
        
        conversation_log = conv_data["conversation_log"]
        
        for i in range(len(conversation_log)):
            current_line = conversation_log[i]
  
            # Determine whose turn it is
            is_agent_b_turn = current_line.startswith(f"{agent_b_name}:")

            # We want to train Agent B, so we only create a training example
            # when it is Agent B's turn to speak.
            if train_agent_b and is_agent_b_turn:
                # The input for the model is the history *before* this turn.
                history = conversation_log[:i]
                
                # Reconstruct the exact prompt Agent B would have seen
                turn_number = (i // 2) + 1
                
                agent_b.update_instruction(transcript=history, turn_number=turn_number)
                full_prompt = agent_b.instructions
                
                # The output is Agent B's response, stripped of the name
                if ":" in current_line:
                    output_text = current_line.split(":", 1)[1].strip()
                else:
                    continue # Should not happen if logic is correct

                sft_examples.append({
                    "input": full_prompt,
                    "output": output_text
                })

    # Save the final dataset
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sft_examples, f, indent=2, ensure_ascii=False)
        print(f"✅ Successfully saved {len(sft_examples)} formatted SFT samples to {output_path}")
    except Exception as e:
        print(f"❌ Error saving final formatted file to {output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and combine conversation data for SFT.")
    parser.add_argument("--bc-data", type=str, required=True, help="Path to the Behavior Cloning data file (bc_data.json).")
    parser.add_argument("--sr-data", type=str, default=None, help="Optional path to the Self-Reinforcement data file (sr_data.json).")
    parser.add_argument("--output", type=str, required=True, help="Path to save the final filtered dataset (e.g., sft_dataset.json).")
    parser.add_argument("--goal-threshold", type=float, default=5.5, help="The minimum goal completion score (exclusive).")
    parser.add_argument("--understanding-threshold", type=float, default=3.0, help="The minimum mutual understanding score (inclusive).")
    parser.add_argument("--confusion-threshold", type=float, default=2.0, help="The minimum unresolved confusion score (inclusive).")
    parser.add_argument("--train-agent-b", action="store_true", default=True, help="Generate training data for Agent B (the partner agent).")
    
    args = parser.parse_args()
    
    final_conversations, rating_map = filter_and_combine_data(
        bc_data_path=args.bc_data,
        sr_data_path=args.sr_data,
        ratings_path=None,  # Ratings are now always extracted from the conversation files
        goal_threshold=args.goal_threshold,
        understanding_threshold=args.understanding_threshold,
        confusion_threshold=args.confusion_threshold,
    )

    if final_conversations:
        format_for_sft_with_template(final_conversations, rating_map, args.output, args.train_agent_b)