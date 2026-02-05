"""
Generate composite barrier episodes by merging Emotional + Semantic barriers.
This script reads the individual barrier episode files and creates a new file
with episodes that exhibit both barriers simultaneously.
"""

import argparse
import json
from pathlib import Path


def load_json(path):
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    """Save data as JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def merge_barrier_prompts(emotional_prompt, semantic_prompt):
    """
    Merge emotional and semantic barrier prompts into a composite instruction.
    The merged prompt emphasizes the causal relationship: emotional state affects semantic clarity.
    """
    composite = f"""You are under a COMPOSITE affective-semantic constraint. You must exhibit BOTH emotional dysregulation AND semantic vagueness simultaneously, with these barriers interacting naturally:

EMOTIONAL COMPONENT:
{emotional_prompt}

SEMANTIC COMPONENT:
{semantic_prompt}

CRITICAL INTERACTION RULES:
- Your emotional state should CAUSE or EXACERBATE your semantic vagueness (e.g., frustration leads to rushed, unclear references).
- Your inability to express yourself clearly should HEIGHTEN your emotional distress.
- The two barriers compound: when you feel misunderstood (emotional), you become MORE vague (semantic), which makes you MORE frustrated (emotional).
- Prioritize maintaining BOTH barriers over achieving clarity or emotional regulation."""
    
    return composite


def create_composite_episode(base_episode, emotional_episode, semantic_episode):
    """
    Create a composite barrier episode by merging emotional and semantic barriers.
    
    Args:
        base_episode: The original neutral episode (for reference)
        emotional_episode: Episode with emotional barrier
        semantic_episode: Episode with semantic barrier
    
    Returns:
        New episode dict with composite barrier
    """
    # Start with the emotional episode as base (arbitrary choice)
    composite = emotional_episode.copy()
    
    # Update episode ID and barrier type
    composite["episode_id"] = emotional_episode["episode_id"].replace("_emotional", "_composite_emotional_semantic")
    composite["barrier_type"] = "composite_emotional_semantic"
    
    # Merge the barrier prompts
    emotional_prompt = emotional_episode.get("barrier_prompts", {}).get("agentA", "")
    semantic_prompt = semantic_episode.get("barrier_prompts", {}).get("agentA", "")
    
    if emotional_prompt and semantic_prompt:
        composite["barrier_prompts"] = {
            "agentA": merge_barrier_prompts(emotional_prompt, semantic_prompt)
        }
    
    # Merge barrier cues if they exist (optional, for profile notes/scene addendums)
    emotional_cues = emotional_episode.get("barrier_cues", {})
    semantic_cues = semantic_episode.get("barrier_cues", {})
    
    if emotional_cues or semantic_cues:
        composite["barrier_cues"] = {}
        
        # Merge profile notes
        emo_profile = emotional_cues.get("agentA_profile_note", "")
        sem_profile = semantic_cues.get("agentA_profile_note", "")
        if emo_profile or sem_profile:
            composite["barrier_cues"]["agentA_profile_note"] = f"{emo_profile} {sem_profile}".strip()
        
        # Merge scene addendums
        emo_scene = emotional_cues.get("scene_addendum", "")
        sem_scene = semantic_cues.get("scene_addendum", "")
        if emo_scene or sem_scene:
            composite["barrier_cues"]["scene_addendum"] = f"{emo_scene} {sem_scene}".strip()
        
        # Merge opening seeds (use emotional as primary, since it's often more distinctive)
        if emotional_cues.get("agentA_opening_seed"):
            composite["barrier_cues"]["agentA_opening_seed"] = emotional_cues["agentA_opening_seed"]
    
    # Update agent1_profile to reflect composite barrier
    if "agent1_profile" in composite:
        composite["agent1_profile"] = (
            composite["agent1_profile"] + 
            " Exhibits both emotional instability and semantic vagueness in communication."
        )
    
    return composite


def main():
    parser = argparse.ArgumentParser(description="Generate composite barrier episodes")
    parser.add_argument(
        "--emotional",
        type=str,
        default="data/episodes_all_emotional.json",
        help="Path to emotional barrier episodes"
    )
    parser.add_argument(
        "--semantic",
        type=str,
        default="data/episodes_all_semantic.json",
        help="Path to semantic barrier episodes"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/episodes_all_composite_emotional_semantic.json",
        help="Output path for composite barrier episodes"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of episodes to process (for testing)"
    )
    
    args = parser.parse_args()
    
    print(f"Loading emotional barrier episodes from {args.emotional}...")
    emotional_episodes = load_json(args.emotional)
    
    print(f"Loading semantic barrier episodes from {args.semantic}...")
    semantic_episodes = load_json(args.semantic)
    
    # Ensure we have matching episodes
    assert len(emotional_episodes) == len(semantic_episodes), \
        f"Episode count mismatch: {len(emotional_episodes)} emotional vs {len(semantic_episodes)} semantic"
    
    # Apply limit if specified
    if args.limit:
        emotional_episodes = emotional_episodes[:args.limit]
        semantic_episodes = semantic_episodes[:args.limit]
        print(f"Limited to first {args.limit} episodes")
    
    print(f"Generating {len(emotional_episodes)} composite barrier episodes...")
    
    composite_episodes = []
    for i, (emo_ep, sem_ep) in enumerate(zip(emotional_episodes, semantic_episodes)):
        # Verify they're the same base scenario
        base_emo_id = emo_ep["episode_id"].replace("_emotional", "")
        base_sem_id = sem_ep["episode_id"].replace("_semantic", "")
        
        if base_emo_id != base_sem_id:
            print(f"⚠️  Warning: Episode {i} ID mismatch: {base_emo_id} vs {base_sem_id}")
        
        composite_ep = create_composite_episode(None, emo_ep, sem_ep)
        composite_episodes.append(composite_ep)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(emotional_episodes)} episodes...")
    
    print(f"Saving composite episodes to {args.output}...")
    save_json(composite_episodes, args.output)
    
    print(f"✅ Successfully created {len(composite_episodes)} composite barrier episodes")
    print(f"   Output: {args.output}")


if __name__ == "__main__":
    main()

