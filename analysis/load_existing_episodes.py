#!/usr/bin/env python3
"""
Load Existing Barrier Episodes

Utility script to work with pre-generated barrier episodes stored in separate files.
This is designed for the social-decipher data structure where:
- episode_sample.jsonl contains baseline episodes
- episodes_semantic.json contains semantic barrier episodes 
- episodes_cultural.json contains cultural barrier episodes
- episodes_emotional.json contains emotional barrier episodes
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_baseline_episodes(
    file_path: str = "data/episode_sample.jsonl", 
    max_episodes: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load baseline episodes from JSONL file.
    
    Args:
        file_path: Path to baseline episodes JSONL file
        max_episodes: Maximum number of episodes to load
        
    Returns:
        List of baseline episode dictionaries
    """
    episodes = []
    
    if not Path(file_path).exists():
        print(f"❌ Baseline episodes file not found: {file_path}")
        return episodes
    
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if max_episodes and i >= max_episodes:
                    break
                line = line.strip()
                if line:
                    ep = json.loads(line)
                    ep["barrier_type"] = "baseline"
                    episodes.append(ep)
        
        print(f"✅ Loaded {len(episodes)} baseline episodes")
        
    except Exception as e:
        print(f"❌ Error loading baseline episodes: {e}")
    
    return episodes

def load_barrier_episodes(
    barrier_type: str,
    file_path: str,
    max_episodes: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load barrier episodes from JSON file.
    
    Args:
        barrier_type: Type of barrier ("semantic", "cultural", "emotional")
        file_path: Path to barrier episodes JSON file
        max_episodes: Maximum number of episodes to load
        
    Returns:
        List of barrier episode dictionaries
    """
    episodes = []
    
    if not Path(file_path).exists():
        print(f"❌ {barrier_type} episodes file not found: {file_path}")
        return episodes
    
    try:
        with open(file_path, 'r') as f:
            barrier_episodes = json.load(f)
        
        # Take the episodes we need
        episodes_to_take = len(barrier_episodes)
        if max_episodes:
            episodes_to_take = min(max_episodes, len(barrier_episodes))
        
        for i in range(episodes_to_take):
            ep = barrier_episodes[i].copy()
            # Standardize barrier type naming
            barrier_type_map = {
                "semantic": "semantic_structure",
                "cultural": "cultural_style", 
                "emotional": "emotional_influence"
            }
            ep["barrier_type"] = barrier_type_map.get(barrier_type, barrier_type)
            episodes.append(ep)
        
        print(f"✅ Loaded {len(episodes)} {barrier_type} episodes")
        
    except Exception as e:
        print(f"❌ Error loading {barrier_type} episodes: {e}")
    
    return episodes

def load_all_episodes(
    baseline_file: str = "data/episode_sample.jsonl",
    max_episodes: int = 5
) -> List[Dict[str, Any]]:
    """
    Load all episodes (baseline + barriers) from existing files.
    
    Args:
        baseline_file: Path to baseline episodes file
        max_episodes: Maximum episodes per barrier type
        
    Returns:
        List of all episodes with barrier_type field added
    """
    print(f"📂 Loading all episodes (max {max_episodes} per type)...")
    
    all_episodes = []
    
    # Allow overriding filenames via environment variables
    baseline_file = os.environ.get("BASELINE_EPISODES", baseline_file)
    semantic_path = os.environ.get("SEMANTIC_EPISODES", "data/episodes_semantic.json")
    cultural_path = os.environ.get("CULTURAL_EPISODES", "data/episodes_cultural.json")
    emotional_path = os.environ.get("EMOTIONAL_EPISODES", "data/episodes_emotional.json")

    # Load baseline episodes
    baseline_episodes = load_baseline_episodes(baseline_file, max_episodes)
    all_episodes.extend(baseline_episodes)
    
    # Load barrier episodes
    barrier_files = {
        "semantic": semantic_path,
        "cultural": cultural_path,
        "emotional": emotional_path,
    }
    
    for barrier_type, file_path in barrier_files.items():
        barrier_episodes = load_barrier_episodes(barrier_type, file_path, max_episodes)
        all_episodes.extend(barrier_episodes)
    
    print(f"🎯 Total episodes loaded: {len(all_episodes)}")
    
    # Print breakdown
    breakdown = {}
    for ep in all_episodes:
        barrier_type = ep.get("barrier_type", "unknown")
        breakdown[barrier_type] = breakdown.get(barrier_type, 0) + 1
    
    print("📊 Episode breakdown:")
    for barrier_type, count in breakdown.items():
        print(f"  {barrier_type}: {count}")
    
    return all_episodes

def get_matched_episodes(
    baseline_file: str = "data/episode_sample.jsonl",
    num_episodes: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get matched sets of episodes (same base episode with different barriers).
    
    Args:
        baseline_file: Path to baseline episodes file
        num_episodes: Number of base episodes to use
        
    Returns:
        Dictionary mapping barrier types to episode lists
    """
    print(f"🔗 Loading matched episode sets (first {num_episodes} episodes)...")
    
    # Load all episodes
    all_episodes = load_all_episodes(baseline_file, num_episodes)
    
    # Group by barrier type
    episodes_by_type = {
        "baseline": [],
        "semantic_structure": [],
        "cultural_style": [],
        "emotional_influence": []
    }
    
    for ep in all_episodes:
        barrier_type = ep.get("barrier_type", "baseline")
        if barrier_type in episodes_by_type:
            episodes_by_type[barrier_type].append(ep)
    
    # Ensure we have the same number for each type (take minimum)
    min_count = min(len(episodes) for episodes in episodes_by_type.values() if episodes)
    
    for barrier_type in episodes_by_type:
        episodes_by_type[barrier_type] = episodes_by_type[barrier_type][:min_count]
    
    print(f"✅ Matched {min_count} episodes per barrier type")
    
    return episodes_by_type

def main():
    """Demo usage of episode loading functions"""
    print("🔬 Episode Loading Demo")
    print("=" * 40)
    
    # Test loading all episodes
    all_episodes = load_all_episodes(max_episodes=3)
    
    print("\n" + "=" * 40)
    
    # Test loading matched episodes  
    matched_episodes = get_matched_episodes(num_episodes=3)
    
    # Show sample episode from each type
    print("\n📋 Sample episodes:")
    for barrier_type, episodes in matched_episodes.items():
        if episodes:
            sample_ep = episodes[0]
            episode_id = sample_ep.get("episode_id", "unknown")
            scenario = sample_ep.get("scenario", "")[:100] + "..."
            print(f"\n{barrier_type}:")
            print(f"  ID: {episode_id}")
            print(f"  Scenario: {scenario}")
            
            # Show barrier-specific info
            if "barrier_prompts" in sample_ep:
                barrier_prompt = str(sample_ep["barrier_prompts"]).get("agentA", "")[:100]
                print(f"  Barrier prompt: {barrier_prompt}...")

if __name__ == "__main__":
    main()