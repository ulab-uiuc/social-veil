#!/usr/bin/env python3
"""
Load existing episodes from barrier-specific files
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

def load_all_episodes(
    baseline_file: str = "data/episode_all.jsonl", 
    max_episodes: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load episodes from all barrier files"""
    
    project_root = Path(__file__).parent.parent
    episodes = []
    
    # File mappings
    file_mappings = {
        "baseline": baseline_file,
        "semantic_structure": "data/episodes_all_semantic.json", 
        "cultural_style": "data/episodes_all_cultural.json",
        "emotional_influence": "data/episodes_all_emotional.json"
    }
    
    for barrier_type, file_path in file_mappings.items():
        full_path = project_root / file_path
        
        if not full_path.exists():
            print(f"⚠️ File not found: {full_path}")
            continue
            
        try:
            if file_path.endswith('.jsonl'):
                # JSONL format
                with open(full_path, 'r') as f:
                    for line in f:
                        episode = json.loads(line.strip())
                        if barrier_type == "baseline":
                            episode["barrier_type"] = "baseline"
                        episodes.append(episode)
            else:
                # JSON array format
                with open(full_path, 'r') as f:
                    file_episodes = json.load(f)
                    if isinstance(file_episodes, list):
                        episodes.extend(file_episodes)
                    else:
                        episodes.append(file_episodes)
                        
            print(f"📂 Loaded {barrier_type} episodes from {file_path}")
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    
    if max_episodes and len(episodes) > max_episodes:
        episodes = episodes[:max_episodes]
        print(f"🔄 Limited to {max_episodes} episodes")
    
    print(f"✅ Total episodes loaded: {len(episodes)}")
    return episodes