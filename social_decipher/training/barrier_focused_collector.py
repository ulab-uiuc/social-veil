"""
Barrier-Focused Data Collection
Specialized for barrier adaptation research - maintains scientific control
while enabling interactive learning.
"""

import json
from typing import Dict, List, Any, Optional
from .data_collector import BarrierDataCollector, TrainingConversation


class BarrierFocusedCollector(BarrierDataCollector):
    """
    Specialized collector for barrier adaptation research.
    
    Key differences from Sotopia-π:
    1. Uses controlled episode sets (no new scenario generation)
    2. Focuses on barrier type variations
    3. Maintains scientific rigor for research
    """
    
    def collect_systematic_barrier_data(
        self,
        base_episodes: List[Dict[str, Any]],
        barrier_episodes: Dict[str, List[Dict[str, Any]]],  # {"semantic": [...], "cultural": [...], "emotional": [...]}
        conversations_per_episode: int = 3
    ) -> Dict[str, List[TrainingConversation]]:
        """
        Collect data systematically across barrier types.
        
        This maintains scientific control by using:
        - Same base scenarios
        - Systematic barrier variations
        - Controlled comparison groups
        """
        
        all_data = {
            "baseline": [],
            "semantic": [],
            "cultural": [], 
            "emotional": []
        }
        
        print("🔬 Collecting systematic barrier adaptation data...")
        
        # Baseline: Original episodes without barriers
        print("\n📊 Collecting baseline (no barriers)...")
        baseline_data = self.collect_behavior_cloning_data(
            base_episodes, conversations_per_episode
        )
        all_data["baseline"] = baseline_data
        
        # Barrier types: Controlled variations
        for barrier_type, episodes in barrier_episodes.items():
            print(f"\n🚧 Collecting {barrier_type} barrier data...")
            
            # Expert demonstrations (BC)
            bc_data = self.collect_behavior_cloning_data(
                episodes, conversations_per_episode
            )
            
            # Self-reinforcement (SR)  
            sr_data = self.collect_self_reinforcement_data(
                episodes, conversations_per_episode
            )
            
            all_data[barrier_type] = bc_data + sr_data
            
        return all_data
    
    def analyze_barrier_coverage(
        self, 
        conversations: Dict[str, List[TrainingConversation]]
    ) -> Dict[str, Any]:
        """Analyze coverage of barrier types and scenarios"""
        
        analysis = {
            "total_conversations": sum(len(convs) for convs in conversations.values()),
            "barrier_type_counts": {},
            "scenario_coverage": {},
            "quality_distribution": {}
        }
        
        for barrier_type, convs in conversations.items():
            analysis["barrier_type_counts"][barrier_type] = len(convs)
            
            # Track scenario diversity within each barrier type
            scenarios = set()
            for conv in convs:
                if conv.barrier_info and "scenario" in conv.barrier_info:
                    scenarios.add(conv.barrier_info["scenario"][:50])  # First 50 chars
                    
            analysis["scenario_coverage"][barrier_type] = len(scenarios)
        
        print("\n📈 Barrier Coverage Analysis:")
        print(f"   Total conversations: {analysis['total_conversations']}")
        for barrier_type, count in analysis["barrier_type_counts"].items():
            scenarios = analysis["scenario_coverage"].get(barrier_type, 0)
            print(f"   {barrier_type}: {count} conversations across {scenarios} scenarios")
            
        return analysis


def load_barrier_episode_sets(data_dir: str = "data") -> Dict[str, List[Dict[str, Any]]]:
    """Load pre-generated barrier episode sets"""
    
    import os
    
    episode_sets = {}
    
    # Load different barrier types
    barrier_files = {
        "semantic": "episodes_semantic.json",
        "cultural": "episodes_cultural.json", 
        "emotional": "episodes_emotional.json"
    }
    
    for barrier_type, filename in barrier_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                episode_sets[barrier_type] = json.load(f)
            print(f"📁 Loaded {len(episode_sets[barrier_type])} {barrier_type} episodes")
        else:
            print(f"⚠️ {filepath} not found - run barrier creation first")
            episode_sets[barrier_type] = []
    
    return episode_sets


# Example usage for your research approach
def collect_barrier_research_data(
    base_episodes_file: str = "data/episode_original.jsonl",
    output_dir: str = "training_data/barrier_research"
):
    """
    Collect data for barrier adaptation research.
    
    This approach maintains scientific rigor while enabling learning:
    1. Controlled episode sets
    2. Systematic barrier variations  
    3. Comparable training data
    """
    
    # Load base episodes
    with open(base_episodes_file, 'r') as f:
        base_episodes = [json.loads(line) for line in f if line.strip()]
    
    # Load barrier episode sets (your current augmentation)
    barrier_episodes = load_barrier_episode_sets()
    
    # Collect systematic data
    collector = BarrierFocusedCollector(output_dir=output_dir)
    
    all_data = collector.collect_systematic_barrier_data(
        base_episodes=base_episodes,
        barrier_episodes=barrier_episodes,
        conversations_per_episode=3
    )
    
    # Analyze coverage
    coverage = collector.analyze_barrier_coverage(all_data)
    
    print("\n✅ Barrier research data collection complete!")
    print(f"📊 Collected systematic data for barrier adaptation research")
    print(f"🔬 Maintains scientific control while enabling interactive learning")
    
    return all_data, coverage