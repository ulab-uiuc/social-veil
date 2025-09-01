#!/usr/bin/env python3
"""
Run Barrier Analysis

Script to run barrier representation analysis using real prompts from social_task.yaml.
Analyzes how different barrier types affect model internal representations.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.barrier_representation_analysis import BarrierRepresentationAnalyzer
from analysis.utils import check_dependencies

def run_full_analysis(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    episodes_file: str = "data/episode_sample.jsonl",
    severity: float = 0.8
):
    """Run the full barrier analysis"""
    print("🔬 Full Barrier Representation Analysis")
    print("=" * 50)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        return
    
    # Configuration
    config = {
        "model_name": model_name,
        "device": "auto", 
        "episodes_file": episodes_file,
        "severity": severity
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create analyzer
    analyzer = BarrierRepresentationAnalyzer(**config)
    
    # Run analysis
    analyzer.run_full_analysis()
    
    print("\n🎯 Full analysis complete!")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run barrier representation analysis using real social_task.yaml prompts")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name to analyze")
    parser.add_argument("--episodes", type=str, default="data/episode_all.jsonl",
                       help="Baseline episodes file to use (barrier episodes loaded automatically)")
    
    args = parser.parse_args()
    
    run_full_analysis(
        model_name=args.model,
        episodes_file=args.episodes,
    )
    
    print("\n💡 Analysis Tips:")
    print("  1. Check preliminary_internal_states_pca.png for SafeSwitch-style visualization")
    print("  2. Review analysis_summary.md for statistical results")
    print("  3. Look at visualizations to see cluster separation")
    print("  4. If no significant effects, try:")
    print("     - Using more episodes (5-10)")
    print("     - Different model or layers")
    print("\n📂 All prompts use actual templates from configs/social_task.yaml")

if __name__ == "__main__":
    main()