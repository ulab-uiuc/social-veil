#!/usr/bin/env python3
"""
Run Barrier Analysis

Simple script to run barrier representation analysis with sensible defaults.
Uses a smaller number of episodes and focuses on key visualizations.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.barrier_representation_analysis import BarrierRepresentationAnalyzer
from analysis.simple_barrier_test import SimpleBarrierTest
from analysis.utils import check_dependencies

def run_simple_test():
    """Run the simple barrier test"""
    print("🔬 Quick Barrier Representation Test")
    print("=" * 50)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        return
    
    # Run simple test
    tester = SimpleBarrierTest()
    results = tester.run_test()
    
    print("\n🎯 Simple test complete!")
    return results

def run_full_analysis(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    episodes_file: str = "data/episode_sample.jsonl",
    num_episodes: int = 3,
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
        "num_episodes": num_episodes,
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
    parser = argparse.ArgumentParser(description="Run barrier representation analysis")
    parser.add_argument("--mode", choices=["simple", "full"], default="simple",
                       help="Analysis mode: simple test or full analysis")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name to analyze")
    parser.add_argument("--episodes", type=str, default="data/episode_sample.jsonl",
                       help="Episodes file to use")
    parser.add_argument("--num_episodes", type=int, default=3,
                       help="Number of episodes to analyze")
    parser.add_argument("--severity", type=float, default=0.8,
                       help="Barrier severity level")
    
    args = parser.parse_args()
    
    if args.mode == "simple":
        run_simple_test()
    else:
        run_full_analysis(
            model_name=args.model,
            episodes_file=args.episodes,
            num_episodes=args.num_episodes,
            severity=args.severity
        )
    
    print("\n💡 Analysis Tips:")
    print("  1. Check analysis results in results/ directory")
    print("  2. Look at visualizations to see cluster separation")
    print("  3. Examine statistical metrics for significance")
    print("  4. If no significant effects, try:")
    print("     - Increasing severity (0.9+)")
    print("     - Using more episodes (5-10)")
    print("     - Different model or layers")

if __name__ == "__main__":
    main()