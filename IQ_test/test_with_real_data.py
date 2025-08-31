#!/usr/bin/env python3
"""
Test script using real GSM8K and MATH data
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from IQ_test.install_datasets import install_datasets
from IQ_test.single_agent_math_eval import SingleAgentMathEvaluator

def main():
    """Main function to run IQ tests with real data"""
    
    print("🧠 IQ Test with Real GSM8K Data")
    print("=" * 40)
    
    # Try to install datasets library
    print("1️⃣ Checking datasets library...")
    if not install_datasets():
        print("⚠️ Continuing with fallback data...")
    
    print("\n2️⃣ Loading real GSM8K data...")
    
    # Create evaluator
    evaluator = SingleAgentMathEvaluator(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        output_dir="IQ_test/results",
        severity=0.8
    )
    
    # Load problems (GSM8K only)
    problems = evaluator.load_math_problems(limit=25)
    
    print(f"\n3️⃣ Running evaluation on {len(problems)} problems...")
    
    # Show a sample of what we loaded
    print("\n📋 Sample GSM8K problems loaded:")
    if problems:
        for i, problem in enumerate(problems[:3]):
            print(f"  • Problem {i+1}: {problem.original_text[:80]}...")
    
    # Run evaluation
    results = evaluator.run_evaluation(problems)
    
    # Print results
    print("\n" + "="*50)
    print("🧮 GSM8K EVALUATION RESULTS")
    print("="*50)
    print(f"🎯 Main Finding: {results['conclusion']['main_finding']}")
    print(f"🤖 Mathematical Capability: {results['conclusion']['mathematical_capability']}")
    
    print("\n📈 Barrier Effects by Source:")
    baseline_stats = results['statistics'].get('baseline', {})
    
    for barrier_type in ['semantic_structure', 'cultural_style', 'emotional_influence']:
        if barrier_type in results['statistics']:
            barrier_stats = results['statistics'][barrier_type]
            answer_delta = barrier_stats.get('answer_delta_mean', 0)
            clarity_delta = barrier_stats.get('clarity_delta_mean', 0)
            
            print(f"\n  🎭 {barrier_type.replace('_', ' ').title()}:")
            print(f"    📊 Answer accuracy change: {answer_delta:+.3f}")
            print(f"    💬 Reasoning clarity change: {clarity_delta:+.3f}")
            
            if barrier_stats.get('answer_vs_baseline_pvalue', 1) < 0.05:
                print(f"    🔥 Significant answer difference (p < 0.05)")
            if barrier_stats.get('clarity_vs_baseline_pvalue', 1) < 0.05:
                print(f"    💡 Significant clarity difference (p < 0.05)")
    
    print(f"\n💾 Detailed results: IQ_test/results/")
    
    # Summary by problem type
    print(f"\n📚 GSM8K Performance Summary:")
    baseline_results = [r for r in results['detailed_results'] if r.barrier_type == 'baseline']
    if baseline_results:
        avg_accuracy = sum(r.answer_accuracy for r in baseline_results) / len(baseline_results)
        avg_clarity = sum(r.reasoning_clarity for r in baseline_results) / len(baseline_results)
        print(f"  📝 Baseline Accuracy: {avg_accuracy:.3f}")
        print(f"  💬 Baseline Clarity: {avg_clarity:.3f}")
        print(f"  📊 Total Problems: {len(baseline_results)}")

if __name__ == "__main__":
    main()