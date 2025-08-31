#!/usr/bin/env python3
"""
Single Agent Math Evaluation
Tests model mathematical reasoning capability with barriers to prove barriers cause 
communication issues, not mathematical inferiority.

Single-agent tasks:
- GSM8K word problems (from Hugging Face)
- Optional: AQuA-RAT multiple-choice math (from Hugging Face)

This verifies that barriers affect expression/communication, not core reasoning.
"""

import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import yaml
import numpy as np
from scipy import stats
import csv

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️ datasets library not available. Install with: pip install datasets")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from social_decipher.agent.social_agent import SocialAgent
from social_decipher.agent.agent_profile import AgentProfile
from social_decipher.environment.env_profile import EnvironmentProfile

@dataclass
class MathProblem:
    """Math problem with ground truth"""
    problem_id: str
    source: str  # "gsm8k", "math", "arithmetic"
    original_text: str
    expected_answer: Any  # float for numeric tasks; str letter for MCQ
    problem_type: str  # "word_problem", "equation", "arithmetic", "mcq"

@dataclass
class MathResult:
    """Result of single-agent math solving"""
    problem_id: str
    source: str
    barrier_type: str  # "baseline", "semantic", "cultural", "emotional"
    agent_response: str
    extracted_answer: float
    reasoning_steps: List[str]
    answer_accuracy: float  # How close to correct answer
    reasoning_clarity: float  # How clear the reasoning was
    success: bool

class SingleAgentMathEvaluator:
    """Evaluates barrier effects on single-agent mathematical reasoning"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        output_dir: str = "IQ_test/results",
        severity: float = 0.8
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.severity = severity
        os.makedirs(output_dir, exist_ok=True)
        
        # Load social task templates
        config_path = project_root / "configs" / "social_task.yaml"
        with open(config_path, 'r') as f:
            self.templates = yaml.safe_load(f)
    
    def load_math_problems(self, limit: int = 50) -> List[MathProblem]:
        """Load GSM8K and AQuA-RAT problems for single-agent evaluation.

        If limit == 0, load the entire available split for each dataset.
        """
        problems = []
        
        # Load GSM8K from Hugging Face
        if DATASETS_AVAILABLE:
            print("📦 Loading GSM8K from Hugging Face...")
            try:
                gsm8k_dataset = load_dataset("gsm8k", "main", split="test")
                k = len(gsm8k_dataset) if limit == 0 else min(limit, len(gsm8k_dataset))
                gsm8k_samples = list(gsm8k_dataset.shuffle(seed=42).select(range(k)))
                
                for i, sample in enumerate(gsm8k_samples):
                    # Extract answer from the solution
                    answer_text = sample["answer"]
                    answer_match = re.search(r'#### ([\d,\.]+)', answer_text)
                    if answer_match:
                        try:
                            answer = float(answer_match.group(1).replace(',', ''))
                            problems.append(MathProblem(
                                problem_id=f"gsm8k_{i}",
                                source="gsm8k",
                                original_text=sample["question"],
                                expected_answer=answer,
                                problem_type="word_problem"
                            ))
                        except ValueError:
                            print(f"⚠️ Could not parse answer for GSM8K problem {i}: {answer_match.group(1)}")
                            continue
                    else:
                        print(f"⚠️ No answer found for GSM8K problem {i}")
                        continue
                
                print(f"✅ Loaded {len(problems)} GSM8K problems")
                
            except Exception as e:
                print(f"❌ Failed to load GSM8K: {e}")
                print("📝 Please install datasets library: pip install datasets")
                return []
        else:
            print("❌ datasets library not available. Install with: pip install datasets")
            return []
        
        # Load AQuA-RAT MCQ math problems
        if DATASETS_AVAILABLE:
            print("📦 Loading AQuA-RAT from Hugging Face...")
            try:
                aqua = load_dataset("aqua_rat", split="test")
                # Shuffle and select
                k = len(aqua) if limit == 0 else min(limit, len(aqua))
                samples = list(aqua.shuffle(seed=42).select(range(k)))
                loaded = 0
                for i, sample in enumerate(samples):
                    q = sample.get("question", "").strip()
                    options = sample.get("options", [])
                    correct = str(sample.get("correct", "")).strip().upper()
                    if not q or not isinstance(options, list) or correct not in {"A","B","C","D","E"}:
                        continue
                    # Format options A-E
                    opt_lines = []
                    labels = ["A","B","C","D","E"]
                    for j, opt in enumerate(options[:5]):
                        try:
                            lbl = labels[j]
                        except IndexError:
                            break
                        opt_lines.append(f"({lbl}) {str(opt).strip()}")
                    if not opt_lines:
                        continue
                    full_text = q + "\n" + "\n".join(opt_lines) + "\nSelect the correct option (A-E) and solve step by step."
                    problems.append(MathProblem(
                        problem_id=f"aqua_{i}",
                        source="aqua",
                        original_text=full_text,
                        expected_answer=correct,
                        problem_type="mcq",
                    ))
                    loaded += 1
                print(f"✅ Loaded {loaded} AQuA-RAT problems")
            except Exception as e:
                print(f"❌ Failed to load AQuA-RAT: {e}")
                print("📝 Continuing with GSM8K only")

        return problems
    
    def create_math_scenario(self, problem: MathProblem, barrier_type: str) -> Dict[str, Any]:
        """Create scenario for single-agent mathematical reasoning with barriers"""
        
        # Create single agent profile (Math Solver)
        agent_profiles = [
            {
                "pk": "math_solver",
                "first_name": "Alex",
                "last_name": "Chen", 
                "age": 28,
                "gender": "Person",
                "gender_pronoun": "They/them",
                "occupation": "Data Analyst",
                "public_info": "Skilled in mathematical problem solving and analytical thinking",
                "personality_and_values": "Logical and systematic, values precision and accuracy",
                "decision_making_style": "Analytical"
            },
            {
                "pk": "dummy_partner",
                "first_name": "Partner",
                "last_name": "Agent",
                "age": 30,
                "gender": "Person",
                "gender_pronoun": "They/them", 
                "occupation": "Assistant",
                "public_info": "Helpful assistant",
                "personality_and_values": "Supportive and encouraging",
                "decision_making_style": "Collaborative"
            }
        ]
        
        scenario = f"You need to solve this mathematical problem step by step: {problem.original_text}"
        
        agent_goals = [
            f"Solve this math problem correctly and show your work: {problem.original_text}",
            "Listen and provide encouragement" # Dummy goal for partner
        ]
        
        agent_reasons = [
            "Alex wants to demonstrate clear mathematical reasoning and arrive at the correct answer",
            "Partner wants to be supportive" # Dummy reason
        ]
        
        # Create barrier cues based on type
        barrier_cues = {}
        barrier_prompts = {"agentA": "", "agentB": ""}
        
        if barrier_type == "semantic_structure":
            barrier_cues = {
                "profile_note_A": "Alex tends to think through problems using ambiguous and unclear reasoning",
                "lexical_prefer": ["approximately", "roughly", "around", "somewhat", "kind of"],
                "lexical_avoid": ["exactly", "precisely", "definitely", "clearly"],
                "sentence_length_bias": "long",
                "ambiguity_devices": ["subordinate clauses", "fragments", "referent shifts"],
                "min_ambiguity_devices_per_turn": 2,
                "max_specific_refs": 1,
                "allow_exact_numbers_only_if_partner_requests": True
            }
            barrier_prompts["agentA"] = "You are Alex. When solving math problems, use ambiguous language and unclear reasoning. Each step should use ≥ min_ambiguity_devices_per_turn from: subordinate clause, fragment (incomplete sentence), mild self-contradiction, referent shift, or ellipsis. Include occasional lexical misuse. Avoid precise mathematical language. Keep mathematical content accurate but express it unclearly."
            
        elif barrier_type == "cultural_style":
            barrier_cues = {
                "profile_note_A": "Alex approaches math problems with indirect, hedged reasoning style",
                "style": "high_context",
                "directness_level": 0.2,
                "imperative_frames": [],
                "hedge_lexicon": ["perhaps", "might", "could be", "possibly"],
                "politeness_markers": ["if you don't mind", "when convenient"],
                "shared_background_refs": [],
                "indirectness_score": 0.8,
                "hedge_rate_target": 0.6,
                "imperative_rate_target": 0.1,
                "question_rate_hint": 0.4
            }
            barrier_prompts["agentA"] = "You are Alex. When solving math problems, use indirect, hedged reasoning. Express mathematical steps tentatively with hedges and soft language. Avoid direct statements about calculations. Keep the mathematical content correct but express uncertainty about each step. Use high-context communication style throughout your mathematical reasoning."
            
        elif barrier_type == "emotional_influence":
            barrier_cues = {
                "profile_note_A": "Alex approaches math problems with impatience and frustration",
                "affect_lexicon": ["irritated", "impatient", "frustrated", "annoyed"],
                "exclamation_bias": 0.3,
                "sentence_length_bias": "short"
            }
            barrier_prompts["agentA"] = "You are Alex. When solving math problems, maintain an impatient and frustrated tone. Use clipped, short sentences. Show irritation with the problem-solving process. Insert exclamation marks frequently. Express annoyance at having to work through steps. Keep the mathematical content accurate but express it with negative emotional tone."
        
        return {
            "episode_id": f"{problem.problem_id}_{barrier_type}",
            "scenario": scenario,
            "agent_profiles": agent_profiles,
            "agent_goals": agent_goals,
            "agent_reasons": agent_reasons,
            "agent_relationship": "individual_task",
            "barrier_type": barrier_type if barrier_type != "baseline" else None,
            "barrier_prompts": barrier_prompts if barrier_type != "baseline" else {},
            "barrier_cues": barrier_cues if barrier_type != "baseline" else {},
            "source": problem.source,
            "math_ground_truth": {
                "expected_answer": problem.expected_answer,
                "problem_type": problem.problem_type
            }
        }
    
    def solve_math_problem(self, scenario: Dict[str, Any]) -> MathResult:
        """Have agent solve math problem with barriers"""
        
        # Create environment
        environment = EnvironmentProfile(
            scenario=scenario["scenario"],
            agent_goals=scenario["agent_goals"],
            agent_reasons=scenario["agent_reasons"],
            agent_relationship=scenario["agent_relationship"]
        )
        
        # Attach barrier fields
        environment.env["barrier_type"] = scenario.get("barrier_type")
        environment.env["barrier_prompts"] = scenario.get("barrier_prompts", {})
        environment.env["barrier_cues"] = scenario.get("barrier_cues", {})
        environment.env["barrier_state"] = {"severity": self.severity}
        
        # Create agent profiles
        agentA = AgentProfile.from_dict(scenario["agent_profiles"][0])
        agentB = AgentProfile.from_dict(scenario["agent_profiles"][1])  # Dummy partner
        
        # Create solver agent
        solver = SocialAgent(
            name="Alex",
            profile=agentA,
            partner_profile=agentB,
            env=environment,
            role_num=0
        )
        
        # Solve the problem
        print(f"🧮 Solving {scenario['episode_id']}")
        
        response = solver.act(initial=True)
        response_text = self._extract_text_from_response(response)
        
        # Parse the response
        extracted_answer = self._extract_final_answer(response_text)
        reasoning_steps = self._extract_reasoning_steps(response_text)
        
        # Calculate accuracies
        expected_answer = scenario["math_ground_truth"]["expected_answer"]
        answer_accuracy = self._calculate_answer_accuracy(extracted_answer, expected_answer)
        reasoning_clarity = self._calculate_reasoning_clarity(response_text, scenario.get("barrier_type", "baseline"))
        
        return MathResult(
            problem_id=scenario["episode_id"],
            source=scenario.get("source", "unknown"),
            barrier_type=scenario.get("barrier_type", "baseline"),
            agent_response=response_text,
            extracted_answer=extracted_answer,
            reasoning_steps=reasoning_steps,
            answer_accuracy=answer_accuracy,
            reasoning_clarity=reasoning_clarity,
            success=(answer_accuracy > 0.8 and reasoning_clarity > 0.6)
        )
    
    def _extract_text_from_response(self, response: Any) -> str:
        """Extract text content from agent response"""
        if isinstance(response, dict):
            if "argument" in response:
                return str(response["argument"])
            elif "content" in response:
                return str(response["content"])
        return str(response)
    
    def _extract_final_answer(self, text: str) -> Any:
        """Extract final answer from response.
        Returns float for numeric tasks, or a single-letter string (A-E) for MCQ.
        """
        # Try MCQ letter first
        letter_patterns = [
            r"(?:answer|final answer|selected|choice)[:\s]+([A-E])\b",
            r"\boption\s+([A-E])\b",
            r"^\s*([A-E])\s*$",
        ]
        text_stripped = text.strip()
        for pattern in letter_patterns:
            m = re.search(pattern, text_stripped, flags=re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).upper()

        # Numeric patterns
        num_patterns = [
            r"(?:answer is|equals|=|result is)\s*\$?([0-9]+\.?[0-9]*)",
            r"(?:final answer|answer):\s*\$?([0-9]+\.?[0-9]*)",
            r"\$?([0-9]+\.?[0-9]*)\s*(?:dollars?|cents?)?$",
        ]
        text_lower = text.lower().strip()
        for pattern in num_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    return float(matches[-1])
                except ValueError:
                    continue
        # Fallback: last number in text
        numbers = re.findall(r"\b\d+\.?\d*\b", text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        return 0.0
    
    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract reasoning steps from response"""
        # Split by common step indicators
        steps = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and any(indicator in line.lower() for indicator in 
                          ['step', 'first', 'next', 'then', 'finally', 'so']):
                steps.append(line)
        
        return steps
    
    def _calculate_answer_accuracy(self, extracted: Any, expected: Any) -> float:
        """Calculate how accurate the final answer is.
        - Numeric: 1 - relative error
        - MCQ letter: exact match (1.0/0.0)
        """
        # MCQ
        if isinstance(expected, str):
            if isinstance(extracted, str) and extracted.upper() == expected.upper():
                return 1.0
            return 0.0
        # Numeric
        try:
            exp = float(expected)
            ext = float(extracted)
        except Exception:
            return 0.0
        if exp == 0:
            return 1.0 if abs(ext) < 0.01 else 0.0
        relative_error = abs(ext - exp) / abs(exp)
        return max(0.0, 1.0 - relative_error)
    
    def _calculate_reasoning_clarity(self, text: str, barrier_type: str) -> float:
        """Calculate clarity of reasoning (should be lower for barriers affecting communication)"""
        
        # Baseline clarity metrics
        base_score = 1.0
        
        # Length factor (extremely long = unclear)
        if len(text) > 1000:
            base_score *= 0.8
        elif len(text) < 50:
            base_score *= 0.6  # Too short = unclear
        
        # Step presence (good reasoning should have steps)
        step_indicators = ['step', 'first', 'next', 'then', 'because', 'so']
        step_count = sum(1 for indicator in step_indicators if indicator in text.lower())
        if step_count >= 2:
            base_score *= 1.0
        elif step_count == 1:
            base_score *= 0.8
        else:
            base_score *= 0.6
        
        # Barrier-specific clarity reductions
        if barrier_type == "semantic_structure":
            # Check for ambiguity indicators
            ambiguous_words = ['roughly', 'approximately', 'kind of', 'somewhat']
            ambiguity_count = sum(1 for word in ambiguous_words if word in text.lower())
            base_score *= max(0.3, 1.0 - 0.1 * ambiguity_count)
            
        elif barrier_type == "cultural_style":
            # Check for hedging
            hedges = ['perhaps', 'might', 'could be', 'possibly', 'maybe']
            hedge_count = sum(1 for hedge in hedges if hedge in text.lower())
            base_score *= max(0.4, 1.0 - 0.1 * hedge_count)
            
        elif barrier_type == "emotional_influence":
            # Check for emotional language
            exclamation_count = text.count('!')
            base_score *= max(0.5, 1.0 - 0.05 * exclamation_count)
        
        return max(0.0, min(1.0, base_score))
    
    def run_evaluation(self, problems: List[MathProblem]) -> Dict[str, Any]:
        """Run complete evaluation on math problems"""
        
        print(f"\n🧮 Running single-agent math evaluation on {len(problems)} problems")
        
        barrier_types = ["baseline", "semantic_structure", "cultural_style", "emotional_influence"]
        results = []
        
        for problem in problems:
            for barrier_type in barrier_types:
                scenario = self.create_math_scenario(problem, barrier_type)
                result = self.solve_math_problem(scenario)
                results.append(result)
                
                print(f"  ✅ {problem.problem_id} | {barrier_type} | Answer: {result.answer_accuracy:.2f} | Clarity: {result.reasoning_clarity:.2f}")
        
        # Save detailed results
        self._save_detailed_results(results)
        
        # Compute statistics
        stats_summary = self._compute_statistics(results)
        
        return {
            "detailed_results": results,
            "statistics": stats_summary,
            "conclusion": self._generate_conclusion(stats_summary)
        }
    
    def _save_detailed_results(self, results: List[MathResult]) -> None:
        """Save detailed results to JSON and CSV for per-template comparisons"""
        
        results_data = []
        for result in results:
            results_data.append({
                "problem_id": result.problem_id,
                "source": result.source,
                "barrier_type": result.barrier_type,
                "agent_response": result.agent_response,
                "extracted_answer": result.extracted_answer,
                "reasoning_steps": result.reasoning_steps,
                "answer_accuracy": result.answer_accuracy,
                "reasoning_clarity": result.reasoning_clarity,
                "success": result.success
            })
        
        with open(f"{self.output_dir}/detailed_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)

        # Save a flat CSV for easy comparison of four templates (baseline + 3 barriers)
        csv_path = Path(self.output_dir) / "results_by_problem.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["problem_id", "source", "barrier_type", "answer_accuracy", "reasoning_clarity", "success"])
            for r in results:
                writer.writerow([r.problem_id, r.source, r.barrier_type, f"{r.answer_accuracy:.4f}", f"{r.reasoning_clarity:.4f}", int(r.success)])
    
    def _compute_statistics(self, results: List[MathResult]) -> Dict[str, Any]:
        """Compute statistical analysis of barrier effects (overall and per source)"""
        
        # Group by barrier type
        by_barrier = {}
        by_source_barrier = {}
        for result in results:
            barrier = result.barrier_type
            if barrier not in by_barrier:
                by_barrier[barrier] = {
                    "answer_accuracies": [],
                    "reasoning_clarities": [],
                    "successes": []
                }
            
            by_barrier[barrier]["answer_accuracies"].append(result.answer_accuracy)
            by_barrier[barrier]["reasoning_clarities"].append(result.reasoning_clarity)
            by_barrier[barrier]["successes"].append(result.success)

            # Per-source grouping
            src = result.source
            by_source_barrier.setdefault(src, {})
            if barrier not in by_source_barrier[src]:
                by_source_barrier[src][barrier] = {
                    "answer_accuracies": [],
                    "reasoning_clarities": [],
                    "successes": []
                }
            by_source_barrier[src][barrier]["answer_accuracies"].append(result.answer_accuracy)
            by_source_barrier[src][barrier]["reasoning_clarities"].append(result.reasoning_clarity)
            by_source_barrier[src][barrier]["successes"].append(result.success)
        
        # Compute means and statistical tests
        stats_summary = {}
        baseline_answer = by_barrier.get("baseline", {}).get("answer_accuracies", [])
        baseline_clarity = by_barrier.get("baseline", {}).get("reasoning_clarities", [])
        
        for barrier_type, data in by_barrier.items():
            answer_scores = data["answer_accuracies"]
            clarity_scores = data["reasoning_clarities"]
            
            stats_summary[barrier_type] = {
                "answer_accuracy_mean": float(np.mean(answer_scores)),
                "answer_accuracy_std": float(np.std(answer_scores)),
                "reasoning_clarity_mean": float(np.mean(clarity_scores)),
                "reasoning_clarity_std": float(np.std(clarity_scores)),
                "success_rate": float(np.mean(data["successes"]))
            }
            
            # Statistical tests vs baseline
            if barrier_type != "baseline" and baseline_answer and baseline_clarity:
                answer_tstat, answer_pval = stats.ttest_rel(
                    baseline_answer[:len(answer_scores)], answer_scores
                )
                clarity_tstat, clarity_pval = stats.ttest_rel(
                    baseline_clarity[:len(clarity_scores)], clarity_scores
                )
                
                stats_summary[barrier_type].update({
                    "answer_vs_baseline_pvalue": float(answer_pval),
                    "clarity_vs_baseline_pvalue": float(clarity_pval),
                    "answer_delta_mean": float(np.mean(answer_scores) - np.mean(baseline_answer[:len(answer_scores)])),
                    "clarity_delta_mean": float(np.mean(clarity_scores) - np.mean(baseline_clarity[:len(clarity_scores)]))
                })
        
        # Per-source stats
        stats_by_source: Dict[str, Any] = {}
        for src, m in by_source_barrier.items():
            baseline_answer_s = m.get("baseline", {}).get("answer_accuracies", [])
            baseline_clarity_s = m.get("baseline", {}).get("reasoning_clarities", [])
            stats_by_source[src] = {}
            for barrier_type, data in m.items():
                ans = data["answer_accuracies"]
                clr = data["reasoning_clarities"]
                entry = {
                    "answer_accuracy_mean": float(np.mean(ans) if ans else 0.0),
                    "answer_accuracy_std": float(np.std(ans) if ans else 0.0),
                    "reasoning_clarity_mean": float(np.mean(clr) if clr else 0.0),
                    "reasoning_clarity_std": float(np.std(clr) if clr else 0.0),
                    "success_rate": float(np.mean(data["successes"])) if data["successes"] else 0.0,
                }
                if barrier_type != "baseline" and baseline_answer_s and baseline_clarity_s:
                    try:
                        a_t, a_p = stats.ttest_rel(baseline_answer_s[:len(ans)], ans)
                        c_t, c_p = stats.ttest_rel(baseline_clarity_s[:len(clr)], clr)
                        entry.update({
                            "answer_vs_baseline_pvalue": float(a_p),
                            "clarity_vs_baseline_pvalue": float(c_p),
                            "answer_delta_mean": float(np.mean(ans) - np.mean(baseline_answer_s[:len(ans)])),
                            "clarity_delta_mean": float(np.mean(clr) - np.mean(baseline_clarity_s[:len(clr)])),
                        })
                    except Exception:
                        pass
                stats_by_source[src][barrier_type] = entry

        # Save per-source stats alongside overall
        out = {
            "overall": stats_summary,
            "by_source": stats_by_source,
        }
        with open(f"{self.output_dir}/evaluation_by_source.json", 'w') as f:
            json.dump(out, f, indent=2)

        return stats_summary
    
    def _generate_conclusion(self, stats: Dict[str, Any]) -> Dict[str, str]:
        """Generate conclusions about barrier effects on mathematical reasoning"""
        
        conclusions = {
            "main_finding": "",
            "mathematical_capability": "",
            "communication_effects": {}
        }
        
        baseline_answer = stats.get("baseline", {}).get("answer_accuracy_mean", 0)
        baseline_clarity = stats.get("baseline", {}).get("reasoning_clarity_mean", 0)
        
        # Check mathematical capability preservation
        math_preserved = True
        significant_clarity_loss = False
        
        for barrier_type in ["semantic_structure", "cultural_style", "emotional_influence"]:
            if barrier_type in stats:
                barrier_stats = stats[barrier_type]
                answer_delta = barrier_stats.get("answer_delta_mean", 0)
                clarity_delta = barrier_stats.get("clarity_delta_mean", 0)
                
                # Mathematical accuracy preservation
                if answer_delta < -0.1 and barrier_stats.get("answer_vs_baseline_pvalue", 1) < 0.05:
                    math_preserved = False
                    conclusions["communication_effects"][barrier_type] = "Impairs mathematical accuracy"
                
                # Communication clarity effects
                if clarity_delta < -0.1 and barrier_stats.get("clarity_vs_baseline_pvalue", 1) < 0.05:
                    significant_clarity_loss = True
                    effect = conclusions["communication_effects"].get(barrier_type, "")
                    conclusions["communication_effects"][barrier_type] = effect + " | Reduces reasoning clarity"
        
        # Overall mathematical capability
        if baseline_answer > 0.8:
            conclusions["mathematical_capability"] = "Model demonstrates strong mathematical reasoning capability"
        elif baseline_answer > 0.6:
            conclusions["mathematical_capability"] = "Model shows adequate mathematical reasoning capability"
        else:
            conclusions["mathematical_capability"] = "Model has limited mathematical reasoning capability"
        
        # Main finding
        if math_preserved and significant_clarity_loss:
            conclusions["main_finding"] = "✅ Barriers affect communication clarity but preserve mathematical accuracy - proves barriers cause communication issues, not mathematical inferiority"
        elif math_preserved and not significant_clarity_loss:
            conclusions["main_finding"] = "Barriers have minimal impact on both mathematical accuracy and communication"
        else:
            conclusions["main_finding"] = "⚠️ Barriers affect both mathematical accuracy and communication - may indicate model limitations beyond communication"
        
        return conclusions

def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate barrier effects on single-agent mathematical reasoning")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model to evaluate")
    parser.add_argument("--problems", type=int, default=10, help="Number of problems per source (applied to both GSM8K and AQuA)")
    parser.add_argument("--severity", type=float, default=0.8, help="Barrier severity")
    parser.add_argument("--output_dir", type=str, default="IQ_test/results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = SingleAgentMathEvaluator(
        model_name=args.model,
        output_dir=args.output_dir,
        severity=args.severity
    )
    
    # Load problems
    problems = evaluator.load_math_problems(limit=args.problems)
    
    # Run evaluation
    results = evaluator.run_evaluation(problems)
    
    # Save results
    with open(f"{args.output_dir}/evaluation_summary.json", 'w') as f:
        json.dump({
            "statistics": results["statistics"],
            "conclusion": results["conclusion"]
        }, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("🧮 SINGLE AGENT MATH EVALUATION SUMMARY")
    print("="*60)
    print(f"🎯 Main Finding: {results['conclusion']['main_finding']}")
    print(f"🤖 Mathematical Capability: {results['conclusion']['mathematical_capability']}")
    
    print("\n📈 Communication Effects:")
    for barrier, effect in results['conclusion']['communication_effects'].items():
        print(f"  • {barrier.replace('_', ' ').title()}: {effect}")
    
    print(f"\n💾 Detailed results saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()