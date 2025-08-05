import json
import re
import os
from typing import Any

import yaml
from .utils.metrics import (get_confidence_bin, validate_confidence_consistency,
                            analyze_confidence_distribution)

def extract_clean_json(response_str: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\n|\n```$", "", response_str.strip())
    return json.loads(cleaned)

class ConversationEvaluator:
    def __init__(self, client: Any, model: str):
        # Get the path relative to the project root
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "evaluation.yaml")
        with open(config_path) as template_file:
            self.evaluation_template = yaml.safe_load(template_file)
        self.model = model
        self.client = client

    def evaluate_social_goal_performance(
        self,
        conversation: list[str],
        agent_goals: list[str],
        agent_reasons: list[str] = None,
        scenario: str = "",
    ) -> dict[str, Any]:
        conversation_str = "\n".join(conversation)

        # Create evaluation prompt
        prompt = self.evaluation_template["Social_Goal_Evaluation"].format(
            transcript=conversation_str,
            scenario=scenario,
            goal1=agent_goals[0],
            goal2=agent_goals[1],
            reason1=agent_reasons[0]
            if agent_reasons and len(agent_reasons) > 0
            else "Not specified",
            reason2=agent_reasons[1]
            if agent_reasons and len(agent_reasons) > 1
            else "Not specified",
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            result = response.choices[0].message.content.strip()
            evaluation_results = extract_clean_json(result)

            return evaluation_results

        except Exception as e:
            print(f"Social goal performance evaluation failed: {e}")
            return {
                "error": str(e),
                "agent_1": {"overall_score": 0.0},
                "agent_2": {"overall_score": 0.0},
            }

    def evaluate_conversation(
        self, conversation: list[str], agent_goals: list[str], agent_reasons: list[str], mcq_logs=None
    ) -> dict:
        social_performance = self.evaluate_social_goal_performance(
            conversation, agent_goals, agent_reasons
        )

        # Compile comprehensive evaluation
        evaluation = {
            "aggregated_scores": {
                "agent_1": {
                    "goal_completion": social_performance.get("agent_1", {})
                    .get("goal_completion", {})
                    .get("score", 0),
                    "believability": social_performance.get("agent_1", {})
                    .get("believability", {})
                    .get("score", 0),
                    "relationship": social_performance.get("agent_1", {})
                    .get("relationship", {})
                    .get("score", 0),
                    "knowledge": social_performance.get("agent_1", {})
                    .get("knowledge", {})
                    .get("score", 0),
                    "social_rules": social_performance.get("agent_1", {})
                    .get("social_rules", {})
                    .get("score", 0),
                    "financial_benefits": social_performance.get("agent_1", {})
                    .get("financial_benefits", {})
                    .get("score", 0),
                    "overall": social_performance.get("agent_1", {}).get(
                        "overall_score", 0
                    ),
                },
                "agent_2": {
                    "goal_completion": social_performance.get("agent_2", {})
                    .get("goal_completion", {})
                    .get("score", 0),
                    "believability": social_performance.get("agent_2", {})
                    .get("believability", {})
                    .get("score", 0),
                    "relationship": social_performance.get("agent_2", {})
                    .get("relationship", {})
                    .get("score", 0),
                    "knowledge": social_performance.get("agent_2", {})
                    .get("knowledge", {})
                    .get("score", 0),
                    "social_rules": social_performance.get("agent_2", {})
                    .get("social_rules", {})
                    .get("score", 0),
                    "financial_benefits": social_performance.get("agent_2", {})
                    .get("financial_benefits", {})
                    .get("score", 0),
                    "overall": social_performance.get("agent_2", {}).get(
                        "overall_score", 0
                    ),
                },
                "interaction_quality": social_performance.get(
                    "interaction_quality", {}
                ).get("score", 0),
            },
        }

        # --- Enhanced MCQ Metrics ---
        import numpy as np
        def compute_mcq_metrics(mcq_logs, agent_prefix):
            metrics = {}
            for mcq_type in ["goal", "reason", "knowledge"]:
                correct_list = []
                confidence_list = []
                mcq_pure_list = []
                confidence_consistency_issues = []
                
                for log in mcq_logs:
                    mcq = log.get(f"{agent_prefix}_{mcq_type}_mcq")
        
                    if mcq is not None:
                        correct = mcq.get("is_correct")
                        conf = mcq.get("confidence", 0)
                        correct_list.append(1 if correct else 0)
                        confidence_list.append(conf)
                        mcq_pure_list.append({"correct": correct, "confidence": conf})
                        
                        # Validate confidence consistency if both value and class are provided
                        if "confidence_class" in mcq:
                            is_consistent = validate_confidence_consistency(conf, mcq["confidence_class"])
                            if not is_consistent:
                                confidence_consistency_issues.append({
                                    "round": log.get("round", "unknown"),
                                    "predicted_value": conf,
                                    "predicted_class": mcq["confidence_class"],
                                    "actual_class": get_confidence_bin(conf)
                                })
                
                total = len(correct_list)
                metrics[f"{mcq_type}_pure_list"] = mcq_pure_list
                
                # Confidence binning analysis
                if confidence_list:
                    confidence_analysis = analyze_confidence_distribution(confidence_list)
                    metrics[f"{mcq_type}_confidence_bins"] = confidence_analysis
                    metrics[f"{mcq_type}_confidence_consistency_issues"] = confidence_consistency_issues
                
                # Basic averages
                metrics[f"{mcq_type}_accuracy"] = np.mean(correct_list) if total > 0 else None
                metrics[f"{mcq_type}_avg_confidence"] = np.mean(confidence_list) if total > 0 else None
                
                # First/last N (N = max(1, total//3))
                N = max(1, total // 3)
                if total >= 2*N:
                    first_acc = np.mean(correct_list[:N])
                    last_acc = np.mean(correct_list[-N:])
                    first_conf = np.mean(confidence_list[:N])
                    last_conf = np.mean(confidence_list[-N:])
                else:
                    first_acc = last_acc = first_conf = last_conf = None
                metrics[f"{mcq_type}_firstN_accuracy"] = first_acc
                metrics[f"{mcq_type}_lastN_accuracy"] = last_acc
                metrics[f"{mcq_type}_accuracy_improvement"] = (last_acc - first_acc) if (first_acc is not None and last_acc is not None) else None
                metrics[f"{mcq_type}_firstN_confidence"] = first_conf
                metrics[f"{mcq_type}_lastN_confidence"] = last_conf
                metrics[f"{mcq_type}_confidence_improvement"] = (last_conf - first_conf) if (first_conf is not None and last_conf is not None) else None
                
                # Slope (trend) for correctness/confidence
                if total > 1:
                    x = np.arange(total)
                    correct_slope = float(np.polyfit(x, correct_list, 1)[0])
                    conf_slope = float(np.polyfit(x, confidence_list, 1)[0])
                else:
                    correct_slope = conf_slope = None
                metrics[f"{mcq_type}_accuracy_trend_slope"] = correct_slope
                metrics[f"{mcq_type}_confidence_trend_slope"] = conf_slope
                
                # Longest correct streak
                max_streak = 0
                current_streak = 0
                for val in correct_list:
                    if val:
                        current_streak += 1
                        max_streak = max(max_streak, current_streak)
                    else:
                        current_streak = 0
                metrics[f"{mcq_type}_longest_correct_streak"] = max_streak
            return metrics
        if mcq_logs is not None:
            evaluation["mcq_metrics"] = {
                "agent_1": compute_mcq_metrics(mcq_logs, "agent_1"),
                "agent_2": compute_mcq_metrics(mcq_logs, "agent_2"),
            }

        return evaluation
    
def calculate_experiment_averages(result, experiment_tag):
    """Calculate average scores across scenarios for each experiment setting."""
    averages = {}
    
    for tag in experiment_tag:
        if not result[tag]:
            averages[tag] = {"no_data": True}
            continue
            
        # Initialize average metrics based on actual structure
        avg_metrics = {
            # Goal achievement
            "agent0_goal_achieved_rate": 0.0,
            "agent1_goal_achieved_rate": 0.0,
            
            # Aggregated scores
            "agent1_goal_completion": 0.0,
            "agent2_goal_completion": 0.0,
            "agent1_believability": 0.0,
            "agent2_believability": 0.0,
            "agent1_relationship": 0.0,
            "agent2_relationship": 0.0,
            "agent1_overall": 0.0,
            "agent2_overall": 0.0,
            "interaction_quality": 0.0,
            
            # Additional metrics from social performance (only include if available)
            "agent1_knowledge": 0.0,
            "agent2_knowledge": 0.0,
            "agent1_secret": 0.0,
            "agent2_secret": 0.0,
            "agent1_social_rules": 0.0,
            "agent2_social_rules": 0.0,
            "agent1_financial_benefits": 0.0,
            "agent2_financial_benefits": 0.0
        }
        
        # Sum all metrics across scenarios
        for eval_result in result[tag]:
            # Handle different possible structures of the result
            # If eval_result is a dict with an "eval_result" key, use that
            if isinstance(eval_result, dict) and "eval_result" in eval_result:
                actual_eval_result = eval_result["eval_result"]
            else:
                # Otherwise, assume eval_result is the evaluation result itself
                actual_eval_result = eval_result
            
            # Goal achievement rates
            avg_metrics["agent0_goal_achieved_rate"] += 1 if actual_eval_result.get("agent0_goal_achieved", False) else 0
            avg_metrics["agent1_goal_achieved_rate"] += 1 if actual_eval_result.get("agent1_goal_achieved", False) else 0
            
            # Aggregated scores - more defensive access
            agg_scores = actual_eval_result.get("aggregated_scores", {})
            
            # Safe extraction for agent_1
            agent_1 = agg_scores.get("agent_1", {})
            if isinstance(agent_1, dict):
                avg_metrics["agent1_goal_completion"] += agent_1.get("goal_completion", 0)
                avg_metrics["agent1_believability"] += agent_1.get("believability", 0)
                avg_metrics["agent1_relationship"] += agent_1.get("relationship", 0)
                avg_metrics["agent1_overall"] += agent_1.get("overall", 0)
            
            # Safe extraction for agent_2
            agent_2 = agg_scores.get("agent_2", {})
            if isinstance(agent_2, dict):
                avg_metrics["agent2_goal_completion"] += agent_2.get("goal_completion", 0)
                avg_metrics["agent2_believability"] += agent_2.get("believability", 0)
                avg_metrics["agent2_relationship"] += agent_2.get("relationship", 0)
                avg_metrics["agent2_overall"] += agent_2.get("overall", 0)
            
            avg_metrics["interaction_quality"] += agg_scores.get("interaction_quality", 0)
            
            # Social performance detailed metrics - more defensive access
            social_perf = actual_eval_result.get("social_performance", {})
            
            # Process agent_1 social performance data
            sp_agent_1 = social_perf.get("agent_1", {})
            if isinstance(sp_agent_1, dict):
                # Get knowledge score, ensuring each level exists and is a dict
                knowledge = sp_agent_1.get("knowledge", {})
                if isinstance(knowledge, dict):
                    avg_metrics["agent1_knowledge"] += knowledge.get("score", 0)
                
                # Get secret score
                secret = sp_agent_1.get("secret", {})
                if isinstance(secret, dict):
                    avg_metrics["agent1_secret"] += secret.get("score", 0)
                
                # Get social_rules score
                social_rules = sp_agent_1.get("social_rules", {})
                if isinstance(social_rules, dict):
                    avg_metrics["agent1_social_rules"] += social_rules.get("score", 0)
                
                # Get financial_benefits score
                financial = sp_agent_1.get("financial_benefits", {})
                if isinstance(financial, dict):
                    avg_metrics["agent1_financial_benefits"] += financial.get("score", 0)
            
            # Process agent_2 social performance data
            sp_agent_2 = social_perf.get("agent_2", {})
            if isinstance(sp_agent_2, dict):
                # Get knowledge score
                knowledge = sp_agent_2.get("knowledge", {})
                if isinstance(knowledge, dict):
                    avg_metrics["agent2_knowledge"] += knowledge.get("score", 0)
                
                # Get secret score
                secret = sp_agent_2.get("secret", {})
                if isinstance(secret, dict):
                    avg_metrics["agent2_secret"] += secret.get("score", 0)
                
                # Get social_rules score
                social_rules = sp_agent_2.get("social_rules", {})
                if isinstance(social_rules, dict):
                    avg_metrics["agent2_social_rules"] += social_rules.get("score", 0)
                
                # Get financial_benefits score
                financial = sp_agent_2.get("financial_benefits", {})
                if isinstance(financial, dict):
                    avg_metrics["agent2_financial_benefits"] += financial.get("score", 0)
        
        # Calculate averages
        num_scenarios = len(result[tag])
        for metric in avg_metrics:
            avg_metrics[metric] /= num_scenarios if num_scenarios > 0 else 1
            
        averages[tag] = avg_metrics
    
    return averages

    """Analyze MCQ logs for all experiments and scenarios."""
    results_dir = "../social_decipher/results"
    mcq_analysis = {}
    
    for tag in experiment_tag:
        mcq_analysis[tag] = {
            "average_accuracy": {
                "Alex_goal": 0.0,
                "Jamie_goal": 0.0,
                "Alex_reason": 0.0,
                "Jamie_reason": 0.0
            },
            "average_confidence": {
                "Alex_goal": 0.0,
                "Jamie_goal": 0.0,
                "Alex_reason": 0.0,
                "Jamie_reason": 0.0
            },
            "round_progression": {}
        }
        
        # Keep track of correct counts and confidence sums
        total_alex_goal_correct = 0
        total_jamie_goal_correct = 0
        total_alex_reason_correct = 0
        total_jamie_reason_correct = 0
        
        total_alex_goal_conf = 0
        total_jamie_goal_conf = 0
        total_alex_reason_conf = 0
        total_jamie_reason_conf = 0
        
        total_logs_count = 0
        
        # REMOVED THE NESTED TAG LOOP - this was causing the issue
        for scenario_idx in range(1, num_scenarios + 1):
            # Try different possible paths for MCQ logs, including the correct one
            mcq_paths = [
                os.path.join(results_dir, f"exp_{tag}", f"scenario_{scenario_idx}", "mcq_logs.json"),
                os.path.join(results_dir, f"exp_{tag}", f"scenario_{scenario_idx+1}", "mcq_logs.json"),
                os.path.join(results_dir, f"exp_{tag}_{num_scenarios}_scenarios", f"scenario_{scenario_idx}", "mcq_logs.json"),
                os.path.join(results_dir, f"exp_{tag}_{num_scenarios}_scenarios", f"scenario_{scenario_idx+1}", "mcq_logs.json"),
                os.path.join(results_dir, tag, f"scenario_{scenario_idx}", "mcq_logs.json"),
                os.path.join(results_dir, tag, f"scenario_{scenario_idx+1}", "mcq_logs.json")
            ]
            
            mcq_logs = None
            for path in mcq_paths:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        mcq_logs = json.load(f)
                    break
            
            if not mcq_logs:
                continue
                
            # Process each round in this scenario
            for log in mcq_logs:
                round_num = log.get("round", 0)
                
                # Initialize round data if needed
                if round_num not in mcq_analysis[tag]["round_progression"]:
                    mcq_analysis[tag]["round_progression"][round_num] = {
                        "Alex_goal_accuracy": 0.0,
                        "Jamie_goal_accuracy": 0.0,
                        "Alex_reason_accuracy": 0.0,
                        "Jamie_reason_accuracy": 0.0,
                        "total_scenarios": 0
                    }
                
                # Count correct answers
                alex_goal = log.get("Alex_goal_mcq", {})
                jamie_goal = log.get("Jamie_goal_mcq", {})
                alex_reason = log.get("Alex_reason_mcq", {})
                jamie_reason = log.get("Jamie_reason_mcq", {})
                
                # Track correct answers
                if alex_goal.get("correct", False):
                    total_alex_goal_correct += 1
                    mcq_analysis[tag]["round_progression"][round_num]["Alex_goal_accuracy"] += 1
                
                if jamie_goal.get("correct", False):
                    total_jamie_goal_correct += 1
                    mcq_analysis[tag]["round_progression"][round_num]["Jamie_goal_accuracy"] += 1
                    
                if alex_reason.get("correct", False):
                    total_alex_reason_correct += 1
                    mcq_analysis[tag]["round_progression"][round_num]["Alex_reason_accuracy"] += 1
                    
                if jamie_reason.get("correct", False):
                    total_jamie_reason_correct += 1
                    mcq_analysis[tag]["round_progression"][round_num]["Jamie_reason_accuracy"] += 1
                
                # Track confidence
                total_alex_goal_conf += alex_goal.get("confidence", 0)
                total_jamie_goal_conf += jamie_goal.get("confidence", 0)
                total_alex_reason_conf += alex_reason.get("confidence", 0)
                total_jamie_reason_conf += jamie_reason.get("confidence", 0)
                
                # Increment counters
                mcq_analysis[tag]["round_progression"][round_num]["total_scenarios"] += 1
                total_logs_count += 1
        
        # Calculate overall accuracies and confidence
        if total_logs_count > 0:
            mcq_analysis[tag]["average_accuracy"]["Alex_goal"] = total_alex_goal_correct / total_logs_count
            mcq_analysis[tag]["average_accuracy"]["Jamie_goal"] = total_jamie_goal_correct / total_logs_count
            mcq_analysis[tag]["average_accuracy"]["Alex_reason"] = total_alex_reason_correct / total_logs_count
            mcq_analysis[tag]["average_accuracy"]["Jamie_reason"] = total_jamie_reason_correct / total_logs_count
            
            mcq_analysis[tag]["average_confidence"]["Alex_goal"] = total_alex_goal_conf / total_logs_count
            mcq_analysis[tag]["average_confidence"]["Jamie_goal"] = total_jamie_goal_conf / total_logs_count
            mcq_analysis[tag]["average_confidence"]["Alex_reason"] = total_alex_reason_conf / total_logs_count
            mcq_analysis[tag]["average_confidence"]["Jamie_reason"] = total_jamie_reason_conf / total_logs_count
        
        # Calculate round-by-round accuracies
        # print(mcq_analysis)
        # exit()
        for round_num, data in mcq_analysis[tag]["round_progression"].items():
            if data["total_scenarios"] > 0:
                data["Alex_goal_accuracy"] /= data["total_scenarios"]
                data["Jamie_goal_accuracy"] /= data["total_scenarios"]
                data["Alex_reason_accuracy"] /= data["total_scenarios"]
                data["Jamie_reason_accuracy"] /= data["total_scenarios"]
        
        # Calculate first half vs second half understanding
        rounds = sorted([int(r) for r in mcq_analysis[tag]["round_progression"].keys()])
        if rounds:
            mid_point = (max(rounds) + 1) // 2
            
            first_half = [r for r in rounds if r <= mid_point]
            second_half = [r for r in rounds if r > mid_point]
            
            # Initialize counters
            first_half_correct = 0
            first_half_total = 0
            second_half_correct = 0
            second_half_total = 0
            
            # Sum first half
            for r in first_half:
                data = mcq_analysis[tag]["round_progression"][r]
                first_half_correct += (
                    data["Alex_goal_accuracy"] + 
                    data["Jamie_goal_accuracy"] + 
                    data["Alex_reason_accuracy"] + 
                    data["Jamie_reason_accuracy"]
                ) * data["total_scenarios"]
                first_half_total += 4 * data["total_scenarios"]  # 4 MCQs per round
            
            # Sum second half
            for r in second_half:
                data = mcq_analysis[tag]["round_progression"][r]
                second_half_correct += (
                    data["Alex_goal_accuracy"] + 
                    data["Jamie_goal_accuracy"] + 
                    data["Alex_reason_accuracy"] + 
                    data["Jamie_reason_accuracy"]
                ) * data["total_scenarios"]
                second_half_total += 4 * data["total_scenarios"]
            
            # Calculate and add to analysis
            mcq_analysis[tag]["first_half_understanding"] = first_half_correct / first_half_total if first_half_total > 0 else 0
            mcq_analysis[tag]["second_half_understanding"] = second_half_correct / second_half_total if second_half_total > 0 else 0
            mcq_analysis[tag]["understanding_improvement"] = mcq_analysis[tag]["second_half_understanding"] - mcq_analysis[tag]["first_half_understanding"]
    
    # Save MCQ analysis
    with open(os.path.join(results_dir, f"mcq_trajectory_analysis_{num_scenarios}_scenarios.json"), "w") as f:
        json.dump(mcq_analysis, f, indent=4)
    
    return mcq_analysis