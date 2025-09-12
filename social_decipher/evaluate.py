import json
import re
import os
from openai import OpenAI
from typing import Any, Optional
import numpy as np
import time
import random

import yaml
from .utils.metrics import (get_confidence_bin, validate_confidence_consistency,
                            analyze_confidence_distribution)
from .utils.error_handler import api_calling_error_exponential_backoff

def extract_clean_json(response_str: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\n|\n```$", "", response_str.strip())
    return json.loads(cleaned)

class ConversationEvaluator:
    def __init__(self, model: str):
        # Get the path relative to the project root
        eval_config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "evaluation.yaml")
        with open(eval_config_path) as template_file:
            self.evaluation_template = yaml.safe_load(template_file)

        # Load main config to get the evaluator-specific API key
        main_config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
        with open(main_config_path) as config_file:
            main_config = yaml.safe_load(config_file)
        
        evaluator_api_key = main_config.get("EVALUATOR_OPENAI_API_KEY")
        if not evaluator_api_key:
            raise ValueError("EVALUATOR_OPENAI_API_KEY not found in config.yaml")

        self.model = model
        self.client = OpenAI(api_key=evaluator_api_key)

    @api_calling_error_exponential_backoff()
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip()
        evaluation_results = extract_clean_json(result)

        return evaluation_results

    @api_calling_error_exponential_backoff()
    def evaluate_conversation(
        self,
        conversation: list[str],
        agent_goals: list[str],
        agent_reasons: list[str],
        scenario: str = "",
        mcq_logs=None,
        barrier_type: Optional[str] = None,
    ) -> dict:
        social_performance = self.evaluate_social_goal_performance(
            conversation, agent_goals, agent_reasons
        )


        print(f"Barrier type for evaluation: {barrier_type}")

        barrier_scores = None
        if self.evaluation_template.get("Barrier_Evaluation"):
            transcript_text = "\n".join(conversation)
            barrier_prompt = self.evaluation_template["Barrier_Evaluation"].format(
                transcript=transcript_text,
                scenario=scenario,
                agent_a_goal=agent_goals[0],
                agent_b_goal=agent_goals[1],
            )
         
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": barrier_prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message.content.strip()
            barrier_scores = extract_clean_json(content)
            
            print(barrier_scores)
           

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
                "episode_level": {
                    "unresolved_confusion": None,
                    "mutual_understanding": None,
                },
            },
        }
        print(f"Barrier scores: {barrier_scores}")
    
        if isinstance(barrier_scores, dict):
            ep = barrier_scores.get("episode_level", {})
            evaluation["aggregated_scores"]["episode_level"][
                "unresolved_confusion"
            ] = ep.get("unresolved_confusion", {}).get("score", 0)
            evaluation["aggregated_scores"]["episode_level"][
                "mutual_understanding"
            ] = ep.get("mutual_understanding", {}).get("score", 0)

        print("Evaluation Results:", json.dumps(evaluation, indent=2))
        
        # --- Enhanced MCQ Metrics ---
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
