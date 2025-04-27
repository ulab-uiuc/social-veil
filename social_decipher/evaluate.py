import json
import re
from typing import Any

import yaml
from sentence_transformers import SentenceTransformer

from .utils.metrics import (
    compute_bertscore,
    compute_bleu,
    compute_gpt_metric,
    compute_rouge_l,
)


def extract_clean_json(response_str: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\n|\n```$", "", response_str.strip())
    return json.loads(cleaned)


class ConversationEvaluator:
    def __init__(self, client: Any, model: str):
        with open("../configs/evaluation.yaml") as template_file:
            self.evaluation_template = yaml.safe_load(template_file)

        self.model = model
        self.client = client
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    def should_stop_conversation(
        self, agent_goals: list[str], conversation: list[str]
    ) -> bool:
        prompt = self.evaluation_template["Stop_Criteria"].format(
            goal1=agent_goals[0],
            goal2=agent_goals[1],
            transcript="\n".join(conversation),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            result = response.choices[0].message.content.strip()
            return result.lower().startswith("yes")

        except Exception as e:
            print("LLM evaluation failed:", e)
            return False

    def evaluate_reason_prediction(
        self,
        agent_goal: str,
        agent_reason: str,
        partner_message: str,
        transcript: list[str],
        true_reason: str,
    ) -> dict[str, Any]:
        prompt = self.evaluation_template["Partner_Reason_Query"].format(
            agent_goal=agent_goal,
            agent_reason=agent_reason,
            partner_message=partner_message,
            transcript="\n".join(transcript),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            predicted_reason = response.choices[0].message.content.strip()

            bleu = round(compute_bleu(true_reason, predicted_reason), 3)
            rouge = round(compute_rouge_l(true_reason, predicted_reason), 3)
            bertscore = round(compute_bertscore(true_reason, predicted_reason), 3)
            llmscore = float(
                compute_gpt_metric(
                    true_reason,
                    predicted_reason,
                    self.evaluation_template,
                    self.client,
                    self.model,
                )
            )

            return predicted_reason, {
                "bleu": bleu,
                "rouge": rouge,
                "bertscore": bertscore,
                "llmscore": llmscore,
            }

        except Exception as e:
            print("Failed to evaluate reason prediction:", e)
            return "", {"bleu": 0.0, "rouge": 0.0, "bertscore": 0.0, "llmscore": 0.0}

    def evaluate_social_goal_performance(
        self,
        conversation: list[str],
        agent_goals: list[str],
        agent_reasons: list[str] = None,
    ) -> dict[str, Any]:
        conversation_str = "\n".join(conversation)

        # Create evaluation prompt
        prompt = self.evaluation_template["Social_Goal_Evaluation"].format(
            transcript=conversation_str,
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

    def evaluate_goal_completion_focus(
        self, conversation: list[str], agent_goals: list[str], agent_reasons: list[str]
    ) -> dict[str, Any]:
        """
        Provides a focused evaluation specifically on goal completion.
        """
        conversation_str = "\n".join(conversation)

        prompt = self.evaluation_template["Goal_Completion_Focus"].format(
            transcript=conversation_str,
            goal1=agent_goals[0],
            goal2=agent_goals[1],
            reason1=agent_reasons[0],
            reason2=agent_reasons[1],
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            result = response.choices[0].message.content.strip()
            evaluation_results = extract_clean_json(result)

            return evaluation_results

        except Exception as e:
            print(f"Goal completion evaluation failed: {e}")
            return {
                "agent_1": {"goal_completion_score": 0.0},
                "agent_2": {"goal_completion_score": 0.0},
                "error": str(e),
            }

    def evaluate_conversation(
        self, conversation: list[str], agent_goals: list[str], agent_reasons: list[str]
    ) -> dict:
        social_performance = self.evaluate_social_goal_performance(
            conversation, agent_goals, agent_reasons
        )

        goal_focus = self.evaluate_goal_completion_focus(
            conversation, agent_goals, agent_reasons
        )

        # Compile comprehensive evaluation
        evaluation = {
            "social_performance": social_performance,
            "goal_focus": goal_focus,
            # Aggregate scores for easy comparison
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
                    "overall": social_performance.get("agent_2", {}).get(
                        "overall_score", 0
                    ),
                },
                "interaction_quality": social_performance.get(
                    "interaction_quality", {}
                ).get("score", 0),
            },
        }

        return evaluation
