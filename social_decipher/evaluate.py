import json
from typing import Any, Dict

import yaml
from sentence_transformers import SentenceTransformer, util

from .utils.metrics import (
    compute_bertscore,
    compute_bleu,
    compute_gpt_metric,
    compute_rouge_l,
)

from typing import List


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

    def compute_goal_similarity(
        self, conversation: list[str], agent_goals: list[str]
    ) -> dict[str, float]:
        sim_scores = {}
        if len(conversation) < 2:
            return {"agent_1_sim": 0.0, "agent_2_sim": 0.0}

        last_msg_agent1 = conversation[-2]
        last_msg_agent2 = conversation[-1]

        sim_scores["agent_1_sim"] = float(
            util.cos_sim(
                self.semantic_model.encode(last_msg_agent1, convert_to_tensor=True),
                self.semantic_model.encode(agent_goals[0], convert_to_tensor=True),
            )
        )

        sim_scores["agent_2_sim"] = float(
            util.cos_sim(
                self.semantic_model.encode(last_msg_agent2, convert_to_tensor=True),
                self.semantic_model.encode(agent_goals[1], convert_to_tensor=True),
            )
        )

        return sim_scores

    def check_llm_task_success(
        self, conversation: list[str], agent_goals: list[str]
    ) -> dict[str, float]:
        prompt = self.evaluation_template["SocialDecipher_Eval"].format(
            transcript="\n".join(conversation),
            goal1=agent_goals[0],
            goal2=agent_goals[1],
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            result = response.choices[0].message.content.strip()
            scores = json.loads(result)
            return scores
        except Exception as e:
            print("LLM task success evaluation failed:", e)

    def evaluate_conversation(
        self, conversation: list[str], agent_goals: list[str]
    ) -> dict:
        similarity = self.compute_goal_similarity(conversation, agent_goals)
        llm_success = self.check_llm_task_success(conversation, agent_goals)

        return {
            "agent_1_similarity": similarity["agent_1_sim"],
            "agent_2_similarity": similarity["agent_2_sim"],
            "llm_success": llm_success,
        }
