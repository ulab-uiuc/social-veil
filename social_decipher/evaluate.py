import json
import yaml
from openai import OpenAI
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

class ConversationEvaluator:
    def __init__(self, client: Any, model: str):
        with open("../configs/evaluation.yaml", "r") as template_file:
            self.evaluation_template = yaml.safe_load(template_file)

        self.model = model
        self.client = client
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    def should_stop_conversation(self, agent_goals: List[str], conversation: List[str]) -> bool:

        prompt = self.evaluation_template["Stop_Criteria"].format(
            goal1=agent_goals[0],
            goal2=agent_goals[1],
            transcript="\n".join(conversation)
        )

        try:
            response = self.client.chat.completions.create(
                model= self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            result = response.choices[0].message.content.strip()
            return result.lower().startswith("yes")
        
        except Exception as e:
            print("LLM evaluation failed:", e)
            return False

    def count_turns(self, conversation: List[str]) -> int:
        return len(conversation)

    def compute_goal_similarity(self, conversation: List[str], agent_goals: List[str]) -> Dict[str, float]:
        sim_scores = {}
        if len(conversation) < 2:
            return {"agent_1_sim": 0.0, "agent_2_sim": 0.0}

        last_msg_agent1 = conversation[-2]
        last_msg_agent2 = conversation[-1]

        sim_scores["agent_1_sim"] = float(util.cos_sim(
            self.semantic_model.encode(last_msg_agent1, convert_to_tensor=True),
            self.semantic_model.encode(agent_goals[0], convert_to_tensor=True)
        ))

        sim_scores["agent_2_sim"] = float(util.cos_sim(
            self.semantic_model.encode(last_msg_agent2, convert_to_tensor=True),
            self.semantic_model.encode(agent_goals[1], convert_to_tensor=True)
        ))

        return sim_scores

    def check_llm_task_success(self, conversation: List[str], agent_goals: List[str]) -> Dict[str, float]:

        prompt = self.evaluation_template["SocialDecipher_Eval"].format(
            transcript="\n".join(conversation),
            goal1=agent_goals[0],
            goal2=agent_goals[1]
        )

        try:
            response = self.client.chat.completions.create(
                model= self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            result = response.choices[0].message.content.strip()
            scores = json.loads(result)
            return scores
        except Exception as e:
            print("LLM task success evaluation failed:", e)

    def evaluate_conversation(self, conversation: List[str], agent_goals: List[str]) -> Dict:
        turn_count = self.count_turns(conversation)
        similarity = self.compute_goal_similarity(conversation, agent_goals)
        llm_success = self.check_llm_task_success(conversation, agent_goals)

        return {
            "num_turns": turn_count,
            "agent_1_similarity": similarity["agent_1_sim"],
            "agent_2_similarity": similarity["agent_2_sim"],
            "llm_success": llm_success
        }
