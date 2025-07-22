import json
import random
from collections import defaultdict
import sys
import yaml
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from social_decipher.environment.mcq_generator import SotopiaMCQGenerator
from openai import OpenAI

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs/config.yaml")
with open(CONFIG_PATH, "r") as f:
    _config = yaml.safe_load(f)

os.environ["OPENAI_API_KEY"] = _config.get("OPENAI_API_KEY") 

client = OpenAI()
mcq_gen = SotopiaMCQGenerator(client=client)

# Load datasets
with open("./processed_sotopia/sotopia_cleaned.jsonl") as f:
    all_episodes_90 = [json.loads(l) for l in f]
with open("./processed_sotopia/agent.jsonl") as f:
    agent_data = [json.loads(l) for l in f]
with open("./processed_sotopia/scenario.jsonl") as f:
    scenario_data = [json.loads(l) for l in f]
with open("./processed_sotopia/scenario_hard.jsonl") as f:
    scenario_hard_data = [json.loads(l) for l in f]

# get scenario_all_data by merging scenario_data and scenario_hard_data
scenario_all_data = scenario_data + scenario_hard_data

SOTOPIA_HARD_ENVS = ["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]
SOTOPIA_ALL_ENVS = list(set(ep["environment_id"] for ep in all_episodes_90)) 

print(len(SOTOPIA_ALL_ENVS))
# Indexing
agent_lookup = {a["pk"]: a for a in agent_data}
scenario_lookup = {s["pk"]: s for s in scenario_all_data}

episodes_by_env = defaultdict(list)
for ep in all_episodes_90:
    episodes_by_env[ep["environment_id"]].append(ep)

def clean_sotopia_goals(goals):
    """Clean goals by removing tags and extracting the core goal"""
    cleaned_goals = []
    for goal in goals:
        # Extract extra_info if present
        extra_info_start = goal.find("<extra_info>")
        extra_info_end = goal.find("</extra_info>")
        if extra_info_start != -1 and extra_info_end != -1:
            goal = goal[:extra_info_start].strip() + goal[extra_info_end + 13:].strip()
            
        # Extract clarification_hint if present
        hint_start = goal.find("<clarification_hint>")
        hint_end = goal.find("</clarification_hint>")
        if hint_start != -1 and hint_end != -1:
            goal = goal[:hint_start].strip() + goal[hint_end + 21:].strip()
            
        cleaned_goals.append(goal.strip())
    return cleaned_goals

# === Helper function to format episode first ===
def format_episode_basic(ep, scenario):
    """First step: Format episode with clean agent goals"""
    agent_ids = ep["agent_ids"]
    agents = [agent_lookup.get(agent_ids[0]), agent_lookup.get(agent_ids[1])]
    
    # Clean the goals first

    raw_goals = scenario["agent_goals"]
    clean_goals = raw_goals
    relationship = scenario["relationship"]

    def format_profile(profile):
        return (
            f"{profile['first_name']} {profile['last_name']}, a {profile['age']}-year-old {profile['occupation']}, "
            f"described as {profile['personality_and_values']}. Public info: {profile['public_info']} "
            f"Decision-making style: {profile['decision_making_style']}."
        )
    
    return {
        "episode_id": ep["episode_id"],
        "environment_id": ep["environment_id"],
        "scenario": scenario["scenario"],
        "codename": scenario["codename"],
        "agent_profiles": agents,
        "agent_goals": clean_goals,  # Use cleaned goals
        "agent_relationship": relationship,
        "agent1_profile": format_profile(agents[0]),
        "agent2_profile": format_profile(agents[1])
    }

# === Helper function to generate MCQs for formatted episode ===
def generate_mcqs_for_formatted_episode(formatted_episode):
    """Second step: Generate MCQs using clean formatted data"""
    sotopia_input = {
        "scenario": formatted_episode["scenario"],
        "agent1_goal": formatted_episode["agent_goals"][0],  # Already cleaned
        "agent2_goal": formatted_episode["agent_goals"][1],  # Already cleaned
        "relationship": formatted_episode["agent_relationship"],
        "agent1_profile": formatted_episode["agent1_profile"],
        "agent2_profile": formatted_episode["agent2_profile"]
    }

    try:
        mcq_result = mcq_gen.generate_mcqs_for_sotopia(sotopia_input)
    except Exception as e:
        print(f"Skipping episode due to MCQ generation error: {e}")
        return None

    # Extract MCQs from the correct structure
    mcqs_data = mcq_result.get("mcqs", {})
    agent_goals_mcqas = mcqs_data.get("goals", []) if isinstance(mcqs_data, dict) else []
    agent_reasons_mcqas = mcqs_data.get("reasons", []) if isinstance(mcqs_data, dict) else []
    agent_knowledge_mcqas = mcqs_data.get("knowledge", []) if isinstance(mcqs_data, dict) else []

    # Add MCQ data to formatted episode
    formatted_episode.update({
        "agent1_reason": mcq_result.get("agent1_reason", ""),
        "agent2_reason": mcq_result.get("agent2_reason", ""),
        "agent1_private_knowledge": mcq_result.get("agent1_private_knowledge", ""),
        "agent2_private_knowledge": mcq_result.get("agent2_private_knowledge", ""),
        "agent_goals_mcqas": agent_goals_mcqas,
        "agent_reasons_mcqas": agent_reasons_mcqas,
        "agent_knowledge_mcqas": agent_knowledge_mcqas
    })

    return formatted_episode

count = 0
for sid in SOTOPIA_ALL_ENVS:
    if sid in scenario_lookup.keys():
        count += 1

# === 2. episode_sample.jsonl ===
with open("episode_sample.jsonl", "w") as out_all:
    for sid in SOTOPIA_ALL_ENVS[:5]:
        scenario = scenario_lookup[sid]
        candidates = episodes_by_env.get(sid, [])
        selected_eps = random.sample(candidates, k=2)
 
        for ep in selected_eps:
            # Step 1: Format episode with clean goals
            formatted = format_episode_basic(ep, scenario)
            if formatted is None:
                continue
            
            # Step 2: Generate MCQs using clean data
            final_episode = generate_mcqs_for_formatted_episode(formatted)
            if final_episode is None:
                continue
                
            out_all.write(json.dumps(final_episode) + "\n")
exit()
# === 2. episode_all.jsonl ===
with open("episode_all.jsonl", "w") as out_all:
    for sid in SOTOPIA_ALL_ENVS:
        scenario = scenario_lookup[sid]
        candidates = episodes_by_env.get(sid, [])
        selected_eps = random.sample(candidates, k=2)
 
        for ep in selected_eps:
            # Step 1: Format episode with clean goals
            formatted = format_episode_basic(ep, scenario)
            if formatted is None:
                continue
            
            # Step 2: Generate MCQs using clean data
            final_episode = generate_mcqs_for_formatted_episode(formatted)
            if final_episode is None:
                continue
                
            out_all.write(json.dumps(final_episode) + "\n")

# === 2. episode_hard.jsonl ===
with open("episode_hard.jsonl", "w") as out_hard:
    for sid in SOTOPIA_HARD_ENVS:
        if sid not in scenario_lookup:
            continue
        scenario = scenario_lookup[sid]
        candidates = episodes_by_env.get(sid, [])
        selected_eps = random.sample(candidates, k=min(2, len(candidates)))
        for ep in selected_eps:
            # Step 1: Format episode with clean goals
            formatted = format_episode_basic(ep, scenario)
            if formatted is None:
                continue
            
            # Step 2: Generate MCQs using clean data
            final_episode = generate_mcqs_for_formatted_episode(formatted)
            if final_episode is None:
                continue
                
            out_hard.write(json.dumps(final_episode) + "\n")