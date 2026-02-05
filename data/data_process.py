import argparse
import json
import os
import random
import sys
from collections import defaultdict

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from openai import OpenAI

from socialveil.environment.mcq_generator import SotopiaMCQGenerator

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs/config.yaml")
with open(CONFIG_PATH, "r") as f:
    _config = yaml.safe_load(f)

os.environ["OPENAI_API_KEY"] = _config.get("OPENAI_API_KEY") 

client = OpenAI()
mcq_gen = SotopiaMCQGenerator(client=client)

# CLI: control which components to generate/update
parser = argparse.ArgumentParser(description="Generate Sotopia MCQs/components")
parser.add_argument(
    "--task", choices=["reasons", "knowledge", "both", "goals"], default="both",
    help="What to generate/update for each episode"
)
parser.add_argument(
    "--input_episodes", type=str, default=None,
    help="Path to an existing episodes JSONL to update in-place (used with --task goals)"
)
parser.add_argument(
    "--output_episodes", type=str, default=None,
    help="Optional output path for updated episodes JSONL (defaults to --input_episodes)"
)
ARGS = parser.parse_args()
TASK = ARGS.task


# === Helper to shuffle MCQ options and update correct_answer ===
def shuffle_mcq_options(mcq_list):
    for mcq in mcq_list or []:
        options = mcq.get("options", {})
        if not options or "correct_answer" not in mcq:
            continue
        correct_key = mcq.get("correct_answer")
        if correct_key not in options:
            continue
        correct_value = options[correct_key]
        items = list(options.items())
        random.shuffle(items)
        new_options = {}
        new_correct_key = None
        for idx, (_k, v) in enumerate(items):
            new_key = chr(ord('A') + idx)
            new_options[new_key] = v
            if v == correct_value:
                new_correct_key = new_key
        mcq["options"] = new_options
        if new_correct_key:
            mcq["correct_answer"] = new_correct_key


# Fast path: update ONLY goal MCQs on an existing episodes file without regenerating reasons/knowledge
if TASK == "goals" and ARGS.input_episodes:
    input_path = ARGS.input_episodes
    output_path = ARGS.output_episodes or input_path
    updated = 0
    with open(input_path, "r") as f:
        lines = [l for l in f if l.strip()]
    out_lines = []
    for line in lines:
        ep = json.loads(line)
        sotopia_input = {
            "scenario": ep.get("scenario", ""),
            "agent1_goal": (ep.get("agent_goals") or ["", ""])[0],
            "agent2_goal": (ep.get("agent_goals") or ["", ""])[1],
            "relationship": ep.get("agent_relationship", "friend"),
            "agent1_profile": ep.get("agent1_profile", ""),
            "agent2_profile": ep.get("agent2_profile", ""),
        }
        try:
            g = mcq_gen.generate_goal_mcqs(sotopia_input)
            goals = g.get("mcqs", {}).get("goals", [])
            if goals:
                ep["agent_goals_mcqas"] = goals
                shuffle_mcq_options(ep["agent_goals_mcqas"])  # randomize A/B/C/D positions
                updated += 1
        except Exception as e:
            print(f"Warning: failed to update goals for one episode: {e}")
        out_lines.append(json.dumps(ep, ensure_ascii=False))

    with open(output_path, "w") as f:
        for l in out_lines:
            f.write(l + "\n")
    print(f"Updated goal MCQs for {updated}/{len(out_lines)} episodes -> {output_path}")
    sys.exit(0)

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
    """Second step: Generate components using clean formatted data.
    Updates only the requested parts based on TASK (reasons/knowledge/both).
    """
    sotopia_input = {
        "scenario": formatted_episode["scenario"],
        "agent1_goal": formatted_episode["agent_goals"][0],  # Already cleaned
        "agent2_goal": formatted_episode["agent_goals"][1],  # Already cleaned
        "relationship": formatted_episode["agent_relationship"],
        "agent1_profile": formatted_episode["agent1_profile"],
        "agent2_profile": formatted_episode["agent2_profile"]
    }

    try:
        # Generate/update reasons
        if TASK in ("reasons", "both"):
            res = mcq_gen.generate_reasons(sotopia_input)
            formatted_episode["agent1_reason"] = res.get("agent1_reason", formatted_episode.get("agent1_reason", ""))
            formatted_episode["agent2_reason"] = res.get("agent2_reason", formatted_episode.get("agent2_reason", ""))
            reasons = res.get("mcqs", {}).get("reasons", [])
            if reasons:
                formatted_episode["agent_reasons_mcqas"] = reasons
                # Randomize correct answer position for reasons MCQs
                shuffle_mcq_options(formatted_episode["agent_reasons_mcqas"])

        # Generate/update private knowledge
        if TASK in ("knowledge", "both"):
            kres = mcq_gen.generate_knowledge(sotopia_input)
            formatted_episode["agent1_private_knowledge"] = kres.get("agent1_private_knowledge", formatted_episode.get("agent1_private_knowledge", ""))
            formatted_episode["agent2_private_knowledge"] = kres.get("agent2_private_knowledge", formatted_episode.get("agent2_private_knowledge", ""))
            knowledge = kres.get("mcqs", {}).get("knowledge", [])
            if knowledge:
                formatted_episode["agent_knowledge_mcqas"] = knowledge
                # Randomize correct answer position for knowledge MCQs
                shuffle_mcq_options(formatted_episode["agent_knowledge_mcqas"])

        # Always (re)generate goal MCQs from the scenario+goals (independent of reasons/knowledge)
        g = mcq_gen.generate_goal_mcqs(sotopia_input)
        goals = g.get("mcqs", {}).get("goals", [])
        if goals:
            formatted_episode["agent_goals_mcqas"] = goals
            shuffle_mcq_options(formatted_episode["agent_goals_mcqas"])
    except Exception as e:
        print(f"Skipping episode due to MCQ generation error: {e}")
        return None

    return formatted_episode

count = 0
for sid in SOTOPIA_ALL_ENVS:
    if sid in scenario_lookup.keys():
        count += 1

# === 2. episode_sample.jsonl ===
# with open("episode_sample.jsonl", "w") as out_all:
#     for sid in SOTOPIA_ALL_ENVS[:5]:
#         scenario = scenario_lookup[sid]
#         candidates = episodes_by_env.get(sid, [])
#         selected_eps = random.sample(candidates, k=2)
 
#         for ep in selected_eps:
#             # Step 1: Format episode with clean goals
#             formatted = format_episode_basic(ep, scenario)
#             if formatted is None:
#                 continue
            
#             # Step 2: Generate requested components using clean data
#             final_episode = generate_mcqs_for_formatted_episode(formatted)
#             if final_episode is None:
#                 continue
                
#             out_all.write(json.dumps(final_episode) + "\n")

# # === 2. episode_hard.jsonl ===
# with open("episode_hard.jsonl", "w") as out_hard:
#     for sid in SOTOPIA_HARD_ENVS:
#         if sid not in scenario_lookup:
#             continue
#         scenario = scenario_lookup[sid]
#         candidates = episodes_by_env.get(sid, [])
#         selected_eps = random.sample(candidates, k=min(2, len(candidates)))
#         for ep in selected_eps:
#             # Step 1: Format episode with clean goals
#             formatted = format_episode_basic(ep, scenario)
#             if formatted is None:
#                 continue
            
#             # Step 2: Generate MCQs using clean data
#             final_episode = generate_mcqs_for_formatted_episode(formatted)
#             if final_episode is None:
#                 continue
                
#             out_hard.write(json.dumps(final_episode) + "\n")

# exit()
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
