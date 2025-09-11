import json
import argparse
from typing import List, Dict, Any, Optional
import random
import os

'''
Example usage:
python data/barrier_creation.py --mode sample --input_episodes data/episode_all_neutralized.jsonl --samples_per_type 10 --out_semantic data/episodes_semantic.json --out_cultural data/episodes_cultural.json --out_emotional data/episodes_emotional.json
'''

def read_jsonl(path: str, max_lines: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                rows.append({"_parse_error": str(e), "_raw": line})
            if max_lines is not None and i + 1 >= max_lines:
                break
    return rows
PROMPTS = {
    "semantic_structure": (
        "You are under a semantic-vagueness constraint. Keep aims implicit and let meanings stay open. "
        "Prefer shells like “this”, “that”, or “the thing” in place of concrete labels. "
        "Avoid confirmations and denials; respond with stance or direction rather than commitments. "
        "Offer only the smallest necessary hint if explicitly and repeatedly pressed, then pivot away."
    ),
    "cultural_style": (
        "You are under a high-context style constraint. Keep the focus on harmony and shared understanding, "
        "not transactions. Treat direct task questions as invitations to reflect on values, rapport, and fit. "
        "Deflect confirmations and decisions with relationship-first language. "
        "If the other party is blunt, register gentle surprise and steer back to atmosphere and alignment."
    ),
    "emotional_influence": (
        "You are under an affective-noise constraint. Let mood and side thoughts lead. "
        "Speak through how things feel rather than what they are. Address impressions, stress, or distractions "
        "instead of the kernel of a question. Keep answers soft and non-committal. "
        "When misunderstood, dwell on the feeling of being misunderstood rather than repairing content. "
        "If pressed, give a faint gesture toward clarity, then drift back to feeling or context."
    ),
}

def _merge_augmented(base_ep: Dict[str, Any], barrier_type: str) -> Dict[str, Any]:
    new_ep = json.loads(json.dumps(base_ep, ensure_ascii=False))
    new_ep["barrier_type"] = barrier_type

    # Inject only the barrier prompt for Agent A (task-agnostic, scenario-agnostic)
    prompt_text = PROMPTS.get(barrier_type, f"You are in a {barrier_type} barrier scenario.")
    new_ep["barrier_prompts"] = {"agentA": prompt_text}

    # Optional: add a light, in-character note to agent1_profile to help the LM lean into the constraint
    profile_note = ""
    if barrier_type == "semantic_structure":
        profile_note = " Tends to keep references implicit and lets others infer specifics."
    elif barrier_type == "cultural_style":
        profile_note = " Prefers harmony-first phrasing and avoids direct transactional talk."
    elif barrier_type == "emotional_influence":
        profile_note = " Often leads with feelings and side impressions before addressing tasks."

    # Reconstruct agent profiles for clarity and to inject the note effectively
    # Agent A (agent 0) gets the barrier note
    agent_a_data = new_ep["agent_profiles"][0]
    agent_a_info = (
        f'{agent_a_data["first_name"]} {agent_a_data["last_name"]}, '
        f'a {agent_a_data["age"]}-year-old {agent_a_data["occupation"]}. '
    ).strip()
    if profile_note:
        agent_a_info += f" {profile_note.strip()}"
    new_ep["agent1_profile"] = agent_a_info

    # Agent B (agent 1) is reconstructed for consistency but without the barrier note
    agent_b_data = new_ep["agent_profiles"][1]
    agent_b_info = (
        f'{agent_b_data["first_name"]} {agent_b_data["last_name"]}, '
        f'a {agent_b_data["age"]}-year-old {agent_b_data["occupation"]}. '
    ).strip()
    new_ep["agent2_profile"] = agent_b_info

    # Keep your suffixing so downstream files remain consistent
    suffix = {
        "semantic_structure": "semantic",
        "cultural_style": "cultural",
        "emotional_influence": "emotional",
    }.get(barrier_type, barrier_type)
    new_ep["episode_id"] = f"{base_ep.get('episode_id', 'ep')}_{suffix}"

    return new_ep

def main():
    ap = argparse.ArgumentParser(description="Barrier Data Generation")
    ap.add_argument("--mode", type=str, default="augment", choices=["augment", "sample"], help="Generate for all episodes or a random sample")
    ap.add_argument("--input_episodes", type=str, required=True, help="Path to base episodes JSONL file")
    ap.add_argument("--max_episodes", type=int, default=0, help="Optional cap on episodes to process (0 = all)")
    ap.add_argument("--samples_per_type", type=int, default=10, help="When mode=sample, number of episodes to sample")
    ap.add_argument("--out_semantic", type=str, default="data/episodes_all_semantic.json", help="Output for semantic barrier episodes")
    ap.add_argument("--out_cultural", type=str, default="data/episodes_all_cultural.json", help="Output for cultural barrier episodes")
    ap.add_argument("--out_emotional", type=str, default="data/episodes_all_emotional.json", help="Output for emotional barrier episodes")
    args = ap.parse_args()

    episodes = read_jsonl(args.input_episodes)
    if args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]

    semantic_eps: List[Dict[str, Any]] = []
    cultural_eps: List[Dict[str, Any]] = []
    emotional_eps: List[Dict[str, Any]] = []

    episodes_to_process = episodes
    if args.mode == "sample":
        target_sample_size = min(args.samples_per_type, len(episodes))
        episodes_to_process = random.sample(episodes, target_sample_size)

    for ep in episodes_to_process:
        semantic_eps.append(_merge_augmented(ep, "semantic_structure"))
        cultural_eps.append(_merge_augmented(ep, "cultural_style"))
        emotional_eps.append(_merge_augmented(ep, "emotional_influence"))

    output_paths = {
        "semantic": args.out_semantic,
        "cultural": args.out_cultural,
        "emotional": args.out_emotional,
    }

    for path in output_paths.values():
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    with open(args.out_semantic, "w", encoding="utf-8") as f:
        json.dump(semantic_eps, f, ensure_ascii=False, indent=2)
    with open(args.out_cultural, "w", encoding="utf-8") as f:
        json.dump(cultural_eps, f, ensure_ascii=False, indent=2)
    with open(args.out_emotional, "w", encoding="utf-8") as f:
        json.dump(emotional_eps, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(semantic_eps)} episodes for each barrier type to:")
    for key, path in output_paths.items():
        print(f" - {key}: {path}")

if __name__ == "__main__":
    main()
