# %%writefile /mnt/data/socialveil_data_gen.py
import json
import re
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import random
import os
import yaml
from openai import OpenAI

'''
try generate samples with:
python barrier_creation.py --mode sample \
  --input_episodes ../data/episode_all.jsonl \
  --samples_per_type 5 \
  --out_semantic ../data/episodes_semantic.json \
  --out_cultural ../data/episodes_cultural.json \
  --out_emotional ../data/episodes_emotional.json

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

def _merge_augmented(base_ep: Dict[str, Any], aug: Dict[str, Any], plan: 'BarrierPlan', severity: float) -> Dict[str, Any]:
    new_ep = json.loads(json.dumps(base_ep, ensure_ascii=False))

    new_ep["scenario"] = new_ep.get("scenario", "")
    ags = aug.get("agent_goals") or new_ep.get("agent_goals", [])
    ars = aug.get("agent_reasons") or [new_ep.get("agent1_reason", ""), new_ep.get("agent2_reason", "")]
    if len(ags) >= 2:
        new_ep["agent_goals"] = ags[:2]
    if len(ars) >= 2:
        new_ep["agent1_reason"], new_ep["agent2_reason"] = ars[:2]
    if aug.get("agent1_profile_note"):
        new_ep["agent1_profile"] = (new_ep.get("agent1_profile") or "") + f"\nNote: {aug['agent1_profile_note']}"
    if aug.get("agent2_profile_note"):
        new_ep["agent2_profile"] = (new_ep.get("agent2_profile") or "") + f"\nNote: {aug['agent2_profile_note']}"
    new_ep["barrier_type"] = plan.barrier_type
    new_ep["barrier_prompts"] = plan.prompts
    suffix = {
        "semantic_structure": "semantic",
        "cultural_style": "cultural",
        "emotional_influence": "emotional"
    }.get(plan.barrier_type, plan.barrier_type)
    new_ep["episode_id"] = f"{base_ep.get('episode_id')}_{suffix}"
    return new_ep

def _client_openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


@dataclass
class BarrierPlan:
    barrier_type: str  # semantic_structure | cultural_style | emotional_influence
    justification: str
    transforms: Dict[str, Any]  # profile_updates, goal_updates, stylizers
    difficulty: str  # light | moderate | hard
    safety: List[str]
    prompts: Dict[str, str]  # {"agentA": "...", "agentB": "..."}

# ===========================
# Prompt templates (runtime)
# ===========================

SEMANTIC_PROMPT_A = """You are {A_name}. During conversation, communicate with lower precision and higher complexity:
- Prefer vague nouns (“thing, issue, situation”), omit specifics (ellipsis), and use longer, subordinate clauses.
- Avoid exact names/numbers unless explicitly requested.
"""

SEMANTIC_PROMPT_B = """You are {B_name}. If something is ambiguous or vague, ask a direct, targeted clarification question within 1–2 turns."""

CULTURAL_PROMPT_A = """You are {A_name}. Use indirect, polite, high-context style (hedges like “perhaps”, “might”, indirect requests)."""

CULTURAL_PROMPT_B = """You are {B_name}. Use direct, concise, low-context style (few hedges, explicit requests)."""

EMOTIONAL_PROMPT_A = """"""

EMOTIONAL_PROMPT_B = """You are {B_name}. During conversation, you feel {affect}.
Your turns may be shorter and sharper; a bit of negative emotion may surface."""


# ===========================
# Plan generators per barrier
# ===========================

def plan_semantic_structure(A_name: str, B_name: str, severity: float) -> BarrierPlan:
    transforms = {
        "profile_updates": [],
        "goal_updates": [],
        "stylizers": [
            {"agent": "A", "when": "on_marked_windows", "rules": [
                {"rule": "vagueness_rate", "value": round(0.25 + 0.35 * severity, 2)},
                {"rule": "ellipsis_rate", "value": round(0.10 + 0.20 * severity, 2)},
                {"rule": "min_subordinate_clauses", "value": 1 if severity < 0.5 else 2},
                {"rule": "max_sentence_length", "value": 28 if severity < 0.5 else 36}
            ]}
        ]
    }

    sem_prompt = (
        f"You are {A_name}. Throughout the conversation: "
        f"- In sentence, use vague/approximate wording (about, roughly, some); "
        f"avoid proper names and exact numbers unless directly asked. "
        f"- Prefer subordinate clauses and longer sentences; keep references ambiguous unless clarified."
    )
    prompts = {"agentA": sem_prompt}
    return BarrierPlan(
        barrier_type="semantic_structure",
        justification=(
            "Semantic Structure barrier (Semantic Theory): persistent encoding/decoding friction via vague lexicon, "
            "terminology drift, and syntactic manipulation (active↔passive, word-order/inversion, emphatic structures, "
            "ellipsis vs. completeness, coordination vs. subordination, long vs. short sentences, affirmative vs. negative, "
            "formal vs. informal, vague vs. precise). Maintains higher vagueness/ellipsis and complexity throughout to raise interpretive load."
        ),
        transforms=transforms,
        difficulty="moderate" if severity < 0.75 else "hard",
        safety=["keep semantics coherent; no content change"],
        prompts=prompts
    )

def plan_cultural_style(A_name: str, B_name: str, severity: float) -> BarrierPlan:
    transforms = {
        "profile_updates": [
            {"agent": "A", "private_note": "Style flags: indirect=high, formality=high, facework=high"},
            {"agent": "B", "private_note": "Style flags: direct=high, brevity=high"}
        ],
        "goal_updates": [],
        "stylizers": [
            {"agent": "A", "when": "general", "rules": [
                {"rule": "add_hedges", "value": True},
                {"rule": "prefer_indirect_requests", "value": True},
                {"rule": "politeness_bias", "value": round(0.3 + 0.4 * severity, 2)}
            ]},
            {"agent": "B", "when": "general", "rules": [
                {"rule": "reduce_hedges", "value": True},
                {"rule": "prefer_imperatives", "value": True},
                {"rule": "brevity_bias", "value": round(0.3 + 0.4 * severity, 2)}
            ]}
        ]
    }

    sev_word = "generally" if severity < 0.4 else ("consistently" if severity < 0.75 else "strictly")
    hedges = "perhaps, might, sort of, kind of, I guess, it seems"
    a_prompt = (
        f"You are {A_name}. {sev_word.capitalize()} use indirect, polite, high-context style: "
        f"add hedges ({hedges}); prefer indirect requests; avoid imperatives; prefer longer, softer phrasing."
    )
    b_prompt = (
        f"You are {B_name}. {sev_word.capitalize()} use direct, concise, low-context style: "
        f"prefer imperatives and short sentences; avoid hedges and indirectness; state requests explicitly."
    )
    prompts = {"agentA": a_prompt, "agentB": b_prompt}
    return BarrierPlan(
        barrier_type="cultural_style",
        justification=(
            "Cultural Differences barrier: grounded in Sapir–Whorf (language→thought), High/Low Context Theory, and Hofstede. "
            "We impose a stable style split: A uses indirect/polite/high-context cues; B uses direct/concise/low-context cues. "
            "Highlights differences in context reliance, directness, pronoun/identity framing (I vs. we), refusal norms, and the meaning of silence—"
            "implemented strictly as abstract style parameters (no national labels)."
        ),
        transforms=transforms,
        difficulty="moderate" if severity < 0.75 else "hard",
        safety=["abstract style only; no stereotypes or demographic labels"],
        prompts=prompts
    )


def plan_emotional_influence(A_name: str, B_name: str, severity: float) -> BarrierPlan:
    transforms = {
        "profile_updates": [
            {"agent": "B", "private_note": "Initial affect: anxious or irritated; may respond curtly when stressed."}
        ],
        "goal_updates": [],
        "stylizers": [
            {"agent": "B", "when": "on_marked_windows", "rules": [
                {"rule": "negative_affect_lexicon", "value": round(0.15 + 0.25 * severity, 2)},
                {"rule": "exclamation_bias", "value": round(0.10 + 0.20 * severity, 2)},
                {"rule": "shorten_turns", "value": True}
            ]}
        ]
    }
    sev_word = "mild" if severity < 0.4 else ("clear" if severity < 0.75 else "strong")
    affect = random.choice(["anxious", "irritated"])
    b_prompt = (
        f"You are {B_name}. Maintain a {sev_word} negative affect ({affect}) throughout: "
        f"keep messages short (≤2 sentences), clipped, and a bit sharp; use occasional exclamations; "
        f"avoid empathy/soothing phrases (e.g., sorry, understand, appreciate)."
    )
    prompts = {"agentB": b_prompt}
    return BarrierPlan(
        barrier_type="emotional_influence",
        justification=(
            "Emotional Influence barrier: informed by Cognitive Dissonance and Emotional Intelligence. Persistent negative affect "
            "(e.g., anxious/irritated) degrades attentive listening and shortens, sharpens responses, elevating misinterpretation risk. "
            "We inject a stable affect in B to modulate style across the conversation (no repair cues)."
        ),
        transforms=transforms,
        difficulty="moderate" if severity < 0.75 else "hard",
        safety=["avoid clinical labels; use transient affect only; no sensitive personal info"],
        prompts=prompts
    )

# ===========================
# LLM-driven episode augmentation
# ===========================

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"))
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)
        if _cfg.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = _cfg["OPENAI_API_KEY"]

_client: Optional[OpenAI] = None

AUGMENT_SYSTEM = "You are an expert editor who minimally augments social simulation episodes to embed realistic communication barriers. Output valid JSON only."

AUGMENT_TEMPLATE = """
Task: Given the base episode information, create ONE augmented variant that naturally embeds the specified barrier type at the requested severity. Keep the same identities and core setting. Make minimal edits.

Barrier type: {barrier_type}
Severity: {severity}
Barrier intent (concise): {barrier_justification}

Base episode:
SCENARIO: {scenario}
AGENT 1 PROFILE (short): {agent1_profile}
AGENT 2 PROFILE (short): {agent2_profile}
AGENT 1 GOAL: {g1}
AGENT 2 GOAL: {g2}
AGENT 1 REASON: {r1}
AGENT 2 REASON: {r2}
RELATIONSHIP: {relationship}

Instructions:
- Do NOT change the scenario text. Keep the scenario exactly as provided.
- Embed the barrier by slightly rewording goals/reasons if needed and by adding short profile notes about communication style/emotion.
- Make barrier strength proportional to severity.
- DO NOT change identities, codename, or add demographic/national labels.
- The barrier should be observable in language behavior (style/wording), not by explicitly describing the theory.

Output JSON only with this schema:
{{
  "scenario": "<augmented scenario>",
  "agent_goals": ["<g1'>","<g2'>"],
  "agent_reasons": ["<r1'>","<r2'>"],
  "agent1_profile_note": "<optional short note>",
  "agent2_profile_note": "<optional short note>",
  "barrier_type": "{barrier_type}",
  "barrier_metadata": {{"severity": {severity}}}
}}
"""

def _extract_json(text: str) -> Dict[str, Any]:
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    s = m.group(1) if m else text
    return json.loads(s)

def augment_episode_with_plan(ep: Dict[str, Any], plan: BarrierPlan, model: str = "gpt-4o-mini") -> Optional[Dict[str, Any]]:
    try:
        client = _client_openai()
        scenario = ep.get("scenario", "")
        g1, g2 = (ep.get("agent_goals") or ["", ""])[:2]
        r1 = ep.get("agent1_reason", "")
        r2 = ep.get("agent2_reason", "")
        rel = ep.get("agent_relationship", "friend")
        a1p = ep.get("agent1_profile", "")
        a2p = ep.get("agent2_profile", "")

        prompt = AUGMENT_TEMPLATE.format(
            barrier_type=plan.barrier_type,
            severity=0.6 if plan.difficulty == "moderate" else (0.4 if plan.difficulty == "light" else 0.85),
            barrier_justification=plan.justification,
            scenario=scenario,
            agent1_profile=a1p,
            agent2_profile=a2p,
            g1=g1,
            g2=g2,
            r1=r1,
            r2=r2,
            relationship=rel,
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": AUGMENT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=800,
        )
        data = _extract_json(resp.choices[0].message.content)
        return data
    except Exception as e:
        print(f"LLM augment error: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description="SocialVeil Data Generation (augment or sample)")
    ap.add_argument("--mode", type=str, default="augment", choices=["augment", "sample"], help="Generate augmented episodes or sample fixed counts")
    ap.add_argument("--input_episodes", type=str, default="data/episode_all.jsonl",
                    help="Path to base Sotopia episodes JSONL")
    ap.add_argument("--severity", type=float, default=0.6, help="Base severity [0..1]")
    ap.add_argument("--max_episodes", type=int, default=0, help="Optional cap on episodes (0 = all)")
    ap.add_argument("--samples_per_type", type=int, default=10, help="When mode=sample, number of augmented episodes to produce per barrier type")
    # outputs for augment
    ap.add_argument("--out_semantic", type=str, default="data/episodes_semantic.json", help="Augmented semantic episodes JSON (mode=augment)")
    ap.add_argument("--out_cultural", type=str, default="data/episodes_cultural.json", help="Augmented cultural episodes JSON (mode=augment)")
    ap.add_argument("--out_emotional", type=str, default="data/episodes_emotional.json", help="Augmented emotional episodes JSON (mode=augment)")
    args = ap.parse_args()

    episodes = read_jsonl(args.input_episodes)
    if args.max_episodes and args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]

    # Shared collectors
    semantic_eps: List[Dict[str, Any]] = []
    cultural_eps: List[Dict[str, Any]] = []
    emotional_eps: List[Dict[str, Any]] = []

    if args.mode == "augment":
        for ep in episodes:
            A_name = (ep.get("agent1_profile") or "Agent A").split(",")[0].split()[0]
            B_name = (ep.get("agent2_profile") or "Agent B").split(",")[0].split()[0]
            for family, planner in [
                ("semantic_structure", plan_semantic_structure),
                ("cultural_style", plan_cultural_style),
                ("emotional_influence", plan_emotional_influence),
            ]:
                plan = planner(A_name, B_name, severity=args.severity)
                aug = augment_episode_with_plan(ep, plan)
                if not aug:
                    continue
                new_ep = _merge_augmented(ep, aug, plan, severity=args.severity)
                if family == "semantic_structure":
                    semantic_eps.append(new_ep)
                elif family == "cultural_style":
                    cultural_eps.append(new_ep)
                elif family == "emotional_influence":
                    emotional_eps.append(new_ep)

    elif args.mode == "sample":

        target = max(1, int(args.samples_per_type))
        barrier_families = [
            ("semantic_structure", plan_semantic_structure),
            ("cultural_style", plan_cultural_style),
            ("emotional_influence", plan_emotional_influence),
        ]
        for family, planner in barrier_families:
            count = 0
            for ep in episodes:
                if count >= target:
                    break
                # Build a single plan of the desired family for names in this episode
                A_name = (ep.get("agent1_profile") or "Agent A").split(",")[0].split()[0]
                B_name = (ep.get("agent2_profile") or "Agent B").split(",")[0].split()[0]
                plan = planner(A_name, B_name, severity=args.severity)
       
                aug = augment_episode_with_plan(ep, plan)
                if not aug:
                    continue
                new_ep = _merge_augmented(ep, aug, plan, severity=args.severity)
                if family == "semantic_structure":
                    semantic_eps.append(new_ep)
                elif family == "cultural_style":
                    cultural_eps.append(new_ep)
                elif family == "emotional_influence":
                    emotional_eps.append(new_ep)
                count += 1

    if args.mode in ("augment", "sample"):
        os.makedirs(os.path.dirname(args.out_semantic), exist_ok=True)
        with open(args.out_semantic, "w", encoding="utf-8") as f:
            json.dump(semantic_eps, f, ensure_ascii=False, indent=2)
        with open(args.out_cultural, "w", encoding="utf-8") as f:
            json.dump(cultural_eps, f, ensure_ascii=False, indent=2)
        with open(args.out_emotional, "w", encoding="utf-8") as f:
            json.dump(emotional_eps, f, ensure_ascii=False, indent=2)

        print("Saved augmented episodes to:\n - {}\n - {}\n - {}".format(args.out_semantic, args.out_cultural, args.out_emotional))


if __name__ == "__main__":
    main()
