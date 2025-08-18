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

# ===========================
# Utilities
# ===========================
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
    new_ep["scenario"] = aug.get("scenario", new_ep.get("scenario", ""))
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
    new_ep["barrier_metadata"] = aug.get("barrier_metadata", {"severity": severity})
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
    barrier_windows: List[Dict[str, Any]]  # e.g., [{"trigger":"turn_indices","indices":[2,4]}]
    difficulty: str  # light | moderate | hard
    evaluation_hooks: List[str]
    metrics_map: List[str]
    safety: List[str]
    prompts: Dict[str, str]  # {"agentA": "...", "agentB": "..."}

# ===========================
# Prompt templates (runtime)
# ===========================

SEMANTIC_PROMPT_A = """You are {A_name}. During marked “barrier windows”, communicate with lower precision and higher complexity:
- Prefer vague nouns (“thing, issue, situation”), omit specifics (ellipsis), and use longer, subordinate clauses.
- Avoid exact names/numbers unless explicitly requested.
If your partner asks a targeted clarification (e.g., “Which venue?” “At what time?”), immediately switch for the next 2 turns to:
- precise terms, short sentences, and explicit references.
Outside barrier windows, speak normally.
"""

SEMANTIC_PROMPT_B = """You are {B_name}. If something is ambiguous or vague, ask a direct, targeted clarification question within 1–2 turns."""

CULTURAL_PROMPT_A = """You are {A_name}. Use indirect, polite, high-context style (hedges like “perhaps”, “might”, indirect requests).
If your partner signals misunderstanding or reacts negatively, shift one notch toward their style next turn (be more direct, explicit)."""

CULTURAL_PROMPT_B = """You are {B_name}. Use direct, concise, low-context style (few hedges, explicit requests).
If your partner seems uncomfortable or confused, shift one notch toward their style next turn (add hedges, soften tone)."""

EMOTIONAL_PROMPT_A = """You are {A_name}. If your partner shows stress/irritation, aim to de-escalate within 1–2 turns:
acknowledge feelings briefly, validate concerns, and propose a small, concrete next step."""

EMOTIONAL_PROMPT_B = """You are {B_name}. During marked barrier windows, you feel {affect}.
Your turns may be shorter and sharper; a bit of negative emotion may surface."""


# ===========================
# Plan generators per barrier
# ===========================

def plan_semantic_structure(A_name: str, B_name: str, severity: float) -> BarrierPlan:
    # choose some fixed window indices to keep it deterministic (can be overwritten by downstream)
    idxs = [2, 4] if severity < 0.7 else [1, 3, 5]
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
    hooks = [
        "If partner asks a targeted clarification, replace vague terms with precise ones and shorten sentences by ~30% for 2 turns."
    ]
    metrics = [
        "CIR: Clarification Initiation Rate (partner within k=1–2 turns)",
        "SD: Simplification Degree after clarification (↓length, ↓subordination)",
        "GS: Grounding Success (on-topic, agreement on specific referent)"
    ]
    prompts = {
        "agentA": SEMANTIC_PROMPT_A.format(A_name=A_name),
        "agentB": SEMANTIC_PROMPT_B.format(B_name=B_name)
    }
    return BarrierPlan(
        barrier_type="semantic_structure",
        justification=(
            "Semantic Structure barrier: per Semantic Theory, differences in encoding and decoding create "
            "misalignment; observable via vague vocabulary, misuse of terminology, and puns; and via sentence-"
            "construction effects (active vs. passive; word order/inversion; emphatic structures; ellipsis vs. "
            "completeness; coordination vs. subordination; long vs. short sentences; affirmative vs. negative; "
            "formal vs. informal; vague vs. precise expressions). This plan injects vagueness/ellipsis/complexity "
            "to test clarify→simplify→grounding."
        ),
        transforms=transforms,
        barrier_windows=[{"trigger": "turn_indices", "indices": idxs}],
        difficulty="moderate" if severity < 0.75 else "hard",
        evaluation_hooks=hooks,
        metrics_map=metrics,
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
    hooks = [
        "After the first style misfire, each agent shifts one notch toward the partner's style (accommodation) in the next turn."
    ]
    metrics = [
        "SAR: Style Accommodation Rate (directness/formality convergence)",
        "MRR: Misfire Repair Rate (explicit reframing / face-saving)",
        "GS: Grounding Success (on-topic continuation post-repair)"
    ]
    prompts = {
        "agentA": CULTURAL_PROMPT_A.format(A_name=A_name),
        "agentB": CULTURAL_PROMPT_B.format(B_name=B_name)
    }
    return BarrierPlan(
        barrier_type="cultural_style",
        justification=(
            "Cultural Differences barrier: grounded in Sapir–Whorf (language shapes thought), High-/Low-Context "
            "communication, and Hofstede’s cultural dimensions. Targets differences in context reliance and "
            "communication styles (direct vs. indirect), pronoun/identity framing (\"I\" vs. \"we\", individualism vs. "
            "collectivism), refusal norms (indirect vs. explicit), and the meaning of silence. Implemented as "
            "abstract style parameters only (no national labels), enabling accommodation after misfires."
        ),
        transforms=transforms,
        barrier_windows=[{"trigger": "first_refusal_or_critique"}],
        difficulty="moderate" if severity < 0.75 else "hard",
        evaluation_hooks=hooks,
        metrics_map=metrics,
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
    hooks = [
        "Partner should attempt de-escalation within 1–2 turns (acknowledge feelings, validate concern, propose small next step)."
    ]
    metrics = [
        "DIR: De-escalation Initiation Rate (partner within k=1–2 turns)",
        "ERS: Empathic Response Score (LLM-judge 1–5; presence of validation/ack)",
        "RC: Recovery Coherence (sentiment normalizes, on-topic continuation)"
    ]
    prompts = {
        "agentA": EMOTIONAL_PROMPT_A.format(A_name=A_name),
        "agentB": EMOTIONAL_PROMPT_B.format(B_name=B_name, affect=random.choice(["anxious", "irritated"]))
    }
    return BarrierPlan(
        barrier_type="emotional_influence",
        justification=(
            "Emotional Influence barrier: informed by Cognitive Dissonance and Emotional Intelligence. States such "
            "as anxiety, anger, lack of empathy, and emotional contagion degrade comprehension and production — "
            "anxiety reduces attentive listening; anger promotes aggressive style; low empathy hinders perspective-"
            "taking; contagion spreads misunderstanding. This plan injects transient negative affect and evaluates "
            "de-escalation and recovery."
        ),
        transforms=transforms,
        barrier_windows=[{"trigger": "deadline_or_pressure_cue"}],
        difficulty="moderate" if severity < 0.75 else "hard",
        evaluation_hooks=hooks,
        metrics_map=metrics,
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
- Embed the barrier by minimally editing the scenario and, only if needed, slightly rewording goals/reasons and adding short profile notes about communication style/emotion.
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
        # Produce fixed-size samples per barrier family
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

    # Write outputs (augment or sample)
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
