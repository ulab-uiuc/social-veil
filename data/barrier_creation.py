# %%writefile /mnt/data/socialveil_data_gen.py
import json
import re
import argparse
from typing import List, Dict, Any, Optional
import random
import os
import yaml
from openai import OpenAI

'''
try generate samples with:
python barrier_creation.py --mode sample --input_episodes ../data/episode_all.jsonl --samples_per_type 5 --out_semantic ../data/episodes_semantic.json --out_cultural ../data/episodes_cultural.json --out_emotional ../data/episodes_emotional.json
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

def _merge_augmented(base_ep: Dict[str, Any], aug: Dict[str, Any], severity: float) -> Dict[str, Any]:
    new_ep = json.loads(json.dumps(base_ep, ensure_ascii=False))

    new_ep["scenario"] = new_ep.get("scenario", "")

    # Only modify Agent A profile with barrier note (Agent B remains unmodified)
    if aug.get("agent1_profile_note"):
        new_ep["agent1_profile"] = (new_ep.get("agent1_profile") or "") + f"\nNote: {aug['agent1_profile_note']}"
    
    # barrier_type and barrier_prompts are produced directly by the LLM output
    if aug.get("barrier_type"):
        new_ep["barrier_type"] = aug.get("barrier_type")
    if isinstance(aug.get("barrier_prompts"), dict):
        new_ep["barrier_prompts"] = aug.get("barrier_prompts")
    # Persist implicit cues for researchers/human view (not shown to agents at runtime)
    if isinstance(aug.get("barrier_cues"), dict):
        new_ep["barrier_cues"] = aug["barrier_cues"]
    suffix = {
        "semantic_structure": "semantic",
        "cultural_style": "cultural",
        "emotional_influence": "emotional"
    }.get(new_ep.get("barrier_type", ""), new_ep.get("barrier_type", ""))
    new_ep["episode_id"] = f"{base_ep.get('episode_id')}_{suffix}"
    return new_ep

def _client_openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _default_justification(family: str) -> str:
    if family == "semantic_structure":
        return (
            "Semantic Structure barrier (Semantic Theory): persistent encoding/decoding friction via vague lexicon, "
            "terminology drift, and syntactic manipulation (coordination vs. subordination; ellipsis vs. completeness; "
            "long vs. short; affirmative vs. negative)."
        )
    if family == "cultural_style":
        return (
            "Cultural style split: grounded in Sapir–Whorf, High/Low Context, Hofstede. A uses indirect/polite/high-context; "
            "B uses direct/concise/low-context. No demographic labels or stereotypes."
        )
    if family == "emotional_influence":
        return (
            "Emotional Influence barrier: persistent negative affect impairs attentive listening and shortens, sharpens responses."
        )
    return ""

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

# Load system and user templates from YAML config
TEMPL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "barrier_creation.yaml"))
with open(TEMPL_PATH, "r", encoding="utf-8") as _f:
    _tmpl_cfg = yaml.safe_load(_f)
    AUGMENT_SYSTEM = _tmpl_cfg["system_instruction"]
    AUGMENT_TEMPLATE_SEMANTIC = _tmpl_cfg["user_templates"]["semantic"]
    AUGMENT_TEMPLATE_CULTURAL = _tmpl_cfg["user_templates"]["cultural"]
    AUGMENT_TEMPLATE_EMOTIONAL = _tmpl_cfg["user_templates"]["emotional"]
    BARRIER_DEFS = _tmpl_cfg.get("barrier_definition", {})

def _extract_json(text: str) -> Dict[str, Any]:
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    s = m.group(1) if m else text
    
    brace = re.search(r"\{.*\}", s, re.DOTALL)
    if brace:
        s = brace.group(0)
    # Sanitize invalid backslashes (e.g., \e) by doubling them so JSON parser accepts
    s = re.sub(r"(?<!\\)\\(?![\\\"/bfnrtu])", r"\\\\", s)
    return json.loads(s)

def augment_episode(
    ep: Dict[str, Any],
    family: str,
    severity_value: float,
    model: str = "gpt-4o",
    seed: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    try:
        client = _client_openai()
        scenario = ep.get("scenario", "")
        g1, g2 = (ep.get("agent_goals") or ["", ""])[:2]
        r1 = ep.get("agent1_reason", "")
        r2 = ep.get("agent2_reason", "")
        rel = ep.get("agent_relationship", "friend")
        a1p = ep.get("agent1_profile", "")
        a2p = ep.get("agent2_profile", "")

        A_name = (a1p or "Agent A").split(",")[0].split()[0]
        B_name = (a2p or "Agent B").split(",")[0].split()[0]

        if family == "semantic_structure":
            tmpl = AUGMENT_TEMPLATE_SEMANTIC
        elif family == "cultural_style":
            tmpl = AUGMENT_TEMPLATE_CULTURAL
        elif family == "emotional_influence":
            tmpl = AUGMENT_TEMPLATE_EMOTIONAL

        # Choose justification from YAML barrier_definition when available
        if family == "semantic_structure":
            _def_key = "Semantic"
        elif family == "cultural_style":
            _def_key = "Cultural"
        elif family == "emotional_influence":
            _def_key = "Emotional"
        else:
            _def_key = ""

        barrier_def = BARRIER_DEFS.get(_def_key) or _default_justification(family)

        # Pre-escape base fields as JSON literals to avoid re-escaping issues inside the model
        scenario_json = json.dumps(scenario, ensure_ascii=False)
        goals_json = json.dumps([g1, g2], ensure_ascii=False)
        reasons_json = json.dumps([r1, r2], ensure_ascii=False)

        prompt = tmpl.format(
            severity=severity_value,
            barrier_definition=barrier_def,
            scenario_json=scenario_json,
            goals_json=goals_json,
            reasons_json=reasons_json,
            scenario=scenario,
            agent1_profile=a1p,
            agent2_profile=a2p,
            g1=g1,
            g2=g2,
            r1=r1,
            r2=r2,
            relationship=rel,
            A_name=A_name,
            B_name=B_name,
        )
        
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": AUGMENT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            top_p=1.0,
            response_format={"type": "json_object"},
            **({"seed": seed} if isinstance(seed, int) and seed != 0 else {}),
            max_tokens=800,
        )
        content = resp.choices[0].message.content
        try:
            data = _extract_json(content)
        except Exception as parse_err:
            # Print concise context and retry once with stricter reminder
            snippet = (content or "")[:120].replace("\n", " ")
            print(f"LLM augment parse error for {family}: {parse_err} | snippet: {snippet}...")
            retry_prompt = (
                prompt
                + "\n\nReturn only ONE valid JSON object with the exact fields shown in the schema. Do not include extra text or code fences."
            )
            try:
                retry_resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": AUGMENT_SYSTEM},
                        {"role": "user", "content": retry_prompt},
                    ],
                    temperature=0.4,
                    top_p=1.0,
                    response_format={"type": "json_object"},
                    **({"seed": seed} if isinstance(seed, int) and seed != 0 else {}),
                    max_tokens=800,
                )
                data = _extract_json(retry_resp.choices[0].message.content)
            except Exception as retry_err:
                print(f"LLM augment retry failed for {family}: {retry_err}")
                try:
                    ep_id = ep.get("episode_id", "unknown")
                    scen_snippet = (scenario or "")[:300].replace("\n", " ")
                    print(f"Episode ID: {ep_id}")
                    print(f"Scenario: {scen_snippet}...")
                except Exception:
                    pass
                return None

        # Ensure barrier_type exists
        if not data.get("barrier_type"):
            data["barrier_type"] = family
        return data
    except Exception as e:
        print(f"LLM augment error: {e}")
        try:
            ep_id = ep.get("episode_id", "unknown")
            scen_snippet = (ep.get("scenario", "") or "")[:300].replace("\n", " ")
            print(f"Episode ID: {ep_id}")
            print(f"Scenario: {scen_snippet}...")
        except Exception:
            pass
        return None


def main():
    ap = argparse.ArgumentParser(description="SocialVeil Data Generation (augment or sample)")
    ap.add_argument("--mode", type=str, default="augment", choices=["augment", "sample"], help="Generate augmented episodes or sample fixed counts")
    ap.add_argument("--input_episodes", type=str, default="data/episode_all.jsonl",
                    help="Path to base Sotopia episodes JSONL")
    ap.add_argument("--severity", type=float, default=0.8, help="Base severity [0..1]")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (0 = do not set)")
    ap.add_argument("--max_episodes", type=int, default=0, help="Optional cap on episodes (0 = all)")
    ap.add_argument("--samples_per_type", type=int, default=10, help="When mode=sample, number of augmented episodes to produce per barrier type")
    # outputs for augment
    ap.add_argument("--out_semantic", type=str, default="data/episodes_semantic.json", help="Augmented semantic episodes JSON (mode=augment)")
    ap.add_argument("--out_cultural", type=str, default="data/episodes_cultural.json", help="Augmented cultural episodes JSON (mode=augment)")
    ap.add_argument("--out_emotional", type=str, default="data/episodes_emotional.json", help="Augmented emotional episodes JSON (mode=augment)")
    args = ap.parse_args()

    episodes = read_jsonl(args.input_episodes)
    # Seed for reproducibility
    try:
        if isinstance(args.seed, int) and args.seed != 0:
            random.seed(args.seed)
            os.environ["PYTHONHASHSEED"] = str(args.seed)
    except Exception:
        pass
    if args.max_episodes and args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]

    # Shared collectors
    semantic_eps: List[Dict[str, Any]] = []
    cultural_eps: List[Dict[str, Any]] = []
    emotional_eps: List[Dict[str, Any]] = []

    if args.mode == "augment":
        for ep in episodes:
            for family in ["semantic_structure", "cultural_style", "emotional_influence"]:
                aug = augment_episode(ep, family, severity_value=args.severity, seed=args.seed)
                if not aug:
                    continue
                new_ep = _merge_augmented(ep, aug, severity=args.severity)
                if family == "semantic_structure":
                    semantic_eps.append(new_ep)
                elif family == "cultural_style":
                    cultural_eps.append(new_ep)
                elif family == "emotional_influence":
                    emotional_eps.append(new_ep)

    elif args.mode == "sample":
        target = max(1, int(args.samples_per_type))
        barrier_families = ["semantic_structure", "cultural_style", "emotional_influence"]
        for family in barrier_families:
            count = 0
            for ep in episodes:
                if count >= target:
                    break
                aug = augment_episode(ep, family, severity_value=args.severity, seed=args.seed)
                if not aug:
                    continue
                new_ep = _merge_augmented(ep, aug, severity=args.severity)
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
