import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Tuple, Set

import yaml
from openai import OpenAI


def _strip_parentheticals(text: str) -> str:
    try:
        return re.sub(r"\([^)]*\)", "", text)
    except Exception:
        return text


def _genericize_venues(text: str) -> str:
    try:
        replacements = {
            "Target": "a store",
            "Walmart": "a store",
            "IKEA": "a store",
            "Starbucks": "a cafe",
            "Craigslist": "an online listing",
            "Facebook Marketplace": "an online listing",
            "charity dinner": "a community event",
            "Charity dinner": "a community event",
            "charity": "a community cause",
        }
        s = text
        for k, v in replacements.items():
            s = re.sub(rf"\b{k}\b", v, s, flags=re.IGNORECASE)
        return s
    except Exception:
        return text


def _remove_role_clauses(text: str) -> str:
    try:
        s = text
        # Remove/neutralize clauses that reveal roles/responsibilities
        patterns_replacements: List[Tuple[str, str]] = [
            (r"\bone of them[^\.,;]*", "they cross paths while an activity is ongoing"),
            (r"\bis in charge of[^\.,;]*", ""),
            (r"\bis responsible for[^\.,;]*", ""),
            (r"\bassigned to[^\.,;]*", ""),
            (r"\bvolunteer(s|ed)?\b[^\.,;]*", ""),
            (r"\bdesignated\s+(buyer|seller)\b", "person"),
            (r"\bbuyer\b", "person"),
            (r"\bseller\b", "person"),
            (r"\brepresentative\b", "person"),
            (r"\bmanager\b", "organizer"),
        ]
        for pat, repl in patterns_replacements:
            s = re.sub(pat, repl, s, flags=re.IGNORECASE)
        # Remove excessive separators left by deletions
        s = re.sub(r"\s+,\s+", ", ", s)
        return s
    except Exception:
        return text


def _remove_amounts(text: str) -> str:
    try:
        s = text
        # Replace explicit monetary figures and percentages with generic phrasing
        s = re.sub(r"\$\s*\d+(?:[\.,]\d+)?", "some amount", s)
        s = re.sub(r"\b\d+(?:[\.,]\d+)?\s*(?:dollars|bucks|USD)\b", "some amount", s, flags=re.IGNORECASE)
        s = re.sub(r"\b\d+(?:[\.,]\d+)?\s*(?:%|percent)\b", "a certain share", s, flags=re.IGNORECASE)
        return s
    except Exception:
        return text


def _tidy(text: str) -> str:
    try:
        s = re.sub(r"\s+", " ", text).strip()
        s = re.sub(r",\s*$", "", s)
        # Collapse to one concise sentence if it's overly long with many commas
        if len(s) > 220:
            # Keep the first clause up to ~180 chars
            s = s[:180].rsplit(",", 1)[0].strip()
        return s
    except Exception:
        return text


def neutralize_scenario_text(text: str) -> str:
    s = text or ""
    s = _strip_parentheticals(s)
    s = _genericize_venues(s)
    s = _remove_role_clauses(s)
    s = _remove_amounts(s)
    s = _tidy(s)
    return s


_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        # Load key from config if present
        try:
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"))
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    if isinstance(cfg, dict) and cfg.get("OPENAI_API_KEY"):
                        os.environ["OPENAI_API_KEY"] = cfg["OPENAI_API_KEY"]
        except Exception:
            pass
        _openai_client = OpenAI()
    return _openai_client


def llm_neutralize_scenario(
    text: str,
    model: str = "gpt-4o-mini",
    seed: int | None = None,
) -> str:
    prompt = (
        "Rewrite the following scenario into ONE short sentence that preserves the general setting, "
        "but removes or obscures explicit roles/goals/privileged information of either party. "
        "Avoid revealing who is in charge, target prices, amounts, or concrete responsibilities. "
        "Keep it natural and specific enough to set context, but ambiguous about who does what.\n\n"
        f"Scenario: {text}"
    )
    try:
        client = _get_openai_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You concisely generalize scenarios to be context-preserving but role-ambiguous."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=120,
            **({"seed": seed} if isinstance(seed, int) and seed != 0 else {}),
        )
        content = (resp.choices[0].message.content or "").strip()
        content = re.sub(r"\s+", " ", content)
        return content or text
    except Exception:
        # Fallback to heuristics
        return neutralize_scenario_text(text)


def build_neutralization_map(
    scenarios: Iterable[str],
    method: str,
    model: str,
    seed: int | None,
    cache_path: str | None,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    # Load existing cache if provided
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                if isinstance(cached, dict):
                    mapping.update({str(k): str(v) for k, v in cached.items()})
        except Exception:
            pass

    unique_scenarios: List[str] = []
    seen: Set[str] = set()
    for s in scenarios:
        if s not in seen:
            seen.add(s)
            unique_scenarios.append(s)

    for s in unique_scenarios:
        if s in mapping:
            continue
        if method == "llm":
            mapping[s] = llm_neutralize_scenario(s, model=model, seed=seed)
        else:
            mapping[s] = neutralize_scenario_text(s)

    # Persist updated cache
    if cache_path:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return mapping


def _is_probably_jsonl(path: str) -> bool:
    return path.lower().endswith(".jsonl")


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _read_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("Expected a JSON array at root")


def process_episodes(
    episodes: Iterable[Dict[str, Any]],
    preserve_detailed: bool,
    method: str,
    model: str,
    seed: int | None,
    cache_path: str | None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    # Build mapping for unique scenarios to ensure consistency across duplicates
    originals = [str((ep or {}).get("scenario", "")) for ep in episodes]
    mapping = build_neutralization_map(originals, method, model, seed, cache_path)
    for ep in episodes:
        ep_new = json.loads(json.dumps(ep, ensure_ascii=False))
        original = str(ep_new.get("scenario", ""))
        if preserve_detailed and original and "scenario_detailed" not in ep_new:
            ep_new["scenario_detailed"] = original
        ep_new["scenario"] = mapping.get(original, original)
        out.append(ep_new)
    return out


def main():
    ap = argparse.ArgumentParser(description="Neutralize scenario text to reduce role leakage.")
    ap.add_argument("--input", required=True, help="Path to episodes .json or .jsonl")
    ap.add_argument("--output", help="Path to write updated episodes (json/jsonl). If omitted with --inplace, overwrites input.")
    ap.add_argument("--inplace", action="store_true", help="Overwrite the input file in place (requires write permission).")
    ap.add_argument("--preserve_detailed", action="store_true", help="Copy original scenario to scenario_detailed.")
    ap.add_argument("--method", choices=["llm", "heuristic"], default="llm", help="Neutralization method: LLM or heuristic.")
    ap.add_argument("--model", default="gpt-4o-mini", help="LLM model for rewriting (when --method llm).")
    ap.add_argument("--seed", type=int, default=0, help="Optional LLM seed for determinism (0 = unset).")
    ap.add_argument("--cache", default="data/scenario_neutralization_cache.json", help="Path to cache mapping original->neutralized.")
    args = ap.parse_args()

    src = args.input
    dst = args.output
    if args.inplace and not dst:
        dst = src
    if not dst:
        # Default output path next to input
        base, ext = os.path.splitext(src)
        dst = base + "_neutralized" + ext

    changed = 0
    if _is_probably_jsonl(src):
        episodes = list(_iter_jsonl(src))
        result = process_episodes(
            episodes,
            preserve_detailed=args.preserve_detailed,
            method=args.method,
            model=args.model,
            seed=args.seed,
            cache_path=args.cache,
        )
        with open(dst, "w", encoding="utf-8") as f:
            for ep in result:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
        changed = len(result)
    else:
        episodes = _read_json_array(src)
        result = process_episodes(
            episodes,
            preserve_detailed=args.preserve_detailed,
            method=args.method,
            model=args.model,
            seed=args.seed,
            cache_path=args.cache,
        )
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        changed = len(result)

    print(f"Neutralized scenarios for {changed} episode(s) -> {dst}")


if __name__ == "__main__":
    main()

