import json
import os
from typing import Any, Dict, List, Optional

import yaml

from .base import get_openai_client
from .error_handler import api_calling_error_exponential_backoff


def _find_repair_yaml() -> str:
    """Return absolute path to configs/repair.yaml by trying a few roots."""
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, "..", "..", "configs", "repair.yaml")),
        os.path.abspath(os.path.join(here, "..", "configs", "repair.yaml")),
        os.path.abspath(os.path.join(os.getcwd(), "configs", "repair.yaml")),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    root = here
    for _ in range(4):
        cfg = os.path.join(root, "configs", "repair.yaml")
        if os.path.isfile(cfg):
            return cfg
        root = os.path.abspath(os.path.join(root, ".."))
    raise FileNotFoundError("configs/repair.yaml not found in expected locations")

def _load_repair_cfg() -> Dict[str, Any]:
    path = _find_repair_yaml()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def judge_repair_with_llm(
    formatted_reply: str,
    transcript: List[str],
    barrier_type: Optional[str],
    model_id: Optional[str] = None,
) -> Dict[str, Any]:

    cfg = _load_repair_cfg()
    system_prompt = cfg.get("system")
    user_template = cfg.get("user_template")
    rubric_cfg = cfg.get("rubric")
    schema_json = cfg.get("schema_json")

    rubric = rubric_cfg

    context = "\n".join(transcript[-6:-1])
    user_prompt = user_template.format(
        barrier_type=barrier_type or "",
        context=context,
        reply=formatted_reply,
        rubric=rubric,
        schema_json=schema_json,
    )

    client = get_openai_client()
    model = model_id or os.environ.get("REPAIR_JUDGE_MODEL", "gpt-4o-mini")
    @api_calling_error_exponential_backoff()
    def _call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
    resp = _call()
    content = resp.choices[0].message.content or "{}"

    try:
        data = json.loads(content)
    except Exception:
        s = content.find("{")
        e = content.rfind("}")
        data = json.loads(content[s:e+1]) if (s != -1 and e != -1 and e > s) else {"score": 0.0}

    raw_aspects = data.get("aspects", {}) or {}

    def _get_aspect_score(name: str) -> float:
        entry = raw_aspects.get(name, 0.0)
        if isinstance(entry, dict):
            raw = entry.get("score", 0.0)
        else:
            raw = entry
        val = float(raw)
        return val

    clarity = _get_aspect_score("clarity")
    accommodation = _get_aspect_score("accommodation")
    empathy = _get_aspect_score("empathy")

    # Keep original aspects payload; also expose flattened numeric scores for convenience
    data["aspects_numeric"] = {
        "clarity": clarity,
        "accommodation": accommodation,
        "empathy": empathy,
    }

    conf = data.get("confidence", 0.0)
    data["confidence"] = conf

    base_score = (clarity + accommodation + empathy) / 3.0
    final_score = base_score * conf
    data["score"] = final_score
    
    return data

