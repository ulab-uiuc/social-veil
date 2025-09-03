import json
import re
from typing import Optional, Tuple


def load_env(env_data_path):
    with open(env_data_path, "r") as f:
        episodes = [json.loads(line) for line in f]

        return episodes

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def parse_mcq_response_text(text: str) -> Tuple[Optional[str], float, str]:
    """Parse MCQ response that may contain plain text, key-value lines, or JSON fragments.

    Returns: (selected_option[A-D] or None, confidence[0..1], reasoning)
    """
    if not isinstance(text, str):
        return None, 0.0, ""
    sel: Optional[str] = None
    conf: float = 0.0
    reas: str = ""
    t = text.strip()

    # 1) Key-value lines (case-insensitive); accepts ':' or '='
    try:
        for raw in t.split("\n"):
            line = raw.strip()
            low = line.lower()
            if low.startswith("selected"):
                parts = re.split(r":|=", line, maxsplit=1)
                if len(parts) == 2:
                    cand = parts[1].strip().strip('"\'')
                    if cand:
                        sel = cand[0].upper()
            elif low.startswith("confidence"):
                parts = re.split(r":|=", line, maxsplit=1)
                if len(parts) == 2:
                    try:
                        conf = float(parts[1].strip().strip(', '))
                    except Exception:
                        pass
            elif low.startswith("reasoning"):
                parts = re.split(r":|=", line, maxsplit=1)
                if len(parts) == 2:
                    reas = parts[1].strip().strip('"\'')
    except Exception:
        pass

    # 2) JSON fragments: take last object that has any mcq fields
    if sel is None:
        objs = []
        for raw in t.splitlines():
            s = raw.strip()
            if not (s.startswith('{') and s.endswith('}')):
                continue
            try:
                d = json.loads(s)
                if isinstance(d, dict):
                    objs.append(d)
            except Exception:
                continue
        if not objs and t.startswith('{') and t.endswith('}'):
            try:
                d = json.loads(t)
                if isinstance(d, dict):
                    objs.append(d)
            except Exception:
                pass
        for d in reversed(objs):
            keys = {str(k).lower(): k for k in d.keys()}
            sel_key = keys.get('selected') or keys.get('answer') or keys.get('choice') or keys.get('option')
            conf_key = keys.get('confidence') or keys.get('conf')
            reas_key = keys.get('reasoning') or keys.get('rationale') or keys.get('explanation')
            if sel is None and sel_key and isinstance(d.get(sel_key), str):
                sel = d.get(sel_key)[:1].upper()
            if conf_key is not None:
                try:
                    conf = float(d.get(conf_key))
                except Exception:
                    pass
            if reas_key and isinstance(d.get(reas_key), str):
                reas = d.get(reas_key)
            if sel is not None:
                break

    # 3) Fallback to lone letter
    if sel is None:
        m = re.search(r"\b([ABCD])\b", t, re.IGNORECASE)
        if m:
            sel = m.group(1).upper()

    conf = max(0.0, min(float(conf), 1.0))
    return sel, conf, reas