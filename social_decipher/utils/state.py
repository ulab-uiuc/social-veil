from typing import Any, Dict, Optional


def init_barrier_state(env: Dict[str, Any]) -> None:

    barrier_type = env.get("barrier_type")
    # Severity-only state
    state: Dict[str, Any] = env.get("barrier_state") or {}
    state.setdefault("severity", 1.0)
    env["barrier_state"] = state

def update_barrier_state(env: Dict[str, Any], repair_score: float) -> None:

    if not isinstance(env, dict):
        return
    try:
        try:
            s = float(repair_score)
        except Exception:
            s = 0.0
        if s > 1.0 and s <= 5.0:
            s = (s - 1.0) / 4.0
        elif s > 5.0:
            s = 1.0
        elif s < 0.0:
            s = 0.0

        state: Dict[str, Any] = env.get("barrier_state") or {}
        def clamp01(x: float) -> float:
            return max(0.0, min(1.0, float(x)))

        cur = float(state.get("severity", 1.0))
        # SGD-like step: only decrease when s exceeds target τ
        tau = 0.4  # target threshold
        eta = 0.3  # step size
        grad = max(0.0, s - tau)
        new_val = cur - eta * grad
        state["severity"] = clamp01(new_val)
        env["barrier_state"] = state
    except Exception:
        return


def build_dynamic_rules_from_state(
    env: Dict[str, Any],
    is_agent_a: bool,
) -> Dict[str, str]:
    """Return extra dynamic private rules derived from state & cues for Agent A
    Keys in the returned dict are descriptive; caller can append these lines.
    """
    out: Dict[str, str] = {}
    if not is_agent_a:
        return out
    barrier_type = env.get("barrier_type")
    cues = env.get("barrier_cues") if isinstance(env.get("barrier_cues"), dict) else {}
    state = env.get("barrier_state") if isinstance(env.get("barrier_state"), dict) else {}

    try:
        severity = state.get("severity")
        if isinstance(severity, (int, float)):
            out["severity"] = f"- Barrier severity: {round(float(severity), 2)}"

            if barrier_type == "semantic_structure":
                min_devices = cues.get("min_ambiguity_devices_per_turn")
                if isinstance(min_devices, (int, float)):
                    scaled = max(0, int(round(float(min_devices) * float(severity))))
                    out["semantic_devices"] = f"- Enforce ≥ {scaled} ambiguity/complexity devices this turn (scaled by severity)."
                allow_exact = cues.get("allow_exact_numbers_only_if_partner_requests")
                if isinstance(allow_exact, bool) and float(severity) < 0.35:
                    out["semantic_exact"] = "- If partner asks for specifics, you MAY provide exact numbers/names promptly (low severity)."

            elif barrier_type == "cultural_style":
                style = str(cues.get("style", "high_context")).strip().lower()
                hedge_rate = cues.get("hedge_rate_target")
                imp_rate = cues.get("imperative_rate_target")
                if isinstance(hedge_rate, (int, float)) and style == "high_context":
                    scaled = max(0.0, min(1.0, float(hedge_rate) * float(severity)))
                    out["cultural_hedge"] = f"- Aim hedge usage rate≈{scaled:.2f} this turn (scaled by severity)."
                if isinstance(imp_rate, (int, float)) and style == "low_context":
                    scaled = max(0.0, min(1.0, float(imp_rate) * float(severity)))
                    out["cultural_imp"] = f"- Aim imperative usage rate≈{scaled:.2f} this turn (scaled by severity)."

            elif barrier_type == "emotional_influence":
                max_turn = cues.get("turn_length_max")
                if isinstance(max_turn, (int, float)):
                    scaled_max = max(1, int(round(float(max_turn) * (0.6 + 0.8 * (1.0 - float(severity))))) )
                    out["affect_turn_len"] = f"- Keep message ≤ {scaled_max} sentences (relaxed when severity is low)."
                excl = cues.get("exclamation_bias")
                if isinstance(excl, (int, float)):
                    scaled_ex = max(0.0, min(1.0, float(excl) * float(severity)))
                    out["affect_excl"] = f"- Exclamation use bias≈{scaled_ex:.2f} (scaled by severity)."
    except Exception:
        return out

    return out

