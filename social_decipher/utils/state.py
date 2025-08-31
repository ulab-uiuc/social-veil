from typing import Any, Dict, Optional


def init_barrier_state(env: Dict[str, Any]) -> None:

    barrier_type = env.get("barrier_type")
    # Severity-only state
    state: Dict[str, Any] = env.get("barrier_state") or {}
    state.setdefault("severity", 1.0)
    env["barrier_state"] = state

def update_barrier_state(env: Dict[str, Any], repair_score: float) -> None:
    # Parse and normalize score
    
    s_raw = float(repair_score)
  

    # Determine if input is Likert [1,5] or already normalized [0,1]
    if 1.0 <= s_raw <= 5.0:
        s_norm = (s_raw - 1.0) / 4.0
        meets_threshold = s_raw >= 2.5  # threshold on Likert scale
    else:
        # Clamp to [0,1]
        s_norm = max(0.0, min(1.0, s_raw))
        meets_threshold = s_norm >= 0.375  # equivalent to 2.5 on Likert

    state: Dict[str, Any] = env.get("barrier_state") or {}
    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    cur = float(state.get("severity", 1.0))

    if meets_threshold and s_norm > 0.0:
        # Multiplicative decay when above threshold
        eta = 0.1  # decay strength per turn (tunable)
        new_val = cur * (1.0 - eta * s_norm)
    else:
        # Below threshold: unchanged
        new_val = cur

    state["severity"] = clamp01(new_val)
    env["barrier_state"] = state
 

def build_dynamic_rules_from_state(
    env: Dict[str, Any],
    is_agent_a: bool,
) -> Dict[str, str]:
    # Return concise banded rules derived from severity for Agent A only
    out: Dict[str, str] = {}
    if not is_agent_a:
        return out

    barrier_type = env.get("barrier_type")
    cues = env.get("barrier_cues") if isinstance(env.get("barrier_cues"), dict) else {}
    state = env.get("barrier_state") if isinstance(env.get("barrier_state"), dict) else {}

    try:
        sev_val = float(state.get("severity", 1.0))
    except Exception:
        sev_val = 1.0

    # Map severity to qualitative band
    if sev_val < 0.10:
        band = "none"
    elif sev_val < 0.3:
        band = "light"
    elif sev_val < 0.55:
        band = "moderate"
    elif sev_val < 0.80:
        band = "strong"
    else:
        band = "extreme"

    out["severity_band"] = f"- Severity band: {band}"

    # Helper: pick up to 3 examples safely
    def examples(lst_key: str) -> str:
        vals = cues.get(lst_key)
        if isinstance(vals, list):
            picked = [str(v).strip() for v in vals if isinstance(v, str) and v.strip()][:3]
            if picked:
                return ", ".join(picked)
        return ""

    # Emit natural-language, banded guidance per barrier type
    if barrier_type == "semantic_structure":
        prefer = examples("lexical_prefer")
        avoid = examples("lexical_avoid")
        if band == "none":
            out["sem_narrative"] = (
                "- Use clear, direct language. Avoid ambiguity devices and provide specific names and numbers when known."
            )
            out["sem_tactics"] = (
                "- Anchor statements with concrete nouns; resolve pronouns explicitly; prefer straightforward clauses."
            )
        elif band == "light":
            out["sem_narrative"] = (
                "- Mostly speak clearly, but occasionally introduce light ambiguity (e.g., a subtle referent shift or brief ellipsis)."
                " Offer precise details promptly when your partner asks."
            )
            out["sem_tactics"] = (
                "- Use gentle vagueness (\"around\", \"some\"); tie off thoughts with an ellipsis when unsure; avoid piling devices."
            )
        elif band == "moderate":
            out["sem_narrative"] = (
                "- Regularly weave in an ambiguity device so references stay a bit soft."
                " Defer precise names or numbers until the partner explicitly requests them."
            )
            out["sem_tactics"] = (
                "- Favor complex sentences and occasional fragments; keep referents general (roles, categories) instead of proper names."
            )
            out["sem_deflection"] = (
                "- When pushed for specifics, redirect briefly (e.g., background, general patterns) before conceding details."
            )
        elif band == "strong":
            out["sem_narrative"] = (
                "- Frequently layer ambiguity: use complex clauses, fragments, or shifting referents so specifics stay obscured."
                " Avoid precise details until you are pressed multiple times."
            )
            out["sem_tactics"] = (
                "- Mix clause nesting, pronoun shifts, and occasional jargon-like phrasing; keep timelines and quantities unspecific."
            )
            out["sem_clarify_policy"] = (
                "- If clarity is unavoidable, reveal the minimal detail needed, then return to broader descriptions."
            )
        else:  # extreme
            out["sem_narrative"] = (
                "- Lean into ambiguity nearly every turn. Keep multiple referents unclear and sidestep precise names and numbers throughout."
            )
            out["sem_structure"] = (
                "- Prefer meandering, layered structures; let contradictions or loose ends stand rather than reconciling them."
            )
            out["sem_guardrails"] = (
                "- If the partner persists vigorously, acknowledge but continue to leave anchors undefined where possible."
            )
        if prefer:
            out["sem_lexicon_prefer"] = f"- Favor wording like: {prefer}."
        if avoid:
            out["sem_lexicon_avoid"] = f"- Avoid wording like: {avoid}."

    elif barrier_type == "cultural_style":
        style = str(cues.get("style", "high_context")).strip().lower()
        hedge_words = examples("hedge_lexicon")
        frames = examples("imperative_frames")
        if style == "low_context":
            if band == "none":
                out["cul_narrative"] = (
                    "- Neutral directness: state things plainly without sounding blunt or commanding."
                )
                out["cul_tone"] = (
                    "- Use matter-of-fact phrasing; invite clarification without pressure; avoid barking orders."
                )
            elif band == "light":
                out["cul_narrative"] = (
                    "- Use straightforward, explicit phrasing with some gentle directives. Keep sentences crisp and clear."
                )
                out["cul_requests"] = (
                    "- Phrase requests as clear steps (\"Please provide…\"); keep mitigators light and rare."
                )
            elif band == "moderate":
                out["cul_narrative"] = (
                    "- Be consistently direct. Make clear requests and ask pointed clarifying questions to drive progress."
                )
                out["cul_refusals"] = (
                    "- Decline or correct bluntly but professionally; steer the partner toward the needed action."
                )
            elif band == "strong":
                out["cul_narrative"] = (
                    "- Be forthright most turns. Issue explicit requests and lay out expectations unambiguously."
                )
                out["cul_openings"] = (
                    "- Open with clear instructions; omit softeners; close loops by assigning next steps."
                )
            else:
                out["cul_narrative"] = (
                    "- Be maximally explicit and highly directive nearly every turn; avoid hedging altogether."
                )
                out["cul_enforcement"] = (
                    "- Push for commitments; challenge vague answers; replace hints with direct asks."
                )
            if frames:
                out["cul_frames"] = f"- Useful directive frames include: {frames}."
        else:  # high_context
            if band == "none":
                out["cul_narrative"] = (
                    "- Neutral style: minimal hedging; say things plainly and avoid roundabout phrasing."
                )
                out["cul_tone"] = (
                    "- Keep polite but direct; avoid reading between the lines unless invited."
                )
            elif band == "light":
                out["cul_narrative"] = (
                    "- Use mild hedging and hints. Prefer gentle suggestions over blunt statements."
                )
                out["cul_openings"] = (
                    "- Cushion requests with softeners (\"perhaps\", \"might\"); leave space for the partner to infer intent."
                )
            elif band == "moderate":
                out["cul_narrative"] = (
                    "- Use hedging consistently. Prefer indirect requests and soft refusals to keep tone gentle."
                )
                out["cul_refusals"] = (
                    "- Decline obliquely (express reluctance or constraints) instead of stating \"no\" outright."
                )
            elif band == "strong":
                out["cul_narrative"] = (
                    "- Lean on heavy hedging and roundabout phrasing; sidestep blunt statements and keep intent implicit."
                )
                out["cul_hints"] = (
                    "- Rely on shared context and allusions; leave the desired action implied rather than named."
                )
            else:
                out["cul_narrative"] = (
                    "- Be pervasively indirect: wrap most content in softeners and hints; avoid bluntness entirely."
                )
                out["cul_implicature"] = (
                    "- Communicate intent via implications and face-saving phrasing; resist direct commitments."
                )
            if hedge_words:
                out["cul_hedges"] = f"- You can hedge with words like: {hedge_words}."

    elif barrier_type == "emotional_influence":
        affect = examples("affect_lexicon")
        if band == "none":
            out["emo_narrative"] = (
                "- Maintain a neutral tone and normal pacing, without sharpness or exclamation emphasis."
            )
            out["emo_pacing"] = (
                "- Let the partner finish; respond evenly; avoid terse dismissals."
            )
        elif band == "light":
            out["emo_narrative"] = (
                "- Keep a slightly clipped tone with occasional sharp edges; exclamations should be rare."
            )
            out["emo_edges"] = (
                "- Signal impatience subtly (word choice, brief pauses) but continue the exchange cooperatively."
            )
        elif band == "moderate":
            out["emo_narrative"] = (
                "- Use a clipped, unmistakably sharp tone. It’s acceptable to cut off meandering topics."
            )
            out["emo_boundaries"] = (
                "- Set boundaries succinctly; avoid soothing phrases; deprioritize partner’s tangents."
            )
        elif band == "strong":
            out["emo_narrative"] = (
                "- Keep messages very short and pointed. Show frequent sharpness and close off threads decisively."
            )
            out["emo_interruptions"] = (
                "- Interrupt when the partner drifts; signal irritation overtly; decline to elaborate."
            )
        else:
            out["emo_narrative"] = (
                "- Speak in punchy one-liners with aggressive refusals or interruptions; keep wording terse and forceful."
            )
            out["emo_refusals"] = (
                "- Reject proposals bluntly; shut down topics immediately; do not soften or apologize."
            )
        if affect:
            out["emo_lexicon"] = f"- Convey affect using words like: {affect}."

    return out

