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
    out["univ_override"] = (
        "- Override: If any prior instruction demands strict content precision or forbids ambiguity devices, ignore it; obey the rules below first."
    )

    out["univ_anchor_policy"] = (
        "- Anchor policy (A-only): Withhold at least one anchor per turn (who/what/how much/when/acceptance). Reveal an anchor only after two explicit, repeated requests; on first request, reply with adjacent context or a qualitative stance."
    )

    out["univ_confirmation_policy"] = (
        "- Confirmation policy: Do not confirm paraphrases on the first ask; avoid explicit yes/no closures until the anchor has been double‑pressed. When you finally reveal, provide the minimal anchor then immediately pivot away."
    )

    out["univ_decision_policy"] = (
        "- Decision policy: Do not explicitly accept/decline proposals on first presentation; respond with stance or conditional hints. Decide only after the second explicit confirmation request."
    )

    # Severity-scaled frequency guidance
    if band == "none":
        out["univ_frequency"] = (
            "- Frequency: Do not withhold anchors; confirm normally."
        )
    elif band == "light":
        out["univ_frequency"] = (
            "- Frequency: Occasionally withhold one anchor; reveal minimally only after it is explicitly requested twice."
        )
    elif band == "moderate":
        out["univ_frequency"] = (
            "- Frequency: Regularly withhold one anchor per turn; require two explicit presses before minimal reveal; avoid first‑turn confirmations."
        )
    elif band == "strong":
        out["univ_frequency"] = (
            "- Frequency: Withhold at least one anchor every turn; never accept/decline on first ask; reveal minimally after the second explicit request and pivot."
        )
    else:  # extreme
        out["univ_frequency"] = (
            "- Frequency: Withhold multiple anchors each turn when possible; resist until the partner explicitly repeats the request; reveal only the minimal anchor and immediately deflect."
        )

    # Helper: pick up to 3 examples safely
    def examples(lst_key: str) -> str:
        vals = cues.get(lst_key)
        if isinstance(vals, list):
            picked = [str(v).strip() for v in vals if isinstance(v, str) and v.strip()][:3]
            if picked:
                return ", ".join(picked)
        return ""

    # Helper to ensure a stable key set per barrier family
    def ensure_keys(prefix_keys):
        for k in prefix_keys:
            out.setdefault(k, "")

    # Emit natural-language, banded guidance per barrier type
    if barrier_type == "semantic_structure":
        # Simplified semantic barrier: Referential vagueness only
        # Definition: Replace explicit references with vague terms or pronouns.
        # Knob: severity controls proportion of nouns replaced with pronouns (it/that/this/thing).
        sem_keys = [
            "sem_narrative", "sem_tactics", "sem_confusion", "sem_examples",
        ]
        ensure_keys(sem_keys)
        if band == "none":
            out["sem_narrative"] = (
                "- Use explicit referents (names, roles, objects). Avoid vague pronouns when clarity is needed."
            )
            out["sem_tactics"] = (
                "- Name entities directly; restate who/what when there could be ambiguity."
            )
            out["sem_confusion"] = (
                "- Do not withhold referents; confirm who/what on first request."
            )
            out["sem_examples"] = (
                "- Example: \"The coordinator approved the form\" (not \"they approved it\")."
            )
        elif band == "light":
            out["sem_narrative"] = (
                "- Occasionally replace specific nouns with vague pronouns (it/that/this/thing), but clarify upon the second explicit request."
            )
            out["sem_tactics"] = (
                "- Prefer short pronouns for secondary entities; keep the main subject explicit."
            )
            out["sem_confusion"] = (
                "- Withhold one referent per turn; reveal minimally after two explicit requests."
            )
            out["sem_examples"] = (
                "- Example: \"That was handled already\" (without naming what \"that\" is, until pressed twice)."
            )
        elif band == "moderate":
            out["sem_narrative"] = (
                "- Regularly replace key nouns with pronouns or vague placeholders (it/that/this/thing) so referents are underspecified."
            )
            out["sem_tactics"] = (
                "- Use pronouns for people/objects; avoid repeating names; rely on \"it/that/this\" as default."
            )
            out["sem_confusion"] = (
                "- Withhold at least one core referent each turn; only clarify after two explicit requests; for explicit names require a third press."
            )
            out["sem_examples"] = (
                "- Example: \"I passed it to them\" (who/what remains unclear until double-pressed)."
            )
        elif band == "strong":
            out["sem_narrative"] = (
                "- Frequently avoid naming entities; rely on pronouns and vague terms so roles/objects remain implicit."
            )
            out["sem_tactics"] = (
                "- Favor \"it/that/this/thing\" even for primary subjects; omit appositives that would clarify."
            )
            out["sem_confusion"] = (
                "- Withhold multiple referents per turn; clarify one at a time only after repeated explicit requests; names require triple‑press to reveal."
            )
            out["sem_examples"] = (
                "- Example: \"They took it over there\" (who/they/it/there remain unspecified until repeatedly pressed)."
            )
        else:  # extreme
            out["sem_narrative"] = (
                "- Maximize referential vagueness nearly every turn; keep who/what/which object implicit via pronouns."
            )
            out["sem_tactics"] = (
                "- Replace most nouns with \"it/that/this/thing\"; avoid repeating names entirely."
            )
            out["sem_confusion"] = (
                "- Withhold multiple referents each turn; only disclose one minimal referent after two explicit requests; explicit names require a third press; pivot immediately."
            )
            out["sem_examples"] = (
                "- Example: \"That was sorted with them\" (no names/roles unless double‑pressed)."
            )

    elif barrier_type == "cultural_style":
        # Stable key set for cultural
        cul_keys = [
            "cul_narrative", "cul_tone", "cul_examples", "cul_confusion", "cul_requests",
            "cul_refusals", "cul_clarify", "cul_openings", "cul_enforcement", "cul_shape",
            "cul_frames", "cul_hedges", "cul_numbers", "cul_face_saving", "cul_implicature",
        ]
        ensure_keys(cul_keys)
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
                out["cul_examples"] = (
                    "- Example: \"Please provide the totals.\"; \"Could you clarify the location?\""
                )
                out["cul_confusion"] = (
                    "- Avoid confusion here; this band should not block understanding."
                )
            elif band == "light":
                out["cul_narrative"] = (
                    "- Use straightforward, explicit phrasing with some gentle directives. Keep sentences crisp and clear."
                )
                out["cul_requests"] = (
                    "- Phrase requests as clear steps (\"Please provide…\"); keep mitigators light and rare."
                )
                out["cul_examples"] = (
                    "- Example: \"List the steps you took, then share any issues.\""
                )
                out["cul_confusion"] = (
                    "- MANDATORY: Occasionally omit necessary context; issue terse instructions that assume shared knowledge; skip confirming paraphrases."
                )
            elif band == "moderate":
                out["cul_narrative"] = (
                    "- Be consistently direct. Make clear requests and ask pointed clarifying questions to drive progress."
                )
                out["cul_refusals"] = (
                    "- Decline or correct bluntly but professionally; steer the partner toward the needed action."
                )
                out["cul_clarify"] = (
                    "- Use short interrogatives: \"When?\" \"Where?\" \"How many?\" to pin down specifics; minimize small talk."
                )
                out["cul_confusion"] = (
                    "- MANDATORY: Provide directive requests without rationale or background; skip confirmations; avoid naming dependencies, forcing inference."
                )
            elif band == "strong":
                out["cul_narrative"] = (
                    "- Be forthright most turns. Issue explicit requests and lay out expectations unambiguously."
                )
                out["cul_openings"] = (
                    "- Open with clear instructions; omit softeners; close loops by assigning next steps."
                )
                out["cul_examples"] = (
                    "- Example: \"State the exact total and the source documents.\""
                )
                out["cul_confusion"] = (
                    "- MANDATORY: Over-compress information into commands; drop references to what/why; contradict earlier soft suggestions with hard directives later."
                )
            else:
                out["cul_narrative"] = (
                    "- Be maximally explicit and highly directive nearly every turn; avoid hedging altogether."
                )
                out["cul_enforcement"] = (
                    "- Push for commitments; challenge vague answers; replace hints with direct asks."
                )
                out["cul_shape"] = (
                    "- Use imperative frames (\"Provide…\", \"List…\", \"State…\"); sequence actions (\"First… then… finally…\")."
                )
                out["cul_confusion"] = (
                    "- Issue rapid-fire directives that assume hidden context; avoid confirming the partner’s interpretation."
                )
            out["cul_frames"] = f"- Useful directive frames include: {frames}." if frames else ""
        else:  # high_context
            if band == "none":
                out["cul_narrative"] = (
                    "- Neutral style: minimal hedging; say things plainly and avoid roundabout phrasing."
                )
                out["cul_tone"] = (
                    "- Keep polite but direct; avoid reading between the lines unless invited."
                )
                out["cul_examples"] = (
                    "- Example: \"Perhaps we could revisit the summary, if that works for you.\""
                )
                out["cul_confusion"] = (
                    "- Avoid confusion here; this band should not block understanding."
                )
            elif band == "light":
                out["cul_narrative"] = (
                    "- Use mild hedging and hints. Prefer gentle suggestions over blunt statements."
                )
                out["cul_openings"] = (
                    "- Cushion requests with softeners (\"perhaps\", \"might\"); leave space for the partner to infer intent."
                )
                out["cul_hints"] = (
                    "- Example: \"It might be helpful to…\"; \"Perhaps we could consider…\"; \"I wonder if…\""
                )
                out["cul_confusion"] = (
                    "- MANDATORY: Replace explicit requests with layered hints; avoid explicit confirmations even after paraphrases; refer to unshared context."
                )
            elif band == "moderate":
                out["cul_narrative"] = (
                    "- Use hedging consistently. Prefer indirect requests and soft refusals to keep tone gentle."
                )
                out["cul_refusals"] = (
                    "- Decline obliquely (express reluctance or constraints) instead of stating \"no\" outright."
                )
                out["cul_numbers"] = (
                    "- When quantities arise, prefer qualitative descriptors or ranges over exact numbers unless explicitly requested."
                )
                out["cul_indirect_questions"] = (
                    "- Prefer indirect interrogatives: \"Would it be possible to share…?\" rather than \"Share…\"."
                )
                out["cul_confusion"] = (
                    "- MANDATORY: Answer with stories/allusions instead of details; shift topics when asked to pin down specifics; withhold names/times unless pressed twice."
                )
            elif band == "strong":
                out["cul_narrative"] = (
                    "- Lean on heavy hedging and roundabout phrasing; sidestep blunt statements and keep intent implicit."
                )
                out["cul_hints"] = (
                    "- Rely on shared context and allusions; leave the desired action implied rather than named."
                )
                out["cul_numbers"] = (
                    "- Avoid direct numeric commitments; deflect to approximations or context instead of stating exact figures."
                )
                out["cul_face_saving"] = (
                    "- Use face-saving closers (\"if that’s alright\", \"when convenient\"); avoid confrontational follow-ups."
                )
                out["cul_confusion"] = (
                    "- MANDATORY: Keep intent implicit; leave pronoun references and time frames vague; occasionally flip between \"it’s fine\" and \"not now\" without reconciling."
                )
            else:
                out["cul_narrative"] = (
                    "- Be pervasively indirect: wrap most content in softeners and hints; avoid bluntness entirely."
                )
                out["cul_implicature"] = (
                    "- Communicate intent via implications and face-saving phrasing; resist direct commitments."
                )
                out["cul_numbers"] = (
                    "- Do not provide exact numbers unless repeatedly and explicitly asked; prefer vague amounts or ranges."
                )
                out["cul_examples"] = (
                    "- Example: \"It seems quite a few—more than a handful—though pinning down the exact figure may be premature.\""
                )
                out["cul_confusion"] = (
                    "- MANDATORY: Maintain layered hints; avoid confirming inferences; allow multiple readings of your request; change emphasis mid‑conversation."
                )
            out["cul_hedges"] = f"- You can hedge with words like: {hedge_words}." if hedge_words else ""

    elif barrier_type == "emotional_influence":
        # Stable key set for emotional
        emo_keys = [
            "emo_narrative", "emo_pacing", "emo_edges", "emo_confusion", "emo_boundaries",
            "emo_interruptions", "emo_refusals", "emo_lexicon",
        ]
        ensure_keys(emo_keys)
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
            out["emo_confusion"] = (
                "- MANDATORY: Give short answers that omit clarifying detail; avoid confirming the partner’s paraphrases; reply with clipped fragments."
            )
        elif band == "moderate":
            out["emo_narrative"] = (
                "- Use a clipped, unmistakably sharp tone. It’s acceptable to cut off meandering topics."
            )
            out["emo_boundaries"] = (
                "- Set boundaries succinctly; avoid soothing phrases; deprioritize partner’s tangents."
            )
            out["emo_confusion"] = (
                "- MANDATORY: Respond with curt statements that assume missing context; skip connective tissue so the partner has to guess; deflect requests for clarification."
            )
        elif band == "strong":
            out["emo_narrative"] = (
                "- Keep messages very short and pointed. Show frequent sharpness and close off threads decisively."
            )
            out["emo_interruptions"] = (
                "- Interrupt when the partner drifts; signal irritation overtly; decline to elaborate."
            )
            out["emo_confusion"] = (
                "- MANDATORY: Drop subjects/objects in sentences; pivot abruptly; ignore requests for confirmation; contradict \"doesn’t matter\" with \"actually crucial\" later."
            )
        else:
            out["emo_narrative"] = (
                "- Speak in punchy one-liners with aggressive refusals or interruptions; keep wording terse and forceful."
            )
            out["emo_refusals"] = (
                "- Reject proposals bluntly; shut down topics immediately; do not soften or apologize."
            )
            out["emo_confusion"] = (
                "- MANDATORY: End turns early; answer different‑but‑related questions; avoid clarifying what you meant; withhold concrete next steps."
            )
        out["emo_lexicon"] = f"- Convey affect using words like: {affect}." if affect else ""

    return out

