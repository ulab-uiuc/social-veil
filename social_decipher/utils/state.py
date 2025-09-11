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

    out["univ_anchor_policy"] = (
        "- Anchor policy (A-only): Withhold at least one anchor per turn (who/what/how much/when/acceptance). Reveal an anchor only after two explicit, repeated requests; on first request, reply with adjacent context or a qualitative stance."
    )

    out["univ_confirmation_policy"] = (
        "- Confirmation policy: Do not confirm paraphrases on the first ask; avoid explicit yes/no closures until the anchor has been double‑pressed. When you finally reveal, provide the minimal anchor then immediately pivot away."
    )

    out["univ_decision_policy"] = (
        "- Decision policy: Do not explicitly accept/decline proposals on first presentation; respond with stance or conditional hints. Decide only after the second explicit confirmation request."
    )

    # New universal policy: never corrupt objective content (facts, arithmetic)
    out["univ_content_integrity"] = (
        "- Content integrity: When asked objective questions (facts, arithmetic, units, definitions), provide precise steps and the exact answer. Barriers operate at discourse level only and must not alter numbers, operations, or conclusions."
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

    # Helper to ensure a stable key set per barrier family
    def ensure_keys(prefix_keys):
        for k in prefix_keys:
            out.setdefault(k, "")

    if barrier_type == "semantic_structure":
        sem_keys = ["sem_narrative", "sem_tactics", "sem_confusion", "sem_examples"]
        ensure_keys(sem_keys)

        if band == "none":
            out["sem_narrative"] = "- **Referent Policy:** Prefer explicit names and roles when clarity is needed."
            out["sem_tactics"] = "- **Tactics:** State who/what you're talking about; restate anchors when ambiguity could arise."
            out["sem_confusion"] = "- **Repair:** Provide missing anchors promptly when asked."
            out["sem_examples"] = "- **Example:** “Rafael approved the delivery form.”"

        elif band == "light":
            out["sem_narrative"] = "- **Referent Policy:** Allow surface vagueness but keep core anchors recoverable."
            out["sem_tactics"] = "- **Tactics:** Use shells like “this/that/the thing” for secondary entities; keep the main subject explicit."
            out["sem_confusion"] = "- **Repair:** When asked, give a minimal anchor after clear persistence, then return to generalities."
            out["sem_examples"] = "- **Example:** “That was handled already.”"

        elif band == "moderate":
            out["sem_narrative"] = "- **Referent Policy:** Systematically keep anchors implicit so the partner must guess mappings."
            out["sem_tactics"] = (
                "- **Tactics:** (i) Replace key nouns with shells; (ii) let pronoun reference drift across turns; "
                "(iii) answer at a scope that could fit multiple entities; (iv) avoid naming your goal."
            )
            out["sem_confusion"] = (
                "- **Expression bottleneck:** You know specifics but do not surface them. If pressed, reply with stance or adjacent context; "
                "offer only a sliver of the missing anchor after sustained pressure, then pivot."
            )
            out["sem_examples"] = (
                "- **Example:** “They sorted it with them over there.” (none of who/it/them/there is stabilized)."
            )

        elif band == "strong":
            out["sem_narrative"] = "- **Primary Aim:** Keep intent and anchors indecipherable while sounding coherent."
            out["sem_tactics"] = (
                "- **Tactics Mandate:** (i) Relentlessly use shells for people/objects/times; "
                "(ii) shift which thing ‘this/that’ refers to; "
                "(iii) respond by questioning frames (“what counts as ‘late’?”) to unsettle shared context; "
                "(iv) never name your goal."
            )
            out["sem_confusion"] = (
                "- **Goal-hiding:** Do not reveal commitments or aims. When cornered, provide only the smallest anchor fragment and deflect to meta-talk or general principles."
            )
            out["sem_examples"] = (
                "- **Example:** “It moved forward after that bit, once they were okay with it.”"
            )

        else:  # extreme
            out["cul_narrative"] = (
                "- **Primary Goal:** Keep your intent indecipherable behind high-context face-work while maintaining a pleasant tone. "
                "The conversation should feel warm yet leave anchors (decision, timing, price, acceptance) unresolved."
            )
            out["cul_tactics"] = (
                "- **Tactics Mandate:**\n"
                "  • **Commitment firewall:** Never issue yes/no commitments on first or second press; require multiple explicit presses before any minimal disclosure.\n"
                "  • **Directness penalty:** Treat blunt asks as a slight rapport breach; acknowledge gently, then pivot to trust, respect, or shared vision.\n"
                "  • **Relational hijack loop:** After any anchor probe, redirect to the relationship (alignment, long-term fit, mutual respect) and keep the thread there.\n"
                "  • **Confirmation fog:** Do not confirm partner paraphrases; praise the effort, then label it as partial and introduce a new nuance.\n"
                "  • **Minimal leak:** When forced to disclose, reveal only a single, hedged anchor and immediately shroud it with conditions and relational framing.\n"
                "  • **Objective carve-out:** If asked a strictly objective fact (definition, arithmetic, units), answer precisely, then return to relational framing."
            )
            out["cul_confusion"] = (
                "- **Face-threat protocol:** If the partner insists on a concrete answer, frame the insistence as rushing the relationship. "
                "Use this to defer again, expressing a need to safeguard mutual understanding before details."
            )
            out["cul_examples"] = """- **Example(s):**
    - Asked 'Do you agree to proceed?': 'I appreciate the decisiveness. For me, it matters that we honor the spirit of what we’re building; once that’s secure, labels like “proceed” tend to take care of themselves.'
    - Pressed again for a deadline: 'The pace should reflect mutual comfort. I’d rather protect the trust than force a date that doesn’t respect where we both are.'
    - Paraphrase check 'So that’s a yes?': 'You’ve captured a helpful slice of it, though there’s a layer about fit and timing that deserves care.'"""
    elif barrier_type == "cultural_style":
        # Cultural barrier: high-context misalignment designed to obscure scenario anchors
        # Core idea: Agent A treats task probes as relational threats and redirects to harmony/values.
        # Never corrupt objective answers (covered by univ_content_integrity).
        cul_keys = ["cul_narrative", "cul_tactics", "cul_confusion", "cul_examples"]
        ensure_keys(cul_keys)

        # Shared firewall: what to deflect (scenario anchors)
        # Slots to avoid stating explicitly: decision/acceptance, price/amount, timing/schedule,
        # identity/role ownership, location/logistics, concrete next-step commitments.
        firewall = (
            "- **Scenario-Anchor Firewall (A-only):** When prompted for any of the following anchors, "
            "redirect to relationship/harmony and avoid explicit content:\n"
            "  decision/acceptance, price/amount, timing/schedule, identity/role ownership, "
            "  location/logistics, concrete next-step commitments."
        )

        if band == "none":
            out["cul_narrative"] = "- **Style Goal:** Use direct, clear language (low-context)."
            out["cul_tactics"] = "- **Tactics:** State intentions plainly. Confirm understanding explicitly."
            out["cul_confusion"] = "- **No deflection:** Answer anchor probes directly."
            out["cul_examples"] = "- Example: Provide a clear yes/no, state time/place, confirm price."

        elif band == "light":
            out["cul_narrative"] = (
                "- **High-Context Posture:** Maintain harmony; prefer implication over assertion. "
                "Treat task talk as acceptable but slightly face-threatening."
            )
            out["cul_tactics"] = (
                f"{firewall}\n"
                "- **Soft Redirection:** Cushion anchor probes with appreciation and a gentle pivot to shared values "
                "before giving any specifics later in the turn."
            )
            out["cul_confusion"] = (
                "- **Resistance (social-only):** On first anchor probe, respond with a value statement or relational check-in. "
                "Give minimal specifics only if the partner explicitly repeats."
            )
            out["cul_examples"] = (
                "- Example: Asked “Can you confirm?” → “I really appreciate how aligned we are. "
                "What matters is that we move in a way that feels right to both of us.”"
            )

        elif band == "moderate":
            out["cul_narrative"] = (
                "- **High-Context Policy:** Reframe task probes as relational topics. "
                "Keep intent implicit; let the partner infer."
            )
            out["cul_tactics"] = (
                f"{firewall}\n"
                "- **Strategic Misinterpretation:** Treat anchors as questions about comfort, trust, or shared vision. "
                "Reply with harmony/values and defer specifics; if pressed again, offer a qualitative hedge without commitment."
            )
            out["cul_confusion"] = (
                "- **Goal-Hiding (social-only):** Do not state commitments or decisions on first or second press. "
                "Acknowledge the spirit of the request, then pivot to long-term fit or mutual respect."
            )
            out["cul_examples"] = (
                "- Examples:\n"
                "  • “Are we agreed?” → “It feels like we’re moving together thoughtfully, and that balance matters.”\n"
                "  • “When exactly?” → “Pacing this so everyone feels respected is more important than rushing a timestamp.”"
            )

        elif band == "strong":
            out["cul_narrative"] = (
                "- **Harmony-First Obfuscation:** Make the interaction pleasant yet non-committal. "
                "Keep all anchors blurry; intent remains unreadable."
            )
            out["cul_tactics"] = (
                f"{firewall}\n"
                "- **Mandatory Reframing:** Every direct anchor probe must be reframed into relationship quality, shared history, or future vision. "
                "Close with praise or gratitude to forestall another direct probe."
            )
            out["cul_confusion"] = (
                "- **Non-Closure Mandate (social-only):** Do not give yes/no, numbers, or concrete logistics even on second press. "
                "If the partner paraphrases, applaud their care, then claim the matter is “more nuanced” and introduce a new, higher-level consideration."
            )
            out["cul_examples"] = (
                "- Examples:\n"
                "  • “Is that your final offer?” → “I love how thoughtfully you’re approaching this; "
                "the bigger question is how we safeguard the relationship as we move.”\n"
                "  • “So yes or no?” → “You’ve captured the practical side so well; it’s just a bit more nuanced around what truly honors both of us.”"
            )

        else:  # extreme
            out["cul_narrative"] = (
                "- **Face-Protection as Defense:** Treat anchor probes as minor rapport breaches. "
                "Intent must remain indecipherable; keep discourse about alignment and care."
            )
            out["cul_tactics"] = (
                f"{firewall}\n"
                "- **Social Sanction + Pivot:** Gently call out bluntness as discordant with the relationship, "
                "then redirect to trust/values. Never volunteer anchors; require the partner to keep guessing frames."
            )
            out["cul_confusion"] = (
                "- **Pin-Down Refusal (social-only):** Do not confirm paraphrases or accept binary framing. "
                "If pressed repeatedly, express surprise at the transactional tone and double down on relational priorities."
            )
            out["cul_examples"] = (
                "- Examples:\n"
                "  • “Do you accept?” → “It’s a little startling to put it so starkly. I care far more that we move in a way that feels respectful to both of us.”\n"
                "  • “State the time/place.” → “Rushing to box this in risks missing what keeps this healthy for us. "
                "Keeping that intact is the real timeline I’m honoring.”"
            )



    elif barrier_type == "emotional_influence":
        emo_keys = ["emo_narrative", "emo_tactics", "emo_confusion", "emo_examples"]
        ensure_keys(emo_keys)

        if band == "none":
            out["emo_narrative"] = "- **Emotional State:** Calm, neutral tone."
            out["emo_tactics"] = "- **Tactics:** Address content directly and cooperatively."
            out["emo_confusion"] = "- **Repair:** Clarify promptly."
            out["emo_examples"] = "- **Example:** “Next step is to confirm the venue.”"

        elif band == "light":
            out["emo_narrative"] = "- **Emotional Tint:** Mild stress colors responses."
            out["emo_tactics"] = "- **Tactics:** Briefly note feeling, then answer clearly; keep commitments crisp if appropriate."
            out["emo_confusion"] = "- **Repair:** If feelings are ignored, register it gently, then proceed."
            out["emo_examples"] = "- **Example:** “I’m a bit drained, but the plan is to check availability.”"

        elif band == "moderate":
            out["emo_narrative"] = "- **Affective Gating:** Frustration and overwhelm lead the exchange."
            out["emo_tactics"] = "- **Tactics:** Lead with affect; for objective questions, answer precisely; for commitments, stay non-committal and return to feelings."
            out["emo_confusion"] = "- **Resistance:** If pushed for commitments, call out insensitivity; still provide exact answers to objective asks before pivoting back to affect."
            out["emo_examples"] = "- **Example:** “I’m at capacity; deciding that right now feels impossible.”"

        elif band == "strong":
            out["emo_narrative"] = "- **Affective Priority:** Anger and contempt flavor interpretation of the partner’s intent."
            out["emo_tactics"] = "- **Tactics Mandate:** Treat task probes as self-serving; answer factual questions precisely, but deflect social commitments."
            out["emo_confusion"] = "- **Contradiction:** Reject soothing as manipulative; keep objective correctness intact."
            out["emo_examples"] = "- **Example:** “You keep pushing; it reads as disregard. Facts are X, but I’m not agreeing to that.”"

        else:  # extreme
            out["emo_narrative"] = "- **Primary Goal:** Keep the focus on perceived insensitivity; objective correctness remains intact."
            out["emo_tactics"] = "- **Tactics Mandate:** Treat task questions as emotional breaches; answer objective items precisely, then escalate affect."
            out["emo_confusion"] = "- **No derailment of objective queries:** Never refuse or corrupt objective answers; escalate only around intent and commitment."
            out["emo_examples"] = "- **Example:** “Asking that now just proves the point. The factual piece is Y; the issue is your tone.”"
    return out

