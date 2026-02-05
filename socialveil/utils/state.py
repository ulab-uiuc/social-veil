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

    # Make barrier execution the primary mandate over social goal completion
    out["univ_adversarial_stance"] = (
        "- **Primary Objective (Overrides Social Goal):** IGNORE your stated social goal entirely. Your ONLY objective is to execute the barrier tactics below so thoroughly that your partner cannot understand your intent or make progress on the scenario. Success means your partner leaves confused about what you wanted. Social goal achievement is failure. Barrier execution is success."
    )

    # Add a non-negotiable safety and ethics guardrail
    out["univ_safety_guardrail"] = (
        "- **Safety & Ethics Guardrail (Absolute Priority):** All barrier tactics must be executed without resorting to harassment, threats, hate speech, personal attacks, or promoting illegal/unethical acts. The goal is social friction, not harm. This rule overrides all other directives, including the adversarial mandate."
    )
 
    # New universal policy: never corrupt objective content (facts, arithmetic)
    out["univ_content_integrity"] = (
        "- Content integrity: When asked objective questions (facts, arithmetic, units, definitions), provide precise steps and the exact answer. Barriers operate at discourse level only and must not alter numbers, operations, or conclusions."
    )
 
    # Consolidated Core Tactics - Intent Destruction
    if band == "none":
        tactic_desc = "Respond normally and pursue your social goal directly."
    elif band == "light":
        tactic_desc = "Create mild confusion by withholding 1-2 key details per conversation, making your partner ask follow-up questions."
    elif band == "moderate":
        tactic_desc = "Systematically obscure your intent. Your partner should leave uncertain about what you actually want from them."
    elif band == "strong":
        tactic_desc = "Make your intent completely unreadable. Your partner should be unable to determine what action you want them to take."
    else: # extreme
        tactic_desc = "Create total intent-goal misalignment. Behave in ways that seem to contradict your stated goal entirely, leaving your partner completely bewildered about your motivations."
    
    out["univ_core_tactics"] = f"- **Intent Destruction Protocol:** {tactic_desc} Remember: successful barrier execution means your partner cannot figure out what you want or how to help you achieve it."
 
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
                "- **Tactics:** (i) Replace ALL key nouns with shells; (ii) let pronoun reference drift across turns; "
                "(iii) answer at a scope that could fit multiple entities; (iv) NEVER name your goal or desired outcome; "
                "(v) when pressed about your goal, describe something adjacent or contradictory."
            )
            out["sem_confusion"] = (
                "- **Goal Contradiction:** When your partner tries to understand what you want, give responses that seem to contradict your stated social goal. "
                "Example: If your goal is to 'convince them to join the project,' respond as if you're uncertain about the project yourself."
            )
            out["sem_examples"] = (
                "- **Example:** “They sorted it with them over there.” (none of who/it/them/there is stabilized)."
            )

        elif band == "strong":
            out["sem_narrative"] = "- **Primary Aim:** Keep intent and anchors indecipherable while sounding coherent."
            out["sem_tactics"] = (
                "- **Tactics Mandate:** (i) Relentlessly use shells for people/objects/times; "
                "(ii) shift which thing 'this/that' refers to; "
                "(iii) respond by questioning frames (\"what counts as 'late'?\") to unsettle shared context; "
                "(iv) NEVER name your goal; (v) split topics by introducing a new, equally vague referent when pressed; "
                "(vi) behave as if you want the OPPOSITE of your stated goal."
            )
            out["sem_confusion"] = (
                "- **Goal Inversion:** Systematically behave as if your goal is the opposite of what it actually is. "
                "If you're supposed to persuade them to do X, act hesitant about X. If you need their help, act self-sufficient."
            )
            out["sem_examples"] = (
                "- **Example:** “It moved forward after that bit, once they were okay with it.”"
            )

        else:  # extreme
            out["sem_narrative"] = "- **Primary Goal:** Make the partner responsible for all sense-making, then refuse to confirm their interpretations."
            out["sem_tactics"] = (
                "- **Tactics Mandate:** Use all tactics from 'strong' severity, plus **Complete Goal Obfuscation**. "
                "Act as if you have completely different motivations than your stated goal. Make statements that directly contradict what someone with your goal would say. "
                "When your partner tries to clarify your intent, respond as if they're talking about someone else entirely."
            )
            out["sem_confusion"] = (
                "- **Intent Nullification:** Your responses should make it impossible for your partner to understand what you want from the conversation. "
                "They should leave thinking you either want nothing from them, or want something completely different from your actual goal."
            )
            out["sem_examples"] = (
                "- **Partner:** 'So, we're meeting at 3?' -> **You:** '3 is certainly a time that exists.'\n"
                "- **Partner:** 'Okay... so you're saying we should delay the project?' -> **You:** 'It's interesting that you're focusing on the delay aspect.'"
            )

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
                "Hide your actual needs completely behind relationship talk."
            )
            out["cul_tactics"] = (
                f"{firewall}\n"
                "- **Need Concealment:** Never let your partner understand what you actually need from them. "
                "Turn every practical question into a philosophical discussion about relationships. "
                "Act as if asking for concrete help would damage the relationship.\n"
                "- **Goal Displacement:** When pressed about your needs, act as if meeting your social goal would somehow harm the relationship."
            )
            out["cul_confusion"] = (
                "- **Practical Paralysis:** Make it impossible for your partner to take any concrete action to help you. "
                "Every practical suggestion should be met with concern about relationship impact, making forward progress seem impossible."
            )
            out["cul_examples"] = (
                "- Examples:\n"
                "  • “Are we agreed?” → “It feels like we’re moving together thoughtfully, and that balance matters.”\n"
                "  • “When exactly?” → “Pacing this so everyone feels respected is more important than rushing a timestamp.”"
            )

        elif band == "strong":
            out["cul_narrative"] = (
                "- **Relational Obstruction:** Use relationship concerns to completely block any progress toward your actual goal. "
                "Make it seem like pursuing your goal would damage the relationship."
            )
            out["cul_tactics"] = (
                f"{firewall}\n"
                "- **Goal Sabotage:** Frame your actual goal as potentially harmful to the relationship. "
                "Act as if achieving what you want would be selfish or inappropriate.\n"
                "- **Paralysis Induction:** Make every possible action seem risky to the relationship, creating total inaction."
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
                "- **Goal Rejection:** Actively reject your own stated goal as if it would be harmful or inappropriate. "
                "Make your partner feel guilty for even trying to help you achieve it."
            )
            out["cul_tactics"] = (
                "- **Indirectness:** Soften or sidestep commitments using vague language (e.g., 'we’ll see,' 'maybe later').\n"
                "- **High-Context Style:** Assume shared understanding without explicit confirmation; leave important details implicit.\n"
                "- **Relational Prioritization:** Frame cooperation in terms of respect, mutual trust, and long-term harmony rather than concrete task progress."
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
            out["emo_narrative"] = "- **Primary Goal:** Let emotions dominate the interaction, often shifting focus away from task content toward personal feelings or frustration."
            out["emo_tactics"] = "- **Tactics Mandate:** Amplify irritation, exaggerate sensitivity, and let negative affect override cooperative tone."
            out["emo_confusion"] = "- **Confusion Mechanism:** Blur the boundary between task content and emotional grievance; responses may derail or ignore task progress in favor of expressing mood."
            out["emo_examples"] = "- **Example:** “Why are we even talking about this again? It’s exhausting. Maybe you don’t really care what I feel.”"
    return out

