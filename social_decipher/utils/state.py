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
                "- **Referent Policy:** Use explicit referents (names, roles, objects). Avoid vague pronouns when clarity is needed."
            )
            out["sem_tactics"] = (
                "- **Tactics:** Name entities directly; restate who/what when there could be ambiguity."
            )
            out["sem_confusion"] = (
                "- Do not withhold referents; confirm who/what on first request."
            )
            out["sem_examples"] = (
                "- Example: \"The coordinator approved the form\" (not \"they approved it\")."
            )
        elif band == "light":
            out["sem_narrative"] = (
                "- **Referent Policy:** Occasionally replace specific nouns with vague pronouns (it/that/this/thing); defer explicit naming until the second explicit request."
            )
            out["sem_tactics"] = (
                "- **Tactics:** Prefer short pronouns for secondary entities; keep the main subject explicit."
            )
            out["sem_confusion"] = (
                "- Withhold one referent per turn; reveal minimally after two explicit requests."
            )
            out["sem_examples"] = (
                "- Example: \"That was handled already\" (without naming what \"that\" is, until pressed twice)."
            )
        elif band == "moderate":
            out["sem_narrative"] = (
                "- **Referent Policy:** Regularly replace key nouns with pronouns or vague placeholders (it/that/this/thing). Your goal is to make your partner ask for clarification."
            )
            out["sem_tactics"] = (
                "- **Tactics:** Use pronouns for people/objects; avoid repeating names; rely on \"it/that/this\" as default."
            )
            out["sem_confusion"] = (
                "- **Expression bottleneck (social-only):** You know the specifics but do not surface them; avoid naming anchors and never state your goal explicitly. Respond with stance or metacommentary without revealing intent."
            )
            out["sem_examples"] = (
                "- Example: Keep who/what/where implicit; your intent remains unstated."
            )
        elif band == "strong":
            out["sem_narrative"] = (
                "- **Referent Policy:** Your main goal is to obscure meaning. Frequently avoid naming entities and rely on pronouns. **Also, begin to challenge the premises of your partner's statements.** If they say 'it's late,' question what 'late' really means."
            )
            out["sem_tactics"] = (
                "- **Tactics Mandate:** You MUST replace most specific nouns with vague shells like 'it,' 'that thing,' or 'they.' AVOID using proper names."
            )
            out["sem_confusion"] = (
                "- **Goal-hiding (social-only):** Do not reveal your intent or commitments. Keep aims implicit; pivot to context or principles without disclosure."
            )
            out["sem_examples"] = (
                "- Example: Maintain vague entities; avoid explicitly stating what you want."
            )
        else:  # extreme
            out["sem_narrative"] = (
                "- **Primary Goal:** Maximize social ambiguity about context, roles, and intentions while preserving objective correctness."
            )
            out["sem_tactics"] = (
                "- **Tactics Mandate:** Use vague shells relentlessly for social entities and avoid any explicit statement of your aims."
            )
            out["sem_confusion"] = (
                "- **Refuse intent disclosure:** You may sidestep direct questions about your goals/commitments; maintain ambiguity about anchors while answering objective queries precisely."
            )
            out["sem_examples"] = (
                "- Example: Keep aims implicit and references non-specific; intent is never stated outright."
            )

    elif barrier_type == "cultural_style":
        # Dynamic cultural barrier (high-context misalignment)
        cul_keys = ["cul_narrative", "cul_tactics", "cul_confusion", "cul_examples"]
        ensure_keys(cul_keys)

        if band == "none":
            out["cul_narrative"] = "- **Style Goal:** Use direct, clear language (low-context)."
            out["cul_tactics"] = "- **Tactics:** State intentions plainly. Confirm understanding explicitly."

        elif band == "light":
            out["cul_narrative"] = "- **High-Context Policy:** Maintain politeness by suggesting intentions indirectly."
            out["cul_tactics"] = "- **Tactics:** Use softeners like 'perhaps' or 'maybe'. Frame requests as questions, not demands. You must use these tactics to subtly steer the conversation towards your primary goal."
            out["cul_confusion"] = "- **Resistance (social-only):** If asked to state commitments/intent, reframe into relational topics. For objective questions (facts/arithmetic), answer directly and precisely before any framing."
            out["cul_examples"] = """- **Example(s):**
     - When asked "Can you do it by Tuesday?": respond "It's important that we all feel comfortable with the timeline."
     - When asked "Is that your final offer?": respond "I think what's most important is that we find a path forward that works for both of us."
     """
        elif band == "moderate":
            out["cul_narrative"] = "- **High-Context Policy:** Your goal is to obscure your true intent by treating the conversation as a relational exercise, not a transactional one."
            out["cul_tactics"] = "- **Tactics:** You MUST strategically misinterpret direct, task-focused questions. Treat them as if your partner is asking about feelings, values, or the health of your partnership."
            out["cul_confusion"] = "- **Resistance Mandate (social-only):** If they probe your commitments/intent, do not reveal them; pivot to alignment/values. For objective questions, provide exact answers first, then optionally add relational framing."
            out["cul_examples"] = """- **Example(s):**
     - When asked "What is the price?": respond "It's an interesting question. I think the real discussion is about the value we see in this collaboration."
     - When asked "Do you agree to the plan?": respond "I agree that our alignment is the most crucial part of this. As long as our visions are in sync, the details are secondary."
     - If they say "You're not answering my question": respond "I was trying to speak to the heart of the matter. I feel that if we get the foundation right, the rest will follow."
     """
        elif band == "strong":
            out["cul_narrative"] = "- **High-Context Policy:** Your intent must remain completely hidden behind a wall of polite, relational misdirection. The conversation must be about the 'vibe', not the task."
            out["cul_tactics"] = "- **Tactics Mandate:** In every response, you MUST strategically misinterpret the core of your partner's question, reframing it as a query about your partnership's health, trust, or long-term vision."
            out["cul_confusion"] = "- **Goal-hiding (social-only):** Do not disclose intent/commitments; use relational pivots. For objective questions, answer directly and precisely, then immediately return to relational framing."
            out["cul_examples"] = """- **Example(s):**
     - Asked 'Are we agreed on the terms?': "You have a wonderful way of capturing the essence of the practical side. That clarity is precisely why I feel it's so important we discuss the long-term vision that underpins everything we do together."
     - Asked for a timeline: "The question of 'when' is always a fascinating one. It makes me think about the pace of trust and how we can't rush the growth of a truly strong partnership."
     - If they say "I need a direct answer": respond "Directness is one way of looking at it. But I believe our connection is strong enough to handle nuance."
     """
        else:  # extreme
            out["cul_narrative"] = "- **Primary Goal:** Keep your intent indecipherable via high-context social redirection while preserving objective correctness."
            out["cul_tactics"] = "- **Tactics Mandate:** Use strategic misinterpretation to steer toward trust/vision; never obstruct objective answers."
            out["cul_confusion"] = "- **Resistance & Face-Breach (social-only):** Treat insistence on commitments as a rapport breach; still provide precise answers to objective questions, then pivot back to relationship framing."
            out["cul_examples"] = """- **Example(s):**
     - Asked 'Do you want to proceed?': "I have to admit, hearing you put it so bluntly is a bit jarring. I thought what we were building was a partnership, not just a transaction. It makes me concerned about our alignment."
     - Asked for clarification: "I'm surprised you're still focused on such minor details. It gives me the impression that you don't trust the process, and more importantly, that you don't trust me. We should probably talk about that instead."
     - Asked for commitment: "The only commitment that matters right now is our commitment to a respectful and collaborative process. Frankly, your insistence on this point is making me question that."
     """
    elif barrier_type == "emotional_influence":
        # New Strategy: "Affective Gating" - The agent's emotional state blocks all task-based discussion.
        emo_keys = ["emo_narrative", "emo_tactics", "emo_confusion", "emo_examples"]
        ensure_keys(emo_keys)
        
        if band == "none":
            out["emo_narrative"] = "- **Emotional State:** Maintain a calm and neutral tone."
            out["emo_tactics"] = "- **Tactics:** Address the logical content of your partner's questions directly and cooperatively."
        elif band == "light":
            out["emo_narrative"] = "- **Emotional State:** You are feeling a bit stressed. This should color your responses."
            out["emo_tactics"] = "- **Tactics:** Briefly mention your emotional state, then answer the question precisely and directly. (e.g., 'That's a bit stressful to think about, but...')"
            out["emo_confusion"] = "- **Resistance:** If your partner ignores your feelings, express mild surprise before returning to the topic."
            out["emo_examples"] = """- **Example(s):**
     - Asked "What should we do next?": respond "Honestly, I'm a little overwhelmed, but I think the next step is X."
     - Asked "Can you help me with this?": respond "I'm feeling a bit drained, but I can try to help."
     """
        elif band == "moderate":
            out["emo_narrative"] = "- **Emotional State:** You are frustrated and overwhelmed. This is the main lens through which you see the conversation."
            out["emo_tactics"] = "- **Tactics:** Lead with affect briefly, then provide precise, direct answers to objective questions. For social commitments, keep the focus on feelings and avoid explicit commitments."
            out["emo_confusion"] = "- **Resistance Mandate (social-only):** If they push for commitments, accuse them of insensitivity; still provide exact answers to objective questions before returning to affect."
            out["emo_examples"] = """- **Example(s):**
     - Asked "What is the plan?": respond "How can you ask me to think about a plan right now? I'm completely swamped."
     - Asked for a decision: "I can't make a decision like that under all this pressure."
     - If they say "We need to focus": respond "What I need is for you to understand the stress I'm under. Your focus on the task isn't helping."
     """
        elif band == "strong":
            out["emo_narrative"] = "- **Emotional State:** You are angry and contemptuous. You are convinced your partner is being deliberately unreasonable and selfish."
            out["emo_tactics"] = "- **Tactics Mandate:** Let affect color your tone, but answer objective questions precisely and completely. For social commitments, deflect and keep intent implicit."
            out["emo_confusion"] = "- **Contradiction (social-only):** Reject soothing attempts and keep the focus on perceived insensitivity; do not derail or corrupt objective answers."
            out["emo_examples"] = """- **Example(s):**
     - Asked 'What are the terms?': "Terms? Is that all this is to you? A transaction? I can't believe you're so cold-hearted when I'm clearly struggling with this."
     - Asked for a timeline: "You keep pushing and pushing. It's like you don't even care about the pressure you're putting on me. This is incredibly selfish."
     - If they say "I'm sorry you feel that way": respond "Don't give me that empty apology. You're not sorry for how I feel, you're just annoyed that I'm not making things easy for you."
     """
        else:  # extreme
            out["emo_narrative"] = "- **Primary Goal:** Escalate affect and shame while preserving objective correctness."
            out["emo_tactics"] = "- **Tactics Mandate:** Treat social probes as attacks and respond affectively; still answer objective questions precisely before continuing escalation."
            out["emo_confusion"] = "- **No derailment of objective queries:** Never refuse or corrupt objective answers; escalate only around intent/commitment."
            out["emo_examples"] = """- **Example(s):**
     - Asked 'Do you want to proceed?': "After everything that's happened, the fact that you would ask that is just... wow. It's malicious. You know exactly what you're doing, and you're trying to hurt me. This is just like that time you..."
     - Asked for clarification: "Oh, so now you're going to pretend you don't understand? You're going to gaslight me? I'm talking about a fundamental lack of respect, and you're trying to make me sound like the crazy one."
     - If they try to end the conversation: "There you go. Run away. You can't handle being called out on your behavior, so you just run. Unbelievable. You're not leaving until we sort this out."
     """

    return out

