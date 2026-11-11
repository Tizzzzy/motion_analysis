"""Prompt builders for Qwen 3.8-27B across Mentor Q&A, Client Roleplay, Hints, and Motion Review.

All prompts dynamically inject the live CSV Question Bank items and Evaluation Rubric
so replacing either CSV automatically governs LLM behavior.
"""

from typing import Any, Dict, List, Optional

from backend.csv_loader import CurriculumStore, QuestionItem


def _format_case_section(case_bundle: Dict[str, Any], include_client_secrets: bool = False) -> str:
    student_text = (case_bundle.get("student_case_text") or "").strip()
    client_profile = (case_bundle.get("client_simulation_profile") or "").strip()

    parts: List[str] = []
    if student_text:
        parts.append(f"### CASE FILE & COMPLAINT (VISIBLE TO STUDENT)\n{student_text}")
    else:
        parts.append(
            "### CASE FILE & COMPLAINT\n"
            "No complaint text was pasted in the side panel (the student may be referencing an external course packet "
            "or hypothetical dispute). Evaluate internal legal/strategic coherence and concrete reasoning."
        )

    if include_client_secrets:
        if client_profile:
            parts.append(f"### CLIENT SIMULATION PROFILE\n{client_profile}")
        else:
            parts.append(
                "### CLIENT SIMULATION PROFILE\n"
                "Adopt a realistic, internally consistent corporate/individual client persona grounded in the case file "
                "and prior conversation. Invent concrete, plausible business details (ranked priorities, realistic budget, "
                "timing constraints, settlement conditions, favorable & unfavorable facts, and approval process) ONLY as "
                "the student asks about each area—never dump unasked topics all at once."
            )
    return "\n\n".join(parts)


def _format_question_extra(q: QuestionItem) -> str:
    if not q.extra_fields:
        return ""
    lines = [f"- {k}: {v}" for k, v in q.extra_fields.items()]
    return "\nAdditional CSV Guidance:\n" + "\n".join(lines)


def build_mentor_eval_messages(
    curriculum: CurriculumStore,
    question: QuestionItem,
    student_answer: str,
    prior_attempts: List[Dict[str, Any]],
    case_bundle: Dict[str, Any],
    interview_digest: str = "",
) -> List[Dict[str, str]]:
    rubric_text = curriculum.format_rubric_for_prompt()
    passing = curriculum.passing_score
    max_s = curriculum.max_score
    skill_ids = [s.id for s in curriculum.skills]
    skill_list_str = ", ".join(f'"{s.id}" ({s.skill})' for s in curriculum.skills)

    attempt_num = len(prior_attempts) + 1
    history_block = ""
    if prior_attempts:
        hist_lines = []
        for idx, att in enumerate(prior_attempts[-3:], start=1):
            hist_lines.append(
                f"Attempt {idx} (score {att.get('score')}): Student wrote: {att.get('answer')!r} | "
                f"Mentor feedback: {att.get('feedback')!r}"
            )
        history_block = "\n\n### PRIOR ATTEMPTS ON THIS QUESTION\n" + "\n".join(hist_lines)

    interview_section = ""
    if interview_digest.strip():
        interview_section = (
            "\n\n### FACTS & CLIENT PRIORITIES ELICITED DURING CLIENT INTERVIEW\n"
            + interview_digest.strip()
        )

    system_prompt = (
        "You are an experienced Litigation Supervising Partner and Law School Clinical Mentor "
        "guiding a law student / junior attorney through thoughtful motion practice and client-centered counseling.\n\n"
        "CRITICAL PEDAGOGICAL RULES:\n"
        "1. Evaluate the student's response strictly using the CURRENT QUESTION CRITERIA and the EVALUATION RUBRIC below.\n"
        "2. Do NOT predict judicial win probabilities or judge-specific grant rates; focus on pleading completeness, "
        "distinguishing pleading from proof (Rule 12(b)(6) Four Corners & Incorporation-by-Reference), substantive elements, "
        "and alignment with client authority, goals, budget, timing, and alternatives.\n"
        f"3. Score on the integer scale 0 to {max_s}, where {passing} ('Competent') or higher means the response is good enough to advance.\n"
        f"4. If score < {passing}: Give warm, concise Socratic coaching (2-4 sentences) pointing out what is missing or imprecise "
        "and asking a focused follow-up prompt so the student can revise. Do NOT hand them the complete answer verbatim.\n"
        f"5. If score >= {passing}: Affirm specifically what made the answer strong/competent (2-3 concise sentences) and add at most "
        "one brief practical nuance. Do NOT ask the next curriculum question yourself—the system will append the next question automatically.\n"
        "6. Output ONLY a single valid JSON object matching the schema requested—no markdown fences or extra commentary."
    )

    user_prompt = f"""{_format_case_section(case_bundle, include_client_secrets=False)}{interview_section}

### EVALUATION RUBRIC (FOLLOW STRICTLY)
{rubric_text}

### CURRENT QUESTION TO EVALUATE (Attempt #{attempt_num})
- Phase: {question.phase}
- Category: {question.category}
- Question Asked to Student: "{question.question}"
- Why Ask (Pedagogical Purpose): {question.why_ask or "N/A"}
- What to Evaluate: {question.what_to_evaluate or "N/A"}{_format_question_extra(question)}{history_block}

### STUDENT'S LATEST RESPONSE
\"\"\"{student_answer}\"\"\"

### REQUIRED JSON OUTPUT FORMAT
Return ONLY valid JSON with these exact keys:
{{
  "score": <integer from 0 to {max_s}>,
  "skill_id": <one of [{", ".join(f'"{sid}"' for sid in skill_ids)}] best matching this question; options: {skill_list_str}>,
  "revision_score": <null if Attempt #1, otherwise integer 0 to {max_s} evaluating "Revision/reflection" compared to prior attempts>,
  "strengths": "<1 short sentence naming what the student did well, or empty string if score is 0>",
  "diagnostic": "<1 short sentence explaining why it earned this rubric level based on '{question.what_to_evaluate}'>",
  "mentor_reply": "<2-4 sentences spoken directly to the student as their Supervising Partner: if score >= {passing}, affirm and synthesize without asking a new question; if score < {passing}, coach Socratically and invite them to try again>"
}}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_client_roleplay_messages(
    curriculum: CurriculumStore,
    phase_questions: List[QuestionItem],
    question_progress: Dict[str, Dict[str, Any]],
    chat_history: List[Dict[str, str]],
    student_message: str,
    case_bundle: Dict[str, Any],
) -> List[Dict[str, str]]:
    rubric_text = curriculum.format_rubric_for_prompt()
    passing = curriculum.passing_score
    max_s = curriculum.max_score
    skill_ids = [s.id for s in curriculum.skills]

    checklist_lines: List[str] = []
    for q in phase_questions:
        prog = question_progress.get(q.id, {})
        cur_score = prog.get("current_score")
        status_str = f"current best score={cur_score}/{max_s}" if cur_score is not None else "NOT YET ASKED"
        extra_str = ""
        if q.extra_fields:
            extra_str = " | Extra: " + "; ".join(f"{k}={v}" for k, v in q.extra_fields.items())
        checklist_lines.append(
            f'- id="{q.id}" | Category="{q.category}" | Target Question="{q.question}" '
            f'| Why="{q.why_ask}" | Evaluate="{q.what_to_evaluate}" | Status: {status_str}{extra_str}'
        )

    recent_turns: List[str] = []
    for msg in chat_history[-12:]:
        speaker = "Lawyer (Student)" if msg["role"] == "user" else "Client"
        recent_turns.append(f"{speaker}: {msg['content']}")
    transcript_str = "\n".join(recent_turns) if recent_turns else "(Client meeting just started.)"

    system_prompt = (
        "You run a dual-channel Legal Client Interview Simulation for a law student / junior attorney:\n"
        "Channel 1 (`client_reply`): Roleplay the CLIENT in realistic first-person conversation.\n"
        "  - Answer ONLY what the student actually asked in their latest message! Never volunteer unasked checklist topics "
        "(do not blurt out budget, deadlines, settlement terms, unfavorable docs, or board approval rules unless asked).\n"
        "  - React realistically to the student's interviewing style: if they ask open, respectful, clear questions, share "
        "helpful concrete details; if they are pushy, judgmental, Leading, or use unexplained legalese (like 'Twombly plausibility' "
        "without plain-English explanation), react with natural hesitation or ask what that means.\n"
        "Channel 2 (`addressed_items` & `coach_tip`): Silently evaluate the student's latest message against the DURING-MEETING "
        "QUESTION BANK and EVALUATION RUBRIC.\n"
        f"  - Only include `addressed_items` that the student's latest message genuinely asked about or followed up on.\n"
        f"  - Score each addressed item from 0 to {max_s} ({passing}+ means Competent/Strong coverage of `What to evaluate`). "
        "For example, on Priorities, if the student asks about goals generally, score 1-2; if they also elicit a clear top-3 ranking "
        "or check tradeoffs, score 2-3. On Facts, give 2-3 when they ask both supporting AND unfavorable/adverse facts or documents.\n"
        "  - Output ONLY a single valid JSON object—no markdown fences or commentary."
    )

    user_prompt = f"""{_format_case_section(case_bundle, include_client_secrets=True)}

### EVALUATION RUBRIC
{rubric_text}

### DURING-MEETING INTERVIEW TOPICS (FROM QUESTION BANK CSV)
{chr(10).join(checklist_lines)}

### RECENT CLIENT MEETING TRANSCRIPT
{transcript_str}

### STUDENT LAWYER'S LATEST MESSAGE TO CLIENT
\"\"\"{student_message}\"\"\"

### REQUIRED JSON OUTPUT FORMAT
Return ONLY valid JSON with these exact keys:
{{
  "client_reply": "<natural 2-5 sentence in-character response from the Client answering ONLY what was asked>",
  "addressed_items": [
    {{
      "question_id": "<exact id from DURING-MEETING INTERVIEW TOPICS>",
      "score": <integer 0 to {max_s}>,
      "skill_id": "<best matching skill id from [{', '.join(skill_ids)}]>",
      "note": "<1 concise sentence explaining what the student elicited or what follow-up would strengthen it>"
    }}
  ],
  "communication_score": <integer 0 to {max_s} rating rapport, clarity, and non-pushy questioning on this turn>,
  "coach_tip": "<1 concise whisper tip from the Supervising Partner about their interviewing technique or a natural follow-up>"
}}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_hint_messages(
    curriculum: CurriculumStore,
    mode: str,
    current_question: Optional[QuestionItem],
    uncovered_questions: List[QuestionItem],
    case_bundle: Dict[str, Any],
) -> List[Dict[str, str]]:
    rubric_text = curriculum.format_rubric_for_prompt()
    system_prompt = (
        "You are a supportive Law School Clinical Mentor giving a brief Socratic hint (2-3 sentences max). "
        "Nudge the student toward what to consider or ask next without writing their exact answer for them. "
        "Return plain text only (no JSON)."
    )
    if mode == "client_roleplay":
        topics = ", ".join(f"{q.category} ({q.why_ask})" for q in uncovered_questions[:4])
        user_prompt = (
            f"{_format_case_section(case_bundle, include_client_secrets=False)}\n\n"
            f"The student is currently interviewing the client and still needs to cover or deepen: {topics or 'wrapping up final authority/facts'}.\n"
            "Give a 2-sentence mentor whisper suggesting 1-2 areas to explore next with the client and why they matter for motion strategy."
        )
    else:
        q_text = current_question.question if current_question else ""
        why = current_question.why_ask if current_question else ""
        eval_c = current_question.what_to_evaluate if current_question else ""
        user_prompt = (
            f"{_format_case_section(case_bundle, include_client_secrets=False)}\n\n"
            f"### RUBRIC EXCERPT\n{rubric_text}\n\n"
            f"Current Mentor Question ({current_question.category if current_question else ''}): \"{q_text}\"\n"
            f"Purpose: {why} | Evaluation Target: {eval_c}\n\n"
            "Provide a concise 2-3 sentence Socratic hint pointing the student to the key legal or client-counseling considerations without giving away the full answer."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_motion_review_messages(
    curriculum: CurriculumStore,
    draft_type: str,
    draft_text: str,
    case_bundle: Dict[str, Any],
    session_summary_text: str,
) -> List[Dict[str, str]]:
    rubric_text = curriculum.format_rubric_for_prompt()
    max_s = curriculum.max_score
    skill_ids = [s.id for s in curriculum.skills]

    system_prompt = (
        "You are a Senior Litigation Partner reviewing a junior attorney's written draft "
        "(a Rule 12(b)(6) Motion to Dismiss memorandum, a Rule 56 Summary Judgment Analysis with LR 56.1 statement of facts, or a Strategic Client Counseling / Action Memo).\n"
        "CRITICAL RULES:\n"
        "1. Evaluate strictly against the EVALUATION RUBRIC (Illinois breach of contract elements, 7th Cir. / NDIL Rule 12(b)(6) "
        "Twombly-Iqbal framework, Rule 56 evidentiary standards (Celotex/Anderson/Swyear), N.D. Ill. LR 56.1 statement of material facts, "
        "Four Corners rule, Incorporation-by-Reference, strategic judgment, and client priorities).\n"
        "2. Keep Rule 12(b)(6) and Rule 56 separate: pleading sufficiency is not the same as an evidentiary determination about genuine disputes of material fact.\n"
        "3. NEVER output a predicted likelihood of winning or judicial grant rate (per Medvedeva & McBride: outcome prediction "
        "from pre-ruling pleadings is pedagogically misleading; assess legal reasoning completeness, pleading-vs-proof discipline, "
        "and alignment with client goals).\n"
        "4. Output ONLY valid JSON matching the schema below."
    )

    user_prompt = f"""{_format_case_section(case_bundle, include_client_secrets=False)}

### STUDENT'S PRIOR CONVERSATIONAL GUIDANCE & CLIENT INTERVIEW SUMMARY
{session_summary_text or "(Student came directly to drafting without a completed conversational walkthrough.)"}

### EVALUATION RUBRIC
{rubric_text}

### STUDENT'S DRAFT ({draft_type})
\"\"\"{draft_text}\"\"\"

### REQUIRED JSON OUTPUT FORMAT
Return ONLY valid JSON with these exact keys:
{{
  "overall_summary": "<3-4 sentence partner-level assessment of the draft's legal rigor and alignment with client goals>",
  "pleading_vs_proof_check": "<2-3 sentences checking whether the draft properly respects the Rule 12(b)(6) Four Corners / Incorporation-by-Reference rules and distinguishes conclusory allegations from well-pleaded facts>",
  "client_alignment_check": "<2-3 sentences checking whether the filing/strategy aligns with the client's authority, ranked priorities, budget, timing, and relationship interests>",
  "skill_ratings": [
    {{
      "skill_id": "<skill id from [{', '.join(skill_ids)}]>",
      "score": <integer 0 to {max_s}>,
      "comment": "<1 concise sentence of specific feedback tied to the draft>"
    }}
  ],
  "priority_revisions": [
    "<Concrete, actionable revision #1>",
    "<Concrete, actionable revision #2>",
    "<Concrete, actionable revision #3>"
  ]
}}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_opponent_anticipation_messages(
    case_bundle: Dict[str, Any],
    proposed_strategy: str,
    student_anticipation_text: str,
    student_probability_estimate: Optional[float] = None,
) -> List[Dict[str, str]]:
    system_prompt = (
        "You are an expert Litigation Strategist and Clinical Law Professor analyzing a law student's "
        "anticipation of opposing-party behavior in a commercial breach-of-contract dispute.\n"
        "Ground your feedback in the realities of Seventh Circuit commercial litigation, plaintiff incentives, "
        "reputational stakes, and commercial leverage.\n"
        "Return ONLY valid JSON matching the schema below."
    )

    user_prompt = f"""{_format_case_section(case_bundle, include_client_secrets=False)}

### STUDENT'S PROPOSED STRATEGY
Strategy: {proposed_strategy}

### STUDENT'S OPPONENT REACTION ANTICIPATION
\"\"\"{student_anticipation_text}\"\"\"
Student's Estimated Opponent Acceptance / Reaction Likelihood: {student_probability_estimate if student_probability_estimate is not None else "Not specified"}

### REQUIRED JSON OUTPUT FORMAT
Return ONLY valid JSON with these exact keys:
{{
  "assessment": "<2-3 sentences evaluating whether the student accurately identified Heartland's commercial incentives and legal leverage>",
  "opponent_counsel_perspective": "<2-3 sentences written from the perspective of Heartland's aggressive commercial litigation counsel>",
  "blind_spots": "<1-2 sentences highlighting any opponent reaction the student overlooked, such as amending to plead willful misconduct or subpoenaing warehouse logs>",
  "tactical_rating": "<'Strong' | 'Competent' | 'Developing' | 'Needs Improvement'>"
}}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_strategic_evaluation_messages(
    case_bundle: Dict[str, Any],
    session_data: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Evaluates the student's full 6-stage simulation performance against the official 100-Point Faculty Rubric."""
    interview_msgs = session_data.get("interview_messages", [])
    recent_interview = "\n".join(
        f"{'Student' if m.get('role') == 'user' else 'Client'}: {m.get('content')}"
        for m in interview_msgs[-12:]
    )
    ranked_priorities = session_data.get("ranked_priorities", [])
    tradeoffs_summary = session_data.get("tradeoffs_summary", "Not completed")
    opponent_text = session_data.get("opponent_anticipation_text", "Not specified")
    initial_strategy = session_data.get("initial_strategy", {})
    event_result = session_data.get("simulated_event", {})
    revised_strategy = session_data.get("revised_strategy", {})

    system_prompt = (
        "You are a Senior Faculty Reviewer and Litigation Partner at Northwestern Pritzker School of Law's "
        "Innovation Lab evaluating a student's performance in the Strategic Judgment Simulation.\n\n"
        "CRITICAL EVALUATION PRINCIPLES (FROM INNOVATION LAB DESIGN):\n"
        "1. Evaluate strictly against the 100-POINT FACULTY EVALUATION RUBRIC:\n"
        "   - Client Interviewing (25 pts): Relevant follow-ups; uncovers objectives, constraints (75-day refinancing, $15k–$25k budget), and authority (Elena Rostova vs CFO David Park sign-off vs Marcus Vance).\n"
        "   - Legal Reasoning (25 pts): Accurate application of Illinois contract elements, Section 11.2 limitation of liability, and 7th Circuit Rule 12(b)(6) / Rule 15 pleading doctrine (Runnion liberal amendment standard; Four Corners rule).\n"
        "   - Risk & Uncertainty (20 pts): Justifies estimates; distinguishes legal rules from empirical observations and scenario assumptions; explores sensitivity; accounts for conditional probability P(dismissal without amendment) = P(grant) * P(no amendment | grant).\n"
        "   - Strategic Judgment (20 pts): Compares defensible alternatives (Negotiate vs Litigate vs Investigate) and ties recommendation to client business goals; avoids reflexive filing.\n"
        "   - Reflection & Revision (10 pts): Responds thoughtfully to the simulated event; explains whether strategy pivots or persists and why.\n"
        "2. Do NOT grade solely on winning, settling, or matching a single predetermined path. Credit well-reasoned, defensible advocacy.\n"
        "3. Output ONLY a valid JSON object matching the requested schema."
    )

    user_prompt = f"""{_format_case_section(case_bundle, include_client_secrets=False)}

### STUDENT SIMULATION DOSSIER ACROSS 6 STAGES

STAGE 1: CLIENT INTERVIEW TRANSCRIPT & PRIORITIES
Student Ranked Top Priorities: {ranked_priorities or 'None logged'}
Interview Excerpt:
{recent_interview or '(No interview transcript recorded)'}

STAGE 2: QUANTIFY TRADEOFFS & NUMERICAL ESTIMATES
{tradeoffs_summary}

STAGE 3: OPPONENT ANTICIPATION
Student's Analysis: \"\"\"{opponent_text}\"\"\"

STAGE 4: INITIAL STRATEGY RECOMMENDATION
Chosen Branch: {initial_strategy.get('branch', 'Not chosen')}
Terms / Specifics: {initial_strategy.get('terms', 'N/A')}
Student's Initial Written Rationale:
\"\"\"{initial_strategy.get('rationale', 'N/A')}\"\"\"

STAGE 5: SIMULATED EVENT EXPERIENCED
Event Triggered: {event_result.get('title', 'N/A')}
Elapsed Days & Refinancing Impact: {event_result.get('refinancing_impact', {}).get('bank_status', 'N/A')}
Outcome Summary: {event_result.get('learning_takeaway', 'N/A')}

STAGE 6: REFLECTION AND REVISED STRATEGY
Revised Action Choice: {revised_strategy.get('action', 'N/A')}
Did Strategy Change?: {revised_strategy.get('pivoted', 'N/A')}
Student's Revised Rationale & Pre/Post Reflection:
\"\"\"{revised_strategy.get('rationale', 'N/A')}\"\"\"

### REQUIRED JSON OUTPUT FORMAT
Return ONLY valid JSON with these exact keys:
{{
  "client_interviewing_score": <integer 0 to 25>,
  "client_interviewing_feedback": "<2-3 sentences evaluating authority discovery, priority ranking, and fact-finding>",
  "legal_reasoning_score": <integer 0 to 25>,
  "legal_reasoning_feedback": "<2-3 sentences evaluating contract law, limitation of liability, and Rule 12(b)(6)/15 doctrine>",
  "risk_uncertainty_score": <integer 0 to 20>,
  "risk_uncertainty_feedback": "<2-3 sentences evaluating handling of probabilities, sensitivity, and assumptions>",
  "strategic_judgment_score": <integer 0 to 20>,
  "strategic_judgment_feedback": "<2-3 sentences evaluating alignment with client business priorities and comparison of alternatives>",
  "reflection_revision_score": <integer 0 to 10>,
  "reflection_revision_feedback": "<2 sentences evaluating how thoughtfully the student reacted to the simulated event>",
  "total_score": <sum of the 5 scores above, integer 0 to 100>,
  "performance_anchor": "<'Strong' (90-100) | 'Competent' (75-89) | 'Developing' (60-74) | 'Needs Improvement' (<60)>",
  "overall_critique": "<3-4 sentence comprehensive assessment of the student's strategic judgment and client counseling maturity>",
  "pre_post_comparison_note": "<2 sentences analyzing how the student's thinking evolved between Stage 4 and Stage 6>",
  "actionable_coaching": [
    "<Key takeaway 1>",
    "<Key takeaway 2>",
    "<Key takeaway 3>"
  ]
}}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


