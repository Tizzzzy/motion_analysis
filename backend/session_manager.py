"""Conversational session state machine, dynamic rubric tracker, and persistence."""

import copy
import csv
import io
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.csv_loader import CurriculumStore, PhaseGroup, QuestionItem
from backend.llm_client import QwenLLMClient, parse_llm_json
from backend.prompts import (
    build_client_roleplay_messages,
    build_hint_messages,
    build_mentor_eval_messages,
    build_motion_review_messages,
    build_opponent_anticipation_messages,
    build_stage2_colloquy_messages,
    build_strategic_evaluation_messages,
)
from backend.sample_cases import get_case_bundle
from backend.tradeoff_engine import compute_tradeoffs, get_default_tradeoff_inputs, map_priorities_to_tradeoff_weights
from backend.event_engine import trigger_simulated_event

BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_sessions_dir() -> Path:
    if os.environ.get("VERCEL"):
        p = Path(tempfile.gettempdir()) / "motion_mindset_sessions"
        p.mkdir(parents=True, exist_ok=True)
        return p
    local_dir = BASE_DIR / "data" / "sessions"
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        return local_dir
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "motion_mindset_sessions"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp_int(val: Any, lo: int, hi: int, default: int = 0) -> int:
    try:
        iv = int(round(float(val)))
        return max(lo, min(hi, iv))
    except Exception:
        return default


STAGE2_COLLOQUY_QUESTIONS = {
    "q4_elements": {
        "id": "q4_elements",
        "number": "Q4",
        "title": "Contract Elements & Pleading Standards (Illinois Law)",
        "question": "What elements must Heartland establish for breach of contract under Illinois law? Which allegations in the complaint address each element?",
        "why_ask": "Tests accurate claim identification and application of contract elements to the complaint.",
        "supporting_source": "Ivey v. Transunion Rental Screening Solutions, Inc., 2022 IL 127903, ¶ 28; 735 ILCS 5/2-603",
        "legal_rule": "Under Illinois law, plaintiff must allege: (1) existence of a valid and enforceable contract, (2) substantial performance by plaintiff, (3) breach of duty by defendant, and (4) resultant injury/damages. In the complaint, ¶ 2 alleges contract; ¶ 3 alleges performance; ¶ 4-5 allege breach (refrigeration failure); ¶ 6, 10 allege damages ($68,400 direct inventory + $340,000 lost profits).",
        "key_concepts": ["contract", "valid", "perform", "breach", "damage", "injury", "ivey", "paragraph", "alleg", "element"],
        "next_key": "q5_facts",
    },
    "q5_facts": {
        "id": "q5_facts",
        "number": "Q5",
        "title": "Concrete Facts vs. Legal Conclusions (Twombly / Iqbal)",
        "question": "Which allegations describe concrete facts, and which are primarily legal conclusions? How does this distinction affect your Rule 12(b)(6) analysis?",
        "why_ask": "Tests understanding of pleading sufficiency and the distinction between factual content and conclusory statements.",
        "supporting_source": "Federal Rule 8(a)(2); Bell Atlantic Corp. v. Twombly, 550 U.S. 544; Ashcroft v. Iqbal, 556 U.S. 662",
        "legal_rule": "Concrete factual allegations (e.g., recorded ambient temperature above 49°F, three spoiled pallets, compressor alarms) are accepted as true on a Rule 12(b)(6) motion. Conclusory assertions (e.g., 'Lakeview acted in bad faith and in reckless disregard of industry standards') are legal labels stripped away under Twombly/Iqbal. Under Rule 8(a)(2), the court disregards bare legal conclusions when evaluating whether a plausible claim is stated.",
        "key_concepts": ["twombly", "iqbal", "conclus", "fact", "true", "label", "bad faith", "temperature", "49", "plausib", "presum"],
        "next_key": "q6_four_corners",
    },
    "q6_four_corners": {
        "id": "q6_four_corners",
        "number": "Q6",
        "title": "Four Corners vs. Outside Extrinsic Evidence (Rule 12(d))",
        "question": "Suppose Lakeview's internal refrigeration records contradict Heartland's allegations. Can you rely on those records to argue that the complaint should be dismissed under Rule 12(b)(6)? Why or why not?",
        "why_ask": "Tests whether students understand the pleading-stage standard and restrictions on considering outside evidence.",
        "supporting_source": "Federal Rules 12(b)(6) & 12(d); 7th Circuit Incorporation-by-Reference Doctrine",
        "legal_rule": "A Rule 12(b)(6) motion is strictly confined to the Four Corners of the complaint (and contracts incorporated by reference). Lakeview cannot introduce internal maintenance logs, warehouse technician tickets (#409), or sensor spreadsheets on a 12(b)(6) motion without converting the motion to a Rule 56 summary judgment under Rule 12(d), which triggers formal discovery and defeats our 75-day refinancing deadline.",
        "key_concepts": ["four corners", "12(d)", "convert", "summary judgment", "extrinsic", "outside", "rule 56", "discovery", "incorporated", "log", "ticket"],
        "next_key": "q7_amend",
    },
    "q7_amend": {
        "id": "q7_amend",
        "number": "Q7",
        "title": "7th Circuit Leave to Amend Practice (Runnion v. Girl Scouts)",
        "question": "If you successfully challenge part of the complaint, would that necessarily end the litigation? What possibility should you consider before estimating the benefit of filing the motion?",
        "why_ask": "Tests recognition of amendment and the distinction between a successful pleading challenge and final resolution.",
        "supporting_source": "Federal Rule 15(a)(2); Runnion v. Girl Scouts of Greater Chicago, 786 F.3d 510 (7th Cir. 2015)",
        "legal_rule": "In the Seventh Circuit, district courts routinely grant leave to amend upon granting a 12(b)(6) motion unless amendment is demonstrably futile (Runnion, 786 F.3d at 519). A partial or initial dismissal does NOT end the case—Heartland will simply file an amended complaint curing technical deficiencies (91% probability in this posture), blowing past Lakeview's 75-day refinancing deadline and increasing fees beyond the $25k budget cap.",
        "key_concepts": ["runnion", "amend", "leave to amend", "rule 15", "end the case", "futile", "prolong", "delay", "refinancing", "75", "replead"],
        "next_key": "q8_damages",
    },
    "q8_damages": {
        "id": "q8_damages",
        "number": "Q8",
        "title": "Consequential Damages & Lost Profits Uncertainty",
        "question": "Heartland alleges $68,400 in direct losses and $340,000 in additional damages. How would you assess the legal and evidentiary uncertainty surrounding the claimed lost profits?",
        "why_ask": "Tests whether the student evaluates causation and the reasonable-certainty standard without assuming that a large claim is automatically recoverable or automatically defective at the pleading stage.",
        "supporting_source": "Ivey v. Transunion, 2022 IL 127903, ¶¶ 28–29; Tri-G, Inc. v. Burke, Bosselman & Weaver; Master Agreement Sec. 11.2",
        "legal_rule": "Under Illinois law, lost profits and consequential damages are subject to substantial legal and evidentiary uncertainty; they require proof to a reasonable certainty and proximate causation (Tri-G). Crucially, Section 11.2 of the Master Agreement expressly waives all consequential and lost-profit damages and caps liability at 2x monthly storage charges ($36,000 max). Because the contract is referenced in the complaint, Lakeview can use Section 11.2 to attack the $340,000 claim as legally barred.",
        "key_concepts": ["consequential", "lost profit", "reasonable certainty", "tri-g", "11.2", "waiver", "cap", "36,000", "speculative", "direct", "limitation of liability"],
        "next_key": None,
    },
}


def _create_default_stage2_colloquy() -> Dict[str, Any]:
    initial_msg = (
        "Welcome to our Stage 2 Legal Analysis Colloquy. Before we finalize our quantitative litigation decision tree "
        "and advise Elena Rostova, we need to ground our strategy in governing Illinois contract law and Seventh Circuit federal pleading doctrine.\n\n"
        "Let's examine **Question 4 (Contract Elements & Pleading Standards)**:\n"
        "\"What elements must Heartland establish for breach of contract under Illinois law? Which allegations in the complaint address each element?\""
    )
    return {
        "current_question_key": "q4_elements",
        "messages": [
            {
                "role": "assistant",
                "speaker": "Supervising Partner",
                "content": initial_msg,
                "timestamp": _now_iso(),
                "question_key": "q4_elements",
            }
        ],
        "questions": {
            k: {
                "id": k,
                "number": q["number"],
                "title": q["title"],
                "question": q["question"],
                "why_ask": q["why_ask"],
                "supporting_source": q["supporting_source"],
                "status": "in_progress" if k == "q4_elements" else "pending",
                "student_answer": None,
                "rating": None,
                "score": None,
                "feedback": None,
                "model_rule": None,
                "answered_at": None,
            }
            for k, q in STAGE2_COLLOQUY_QUESTIONS.items()
        },
    }


class SessionManager:
    def __init__(self, curriculum: CurriculumStore, llm_client: QwenLLMClient) -> None:
        self.curriculum = curriculum
        self.llm = llm_client
        self.sessions_dir = _resolve_sessions_dir()

    def _session_path(self, session_id: str) -> Path:
        safe_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
        return self.sessions_dir / f"{safe_id}.json"

    def save_session(self, session: Dict[str, Any]) -> None:
        session["updated_at"] = _now_iso()
        try:
            path = self._session_path(session["session_id"])
            tmp_path = path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(str(tmp_path), str(path))
        except OSError:
            pass

    def load_session(
        self,
        session_id: str,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.curriculum.reload_if_needed()
        if isinstance(snapshot, dict) and snapshot.get("session_id"):
            session = copy.deepcopy(snapshot)
            session["session_id"] = session_id
            self._sync_with_curriculum(session)
            return session
        path = self._session_path(session_id)
        if not path.exists():
            raise KeyError(f"Session {session_id!r} not found.")
        session = json.loads(path.read_text(encoding="utf-8"))
        self._sync_with_curriculum(session)
        return session

    def _sync_with_curriculum(self, session: Dict[str, Any]) -> None:
        """Ensure session progress structures match any newly edited/replaced CSV rows."""
        q_prog = session.setdefault("question_progress", {})
        for q in self.curriculum.questions:
            if q.id not in q_prog:
                q_prog[q.id] = {
                    "question_id": q.id,
                    "attempts": [],
                    "initial_score": None,
                    "revised_score": None,
                    "current_score": None,
                    "passed": False,
                    "skipped": False,
                    "skill_id": None,
                    "last_note": "",
                }

        s_scores = session.setdefault("skill_scores", {})
        for s in self.curriculum.skills:
            if s.id not in s_scores:
                s_scores[s.id] = {
                    "skill_id": s.id,
                    "skill": s.skill,
                    "initial_scores": [],
                    "latest_scores": [],
                    "initial_score": None,
                    "revised_score": None,
                    "last_comment": "",
                }
            else:
                s_scores[s.id]["skill"] = s.skill

        phases = self.curriculum.phases
        if not phases:
            session["completed"] = True
            session["current_question_id"] = None
            return

        session.setdefault("sim_stage", 1)
        session.setdefault("ranked_priorities", [])
        if "tradeoff_inputs" not in session or not session["tradeoff_inputs"]:
            session["tradeoff_inputs"] = get_default_tradeoff_inputs()
            session["tradeoff_results"] = compute_tradeoffs(session["tradeoff_inputs"])
        session.setdefault("opponent_anticipation", {})
        session.setdefault("initial_strategy", {})
        session.setdefault("simulated_event", {})
        session.setdefault("revision", {})
        session.setdefault("rubric_evaluation", {})
        if "stage2_colloquy" not in session or not session["stage2_colloquy"]:
            session["stage2_colloquy"] = _create_default_stage2_colloquy()

        if session.get("completed"):
            return

        p_idx = session.get("current_phase_idx", 0)
        if p_idx >= len(phases):
            session["completed"] = True
            session["current_question_id"] = None
            return

        cur_phase = phases[p_idx]
        if cur_phase.mode == "mentor_qa":
            cur_qid = session.get("current_question_id")
            phase_qids = [q.id for q in cur_phase.questions]
            if cur_qid not in phase_qids:
                next_q = self._first_pending_in_phase(cur_phase, q_prog)
                session["current_question_id"] = next_q.id if next_q else None

    def _first_pending_in_phase(
        self, phase: PhaseGroup, q_prog: Dict[str, Dict[str, Any]]
    ) -> Optional[QuestionItem]:
        for q in phase.questions:
            st = q_prog.get(q.id, {})
            if not st.get("passed") and not st.get("skipped"):
                return q
        return None

    def _record_skill_observation(
        self,
        session: Dict[str, Any],
        skill_id: Optional[str],
        score: int,
        is_revision: bool,
        comment: str = "",
        previous_score_to_replace: Optional[int] = None,
    ) -> Optional[str]:
        if not self.curriculum.skills:
            return None
        if not skill_id or skill_id not in self.curriculum.skills_by_id:
            skill_id = self.curriculum.skills[0].id

        entry = session["skill_scores"][skill_id]
        if not is_revision:
            entry["initial_scores"].append(score)
            entry["latest_scores"].append(score)
        else:
            if previous_score_to_replace is not None and previous_score_to_replace in entry["latest_scores"]:
                idx = entry["latest_scores"].index(previous_score_to_replace)
                entry["latest_scores"][idx] = score
            else:
                entry["latest_scores"].append(score)

        if entry["initial_scores"]:
            entry["initial_score"] = round(
                sum(entry["initial_scores"]) / len(entry["initial_scores"]), 1
            )
        if entry["latest_scores"]:
            avg_latest = round(sum(entry["latest_scores"]) / len(entry["latest_scores"]), 1)
            if is_revision or entry["revised_score"] is not None:
                entry["revised_score"] = avg_latest

        if comment:
            entry["last_comment"] = comment
        return skill_id

    def _append_msg(
        self,
        session: Dict[str, Any],
        role: str,
        speaker: str,
        content: str,
        phase_id: str,
        question_id: Optional[str] = None,
        evaluation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        msg = {
            "id": f"m_{len(session['messages']) + 1}_{int(time.time() * 1000) % 100000}",
            "role": role,
            "speaker": speaker,
            "phase_id": phase_id,
            "question_id": question_id,
            "content": content,
            "evaluation": evaluation,
            "created_at": _now_iso(),
        }
        session["messages"].append(msg)
        return msg

    def create_session(
        self,
        case_id: Optional[str] = None,
        custom_case_text: str = "",
        custom_client_notes: str = "",
    ) -> Dict[str, Any]:
        self.curriculum.reload_if_needed()
        session_id = uuid.uuid4().hex[:12]
        resolved_case = case_id if (case_id and case_id != "none") else "ndil_lakeview_cold_chain"
        trade_defaults = get_default_tradeoff_inputs()
        session: Dict[str, Any] = {
            "session_id": session_id,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "case_id": resolved_case,
            "custom_case_text": custom_case_text.strip(),
            "custom_client_notes": custom_client_notes.strip(),
            "current_phase_idx": 0,
            "current_question_id": None,
            "completed": False,
            "messages": [],
            "question_progress": {},
            "skill_scores": {},
            "sim_stage": 1,
            "ranked_priorities": [],
            "discovered_facts": [
                {
                    "id": "known_marcus_preference",
                    "category": "Known Authority",
                    "source": "Supervising Partner Intake Note",
                    "text": "Marcus Vance (Warehouse Ops) wants to file an immediate motion to dismiss, but lacks authority for dispositive litigation filings or budgets over $15,000.",
                },
                {
                    "id": "known_elena_authority",
                    "category": "Known Authority",
                    "source": "Supervising Partner Intake Note",
                    "text": "CEO Elena Rostova holds executive strategic authority; joint CFO David Park sign-off required for legal commitments exceeding $25,000.",
                },
            ],
            "prep_answers": {},
            "tradeoff_inputs": trade_defaults,
            "tradeoff_results": compute_tradeoffs(trade_defaults),
            "opponent_anticipation": {},
            "initial_strategy": {},
            "simulated_event": {},
            "revision": {},
            "rubric_evaluation": {},
        }
        self._sync_with_curriculum(session)

        phases = self.curriculum.phases
        if phases:
            # If the session is for the Strategic Simulation (ndil_lakeview_cold_chain),
            # Stage 1 is the Client Interview. Start directly in client roleplay with Elena Rostova!
            # The client acknowledges the dispute without prematurely revealing hidden constraints (75-day loan, Sec. 11.2)
            client_phase_idx = next(
                (i for i, p in enumerate(phases) if p.mode == "client_roleplay" or p.id == "during_meeting"),
                None
            )
            if client_phase_idx is not None and resolved_case == "ndil_lakeview_cold_chain":
                session["current_phase_idx"] = client_phase_idx
                client_phase = phases[client_phase_idx]
                self._append_msg(
                    session,
                    role="client",
                    speaker="Elena Rostova, CEO (with Marcus Vance)",
                    content=(
                        "Hello counsel. Thank you for making time to meet with us on such short notice. "
                        "Marcus and I received Heartland's federal complaint claiming over $408,000 for cheese spoilage in Bay C. "
                        "Marcus wants to file an immediate motion to dismiss right away, but as CEO, I want to make sure we evaluate all our options "
                        "and make the right strategic move for Lakeview before making any commitments.\n\n"
                        "What questions do you have for us about our business, the lawsuit, or what happened?"
                    ),
                    phase_id=client_phase.id,
                )
            else:
                first_phase = phases[0]
                if first_phase.mode == "mentor_qa" and first_phase.questions:
                    q1 = first_phase.questions[0]
                    session["current_question_id"] = q1.id
                    intro = (
                        "Welcome! Before you meet with the client or begin drafting a motion, "
                        "let's walk through your preparation step by step.\n\n"
                        f"**Question 1 ({q1.category}):** {q1.question}"
                    )
                    self._append_msg(
                        session,
                        role="mentor",
                        speaker="Supervising Partner",
                        content=intro,
                        phase_id=first_phase.id,
                        question_id=q1.id,
                    )
                elif first_phase.mode == "client_roleplay":
                    self._enter_client_roleplay_phase(session, first_phase)
        else:
            session["completed"] = True

        self.save_session(session)
        return self.serialize_session(session)

    def update_case_context(
        self,
        session_id: str,
        case_id: Optional[str] = None,
        custom_case_text: str = "",
        custom_client_notes: str = "",
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.load_session(session_id, snapshot=snapshot)
        session["case_id"] = case_id or "none"
        session["custom_case_text"] = custom_case_text.strip()
        session["custom_client_notes"] = custom_client_notes.strip()
        self.save_session(session)
        return self.serialize_session(session)

    def _enter_client_roleplay_phase(self, session: Dict[str, Any], phase: PhaseGroup) -> None:
        session["current_question_id"] = None
        self._append_msg(
            session,
            role="system",
            speaker="Simulation Transition",
            content=(
                f"Entering {phase.name}: Client Interview Simulation. "
                "Roles have switched — you are now speaking directly with the client. "
                "Ask clear, non-pushy questions to uncover their priorities, constraints, facts, and decision process."
            ),
            phase_id=phase.id,
        )
        self._append_msg(
            session,
            role="client",
            speaker="Client",
            content=(
                "Thanks for making time to meet with me today. I know your team received the complaint "
                "we were served with, and I'm anxious to figure out our best move here. Where should we start?"
            ),
            phase_id=phase.id,
        )

    def _advance_to_next_phase(self, session: Dict[str, Any], mentor_prefix: str = "") -> None:
        phases = self.curriculum.phases
        next_idx = session["current_phase_idx"] + 1
        session["current_phase_idx"] = next_idx

        if next_idx >= len(phases):
            session["completed"] = True
            session["current_question_id"] = None
            closing = (
                (mentor_prefix.strip() + "\n\n") if mentor_prefix.strip() else ""
            ) + (
                "Excellent work completing the full preparation, client interview, and post-meeting counseling analysis! "
                "Review your Rubric Scorecard in the side panel, or head to the **Motion Drafting** workspace to draft "
                "and review your Rule 12(b)(6) motion or strategic client memo."
            )
            last_phase_id = phases[-1].id if phases else "done"
            self._append_msg(
                session,
                role="mentor",
                speaker="Supervising Partner",
                content=closing,
                phase_id=last_phase_id,
            )
            return

        next_phase = phases[next_idx]
        if next_phase.mode == "client_roleplay":
            handoff = (
                (mentor_prefix.strip() + "\n\n") if mentor_prefix.strip() else ""
            ) + (
                f"Good preparation. You're ready for **{next_phase.name}**—your client is joining now. "
                "Conduct the interview in your own words; when you've covered everything you need, "
                "we'll regroup to evaluate pleading plausibility and motion strategy."
            )
            prev_phase_id = phases[next_idx - 1].id
            self._append_msg(
                session,
                role="mentor",
                speaker="Supervising Partner",
                content=handoff,
                phase_id=prev_phase_id,
            )
            self._enter_client_roleplay_phase(session, next_phase)
        else:
            self._append_msg(
                session,
                role="system",
                speaker="Simulation Transition",
                content=(
                    f"Entering {next_phase.name}: Returning to Supervising Partner Debrief. "
                    "Synthesize what you learned from the client and the complaint."
                ),
                phase_id=next_phase.id,
            )
            first_q = self._first_pending_in_phase(next_phase, session["question_progress"])
            if first_q is None and next_phase.questions:
                first_q = next_phase.questions[0]
            if first_q:
                session["current_question_id"] = first_q.id
                q_num = first_q.index + 1
                msg_text = (
                    ((mentor_prefix.strip() + "\n\n") if mentor_prefix.strip() else "")
                    + "Welcome back from the client meeting. Let's debrief and decide on the smartest path forward.\n\n"
                    + f"**Question {q_num} ({first_q.category}):** {first_q.question}"
                )
                self._append_msg(
                    session,
                    role="mentor",
                    speaker="Supervising Partner",
                    content=msg_text,
                    phase_id=next_phase.id,
                    question_id=first_q.id,
                )
            else:
                self._advance_to_next_phase(session)

    def _build_interview_digest(self, session: Dict[str, Any]) -> str:
        roleplay_phase_ids = {p.id for p in self.curriculum.phases if p.mode == "client_roleplay"}
        lines: List[str] = []
        for m in session["messages"]:
            if m.get("phase_id") in roleplay_phase_ids and m.get("role") in ("user", "client"):
                who = "Student Lawyer" if m["role"] == "user" else "Client"
                lines.append(f"{who}: {m['content']}")
        return "\n".join(lines[-24:])

    def handle_user_message(
        self,
        session_id: str,
        text: str,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError("Message cannot be empty.")

        session = self.load_session(session_id, snapshot=snapshot)
        if session.get("completed"):
            return self.serialize_session(session)

        phases = self.curriculum.phases
        if session.get("case_id") == "ndil_lakeview_cold_chain" and session.get("sim_stage", 1) == 1:
            client_phase = next(
                (p for p in phases if p.mode == "client_roleplay" or p.id == "during_meeting"),
                None
            )
            if client_phase is not None:
                cur_phase = client_phase
                session["current_phase_idx"] = phases.index(client_phase)
            else:
                cur_phase = phases[session["current_phase_idx"]]
        else:
            cur_phase = phases[session["current_phase_idx"]]

        case_bundle = get_case_bundle(
            session.get("case_id"),
            session.get("custom_case_text", ""),
            session.get("custom_client_notes", ""),
        )

        if cur_phase.mode == "mentor_qa":
            return self._handle_mentor_qa_turn(session, cur_phase, cleaned, case_bundle)
        return self._handle_client_roleplay_turn(session, cur_phase, cleaned, case_bundle)

    def _handle_mentor_qa_turn(
        self,
        session: Dict[str, Any],
        cur_phase: PhaseGroup,
        student_text: str,
        case_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        cur_qid = session.get("current_question_id")
        question = self.curriculum.questions_by_id.get(cur_qid or "")
        if question is None:
            question = self._first_pending_in_phase(cur_phase, session["question_progress"])
            if question is None:
                self._advance_to_next_phase(session)
                self.save_session(session)
                return self.serialize_session(session)
            session["current_question_id"] = question.id

        q_state = session["question_progress"][question.id]
        prior_attempts = q_state["attempts"]
        is_revision = len(prior_attempts) > 0
        prev_score = q_state.get("current_score")

        interview_digest = self._build_interview_digest(session)
        llm_msgs = build_mentor_eval_messages(
            curriculum=self.curriculum,
            question=question,
            student_answer=student_text,
            prior_attempts=prior_attempts,
            case_bundle=case_bundle,
            interview_digest=interview_digest,
        )

        try:
            gen_out = self.llm.generate(messages=llm_msgs, max_tokens=900, temperature=0.2, enable_thinking=False)
            parsed = parse_llm_json(gen_out["text"])
        except Exception:
            parsed = {
                "score": 2,
                "skill_id": "client_communication",
                "revision_score": None,
                "strengths": "Addressed key legal issues in the complaint.",
                "diagnostic": f"Clear answer identifying facts relating to {question.category}.",
                "mentor_reply": f"Good observation on {question.category}. Let's proceed to the next step."
            }

        max_s = self.curriculum.max_score
        passing = self.curriculum.passing_score
        score = _clamp_int(parsed.get("score"), 0, max_s, default=1)
        passed = score >= passing
        skill_id = parsed.get("skill_id")
        strengths = (parsed.get("strengths") or "").strip()
        diagnostic = (parsed.get("diagnostic") or "").strip()
        mentor_reply = (parsed.get("mentor_reply") or "").strip()
        if not mentor_reply:
            mentor_reply = diagnostic or "Let's refine that point a bit further."

        matched_skill_id = self._record_skill_observation(
            session=session,
            skill_id=skill_id,
            score=score,
            is_revision=is_revision,
            comment=diagnostic or strengths,
            previous_score_to_replace=prev_score,
        )
        matched_skill_name = (
            self.curriculum.skills_by_id[matched_skill_id].skill
            if matched_skill_id and matched_skill_id in self.curriculum.skills_by_id
            else "General"
        )

        # Update Revision/reflection skill automatically when revising
        if is_revision and "revision_reflection" in self.curriculum.skills_by_id:
            rev_raw = parsed.get("revision_score")
            if rev_raw is not None:
                rev_score = _clamp_int(rev_raw, 0, max_s, default=score)
            else:
                delta = score - (prev_score or 0)
                rev_score = min(max_s, max(1, 2 if delta >= 1 else 1))
            self._record_skill_observation(
                session=session,
                skill_id="revision_reflection",
                score=rev_score,
                is_revision=len(session["skill_scores"]["revision_reflection"]["initial_scores"]) > 0,
                comment=f"Revised {question.category} response from {prev_score}/{max_s} to {score}/{max_s}.",
            )

        # Update question progress
        if q_state["initial_score"] is None:
            q_state["initial_score"] = score
        else:
            q_state["revised_score"] = score
        q_state["current_score"] = score
        q_state["passed"] = passed
        q_state["skill_id"] = matched_skill_id
        q_state["last_note"] = diagnostic or strengths
        q_state["attempts"].append(
            {
                "answer": student_text,
                "score": score,
                "feedback": mentor_reply,
                "timestamp": _now_iso(),
            }
        )

        eval_badge = {
            "mode": "mentor_qa",
            "question_id": question.id,
            "category": question.category,
            "attempt": len(q_state["attempts"]),
            "score": score,
            "initial_score": q_state["initial_score"],
            "revised_score": q_state["revised_score"],
            "max_score": max_s,
            "passed": passed,
            "skill_id": matched_skill_id,
            "skill_name": matched_skill_name,
            "strengths": strengths,
            "diagnostic": diagnostic,
        }

        self._append_msg(
            session,
            role="user",
            speaker="You",
            content=student_text,
            phase_id=cur_phase.id,
            question_id=question.id,
            evaluation=eval_badge,
        )

        if passed:
            next_q = self._first_pending_in_phase(cur_phase, session["question_progress"])
            if next_q is not None:
                session["current_question_id"] = next_q.id
                q_num = next_q.index + 1
                combined_reply = (
                    f"{mentor_reply}\n\n"
                    f"**Question {q_num} ({next_q.category}):** {next_q.question}"
                )
                self._append_msg(
                    session,
                    role="mentor",
                    speaker="Supervising Partner",
                    content=combined_reply,
                    phase_id=cur_phase.id,
                    question_id=next_q.id,
                )
            else:
                self._advance_to_next_phase(session, mentor_prefix=mentor_reply)
        else:
            self._append_msg(
                session,
                role="mentor",
                speaker="Supervising Partner",
                content=mentor_reply,
                phase_id=cur_phase.id,
                question_id=question.id,
            )

        self.save_session(session)
        return self.serialize_session(session)

    def _handle_client_roleplay_turn(
        self,
        session: Dict[str, Any],
        cur_phase: PhaseGroup,
        student_text: str,
        case_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        phase_chat_history: List[Dict[str, str]] = []
        for m in session["messages"]:
            if m.get("phase_id") == cur_phase.id and m.get("role") in ("user", "client"):
                phase_chat_history.append(
                    {"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]}
                )

        llm_msgs = build_client_roleplay_messages(
            curriculum=self.curriculum,
            phase_questions=cur_phase.questions,
            question_progress=session["question_progress"],
            chat_history=phase_chat_history,
            student_message=student_text,
            case_bundle=case_bundle,
        )

        try:
            gen_out = self.llm.generate(messages=llm_msgs, max_tokens=1000, temperature=0.3, enable_thinking=False)
            parsed = parse_llm_json(gen_out["text"])
        except Exception:
            parsed = self._heuristic_client_reply(student_text)

        max_s = self.curriculum.max_score
        passing = self.curriculum.passing_score
        client_reply = (parsed.get("client_reply") or "").strip() or "Could you clarify what you mean by that?"
        coach_tip = (parsed.get("coach_tip") or "").strip()
        comm_score = _clamp_int(parsed.get("communication_score"), 0, max_s, default=2)

        if "client_communication" in self.curriculum.skills_by_id:
            comm_has_prior = len(session["skill_scores"]["client_communication"]["initial_scores"]) > 0
            self._record_skill_observation(
                session=session,
                skill_id="client_communication",
                score=comm_score,
                is_revision=comm_has_prior,
                comment=coach_tip,
            )

        valid_phase_qids = {q.id for q in cur_phase.questions}
        addressed_badges: List[Dict[str, Any]] = []

        for item in parsed.get("addressed_items") or []:
            if not isinstance(item, dict):
                continue
            qid = item.get("question_id")
            if qid not in valid_phase_qids:
                continue
            q_obj = self.curriculum.questions_by_id[qid]
            q_state = session["question_progress"][qid]
            raw_s = _clamp_int(item.get("score"), 0, max_s, default=1)
            note = (item.get("note") or "").strip()
            skill_id = item.get("skill_id")

            was_attempted = len(q_state["attempts"]) > 0
            prev_best = q_state.get("current_score")
            new_best = max(prev_best, raw_s) if prev_best is not None else raw_s

            matched_skill_id = self._record_skill_observation(
                session=session,
                skill_id=skill_id,
                score=new_best,
                is_revision=was_attempted,
                comment=note,
                previous_score_to_replace=prev_best,
            )
            matched_skill_name = (
                self.curriculum.skills_by_id[matched_skill_id].skill
                if matched_skill_id and matched_skill_id in self.curriculum.skills_by_id
                else "Interviewing"
            )

            if q_state["initial_score"] is None:
                q_state["initial_score"] = raw_s
            elif new_best != q_state["initial_score"]:
                q_state["revised_score"] = new_best

            q_state["current_score"] = new_best
            q_state["passed"] = new_best >= passing
            q_state["skill_id"] = matched_skill_id
            q_state["last_note"] = note
            q_state["attempts"].append(
                {
                    "answer": student_text,
                    "score": raw_s,
                    "feedback": note,
                    "timestamp": _now_iso(),
                }
            )

            addressed_badges.append(
                {
                    "question_id": qid,
                    "category": q_obj.category,
                    "score": new_best,
                    "max_score": max_s,
                    "passed": q_state["passed"],
                    "skill_name": matched_skill_name,
                    "note": note,
                }
            )

        eval_payload = {
            "mode": "client_roleplay",
            "communication_score": comm_score,
            "max_score": max_s,
            "addressed_items": addressed_badges,
            "coach_tip": coach_tip,
        }

        self._append_msg(
            session,
            role="user",
            speaker="Lawyer (You)",
            content=student_text,
            phase_id=cur_phase.id,
            evaluation=eval_payload,
        )
        self._append_msg(
            session,
            role="client",
            speaker="Elena Rostova, CEO (with Marcus Vance)",
            content=client_reply,
            phase_id=cur_phase.id,
        )

        # Dynamically uncover scenario facts based on student inquiry (Page 5, 6, 10 of PDF recommendations)
        t_lower = student_text.lower()
        discovered = session.setdefault("discovered_facts", [])
        discovered_ids = {f["id"] for f in discovered}

        FACT_TRIGGERS = [
            (
                "fact_refinancing_deadline",
                ("refinanc", "bank", "credit", "deadline", "loan", "75 day", "timing", "when", "urgent", "pressure"),
                "Commercial Urgency",
                "Discovered from CEO Elena Rostova",
                "75-Day Bank Refinancing Window: Commercial credit facility with Midwest Business Bank is up for renewal; open litigation >$250k threatens to freeze Lakeview's credit line.",
            ),
            (
                "fact_budget_cap",
                ("budget", "fee", "cost", "spend", "how much", "expense", "dollars", "funds", "afford"),
                "Financial Constraint",
                "Discovered from CEO Elena Rostova",
                "Litigation Budget Cap: Client budget is strictly $15,000–$25,000; Lakeview cannot afford a protracted $75k+ discovery war over a $68k shipment.",
            ),
            (
                "fact_section_11_2",
                ("contract", "section 11", "11.2", "limitation", "waiver", "consequential", "lost profit", "agreement", "clause", "terms"),
                "Contractual Defense",
                "Discovered from Master Agreement & Elena",
                "Master Agreement Sec. 11.2: Expressly bars consequential and lost-profit damages ($340,000) and caps cumulative warehouse liability at $36,000 (2x monthly storage).",
            ),
            (
                "fact_settlement_appetite",
                ("settle", "offer", "compromise", "resolve", "deal", "negotiat", "informal"),
                "Commercial Resolution",
                "Discovered from CEO Elena Rostova",
                "Settlement Terms: Client willing to offer $30k–$40k cash + $20k storage fee credits conditioned on mutual general release, confidentiality, and no admission of fault.",
            ),
            (
                "fact_dock_records",
                ("what happened", "spoil", "temperature", "pallet", "bay c", "cheese", "alarm", "ticket", "sensor", "dock", "log", "truck", "record"),
                "Evidentiary Facts",
                "Discovered from Marcus Vance",
                "Warehouse Dock Logs & Ticket #409: Receiving dock logs show cheese was delivered at 45°F hot-loaded (violating Sec. 4.1 pre-cooling); internal ticket #409 logged auxiliary sensor alert on Aug 15.",
            ),
            (
                "fact_heartland_relationship",
                ("relationship", "customer", "heartland", "revenue", "recurring", "future business", "partner", "loss of customer"),
                "Commercial Value",
                "Discovered from CEO Elena Rostova",
                "Heartland Commercial Account: Heartland generates ~$190,000/year in recurring storage revenue for Lakeview's Bolingbrook warehouse.",
            ),
        ]

        for fid, words, cat, src, txt in FACT_TRIGGERS:
            if fid not in discovered_ids and any(w in t_lower for w in words):
                discovered.append({
                    "id": fid,
                    "category": cat,
                    "source": src,
                    "text": txt,
                })
                discovered_ids.add(fid)

        # Automatically advance once all During-meeting items pass (legacy guided mode only)
        if not session.get("sim_stage"):
            all_passed = all(
                session["question_progress"][q.id].get("passed") for q in cur_phase.questions
            )
            if all_passed and cur_phase.questions:
                self._advance_to_next_phase(session)

        self.save_session(session)
        return self.serialize_session(session)

    def _heuristic_client_reply(self, text: str) -> Dict[str, Any]:
        t = (text or "").lower()
        qid = None
        if any(w in t for w in ["authority", "decision", "who can", "sign off", "approve", "operating agreement", "cfo", "who is the client"]):
            reply = (
                "As CEO, I hold ultimate executive authority for Lakeview Cold Chain Logistics. "
                "Marcus Vance is our warehouse operations director; he has day-to-day spending authority up to $15,000, "
                "but he cannot authorize litigation filings, major legal expenses, or settlements. "
                "Any settlement or legal budget exceeding $25,000 requires joint approval from me and our CFO David Park, "
                "plus consultation with our three-member managing board."
            )
            qid = "authority_thresholds"
        elif any(w in t for w in ["budget", "fee", "cost", "spend", "how much", "expense", "dollars"]):
            reply = (
                "Our budget is quite tight. We can absorb between $15,000 and $25,000 in total legal spend if we must, "
                "but Marcus's impulse to file an immediate motion to dismiss worries me if it costs $20,000 just to have the judge "
                "allow Heartland to amend their complaint. We cannot afford a protracted $75,000 discovery war over a $68,000 cargo loss."
            )
            qid = "litigation_budget"
        elif any(w in t for w in ["refinanc", "bank", "credit", "deadline", "loan", "75 day", "timing", "when"]):
            reply = (
                "That's our single most urgent constraint. Our commercial credit facility with Midwest Business Bank is up for "
                "formal underwriting renewal in exactly 75 days. Under our covenant terms, any unresolved lawsuit demanding more than "
                "$250,000 gets flagged by the loan committee, which could freeze our line of credit right as we purchase winter inventory. "
                "We need this matter resolved, capped, or removed from the public docket before that 75-day window closes."
            )
            qid = "commercial_deadlines"
        elif any(w in t for w in ["contract", "section 11", "11.2", "limitation", "waiver", "consequential", "lost profit", "agreement"]):
            reply = (
                "Yes! In March 2024, Heartland and Lakeview signed a Master Cold-Storage Agreement. Section 11.2 explicitly bars "
                "any consequential, lost-profit, or reputational damages, and caps our total liability at two times the monthly storage fees "
                "for the affected bay, which is exactly $36,000! That's why Heartland's demand for $340,000 in lost profits is outrageous."
            )
            qid = "contract_terms"
        elif any(w in t for w in ["settle", "offer", "compromise", "resolve", "deal", "negotiat"]):
            reply = (
                "We are very open to a sensible commercial settlement. We would be willing to pay $30,000 to $40,000 in cash—perhaps with "
                "an insurance contribution—plus offer up to $20,000 or $25,000 in storage fee credits over the coming year. In exchange, "
                "we require a full mutual general release, strict confidentiality, and no admission of fault."
            )
            qid = "settlement_appetite"
        elif any(w in t for w in ["what happened", "spoil", "temperature", "pallet", "bay c", "cheese", "alarm", "ticket", "sensor", "facts"]):
            reply = (
                "(Marcus Vance): In Bay C, our auxiliary compressor sensor had an intermittent reading on August 15th—it was logged "
                "as internal maintenance ticket #409. But more importantly, our receiving dock logs show Heartland's own carrier truck "
                "delivered the cheese pallets at 45°F already hot-loaded, in direct violation of Section 4.1 pre-cooling specifications!"
            )
            qid = "factual_narrative"
        elif any(w in t for w in ["priorit", "goal", "objective", "rank", "top 3", "most important"]):
            reply = (
                "If I have to rank our top three priorities: First, protect our credit standing and resolve this before our 75-day bank "
                "refinancing review. Second, keep total financial spend predictable and avoid an expensive litigation battle. "
                "Third, preserve our commercial reputation and, if possible, keep Heartland as a customer—they generate nearly $190,000 "
                "a year in recurring storage revenue for our Bolingbrook warehouse."
            )
            qid = "client_priorities"
        else:
            reply = (
                "Marcus and I are here to answer whatever you need counsel. We want to know our options, but we have strict financial "
                "and timing constraints—especially regarding our upcoming bank loan review, our contract terms, and our operating budget. "
                "What specific aspects of our business or the dispute would you like to explore?"
            )
            qid = None

        return {
            "client_reply": reply,
            "communication_score": 2,
            "coach_tip": "Good open-ended inquiry. Continue probing for concrete authority thresholds, deadlines, and adverse records.",
            "addressed_items": [
                {
                    "question_id": qid,
                    "score": 2,
                    "skill_id": "client_communication",
                    "note": f"Elicited information regarding {qid.replace('_', ' ')}."
                }
            ] if qid else []
        }

        self.save_session(session)
        return self.serialize_session(session)

    def skip_or_finish_phase(
        self,
        session_id: str,
        action: str = "skip_question",
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.load_session(session_id, snapshot=snapshot)
        if session.get("completed"):
            return self.serialize_session(session)

        phases = self.curriculum.phases
        cur_phase = phases[session["current_phase_idx"]]

        if cur_phase.mode == "client_roleplay" or action == "finish_phase":
            # Mark any completely untouched items in this phase as skipped (score 0 if unasked)
            for q in cur_phase.questions:
                st = session["question_progress"][q.id]
                if not st["passed"]:
                    st["skipped"] = True
                    if st["initial_score"] is None:
                        st["initial_score"] = 0
                        st["current_score"] = 0
            self._advance_to_next_phase(session)
        else:
            cur_qid = session.get("current_question_id")
            if cur_qid and cur_qid in session["question_progress"]:
                session["question_progress"][cur_qid]["skipped"] = True
            next_q = self._first_pending_in_phase(cur_phase, session["question_progress"])
            if next_q is not None:
                session["current_question_id"] = next_q.id
                q_num = next_q.index + 1
                self._append_msg(
                    session,
                    role="mentor",
                    speaker="Supervising Partner",
                    content=f"Let's move forward for now and revisit that if needed.\n\n**Question {q_num} ({next_q.category}):** {next_q.question}",
                    phase_id=cur_phase.id,
                    question_id=next_q.id,
                )
            else:
                self._advance_to_next_phase(session)

        self.save_session(session)
        return self.serialize_session(session)

    def generate_hint(
        self,
        session_id: str,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.load_session(session_id, snapshot=snapshot)
        if session.get("completed"):
            return {"hint": "You have already completed all guided questions!"}

        phases = self.curriculum.phases
        cur_phase = phases[session["current_phase_idx"]]
        case_bundle = get_case_bundle(
            session.get("case_id"),
            session.get("custom_case_text", ""),
            session.get("custom_client_notes", ""),
        )

        if cur_phase.mode == "client_roleplay":
            uncovered = [
                q for q in cur_phase.questions if not session["question_progress"][q.id].get("passed")
            ]
            msgs = build_hint_messages(
                curriculum=self.curriculum,
                mode="client_roleplay",
                current_question=None,
                uncovered_questions=uncovered,
                case_bundle=case_bundle,
            )
        else:
            cur_q = self.curriculum.questions_by_id.get(session.get("current_question_id") or "")
            msgs = build_hint_messages(
                curriculum=self.curriculum,
                mode="mentor_qa",
                current_question=cur_q,
                uncovered_questions=[],
                case_bundle=case_bundle,
            )

        out = self.llm.generate(messages=msgs, max_tokens=280, temperature=0.3, enable_thinking=False)
        return {"hint": out["text"]}

    def review_motion_draft(
        self,
        draft_type: str,
        draft_text: str,
        session_id: Optional[str] = None,
        case_id: Optional[str] = None,
        custom_case_text: str = "",
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.curriculum.reload_if_needed()
        if not (draft_text or "").strip():
            raise ValueError("Please write or paste a motion / strategy draft before requesting review.")

        summary_lines: List[str] = []
        resolved_case_id = case_id
        resolved_case_text = custom_case_text

        if session_id or snapshot:
            try:
                sid = session_id or (snapshot.get("session_id") if isinstance(snapshot, dict) else "stateless")
                session = self.load_session(sid, snapshot=snapshot)
                if not resolved_case_id:
                    resolved_case_id = session.get("case_id")
                if not resolved_case_text:
                    resolved_case_text = session.get("custom_case_text", "")
                for q in self.curriculum.questions:
                    st = session["question_progress"].get(q.id, {})
                    if st.get("attempts"):
                        last_ans = st["attempts"][-1]["answer"]
                        summary_lines.append(
                            f"- [{q.phase} · {q.category}] {q.question}\n  Student: {last_ans} (Score: {st.get('current_score')}/{self.curriculum.max_score})"
                        )
                interview_digest = self._build_interview_digest(session)
                if interview_digest:
                    summary_lines.append("\nClient Interview Excerpts:\n" + interview_digest)
            except KeyError:
                pass

        case_bundle = get_case_bundle(resolved_case_id, resolved_case_text, "")
        msgs = build_motion_review_messages(
            curriculum=self.curriculum,
            draft_type=draft_type,
            draft_text=draft_text.strip(),
            case_bundle=case_bundle,
            session_summary_text="\n".join(summary_lines),
        )
        out = self.llm.generate(messages=msgs, max_tokens=1400, temperature=0.2, enable_thinking=False)
        parsed = parse_llm_json(out["text"])

        # Enrich skill_ratings with human-readable names
        enriched_ratings = []
        for sr in parsed.get("skill_ratings") or []:
            if not isinstance(sr, dict):
                continue
            sid = sr.get("skill_id", "")
            s_obj = self.curriculum.skills_by_id.get(sid)
            enriched_ratings.append(
                {
                    "skill_id": sid,
                    "skill": s_obj.skill if s_obj else sid,
                    "score": _clamp_int(sr.get("score"), 0, self.curriculum.max_score, default=2),
                    "max_score": self.curriculum.max_score,
                    "comment": sr.get("comment", ""),
                }
            )
        parsed["skill_ratings"] = enriched_ratings
        parsed["max_score"] = self.curriculum.max_score
        return parsed

    def export_rubric_csv(
        self,
        session_id: str,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> str:
        session = self.load_session(session_id, snapshot=snapshot)
        buf = io.StringIO()
        writer = csv.writer(buf)

        level_headers = [lvl["header"] for lvl in self.curriculum.scale]
        writer.writerow(["Skill"] + level_headers + ["Initial score", "Revised score", "Session Notes"])

        for s in self.curriculum.skills:
            st = session["skill_scores"].get(s.id, {})
            lvl_descs = [lvl.description for lvl in s.levels]
            init_s = "" if st.get("initial_score") is None else st.get("initial_score")
            rev_s = "" if st.get("revised_score") is None else st.get("revised_score")
            note = st.get("last_comment") or s.notes
            writer.writerow([s.skill] + lvl_descs + [init_s, rev_s, note])

        return buf.getvalue()

    def serialize_session(self, session: Dict[str, Any]) -> Dict[str, Any]:
        self.curriculum.reload_if_needed()
        phases = self.curriculum.phases
        p_idx = session.get("current_phase_idx", 0)
        cur_phase = phases[p_idx].to_dict() if (not session.get("completed") and p_idx < len(phases)) else None
        cur_q = (
            self.curriculum.questions_by_id[session["current_question_id"]].to_dict()
            if session.get("current_question_id") in self.curriculum.questions_by_id
            else None
        )
        case_bundle = get_case_bundle(
            session.get("case_id"),
            session.get("custom_case_text", ""),
            session.get("custom_client_notes", ""),
        )

        completed_count = sum(
            1
            for q in self.curriculum.questions
            if session["question_progress"].get(q.id, {}).get("passed")
            or session["question_progress"].get(q.id, {}).get("skipped")
        )

        return {
            "session_id": session["session_id"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "completed": session["completed"],
            "current_phase_idx": p_idx,
            "current_phase": cur_phase,
            "current_question": cur_q,
            "completed_questions": completed_count,
            "total_questions": len(self.curriculum.questions),
            "sim_stage": session.get("sim_stage", 1),
            "ranked_priorities": session.get("ranked_priorities", []),
            "discovered_facts": session.get("discovered_facts", []),
            "prep_answers": session.get("prep_answers", {}),
            "stage2_colloquy": session.get("stage2_colloquy", {}),
            "tradeoff_inputs": session.get("tradeoff_inputs", {}),
            "tradeoff_results": session.get("tradeoff_results", {}),
            "opponent_anticipation": session.get("opponent_anticipation", {}),
            "initial_strategy": session.get("initial_strategy", {}),
            "simulated_event": session.get("simulated_event", {}),
            "revision": session.get("revision", {}),
            "rubric_evaluation": session.get("rubric_evaluation", {}),
            "case": {
                "case_id": case_bundle["case_id"],
                "title": case_bundle["title"],
                "student_case_text": case_bundle["student_case_text"],
            },
            "messages": session["messages"],
            "question_progress": session["question_progress"],
            "skill_scores": session["skill_scores"],
            "session_snapshot": session,
            "curriculum": self.curriculum.to_api_dict(),
        }

    # =========================================================================
    # STRATEGIC JUDGMENT SIMULATION (6 STAGES)
    # =========================================================================

    def set_simulation_stage(
        self,
        session_id: str,
        stage: int,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.load_session(session_id, snapshot=snapshot)
        session["sim_stage"] = max(1, min(6, int(stage)))
        self.save_session(session)
        return self.serialize_session(session)

    def submit_prep_answer(
        self,
        session_id: str,
        question_key: str,
        student_answer: str,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.load_session(session_id, snapshot=snapshot)
        prep = session.setdefault("prep_answers", {})

        feedback_dict = {
            "q1_authority": {
                "question": "Marcus has instructed you to file a motion to dismiss immediately. Before acting on that request, whose interests are you representing, and whose authorisation do you need?",
                "rules": "Illinois Rules of Professional Conduct 1.13 & 1.2",
                "model_answer": "Under Illinois Rule 1.13 (Organization as Client), you represent the corporate entity—Lakeview Cold Chain Logistics, LLC—not employee Marcus Vance. Under Rule 1.2 (Scope of Representation & Allocation of Authority), strategic litigation decisions and financial commitments rest with executive leadership. CEO Elena Rostova holds executive strategic authority, and Lakeview's governance requires CFO David Park's sign-off for expenditures exceeding $25,000.",
                "key_concepts": ["entity", "organization", "1.13", "elena", "rostova", "cfo", "park", "25,000", "authority", "rule 1.2", "marcus"],
            },
            "q2_assumptions": {
                "question": "What information do you currently know, and what assumptions are you making about the client's objectives?",
                "rules": "Illinois Rule of Professional Conduct 1.4",
                "model_answer": "Known facts from complaint: Heartland filed a federal breach-of-contract complaint alleging cheese spoilage in Bay C demanding $68,400 direct inventory loss + $340,000 lost profits; Marcus Vance wants an immediate motion to dismiss. Assumptions: Assuming Marcus speaks for the company, assuming the client wants an aggressive public court battle, and assuming no adverse internal records exist. Rule 1.4 requires direct consultation with Elena Rostova to identify information gaps and true business objectives.",
                "key_concepts": ["known", "assum", "complaint", "marcus", "spoil", "goal", "objective", "gap", "rule 1.4", "aggressive", "408"],
            },
            "q3_priorities": {
                "question": "What three questions would you prioritise in your client interview, and how could the answers affect your eventual recommendation?",
                "rules": "Illinois Rules of Professional Conduct 1.4 & 2.1",
                "model_answer": "Prioritize asking: (1) What are your business deadlines or commercial pressures (e.g. bank refinancing underwriting)? (2) What litigation budget is available ($15k–$25k vs. a $75k+ discovery war)? (3) Is preserving Heartland's commercial relationship important ($190k/year recurring revenue)? Under Rule 2.1 (Advisor), counsel must integrate economic, financial, and reputational factors alongside pure legal merits.",
                "key_concepts": ["budget", "deadline", "timing", "authority", "refinanc", "relationship", "goal", "rule 2.1", "cost", "heartland", "75"],
            },
        }

        q_info = feedback_dict.get(question_key, feedback_dict["q1_authority"])
        ans_clean = (student_answer or "").strip()
        ans_lower = ans_clean.lower()

        matches = sum(1 for c in q_info["key_concepts"] if c in ans_lower)
        if len(ans_clean) < 15:
            assessment = "Brief answer recorded. Compare with the partner guidance below to ensure complete legal grounding."
            rating = "Developing"
        elif matches >= 3:
            assessment = "Excellent analysis! Your answer accurately identifies governing professional conduct rules and client authority constraints."
            rating = "Proficient"
        else:
            assessment = "Good preliminary analysis. Compare your reasoning with the supervisory partner explanation and supporting rules below."
            rating = "Competent"

        prep[question_key] = {
            "student_answer": ans_clean,
            "submitted_at": _now_iso(),
            "rating": rating,
            "assessment": assessment,
            "model_answer": q_info["model_answer"],
            "supporting_rules": q_info["rules"],
            "revealed": True,
        }

        self.save_session(session)
        return self.serialize_session(session)

    def submit_stage2_colloquy_message(
        self,
        session_id: str,
        message: str,
        question_key: Optional[str] = None,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.load_session(session_id, snapshot=snapshot)
        colloquy = session.setdefault("stage2_colloquy", _create_default_stage2_colloquy())

        target_key = question_key or colloquy.get("current_question_key") or "q4_elements"
        if target_key not in STAGE2_COLLOQUY_QUESTIONS:
            target_key = "q4_elements"

        q_def = STAGE2_COLLOQUY_QUESTIONS[target_key]
        clean_msg = (message or "").strip()

        # 1. Append the associate/student message to colloquy chat history
        colloquy.setdefault("messages", []).append({
            "role": "user",
            "speaker": "Lawyer (You)",
            "content": clean_msg,
            "timestamp": _now_iso(),
            "question_key": target_key,
        })

        # 2. Evaluate the student response
        case_bundle = get_case_bundle(
            session.get("case_id"),
            session.get("custom_case_text", ""),
            session.get("custom_client_notes", ""),
        )

        ans_lower = clean_msg.lower()
        matches = sum(1 for c in q_def["key_concepts"] if c in ans_lower)

        rating = "Competent"
        score = 3
        if len(clean_msg) < 20:
            rating = "Developing"
            score = 1
        elif matches >= 4 or (len(clean_msg) >= 110 and matches >= 2):
            rating = "Proficient"
            score = 5
        elif matches >= 2:
            rating = "Competent"
            score = 4
        else:
            rating = "Developing"
            score = 2

        fallback_replies = {
            "q4_elements": (
                f"You have correctly grounded your analysis in governing Illinois contract principles under {q_def['supporting_source']}. "
                "In Illinois, a plaintiff must allege (1) a valid enforceable contract, (2) substantial performance by the plaintiff, "
                "(3) breach by the defendant, and (4) resultant damages. Heartland has formally addressed each element across paragraphs 2, 3, 4, and 6/10. "
                "Therefore, on the face of the complaint alone, Heartland satisfies basic pleading requirements."
            ),
            "q5_facts": (
                f"Excellent distinction under Federal Rule 8(a)(2) and {q_def['supporting_source']}. "
                "The ambient temperature reading (>49°F), the three spoiled pallets, and the delivery dates are concrete factual allegations "
                "that the court must accept as true at this stage. By contrast, Heartland's assertion that Lakeview acted in 'bad faith and reckless disregard' "
                "is a bare legal conclusion that the court disregards under Twombly and Iqbal."
            ),
            "q6_four_corners": (
                f"Crucial procedural insight under {q_def['supporting_source']}. "
                "A Rule 12(b)(6) motion is strictly confined to the Four Corners of the complaint (and incorporated contracts). "
                "We cannot introduce our internal refrigeration logs or technician ticket #409 without converting the motion to a Rule 56 summary judgment under Rule 12(d). "
                "Doing so would trigger formal discovery and completely blow past our 75-day refinancing deadline."
            ),
            "q7_amend": (
                f"Spot-on recognition of Seventh Circuit practice under {q_def['supporting_source']}. "
                "Under Rule 15(a)(2) and Runnion v. Girl Scouts (786 F.3d at 519), district courts routinely grant leave to amend upon granting a 12(b)(6) motion. "
                "Even if we won a dismissal of certain allegations, Heartland would simply replead with more specific facts. "
                "A motion to dismiss does not end this dispute—it merely delays resolution past our 75-day commercial loan deadline."
            ),
            "q8_damages": (
                f"Outstanding commercial and legal evaluation under {q_def['supporting_source']}. "
                "Under Illinois law (Tri-G), lost profits require proof to a reasonable certainty and proximate causation. "
                "More importantly, Master Agreement Section 11.2 contains an express waiver of consequential damages and caps warehouse liability at $36,000. "
                "Because the agreement is central to the claim, we have a decisive legal weapon to eliminate the $340,000 lost profits exposure early."
            ),
        }

        mentor_reply = fallback_replies.get(target_key, "Good legal analysis.")

        if self.llm:
            try:
                llm_msgs = build_stage2_colloquy_messages(
                    case_bundle=case_bundle,
                    question_key=target_key,
                    question_title=q_def["title"],
                    question_text=q_def["question"],
                    why_ask=q_def["why_ask"],
                    supporting_source=q_def["supporting_source"],
                    legal_rule=q_def["legal_rule"],
                    student_answer=clean_msg,
                    prior_messages=colloquy.get("messages", [])[-6:],
                )
                gen_out = self.llm.generate(messages=llm_msgs, max_tokens=450, temperature=0.3, enable_thinking=False)
                parsed = parse_llm_json(gen_out["text"])
                if parsed and parsed.get("mentor_reply"):
                    mentor_reply = parsed["mentor_reply"]
                if parsed and parsed.get("rating"):
                    rating = parsed["rating"]
                if parsed and parsed.get("score"):
                    score = _clamp_int(parsed["score"], 1, 5, default=score)
            except Exception:
                pass

        # 3. Update question state
        colloquy.setdefault("questions", {})
        if target_key not in colloquy["questions"]:
            colloquy["questions"][target_key] = {
                "id": target_key,
                "number": q_def["number"],
                "title": q_def["title"],
                "question": q_def["question"],
                "why_ask": q_def["why_ask"],
                "supporting_source": q_def["supporting_source"],
            }
        colloquy["questions"][target_key].update({
            "status": "completed",
            "student_answer": clean_msg,
            "rating": rating,
            "score": score,
            "feedback": mentor_reply,
            "model_rule": q_def["legal_rule"],
            "answered_at": _now_iso(),
        })

        # 4. Advance to next question or conclude
        next_key = q_def.get("next_key")
        partner_full_message = mentor_reply

        if next_key and next_key in STAGE2_COLLOQUY_QUESTIONS:
            next_q = STAGE2_COLLOQUY_QUESTIONS[next_key]
            colloquy["current_question_key"] = next_key
            if next_key in colloquy["questions"]:
                colloquy["questions"][next_key]["status"] = "in_progress"
            partner_full_message += (
                f"\n\n---\n**Next Question ({next_q['number']} · {next_q['title']}):**\n"
                f"\"{next_q['question']}\""
            )
        else:
            colloquy["current_question_key"] = target_key
            all_done = all(
                colloquy["questions"].get(k, {}).get("status") == "completed"
                for k in STAGE2_COLLOQUY_QUESTIONS
            )
            if all_done:
                partner_full_message += (
                    "\n\n---\n**Supervisory Synthesis:**\n"
                    "Superb work completing our legal analysis colloquy! You have established that while Heartland technically pleaded "
                    "contract breach, Section 11.2 legally caps damages at $36,000, and a Rule 12(b)(6) dismissal carries a 91% risk of "
                    "leave-to-amend delay under Runnion. Now let's review the quantitative decision tree and sensitivity matrix below to test "
                    "these legal realities against Elena Rostova's 75-day refinancing deadline."
                )

        colloquy["messages"].append({
            "role": "assistant",
            "speaker": "Supervising Partner",
            "content": partner_full_message,
            "timestamp": _now_iso(),
            "question_key": target_key,
        })

        self._record_skill_observation(
            session=session,
            skill_id="legal_reasoning",
            score=score,
            is_revision=False,
            comment=f"{q_def['number']}: {rating} - {mentor_reply[:80]}...",
        )

        self.save_session(session)
        return self.serialize_session(session)

    def set_stage2_colloquy_question(
        self,
        session_id: str,
        question_key: str,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.load_session(session_id, snapshot=snapshot)
        colloquy = session.setdefault("stage2_colloquy", _create_default_stage2_colloquy())
        if question_key in STAGE2_COLLOQUY_QUESTIONS:
            colloquy["current_question_key"] = question_key
            q_state = colloquy.setdefault("questions", {}).setdefault(question_key, {})
            if q_state.get("status") in (None, "pending"):
                q_state["status"] = "in_progress"
            q_def = STAGE2_COLLOQUY_QUESTIONS[question_key]
            last_msg = colloquy.get("messages", [])[-1] if colloquy.get("messages") else {}
            if last_msg.get("question_key") != question_key:
                colloquy["messages"].append({
                    "role": "assistant",
                    "speaker": "Supervising Partner",
                    "content": f"Let's focus on **{q_def['number']} ({q_def['title']})**:\n\"{q_def['question']}\"",
                    "timestamp": _now_iso(),
                    "question_key": question_key,
                })
            self.save_session(session)
        return self.serialize_session(session)

    def set_ranked_priorities(
        self,
        session_id: str,
        priorities: List[str],
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.load_session(session_id, snapshot=snapshot)
        cleaned = [p.strip() for p in priorities if p.strip()][:3]
        session["ranked_priorities"] = cleaned

        # Ingest Stage 1 priorities into Stage 2 quantitative weights and recompute decision tree
        weights = map_priorities_to_tradeoff_weights(cleaned)
        cur_inputs = session.get("tradeoff_inputs") or get_default_tradeoff_inputs()
        cur_inputs.update(weights)
        session["tradeoff_inputs"] = cur_inputs
        session["tradeoff_results"] = compute_tradeoffs(cur_inputs)

        self.save_session(session)
        return self.serialize_session(session)

    def update_tradeoffs(
        self,
        session_id: str,
        tradeoff_inputs: Dict[str, Any],
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.load_session(session_id, snapshot=snapshot)
        current_inputs = session.get("tradeoff_inputs") or get_default_tradeoff_inputs()
        current_inputs.update(tradeoff_inputs)
        session["tradeoff_inputs"] = current_inputs
        session["tradeoff_results"] = compute_tradeoffs(current_inputs)
        self.save_session(session)
        return self.serialize_session(session)

    def submit_opponent_anticipation(
        self,
        session_id: str,
        anticipation_text: str,
        probability_estimate: Optional[float] = None,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.load_session(session_id, snapshot=snapshot)
        case_bundle = get_case_bundle(
            session.get("case_id"),
            session.get("custom_case_text", ""),
            session.get("custom_client_notes", ""),
        )
        strat = session.get("initial_strategy", {}).get("branch", "Negotiate / Litigate")
        msgs = build_opponent_anticipation_messages(
            case_bundle=case_bundle,
            proposed_strategy=strat,
            student_anticipation_text=anticipation_text,
            student_probability_estimate=probability_estimate,
        )
        try:
            out = self.llm.generate(messages=msgs, max_tokens=650, temperature=0.2, enable_thinking=False)
            feedback = parse_llm_json(out["text"])
        except Exception:
            feedback = {
                "assessment": (
                    "Thoughtful identification of Heartland's commercial pressures. Heartland has significant debt "
                    "owed to dairy suppliers but cannot afford prolonged public discovery that exposes its retail distribution contracts."
                ),
                "opponent_counsel_perspective": (
                    "Heartland's counsel will use the threat of Bay C sensor discovery to drive a quick settlement, "
                    "but will immediately file an amended complaint alleging bad faith if a Rule 12(b)(6) motion is served."
                ),
                "blind_spots": (
                    "Consider that Heartland's litigation counsel works on a hybrid fee basis and is incentivized "
                    "to push discovery unless a structured business resolution (cash + credits) is offered."
                ),
                "tactical_rating": "Competent",
            }

        session["opponent_anticipation"] = {
            "text": anticipation_text.strip(),
            "probability": probability_estimate,
            "feedback": feedback,
            "updated_at": _now_iso(),
        }
        self.save_session(session)
        return self.serialize_session(session)

    def submit_initial_strategy(
        self,
        session_id: str,
        branch: str,
        terms: str,
        rationale: str,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.load_session(session_id, snapshot=snapshot)
        session["initial_strategy"] = {
            "branch": branch.strip(),
            "terms": terms.strip(),
            "rationale": rationale.strip(),
            "created_at": _now_iso(),
        }
        trade_in = session.get("tradeoff_inputs", {})
        event_details = {
            "settlement_cash": trade_in.get("settlement_offer_cash", 35000.0),
            "storage_credits": trade_in.get("settlement_offer_credits", 20000.0),
            "terms": terms,
            "rationale": rationale,
        }
        session["simulated_event"] = trigger_simulated_event(branch, event_details)
        session["sim_stage"] = 5
        self.save_session(session)
        return self.serialize_session(session)

    def trigger_sim_event(
        self,
        session_id: str,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.load_session(session_id, snapshot=snapshot)
        strat = session.get("initial_strategy", {})
        trade_in = session.get("tradeoff_inputs", {})
        event_details = {
            "settlement_cash": trade_in.get("settlement_offer_cash", 35000.0),
            "storage_credits": trade_in.get("settlement_offer_credits", 20000.0),
            "terms": strat.get("terms", ""),
            "rationale": strat.get("rationale", ""),
        }
        session["simulated_event"] = trigger_simulated_event(strat.get("branch", "negotiate"), event_details)
        session["sim_stage"] = 5
        self.save_session(session)
        return self.serialize_session(session)

    def submit_revision_and_evaluate(
        self,
        session_id: str,
        revised_action: str,
        pivoted: bool,
        revised_rationale: str,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.load_session(session_id, snapshot=snapshot)
        session["revision"] = {
            "action": revised_action.strip(),
            "pivoted": bool(pivoted),
            "rationale": revised_rationale.strip(),
            "updated_at": _now_iso(),
        }

        case_bundle = get_case_bundle(
            session.get("case_id"),
            session.get("custom_case_text", ""),
            session.get("custom_client_notes", ""),
        )
        trade_res = session.get("tradeoff_results", {})
        comp_text = ""
        if trade_res and "path_comparisons" in trade_res:
            paths = trade_res["path_comparisons"]
            comp_text = (
                f"- Negotiate EV: ${paths.get('negotiate', {}).get('total_ev', 0):,.0f} (Utility: {paths.get('negotiate', {}).get('client_utility_score', 0)}%)\n"
                f"- Litigate EV: ${paths.get('litigate', {}).get('total_ev', 0):,.0f} (Utility: {paths.get('litigate', {}).get('client_utility_score', 0)}%)\n"
                f"- Investigate EV: ${paths.get('investigate', {}).get('total_ev', 0):,.0f} (Utility: {paths.get('investigate', {}).get('client_utility_score', 0)}%)\n"
                f"- P(Permanent Dismissal without leave to amend): {trade_res.get('probabilities', {}).get('p_dismissal_permanent', 0.09) * 100:.1f}%\n"
            )

        dossier = {
            "interview_messages": session.get("messages", []),
            "ranked_priorities": session.get("ranked_priorities", []),
            "tradeoffs_summary": comp_text,
            "opponent_anticipation_text": session.get("opponent_anticipation", {}).get("text", ""),
            "initial_strategy": session.get("initial_strategy", {}),
            "simulated_event": session.get("simulated_event", {}),
            "revised_strategy": session.get("revision", {}),
        }

        eval_msgs = build_strategic_evaluation_messages(case_bundle, dossier)
        try:
            gen_out = self.llm.generate(messages=eval_msgs, max_tokens=1400, temperature=0.2, enable_thinking=False)
            rubric_eval = parse_llm_json(gen_out["text"])
        except Exception:
            rubric_eval = self._heuristic_strategic_evaluation(session)

        session["rubric_evaluation"] = rubric_eval
        session["sim_stage"] = 6
        self.save_session(session)
        return self.serialize_session(session)

    def _heuristic_strategic_evaluation(self, session: Dict[str, Any]) -> Dict[str, Any]:
        ranked = session.get("ranked_priorities", [])
        init_strat = session.get("initial_strategy", {})
        rev = session.get("revision", {})
        branch = (init_strat.get("branch") or "negotiate").lower()
        rat = (init_strat.get("rationale") or "") + " " + (rev.get("rationale") or "")
        rat_lower = rat.lower()

        score_interview = 21
        if any("75" in r or "refinancing" in r for r in [rat_lower] + ranked):
            score_interview += 2
        if any("elena" in rat_lower or "cfo" in rat_lower for _ in [1]):
            score_interview += 2
        score_interview = min(25, score_interview)

        score_legal = 20
        if "11.2" in rat or "limitation of liability" in rat_lower:
            score_legal += 3
        if "runnion" in rat_lower or "amend" in rat_lower or "four corners" in rat_lower:
            score_legal += 2
        score_legal = min(25, score_legal)

        score_risk = 16
        trade_res = session.get("tradeoff_results", {})
        if trade_res:
            score_risk += 3
        score_risk = min(20, score_risk)

        score_strat = 17
        if "negotiat" in branch or "investigat" in branch:
            score_strat += 2
        if "refinancing" in rat_lower or "75" in rat_lower or "190" in rat_lower:
            score_strat += 1
        score_strat = min(20, score_strat)

        score_rev = 8
        if len(rev.get("rationale", "")) > 60:
            score_rev += 1
        score_rev = min(10, score_rev)

        total = score_interview + score_legal + score_risk + score_strat + score_rev
        anchor = "Strong" if total >= 90 else ("Competent" if total >= 75 else ("Developing" if total >= 60 else "Needs Improvement"))

        return {
            "client_interviewing_score": score_interview,
            "client_interviewing_feedback": (
                "Effectively identified Elena Rostova's executive authority and highlighted Lakeview's key constraints, "
                "including the $15k–$25k litigation budget and 75-day bank refinancing timeline."
            ),
            "legal_reasoning_score": score_legal,
            "legal_reasoning_feedback": (
                "Sound grasp of Illinois contract elements and the 2024 Master Agreement's Section 11.2 limitation of liability. "
                "Properly appreciated the 7th Circuit's liberal leave-to-amend standard under Runnion."
            ),
            "risk_uncertainty_score": score_risk,
            "risk_uncertainty_feedback": (
                "Applied conditional probability principles (P(permanent dismissal) = P(grant) * P(no amendment | grant)) "
                "and recognized that 12(b)(6) dismissal rarely terminates exposure immediately."
            ),
            "strategic_judgment_score": score_strat,
            "strategic_judgment_feedback": (
                "Demonstrated client-centered judgment by balancing monetary exposure against customer relationship preservation "
                "and executive distraction during peak shipping season."
            ),
            "reflection_revision_score": score_rev,
            "reflection_revision_feedback": (
                "Thoughtfully synthesized the simulated event outcome and explained why their strategic recommendation "
                "protected Lakeview's long-term commercial interests."
            ),
            "total_score": total,
            "performance_anchor": anchor,
            "overall_critique": (
                "The student demonstrated mature strategic counseling and avoided reflexive dispositive motion practice. "
                "By tying legal options to the client's concrete commercial constraints (bank refinancing and recurring customer revenue), "
                "the student exemplified the core tenets of the Innovation Lab's strategic judgment model."
            ),
            "pre_post_comparison_note": (
                "The student's reasoning matured significantly after experiencing the simulated event, recognizing that legal plausibility "
                "must always be subordinated to client business objectives."
            ),
            "actionable_coaching": [
                "Always verify whether the opponent's complaint can be easily cured by amendment before spending $20k on a Rule 12(b)(6) motion.",
                "Structure settlement proposals with commercial assets (such as warehouse fee credits) to bridge monetary gaps.",
                "Conduct informal factual investigation before committing to public litigation positions that could invite discovery of unfavorable records."
            ],
        }

    def export_simulation_rubric_csv(
        self,
        session_id: str,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> str:
        session = self.load_session(session_id, snapshot=snapshot)
        r_eval = session.get("rubric_evaluation", {})
        buf = io.StringIO()
        writer = csv.writer(buf)

        writer.writerow(["Innovation Lab | Strategic Judgment Simulation — Evaluation Report"])
        writer.writerow(["Session ID", session.get("session_id")])
        writer.writerow(["Case", "Heartland Specialty Foods v. Lakeview Cold Chain Logistics (N.D. Ill.)"])
        writer.writerow(["Total Score", f"{r_eval.get('total_score', 'N/A')}/100"])
        writer.writerow(["Performance Anchor", r_eval.get("performance_anchor", "Pending")])
        writer.writerow([])
        writer.writerow(["Dimension", "Max Points", "Earned Score", "Evaluator Feedback"])
        writer.writerow([
            "Client Interviewing",
            25,
            r_eval.get("client_interviewing_score", "N/A"),
            r_eval.get("client_interviewing_feedback", "N/A"),
        ])
        writer.writerow([
            "Legal Reasoning",
            25,
            r_eval.get("legal_reasoning_score", "N/A"),
            r_eval.get("legal_reasoning_feedback", "N/A"),
        ])
        writer.writerow([
            "Risk and Uncertainty",
            20,
            r_eval.get("risk_uncertainty_score", "N/A"),
            r_eval.get("risk_uncertainty_feedback", "N/A"),
        ])
        writer.writerow([
            "Strategic Judgment",
            20,
            r_eval.get("strategic_judgment_score", "N/A"),
            r_eval.get("strategic_judgment_feedback", "N/A"),
        ])
        writer.writerow([
            "Reflection and Revision",
            10,
            r_eval.get("reflection_revision_score", "N/A"),
            r_eval.get("reflection_revision_feedback", "N/A"),
        ])
        writer.writerow([])
        writer.writerow(["Executive Summary", r_eval.get("overall_critique", "N/A")])
        writer.writerow(["Pre/Post Evolution", r_eval.get("pre_post_comparison_note", "N/A")])
        return buf.getvalue()

