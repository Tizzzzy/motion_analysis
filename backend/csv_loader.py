"""Dynamic, hot-reloading CSV loader for Question Bank and Evaluation Rubric.

Designed so that replacing either CSV file on disk automatically updates
questions, phases, evaluation criteria, and rubric skills without code changes
or server restarts.
"""

import csv
import io
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_QUESTION_BANK_PATH = BASE_DIR / "Client Simulation Question Bank(Question Bank).csv"
DEFAULT_RUBRIC_PATH = BASE_DIR / "Client Simulation Question Bank(Evaluation Rubric).csv"

_ALWAYS_FIRST_RE = re.compile(
    r"\s*[-–—]\s*always\s+question\s*1(?:\s+to\s+ask)?\s*$",
    re.IGNORECASE,
)
_LEVEL_PREFIX_RE = re.compile(r"^\s*(\d+)\s*[-–—:]\s*(.+?)\s*$")


def _read_text_with_fallback(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _clean_cell(val: Optional[str]) -> str:
    if val is None:
        return ""
    lines = [line.strip() for line in val.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "item"


def _infer_phase_mode(phase_name: str) -> str:
    lower = phase_name.lower()
    if any(tok in lower for tok in ("during", "interview", "client meeting", "simulation")):
        return "client_roleplay"
    return "mentor_qa"


class QuestionItem:
    def __init__(
        self,
        id: str,
        index: int,
        phase: str,
        phase_id: str,
        phase_mode: str,
        category: str,
        raw_question: str,
        question: str,
        why_ask: str,
        what_to_evaluate: str,
        always_first: bool = False,
        extra_fields: Optional[Dict[str, str]] = None,
    ) -> None:
        self.id = id
        self.index = index
        self.phase = phase
        self.phase_id = phase_id
        self.phase_mode = phase_mode
        self.category = category
        self.raw_question = raw_question
        self.question = question
        self.why_ask = why_ask
        self.what_to_evaluate = what_to_evaluate
        self.always_first = always_first
        self.extra_fields = extra_fields or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "index": self.index,
            "phase": self.phase,
            "phase_id": self.phase_id,
            "phase_mode": self.phase_mode,
            "category": self.category,
            "raw_question": self.raw_question,
            "question": self.question,
            "why_ask": self.why_ask,
            "what_to_evaluate": self.what_to_evaluate,
            "always_first": self.always_first,
            "extra_fields": self.extra_fields,
        }


class PhaseGroup:
    def __init__(self, id: str, name: str, mode: str, questions: Optional[List[QuestionItem]] = None) -> None:
        self.id = id
        self.name = name
        self.mode = mode
        self.questions = questions or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "mode": self.mode,
            "questions": [q.to_dict() for q in self.questions],
        }


class RubricLevel:
    def __init__(self, score: int, label: str, header: str, description: str) -> None:
        self.score = score
        self.label = label
        self.header = header
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "label": self.label,
            "header": self.header,
            "description": self.description,
        }


class RubricSkill:
    def __init__(
        self,
        id: str,
        skill: str,
        levels: List[RubricLevel],
        notes: str = "",
        extra_fields: Optional[Dict[str, str]] = None,
    ) -> None:
        self.id = id
        self.skill = skill
        self.levels = levels
        self.notes = notes
        self.extra_fields = extra_fields or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "skill": self.skill,
            "levels": [lvl.to_dict() for lvl in self.levels],
            "notes": self.notes,
            "extra_fields": self.extra_fields,
        }


def load_question_bank(path: Path = DEFAULT_QUESTION_BANK_PATH) -> Tuple[List[PhaseGroup], List[QuestionItem]]:
    text = _read_text_with_fallback(path)
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return [], []

    raw_headers = [_clean_cell(h) for h in rows[0]]

    col_when: Optional[int] = None
    col_category: Optional[int] = None
    col_question: Optional[int] = None
    col_why: Optional[int] = None
    col_eval: Optional[int] = None
    extra_cols: List[Tuple[int, str]] = []

    for idx, header in enumerate(raw_headers):
        if not header:
            continue
        hl = header.lower()
        if col_when is None and any(k in hl for k in ("when", "phase", "stage")):
            col_when = idx
        elif col_category is None and any(k in hl for k in ("category", "topic", "area")):
            col_category = idx
        elif col_question is None and any(k in hl for k in ("question", "prompt", "ask student")):
            col_question = idx
        elif col_why is None and any(k in hl for k in ("why", "purpose", "goal", "rationale")):
            col_why = idx
        elif col_eval is None and any(k in hl for k in ("what to evaluate", "evaluate", "criteria", "look for")):
            col_eval = idx
        else:
            extra_cols.append((idx, header))

    if col_question is None and len(raw_headers) >= 3:
        col_question = 2
    if col_when is None and len(raw_headers) >= 1:
        col_when = 0
    if col_category is None and len(raw_headers) >= 2:
        col_category = 1

    raw_items: List[Dict[str, Any]] = []
    last_when = "General"

    for row in rows[1:]:
        if not row or all(not _clean_cell(c) for c in row):
            continue

        def get_col(c_idx: Optional[int]) -> str:
            if c_idx is None or c_idx >= len(row):
                return ""
            return _clean_cell(row[c_idx])

        raw_q = get_col(col_question)
        if not raw_q:
            when_candidate = get_col(col_when)
            if when_candidate:
                last_when = when_candidate
            continue

        when_val = get_col(col_when) or last_when
        last_when = when_val
        category_val = get_col(col_category) or "General"
        why_val = get_col(col_why)
        eval_val = get_col(col_eval)

        always_first = bool(_ALWAYS_FIRST_RE.search(raw_q))
        cleaned_q = _ALWAYS_FIRST_RE.sub("", raw_q).strip()

        extras: Dict[str, str] = {}
        for c_idx, c_name in extra_cols:
            val = get_col(c_idx)
            if val:
                extras[c_name] = val

        raw_items.append(
            {
                "phase": when_val,
                "category": category_val,
                "raw_question": raw_q,
                "question": cleaned_q,
                "why_ask": why_val,
                "what_to_evaluate": eval_val,
                "always_first": always_first,
                "extra_fields": extras,
            }
        )

    phases_by_id: Dict[str, PhaseGroup] = {}
    phase_order: List[str] = []

    for entry in raw_items:
        p_name = entry["phase"]
        p_id = _slugify(p_name)
        if p_id not in phases_by_id:
            phases_by_id[p_id] = PhaseGroup(
                id=p_id,
                name=p_name,
                mode=_infer_phase_mode(p_name),
                questions=[],
            )
            phase_order.append(p_id)

    bucketed: Dict[str, List[Dict[str, Any]]] = {pid: [] for pid in phase_order}
    for entry in raw_items:
        bucketed[_slugify(entry["phase"])].append(entry)

    all_questions: List[QuestionItem] = []
    global_idx = 0
    seen_ids: Dict[str, int] = {}

    for pid in phase_order:
        entries = bucketed[pid]
        entries.sort(key=lambda e: (0 if e["always_first"] else 1))
        phase_group = phases_by_id[pid]
        for entry in entries:
            base_id = f"{pid}__{_slugify(entry['category'])}"
            count = seen_ids.get(base_id, 0) + 1
            seen_ids[base_id] = count
            q_id = base_id if count == 1 else f"{base_id}_{count}"

            item = QuestionItem(
                id=q_id,
                index=global_idx,
                phase=phase_group.name,
                phase_id=pid,
                phase_mode=phase_group.mode,
                category=entry["category"],
                raw_question=entry["raw_question"],
                question=entry["question"],
                why_ask=entry["why_ask"],
                what_to_evaluate=entry["what_to_evaluate"],
                always_first=entry["always_first"],
                extra_fields=entry["extra_fields"],
            )
            phase_group.questions.append(item)
            all_questions.append(item)
            global_idx += 1

    return [phases_by_id[pid] for pid in phase_order], all_questions


def load_evaluation_rubric(path: Path = DEFAULT_RUBRIC_PATH) -> Tuple[List[RubricSkill], List[Dict[str, Any]]]:
    text = _read_text_with_fallback(path)
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return [], []

    raw_headers = [_clean_cell(h) for h in rows[0]]

    col_skill: Optional[int] = None
    col_notes: Optional[int] = None
    level_cols: List[Tuple[int, int, str, str]] = []
    extra_cols: List[Tuple[int, str]] = []
    ignored_score_cols = {"initial score", "revised score", "score", "final score"}

    for idx, header in enumerate(raw_headers):
        if not header:
            continue
        hl = header.lower()
        m_lvl = _LEVEL_PREFIX_RE.match(header)
        if col_skill is None and any(k in hl for k in ("skill", "dimension", "criterion", "competency")):
            col_skill = idx
        elif m_lvl:
            score_num = int(m_lvl.group(1))
            label = m_lvl.group(2).strip()
            level_cols.append((idx, score_num, label, header))
        elif hl in ignored_score_cols:
            continue
        elif col_notes is None and any(k in hl for k in ("note", "guidance", "framework", "reference", "law")):
            col_notes = idx
        else:
            extra_cols.append((idx, header))

    if col_skill is None and raw_headers:
        col_skill = 0

    level_cols.sort(key=lambda item: item[1])

    skills: List[RubricSkill] = []
    seen_ids: Dict[str, int] = {}

    for row in rows[1:]:
        if not row or all(not _clean_cell(c) for c in row):
            continue

        def get_col(c_idx: Optional[int]) -> str:
            if c_idx is None or c_idx >= len(row):
                return ""
            return _clean_cell(row[c_idx])

        skill_name = get_col(col_skill)
        if not skill_name:
            continue

        base_id = _slugify(skill_name)
        cnt = seen_ids.get(base_id, 0) + 1
        seen_ids[base_id] = cnt
        skill_id = base_id if cnt == 1 else f"{base_id}_{cnt}"

        levels: List[RubricLevel] = []
        for c_idx, score_num, label, header in level_cols:
            desc = get_col(c_idx)
            levels.append(
                RubricLevel(
                    score=score_num,
                    label=label,
                    header=header,
                    description=desc,
                )
            )

        notes_val = get_col(col_notes)
        extras: Dict[str, str] = {}
        for c_idx, c_name in extra_cols:
            val = get_col(c_idx)
            if val:
                extras[c_name] = val

        skills.append(
            RubricSkill(
                id=skill_id,
                skill=skill_name,
                levels=levels,
                notes=notes_val,
                extra_fields=extras,
            )
        )

    scale_meta = [
        {"score": score_num, "label": label, "header": header}
        for _, score_num, label, header in level_cols
    ]
    return skills, scale_meta


class CurriculumStore:
    """Hot-reloading store that watches both CSV files by mtime."""

    def __init__(
        self,
        question_bank_path: Optional[Path] = None,
        rubric_path: Optional[Path] = None,
    ) -> None:
        self.question_bank_path = Path(
            question_bank_path
            or os.environ.get("QUESTION_BANK_CSV", str(DEFAULT_QUESTION_BANK_PATH))
        )
        self.rubric_path = Path(
            rubric_path
            or os.environ.get("EVALUATION_RUBRIC_CSV", str(DEFAULT_RUBRIC_PATH))
        )
        self._qb_mtime = -1.0
        self._rubric_mtime = -1.0
        self.phases: List[PhaseGroup] = []
        self.questions: List[QuestionItem] = []
        self.questions_by_id: Dict[str, QuestionItem] = {}
        self.skills: List[RubricSkill] = []
        self.skills_by_id: Dict[str, RubricSkill] = {}
        self.scale: List[Dict[str, Any]] = []
        self.reload_if_needed(force=True)

    def reload_if_needed(self, force: bool = False) -> bool:
        reloaded = False
        qb_mtime = self.question_bank_path.stat().st_mtime if self.question_bank_path.exists() else 0.0
        rb_mtime = self.rubric_path.stat().st_mtime if self.rubric_path.exists() else 0.0

        if force or qb_mtime != self._qb_mtime:
            self.phases, self.questions = load_question_bank(self.question_bank_path)
            self.questions_by_id = {q.id: q for q in self.questions}
            self._qb_mtime = qb_mtime
            reloaded = True

        if force or rb_mtime != self._rubric_mtime:
            self.skills, self.scale = load_evaluation_rubric(self.rubric_path)
            self.skills_by_id = {s.id: s for s in self.skills}
            self._rubric_mtime = rb_mtime
            reloaded = True

        return reloaded

    @property
    def max_score(self) -> int:
        if not self.scale:
            return 3
        return max(item["score"] for item in self.scale)

    @property
    def passing_score(self) -> int:
        scores = sorted(item["score"] for item in self.scale)
        if 2 in scores:
            return 2
        if len(scores) >= 2:
            return scores[len(scores) // 2]
        return 1

    def format_rubric_for_prompt(self) -> str:
        self.reload_if_needed()
        blocks: List[str] = []
        for s in self.skills:
            lines = [f"### Skill: {s.skill} (id: {s.id})"]
            for lvl in s.levels:
                lines.append(f"  - {lvl.score} ({lvl.label}): {lvl.description or 'N/A'}")
            if s.notes:
                indented = "\n    ".join(s.notes.splitlines())
                lines.append(f"  - Reference / Notes:\n    {indented}")
            for k, v in s.extra_fields.items():
                lines.append(f"  - {k}: {v}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def to_api_dict(self) -> Dict[str, Any]:
        self.reload_if_needed()
        return {
            "question_bank_file": self.question_bank_path.name,
            "rubric_file": self.rubric_path.name,
            "question_bank_mtime": self._qb_mtime,
            "rubric_mtime": self._rubric_mtime,
            "total_questions": len(self.questions),
            "passing_score": self.passing_score,
            "max_score": self.max_score,
            "scale": self.scale,
            "phases": [p.to_dict() for p in self.phases],
            "skills": [s.to_dict() for s in self.skills],
        }
