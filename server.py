"""Main Web & API Server for Motion Mindset Conversational Guidance.

Runs identically in two modes:
  1) Local / Quest server (`python3 server.py --port 8821`)
  2) Vercel Python Serverless Function (`api/index.py` importing `MotionGuidanceHandler`)
"""

import argparse
import json
import mimetypes
import os
import posixpath
import re
import socketserver
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse

from backend.csv_loader import CurriculumStore
from backend.llm_client import QwenLLMClient
from backend.sample_cases import list_sample_cases
from backend.session_manager import SessionManager

BASE_DIR = Path(__file__).resolve().parent

_SESSION_ROUTE_RE = re.compile(
    r"^/api/session/([a-zA-Z0-9_-]+)(?:/(case|message|skip|hint|export\.csv))?$"
)
_SIM_ROUTE_RE = re.compile(
    r"^/api/simulation/([a-zA-Z0-9_-]+)(?:/(stage|priorities|tradeoffs|opponent|strategy|event|evaluate|prep|colloquy|export\.(?:csv|json)))?$"
)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class AppContext:
    def __init__(self, vllm_url: str = "auto", local_engine: Any = None) -> None:
        self.curriculum = CurriculumStore()
        self.llm = QwenLLMClient(vllm_url=vllm_url, local_engine=local_engine)
        self.sessions = SessionManager(curriculum=self.curriculum, llm_client=self.llm)


APP_CTX: Optional[AppContext] = None


def get_app_ctx() -> AppContext:
    global APP_CTX
    if APP_CTX is None:
        APP_CTX = AppContext(vllm_url=os.environ.get("VLLM_URL", "auto"))
    return APP_CTX


class MotionGuidanceHandler(BaseHTTPRequestHandler):
    server_version = "MotionMindsetHTTP/2.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _resolve_route(self) -> str:
        parsed = urlparse(self.path)
        route = parsed.path or "/"
        if route in ("/api/index", "/api/index.py"):
            qs = parse_qs(parsed.query)
            sub = (qs.get("__api_path") or [""])[0].lstrip("/")
            route = f"/api/{sub}" if sub else "/api"
        return route

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _send_csv(self, filename: str, csv_text: str) -> None:
        raw = csv_text.encode("utf-8-sig")
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        if not raw.strip():
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object in request body.")
        return data

    def do_GET(self) -> None:
        ctx = get_app_ctx()
        route = self._resolve_route()

        try:
            if route == "/api/config":
                self._send_json(
                    200,
                    {
                        "curriculum": ctx.curriculum.to_api_dict(),
                        "sample_cases": list_sample_cases(),
                        "llm": ctx.llm.check_health(),
                    },
                )
                return

            if route == "/api/health":
                self._send_json(200, {"ok": True, "llm": ctx.llm.check_health()})
                return

            m_sess = _SESSION_ROUTE_RE.match(route)
            if m_sess:
                session_id, subaction = m_sess.group(1), m_sess.group(2)
                if subaction is None:
                    data = ctx.sessions.serialize_session(ctx.sessions.load_session(session_id))
                    self._send_json(200, data)
                    return
                if subaction == "export.csv":
                    csv_str = ctx.sessions.export_rubric_csv(session_id)
                    self._send_csv(f"evaluation_rubric_{session_id}.csv", csv_str)
                    return

            m_sim = _SIM_ROUTE_RE.match(route)
            if m_sim:
                session_id, subaction = m_sim.group(1), m_sim.group(2)
                if subaction is None:
                    data = ctx.sessions.serialize_session(ctx.sessions.load_session(session_id))
                    self._send_json(200, data)
                    return
                if subaction == "export.csv":
                    csv_str = ctx.sessions.export_simulation_rubric_csv(session_id)
                    self._send_csv(f"strategic_judgment_report_{session_id}.csv", csv_str)
                    return

            if route.startswith("/api/"):
                self._send_json(404, {"error": f"Unknown API endpoint: {route}"})
                return

            self._serve_static(route)
        except KeyError as exc:
            self._send_json(404, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"error": str(exc)})

    def do_POST(self) -> None:
        ctx = get_app_ctx()
        route = self._resolve_route()

        try:
            body = self._read_json_body()
            snapshot = body.get("session_snapshot")

            if route == "/api/reload":
                ctx.curriculum.reload_if_needed(force=True)
                self._send_json(
                    200,
                    {
                        "reloaded": True,
                        "curriculum": ctx.curriculum.to_api_dict(),
                    },
                )
                return

            if route == "/api/session/start":
                state = ctx.sessions.create_session(
                    case_id=body.get("case_id"),
                    custom_case_text=body.get("custom_case_text", ""),
                    custom_client_notes=body.get("custom_client_notes", ""),
                )
                self._send_json(200, state)
                return

            if route == "/api/session/hydrate":
                if not isinstance(snapshot, dict) or not snapshot.get("session_id"):
                    raise ValueError("Missing valid session_snapshot.")
                loaded = ctx.sessions.load_session(snapshot["session_id"], snapshot=snapshot)
                self._send_json(200, ctx.sessions.serialize_session(loaded))
                return

            if route == "/api/motion/review":
                review = ctx.sessions.review_motion_draft(
                    draft_type=body.get("draft_type", "Rule 12(b)(6) Motion to Dismiss"),
                    draft_text=body.get("draft_text", ""),
                    session_id=body.get("session_id"),
                    case_id=body.get("case_id"),
                    custom_case_text=body.get("custom_case_text", ""),
                    snapshot=snapshot,
                )
                self._send_json(200, review)
                return

            m_sess = _SESSION_ROUTE_RE.match(route)
            if m_sess:
                session_id, subaction = m_sess.group(1), m_sess.group(2)
                if subaction == "case":
                    state = ctx.sessions.update_case_context(
                        session_id=session_id,
                        case_id=body.get("case_id"),
                        custom_case_text=body.get("custom_case_text", ""),
                        custom_client_notes=body.get("custom_client_notes", ""),
                        snapshot=snapshot,
                    )
                    self._send_json(200, state)
                    return
                if subaction == "message":
                    state = ctx.sessions.handle_user_message(
                        session_id=session_id,
                        text=body.get("message", ""),
                        snapshot=snapshot,
                    )
                    self._send_json(200, state)
                    return
                if subaction == "skip":
                    state = ctx.sessions.skip_or_finish_phase(
                        session_id=session_id,
                        action=body.get("action", "skip_question"),
                        snapshot=snapshot,
                    )
                    self._send_json(200, state)
                    return
                if subaction == "hint":
                    hint = ctx.sessions.generate_hint(
                        session_id=session_id,
                        snapshot=snapshot,
                    )
                    self._send_json(200, hint)
                    return
                if subaction == "export.csv":
                    csv_str = ctx.sessions.export_rubric_csv(
                        session_id=session_id,
                        snapshot=snapshot,
                    )
                    self._send_csv(f"evaluation_rubric_{session_id}.csv", csv_str)
                    return

            m_sim = _SIM_ROUTE_RE.match(route)
            if m_sim:
                session_id, subaction = m_sim.group(1), m_sim.group(2)
                if subaction == "stage":
                    state = ctx.sessions.set_simulation_stage(
                        session_id=session_id,
                        stage=body.get("stage", 1),
                        snapshot=snapshot,
                    )
                    self._send_json(200, state)
                    return
                if subaction == "priorities":
                    state = ctx.sessions.set_ranked_priorities(
                        session_id=session_id,
                        priorities=body.get("priorities", []),
                        snapshot=snapshot,
                    )
                    self._send_json(200, state)
                    return
                if subaction == "prep":
                    state = ctx.sessions.submit_prep_answer(
                        session_id=session_id,
                        question_key=body.get("question_key", "q1_authority"),
                        student_answer=body.get("answer", ""),
                        snapshot=snapshot,
                    )
                    self._send_json(200, state)
                    return
                if subaction == "colloquy":
                    action = body.get("action", "message")
                    if action == "select":
                        state = ctx.sessions.set_stage2_colloquy_question(
                            session_id=session_id,
                            question_key=body.get("question_key", "q4_elements"),
                            snapshot=snapshot,
                        )
                    else:
                        state = ctx.sessions.submit_stage2_colloquy_message(
                            session_id=session_id,
                            message=body.get("message", ""),
                            question_key=body.get("question_key"),
                            snapshot=snapshot,
                        )
                    self._send_json(200, state)
                    return
                if subaction == "tradeoffs":
                    state = ctx.sessions.update_tradeoffs(
                        session_id=session_id,
                        tradeoff_inputs=body.get("tradeoff_inputs", {}),
                        snapshot=snapshot,
                    )
                    self._send_json(200, state)
                    return
                if subaction == "opponent":
                    state = ctx.sessions.submit_opponent_anticipation(
                        session_id=session_id,
                        anticipation_text=body.get("anticipation_text", ""),
                        probability_estimate=body.get("probability_estimate"),
                        snapshot=snapshot,
                    )
                    self._send_json(200, state)
                    return
                if subaction == "strategy":
                    state = ctx.sessions.submit_initial_strategy(
                        session_id=session_id,
                        branch=body.get("branch", "negotiate"),
                        terms=body.get("terms", ""),
                        rationale=body.get("rationale", ""),
                        snapshot=snapshot,
                    )
                    self._send_json(200, state)
                    return
                if subaction == "event":
                    state = ctx.sessions.trigger_sim_event(
                        session_id=session_id,
                        snapshot=snapshot,
                    )
                    self._send_json(200, state)
                    return
                if subaction == "evaluate":
                    state = ctx.sessions.submit_revision_and_evaluate(
                        session_id=session_id,
                        revised_action=body.get("revised_action", "negotiate"),
                        pivoted=body.get("pivoted", False),
                        revised_rationale=body.get("revised_rationale", ""),
                        snapshot=snapshot,
                    )
                    self._send_json(200, state)
                    return
                if subaction == "export.csv":
                    csv_str = ctx.sessions.export_simulation_rubric_csv(
                        session_id=session_id,
                        snapshot=snapshot,
                    )
                    self._send_csv(f"strategic_judgment_report_{session_id}.csv", csv_str)
                    return

            self._send_json(404, {"error": f"Unknown POST endpoint: {route}"})
        except KeyError as exc:
            self._send_json(404, {"error": str(exc)})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"error": str(exc)})

    def _serve_static(self, raw_path: str) -> None:
        clean_rel = posixpath.normpath(unquote(raw_path)).lstrip("/")
        if clean_rel in ("", "."):
            clean_rel = "index.html"

        blocked_prefixes = (".", "backend", "data", "api", "bin")
        if any(clean_rel == p or clean_rel.startswith(p + "/") for p in blocked_prefixes) or clean_rel.startswith("."):
            self._send_json(403, {"error": "Forbidden"})
            return

        target = (BASE_DIR / clean_rel).resolve()
        if not str(target).startswith(str(BASE_DIR.resolve())) or not target.is_file():
            self._send_json(404, {"error": f"File not found: {clean_rel}"})
            return

        ctype, _ = mimetypes.guess_type(str(target))
        if not ctype:
            ctype = "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype = f"{ctype}; charset=utf-8"

        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    global APP_CTX
    ap = argparse.ArgumentParser(description="Motion Mindset Conversational Guidance Web Server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8821)
    ap.add_argument(
        "--vllm-url",
        default=os.environ.get("VLLM_URL", "auto"),
        help="URL of running vllm_server.py, or 'auto' to read data/vllm_endpoint.txt",
    )
    ap.add_argument(
        "--local-vllm",
        action="store_true",
        help="Initialize Qwen 3.8-27B vLLM directly inside this process (run on GPU node).",
    )
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--max_model_len", type=int, default=32768)
    args = ap.parse_args()

    local_engine = None
    if args.local_vllm:
        from vllm_server import VLLMEngineHolder

        local_engine = VLLMEngineHolder()
        local_engine.initialize(
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )

    APP_CTX = AppContext(vllm_url=args.vllm_url, local_engine=local_engine)
    srv = ThreadingHTTPServer((args.host, args.port), MotionGuidanceHandler)
    print(f"Motion Mindset server listening on http://{args.host}:{args.port}")
    print(f"  Question Bank CSV : {APP_CTX.curriculum.question_bank_path}")
    print(f"  Evaluation Rubric : {APP_CTX.curriculum.rubric_path}")
    print(f"  Qwen vLLM Backend : {'in-process' if local_engine else args.vllm_url}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        srv.server_close()


if __name__ == "__main__":
    main()
