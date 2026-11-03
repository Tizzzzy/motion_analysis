"""Client interface to the Qwen 3.8-27B vLLM engine (local cluster, HTTPS tunnel, or in-process)."""

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
ENDPOINT_FILE = BASE_DIR / "data" / "vllm_endpoint.txt"
API_KEY_FILE = BASE_DIR / "data" / "qwen_api_key.txt"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def strip_thinking(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    return cleaned.strip()


def parse_llm_json(text: str) -> Dict[str, Any]:
    """Robustly extract a JSON object from LLM text output."""
    cleaned = strip_thinking(text)
    candidates: List[str] = [cleaned]

    fence_match = _CODE_FENCE_RE.search(cleaned)
    if fence_match:
        candidates.insert(0, fence_match.group(1).strip())

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(cleaned[first_brace : last_brace + 1])

    for cand in candidates:
        if not cand:
            continue
        for attempt in (cand, _TRAILING_COMMA_RE.sub(r"\1", cand)):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

    raise ValueError(f"Could not parse JSON from model output: {cleaned[:300]!r}")


class QwenLLMClient:
    def __init__(
        self,
        vllm_url: Optional[str] = None,
        api_key: Optional[str] = None,
        local_engine: Any = None,
        timeout_sec: int = 58,
    ) -> None:
        self._configured_url = (vllm_url or os.environ.get("VLLM_URL", "auto")).strip()
        self._configured_api_key = api_key
        self.local_engine = local_engine
        self.timeout_sec = timeout_sec

    @property
    def vllm_url(self) -> str:
        env_url = (os.environ.get("VLLM_URL") or "").strip()
        if self._configured_url and self._configured_url.lower() != "auto":
            return self._configured_url.rstrip("/")
        if env_url and env_url.lower() != "auto":
            return env_url.rstrip("/")
        if ENDPOINT_FILE.exists():
            try:
                discovered = ENDPOINT_FILE.read_text(encoding="utf-8").strip()
                if discovered:
                    return discovered.rstrip("/")
            except Exception:
                pass
        return "http://localhost:8001"

    @property
    def api_key(self) -> str:
        if self._configured_api_key is not None:
            return self._configured_api_key.strip()
        env_key = (os.environ.get("QWEN_API_KEY") or "").strip()
        if env_key:
            return env_key
        if API_KEY_FILE.exists():
            try:
                return API_KEY_FILE.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        return ""

    def _build_headers(self, json_body: bool = False) -> Dict[str, str]:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; MotionMindset/2.0)",
            "ngrok-skip-browser-warning": "true",
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        key = self.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def check_health(self) -> Dict[str, Any]:
        if self.local_engine is not None:
            return {
                "connected": bool(getattr(self.local_engine, "ready", False)),
                "mode": "in_process",
                "model": getattr(self.local_engine, "model_name", "Qwen/Qwen3.8-27B"),
            }
        target_url = self.vllm_url
        is_vercel = bool(os.environ.get("VERCEL"))
        if is_vercel and target_url == "http://localhost:8001":
            return {
                "connected": False,
                "mode": "vercel_unconfigured",
                "model": "Qwen/Qwen3.8-27B",
                "error": "Set VLLM_URL in Vercel Environment Variables to your Quest HTTPS tunnel.",
            }
        try:
            req = urllib.request.Request(
                f"{target_url}/health",
                headers=self._build_headers(json_body=False),
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "connected": bool(data.get("ok", False)),
                    "mode": "vercel_tunnel" if is_vercel else "remote_vllm",
                    "model": data.get("model", "Qwen/Qwen3.8-27B"),
                    "uptime_sec": data.get("uptime_sec"),
                }
        except Exception as exc:
            return {
                "connected": False,
                "mode": "vercel_tunnel" if is_vercel else "remote_vllm",
                "model": "Qwen/Qwen3.8-27B",
                "error": str(exc),
            }

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.95,
        enable_thinking: bool = False,
    ) -> Dict[str, Any]:
        if self.local_engine is not None:
            return self.local_engine.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=enable_thinking,
            )

        target_url = self.vllm_url
        if os.environ.get("VERCEL") and target_url == "http://localhost:8001":
            raise RuntimeError(
                "VLLM_URL is not configured on Vercel yet. Add VLLM_URL and QWEN_API_KEY in Vercel Project Settings -> Environment Variables."
            )

        payload = json.dumps(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "enable_thinking": enable_thinking,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{target_url}/generate",
            data=payload,
            headers=self._build_headers(json_body=True),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                body["text"] = strip_thinking(body.get("text", ""))
                return body
        except urllib.error.HTTPError as http_err:
            err_body = http_err.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM HTTP {http_err.code}: {err_body}") from http_err
        except urllib.error.URLError as url_err:
            raise RuntimeError(
                f"Cannot reach Qwen vLLM server ({url_err.reason}). "
                "Ensure your Quest GPU job and HTTPS tunnel (`run_qwen_with_tunnel.sh`) are running."
            ) from url_err
