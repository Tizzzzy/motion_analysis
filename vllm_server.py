"""Dedicated Qwen 3.8-27B vLLM HTTP inference service (built from Qwen_example.py).

Supports optional Bearer API-key authentication (`QWEN_API_KEY`) so it can be
safely exposed to Vercel through an HTTPS tunnel (`cloudflared` or `ngrok`).
"""

import argparse
import hmac
import os
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams

BASE_DIR = Path(__file__).resolve().parent
API_KEY_FILE = BASE_DIR / "data" / "qwen_api_key.txt"

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "/projects/p32143/cache/qwen38_27b")
MODEL_ID_NAME = os.environ.get("QWEN_MODEL_NAME", "Qwen/Qwen3.8-27B")

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    return cleaned.strip()


def _resolve_expected_api_key() -> str:
    env_key = (os.environ.get("QWEN_API_KEY") or "").strip()
    if env_key:
        return env_key
    if API_KEY_FILE.exists():
        try:
            return API_KEY_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return ""


def verify_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> None:
    expected = HOLDER.api_key
    if not expected:
        return
    provided = ""
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    elif x_api_key:
        provided = x_api_key.strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing QWEN_API_KEY.")


class ChatMessage(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = Field(default=1024, ge=16, le=8192)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.05, le=1.0)
    enable_thinking: bool = False
    seed: Optional[int] = None


class VLLMEngineHolder:
    def __init__(self) -> None:
        self.llm: Optional[LLM] = None
        self.lock = threading.Lock()
        self.ready = False
        self.started_at = time.time()
        self.model_path = MODEL_PATH
        self.model_name = MODEL_ID_NAME
        self.api_key = _resolve_expected_api_key()

    def initialize(
        self,
        model_path: str = MODEL_PATH,
        tensor_parallel_size: int = 2,
        dtype: str = "auto",
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.85,
        enforce_eager: bool = True,
        trust_remote_code: bool = True,
        disable_log_stats: bool = True,
        api_key: Optional[str] = None,
    ) -> None:
        if api_key is not None:
            self.api_key = api_key.strip()
        else:
            self.api_key = _resolve_expected_api_key()

        self.model_path = model_path
        print(f"Initializing vLLM Engine for {model_path} (TP={tensor_parallel_size}, max_len={max_model_len})...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
            disable_log_stats=disable_log_stats,
            generation_config="vllm",
        )
        self.ready = True
        print("vLLM Engine is ready!")

    def format_messages(self, messages: List[Dict[str, Any]], enable_thinking: bool = False) -> str:
        assert self.llm is not None
        tokenizer = self.llm.get_tokenizer()
        formatted_msgs = []
        for m in messages:
            if m["role"] == "user" and isinstance(m["content"], str):
                formatted_msgs.append(
                    {"role": "user", "content": [{"type": "text", "text": m["content"]}]}
                )
            else:
                formatted_msgs.append({"role": m["role"], "content": m["content"]})

        try:
            return tokenizer.apply_chat_template(
                formatted_msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                formatted_msgs,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": enable_thinking},
            )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.95,
        enable_thinking: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.ready or self.llm is None:
            raise RuntimeError("vLLM engine is not initialized yet.")

        prompt_str = self.format_messages(messages, enable_thinking=enable_thinking)
        sampling_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p if temperature > 0 else 1.0,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            sampling_kwargs["seed"] = seed
        sampling = SamplingParams(**sampling_kwargs)

        t0 = time.time()
        with self.lock:
            outputs = self.llm.generate([{"prompt": prompt_str}], sampling)
        elapsed = round(time.time() - t0, 2)

        raw_text = outputs[0].outputs[0].text if (outputs and outputs[0].outputs) else ""
        clean_text = strip_thinking(raw_text)
        return {
            "text": clean_text,
            "raw_text": raw_text,
            "model": self.model_name,
            "latency_sec": elapsed,
        }


HOLDER = VLLMEngineHolder()
app = FastAPI(title="Qwen 3.8-27B vLLM Service", version="1.1.0")


@app.get("/health", dependencies=[Depends(verify_api_key)])
def health() -> Dict[str, Any]:
    return {
        "ok": HOLDER.ready,
        "model": HOLDER.model_name,
        "auth_enabled": bool(HOLDER.api_key),
        "uptime_sec": round(time.time() - HOLDER.started_at, 1),
    }


@app.post("/generate", dependencies=[Depends(verify_api_key)])
def generate_endpoint(req: GenerateRequest) -> Dict[str, Any]:
    if not HOLDER.ready:
        raise HTTPException(status_code=503, detail="vLLM engine still loading.")
    try:
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        return HOLDER.generate(
            messages=msgs,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            enable_thinking=req.enable_thinking,
            seed=req.seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Qwen 3.8-27B vLLM HTTP inference worker.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--api_key", default=None, help="Optional Bearer API key (defaults to QWEN_API_KEY env or data/qwen_api_key.txt)")
    ap.add_argument("--model_path", default=MODEL_PATH)
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--enforce_eager", action="store_true", default=True)
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument("--disable_log_stats", action="store_true", default=True)
    args = ap.parse_args()

    tp = args.tensor_parallel_size
    try:
        import torch
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            if tp > available_gpus:
                print(f"Requested tensor_parallel_size={tp} exceeds available GPUs ({available_gpus}). Adjusting to {available_gpus}.")
                tp = max(1, available_gpus)
    except Exception:
        pass

    HOLDER.initialize(
        model_path=args.model_path,
        tensor_parallel_size=tp,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
        disable_log_stats=args.disable_log_stats,
        api_key=args.api_key,
    )

    endpoint_host = socket.gethostname()
    endpoint_file = BASE_DIR / "data" / "vllm_endpoint.txt"
    endpoint_file.parent.mkdir(parents=True, exist_ok=True)
    endpoint_url = f"http://{endpoint_host}:{args.port}"
    endpoint_file.write_text(endpoint_url + "\n", encoding="utf-8")
    print(f"Registered active vLLM endpoint -> {endpoint_url} ({endpoint_file})")
    print(f"API key protection: {'ENABLED' if HOLDER.api_key else 'DISABLED'}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
