#!/usr/bin/env bash
# Launch Qwen 3.8-27B (vLLM) + secure HTTPS tunnel on Northwestern Quest for Vercel.
#
# Usage on an allocated Quest GPU node (recommended):
#   cd /projects/p32143/ai_law_course
#   ./run_qwen_with_tunnel.sh
#
# Or from a Quest login node into an active SLURM GPU job:
#   srun --jobid <JOBID> --overlap /projects/p32143/ai_law_course/run_qwen_with_tunnel.sh
#
# Optional tunnel relay from login node to an already-running GPU worker:
#   ./run_qwen_with_tunnel.sh --target-url http://qgpu2008:8001
#
# Optional static-domain env vars (so you never have to update VLLM_URL in Vercel):
#   CLOUDFLARE_TUNNEL_TOKEN=<token> ./run_qwen_with_tunnel.sh
#   NGROK_DOMAIN=<your-name.ngrok-free.app> ./run_qwen_with_tunnel.sh
set -eo pipefail

SCRIPT_DIR="/projects/p32143/ai_law_course"
DATA_DIR="${SCRIPT_DIR}/data"
BIN_DIR="${SCRIPT_DIR}/bin"
KEY_FILE="${DATA_DIR}/qwen_api_key.txt"
ENDPOINT_FILE="${DATA_DIR}/vllm_endpoint.txt"
PUBLIC_URL_FILE="${DATA_DIR}/public_tunnel_url.txt"
VLLM_LOG="${DATA_DIR}/vllm_server.log"
TUNNEL_LOG="${DATA_DIR}/tunnel.log"

mkdir -p "${DATA_DIR}" "${BIN_DIR}"

PORT=8001
TARGET_URL=""
USE_NGROK=0
VLLM_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --target-url)
      TARGET_URL="$2"
      shift 2
      ;;
    --ngrok)
      USE_NGROK=1
      shift
      ;;
    *)
      VLLM_EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# 1. Ensure persistent QWEN_API_KEY exists so Vercel only needs it configured once.
if [[ -z "${QWEN_API_KEY:-}" ]]; then
  if [[ -s "${KEY_FILE}" ]]; then
    QWEN_API_KEY="$(tr -d '[:space:]' < "${KEY_FILE}")"
  else
    QWEN_API_KEY="qwen_sk_$(openssl rand -hex 24)"
    printf "%s\n" "${QWEN_API_KEY}" > "${KEY_FILE}"
    chmod 600 "${KEY_FILE}"
  fi
fi
export QWEN_API_KEY

VLLM_PID=""
TUNNEL_PID=""

cleanup() {
  echo ""
  echo "Shutting down tunnel and vLLM worker..."
  if [[ -n "${TUNNEL_PID}" ]] && kill -0 "${TUNNEL_PID}" 2>/dev/null; then
    kill "${TUNNEL_PID}" 2>/dev/null || true
  fi
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

check_vllm_ready() {
  local url="$1"
  curl -fsS -H "Authorization: Bearer ${QWEN_API_KEY}" "${url}/health" 2>/dev/null | grep -q '"ok":true'
}

# 2. Start vLLM locally if on a GPU node (unless --target-url was explicitly given).
if [[ -z "${TARGET_URL}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    TARGET_URL="http://127.0.0.1:${PORT}"
    if check_vllm_ready "${TARGET_URL}"; then
      echo "Reusing already-running Qwen vLLM worker at ${TARGET_URL}."
    else
      echo "Launching Qwen 3.8-27B vLLM server on $(hostname):${PORT} (logs -> ${VLLM_LOG})..."
      "${SCRIPT_DIR}/run_qwen_server.sh" --port "${PORT}" "${VLLM_EXTRA_ARGS[@]}" >"${VLLM_LOG}" 2>&1 &
      VLLM_PID=$!

      echo "Waiting ~2 minutes for Qwen 3.8-27B weights to load into GPU memory..."
      ATTEMPTS=0
      until check_vllm_ready "${TARGET_URL}"; do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
          echo "ERROR: vLLM server exited unexpectedly. Last 40 log lines:" >&2
          tail -n 40 "${VLLM_LOG}" >&2 || true
          exit 1
        fi
        ATTEMPTS=$((ATTEMPTS + 1))
        if (( ATTEMPTS % 6 == 0 )); then
          echo "  ...still loading Qwen 3.8-27B (${ATTEMPTS}0s elapsed)"
        fi
        sleep 5
      done
      echo "Qwen 3.8-27B vLLM engine is READY at ${TARGET_URL}!"
    fi
  elif [[ -s "${ENDPOINT_FILE}" ]]; then
    CANDIDATE_URL="$(tr -d '[:space:]' < "${ENDPOINT_FILE}")"
    if check_vllm_ready "${CANDIDATE_URL}"; then
      TARGET_URL="${CANDIDATE_URL}"
      echo "No local GPU on $(hostname); relaying HTTPS tunnel to active GPU worker at ${TARGET_URL}."
    else
      echo "ERROR: $(hostname) has no NVIDIA GPU and ${CANDIDATE_URL} is not reachable." >&2
      echo "Run this script inside an active Quest GPU allocation instead:" >&2
      echo "  srun --jobid <JOBID> --overlap /projects/p32143/ai_law_course/run_qwen_with_tunnel.sh" >&2
      exit 1
    fi
  else
    echo "ERROR: $(hostname) has no NVIDIA GPU. Run this script on a Quest GPU node:" >&2
    echo "  srun --jobid <JOBID> --overlap /projects/p32143/ai_law_course/run_qwen_with_tunnel.sh" >&2
    exit 1
  fi
else
  echo "Waiting for target vLLM worker at ${TARGET_URL}..."
  until check_vllm_ready "${TARGET_URL}"; do
    sleep 3
  done
  echo "Connected to vLLM worker at ${TARGET_URL}."
fi

# 3. Start HTTPS Tunnel (ngrok static domain, Cloudflare Named Tunnel, or Cloudflare Quick Tunnel).
if [[ "${USE_NGROK}" -eq 1 || -n "${NGROK_DOMAIN:-}" ]]; then
  NGROK_BIN="$(command -v ngrok || true)"
  if [[ -z "${NGROK_BIN}" && -x "${BIN_DIR}/ngrok" ]]; then
    NGROK_BIN="${BIN_DIR}/ngrok"
  fi
  if [[ -z "${NGROK_BIN}" ]]; then
    echo "ERROR: ngrok binary not found in PATH or ${BIN_DIR}/ngrok." >&2
    exit 1
  fi
  echo "Starting ngrok HTTPS tunnel -> ${TARGET_URL}..."
  if [[ -n "${NGROK_DOMAIN:-}" ]]; then
    PUBLIC_URL="https://${NGROK_DOMAIN#https://}"
    printf "%s\n" "${PUBLIC_URL}" > "${PUBLIC_URL_FILE}"
    echo "========================================================================"
    echo "  VERCEL ENVIRONMENT VARIABLES (Settings -> Environment Variables)"
    echo "  VLLM_URL=${PUBLIC_URL}"
    echo "  QWEN_API_KEY=${QWEN_API_KEY}"
    echo "========================================================================"
    "${NGROK_BIN}" http --url="${NGROK_DOMAIN}" "${TARGET_URL}" &
    TUNNEL_PID=$!
    wait "${TUNNEL_PID}"
    exit 0
  else
    "${NGROK_BIN}" http "${TARGET_URL}" --log=stdout >"${TUNNEL_LOG}" 2>&1 &
    TUNNEL_PID=$!
  fi
else
  CLOUDFLARED_BIN="$(command -v cloudflared || true)"
  if [[ -z "${CLOUDFLARED_BIN}" ]]; then
    CLOUDFLARED_BIN="${BIN_DIR}/cloudflared"
    if [[ ! -x "${CLOUDFLARED_BIN}" ]]; then
      ARCH="$(uname -m)"
      CF_ASSET="cloudflared-linux-amd64"
      if [[ "${ARCH}" == "aarch64" || "${ARCH}" == "arm64" ]]; then
        CF_ASSET="cloudflared-linux-arm64"
      fi
      echo "Downloading standalone ${CF_ASSET} into ${CLOUDFLARED_BIN}..."
      curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/${CF_ASSET}" -o "${CLOUDFLARED_BIN}"
      chmod +x "${CLOUDFLARED_BIN}"
    fi
  fi

  if [[ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]]; then
    echo "Starting Cloudflare Named Tunnel (HTTP/2)..."
    echo "========================================================================"
    echo "  VERCEL ENVIRONMENT VARIABLES (Settings -> Environment Variables)"
    echo "  VLLM_URL=<your configured Cloudflare public hostname>"
    echo "  QWEN_API_KEY=${QWEN_API_KEY}"
    echo "========================================================================"
    "${CLOUDFLARED_BIN}" tunnel --protocol http2 --no-autoupdate run --token "${CLOUDFLARE_TUNNEL_TOKEN}" &
    TUNNEL_PID=$!
    wait "${TUNNEL_PID}"
    exit 0
  fi

  echo "Starting Cloudflare Quick HTTPS Tunnel (HTTP/2) -> ${TARGET_URL}..."
  : > "${TUNNEL_LOG}"
  "${CLOUDFLARED_BIN}" tunnel --protocol http2 --no-autoupdate --url "${TARGET_URL}" >"${TUNNEL_LOG}" 2>&1 &
  TUNNEL_PID=$!

  PUBLIC_URL=""
  for _ in {1..30}; do
    if ! kill -0 "${TUNNEL_PID}" 2>/dev/null; then
      echo "ERROR: cloudflared exited unexpectedly. Tunnel log:" >&2
      cat "${TUNNEL_LOG}" >&2 || true
      exit 1
    fi
    PUBLIC_URL="$(grep -Eo 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' "${TUNNEL_LOG}" | head -n 1 || true)"
    if [[ -n "${PUBLIC_URL}" ]]; then
      break
    fi
    sleep 1
  done

  if [[ -z "${PUBLIC_URL}" ]]; then
    echo "WARNING: Could not auto-detect trycloudflare.com URL yet. Check ${TUNNEL_LOG}:"
    cat "${TUNNEL_LOG}" || true
  else
    printf "%s\n" "${PUBLIC_URL}" > "${PUBLIC_URL_FILE}"
    echo ""
    echo "========================================================================"
    echo "  Quest Qwen 3.8-27B HTTPS Tunnel is LIVE!"
    echo ""
    echo "  Add these two variables in Vercel -> Project Settings -> Environment Variables:"
    echo "    VLLM_URL      = ${PUBLIC_URL}"
    echo "    QWEN_API_KEY  = ${QWEN_API_KEY}"
    echo ""
    echo "  (Note: QWEN_API_KEY is saved in data/qwen_api_key.txt and stays the same across jobs.)"
    echo "  Keep this terminal/job running while using the Vercel website."
    echo "========================================================================"
    echo ""
  fi

  wait "${TUNNEL_PID}"
fi

