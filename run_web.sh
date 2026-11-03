#!/usr/bin/env bash
# Launch the Motion Mindset Web & API server (connects to Qwen vLLM worker).
# Usage:
#   ./run_web.sh --port 8821 --vllm-url http://qgpu2008:8001
set -eo pipefail

SCRIPT_DIR="/projects/p32143/ai_law_course"
cd "${SCRIPT_DIR}"

if [ -x /home/kjj8053/anaconda3/envs/qwen/bin/python ]; then
    PYTHON_BIN="/home/kjj8053/anaconda3/envs/qwen/bin/python"
elif [ -x /home/kjj8053/anaconda3/bin/python ]; then
    PYTHON_BIN="/home/kjj8053/anaconda3/bin/python"
else
    PYTHON_BIN="python3"
fi

exec "${PYTHON_BIN}" /projects/p32143/ai_law_course/server.py "$@"

