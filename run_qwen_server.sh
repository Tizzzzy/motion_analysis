#!/usr/bin/env bash
# Launch the Qwen 3.8-27B vLLM HTTP inference worker on a GPU node (e.g. qgpu2008).
# Usage (on GPU node):
#   ./run_qwen_server.sh --port 8001
set -eo pipefail

SCRIPT_DIR="/projects/p32143/ai_law_course"
cd "${SCRIPT_DIR}"

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

if [ -f /home/kjj8053/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/kjj8053/anaconda3/etc/profile.d/conda.sh
    conda activate qwen || true
fi

module load vllm/0.19.0

echo "Starting Qwen 3.8-27B vLLM server on $(hostname)..."
python /projects/p32143/ai_law_course/vllm_server.py "$@"

