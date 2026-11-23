#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
runtime_dir="/dev/shm/qwen-token-explorer-$(id -u)"
mkdir -m 700 -p "$runtime_dir"
if [[ ! -O "$runtime_dir" ]]; then
    printf 'Runtime directory must be owned by the current user: %s\n' "$runtime_dir" >&2
    exit 1
fi
export TMPDIR="$runtime_dir"
export GRADIO_TEMP_DIR="$runtime_dir/gradio"
export MPLCONFIGDIR="$runtime_dir/matplotlib"
export TRITON_CACHE_DIR="$runtime_dir/triton"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" TOKENIZERS_PARALLELISM=false
runtime_python="${QWEN_EXPLORER_PYTHON:-$HOME/miniconda3/envs/rtv4/bin/python}"
if [[ ! -x "$runtime_python" ]]; then
    runtime_python="${QWEN_EXPLORER_PYTHON:-python}"
fi
exec "$runtime_python" -B -u -m qwen_token_explorer.app "$@"
