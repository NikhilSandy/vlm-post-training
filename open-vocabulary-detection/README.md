# Qwen open-vocabulary classification and token explorer

This directory contains the Qwen3.5 Gradio vision-token explorer, the original COCO
readout experiments, and the option-attention instance classifier. Shared probing
code is included locally; new runs do not require the original NFS code directory.
Model checkpoints, COCO images/annotations, and generated results are external data.

## Layout

```text
open-vocabulary-detection/
├── qwen_token_explorer/       # Gradio app, shared model probe, visualizations
├── coco_classification/       # Original evaluator, option classifier, VRAM guard
├── scripts/run_gradio.sh      # Gradio environment and server launcher
├── tests/                    # CPU-only model, visualization, and voting checks
├── docs/                     # Detailed Gradio and evaluation instructions
├── requirements.txt          # Versions from the verified runtime
├── pyproject.toml            # Installable packages and console commands
├── evaluate_coco.py          # Compatibility CLI
├── evaluate_option_attention.py
└── run.sh                    # Compatibility Gradio launcher
```

## Setup

Use Python 3.11 or newer and install a CUDA-compatible PyTorch build for the target
machine. Then install the project, or use the existing `rtv4` environment:

```bash
cd open-vocabulary-detection
python -m pip install -e .
```

Both CLIs accept `--checkpoint`, `--images`, and `--annotations`. Their defaults
match this machine's Qwen3.5-4B checkpoint and COCO val2017 dataset. No model weights
or dataset files are included in Git.

## Gradio explorer

```bash
./run.sh --host 127.0.0.1 --port 7861
```

The launcher uses the existing `rtv4` Python when available, falling back to `python`.
Set `QWEN_EXPLORER_PYTHON` and `CUDA_VISIBLE_DEVICES` to override the runtime and GPUs.
After installing the project, `qwen-token-explorer --help` is also available.
See [Gradio documentation](docs/gradio.md) for the readouts and controls.

## Full COCO option-attention evaluation

From the repository root with the existing environment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 \
  /home/nikhileswara/miniconda3/envs/rtv4/bin/python -B -u \
  open-vocabulary-detection/evaluate_option_attention.py \
  --gpu-memory-caps-gib 3 4 1.5 4 \
  --model-memory-budgets-gib 2.25 3.25 0.75 3.25 --min-free-gib 4 \
  --output outputs/qwen_coco_option_attention_new_run
```

The default is every validation image, including single-class images. The prompt
lists the masked GT categories alphabetically as `category = A`, `category = B`, etc.
Each patch votes for the option receiving the largest mean-head attention share.
GT object masks aggregate those patch votes for instance-classification accuracy.

Use `--limit 3` for a smoke check. After installation, `coco-option-attention` is the
equivalent console command. The original cosine/word-attention experiments remain
available through `evaluate_coco.py` or `coco-readout-evaluation`.
See [evaluation documentation](docs/option_attention.md) for pseudocode, output
formats, GPU memory limits, and resume behavior.

## Existing running processes

The reorganization does not restart the existing Gradio server or the full COCO
evaluation. The already-running Gradio server still uses its original NFS working
directory. Future launches use the files in this repository.

The evaluation started before this reorganization stores an exact copy of its
original evaluator, CUDA guard, and probe in
`outputs/qwen_coco_option_attention_val2017/source_snapshot/`. The files match the
run's recorded hashes. If that worker exits before completion, use the saved
`resume-command.txt` there; the reorganized evaluator has different source hashes
and intentionally rejects resuming the old configuration. Do not run a second
worker against an output directory while its current evaluation is running.

## Validation

```bash
cd open-vocabulary-detection
/home/nikhileswara/miniconda3/envs/rtv4/bin/python -B -m unittest discover -s tests -v
```

The tests use CPU fixtures and do not load the Qwen checkpoint or allocate a new
GPU model. Results, checkpoints, local environments, and caches are excluded from
the project sources.
