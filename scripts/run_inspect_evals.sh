#!/usr/bin/env bash
# Lightweight wrapper around tinker_cookbook.eval.run_inspect_evals
# to run Inspect Evals benchmarks against a Tinker model.
#
# Usage:
#   chmod +x scripts/run_inspect_evals.sh
#   MODEL_PATH="tinker://YOUR/MODEL/PATH" #   MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" #   RENDERER_NAME="llama3" #     ./scripts/run_inspect_evals.sh
#
# You can override TASKS to run different benchmarks. The defaults target
# general reasoning + instruction following + math.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-tinker://FIXME}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
RENDERER_NAME="${RENDERER_NAME:-llama3}"

# Default suite: general-purpose assistant metrics
TASKS="${TASKS:-inspect_evals/ifeval,inspect_evals/mmlu_0_shot,inspect_evals/gsm8k}"

echo "[inspect] Running Inspect Evals via tinker_cookbook.eval.run_inspect_evals"
echo "  MODEL_PATH   = ${MODEL_PATH}"
echo "  MODEL_NAME   = ${MODEL_NAME}"
echo "  RENDERER_NAME= ${RENDERER_NAME}"
echo "  TASKS        = ${TASKS}"

python -m tinker_cookbook.eval.run_inspect_evals   model_path="${MODEL_PATH}"   model_name="${MODEL_NAME}"   renderer_name="${RENDERER_NAME}"   tasks="${TASKS}"
