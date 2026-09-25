#!/usr/bin/env bash
#
# experiments/exp5_graph/run.sh — steady-state decode speed, eager vs CUDA graph.
# See .claude/plans/exp5_plan.md. Pick a FREE GPU first (nvidia-smi), e.g.:
#
#   CUDA_VISIBLE_DEVICES=2 bash experiments/exp5_graph/run.sh                 # ouro
#   CUDA_VISIBLE_DEVICES=2 MODEL=huginn-3.5b KV_BUDGET_S=0,1 bash experiments/exp5_graph/run.sh
#
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-ouro-2.6b}"          # ouro-2.6b | huginn-3.5b
MODES="${MODES:-ar,wfd}"
ENGINES="${ENGINES:-eager,graph}"
PREFILL="${PREFILL:-512}"
DECODE="${DECODE:-512}"
KV_BUDGET_S="${KV_BUDGET_S:-0}"      # huginn: "0,1" for the sharing axis
DTV_GAMMA="${DTV_GAMMA:-8}"
SEED="${SEED:-0}"

PYTHONPATH=. python experiments/exp5_graph/run.py \
    --model "${MODEL}" \
    --modes "${MODES}" \
    --engines "${ENGINES}" \
    --prefill-len "${PREFILL}" \
    --decode-len "${DECODE}" \
    --kv-budget-s "${KV_BUDGET_S}" \
    --dtv-gamma "${DTV_GAMMA}" \
    --seed "${SEED}"
