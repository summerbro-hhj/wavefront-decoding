#!/usr/bin/env bash
# exp4 sweep launcher. Pick a FREE gpu first (nvidia-smi):
#   CUDA_VISIBLE_DEVICES=2 bash experiments/exp4_emu/run.sh
# Every knob is an env var; extra args pass through to run.py.
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL=${MODEL:-ouro-2.6b}
SCHEDULERS=${SCHEDULERS:-ar,wfd,dtv}
ALPHAS=${ALPHAS:-0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.92,0.94,0.96,0.98}
BATCH_SIZES=${BATCH_SIZES:-1}
KV_BUDGET_S=${KV_BUDGET_S:-0}
T_TOTAL=${T_TOTAL:-4}
WFD_T_DRAFTS=${WFD_T_DRAFTS:-1}
DTV_T_DRAFT=${DTV_T_DRAFT:-1}
DTV_DRAFT_LENGTHS=${DTV_DRAFT_LENGTHS:-8}
PREFILL_LEN=${PREFILL_LEN:-1024}
DECODE_LEN=${DECODE_LEN:-512}
SEED=${SEED:-0}
# 1 = prefill ONCE per (s, prefill, B) cell and share it across ar/wfd/dtv.
# Each run still reports that one measured prefill as its own prefill_s, so
# walltime_s / speedup_vs_ar keep their meaning — only the decode loop is
# re-executed. Big win at the long PREFILL_LEN entries above, where an
# un-shared prefill dominates every run.
SKIP_PREFILL=${SKIP_PREFILL:-1}

EXTRA=()
if [ "$SKIP_PREFILL" != "0" ]; then EXTRA+=(--skip-prefill); fi

python experiments/exp4_emu/run.py \
  --model "$MODEL" \
  --schedulers "$SCHEDULERS" \
  --alphas "$ALPHAS" \
  --batch-sizes "$BATCH_SIZES" \
  --kv-budget-s "$KV_BUDGET_S" \
  --T-total "$T_TOTAL" \
  --wfd-T-drafts "$WFD_T_DRAFTS" \
  --dtv-T-draft "$DTV_T_DRAFT" \
  --dtv-draft-lengths "$DTV_DRAFT_LENGTHS" \
  --prefill-len "$PREFILL_LEN" \
  --decode-len "$DECODE_LEN" \
  --seed "$SEED" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  "$@"
