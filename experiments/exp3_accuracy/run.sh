#!/usr/bin/env bash
#
# experiments/exp3_accuracy/run.sh
#
# Launcher for experiment 3 (lossy wavefront: accuracy x latency on math).
# Step 1 = test environment: gsm8k / MATH-500 generative accuracy + speedup,
# using the existing AR / wavefront-SSD decode (no lossy levers yet).
#
# Sweep by overriding env vars, e.g.:
#     MODEL=ouro-2.6b DATASET=math500 T_TOTAL=4 T_DRAFT=2 LIMIT=50 bash experiments/exp3_accuracy/run.sh
# Temperature sampling (AR + SSD both sample at T; vary SEED for repeated draws):
#     SAMPLING=temperature TEMPERATURE=0.7 SEED=0 bash experiments/exp3_accuracy/run.sh
#
# No auto best-search in code — run small sets and compare JSONs yourself.
#

set -euo pipefail
cd "$(dirname "$0")/../.."

# Allocator: the WFD varlen path packs (wave_size x prefix) K/V transients every
# advance whose sizes GROW each tick — with T_DRAFT=1/2 (wave 33/17 tokens at
# T_TOTAL=32) the default allocator can't reuse the ever-larger blocks and
# saw-tooths up to the card limit (internal flush-retry, no OOM). expandable
# segments absorb growing allocations smoothly (same setting as exp4).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# FA4 persistent JIT cache (shared with exp2).
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR="${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:-$(pwd)/cache}"

SCRIPT="experiments/exp3_accuracy/run.py"

# ============= USABLE MODELS (--model) =============
#   parcae-1.3b  (T_TOTAL=8)   |   ouro-2.6b (T_TOTAL=4)     ← exp3 main models
#   (also: parcae-140m/370m/770m, ouro-1.4b — see exp2 registry)
#   ouro-1.4b-thinking | ouro-2.6b-thinking — reasoning SFT (same arch); chat
#     models: PROMPT_FORMAT=auto applies the chat template + <think>. Use a
#     large MAX_NEW_TOKENS (2048+) and consider NUM_FEWSHOT=0 (zero-shot user turn).
#   huginn-3.5b
#   NOTE: set T_TOTAL to the model's natural depth (parcae 8, Ouro 4).
# ===================================================

# ============= CONFIG (edit me, or override via env) =============
MODEL="${MODEL:-huginn-3.5b}"                 # see USABLE MODELS above
DATASET="${DATASET:-math500}"                   # gsm8k | math500
LIMIT="${LIMIT:-0}"                          # first N problems (0 = full set: gsm8k 1319 / math500 500)
NUM_FEWSHOT="${NUM_FEWSHOT:-}"                # blank = default (8 gsm8k / 4 math500)
T_TOTAL="${T_TOTAL:-32}"                        # 8 for parcae, 4 for ouro-2.6b
T_DRAFT="${T_DRAFT:-4}"                        # single value (sweep across runs)
DECODE="${DECODE:-both}"                       # both | ar | ssd
SAMPLING="${SAMPLING:-greedy}"                 # greedy | temperature (speculative sampling at T; AR samples at T too)
TEMPERATURE="${TEMPERATURE:-1.0}"              # T > 0, used only when SAMPLING=temperature
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"        # gsm8k ~256, MATH ~512 
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
PROMPT_FORMAT="${PROMPT_FORMAT:-auto}"         # auto | raw | chat  (auto = chat for *-thinking, else raw)
DTYPE="${DTYPE:-bfloat16}"                      # parcae fp16 NaN — bf16/fp32
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-100}"
STATE_INIT="${STATE_INIT:-random}"             # random (model-native like-init, seeded) | zero 
WARMUP="${WARMUP:-1}"
OUT_DIR="${OUT_DIR:-results/exp3}"

# ----- lossy levers -----
# KV_BUDGET_S: R block KV-share budget s (0 or >=T_TOTAL = off/exp2).
KV_BUDGET_S="${KV_BUDGET_S:-1}"
EXIT_THRESHOLD="${EXIT_THRESHOLD:-}"
EXIT_HIDDEN_EPS="${EXIT_HIDDEN_EPS:-}"
# =================================================================

TAG="T${T_TOTAL}_d${T_DRAFT}_s${KV_BUDGET_S}_tau${EXIT_THRESHOLD:-off}_eps${EXIT_HIDDEN_EPS:-off}_${DECODE}_n${LIMIT}"
if [ "${SAMPLING}" = "temperature" ]; then TAG="${TAG}_temp${TEMPERATURE}_seed${SEED}"; fi   # greedy keeps the old names
OUT_FILE="${OUT_DIR}/${MODEL}_${DATASET}_${TAG}.json"
mkdir -p "${OUT_DIR}"

ARGS=(
    --model "${MODEL}"
    --dataset "${DATASET}"
    --limit "${LIMIT}"
    --T-total "${T_TOTAL}"
    --T-draft "${T_DRAFT}"
    --kv-budget-s "${KV_BUDGET_S}"
    --decode "${DECODE}"
    --sampling "${SAMPLING}"
    --temperature "${TEMPERATURE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --max-prompt-tokens "${MAX_PROMPT_TOKENS}"
    --prompt-format "${PROMPT_FORMAT}"
    --dtype "${DTYPE}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --state-init "${STATE_INIT}"
    --warmup "${WARMUP}"
    --output "${OUT_FILE}"
)
if [ -n "${NUM_FEWSHOT}" ]; then ARGS+=(--num-fewshot "${NUM_FEWSHOT}"); fi
if [ -n "${EXIT_THRESHOLD}" ]; then ARGS+=(--exit-threshold "${EXIT_THRESHOLD}"); fi
if [ -n "${EXIT_HIDDEN_EPS}" ]; then ARGS+=(--exit-hidden-eps "${EXIT_HIDDEN_EPS}"); fi

echo "=> PYTHONPATH=. python ${SCRIPT} ${ARGS[*]}"
PYTHONPATH=. python "${SCRIPT}" "${ARGS[@]}"

