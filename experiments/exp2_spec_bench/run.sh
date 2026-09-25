#!/usr/bin/env bash
#
# experiments/exp2_spec_bench/run.sh
#
# Convenience launcher for experiment 2 (wavefront self-speculative decoding
# wall-clock benchmark). Edit the CONFIG block below, or override via env, then:
#
#     bash experiments/exp2_spec_bench/run.sh
#
# Sweeps a comma-separated list of T_draft values in a single run.
#

set -euo pipefail
cd "$(dirname "$0")/../.."

# FlashAttention-4 persistent (static) JIT compile cache — avoids recompiling
# kernels on every run. Cache is namespaced by FA4 source fingerprint, so it
# auto-invalidates if external/flash-attention sources or cutlass/python change.
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR="${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:-$(pwd)/cache}"

# CUDA allocator: expandable segments greatly reduce reserved-memory growth from
# many variable-size allocations (DtV drafts/verify packs + coda grow with the
# sequence, fragmenting the default pool). Cheap, transparent, no code change.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT="experiments/exp2_spec_bench/run.py"

# ============= USABLE MODELS (--model) =============
#   parcae family (T_TOTAL=8, mean_recurrence=8):
#     parcae-140m | parcae-370m | parcae-770m | parcae-1.3b   (SandyResearch/*)
#   Ouro family (T_TOTAL=4, total_ut_steps=4):
#     ouro-1.4b (24 layers) | ouro-2.6b (48 layers)            (ByteDance/Ouro-*)
#     ouro-1.4b-thinking | ouro-2.6b-thinking  — reasoning SFT, same architecture;
#       chat models: PROMPT_FORMAT=auto wraps prompts in the chat template with
#       <think>; they emit long reasoning, so raise MAX_NEW_TOKENS (e.g. 2048+)
#   RDM (T_TOTAL=32):
#     huginn-3.5b
#   NOTE: set T_TOTAL to the model's natural depth (8 for parcae, 4 for Ouro);
#         a larger T_TOTAL runs out-of-distribution (run.py warns).
# ===================================================

# ============= CONFIG (edit me, or override via env) =============
MODEL="${MODEL:-huginn-3.5b}"                 # see USABLE MODELS above
CORPUS="${CORPUS:-spec_bench}"                # debug | spec_bench
INDEX_START="${INDEX_START:-81}"              # SpecBench question_id lower bound (clamped >=81)
INDEX_END="${INDEX_END:-560}"                 # upper bound (clamped <=560). Split across GPUs by range.
T_TOTAL="${T_TOTAL:-32}"                       # 8 for parcae, 4 for ouro-1.4b
T_DRAFT="${T_DRAFT:-4}"                 # comma-separated sweep
SCHEDULER="${SCHEDULER:-wfd}"                 # wfd (wavefront) | dtv (draft-then-verify, nested reuse)
DRAFT_LENGTH="${DRAFT_LENGTH:-8}"             # dtv speculation length (# drafts/round; wfd ignores)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-512}"
PROMPT_FORMAT="${PROMPT_FORMAT:-auto}"        # auto | raw | chat  (auto = chat for *-thinking, else raw)
SAMPLING="${SAMPLING:-greedy}"                # greedy | temperature (speculative sampling at T; AR samples at T too)
TEMPERATURE="${TEMPERATURE:-1.0}"             # T > 0, used only when SAMPLING=temperature
DTYPE="${DTYPE:-bfloat16}"                    # NOTE: parcae fp16 produces NaN — bf16 or fp32 only
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
WARMUP="${WARMUP:-1}"
OUT_DIR="${OUT_DIR:-results/exp2}"
# ==================================================================

TAG="T${T_TOTAL}_d$(echo "${T_DRAFT}" | tr ',' '-')_${SCHEDULER}"
if [ "${SCHEDULER}" = "dtv" ]; then TAG="${TAG}${DRAFT_LENGTH}"; fi
if [ "${SAMPLING}" = "temperature" ]; then TAG="${TAG}_temp${TEMPERATURE}"; fi   # greedy keeps the old names
OUT_FILE="${OUT_DIR}/${MODEL}_${CORPUS}_${TAG}_idx${INDEX_START}-${INDEX_END}.json"
mkdir -p "${OUT_DIR}"

ARGS=(
    --model "${MODEL}"
    --corpus "${CORPUS}"
    --index-start "${INDEX_START}"
    --index-end "${INDEX_END}"
    --T-total "${T_TOTAL}"
    --T-draft "${T_DRAFT}"
    --scheduler "${SCHEDULER}"
    --draft-length "${DRAFT_LENGTH}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --max-prompt-tokens "${MAX_PROMPT_TOKENS}"
    --prompt-format "${PROMPT_FORMAT}"
    --sampling "${SAMPLING}"
    --temperature "${TEMPERATURE}"
    --dtype "${DTYPE}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --warmup "${WARMUP}"
    --output "${OUT_FILE}"
)

echo "=> PYTHONPATH=. python ${SCRIPT} ${ARGS[*]}"
PYTHONPATH=. python "${SCRIPT}" "${ARGS[@]}"

# ============= EXAMPLES =============
# Quick dryrun (no SpecBench download, 8 bundled debug prompts, short generation):
#   CORPUS=debug MAX_NEW_TOKENS=64 bash experiments/exp2_spec_bench/run.sh
#
# Full SpecBench (all question_id 81..560):
#   bash experiments/exp2_spec_bench/run.sh
#
# Split across GPUs by index range (one per GPU, in parallel):
#   CUDA_VISIBLE_DEVICES=0 INDEX_START=81  INDEX_END=200 bash experiments/exp2_spec_bench/run.sh
#   CUDA_VISIBLE_DEVICES=1 INDEX_START=201 INDEX_END=320 bash experiments/exp2_spec_bench/run.sh
#   CUDA_VISIBLE_DEVICES=2 INDEX_START=321 INDEX_END=440 bash experiments/exp2_spec_bench/run.sh
#   CUDA_VISIBLE_DEVICES=3 INDEX_START=441 INDEX_END=560 bash experiments/exp2_spec_bench/run.sh
#
# Single T_draft, smaller model:
#   T_DRAFT=2 MODEL=parcae-370m bash experiments/exp2_spec_bench/run.sh
#
# Draft-then-verify scheduler (nested reuse) instead of wavefront, draft_length=4:
#   SCHEDULER=dtv DRAFT_LENGTH=4 CORPUS=debug MAX_NEW_TOKENS=64 bash experiments/exp2_spec_bench/run.sh
#   # compare vs wfd (default) at the same T_total/T_draft to see the two SSD styles
#
# Ouro-1.4B (natural depth = total_ut_steps = 4, so T_TOTAL=4):
#   MODEL=ouro-1.4b T_TOTAL=4 T_DRAFT=1,2 bash experiments/exp2_spec_bench/run.sh
#   # GPU 1 only (GPU 0 busy): prefix CUDA_VISIBLE_DEVICES=1
#
# Ouro-2.6B-Thinking (chat template applied automatically; long <think> outputs):
#   MODEL=ouro-2.6b-thinking T_TOTAL=4 T_DRAFT=1,2 MAX_NEW_TOKENS=2048 bash experiments/exp2_spec_bench/run.sh
#
# Temperature sampling (T=0.7) for AR and SSD alike; sweep SEED for more samples:
#   SAMPLING=temperature TEMPERATURE=0.7 SEED=0 bash experiments/exp2_spec_bench/run.sh
#   # output gets a "_temp0.7" suffix; greedy runs keep their existing file names
