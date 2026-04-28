#!/usr/bin/env bash
# End-to-end local pipeline: merge LoRA -> serve vLLM -> run validation.
#
# Steps:
#   1. Merge the LoRA adapter into the base model with `llamafactory-cli export`
#      (one-time; cached under MERGED_DIR). Skipped if MERGED_DIR already exists.
#        env: $LF_ENV       (default: LlamaFactory)
#   2. Start vLLM in the background serving the merged model on LOCAL_API_PORT,
#      wait for /v1/models to respond.
#        env: $SERVE_ENV    (default: qwen35_vllm)
#   3. Run agent_r1.src.main_agent_retro_noback against the local endpoint.
#        env: $EVAL_ENV     (default: Retro_R1)
#   4. On exit, stop vLLM (TERM, then KILL after a grace window).
#
# Override anything via env vars. Common ones:
#   MODEL_MODE         sft | base                (default: sft)
#                      sft  -> merge LoRA, serve merged model
#                      base -> serve BASE_MODEL_PATH directly (no merge)
#   ADAPTER_DIR        path to the LoRA adapter directory (ignored when MODEL_MODE=base)
#   MERGED_DIR         where to save the merged model (cached; ignored when MODEL_MODE=base)
#   BASE_MODEL_PATH    base model (used directly when MODEL_MODE=base; merge source when sft)
#   DATASET            chembl | retro            (default: retro)
#   BACKTRACK          true | false              (default: false)
#                      true  -> main_agent_retro  + tool.env=retro        (back_state allowed)
#                      false -> main_agent_retro_noback + tool.env=retro_noback_V4
#   MAX_TURNS          tool-use turns per traj   (default: 100)
#   FORCE_NOLOOP       True | False              (default: True)
#                      only used when BACKTRACK=false
#   VAL_BATCH_SIZE                                (default: 128)
#   VAL_RESUME         True | False              (default: True)
#   TP_SIZE            tensor-parallel size      (default: from CUDA_VISIBLE_DEVICES, fallback 4)
#   LOCAL_API_PORT     vLLM port                 (default: 20011)
#   API_MODEL_NAME     name advertised to vLLM   (default: retro_sft_full)
#   VLLM_MAX_MODEL_LEN vLLM --max-model-len      (default: 65536)
#   GPU_MEM_UTIL       --gpu-memory-utilization  (default: 0.85)
#   SKIP_MERGE         skip step 1 even if dir missing (default: 0)
#   EVAL_MODEL_PATH    override tokenizer path for eval (default: MERGED_DIR)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# See run_eval_rjob.sh: rdchiral, mlp_retrosyn, verl, then repo root.
export PYTHONPATH="${SCRIPT_DIR}/packages/rdchiral:${SCRIPT_DIR}/packages/mlp_retrosyn:${SCRIPT_DIR}/verl:${SCRIPT_DIR}:${PYTHONPATH:-}"

# --- CUDA toolchain from shared GPFS (needed for flashinfer JIT in SERVE_ENV) ---
export CUDA_HOME="${CUDA_HOME:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/soft/cuda/12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# ======================================
# Eval config (mirrors run_eval_rjob.sh)
# ======================================
DATASET="${DATASET:-retro}"
BACKTRACK="${BACKTRACK:-false}"
MAX_TURNS="${MAX_TURNS:-100}"
FORCE_NOLOOP="${FORCE_NOLOOP:-True}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
VAL_RESUME="${VAL_RESUME:-True}"
# Number of times to re-run the eval on the same val set (for measuring result
# variance). Each run gets its own CHECKPOINT_DIR (./runN) and log; after all
# runs finish, per-round metrics and overall mean/std/var are printed.
TOTAL_RUN="${TOTAL_RUN:-1}"

case "$DATASET" in
    chembl) VAL_FILES="./data/reaction_pathway_search/validation_chembl_1000.parquet" ;;
    retro)  VAL_FILES="./data/reaction_pathway_search/validation_retro_190.parquet" ;;
    *) echo "ERROR: unknown DATASET='$DATASET' (expected 'chembl' or 'retro')" >&2; exit 1 ;;
esac

# BACKTRACK picks the main module and the tool.env value:
#   true  -> ToolEnvRetro       (back_state tool enabled)
#   false -> ToolEnvRetroNoBack (no back_state; force_noloop applies)
case "$BACKTRACK" in
    true|True|TRUE|1)
        MAIN_MODULE="agent_r1.src.main_agent_retro"
        TOOL_ENV_NAME="retro"
        BACK_TAG="back"
        ;;
    false|False|FALSE|0)
        MAIN_MODULE="agent_r1.src.main_agent_retro_noback"
        TOOL_ENV_NAME="retro_noback_V4"
        BACK_TAG="noback"
        ;;
    *)
        echo "ERROR: BACKTRACK='$BACKTRACK' (expected true|false)" >&2
        exit 1
        ;;
esac

# ======================================
# Model paths
# ======================================
# MODEL_MODE controls which weights vLLM serves:
#   sft  -> merge LoRA adapter into base and serve the merged model (default)
#   base -> serve BASE_MODEL_PATH directly; step 1 (LoRA merge) is skipped
MODEL_MODE="${MODEL_MODE:-sft}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/shared-storage-user/ai4cmp/models/0401-preview}"
ADAPTER_DIR="${ADAPTER_DIR:-/mnt/shared-storage-gpfs2/wangzifugpfs2/LlamaFactory_saves/retro_sft/full/0401-preview/lora/20260425_235107_20085}"
MERGED_DIR="${MERGED_DIR:-${ADAPTER_DIR}/merged}"
SKIP_MERGE="${SKIP_MERGE:-0}"

case "$MODEL_MODE" in
    sft)
        SERVE_DIR="$MERGED_DIR"
        MODEL_TAG="sft-$(basename "$ADAPTER_DIR")"
        ;;
    base)
        SERVE_DIR="$BASE_MODEL_PATH"
        MODEL_TAG="base-$(basename "$BASE_MODEL_PATH")"
        ;;
    *)
        echo "ERROR: MODEL_MODE='$MODEL_MODE' (expected sft|base)" >&2
        exit 1
        ;;
esac

API_MODEL_NAME="${API_MODEL_NAME:-retro_${MODEL_MODE}}"

# ======================================
# Local vLLM endpoint
# ======================================
LOCAL_API_HOST="${LOCAL_API_HOST:-127.0.0.1}"
LOCAL_API_PORT="${LOCAL_API_PORT:-20011}"
LOCAL_API_URL="http://${LOCAL_API_HOST}:${LOCAL_API_PORT}/v1"

VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-1800}"   # seconds

# Tensor-parallel: default to count of visible GPUs (or 4 if unset).
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    DETECTED_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
else
    DETECTED_GPUS=4
fi
TP_SIZE="${TP_SIZE:-$DETECTED_GPUS}"

# ======================================
# Conda envs — each step runs in its own env
# ======================================
CONDA_BASE="${CONDA_BASE:-/mnt/shared-storage-gpfs2/wangzifugpfs2/miniconda3}"
LF_ENV="${LF_ENV:-LlamaFactory}"
SERVE_ENV="${SERVE_ENV:-qwen35_vllm}"
EVAL_ENV="${EVAL_ENV:-Retro_R1}"

if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: conda.sh not found under CONDA_BASE='$CONDA_BASE'." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ======================================
# Logging
# ======================================
mkdir -p logs
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
VAL_TAG="$(basename "$VAL_FILES" .parquet)"
EXPERIMENT_NAME="ppo_retro_test190_t0_${DATASET}_local_${MODEL_TAG}_${BACK_TAG}"

DEFAULT_CHECKPOINT_DIR="./checkpoints/val/test_${DATASET}_local-${MODEL_TAG}_val-${VAL_TAG}_turns-${MAX_TURNS}_${BACK_TAG}_noloop-${FORCE_NOLOOP}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$DEFAULT_CHECKPOINT_DIR}"

EVAL_LOG="logs/local_eval_${DATASET}_${MODEL_TAG}_turns-${MAX_TURNS}_${BACK_TAG}_noloop-${FORCE_NOLOOP}_run-${RUN_ID}.log"
VLLM_LOG="logs/vllm_${MODEL_TAG}_run-${RUN_ID}.log"
MERGE_LOG="logs/merge_${MODEL_TAG}_run-${RUN_ID}.log"

echo "=== Run Config ==="
echo "RUN_ID=$RUN_ID"
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "DATASET=$DATASET    VAL_FILES=$VAL_FILES"
echo "BACKTRACK=$BACKTRACK  MAIN_MODULE=$MAIN_MODULE  TOOL_ENV_NAME=$TOOL_ENV_NAME"
echo "MODEL_MODE=$MODEL_MODE  MODEL_TAG=$MODEL_TAG  SERVE_DIR=$SERVE_DIR"
echo "BASE_MODEL_PATH=$BASE_MODEL_PATH"
echo "ADAPTER_DIR=$ADAPTER_DIR"
echo "MERGED_DIR=$MERGED_DIR"
echo "API_MODEL_NAME=$API_MODEL_NAME"
echo "LOCAL_API_URL=$LOCAL_API_URL"
echo "TP_SIZE=$TP_SIZE   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN  GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "MAX_TURNS=$MAX_TURNS  FORCE_NOLOOP=$FORCE_NOLOOP  VAL_BATCH_SIZE=$VAL_BATCH_SIZE  VAL_RESUME=$VAL_RESUME  TOTAL_RUN=$TOTAL_RUN"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "LF_ENV=$LF_ENV   SERVE_ENV=$SERVE_ENV   EVAL_ENV=$EVAL_ENV"
echo "MERGE_LOG=$MERGE_LOG"
echo "VLLM_LOG=$VLLM_LOG"
echo "EVAL_LOG=$EVAL_LOG"
echo "==================="

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

# ======================================
# Step 1: merge LoRA into base (one-time; cached) — LF_ENV
# Skipped entirely when MODEL_MODE=base (nothing to merge).
# ======================================
need_merge=1
if [[ "$MODEL_MODE" == "base" ]]; then
    echo "[merge] MODEL_MODE=base; skipping LoRA merge — serving base model directly from $BASE_MODEL_PATH."
    need_merge=0
elif [[ -f "$MERGED_DIR/config.json" ]] && \
     ls "$MERGED_DIR"/*.safetensors >/dev/null 2>&1; then
    need_merge=0
fi
if [[ "$SKIP_MERGE" == "1" ]]; then
    echo "[merge] SKIP_MERGE=1; assuming MERGED_DIR is ready."
    need_merge=0
fi

if [[ $need_merge -eq 1 ]]; then
    echo "[merge] merging LoRA -> $MERGED_DIR (env=$LF_ENV)"
    mkdir -p "$MERGED_DIR"
    MERGE_YAML="$MERGED_DIR/_export_config.yaml"
    cat > "$MERGE_YAML" <<YAML_EOF
### Auto-generated by run_serve_and_eval.sh — LoRA merge config
model_name_or_path: ${BASE_MODEL_PATH}
adapter_name_or_path: ${ADAPTER_DIR}
template: qwen3_5
finetuning_type: lora
trust_remote_code: true

export_dir: ${MERGED_DIR}
export_size: 5
export_device: cpu
export_legacy_format: false
YAML_EOF
    echo "[merge] config:"
    cat "$MERGE_YAML"

    conda activate "$LF_ENV"
    HF_HUB_OFFLINE=1 llamafactory-cli export "$MERGE_YAML" 2>&1 | tee "$MERGE_LOG"
    EXPORT_RC=${PIPESTATUS[0]}
    conda deactivate
    if [[ $EXPORT_RC -ne 0 ]]; then
        echo "[merge] ERROR: llamafactory-cli export failed (rc=$EXPORT_RC). See $MERGE_LOG" >&2
        exit $EXPORT_RC
    fi
    # Copy any *.py files from base so trust_remote_code keeps working after export.
    for f in "$BASE_MODEL_PATH"/*.py; do
        [[ -e "$f" ]] || continue
        bn="$(basename "$f")"
        [[ -e "$MERGED_DIR/$bn" ]] || cp "$f" "$MERGED_DIR/$bn"
    done
    echo "[merge] done -> $MERGED_DIR"
else
    if [[ "$MODEL_MODE" == "base" ]]; then
        :   # already logged above
    else
        echo "[merge] reusing existing merged model at $MERGED_DIR"
    fi
fi

# ======================================
# Step 2: launch vLLM in background — SERVE_ENV
# ======================================
conda activate "$SERVE_ENV"

cleanup() {
    rc=$?
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[cleanup] stopping vLLM (pid=$VLLM_PID) ..."
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[cleanup] vLLM did not exit; sending KILL"
            kill -KILL "$VLLM_PID" 2>/dev/null || true
        fi
    fi
    exit $rc
}
trap cleanup EXIT INT TERM

echo "[vllm] launching on ${LOCAL_API_HOST}:${LOCAL_API_PORT} (TP=$TP_SIZE, env=$SERVE_ENV) — serving $SERVE_DIR"
# Note: --served-model-name controls the value the API expects in `model=...`.
# We pin it to API_MODEL_NAME so the eval framework's API_MODEL_NAME flag matches.
env HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
    vllm serve "$SERVE_DIR" \
    --host "$LOCAL_API_HOST" \
    --port "$LOCAL_API_PORT" \
    --served-model-name "$API_MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --trust-remote-code \
    --dtype bfloat16 \
    --disable-uvicorn-access-log \
    --disable-log-stats \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "[vllm] pid=$VLLM_PID  log=$VLLM_LOG"

# Wait for /v1/models to respond. vLLM startup for a 60GB+ MoE model can take
# a few minutes; bound it with VLLM_READY_TIMEOUT.
echo "[vllm] waiting for readiness (timeout=${VLLM_READY_TIMEOUT}s) ..."
ready=0
for i in $(seq 1 "$VLLM_READY_TIMEOUT"); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[vllm] ERROR: process exited before becoming ready. Tail of log:" >&2
        tail -n 80 "$VLLM_LOG" >&2 || true
        exit 1
    fi
    if curl -sf --max-time 2 "${LOCAL_API_URL}/models" >/dev/null 2>&1; then
        ready=1
        echo "[vllm] ready after ${i}s"
        break
    fi
    sleep 1
done
if [[ $ready -ne 1 ]]; then
    echo "[vllm] ERROR: not ready after ${VLLM_READY_TIMEOUT}s. Tail of log:" >&2
    tail -n 100 "$VLLM_LOG" >&2 || true
    exit 1
fi
curl -s "${LOCAL_API_URL}/models" || true
echo

# Leave SERVE_ENV before switching to EVAL_ENV.
conda deactivate

# ======================================
# Step 3: run validation against local endpoint — EVAL_ENV
# ======================================
conda activate "$EVAL_ENV"

# Point the eval framework at the local vLLM server (replaces the remote URL
# hard-coded in run_eval_rjob.sh: http://100.104.48.113:20010/v1).
export pjlab_APImodel_key="EMPTY"
export pjlab_APImodel_url="${LOCAL_API_URL}"
export CPATH="/usr/include:${CPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Tokenizer/model path used by the framework (loading-only; actual generation
# happens on the vLLM server). Defaults to the merged model so the tokenizer
# matches what's being served; override with EVAL_MODEL_PATH if that env can't
# import the custom Qwen3_5 arch.
EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-$SERVE_DIR}"
# ！do not use the model's chattemplate, use the template of the benchmarks
EVAL_MODEL_PATH="/mnt/shared-storage-user/wangzifu/cache/model/Qwen3.5-0.8B"

NNODES="${NNODES:-1}"
GPUS_PER_NODE=0   # eval runs CPU-only; all GPUs are reserved by vLLM.

echo "=== Eval ==="
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "API_MODEL_NAME=$API_MODEL_NAME"
echo "API_URL=$pjlab_APImodel_url"
echo "EVAL_MODEL_PATH=$EVAL_MODEL_PATH"
echo "EVAL_LOG=$EVAL_LOG"
echo "TOTAL_RUN=$TOTAL_RUN"
echo "============"

RUN_LOGS=()
FINAL_RC=0
for RUN_IDX in $(seq 1 "$TOTAL_RUN"); do
    RUN_CHECKPOINT_DIR="${CHECKPOINT_DIR}/run${RUN_IDX}"
    if [[ "$TOTAL_RUN" == "1" ]]; then
        RUN_EVAL_LOG="$EVAL_LOG"
    else
        RUN_EVAL_LOG="${EVAL_LOG%.log}_run${RUN_IDX}-of-${TOTAL_RUN}.log"
    fi
    RUN_LOGS+=("$RUN_EVAL_LOG")

    echo ""
    echo "=== Eval run ${RUN_IDX}/${TOTAL_RUN} ==="
    echo "RUN_CHECKPOINT_DIR=$RUN_CHECKPOINT_DIR"
    echo "RUN_EVAL_LOG=$RUN_EVAL_LOG"

    set +e
    PYTHONUNBUFFERED=1 python3 -m "$MAIN_MODULE" \
        data.val_files="$VAL_FILES" \
        data.max_prompt_length=32768 \
        data.max_response_length=32768 \
        data.max_start_length=1024 \
        data.max_tool_response_length=4096 \
        data.val_batch_size="$VAL_BATCH_SIZE" \
        actor_rollout_ref.model.path="$EVAL_MODEL_PATH" \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=False \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        trainer.default_local_dir="$RUN_CHECKPOINT_DIR" \
        trainer.val_resume="$VAL_RESUME" \
        trainer.logger="['console']" \
        trainer.project_name=retro_qwen2.5-7b-instruct-1M_10_test \
        trainer.experiment_name="$EXPERIMENT_NAME" \
        trainer.n_gpus_per_node="$GPUS_PER_NODE" \
        trainer.nnodes="$NNODES" \
        tool.debug=True \
        tool.api_max_concurrency=128 \
        tool.max_turns="$MAX_TURNS" \
        tool.topk=10 \
        tool.shuffle=False \
        tool.maxstep=30 \
        tool.force_noloop="$FORCE_NOLOOP" \
        tool.use_batch_tool_calls=False \
        tool.env="$TOOL_ENV_NAME" \
        tool.use_api_model=True \
        tool.api_model_name="$API_MODEL_NAME" \
        2>&1 | tee "$RUN_EVAL_LOG"
    RUN_RC=${PIPESTATUS[0]}
    set -e
    echo "[eval] run ${RUN_IDX}/${TOTAL_RUN} finished with rc=$RUN_RC"
    if [[ $RUN_RC -ne 0 ]]; then
        FINAL_RC=$RUN_RC
    fi
done

# --- Aggregate metrics across runs ---------------------------------------
echo ""
echo "=== Aggregating metrics across ${TOTAL_RUN} run(s) ==="
PYTHONUNBUFFERED=1 python3 - "${RUN_LOGS[@]}" <<'PYAGG'
import math, re, sys
from pathlib import Path

logs = sys.argv[1:]
# Metric keys produced by main_agent_retro*; extend here if more are added.
METRIC_KEYS = [
    "val/test_score/reaction_pathway_search",
    "val/end_score/reaction_pathway_search",
    "val/answer_score/reaction_pathway_search",
    "val/format_score/reaction_pathway_search",
    "val/turns/reaction_pathway_search",
]

def parse_log(path):
    try:
        text = Path(path).read_text(errors="replace")
    except FileNotFoundError:
        return None
    # Prefer the final "Validation metrics: {...}" pprint block for full
    # float precision. pprint may wrap the dict across adjacent string
    # literals, inserting `"\n "` between key and value — the regex allows
    # whitespace/quotes on either side of the `:`. Take the highest-precision
    # (longest) match per key to avoid the truncated `step:0 - ...` summary
    # line that comes later in the same region.
    tail = text[-200_000:]
    marker = "Validation metrics"
    idx = tail.rfind(marker)
    if idx < 0:
        return None
    region = tail[idx:]
    vals = {}
    for key in METRIC_KEYS:
        pat = re.escape(key) + r"['\"\s]*:[\s'\"]*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
        ms = re.findall(pat, region)
        if not ms:
            continue
        # Pick the match with the most digits (highest precision).
        best = max(ms, key=lambda s: len(s))
        vals[key] = float(best)
    return vals or None

runs = []
for i, log in enumerate(logs, 1):
    r = parse_log(log)
    runs.append(r)

# Per-run table
print()
print("Per-run metrics:")
header = ["run"] + [k.split("/")[-2] for k in METRIC_KEYS]
print("  " + "  ".join(f"{h:>14s}" for h in header))
for i, r in enumerate(runs, 1):
    if r is None:
        print(f"  {i:>14d}  " + "  ".join(f"{'<missing>':>14s}" for _ in METRIC_KEYS))
        continue
    row = [f"{i:>14d}"]
    for k in METRIC_KEYS:
        row.append(f"{r[k]:>14.6f}" if k in r else f"{'<n/a>':>14s}")
    print("  " + "  ".join(row))

# Aggregate stats
valid = [r for r in runs if r is not None]
print()
print(f"Aggregate over {len(valid)}/{len(runs)} successful run(s):")
if not valid:
    print("  (no successful runs to aggregate)")
    sys.exit(0)

def stats(xs):
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n  # population variance
    return mean, math.sqrt(var), var, n

for k in METRIC_KEYS:
    xs = [r[k] for r in valid if k in r]
    if not xs:
        continue
    mean, std, var, n = stats(xs)
    short = k.split("/")[-2]
    print(f"  {short:<14s}  mean={mean:.6f}  std={std:.6f}  var={var:.6g}  n={n}  min={min(xs):.6f}  max={max(xs):.6f}")
PYAGG

echo ""
echo "[eval] all runs finished; final rc=$FINAL_RC (vLLM will be stopped by cleanup)"
exit $FINAL_RC
