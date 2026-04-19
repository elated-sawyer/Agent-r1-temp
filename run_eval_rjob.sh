#!/usr/bin/env bash
set -euo pipefail

# Always run from the repo root (directory of this script) so `python -m agent_r1.*` works,
# regardless of the cwd chosen by rjob.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

# ======================================
# Configurable parameters (override via env):
# Example:
# DATASET=retro MAX_TURNS=100 FORCE_NOLOOP=True \
# API_MODEL_NAME=Qwen/Qwen3.5-35B-A3B \
# API_KEY_VAR=pjlab_APImodel_key_bailian \
# API_URL_VAR=pjlab_APImodel_url_bailian \
# bash run_eval_rjob.sh
# ======================================

DATASET="${DATASET:-retro}"
MAX_TURNS="${MAX_TURNS:-100}"
FORCE_NOLOOP="${FORCE_NOLOOP:-True}"
API_MODEL_NAME="${API_MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
API_KEY_VAR="${API_KEY_VAR:-pjlab_APImodel_key}"
API_URL_VAR="${API_URL_VAR:-pjlab_APImodel_url}"
# Set True when the API accepts no key (e.g. local OpenAI-compatible gateway).
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
VAL_RESUME="${VAL_RESUME:-True}"

case "$DATASET" in
    chembl)
        VAL_FILES="./data/reaction_pathway_search/validation_chembl_1000.parquet"
        ;;
    retro)
        VAL_FILES="./data/reaction_pathway_search/validation_retro_190.parquet"
        ;;
    *)
        echo "ERROR: unknown DATASET='$DATASET' (expected 'chembl' or 'retro')" >&2
        exit 1
        ;;
esac

API_LOG_TAG="${API_MODEL_NAME//\//_}"
VAL_TAG="$(basename "$VAL_FILES" .parquet)"
EXPERIMENT_NAME="ppo_retro_test190_t0_${DATASET}"

# Stable path for resume
DEFAULT_CHECKPOINT_DIR="./checkpoints/val/test_${DATASET}_api-${API_LOG_TAG}_val-${VAL_TAG}_turns-${MAX_TURNS}_noloop-${FORCE_NOLOOP}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$DEFAULT_CHECKPOINT_DIR}"

mkdir -p logs

# rjob 下通常没有 SLURM_JOB_ID，这里自己生成一个
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
LOG_FILE="logs/test_${DATASET}_api-${API_LOG_TAG}_val-${VAL_TAG}_turns-${MAX_TURNS}_noloop-${FORCE_NOLOOP}_run-${RUN_ID}.log"

# ======================================
# Runtime / cluster info
# ======================================

NNODES="${NNODES:-1}"

# rjob 这次是 --gpu=0，所以默认 0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    GPUS_PER_NODE=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
else
    GPUS_PER_NODE=0
fi

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-0.8B}"   # only used for tokenizer loading
MODEL_TAG="${MODEL_PATH##*/}"

export CPATH="/usr/include:${CPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# API model credentials
export pjlab_APImodel_key="${!API_KEY_VAR:-}"
export pjlab_APImodel_url="${!API_URL_VAR:-}"


echo "=== Run Config ==="
echo "DATASET=$DATASET"
echo "VAL_FILES=$VAL_FILES"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "API_MODEL_NAME=$API_MODEL_NAME"
echo "API_KEY_VAR=$API_KEY_VAR"
echo "API_URL_VAR=$API_URL_VAR"
echo "MAX_TURNS=$MAX_TURNS"
echo "FORCE_NOLOOP=$FORCE_NOLOOP"
echo "VAL_BATCH_SIZE=$VAL_BATCH_SIZE"
echo "VAL_RESUME=$VAL_RESUME"
echo "LOG_FILE=$LOG_FILE"

echo "=== Runtime Info ==="
echo "RUN_ID=$RUN_ID"
echo "NNODES=$NNODES"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

# ======================================
# Activate env
# ======================================
CONDA_BASE="${CONDA_BASE:-/mnt/shared-storage-user/wangzifu/miniconda3}"
if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: conda.sh not found under CONDA_BASE='$CONDA_BASE'." >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-Retro_R1}"

# ======================================
# Run validation-only pipeline
# ======================================
PYTHONUNBUFFERED=1 python3 -m agent_r1.src.main_agent_retro_noback \
    data.val_files="$VAL_FILES" \
    data.max_prompt_length=32768 \
    data.max_response_length=32768 \
    data.max_start_length=1024 \
    data.max_tool_response_length=4096 \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.val_resume="$VAL_RESUME" \
    trainer.logger="['console','wandb']" \
    trainer.project_name=retro_qwen2.5-7b-instruct-1M_10_test \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    tool.debug=True \
    tool.api_max_concurrency=32 \
    tool.max_turns="$MAX_TURNS" \
    tool.topk=10 \
    tool.shuffle=False \
    tool.maxstep=30 \
    tool.force_noloop="$FORCE_NOLOOP" \
    tool.use_batch_tool_calls=False \
    tool.env='retro_noback_V4' \
    tool.use_api_model=True \
    tool.api_model_name="$API_MODEL_NAME" \
    2>&1 | tee "$LOG_FILE"