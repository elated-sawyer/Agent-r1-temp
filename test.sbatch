#!/bin/bash
#SBATCH --job-name=test_eval_val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=ai_science

set -eo pipefail

#======================================
# Configurable parameters (override via --export):
#   sbatch --export=ALL,DATASET=retro,MAX_TURNS=100,FORCE_NOLOOP=True,API_MODEL_NAME=Qwen/Qwen3.5-35B-A3B,API_KEY_VAR=pjlab_APImodel_key_bailian test.sbatch
#======================================
DATASET="${DATASET:-retro}"
MAX_TURNS="${MAX_TURNS:-100}"
FORCE_NOLOOP="${FORCE_NOLOOP:-True}"
API_MODEL_NAME="${API_MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
API_KEY_VAR="${API_KEY_VAR:-pjlab_APImodel_key}"
API_URL_VAR="${API_URL_VAR:-pjlab_APImodel_url}"

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

EXPERIMENT_NAME="ppo_retro_test190_t0_${DATASET}"
CHECKPOINT_DIR="./checkpoints/retro_qwen2.5-7b-instruct-1M_10_test/${EXPERIMENT_NAME}"

API_LOG_TAG="${API_MODEL_NAME//\//_}"
VAL_TAG="$(basename "$VAL_FILES" .parquet)"
LOG_FILE="logs/test_${DATASET}_api-${API_LOG_TAG}_val-${VAL_TAG}_turns-${MAX_TURNS}_noloop-${FORCE_NOLOOP}_job-${SLURM_JOB_ID}.log"
mkdir -p logs

#======================================
# Environment Variables
#======================================

NNODES="${SLURM_JOB_NUM_NODES:-1}"
GPUS_PER_NODE=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | awk -F',' '{print NF}')

export MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507   # Only used for tokenizer loading
MODEL_TAG="${MODEL_PATH##*/}"

export CPATH="/usr/include:${CPATH:-}"
export HF_HUB_OFFLINE=1

# API model credentials (set these to use API-based evaluation)
export pjlab_APImodel_key="${!API_KEY_VAR}"
export pjlab_APImodel_url="${!API_URL_VAR}"

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
echo "LOG_FILE=$LOG_FILE"
echo "=== Cluster Info ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "NNODES=$NNODES"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
nvidia-smi || true

#======================================
# Run validation-only pipeline
#======================================

srun env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    bash -c "source ~/miniforge3/etc/profile.d/conda.sh && conda activate retro_r1 && \
    PYTHONUNBUFFERED=1 python3 -m agent_r1.src.main_agent_retro_noback \
    data.val_files=\"$VAL_FILES\" \
    data.max_prompt_length=32768 \
    data.max_response_length=32768 \
    data.max_start_length=1024 \
    data.max_tool_response_length=4096 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.logger=['console','wandb'] \
    trainer.project_name=retro_qwen2.5-7b-instruct-1M_10_test \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    tool.debug=True \
    tool.api_max_concurrency=32 \
    tool.max_turns=$MAX_TURNS \
    tool.topk=10 \
    tool.shuffle=False \
    tool.maxstep=30 \
    tool.force_noloop=$FORCE_NOLOOP \
    tool.use_batch_tool_calls=False \
    tool.env='retro_noback_V4' \
    tool.use_api_model=True \
    tool.api_model_name='\"$API_MODEL_NAME\"'" \
    2>&1 | tee "$LOG_FILE"
