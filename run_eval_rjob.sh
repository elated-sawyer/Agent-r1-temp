#!/bin/bash
set -eo pipefail

#======================================
# Locate script dir and cd into the repo
# (rjob's CWD may not be the repo root)
#======================================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#======================================
# Dataset selection: "chembl" or "retro"
# Override via:  rjob submit ... -e DATASET=chembl
#======================================
DATASET="${DATASET:-retro}"

case "$DATASET" in
    chembl)
        VAL_FILES="${VAL_FILES:-./data/reaction_pathway_search/validation_chembl_1000.parquet}"
        ;;
    retro)
        VAL_FILES="${VAL_FILES:-./data/reaction_pathway_search/validation_retro_190.parquet}"
        ;;
    *)
        echo "ERROR: unknown DATASET='$DATASET' (expected 'chembl' or 'retro')" >&2
        exit 1
        ;;
esac

VAL_TAG="$(basename "$VAL_FILES" .parquet)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ppo_retro_test190_t0_${DATASET}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/retro_qwen2.5-7b-instruct-1M_10_test/${EXPERIMENT_NAME}}"

#======================================
# Cluster / runtime variables
#======================================
NNODES="${NNODES:-1}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
else
    GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"

export MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/ai4cmp/models/0401-preview}"
MODEL_TAG="${MODEL_PATH##*/}"

API_MODEL_NAME="${API_MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
API_LOG_TAG="${API_MODEL_NAME//\//_}"

export CPATH="/usr/include:${CPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Job id for log/tmp naming — rjob may set its own; fall back to timestamp
JOB_ID="${RJOB_JOB_ID:-${JOB_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}}"
export RAY_TMPDIR="/tmp/${USER:-$(id -un)}/ray/${JOB_ID}"
mkdir -p "$RAY_TMPDIR"

# API model credentials — must be injected via rjob -e
: "${pjlab_APImodel_key:?pjlab_APImodel_key is not set (pass via rjob -e)}"
: "${pjlab_APImodel_url:?pjlab_APImodel_url is not set (pass via rjob -e)}"
export pjlab_APImodel_key pjlab_APImodel_url

mkdir -p logs
LOG_FILE="logs/test_main_${DATASET}_api-${API_LOG_TAG}_val-${VAL_TAG}_job-${JOB_ID}.log"

echo "=== Run Config ==="
echo "DATASET=$DATASET"
echo "VAL_FILES=$VAL_FILES"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "API_MODEL_NAME=$API_MODEL_NAME"
echo "LOG_FILE=$LOG_FILE"
echo "=== Cluster Info ==="
echo "JOB_ID=$JOB_ID"
echo "NNODES=$NNODES"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
nvidia-smi || true
test -d "$MODEL_PATH" && ls "$MODEL_PATH/tokenizer_config.json" || { echo "MODEL_PATH missing: $MODEL_PATH"; exit 1; }

#======================================
# Tunable hyperparameters (overridable via rjob -e)
#======================================
MAX_TURNS="${MAX_TURNS:-100}"
TOPK="${TOPK:-10}"
MAXSTEP="${MAXSTEP:-30}"
API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-50}"
TOOL_ENV="${TOOL_ENV:-retro_noback_V4}"

#======================================
# Conda env activation (optional)
# Set CONDA_SH and CONDA_ENV via -e if your image ships miniforge/miniconda
#======================================
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-Retro_R1}"
if [[ -f "$CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
else
    echo "NOTE: $CONDA_SH not found; using system python3"
fi

#======================================
# CUDA / cuDNN / framework env
#======================================
export CUDA_HOME="${CUDA_HOME:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/soft/cuda/12.6}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDNN_PATH="${CUDNN_PATH:-/mnt/shared-storage-user/wangzifu/miniconda3/envs/Retro_R1/lib/python3.10/site-packages/nvidia/cudnn}"
export CPLUS_INCLUDE_PATH="${CUDNN_PATH}/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="${CUDNN_PATH}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDNN_PATH}/lib:${LD_LIBRARY_PATH:-}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"

#======================================
# Run evaluation
#======================================
export_pth=./export_model/retro_qwen2.5-7b-instruct-1M/ppo_retro/global_step_190/actor

env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    PYTHONUNBUFFERED=1 python3 -m agent_r1.src.main_agent_retro_noback \
    data.train_files="./data/reaction_pathway_search/train_h4_10.parquet" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=128 \
    data.max_prompt_length=32768 \
    data.max_response_length=32768 \
    data.max_start_length=1024 \
    data.max_tool_response_length=4096 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.wrap_policy.disable=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path="$MODEL_PATH" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.model.fsdp_config.fsdp_size=8 \
    critic.ppo_max_token_len_per_gpu=65536 \
    algorithm.use_process_rewards=False \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.critic_warmup=3 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=retro_qwen2.5-7b-instruct-1M_10_test \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=0 \
    trainer.val_before_train=True \
    tool.api_max_concurrency="$API_MAX_CONCURRENCY" \
    tool.debug=True \
    tool.max_turns="$MAX_TURNS" \
    tool.topk="$TOPK" \
    tool.shuffle=False \
    tool.maxstep="$MAXSTEP" \
    tool.force_noloop=True \
    tool.use_batch_tool_calls=False \
    tool.env="$TOOL_ENV" \
    tool.use_api_model=False \
    tool.api_model_name="$API_MODEL_NAME" \
    2>&1 | tee "$LOG_FILE"
