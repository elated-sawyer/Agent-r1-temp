#!/bin/bash
#SBATCH --job-name=test_chembl_val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=ai_science

set -eo pipefail

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
export pjlab_APImodel_key=$pjlab_APImodel_key
export pjlab_APImodel_url=$pjlab_APImodel_url

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
    data.val_files=\"./data/reaction_pathway_search/validation_chembl_1000.parquet\" \
    data.max_prompt_length=32768 \
    data.max_response_length=32768 \
    data.max_start_length=1024 \
    data.max_tool_response_length=4096 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.default_local_dir=./checkpoints/retro_qwen2.5-7b-instruct-1M_10_test/ppo_retro_test190_t0_chembl \
    trainer.logger=['console','wandb'] \
    trainer.project_name=retro_qwen2.5-7b-instruct-1M_10_test \
    trainer.experiment_name=ppo_retro_test190_t0_chembl \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    tool.debug=True \
    tool.api_max_concurrency=2 \
    tool.max_turns=100 \
    tool.topk=10 \
    tool.shuffle=False \
    tool.maxstep=30 \
    tool.force_noloop=True \
    tool.use_batch_tool_calls=False \
    tool.env='retro_noback_V2' \
    tool.use_api_model=True \
    tool.api_model_name='deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'" \
    2>&1 | tee "logs/test_chembl_${SLURM_JOB_ID}.log"
