#!/bin/bash
#SBATCH --job-name=test_chembl_ppo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=ai_science

set -eo pipefail

#======================================
# Environment Variables
#======================================

NNODES="${SLURM_JOB_NUM_NODES:-1}"
GPUS_PER_NODE=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | awk -F',' '{print NF}')

export MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507   # Qwen/Qwen2.5-Math-7B; Qwen/Qwen3-1.7B; RoadQAQ/Qwen2.5-Math-1.5B-16k-think; microsoft/Phi-4-mini-instruct; Qwen/Qwen3-4B-Instruct-2507
MODEL_TAG="${MODEL_PATH##*/}"

export CPATH="/usr/include:${CPATH:-}"
export HF_HUB_OFFLINE=1
export RAY_TMPDIR="/tmp/$USER/ray/$SLURM_JOB_ID"
mkdir -p "$RAY_TMPDIR"

echo "=== Cluster Info ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "NNODES=$NNODES"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
nvidia-smi || true

#======================================
# Run test_chembl evaluation
#======================================
export_pth=./export_model/retro_qwen2.5-7b-instruct-1M/ppo_retro/global_step_190/actor

srun env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    bash -c "source ~/miniforge3/etc/profile.d/conda.sh && conda activate retro_r1 && \
    PYTHONUNBUFFERED=1 python3 -m agent_r1.src.main_agent_retro_noback \
    data.train_files=\"./data/reaction_pathway_search/train_h4_10.parquet\" \
    data.val_files=\"./data/reaction_pathway_search/validation_chembl_1000.parquet\" \
    data.train_batch_size=128 \
    data.max_prompt_length=32768 \
    data.max_response_length=32768 \
    data.max_start_length=1024 \
    data.max_tool_response_length=4096 \
    actor_rollout_ref.model.path=$MODEL_PATH \
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
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$MODEL_PATH \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.model.fsdp_config.fsdp_size=8 \
    critic.ppo_max_token_len_per_gpu=65536 \
    algorithm.use_process_rewards=False \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.default_local_dir=./checkpoints/retro_qwen2.5-7b-instruct-1M_10_test/ppo_retro_test190_t0_chembl \
    trainer.critic_warmup=3 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=retro_qwen2.5-7b-instruct-1M_10_test \
    trainer.experiment_name=ppo_retro_test190_t0_chembl \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=0 \
    trainer.val_before_train=True \
    tool.max_turns=100 \
    tool.topk=10 \
    tool.shuffle=False \
    tool.maxstep=30 \
    tool.force_noloop=True \
    tool.use_batch_tool_calls=False \
    tool.env='retro_noback_V2'" \
    2>&1 | tee "logs/test_chembl_${SLURM_JOB_ID}.log"
