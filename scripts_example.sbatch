#!/bin/bash
#SBATCH --job-name=ours_mem_grpo
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
# Experiment: OUR METHOD — Memory-Assisted GRPO
#
# This script runs GRPO training WITH the in-context memory system.
# Training proceeds in two phases:
#   Phase 1 (steps 1–250): Vanilla GRPO warm-up with binary reward.
#   Phase 2 (steps 251+):  Memory-assisted GRPO with full pipeline:
#     - Memory retrieval   (use_mem_N=1)
#     - Skill generation   (enable_skill_gen=True)
#     - Memory pruning     (enable_memory_prune=true)
#     - Memory in validation (use_memory_in_validation=True)
#     - Mixed rollouts     (mixed_ratio=0.5: 50% with hints, 50% vanilla)
#     - Hierarchical reward switching handled internally
#
# Compare against: baseline_Grpo_1node_deepscaler.sbatch
#======================================

#======================================
# Environment Variables
#======================================

NNODES="${SLURM_JOB_NUM_NODES:-1}"
GPUS_PER_NODE=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | awk -F',' '{print NF}')

export MODEL_PATH=Qwen/Qwen2.5-Math-7B   # Qwen/Qwen2.5-Math-7B; Qwen/Qwen3-1.7B; RoadQAQ/Qwen2.5-Math-1.5B-16k-think; microsoft/Phi-4-mini-instruct; Qwen/Qwen3-4B-Instruct-2507
export DATA_DIR=$HOME/workspace/pjlab-mem/dataset

# Key trainer flags (single source of truth for EXP_NAME + Hydra overrides below)
TRAIN_FILES="$DATA_DIR/train_data/openr1_no_system.parquet" # $DATA_DIR/train_data/openr1_no_system.parquet; $DATA_DIR/deepscaler/train.parquet
VAL_FILES="$DATA_DIR/valid_no_system.parquet" # $DATA_DIR/valid_no_system.parquet; $DATA_DIR/deepscaler/aime24.parquet
DATASET_TAG="$(basename "$(dirname "$TRAIN_FILES")")_$(basename "$TRAIN_FILES" .parquet)__val_$(basename "$VAL_FILES" .parquet)"
MODEL_TAG="${MODEL_PATH##*/}"
SKILL_RETRIEVAL_EPSILON=0.2
MIXED_RATIO=0.5 # 0.0: pure vanilla, 1.0: pure memory-assisted
# (deprecated) WARMUP_MODE=self_extract # vanilla, frozen_memory, self_extract
# (deprecated) MEMORY_WARMUP_STEPS=50 # 0 = no warm-up; set to e.g. 250 for vanilla GRPO before memory
AVERAGE_REWARD_BAR=0.3  # Per-query average reward threshold for reflexion loop
LOOP_LIMIT=3
USE_MEM_N=1 # 0 = no memory; set to e.g. 1 for vanilla GRPO before memory
MEMORY_UPDATE_POLICY=contrastive_reward_ema # binary_reward_ema, contrastive_reward_ema
MEMORY_PRUNE_POLICY=lowest_q_value # bad_count, negative_count, lowest_q_value
SKILL_SAVE_LIMIT_RATIO=0.5
SKILL_CONFIDENCE_MULTIPLIER=1.5
SKILL_GEN_STRATEGY=contrast_failure_analysis # success_trajectory, contrast_failure_analysis, hybrid
RESPONSE_DIRECTIVE="Please reason step by step, and put your final answer within \\boxed{}."
SKILL_GEN_BACKEND="api" # policy, api
SKILL_GEN_API_MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" # deepseek-ai/DeepSeek-R1-0528-Qwen3-8B, deepseek-ai/DeepSeek-R1-0528-Qwen3-4B, deepseek-ai/DeepSeek-R1-0528-Qwen3-2B
SKILL_GEN_API_TEMPERATURE=0.3
# Short, wandb/local-path-friendly suffix: adjust fields here if you want more in the name
export EXP_NAME="ours_grpo_${MODEL_TAG}__dataset_${DATASET_TAG}__eps${SKILL_RETRIEVAL_EPSILON}__mix${MIXED_RATIO}__memN${USE_MEM_N}__memup${MEMORY_UPDATE_POLICY}__mempr${MEMORY_PRUNE_POLICY}__sklsg${SKILL_GEN_STRATEGY}__sklc${SKILL_CONFIDENCE_MULTIPLIER}__sklbg${SKILL_GEN_BACKEND}__sklgm${SKILL_GEN_API_MODEL}__avg${AVERAGE_REWARD_BAR}__loop${LOOP_LIMIT}__jid${SLURM_JOB_ID:-na}"

export WANDB_PROJECT="mem2param"
# Increase this if math_verify still times out on your cluster/runtime.
export MATH_VERIFY_TIMEOUT_SECONDS=10

export CPATH="/usr/include:${CPATH:-}"
export HF_HUB_OFFLINE=1
export RAY_TMPDIR="/tmp/$USER/ray/$SLURM_JOB_ID"
mkdir -p "$RAY_TMPDIR"

echo "=== Cluster Info ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "EXP_NAME=$EXP_NAME"
echo "NNODES=$NNODES"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
nvidia-smi || true

#======================================
# Run Our Method — Memory-Assisted GRPO
#======================================
srun env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    bash -c "source ~/miniforge3/etc/profile.d/conda.sh && conda activate pjlab_verl && \
    PYTHONUNBUFFERED=1 python3 -m verl.relift.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.kl_ctrl.kl_coef=0.000 \
    data.return_raw_chat=True \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.truncation='left' \
    data.train_batch_size=64 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.max_target_length=8192 \
    data.shuffle=True \
    data.response_directive=\"'$RESPONSE_DIRECTIVE'\" \
    reward_model.reward_manager=\"naive\" \
    custom_reward_function.path=\"./utils_relift_reward/reward_utils/my_reward_func.py\" \
    custom_reward_function.name=\"my_reward_func\" \
    custom_reward_function.overlong_buffer.enable=False \
    custom_reward_function.overlong_buffer.len=4096 \
    custom_reward_function.overlong_buffer.penalty_factor=1.0 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.grad_clip=0.5 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=14336 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16000 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=131072 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.skill_gen_backend=${SKILL_GEN_BACKEND} \
    trainer.skill_gen_api_model=${SKILL_GEN_API_MODEL} \
    trainer.skill_gen_strategy=${SKILL_GEN_STRATEGY} \
    trainer.memory_update_policy=${MEMORY_UPDATE_POLICY} \
    trainer.memory_prune_policy=${MEMORY_PRUNE_POLICY} \
    trainer.skill_save_limit_ratio=${SKILL_SAVE_LIMIT_RATIO} \
    trainer.skill_confidence_multiplier=${SKILL_CONFIDENCE_MULTIPLIER} \
    trainer.average_reward_bar=${AVERAGE_REWARD_BAR} \
    trainer.loop_limit=${LOOP_LIMIT} \
    trainer.cache_pool_size=20 \
    trainer.debug_inspect_n=2 \
    trainer.skill_hint_placement=prepend \
    trainer.skip_param_update=False \
    trainer.rollout_data_dir=$HOME/workspace/pjlab-mem/experience_data/rollout_output \
    trainer.use_mem_N=${USE_MEM_N} \
    trainer.skill_retrieval_epsilon=${SKILL_RETRIEVAL_EPSILON} \
    trainer.mixed_ratio=${MIXED_RATIO} \
    trainer.debug_batch_inspect=True \
    trainer.q_update_alpha=0.1 \
    trainer.q_update_gamma=0.0 \
    trainer.enable_memory_prune=True \
    trainer.memory_prune_positive_rate=0.1 \
    trainer.memory_prune_delta=0.05 \
    trainer.memory_prune_bad_count_limit=5 \
    trainer.memory_limit=50 \
    trainer.strip_memory_for_update=False \
    trainer.enable_skill_gen=True \
    trainer.use_memory_in_validation=True \
    trainer.val_memory_q_min=0.15 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=\"$WANDB_PROJECT\" \
    trainer.experiment_name=\"$EXP_NAME\" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=./train_results/\${WANDB_PROJECT}/\${EXP_NAME} \
    trainer.total_epochs=5" \
    2>&1 | tee "logs/ours_grpo_${SLURM_JOB_ID}.log"
