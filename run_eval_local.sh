#!/usr/bin/env bash
# Run agent_r1 validation on the CURRENT compute node (no rjob).
# Mirrors run_eval_rjob.sh but removes cluster-submission assumptions
# and emits a self-contained, detailed log for later inspection.
#
# Conda env: Retro_R1
#
# Minimum required env (or just use the launch_eval_sft.sh wrapper):
#   MODEL_PATH   HF model dir to evaluate (merged SFT checkpoint, or base model)
#
# Common overrides:
#   DATASET              "retro" | "chembl"                         (default: retro)
#   EXPERIMENT_NAME      wandb / checkpoint tag                      (default: derived from MODEL_TAG)
#   EXP_TAG              extra suffix on EXPERIMENT_NAME             (default: empty)
#   VAL_FILES            path to validation parquet                  (default: per DATASET)
#   NNODES               number of nodes                             (default: 1)
#   TP_SIZE              tensor parallel size                        (default: min(2, NUM_GPUS))
#   MAX_TURNS TOPK MAXSTEP API_MAX_CONCURRENCY TOOL_ENV             (see defaults below)
#   CONDA_SH / CONDA_ENV                                             (default: miniconda3 + Retro_R1)
#   CUDA_HOME / CUDNN_PATH                                           (default: gpfs2 cuda 12.6)
#   LOG_DIR                                                          (default: ./logs)
#
# The framework still reads pjlab_APImodel_{key,url} at import time even
# though tool.use_api_model=False; they must be exported (dummy values work).

set -eo pipefail
set -u
# Report the line where we died — otherwise `set -e` + piped teeing hides it.
trap 'rc=$?; echo "[run_eval_local.sh] ERR rc=$rc at line $LINENO: ${BASH_COMMAND}" >&2' ERR

#======================================
# Locate script dir and cd into the repo
#======================================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#======================================
# Dataset selection
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

#======================================
# Model path & experiment naming
#======================================
: "${MODEL_PATH:?MODEL_PATH must be set (point at the merged SFT dir or base model)}"
export MODEL_PATH
MODEL_TAG="${MODEL_PATH##*/}"
# Avoid collisions when MODEL_PATH ends in /merged (common for our merge script).
if [[ "$MODEL_TAG" == "merged" ]]; then
    MODEL_TAG="$(basename "$(dirname "$MODEL_PATH")")_merged"
fi

EXP_TAG="${EXP_TAG:-}"
if [[ -n "$EXP_TAG" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval_${DATASET}_${MODEL_TAG}_${EXP_TAG}}"
else
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval_${DATASET}_${MODEL_TAG}}"
fi
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
[[ "$GPUS_PER_NODE" -ge 1 ]] || { echo "ERROR: no GPUs detected"; exit 1; }

# TP defaults to min(2, GPUS_PER_NODE); override via env var.
if [[ -z "${TP_SIZE:-}" ]]; then
    if [[ "$GPUS_PER_NODE" -ge 2 ]]; then TP_SIZE=2; else TP_SIZE=1; fi
fi

API_MODEL_NAME="${API_MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
API_LOG_TAG="${API_MODEL_NAME//\//_}"

# Job id for log/tmp naming
JOB_ID="${RJOB_JOB_ID:-${JOB_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}}"
export RAY_TMPDIR="/tmp/${USER:-$(id -un)}/ray/${JOB_ID}"
mkdir -p "$RAY_TMPDIR"

# Pjlab API creds — still required at import time even with use_api_model=False.
# If not already set in the shell, drop in dummies so eval can start.
: "${pjlab_APImodel_key:=__unused__}"
: "${pjlab_APImodel_url:=http://unused.local}"
export pjlab_APImodel_key pjlab_APImodel_url

#======================================
# Conda env activation
#======================================
CONDA_SH="${CONDA_SH:-/mnt/shared-storage-user/wangzifu/miniconda3/etc/profile.d/conda.sh}"
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
export CPATH="/usr/include:${CPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_HOME="${CUDA_HOME:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/soft/cuda/12.6}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDNN_PATH="${CUDNN_PATH:-/mnt/shared-storage-user/wangzifu/miniconda3/envs/Retro_R1/lib/python3.10/site-packages/nvidia/cudnn}"
export CPLUS_INCLUDE_PATH="${CUDNN_PATH}/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="${CUDNN_PATH}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDNN_PATH}/lib:${LD_LIBRARY_PATH:-}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"

#======================================
# Log setup (MUST exist before we tee into it)
#======================================
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/eval_local_${DATASET}_${MODEL_TAG}_val-${VAL_TAG}_job-${JOB_ID}.log"
ENV_LOG="${LOG_FILE%.log}.env.log"
# Hydra's default run.dir is outputs/<date>/<time>, and outputs/ in this repo
# is root-owned (leftover from a containerized rjob). Point hydra at our
# writable log tree instead.
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${LOG_DIR}/hydra/${JOB_ID}}"
mkdir -p "$HYDRA_RUN_DIR"

#======================================
# Tunable hyperparameters
#======================================
MAX_TURNS="${MAX_TURNS:-100}"
TOPK="${TOPK:-10}"
MAXSTEP="${MAXSTEP:-30}"
API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-50}"
TOOL_ENV="${TOOL_ENV:-retro_noback_V4}"

#======================================
# Detailed run-metadata dump (header + separate env snapshot)
#======================================
{
echo "========================================"
echo "  Agent-R1 eval  (direct launch)"
echo "========================================"
echo "JOB_ID:              $JOB_ID"
echo "DATE:                $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "HOSTNAME:            $(hostname)"
echo "PWD:                 $(pwd)"
echo "DATASET:             $DATASET"
echo "VAL_FILES:           $VAL_FILES"
echo "MODEL_PATH:          $MODEL_PATH"
echo "MODEL_TAG:           $MODEL_TAG"
echo "EXPERIMENT_NAME:     $EXPERIMENT_NAME"
echo "CHECKPOINT_DIR:      $CHECKPOINT_DIR"
echo "NNODES:              $NNODES"
echo "GPUS_PER_NODE:       $GPUS_PER_NODE"
echo "TP_SIZE:             $TP_SIZE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "MAX_TURNS:           $MAX_TURNS"
echo "TOPK:                $TOPK"
echo "MAXSTEP:             $MAXSTEP"
echo "API_MAX_CONCURRENCY: $API_MAX_CONCURRENCY"
echo "TOOL_ENV:            $TOOL_ENV"
echo "API_MODEL_NAME:      $API_MODEL_NAME   (use_api_model=False)"
echo "CONDA_ENV:           $CONDA_ENV"
echo "PYTHON:              $(which python3 2>/dev/null || echo '<none>')"
echo "PYTHON VER:          $(python3 --version 2>&1)"
echo "RAY_TMPDIR:          $RAY_TMPDIR"
echo "LOG_FILE:            $LOG_FILE"
echo "ENV_LOG:             $ENV_LOG"
echo "HYDRA_RUN_DIR:       $HYDRA_RUN_DIR"
echo "----------------------------------------"
echo "MODEL_PATH contents:"
if [[ -d "$MODEL_PATH" ]]; then
    ls -lh "$MODEL_PATH" | sed 's/^/  /'
else
    echo "  (not a directory: $MODEL_PATH)"; exit 1
fi
echo "----------------------------------------"
[[ -f "$MODEL_PATH/config.json" ]] || { echo "ERROR: $MODEL_PATH/config.json missing"; exit 1; }
[[ -f "$MODEL_PATH/tokenizer_config.json" ]] || { echo "WARN: $MODEL_PATH/tokenizer_config.json missing"; }
echo "GPU info:"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "(no nvidia-smi)"
echo "========================================"
} | tee "$LOG_FILE"

# Full-fat env snapshot — kept in a sidecar so the main log stays scannable.
# Disable -e/pipefail inside this block: `pip list | head` causes a harmless
# SIGPIPE on pip list that would otherwise kill the whole script.
(
    set +e
    set +o pipefail
    echo "### env ($(date -u '+%Y-%m-%dT%H:%M:%SZ'))"
    env | sort
    echo
    echo "### key package versions"
    python3 - 2>&1 <<'PY'
import importlib
for pkg in ("torch","transformers","vllm","ray","peft","flash_attn","deepspeed","verl","agent_r1","trl"):
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg:14s} {getattr(m,'__version__','?')}")
    except Exception as e:
        print(f"{pkg:14s} (not importable: {type(e).__name__}: {e})")
PY
    echo
    echo "### pip list (top 60)"
    pip list 2>/dev/null | head -n 60
) > "$ENV_LOG" 2>&1 || true

#======================================
# Hydra command
#======================================
HYDRA_CMD=(
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES
    PYTHONUNBUFFERED=1
    python3 -m agent_r1.src.main_agent_retro_noback
        data.train_files=./data/reaction_pathway_search/train_h4_10.parquet
        data.val_files="$VAL_FILES"
        data.train_batch_size=128
        data.max_prompt_length=32768
        data.max_response_length=32768
        data.max_start_length=1024
        data.max_tool_response_length=4096
        actor_rollout_ref.model.path="$MODEL_PATH"
        actor_rollout_ref.actor.optim.lr=1e-6
        actor_rollout_ref.model.use_remove_padding=True
        actor_rollout_ref.actor.ppo_mini_batch_size=64
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
        actor_rollout_ref.actor.use_dynamic_bsz=True
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
        actor_rollout_ref.model.enable_gradient_checkpointing=True
        actor_rollout_ref.actor.fsdp_config.param_offload=True
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
        actor_rollout_ref.actor.fsdp_config.fsdp_size=8
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
        actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE"
        actor_rollout_ref.rollout.max_num_batched_tokens=65536
        actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6
        actor_rollout_ref.rollout.val_kwargs.temperature=0.0
        actor_rollout_ref.rollout.val_kwargs.do_sample=False
        actor_rollout_ref.rollout.val_kwargs.n=1
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
        actor_rollout_ref.ref.fsdp_config.param_offload=True
        critic.optim.lr=1e-5
        critic.model.use_remove_padding=True
        critic.model.path="$MODEL_PATH"
        critic.model.enable_gradient_checkpointing=True
        critic.ppo_micro_batch_size_per_gpu=2
        critic.model.fsdp_config.param_offload=True
        critic.model.fsdp_config.optimizer_offload=True
        critic.model.fsdp_config.fsdp_size=8
        critic.ppo_max_token_len_per_gpu=65536
        algorithm.use_process_rewards=False
        algorithm.adv_estimator=grpo
        algorithm.kl_ctrl.kl_coef=0.001
        trainer.default_local_dir="$CHECKPOINT_DIR"
        trainer.critic_warmup=3
        "trainer.logger=[console]"
        trainer.project_name=retro_qwen2.5-7b-instruct-1M_10_test
        trainer.experiment_name="$EXPERIMENT_NAME"
        trainer.n_gpus_per_node="$GPUS_PER_NODE"
        trainer.nnodes="$NNODES"
        trainer.save_freq=10
        trainer.test_freq=10
        trainer.total_epochs=0
        trainer.val_before_train=True
        tool.api_max_concurrency="$API_MAX_CONCURRENCY"
        tool.debug=True
        tool.max_turns="$MAX_TURNS"
        tool.topk="$TOPK"
        tool.shuffle=False
        tool.maxstep="$MAXSTEP"
        tool.force_noloop=True
        tool.use_batch_tool_calls=False
        tool.env="$TOOL_ENV"
        tool.use_api_model=False
        tool.api_model_name="$API_MODEL_NAME"
        "hydra.run.dir=$HYDRA_RUN_DIR"
        "hydra.sweep.dir=$HYDRA_RUN_DIR"
)

{
echo "----------------------------------------"
echo "Command:"
printf '  %q ' "${HYDRA_CMD[@]}"
echo
echo "----------------------------------------"
echo "[$(date -u '+%H:%M:%SZ')] launching eval ..."
} | tee -a "$LOG_FILE"

START_TS=$(date +%s)
"${HYDRA_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
END_TS=$(date +%s)

(
    set +e
    set +o pipefail
    echo ""
    echo "========================================"
    echo "[$(date -u '+%H:%M:%SZ')] eval finished with exit code: $EXIT_CODE"
    echo "Elapsed:    $((END_TS - START_TS)) s"
    echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
    if [[ -d "$CHECKPOINT_DIR" ]]; then
        echo "Checkpoint dir contents (depth 2):"
        find "$CHECKPOINT_DIR" -maxdepth 2 -mindepth 1 -printf '  %p\n' 2>/dev/null | head -n 40
    fi
    echo "Last eval-prediction-like files, if any:"
    find "$CHECKPOINT_DIR" -maxdepth 4 -type f \( -name '*.jsonl' -o -name '*.json' -o -name 'val_*' \) 2>/dev/null | tail -n 5 | sed 's/^/  /'
    echo "========================================"
) | tee -a "$LOG_FILE"

exit "$EXIT_CODE"
