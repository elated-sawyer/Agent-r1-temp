#!/usr/bin/env bash
# Thin wrapper: run `run_eval_local.sh` against the most recent merged SFT
# checkpoint produced by merge_retro_sft_lora.sh.
#
# Examples:
#   ./launch_eval_sft.sh                                       # retro, auto-TP
#   CUDA_VISIBLE_DEVICES=0,1 ./launch_eval_sft.sh              # pin 2 GPUs
#   DATASET=chembl ./launch_eval_sft.sh                        # chembl eval
#   MODEL_PATH=/path/to/specific/merged ./launch_eval_sft.sh   # override model
#   nohup ./launch_eval_sft.sh > logs/bootstrap.eval.log 2>&1 &
#
# If auto-detection fails, set MODEL_PATH directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_MODEL_TAG="${BASE_MODEL_TAG:-0401-preview}"
SFT_MERGED_ROOT="${SFT_MERGED_ROOT:-/mnt/shared-storage-gpfs2/wangzifugpfs2/LlamaFactory_saves/retro_sft/full/${BASE_MODEL_TAG}/lora}"

if [[ -z "${MODEL_PATH:-}" ]]; then
    # Pick the newest run that has a populated merged/ subdir.
    CANDIDATE=""
    while IFS= read -r d; do
        if [[ -f "$d/merged/config.json" ]]; then
            CANDIDATE="$d/merged"
            break
        fi
    done < <(ls -td "$SFT_MERGED_ROOT"/*/ 2>/dev/null | sed 's:/*$::')

    if [[ -z "$CANDIDATE" ]]; then
        echo "ERROR: no merged SFT checkpoint found under" >&2
        echo "       $SFT_MERGED_ROOT/<run-id>/merged/" >&2
        echo "       Run LlamaFactory/merge_retro_sft_lora.sh first, or pass MODEL_PATH=..." >&2
        exit 1
    fi
    MODEL_PATH="$CANDIDATE"
fi
export MODEL_PATH

# Tag the experiment with the SFT run id so checkpoints/logs are distinguishable.
SFT_RUN_ID="$(basename "$(dirname "$MODEL_PATH")")"
export EXP_TAG="${EXP_TAG:-sft_${SFT_RUN_ID}}"
export DATASET="${DATASET:-retro}"

exec bash "$SCRIPT_DIR/run_eval_local.sh"
