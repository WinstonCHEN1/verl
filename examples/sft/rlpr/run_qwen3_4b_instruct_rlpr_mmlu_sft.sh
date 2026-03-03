#!/usr/bin/env bash
set -xeuo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen3_4b_instruct_rlpr_mmlu_sft.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2
shift 2

TRAIN_FILE=${TRAIN_FILE:-$HOME/data/rlpr_mmlu_sft/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/rlpr_mmlu_sft/validation.parquet}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Instruct}
PROJECT_NAME=${PROJECT_NAME:-rlpr-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-rlpr-sft-qwen3-4b-instruct-mmlu-val}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-2}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-8192}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TEST_FREQ=${TEST_FREQ:-after_each_epoch}
SAVE_FREQ=${SAVE_FREQ:-after_each_epoch}
LR=${LR:-1e-5}

torchrun --standalone --nnodes=1 --nproc_per_node="${nproc_per_node}" \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.messages_key=messages \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    data.use_dynamic_bsz=True \
    data.pad_mode=no_padding \
    optim.lr="${LR}" \
    engine=fsdp \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=True \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger=console \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    "$@"
