#!/usr/bin/env bash
set -x

export WANDB_API_KEY="${WANDB_API_KEY:-wandb_...}"
clip_ratio_low=0.2
clip_ratio_high=0.27

val_temperature=0.7
val_top_p=0.9

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

TRAIN_FILE=${TRAIN_FILE:-/mnt/ali-sh-1/usr/lihaitao/chenguo/verl/datasets/train.parquet}

VAL_FILE_1=${VAL_FILE_1:-/mnt/ali-sh-1/usr/lihaitao/chenguo/verl/datasets/math500_validation.parquet}
VAL_FILE_2=${VAL_FILE_2:-/mnt/ali-sh-1/usr/lihaitao/chenguo/verl/datasets/mmlu_test_processed.parquet}

VAL_FILES="[${VAL_FILE_1},${VAL_FILE_2}]"
# VAL_FILE=${VAL_FILE:-/mnt/ali-sh-1/usr/lihaitao/chenguo/data/mmlu_grpo/validation.parquet}
CKPT_DIR=${CKPT_DIR:-/mnt/ali-sh-1/usr/lihaitao/chenguo/checkpoints/qwen2.5-outcome-rtg-warm}
MODEL_PATH=${MODEL_PATH:-/mnt/ali-sh-1/usr/lihaitao/model/Qwen/Qwen2.5-7B}
# CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-${REPO_ROOT}/examples/data_preprocess/mmlu_reward.py}
CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-${REPO_ROOT}/examples/data_preprocess/unified_reward.py}
TEACHER_SEQUENCE_KEY=${TEACHER_SEQUENCE_KEY:-teacher_sequence}

# 两阶段训练配置
# Phase 1: ENABLE_TEACHER_STEP_REWARD=false (只用outcome reward)
# Phase 2: ENABLE_TEACHER_STEP_REWARD=true (加入teacher step reward)
ENABLE_TEACHER_STEP_REWARD=${ENABLE_TEACHER_STEP_REWARD:-false}

if [ "$ENABLE_TEACHER_STEP_REWARD" = "true" ]; then
    echo "=== 启用第二阶段训练: Outcome + Teacher Step Reward ==="
    TEACHER_STEP_REWARD_ENABLE=True
    # 第二阶段可以用更小的学习率
    ACTOR_LR=${ACTOR_LR:-5e-7}
else
    echo "=== 启用第一阶段训练: 仅用 Outcome Reward ==="
    TEACHER_STEP_REWARD_ENABLE=False
    ACTOR_LR=${ACTOR_LR:-1e-6}
fi

echo "=== 分布式环境变量调试 ==="
echo "WORLD_SIZE=${WORLD_SIZE:-未设置}"
echo "LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE:-未设置}"
echo "RANK=${RANK:-未设置}"
echo "GPU_NUM=${GPU_NUM:-未设置}"
echo "RAY_JOB_ID=${RAY_JOB_ID:-未设置}"
echo "=========================="

if [ -n "${WORLD_SIZE:-}" ]; then
    if [ -n "${LOCAL_WORLD_SIZE:-}" ]; then
        NNODES=$((WORLD_SIZE / LOCAL_WORLD_SIZE))
        echo "检测到 torchrun 环境，计算 NNODES=$NNODES"
    else
        NNODES=$WORLD_SIZE
        echo "检测到 Ray 环境，NNODES=$NNODES"
    fi
else
    NNODES=1
    echo "未检测到分布式环境，NNODES=1"
fi

echo "最终使用 NNODES=$NNODES"
echo "=========================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files=${VAL_FILES} \
    trainer.default_local_dir="${CKPT_DIR}" \
    data.train_batch_size=768 \
    data.max_prompt_length=2048 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=192 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.calculate_sum_pi_squared=True \
    actor_rollout_ref.actor.sum_pi_squared_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16000 \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.teacher_step_reward.enable=${TEACHER_STEP_REWARD_ENABLE} \
    algorithm.teacher_step_reward.teacher_sequence_key="${TEACHER_SEQUENCE_KEY}" \
    algorithm.teacher_step_reward.mix_rm_coef=0.0 \
    algorithm.teacher_step_reward.format_reward_coef=1.0 \
    algorithm.teacher_step_reward.enable_after_steps=50 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_rtg_oucome' \
    trainer.experiment_name='7b-rtg-outcome' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.log_val_generations=20 \
    trainer.total_epochs=10 \
    custom_reward_function.path="${CUSTOM_REWARD_PATH}" \
    custom_reward_function.name=compute_score \
    "$@"
