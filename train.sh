#!/usr/bin/env bash
set -x

export WANDB_API_KEY="${WANDB_API_KEY:-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}"

clip_ratio_low=0.2
clip_ratio_high=0.28

val_temperature=0.7
val_top_p=0.9

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)


TRAIN_FILE=${TRAIN_FILE:-/mnt/ali-sh-1/usr/lihaitao/chenguo/data/mmlu_grpo/235_train.parquet}
VAL_FILE=${VAL_FILE:-/mnt/ali-sh-1/usr/lihaitao/chenguo/data/mmlu_grpo/validation.parquet}
CKPT_DIR=${CKPT_DIR:-/mnt/ali-sh-1/usr/lihaitao/chenguo/checkpoints/mmlu_rloo_qwen3_4b_thinking_morestep}
MODEL_PATH=${MODEL_PATH:-/mnt/ali-sh-1/usr/lihaitao/model/Qwen3/Qwen3-4B-Thinking-2507}
CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-${REPO_ROOT}/examples/data_preprocess/mmlu_reward.py}
TEACHER_SEQUENCE_KEY=${TEACHER_SEQUENCE_KEY:-teacher_sequence}

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
    algorithm.adv_estimator=rloo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1000 \
    data.max_response_length=15000 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.calculate_sum_pi_squared=True \
    actor_rollout_ref.actor.sum_pi_squared_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
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
    algorithm.teacher_step_reward.enable=True \
    algorithm.teacher_step_reward.teacher_sequence_key="${TEACHER_SEQUENCE_KEY}" \
    algorithm.teacher_step_reward.mix_rm_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_rloo' \
    trainer.experiment_name='rloo_mmlu_qwen3_4b_thinking_morestep' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.log_val_generations=1 \
    trainer.total_epochs=100 \
    custom_reward_function.path="${CUSTOM_REWARD_PATH}" \
    custom_reward_function.name=compute_score \
    "$@"
