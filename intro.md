# Intro

这套“`teacher step reward + GRPO`”当前实现可以按一条主链理解：

## 生成与分组  
在训练 step 里，先给每个原始样本分配一个 `uid`，再按 `rollout.n` 重复生成多条回答；同一题的 `n` 条回答共享同一个 `uid`，后面 GRPO 就按这个分组做 group baseline。实现见 [ray_trainer.py]。

## teacher step reward 的 token 级构造  
核心在 [teacher_step_reward.py]：
- 先从 `reward_model` 字段取 `teacher_sequence_key`。
- 把 teacher 序列转成 token id，统计词频 `freq_teacher(y_t)`。
- 对每个响应 token 计算：
  `reward_t = freq_coef * freq_teacher(y_t) - pi_coef * pi_t`，其中 `pi_t = exp(old_log_probs_t)`。
  减去一个序列均值 proxy（`seq_freq_mean`），即每个 token 都减同一个该序列平均 teacher_freq。再加 `sum_pi_squared_coef * Σ_v π(v|s_t)^2`（由 actor 侧 `calculate_sum_pi_squared=True` 提供）。对齐公式
- 最后乘 `response_mask`。

## GRPO 具体怎么算  
在 [core_algos.py] 的 `compute_grpo_outcome_advantage`：
- 按 `uid` 分组。
- 先对每条 rollout 的 token reward 计算 token-level return-to-go：  
  `G_{i,t} = Σ_{t'≥t} r_{i,t'}`。
- 用组内每条 rollout 的总 return `G_{i,0}` 计算 baseline 和 std：  
  `b_g = mean_j(G_{j,0})`，`σ_g = std_j(G_{j,0})`。
- 对每个 token 位置算：  
  `adv_{i,t} = (G_{i,t} - b_g) / (σ_g + eps)`。
- 若组内某个位置只有一条 rollout 在该位置有效，则该位置 adv 直接置 0。
- `returns = G`。

## 总结  
`teacher token-level proxy reward -> 直接作为 token_level_rewards -> 按 uid 做 token-level RTG GRPO 标准化 -> 更新 actor`。
