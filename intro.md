# Intro

这套“`teacher step reward + RLOO`”当前实现可以按一条主链理解：

## 生成与分组  
在训练 step 里，先给每个原始样本分配一个 `uid`，再按 `rollout.n` 重复生成多条回答；同一题的 `n` 条回答共享同一个 `uid`，后面 RLOO 就按这个分组做 LOO baseline。实现见 [ray_trainer.py]。

## teacher step reward 的 token 级构造  
核心在 [teacher_step_reward.py]：
- 先从 `reward_model` 字段取 `teacher_sequence_key`。
- 把 teacher 序列转成 token id，统计词频 `freq_teacher(y_t)`。
- 对每个响应 token 计算：
  `reward_t = freq_coef * freq_teacher(y_t) - pi_coef * pi_t`，其中 `pi_t = exp(old_log_probs_t)`。
  减去一个序列均值 proxy（`seq_freq_mean`），即每个 token 都减同一个该序列平均 teacher_freq。再加 `sum_pi_squared_coef * Σ_v π(v|s_t)^2`（由 actor 侧 `calculate_sum_pi_squared=True` 提供）。对齐公式
- 最后乘 `response_mask`。

## RLOO 具体怎么算  
在 [core_algos.py] 的 `compute_rloo_outcome_advantage`：
- 按 `uid` 分组。
- 对组内每个 token 位置 `t`，做严格 mask-aware 的 leave-one-out baseline：  
  `baseline_{i,t} = mean_{j!=i, valid}(reward_{j,t})`。  
  若该位置没有其他有效样本，adv 置 0。
- `adv_{i,t} = (reward_{i,t} - baseline_{i,t}) * mask_{i,t}`。
- `returns = advantages`。

## 总结  
`teacher token-level proxy reward -> 直接作为 token_level_rewards -> 按 uid 做 token 级 RLOO LOO 差分 -> 更新 actor`。