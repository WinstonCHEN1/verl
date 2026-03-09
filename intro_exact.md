# Intro

## 总体训练链路
1. rollout 生成每个 prompt 的 `n` 条回答，并给同组样本同一个 `uid`。  
2. `compute_log_prob` 回算 `old_log_probs`，同时产出 `sum_pi_squared`，在 `exact` 模式下还会产出 `teacher_avg_prob`。  
3. 进入 teacher step reward：得到 token 级 `reward_tensor`。  
4. `adv_estimator=rloo` 下，按 `uid` 做 token 级 leave-one-out baseline，得到 `advantages`。  
5. actor 用 PPO loss 更新。  

## exact 模式的核心点**
在 [teacher_step_reward.py] 里，reward 变成：

\[
r_t =
\text{freq\_coef}\cdot f_{\text{teacher}}(y_t)
-\text{pi\_coef}\cdot \pi(y_t|s_t)
-\text{teacher\_avg\_prob\_coef}\cdot \bar{\pi}_{\text{teacher}}(s_t)
+\text{sum\_pi\_squared\_coef}\cdot \sum_v \pi(v|s_t)^2
\]

其中 `exact` 模式下：
\[
\bar{\pi}_{\text{teacher}}(s_t)=\sum_{v\in V_{teacher}} w_v\,\pi(v|s_t)
\]
`w_v` 是 teacher token 的频率权重（词频/teacher长度）。

## exact 实现
1. 构建 teacher token 权重对 `(token_ids, token_weights)`：  
   [_build_teacher_token_weight_pairs in fsdp_workers.py]
2. 在 `compute_log_prob` 前把它塞进 batch 的 `non_tensor_batch["teacher_token_weight_pairs"]`：  
   [compute_log_prob in fsdp_workers.py]
3. 在 actor 前向里，从 logits 直接算 `teacher_avg_prob`（逐 token）：  
   [DPActor._forward_micro_batch]
4. 回传 `teacher_avg_prob` 到 trainer，再传给 `compute_teacher_step_proxy_reward`：  
   [_compute_teacher_step_reward in ray_trainer.py]

## RLOO 部分（与proxy版本一致）
在 [core_algos.py]：
1. 按 `uid` 分组。  
2. 每个 token 位置做严格 mask-aware 的 LOO baseline：`baseline_{i,t}=mean_{j!=i,valid}(r_{j,t})`。  
3. `adv_{i,t}=(r_{i,t}-baseline_{i,t})*mask_{i,t}`，无可用 peer 时置 0。  
4. `returns=advantages`。  