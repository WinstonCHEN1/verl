# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch


def _to_teacher_token_ids(teacher_sequence: Any, tokenizer: Any) -> list[int]:
    """Convert teacher sequence into token ids.

    Supports:
    - list/tuple/np.ndarray of token ids
    - plain string (encoded by tokenizer)
    """
    if teacher_sequence is None:
        return []

    if isinstance(teacher_sequence, np.ndarray):
        teacher_sequence = teacher_sequence.tolist()

    if isinstance(teacher_sequence, (list, tuple)):
        if len(teacher_sequence) == 0:
            return []
        if all(isinstance(x, (int, np.integer)) for x in teacher_sequence):
            return [int(x) for x in teacher_sequence]
        teacher_sequence = " ".join(str(x) for x in teacher_sequence)

    if isinstance(teacher_sequence, str):
        token_ids = tokenizer.encode(teacher_sequence, add_special_tokens=False)
        return [int(x) for x in token_ids]

    return []


def compute_teacher_frequency_tensor(
    responses: torch.Tensor,
    reward_model_items: Any,
    tokenizer: Any,
    teacher_sequence_key: str,
) -> torch.Tensor:
    """Compute teacher-token frequency term freq_teacher(y_t) for each sampled token."""
    batch_size, response_len = responses.shape
    freq = torch.zeros((batch_size, response_len), dtype=torch.float32, device=responses.device)

    if reward_model_items is None:
        return freq

    for i in range(batch_size):
        item = reward_model_items[i]
        teacher_sequence = None
        if isinstance(item, dict):
            teacher_sequence = item.get(teacher_sequence_key)
            if teacher_sequence is None and teacher_sequence_key != "ground_truth":
                teacher_sequence = item.get("ground_truth")
        else:
            teacher_sequence = item

        teacher_token_ids = _to_teacher_token_ids(teacher_sequence, tokenizer)
        if len(teacher_token_ids) == 0:
            continue

        counts = Counter(teacher_token_ids)
        denom = float(len(teacher_token_ids))
        for t, token_id in enumerate(responses[i].tolist()):
            freq[i, t] = counts.get(int(token_id), 0) / denom

    return freq


def compute_teacher_step_proxy_reward(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor | None,
    reward_model_items: Any,
    tokenizer: Any,
    cfg: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute token-level proxy reward for teacher-step probability improvement."""
    response_mask = response_mask.to(dtype=torch.float32)
    pi_t = torch.exp(old_log_probs).to(dtype=torch.float32)

    teacher_freq = compute_teacher_frequency_tensor(
        responses=responses,
        reward_model_items=reward_model_items,
        tokenizer=tokenizer,
        teacher_sequence_key=cfg.teacher_sequence_key,
    )

    reward = cfg.freq_coef * teacher_freq - cfg.pi_coef * pi_t

    teacher_avg_prob_proxy = torch.zeros_like(teacher_freq)
    if getattr(cfg, "teacher_avg_prob_coef", 0.0) != 0.0:
        mode = getattr(cfg, "teacher_avg_prob_mode", "seq_freq_mean")
        if mode == "seq_freq_mean":
            valid_cnt = response_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            seq_mean = (teacher_freq * response_mask).sum(dim=-1, keepdim=True) / valid_cnt
            teacher_avg_prob_proxy = seq_mean.expand_as(teacher_freq)
        elif mode == "none":
            teacher_avg_prob_proxy = torch.zeros_like(teacher_freq)
        else:
            raise ValueError(f"Unknown teacher_avg_prob_mode: {mode}")
        reward = reward - float(cfg.teacher_avg_prob_coef) * teacher_avg_prob_proxy
    if cfg.sum_pi_squared_coef != 0.0:
        if sum_pi_squared is None:
            raise ValueError(
                "teacher_step_reward requires sum_pi_squared, please set "
                "actor_rollout_ref.actor.calculate_sum_pi_squared=True"
            )
        reward = reward + cfg.sum_pi_squared_coef * sum_pi_squared.to(dtype=torch.float32)

    reward = reward * response_mask

    if cfg.normalize_per_sequence:
        valid_cnt = response_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        seq_mean = reward.sum(dim=-1, keepdim=True) / valid_cnt
        centered = (reward - seq_mean) * response_mask
        seq_var = (centered.square().sum(dim=-1, keepdim=True) / valid_cnt).clamp_min(float(cfg.eps))
        reward = centered / torch.sqrt(seq_var) * response_mask

    metrics = {
        "teacher_step_reward/proxy_mean": (reward.sum() / response_mask.sum().clamp_min(1.0)).item(),
        "teacher_step_reward/teacher_freq_mean": (
            (teacher_freq * response_mask).sum() / response_mask.sum().clamp_min(1.0)
        ).item(),
        "teacher_step_reward/teacher_avg_prob_proxy_mean": (
            (teacher_avg_prob_proxy * response_mask).sum() / response_mask.sum().clamp_min(1.0)
        ).item(),
        "teacher_step_reward/pi_mean": ((pi_t * response_mask).sum() / response_mask.sum().clamp_min(1.0)).item(),
    }
    if sum_pi_squared is not None:
        sum_pi_squared_f = sum_pi_squared.to(dtype=torch.float32)
        metrics["teacher_step_reward/sum_pi_squared_mean"] = (
            (sum_pi_squared_f * response_mask).sum() / response_mask.sum().clamp_min(1.0)
        ).item()

    return reward, metrics
