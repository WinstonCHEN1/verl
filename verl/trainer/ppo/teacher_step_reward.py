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
import math
import re
from typing import Any

import numpy as np
import torch


FORMAT_PATTERN = re.compile(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL)


def _to_teacher_token_ids(teacher_sequence: Any, tokenizer: Any) -> list[int]:
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


def _has_single_ordered_think_answer_format(text: str) -> bool:
    if not isinstance(text, str):
        return False

    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return False
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return False

    return FORMAT_PATTERN.match(text) is not None


def _find_think_boundaries(token_ids: list[int], tokenizer: Any) -> tuple[int, int]:
    """Find the start and end positions of <think>...</think> in token sequence.

    Returns:
        (think_start, think_end): inclusive start, exclusive end indices
        Returns (-1, -1) if not found
    """
    # Get token ids for think tags
    think_start_tokens = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_tokens = tokenizer.encode("</think>", add_special_tokens=False)

    n = len(token_ids)
    start_len = len(think_start_tokens)
    end_len = len(think_end_tokens)

    think_start = -1
    think_end = -1

    # Find <think>
    for i in range(n - start_len + 1):
        if token_ids[i:i + start_len] == think_start_tokens:
            think_start = i + start_len  # Position after <think>
            break

    if think_start == -1:
        return (-1, -1)

    # Find </think> after think_start
    for i in range(think_start, n - end_len + 1):
        if token_ids[i:i + end_len] == think_end_tokens:
            think_end = i  # Position at </think>
            break

    return (think_start, think_end)


def compute_format_reward_tensor(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer: Any,
    reward_coef: float,
) -> torch.Tensor:
    format_reward = torch.zeros_like(response_mask, dtype=torch.float32)
    if reward_coef == 0.0:
        return format_reward

    response_mask_f = response_mask.to(dtype=torch.float32)
    response_lengths = response_mask_f.sum(dim=-1).to(dtype=torch.long)

    # Get token ids for format tags
    think_start_tokens = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    answer_start_tokens = tokenizer.encode("<answer>", add_special_tokens=False)
    answer_end_tokens = tokenizer.encode("</answer>", add_special_tokens=False)

    for i, seq_len in enumerate(response_lengths.tolist()):
        if seq_len <= 0:
            continue

        response_text = tokenizer.decode(responses[i, :seq_len].tolist(), skip_special_tokens=True)
        if not _has_single_ordered_think_answer_format(response_text):
            continue

        # Only reward the format tag tokens, not all tokens
        # This prevents model from learning to generate short responses to "lock in" format reward
        token_ids = responses[i, :seq_len].tolist()

        def find_tag_positions(tag_tokens):
            positions = []
            tag_len = len(tag_tokens)
            for pos in range(seq_len - tag_len + 1):
                if token_ids[pos:pos + tag_len] == tag_tokens:
                    positions.extend(range(pos, pos + tag_len))
            return positions

        # Reward format tag tokens
        tag_positions = []
        tag_positions.extend(find_tag_positions(think_start_tokens))
        tag_positions.extend(find_tag_positions(think_end_tokens))
        tag_positions.extend(find_tag_positions(answer_start_tokens))
        tag_positions.extend(find_tag_positions(answer_end_tokens))

        for pos in set(tag_positions):
            if pos < seq_len:
                format_reward[i, pos] = float(reward_coef)

        # Also reward think content tokens to encourage detailed reasoning
        # Find think content (between <think> and </think>)
        think_start_pos = -1
        think_end_pos = -1
        for pos in range(seq_len - len(think_start_tokens) + 1):
            if token_ids[pos:pos + len(think_start_tokens)] == think_start_tokens:
                think_start_pos = pos + len(think_start_tokens)
                break
        if think_start_pos >= 0:
            for pos in range(think_start_pos, seq_len - len(think_end_tokens) + 1):
                if token_ids[pos:pos + len(think_end_tokens)] == think_end_tokens:
                    think_end_pos = pos
                    break
        # Reward think content tokens with smaller coefficient
        if think_start_pos >= 0 and think_end_pos > think_start_pos:
            think_content_coef = float(reward_coef) * 0.1  # 10% of format reward for content
            for pos in range(think_start_pos, think_end_pos):
                format_reward[i, pos] = think_content_coef

    return format_reward


def compute_teacher_step_reward_tensor(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    reward_model_items: Any,
    tokenizer: Any,
    teacher_sequence_key: str,
    position_coef: float = 1.0,
    repeat_penalty_coef: float = 0.1,
    length_coef: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute teacher step reward with think-only imitation + length reward (SEQUENCE-LEVEL for clean).

    For clean version: sequence-level reward, all tokens in think share the same reward.

    Args:
        responses: Student response token ids [batch_size, response_len]
        response_mask: Response mask [batch_size, response_len]
        reward_model_items: Items containing teacher sequences
        tokenizer: Tokenizer
        teacher_sequence_key: Key to get teacher sequence
        position_coef: Coefficient for position match within think
        repeat_penalty_coef: Coefficient for repeat penalty within think
        length_coef: Coefficient for length matching reward

    Returns:
        (imitation_reward, length_reward): Two reward tensors
    """
    batch_size, response_len = responses.shape
    device = responses.device
    imitation_reward = torch.zeros((batch_size, response_len), dtype=torch.float32, device=device)
    length_reward = torch.zeros((batch_size, response_len), dtype=torch.float32, device=device)

    if reward_model_items is None:
        return (imitation_reward, length_reward)

    response_lengths = response_mask.sum(dim=-1).to(dtype=torch.long).tolist()

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
        teacher_len = len(teacher_token_ids)
        if teacher_len == 0:
            continue

        # Get student response tokens (actual length)
        seq_len = response_lengths[i]
        if seq_len <= 0:
            continue
        student_token_ids = responses[i, :seq_len].tolist()

        # Find think boundaries in student response
        think_start, think_end = _find_think_boundaries(student_token_ids, tokenizer)
        if think_start == -1 or think_end == -1 or think_start >= think_end:
            # No valid think section found
            continue

        think_len = think_end - think_start

        # Count teacher token frequencies (for repeat penalty)
        teacher_counts = Counter(teacher_token_ids)
        generated_counts: dict[int, int] = {}

        # ========== SEQUENCE-LEVEL: accumulate rewards across think section ==========
        total_imitation_reward = 0.0

        for t in range(think_start, think_end):
            token_id = int(student_token_ids[t])

            # Position-aware reward: match at position t (relative to teacher)
            relative_pos = (t - think_start) / max(think_len, 1)
            teacher_pos = int(relative_pos * teacher_len)

            if teacher_pos < teacher_len and token_id == teacher_token_ids[teacher_pos]:
                total_imitation_reward += position_coef / teacher_len

            # Repeat penalty (based on teacher frequency)
            generated_counts[token_id] = generated_counts.get(token_id, 0) + 1
            teacher_count = teacher_counts.get(token_id, 0)
            excess = max(0, generated_counts[token_id] - teacher_count)
            total_imitation_reward -= repeat_penalty_coef * excess

        # Distribute accumulated reward evenly across think section
        if think_len > 0:
            imitation_reward[i, think_start:think_end] = total_imitation_reward / think_len

        # ========== Length reward (sequence-level, distributed across think) ==========
        # Gaussian reward based on how close think_len is to teacher_len
        # This encourages model to match teacher's reasoning length without hard thresholds
        length_diff_ratio = abs(think_len - teacher_len) / max(teacher_len, 1)
        length_match_score = length_coef * math.exp(-length_diff_ratio ** 2)

        if think_len > 0:
            length_reward[i, think_start:think_end] = length_match_score / think_len

        # Small bonus for non-empty think to prevent <think></think> empty pattern
        # This is NOT a threshold - it's a continuous bonus that scales with think_len
        # Capped at 0.1 to avoid dominating the teacher-length-based reward
        if think_len > 0:
            non_empty_bonus = 0.1 * min(think_len / max(teacher_len, 10), 1.0)
            length_reward[i, think_start:think_end] += non_empty_bonus / think_len

    return (imitation_reward, length_reward)


def compute_teacher_frequency_tensor(
    responses: torch.Tensor,
    reward_model_items: Any,
    tokenizer: Any,
    teacher_sequence_key: str,
) -> torch.Tensor:
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
        rewarded_student_counts: Counter[int] = Counter()
        for t, token_id in enumerate(responses[i].tolist()):
            token_id = int(token_id)
            if rewarded_student_counts[token_id] >= counts.get(token_id, 0):
                continue
            rewarded_student_counts[token_id] += 1
            freq[i, t] = 1.0 / denom

    return freq


def compute_global_repeat_penalty_tensor(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer: Any,
    repeat_penalty_coef: float = 0.1,
    min_repeat_length: int = 3,
) -> torch.Tensor:
    """Compute global repeat penalty tensor - penalizes repeated n-grams globally.

    This penalty applies regardless of format validity, providing a global
    constraint against repetitive generation.

    Args:
        responses: Token ids [batch_size, response_len]
        response_mask: Mask [batch_size, response_len]
        tokenizer: Tokenizer
        repeat_penalty_coef: Penalty coefficient for each repeated token
        min_repeat_length: Minimum length of repeated sequence to trigger penalty

    Returns:
        Penalty tensor with negative values for repeated tokens
    """
    batch_size, response_len = responses.shape
    device = responses.device
    penalty = torch.zeros((batch_size, response_len), dtype=torch.float32, device=device)

    if repeat_penalty_coef <= 0.0:
        return penalty

    response_lengths = response_mask.sum(dim=-1).to(dtype=torch.long).tolist()

    for i in range(batch_size):
        seq_len = response_lengths[i]
        if seq_len <= min_repeat_length * 2:
            continue

        token_ids = responses[i, :seq_len].tolist()

        # Track positions of each token
        token_positions: dict[int, list[int]] = {}
        for pos, token_id in enumerate(token_ids):
            if token_id not in token_positions:
                token_positions[token_id] = []
            token_positions[token_id].append(pos)

        # Penalize tokens that appear too frequently
        for token_id, positions in token_positions.items():
            if len(positions) >= min_repeat_length:
                # Count excess repetitions beyond the first occurrence
                excess_count = len(positions) - 1
                for pos in positions[1:]:  # Skip first occurrence
                    penalty[i, pos] -= repeat_penalty_coef * excess_count

    return penalty


def compute_teacher_step_proxy_reward(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor | None,
    reward_model_items: Any,
    tokenizer: Any,
    cfg: Any,
    teacher_avg_prob_exact: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute teacher step proxy reward using sequence-level imitation + length reward.

    Uses compute_teacher_step_reward_tensor (from clean repo) for sequence-level rewards.
    """
    response_mask = response_mask.to(dtype=torch.float32)
    pi_t = torch.exp(old_log_probs).to(dtype=torch.float32)

    # Use new sequence-level teacher step reward (from clean repo)
    imitation_reward, length_reward = compute_teacher_step_reward_tensor(
        responses=responses,
        response_mask=response_mask,
        reward_model_items=reward_model_items,
        tokenizer=tokenizer,
        teacher_sequence_key=cfg.teacher_sequence_key,
        position_coef=float(getattr(cfg, "position_coef", 1.0)),
        repeat_penalty_coef=float(getattr(cfg, "repeat_penalty_coef", 0.1)),
        length_coef=float(getattr(cfg, "length_coef", 0.1)),
    )

    # Combine rewards: imitation + length - pi_coef * pi_t
    teacher_step_reward = imitation_reward + length_reward
    reward = teacher_step_reward - cfg.pi_coef * pi_t

    teacher_avg_prob_proxy = torch.zeros_like(teacher_step_reward)
    if getattr(cfg, "teacher_avg_prob_coef", 0.0) != 0.0:
        mode = getattr(cfg, "teacher_avg_prob_mode", "seq_freq_mean")
        if mode == "exact":
            if teacher_avg_prob_exact is None:
                raise ValueError(
                    "teacher_step_reward teacher_avg_prob_mode=exact requires teacher_avg_prob tensor from actor "
                    "compute_log_prob path."
                )
            teacher_avg_prob_proxy = teacher_avg_prob_exact.to(dtype=torch.float32)
        elif mode == "seq_freq_mean":
            valid_cnt = response_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            seq_mean = (teacher_step_reward * response_mask).sum(dim=-1, keepdim=True) / valid_cnt
            teacher_avg_prob_proxy = seq_mean.expand_as(teacher_step_reward)
        elif mode == "none":
            teacher_avg_prob_proxy = torch.zeros_like(teacher_step_reward)
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

    # Check format validity - apply penalty for invalid format responses
    batch_size = responses.shape[0]
    format_valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=responses.device)
    response_lengths = response_mask.sum(dim=-1).to(dtype=torch.long)
    for i in range(batch_size):
        seq_len = response_lengths[i].item()
        if seq_len > 0:
            response_text = tokenizer.decode(responses[i, :seq_len].tolist(), skip_special_tokens=True)
            if _has_single_ordered_think_answer_format(response_text):
                format_valid_mask[i] = True

    # Apply format penalty: invalid format responses get zero reward
    format_valid_mask_expanded = format_valid_mask.unsqueeze(-1).expand_as(reward)
    # 0.0 = keep 0% reward for invalid format, 100% for valid format
    format_penalty = 0.0
    reward = reward * (format_penalty + (1 - format_penalty) * format_valid_mask_expanded.to(dtype=torch.float32))

    # Add global repeat penalty (applies regardless of format validity)
    global_repeat_penalty = compute_global_repeat_penalty_tensor(
        responses=responses,
        response_mask=response_mask,
        tokenizer=tokenizer,
        repeat_penalty_coef=float(getattr(cfg, "global_repeat_penalty_coef", 0.1)),
    )
    reward = reward + global_repeat_penalty

    format_reward = compute_format_reward_tensor(
        responses=responses,
        response_mask=response_mask,
        tokenizer=tokenizer,
        reward_coef=float(getattr(cfg, "format_reward_coef", 0.1)),
    )
    reward = reward + format_reward

    reward = reward * response_mask

    if cfg.normalize_per_sequence:
        valid_cnt = response_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        seq_mean = reward.sum(dim=-1, keepdim=True) / valid_cnt
        centered = (reward - seq_mean) * response_mask
        seq_var = (centered.square().sum(dim=-1, keepdim=True) / valid_cnt).clamp_min(float(cfg.eps))
        reward = centered / torch.sqrt(seq_var) * response_mask

    metrics = {
        "teacher_step_reward/proxy_mean": (reward.sum() / response_mask.sum().clamp_min(1.0)).item(),
        "teacher_step_reward/imitation_reward_mean": (
            (imitation_reward * response_mask).sum() / response_mask.sum().clamp_min(1.0)
        ).item(),
        "teacher_step_reward/length_reward_mean": (
            (length_reward * response_mask).sum() / response_mask.sum().clamp_min(1.0)
        ).item(),
        "teacher_step_reward/teacher_avg_prob_proxy_mean": (
            (teacher_avg_prob_proxy * response_mask).sum() / response_mask.sum().clamp_min(1.0)
        ).item(),
        "teacher_step_reward/pi_mean": ((pi_t * response_mask).sum() / response_mask.sum().clamp_min(1.0)).item(),
        "teacher_step_reward/format_reward_mean": (
            format_reward.sum() / response_mask.sum().clamp_min(1.0)
        ).item(),
        "teacher_step_reward/format_pass_rate": (
            (format_reward.sum(dim=-1) > 0).to(dtype=torch.float32).mean().item()
        ),
        "teacher_step_reward/global_repeat_penalty_mean": (
            (global_repeat_penalty * response_mask).sum() / response_mask.sum().clamp_min(1.0)
        ).item(),
    }
    if sum_pi_squared is not None:
        sum_pi_squared_f = sum_pi_squared.to(dtype=torch.float32)
        metrics["teacher_step_reward/sum_pi_squared_mean"] = (
            (sum_pi_squared_f * response_mask).sum() / response_mask.sum().clamp_min(1.0)
        ).item()

    return reward, metrics
