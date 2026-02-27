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

import math
from types import SimpleNamespace

import torch

from verl.trainer.ppo.teacher_step_reward import compute_teacher_step_proxy_reward


class _DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        mapping = {"a": 1, "b": 2, "c": 3}
        return [mapping[ch] for ch in text if ch in mapping]


def test_teacher_step_proxy_reward_formula_on_cpu():
    cfg = SimpleNamespace(
        teacher_sequence_key="teacher_sequence",
        freq_coef=1.0,
        pi_coef=1.0,
        sum_pi_squared_coef=1.0,
        teacher_avg_prob_coef=0.0,
        teacher_avg_prob_mode="none",
        normalize_per_sequence=False,
        eps=1e-6,
    )

    responses = torch.tensor([[1, 2, 9]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 0]], dtype=torch.float32)
    old_log_probs = torch.log(torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float32))
    sum_pi_squared = torch.tensor([[0.4, 0.4, 0.4]], dtype=torch.float32)
    reward_model_items = [{"teacher_sequence": [1, 1, 3]}]

    reward, metrics = compute_teacher_step_proxy_reward(
        responses=responses,
        response_mask=response_mask,
        old_log_probs=old_log_probs,
        sum_pi_squared=sum_pi_squared,
        reward_model_items=reward_model_items,
        tokenizer=_DummyTokenizer(),
        cfg=cfg,
    )

    # freq - pi + sum_pi_squared
    # t0: 2/3 - 0.2 + 0.4 = 0.8666...
    # t1: 0   - 0.5 + 0.4 = -0.1
    # t2 masked
    expected = torch.tensor([[0.8666667, -0.1, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(reward, expected, atol=1e-6, rtol=1e-6)

    assert math.isclose(metrics["teacher_step_reward/proxy_mean"], (0.8666667 - 0.1) / 2, rel_tol=1e-5)


def test_teacher_step_proxy_reward_supports_teacher_text_on_cpu():
    cfg = SimpleNamespace(
        teacher_sequence_key="teacher_sequence",
        freq_coef=1.0,
        pi_coef=0.0,
        sum_pi_squared_coef=0.0,
        teacher_avg_prob_coef=0.0,
        teacher_avg_prob_mode="none",
        normalize_per_sequence=False,
        eps=1e-6,
    )

    responses = torch.tensor([[1, 2, 3]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 1]], dtype=torch.float32)
    old_log_probs = torch.zeros((1, 3), dtype=torch.float32)
    reward_model_items = [{"teacher_sequence": "abac"}]  # token ids [1,2,1,3]

    reward, _ = compute_teacher_step_proxy_reward(
        responses=responses,
        response_mask=response_mask,
        old_log_probs=old_log_probs,
        sum_pi_squared=None,
        reward_model_items=reward_model_items,
        tokenizer=_DummyTokenizer(),
        cfg=cfg,
    )

    expected = torch.tensor([[0.5, 0.25, 0.25]], dtype=torch.float32)
    torch.testing.assert_close(reward, expected, atol=1e-6, rtol=1e-6)


def test_teacher_step_proxy_reward_with_avg_prob_proxy_term_on_cpu():
    cfg = SimpleNamespace(
        teacher_sequence_key="teacher_sequence",
        freq_coef=1.0,
        pi_coef=0.0,
        sum_pi_squared_coef=0.0,
        teacher_avg_prob_coef=1.0,
        teacher_avg_prob_mode="seq_freq_mean",
        normalize_per_sequence=False,
        eps=1e-6,
    )

    responses = torch.tensor([[1, 2, 3]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 1]], dtype=torch.float32)
    old_log_probs = torch.zeros((1, 3), dtype=torch.float32)
    reward_model_items = [{"teacher_sequence": [1, 1, 3]}]

    reward, _ = compute_teacher_step_proxy_reward(
        responses=responses,
        response_mask=response_mask,
        old_log_probs=old_log_probs,
        sum_pi_squared=None,
        reward_model_items=reward_model_items,
        tokenizer=_DummyTokenizer(),
        cfg=cfg,
    )

    # teacher_freq = [2/3, 0, 1/3], seq mean = 1/3
    expected = torch.tensor([[1 / 3, -1 / 3, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(reward, expected, atol=1e-6, rtol=1e-6)
