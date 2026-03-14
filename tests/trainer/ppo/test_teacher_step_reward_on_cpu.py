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

import torch

from verl.trainer.config import TeacherStepRewardConfig
from verl.trainer.ppo.teacher_step_reward import compute_teacher_step_proxy_reward


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(int(token_id)) for token_id in token_ids)


def test_teacher_step_reward_adds_format_bonus_on_last_valid_token():
    tokenizer = DummyTokenizer()
    cfg = TeacherStepRewardConfig(
        freq_coef=0.0,
        pi_coef=0.0,
        sum_pi_squared_coef=0.0,
        teacher_avg_prob_coef=0.0,
        format_reward_coef=0.1,
    )
    text = "<think>a</think><answer>b</answer>"
    token_ids = tokenizer.encode(text)

    reward, metrics = compute_teacher_step_proxy_reward(
        responses=torch.tensor([token_ids + [0, 0]], dtype=torch.long),
        response_mask=torch.tensor([[1.0] * len(token_ids) + [0.0, 0.0]], dtype=torch.float32),
        old_log_probs=torch.zeros((1, len(token_ids) + 2), dtype=torch.float32),
        sum_pi_squared=None,
        reward_model_items=None,
        tokenizer=tokenizer,
        cfg=cfg,
    )

    expected = torch.zeros((1, len(token_ids) + 2), dtype=torch.float32)
    expected[0, len(token_ids) - 1] = 0.1
    assert torch.allclose(reward, expected)
    assert metrics["teacher_step_reward/format_pass_rate"] == 1.0


def test_teacher_step_reward_rejects_repeated_or_disordered_tags():
    tokenizer = DummyTokenizer()
    cfg = TeacherStepRewardConfig(
        freq_coef=0.0,
        pi_coef=0.0,
        sum_pi_squared_coef=0.0,
        teacher_avg_prob_coef=0.0,
        format_reward_coef=0.1,
    )
    invalid_text_1 = "<answer>b</answer><think>a</think>"
    invalid_text_2 = "<think>a</think><answer>b</answer><answer>c</answer>"
    invalid_token_ids_1 = tokenizer.encode(invalid_text_1)
    invalid_token_ids_2 = tokenizer.encode(invalid_text_2)
    max_len = max(len(invalid_token_ids_1), len(invalid_token_ids_2))

    responses = torch.tensor(
        [
            invalid_token_ids_1 + [0] * (max_len - len(invalid_token_ids_1)),
            invalid_token_ids_2 + [0] * (max_len - len(invalid_token_ids_2)),
        ],
        dtype=torch.long,
    )
    response_mask = torch.tensor(
        [
            [1.0] * len(invalid_token_ids_1) + [0.0] * (max_len - len(invalid_token_ids_1)),
            [1.0] * len(invalid_token_ids_2) + [0.0] * (max_len - len(invalid_token_ids_2)),
        ],
        dtype=torch.float32,
    )

    reward, metrics = compute_teacher_step_proxy_reward(
        responses=responses,
        response_mask=response_mask,
        old_log_probs=torch.zeros((2, max_len), dtype=torch.float32),
        sum_pi_squared=None,
        reward_model_items=None,
        tokenizer=tokenizer,
        cfg=cfg,
    )

    assert torch.allclose(reward, torch.zeros_like(reward))
    assert metrics["teacher_step_reward/format_pass_rate"] == 0.0
