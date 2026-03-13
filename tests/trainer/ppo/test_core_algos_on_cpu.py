# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import random
import unittest

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_rloo_outcome_advantage,
    get_adv_estimator_fn,
    register_adv_est,
)


def mock_test_fn():
    pass


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


def test_compute_rloo_token_level_advantage_no_sequence_broadcast():
    token_level_rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    response_mask = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    index = np.array(["g1", "g1"], dtype=object)

    advantages, returns = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    expected = torch.tensor([[-2.0, -2.0], [2.0, 2.0]], dtype=torch.float32)
    assert torch.allclose(advantages, expected)
    assert torch.allclose(returns, expected)


def test_compute_rloo_token_level_advantage_respects_mask_and_single_group():
    token_level_rewards = torch.tensor([[1.0, 9.0], [3.0, 7.0], [5.0, 6.0]], dtype=torch.float32)
    response_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    index = np.array(["g1", "g1", "solo"], dtype=object)

    advantages, returns = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    expected = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(advantages, expected)
    assert torch.allclose(returns, expected)


def test_compute_rloo_token_level_advantage_strict_mask_for_loo_baseline():
    # g1 has 3 samples. At token t=1 only sample-0 is valid.
    token_level_rewards = torch.tensor(
        [[10.0, 2.0], [4.0, 100.0], [1.0, 200.0]],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
        dtype=torch.float32,
    )
    index = np.array(["g1", "g1", "g1"], dtype=object)

    advantages, returns = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # t=0:
    # sample0 baseline=(4+1)/2=2.5 => 7.5
    # sample1 baseline=(10+1)/2=5.5 => -1.5
    # sample2 baseline=(10+4)/2=7.0 => -6.0
    # t=1:
    # only sample0 valid, no valid leave-one-out peers => 0
    expected = torch.tensor([[7.5, 0.0], [-1.5, 0.0], [-6.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(advantages, expected)
    assert torch.allclose(returns, expected)


def test_compute_grpo_token_level_advantage_uses_return_to_go():
    token_level_rewards = torch.tensor(
        [
            [0.3, 0.1, 0.5, 0.2, 0.0],
            [0.0, 0.0, 0.1, 0.0, 0.0],
            [0.2, 0.4, 0.3, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array(["g1", "g1", "g1", "g1"], dtype=object)

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    expected_returns = torch.tensor(
        [
            [1.1, 0.8, 0.7, 0.2, 0.0],
            [0.1, 0.1, 0.1, 0.0, 0.0],
            [1.1, 0.9, 0.5, 0.2, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    total_returns = torch.tensor([1.1, 0.1, 1.1, 0.0], dtype=torch.float32)
    baseline = total_returns.mean()
    scale = total_returns.std(unbiased=False) + 1e-6
    expected_advantages = (expected_returns - baseline) / scale

    assert torch.allclose(returns, expected_returns, atol=1e-6)
    assert torch.allclose(advantages, expected_advantages, atol=1e-5)
    assert advantages[0, 0] > advantages[0, 1] > advantages[0, 2]


def test_compute_grpo_token_level_advantage_zeros_single_valid_position():
    token_level_rewards = torch.tensor(
        [
            [0.5, 0.2, 0.1],
            [0.2, 0.1, 0.0],
            [0.1, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    index = np.array(["g1", "g1", "g1"], dtype=object)

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    expected_returns = torch.tensor(
        [
            [0.8, 0.3, 0.1],
            [0.3, 0.1, 0.0],
            [0.1, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(returns, expected_returns, atol=1e-6)
    assert torch.allclose(advantages[:, 2], torch.zeros(3), atol=1e-6)
    assert torch.allclose(advantages[2, 1:], torch.zeros(2), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
