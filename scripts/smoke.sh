#!/bin/bash

# 1. 激活环境
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate opd

# 2. 运行测试
python -m pytest -q tests/trainer/ppo/test_teacher_step_reward_on_cpu.py