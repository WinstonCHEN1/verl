#!/usr/bin/env python3
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

import re


def extract_choice(solution_str: str) -> str | None:
    if not solution_str:
        return None

    matches = re.findall(r"\b([ABCD])\b", solution_str.upper())
    if matches:
        return matches[-1]

    compact = re.findall(r"ANSWER\s*[:：]?\s*([ABCD])", solution_str.upper())
    if compact:
        return compact[-1]

    return None


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    pred = extract_choice(solution_str)
    gt = str(ground_truth).strip().upper()
    if pred is None:
        return {
            "score": 0.0,
            "pred": None,
            "ground_truth": gt,
            "correct": False,
        }

    correct = pred == gt
    return {
        "score": 1.0 if correct else 0.0,
        "pred": pred,
        "ground_truth": gt,
        "correct": correct,
    }
