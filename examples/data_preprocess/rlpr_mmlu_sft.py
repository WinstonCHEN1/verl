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
"""
Convert RLPR and MMLU-like tabular data into MultiTurnSFTDataset parquet files.
"""

import argparse
import ast
import json
import os
from typing import Any

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlpr_path", required=True, help="Path to the RLPR source file.")
    parser.add_argument("--mmlu_path", required=True, help="Path to the MMLU source file.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/rlpr_mmlu_sft",
        help="Directory to save converted train/validation parquet files.",
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--rlpr_answer_source",
        choices=["ground_truth", "reasoning_score_response"],
        default="ground_truth",
        help="Which RLPR field to use as the assistant target.",
    )
    return parser.parse_args()


def read_table(path: str) -> pd.DataFrame:
    path = os.path.expanduser(path)
    lower = path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if lower.endswith(".json"):
        return pd.read_json(path)
    raise ValueError(f"Unsupported file format for {path}")


def maybe_parse(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(stripped)
            except Exception:
                continue
    return value


def normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return str(content["content"])
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def normalize_messages(prompt_value: Any) -> list[dict[str, str]]:
    prompt_value = maybe_parse(prompt_value)
    if not isinstance(prompt_value, list):
        raise ValueError(f"Expected RLPR prompt to be a list, got {type(prompt_value)}")

    messages = []
    for message in prompt_value:
        message = maybe_parse(message)
        if not isinstance(message, dict):
            raise ValueError(f"Each RLPR prompt item must be a dict, got {type(message)}")
        role = str(message.get("role", "user"))
        content = normalize_content(message.get("content", ""))
        messages.append({"role": role, "content": content})
    return messages


def extract_nested(row: pd.Series, field: str) -> Any:
    if "." not in field:
        return row.get(field)

    current = row
    for key in field.split("."):
        if isinstance(current, pd.Series):
            current = current.get(key)
        else:
            current = maybe_parse(current)
            if not isinstance(current, dict):
                return None
            current = current.get(key)
    return current


def convert_rlpr(df: pd.DataFrame, answer_source: str) -> pd.DataFrame:
    answer_field = {
        "ground_truth": "reward_model.ground_truth",
        "reasoning_score_response": "extra_info.reasoning_score_response",
    }[answer_source]

    records = []
    for idx, row in df.iterrows():
        messages = normalize_messages(row["prompt"])
        answer = extract_nested(row, answer_field)
        answer = normalize_content(maybe_parse(answer)).strip()
        if not answer:
            raise ValueError(f"Empty RLPR answer at row {idx} from source {answer_source}")

        records.append(
            {
                "messages": messages + [{"role": "assistant", "content": answer}],
                "data_source": row.get("data_source", "RLPR"),
                "ability": row.get("ability", None),
                "extra_info": maybe_parse(row.get("extra_info", {})),
            }
        )

    return pd.DataFrame(records)


def build_mmlu_prompt(row: pd.Series) -> str:
    question = normalize_content(row["prompt"]).strip()
    choices = []
    for option in ["A", "B", "C", "D"]:
        if option not in row:
            raise ValueError(f"MMLU row is missing option column {option}")
        choices.append(f"{option}. {normalize_content(row[option]).strip()}")

    return (
        f"{question}\n\n"
        + "\n".join(choices)
        + "\n\nAnswer with the correct option letter only."
    )


def convert_mmlu(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, row in df.iterrows():
        answer = normalize_content(row["answer"]).strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"Unexpected MMLU answer at row {idx}: {answer}")

        records.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer the multiple-choice question with only the correct option letter.",
                    },
                    {"role": "user", "content": build_mmlu_prompt(row)},
                    {"role": "assistant", "content": answer},
                ],
                "data_source": row.get("data_source", "MMLU"),
                "ability": row.get("ability", "multiple_choice"),
                "extra_info": {"source_row": int(idx)},
            }
        )

    return pd.DataFrame(records)


def main():
    args = parse_args()

    rlpr_df = read_table(args.rlpr_path)
    mmlu_df = read_table(args.mmlu_path)

    train_df = convert_rlpr(rlpr_df, answer_source=args.rlpr_answer_source)
    val_df = convert_mmlu(mmlu_df)

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    train_path = os.path.join(save_dir, "train.parquet")
    val_path = os.path.join(save_dir, "validation.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Saved RLPR train parquet to {train_path} with {len(train_df)} rows")
    print(f"Saved MMLU validation parquet to {val_path} with {len(val_df)} rows")

    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(args.hdfs_dir)
        copy(src=save_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
