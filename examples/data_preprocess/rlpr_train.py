import argparse
import os

import pandas as pd
import numpy as np

SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."

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
    raise ValueError(f"Unsupported file format: {path}")


def build_prompt(row: pd.Series) -> list[dict[str, str]]:
    question_key = "question" if "question" in row else "prompt" 
    question = str(row[question_key]).strip()
    subject = str(row["subject"]).strip() if "subject" in row and pd.notna(row["subject"]) else None
    prompt = ( f"Question: {question}\n" )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def convert_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    records = []

    for idx, row in df.iterrows():

        # ground truth 直接读取
        ground_truth = str(row["answer"]).strip()

        records.append(
            {
                "data_source": "train_dataset",
                "prompt": build_prompt(row),
                "ability": row.get("ability", "generation"),
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "split": split,
                    "index": int(idx),
                },
            }
        )

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        default="/mnt/ali-sh-1/usr/lihaitao/chenguo/data/train_dataset/train.parquet",
    )
    parser.add_argument(
        "--local_save_dir",
        default="/mnt/ali-sh-1/usr/lihaitao/chenguo/data/train_dataset_grpo",
    )
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    train_df = convert_split(read_table(args.train_path), split="train")

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_path = os.path.join(local_save_dir, "train.parquet")
    train_df.to_parquet(train_path, index=False)

    print(f"Saved train parquet to {train_path} with {len(train_df)} rows")

    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()