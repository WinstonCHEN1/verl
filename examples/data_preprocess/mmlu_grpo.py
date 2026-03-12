import argparse
import os

import pandas as pd
import numpy as np

SYSTEM_PROMPT = "Please think deeply before your response."


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
    choices = np.array(row["choices"]).astype(str).tolist()
    if not isinstance(choices, (list, tuple)) or len(choices) != 4:
        raise ValueError(f"Expected `choices` to be a length-4 list, got: {choices}")

    if subject:
        question = f"Subject: {subject}\n\n{question}"
    prompt = (
        "Answer the following multiple-choice question.\\n"
        "Choose exactly one option from A, B, C, D.\\n"
        "At the end, output in this exact format: Final answer: <A/B/C/D>.\\n\\n"
        f"Question: {question}\n"
        f"Choices:\n"
        f"A. {str(choices[0]).strip()}\n"
        f"B. {str(choices[1]).strip()}\n"
        f"C. {str(choices[2]).strip()}\n"
        f"D. {str(choices[3]).strip()}\n\n"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def convert_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    records = []
    for idx, row in df.iterrows():
        raw_answer = row["answer"]
        if isinstance(raw_answer, str):
            answer = raw_answer.strip().upper()
        elif isinstance(raw_answer, (int, float)):
            answer = ["A", "B", "C", "D"][int(raw_answer)]
        else:
            answer = str(raw_answer).strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"Unexpected answer at row {idx}: {answer}")

        choices = np.array(row["choices"]).tolist()
        if not isinstance(choices, (list, tuple)) or len(choices) != 4:
            raise ValueError(f"Expected `choices` to be a length-4 list, got: {choices}")

        records.append(
            {
                "data_source": "mmlu",
                "prompt": build_prompt(row),
                "ability": row.get("ability", "multiple_choice"),
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                    "choices": {
                        "A": str(choices[0]).strip(),
                        "B": str(choices[1]).strip(),
                        "C": str(choices[2]).strip(),
                        "D": str(choices[3]).strip(),
                    },
                },
                "extra_info": {
                    "split": split,
                    "index": int(idx),
                    "subject": row.get("subject", None),
                },
            }
        )

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", default="/mnt/ali-sh-1/usr/lihaitao/chenguo/data/mmlu_parquet/mmlu_validation.parquet")
    parser.add_argument("--local_save_dir", default="/mnt/ali-sh-1/usr/lihaitao/chenguo/data/mmlu_grpo")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    val_df = convert_split(read_table(args.val_path), split="validation")

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    val_path = os.path.join(local_save_dir, "validation.parquet")
    val_df.to_parquet(val_path, index=False)

    print(f"Saved validation parquet to {val_path} with {len(val_df)} rows")

    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()