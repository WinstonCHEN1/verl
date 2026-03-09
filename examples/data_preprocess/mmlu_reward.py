import re
from typing import Optional
def extract_choice(solution_str: str) -> Optional[str]:
    if not solution_str:
        return None

    m = re.search(
        r"final\s*answer\s*[:：]\s*<?([A-D])>?",
        solution_str,
        flags=re.IGNORECASE,
    )

    if m:
        return m.group(1).upper()

    return None

def compute_score(data_source, solution_str, ground_truth, extra_info=None):

    pred = extract_choice(solution_str)
    gt = str(ground_truth).strip().upper()

    if pred is None:
        correct = False
    else:
        correct = pred == gt

    return {
        "score": 1.0 if correct else 0.0,
        "acc": 1.0 if correct else 0.0,
    }