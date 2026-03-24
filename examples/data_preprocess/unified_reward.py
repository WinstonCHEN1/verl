import re
from typing import Optional, Dict, Any

# ============ MMLU Scoring ============

def extract_choice(solution_str: str) -> Optional[str]:
    """Extract choice from <answer>X</answer> or 'Final answer: X' format"""
    if not solution_str:
        return None

    # First, try to extract from <answer> tags (priority)
    m = re.search(
        r"<answer>\s*([A-J])\s*</answer>",
        solution_str,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).upper()

    # Fallback: Support A-J (for MMLU-Pro with up to 10 options)
    m = re.search(
        r"final\s*answer\s*[:：]\s*<?([A-J])>?",
        solution_str,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    return None


def compute_mmlu_score(solution_str: str, ground_truth: str) -> Dict[str, float]:
    """Score MMLU multiple choice questions"""
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


# ============ Math Scoring (from math_reward.py) ============

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]
    return retval


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def extract_from_answer_tags(solution_str: str) -> Optional[str]:
    """Extract content from <answer> tags"""
    m = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        solution_str,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return None


def compute_math_score(solution_str: str, ground_truth: str) -> Dict[str, float]:
    """Score math questions by extracting from <answer> tags first, then \\boxed{}"""
    score = 0.0
    try:
        # First, try to extract from <answer> tags (priority)
        answer = extract_from_answer_tags(solution_str)

        # Fallback to \\boxed{} if no <answer> tags found
        if answer is None:
            string_in_last_boxed = last_boxed_only_string(solution_str)
            if string_in_last_boxed is not None:
                answer = remove_boxed(string_in_last_boxed)

        if answer is not None and is_equiv(answer, ground_truth):
            score = 1.0
    except Exception as e:
        print(f"Error in math scoring: {e}")

    return {
        "score": score,
        "acc": score,
    }


# ============ Unified Scoring ============

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[Dict[str, Any]] = None):
    """
    Unified scoring function that dispatches to appropriate scorer based on data_source.

    Args:
        data_source: Dataset identifier (e.g., 'math500', 'mmlu', 'mmlu_pro', 'train_dataset')
        solution_str: Model's generated response
        ground_truth: Expected answer
        extra_info: Additional metadata

    Returns:
        Dict with 'score' and 'acc' keys
    """
    data_source_lower = str(data_source).lower()

    # Math datasets
    if any(x in data_source_lower for x in ['math', 'math500', 'gsm8k', 'aime']):
        return compute_math_score(solution_str, ground_truth)

    # MMLU datasets (including MMLU-Pro)
    elif any(x in data_source_lower for x in ['mmlu', 'multiple_choice']):
        return compute_mmlu_score(solution_str, ground_truth)

    # Default to MMLU scoring for unknown datasets
    # You can add more dataset types here
    else:
        # Try math scoring first (for oxed{}), then MMLU
        math_result = compute_math_score(solution_str, ground_truth)
        if math_result["score"] > 0:
            return math_result

        mmlu_result = compute_mmlu_score(solution_str, ground_truth)
        if mmlu_result["score"] > 0:
            return mmlu_result

        # Return MMLU result (will be 0 if neither matched)
        return mmlu_result
