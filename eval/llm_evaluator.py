import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from typing import Any, List, Optional, Tuple

import torch

warnings.filterwarnings("ignore")

from transformers.utils import logging as hf_logging
hf_logging.disable_progress_bar()
hf_logging.set_verbosity_error()

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DISCERN_ROOT = os.path.dirname(_THIS_DIR)
if _DISCERN_ROOT not in sys.path:
    sys.path.insert(0, _DISCERN_ROOT)

from data_utils.utils import make_unique_id
from utils.json_utils import parse_json_loose
from utils.reproducibility import set_seed, configure_generation_defaults


# --------------------------------------------------------------------------
# Text normalization and choice matching
# --------------------------------------------------------------------------

def standardize_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def exact_choice_match(text: Any, choices: List[str]) -> Optional[str]:
    text_norm = standardize_text(text)
    for choice in choices:
        if standardize_text(choice) == text_norm:
            return choice
    return None


def prefixed_choice_match(text: Any, choices: List[str]) -> Optional[str]:
    """Match predictions like "B. <choice text>" or "(b) <choice text>"."""
    text_norm = standardize_text(text)
    for idx, choice in enumerate(choices):
        choice_norm = standardize_text(choice)
        prefix = standardize_text(f"{chr(ord('A') + idx)}.")
        if text_norm == f"{prefix} {choice_norm}":
            return choice
        if text_norm == f"{chr(ord('A') + idx).lower()} {choice_norm}":
            return choice
        if text_norm.endswith(choice_norm):
            leading = text_norm[: -len(choice_norm)].strip()
            if re.fullmatch(r"\(?\s*[a-z]\s*\)?[\.:]?", leading):
                return choice
    return None


def letter_choice_match(text: Any, choices: List[str]) -> Optional[str]:
    """Match predictions that are just a choice letter, e.g. "B" or "(b)"."""
    text = "" if text is None else str(text).strip()
    match = re.fullmatch(r"\(?\s*([A-Z])\s*\)?[\.:]?", text, flags=re.IGNORECASE)
    if not match:
        return None

    idx = ord(match.group(1).upper()) - ord("A")
    if 0 <= idx < len(choices):
        return choices[idx]
    return None


def parse_standardized_prediction(output_text: str, choices: List[str]) -> Optional[str]:
    """Extract a choice from the LLM standardizer's output, preferring its JSON field."""
    parsed = parse_json_loose(output_text)
    candidates = []
    if isinstance(parsed, dict):
        candidates.append(parsed.get("standardized_prediction", ""))
    candidates.append(output_text)

    for candidate in candidates:
        for matcher in (exact_choice_match, prefixed_choice_match, letter_choice_match):
            match = matcher(candidate, choices)
            if match is not None:
                return match
    return None


# --------------------------------------------------------------------------
# LLM-based standardization (used only when exact matching fails)
# --------------------------------------------------------------------------

def build_standardization_prompt(question: str, choices: List[str], explanation: str) -> List[dict]:
    choices_text = "\n".join(
        f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices)
    )

    system_prompt = (
        "You are an evaluator that standardizes a model's multiple-choice prediction.\n"
        "Your job is NOT to answer the medical question yourself.\n"
        "Infer which listed choice the model predicted using only the model's explanation.\n"
        "The explanation is the reasoning process used by the executor when producing its final answer.\n"
        "Return exactly one of the provided choice texts.\n"
        "Output only valid JSON with this schema:\n"
        '{"standardized_prediction": "exact choice text"}'
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Choices:\n{choices_text}\n\n"
        f"Model explanation:\n{explanation}\n\n"
        "Based only on the model explanation, which choice did the model predict? "
        "Return one exact choice text from the choices."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def standardize_prediction_with_llm(
    model,
    tokenizer,
    question: str,
    choices: List[str],
    explanation: str,
    max_new_tokens: int = 128,
) -> Tuple[Optional[str], str]:
    messages = build_standardization_prompt(question, choices, explanation)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    standardized = parse_standardized_prediction(output_text, choices)

    del model_inputs, generated_ids, output_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return standardized, output_text


# --------------------------------------------------------------------------
# Per-case scoring
# --------------------------------------------------------------------------

def load_cache(cache_file: str) -> dict:
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def score_one_file(
    fname: str,
    data: dict,
    choices: List[str],
    model,
    tokenizer,
    cache: dict,
    cache_file: str,
    max_new_tokens: int,
) -> Optional[dict]:
    """Standardize and score a single result record; None if it isn't a scorable MCQ."""
    question = str(data.get("question", "")).strip()
    gt = str(data.get("ground_truth", "")).strip()
    raw_pred = "" if data.get("pred_answer") is None else str(data.get("pred_answer")).strip()
    explanation = str(data.get("explanation") or data.get("pred_answer") or "").strip()

    if not choices or not isinstance(choices, list) or not gt:
        return None

    gt_choice = exact_choice_match(gt, choices) or gt

    if not raw_pred:
        standardized_pred, method = "", "missing"
    else:
        standardized_pred = exact_choice_match(raw_pred, choices)
        method = "exact"

        if standardized_pred is None:
            method = "llm"
            cache_key = make_unique_id(str(data.get("long_id", "")), question)
            cached = cache.get(cache_key)
            if (
                cached
                and cached.get("raw_prediction") == raw_pred
                and cached.get("standardized_prediction") is not None
            ):
                standardized_pred = cached.get("standardized_prediction")
                method = "cache"
            else:
                standardized_pred, raw_llm_output = standardize_prediction_with_llm(
                    model, tokenizer, question, choices, explanation, max_new_tokens=max_new_tokens,
                )
                # Persist after every LLM call so a crash mid-run doesn't lose completed work.
                cache[cache_key] = {
                    "file": fname,
                    "raw_prediction": raw_pred,
                    "standardized_prediction": standardized_pred,
                    "raw_llm_output": raw_llm_output,
                    "ground_truth": gt,
                    "choices": choices,
                }
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2, ensure_ascii=False)

            if standardized_pred is None:
                standardized_pred = raw_pred
                method = "llm_failed"

    is_correct = standardize_text(standardized_pred) == standardize_text(gt_choice)
    return {
        "method": method,
        "raw_pred": raw_pred,
        "standardized_pred": standardized_pred,
        "gt_choice": gt_choice,
        "correct": is_correct,
    }


# --------------------------------------------------------------------------
# Benchmark metrics
# --------------------------------------------------------------------------

# Benchmarks with a fixed class set are reported via balanced accuracy; the rest via
# accuracy (see the paper, Sec. 4.1 and Supp. B.1).
BACC_BENCHMARKS = {"gtex", "tcga", "panda"}


def infer_benchmark_name(results_dir: str) -> str:
    tail = os.path.basename(os.path.normpath(results_dir))
    prefix = "multipathqa_"
    return tail[len(prefix):] if tail.startswith(prefix) else tail


def balanced_accuracy_from_counts(tp_by_class: dict, total_by_class: dict) -> float:
    recalls = [
        tp_by_class.get(c, 0) / total for c, total in total_by_class.items() if total > 0
    ]
    return sum(recalls) / len(recalls) if recalls else 0.0


def primary_metric(benchmark_name: str, acc: float, bacc: Optional[float]) -> Tuple[str, float]:
    """The single metric this benchmark is reported with: "bacc" or "acc"."""
    if benchmark_name in BACC_BENCHMARKS and bacc is not None:
        return "bacc", bacc
    return "acc", acc


# --------------------------------------------------------------------------
# Evaluation driver
# --------------------------------------------------------------------------

def evaluate_results(
    results_dir: str,
    qwen_ckpt: str,
    cache_file: Optional[str] = None,
    max_new_tokens: int = 128,
    seed: int = 128,
    strict_determinism: bool = False,
    verbose: bool = False,
    summary_file: Optional[str] = None,
) -> Tuple[float, Optional[float]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(seed, strict=strict_determinism)

    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Directory does not exist: {results_dir}")

    cache_file = cache_file or os.path.join(results_dir, "llm_standardized_predictions_cache.json")
    summary_file = summary_file or os.path.join(results_dir, "eval_summary.json")
    cache = load_cache(cache_file)

    tokenizer = AutoTokenizer.from_pretrained(qwen_ckpt)
    model = AutoModelForCausalLM.from_pretrained(qwen_ckpt, torch_dtype="auto", device_map="auto")
    model.eval()
    configure_generation_defaults(model)

    benchmark_name = infer_benchmark_name(results_dir)
    total_mcq = 0
    correct_mcq = 0
    exact_used = 0
    llm_used = 0
    llm_failed = 0
    total_by_class = defaultdict(int)
    tp_by_class = defaultdict(int)

    file_list = sorted(f for f in os.listdir(results_dir) if f.endswith(".json"))
    for fname in file_list:
        with open(os.path.join(results_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)

        result = score_one_file(
            fname, data, data.get("choices"), model, tokenizer, cache, cache_file, max_new_tokens,
        )
        if result is None:
            continue

        total_mcq += 1
        total_by_class[result["gt_choice"]] += 1
        if result["correct"]:
            correct_mcq += 1
            tp_by_class[result["gt_choice"]] += 1

        if result["method"] == "exact":
            exact_used += 1
        elif result["method"] in ("cache", "llm"):
            llm_used += 1
        elif result["method"] == "llm_failed":
            llm_failed += 1

        if verbose:
            print(
                f"{fname}: method={result['method']}, raw={result['raw_pred']!r}, "
                f"standardized={result['standardized_pred']!r}, gt={result['gt_choice']!r}, "
                f"correct={result['correct']}"
            )

    acc_score = correct_mcq / total_mcq if total_mcq > 0 else 0.0
    bacc = balanced_accuracy_from_counts(tp_by_class, total_by_class) \
        if benchmark_name in BACC_BENCHMARKS and total_mcq > 0 else None
    metric_name, metric_value = primary_metric(benchmark_name, acc_score, bacc)

    print(f"[{benchmark_name}] MCQ: {correct_mcq}/{total_mcq} correct  |  "
          f"{metric_name.upper()} = {metric_value * 100:.2f}")
    print(f"[{benchmark_name}] Standardization: exact={exact_used}, llm={llm_used}, llm_failed={llm_failed}")

    summary_data = {
        "benchmark": benchmark_name,
        "total_mcq": total_mcq,
        "correct_mcq": correct_mcq,
        "metric_name": metric_name,
        "metric_value": round(metric_value * 100, 2),
        "exact_used": exact_used,
        "llm_used": llm_used,
        "llm_failed": llm_failed,
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return acc_score, bacc


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-based evaluator: standardize free-form DISCERN predictions into "
                     "MCQ choices, then score them."
    )
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--qwen_ckpt", type=str, required=True, help="Path or HF id for Qwen-3-4B checkpoint.")
    parser.add_argument("--cache_file", type=str, default=None, help="Optional cache path for standardized predictions.")
    parser.add_argument("--summary_file", type=str, default=None, help="Optional path for the JSON accuracy summary.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=128, help="Global seed for reproducible standardization.")
    parser.add_argument("--strict_determinism", action="store_true",
                        help="Use stricter deterministic CUDA kernels when available.")
    parser.add_argument("--verbose", action="store_true", help="Print a line per scored case, not just the summary.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_results(
        results_dir=args.results_dir,
        qwen_ckpt=args.qwen_ckpt,
        cache_file=args.cache_file,
        summary_file=args.summary_file,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        strict_determinism=args.strict_determinism,
        verbose=args.verbose,
    )
