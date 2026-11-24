"""Frozen reranker: scores (caption d_x, answer statement a_k) pairs for Stage 2 (Eq. 2).

The prompt wording ("hypothesis") is kept verbatim from the experiments so that results reproduce."""

import re
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def prepare_runtime(tokenizer, max_length: int = 4096) -> dict:
    system_prompt = (
        "Judge whether the pathology evidence in <Evidence Description> supports "
        "the hypothesis in <Category Description>. "
        "Respond with only 'yes' if the evidence supports the hypothesis, otherwise 'no'."
    )
    prefix = (
        "<|im_start|>system\n"
        + system_prompt
        + "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return {
        "max_length": max_length,
        "prefix_tokens": tokenizer.encode(prefix, add_special_tokens=False),
        "suffix_tokens": tokenizer.encode(suffix, add_special_tokens=False),
        "token_true_id": tokenizer.convert_tokens_to_ids("yes"),
        "token_false_id": tokenizer.convert_tokens_to_ids("no"),
    }


def _format_pair(task: str, evidence: str, category_description: str) -> str:
    return (
        f"<Instruct>:{task}\n"
        f"<Evidence Description>:{evidence}\n"
        f"<Category Description>:{category_description}"
    )


def _normalize_choice_text(text) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _is_meta_choice(choice_text) -> bool:
    normalized = _normalize_choice_text(choice_text)
    return normalized in {
        "all of the above",
        "all the above",
        "all above",
        "none of the above",
        "none above",
    }


def _parse_letter_refs(normalized_text: str) -> List[int]:
    matches = re.findall(r"\b([a-z])\b", normalized_text)
    refs = []
    for letter in matches:
        idx = ord(letter) - ord("a")
        if 0 <= idx < 26 and idx not in refs:
            refs.append(idx)
    return refs


def _expand_choice_text(choice, question_choices) -> str:
    normalized = _normalize_choice_text(choice)
    all_choices = [str(c).strip() for c in (question_choices or [])]
    base_choices = [
        c for c in all_choices
        if _normalize_choice_text(c) != normalized and not _is_meta_choice(c)
    ]

    if normalized in {"all of the above", "all the above", "all above"}:
        if base_choices:
            return (
                "This means all listed non-meta options are supported: "
                + "; ".join(base_choices)
            )
        return "This means all listed options are supported."

    if normalized in {"none of the above", "none above"}:
        if base_choices:
            return (
                "This means none of the listed non-meta options are supported: "
                + "; ".join(base_choices)
            )
        return "This means none of the listed options are supported."

    both_match = re.fullmatch(r"both\s+([a-z])\s+and\s+([a-z])", normalized)
    if both_match:
        refs = _parse_letter_refs(normalized)
        resolved = [all_choices[i] for i in refs if i < len(all_choices)]
        if resolved:
            return "This means both referenced options are supported: " + "; ".join(resolved)

    either_match = re.fullmatch(r"(?:either\s+)?([a-z])\s+or\s+([a-z])", normalized)
    if either_match:
        refs = _parse_letter_refs(normalized)
        resolved = [all_choices[i] for i in refs if i < len(all_choices)]
        if resolved:
            return "This means one of the referenced options is supported: " + "; ".join(resolved)

    neither_match = re.fullmatch(r"neither\s+([a-z])\s+nor\s+([a-z])", normalized)
    if neither_match:
        refs = _parse_letter_refs(normalized)
        resolved = [all_choices[i] for i in refs if i < len(all_choices)]
        if resolved:
            return "This means neither referenced option is supported: " + "; ".join(resolved)

    all_except_match = re.fullmatch(r"all except\s+(.+)", normalized)
    if all_except_match:
        target = all_except_match.group(1).strip()
        excluded = []
        if re.fullmatch(r"[a-z](?:\s*(?:,|and)\s*[a-z])*", target):
            refs = _parse_letter_refs(target)
            excluded = [all_choices[i] for i in refs if i < len(all_choices)]
        else:
            excluded = [c for c in all_choices if _normalize_choice_text(c) == target]

        remaining = [c for c in base_choices if c not in excluded]
        if remaining:
            return (
                "This means all listed non-meta options are supported except: "
                + "; ".join(excluded or [target])
                + ". Supported options: "
                + "; ".join(remaining)
            )

    return str(choice).strip()


NEGATION_TRIGGERS = [
    "not observed",
    "not typically observed",
    "least evident",
    "least likely to be present",
    "absent",
    "not identified",
    "not present",
    "not seen",
    "not noted",
    "not involved",
    "not represented",
    "not typically associated",
    "which is not",
    "is not found",
    "is not identified",
    "is not present",
    "is absent",
]


def _is_negation_question(question_text) -> bool:
    q = str(question_text or "").lower()
    return any(t in q for t in NEGATION_TRIGGERS)


def _build_reranker_task(question: str) -> str:
    if _is_negation_question(question):
        return (
            "Assume the pathology option (hypothesis) is true. "
            "This is a negation-style pathology question, so a correct hypothesis means the named feature "
            "is absent, not observed, or less evident than the alternatives in the slide. "
            "Evaluate how likely the observed evidence would be under this assumption. "
            "If the evidence matches this negation-style hypothesis, respond 'yes'. "
            "If the evidence instead suggests the feature is present or prominent, respond 'no'."
        )
    return (
        "Assume the pathology option (hypothesis) is true. "
        "Evaluate how likely the observed evidence would be under this assumption. "
        "If the evidence is expected given the hypothesis, respond 'yes'. "
        "If the evidence would be unlikely or contradictory, respond 'no'."
    )


def build_statement_text(choice: str, question: str, choices: Optional[List[str]] = None) -> str:
    expanded_option_text = _expand_choice_text(choice, choices or [])
    if _is_negation_question(question):
        return (
            f"Pathology question: {question.strip()}\n"
            f"Hypothesis: The correct answer is '{choice}' because this feature is absent, "
            f"not observed, or less evident than the alternatives in the slide.\n"
            f"If this is true, the slide should indicate that '{expanded_option_text}' is missing, "
            f"not supported by the evidence, or less evident than the other options."
        )
    return (
        f"Pathology question: {question.strip()}\n"
        f"Hypothesis: The correct answer is '{choice}'.\n"
        f"If this is true, the slide should show: {expanded_option_text}"
    )


def _run_batched_inference(
    model, tokenizer, runtime: dict, pairs: List[str], batch_size: int,
) -> List[float]:
    if not pairs:
        return []

    prefix_tokens = runtime["prefix_tokens"]
    suffix_tokens = runtime["suffix_tokens"]
    token_true_id = runtime["token_true_id"]
    token_false_id = runtime["token_false_id"]
    max_length = runtime["max_length"]
    scores = []

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        )
        for i, ids in enumerate(encoded["input_ids"]):
            encoded["input_ids"][i] = prefix_tokens + ids + suffix_tokens

        model_inputs = tokenizer.pad(
            encoded, padding=True, return_tensors="pt", max_length=max_length
        )
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            logits = model(**model_inputs).logits[:, -1, :]

        # P(yes) via softmax over just the yes/no logits.
        batch_probs = F.log_softmax(
            torch.stack([logits[:, token_false_id], logits[:, token_true_id]], dim=1),
            dim=1,
        )[:, 1].exp().tolist()
        scores.extend(float(p) for p in batch_probs)

    return scores


def score_patch_descriptions(
    model,
    tokenizer,
    runtime: dict,
    patch_names: List[str],
    descriptions: Dict[str, str],
    question: str,
    choices: List[str],
    batch_size: int = 8,
) -> Dict[str, Dict[str, float]]:
    uniform = 1.0 / max(1, len(choices))

    scored = [n for n in patch_names if descriptions.get(n, "").strip()]
    all_pairs = []
    pair_index = []
    task = _build_reranker_task(question)

    for pi, name in enumerate(scored):
        desc = descriptions[name].strip()
        for ci, choice in enumerate(choices):
            hyp = build_statement_text(choice, question, choices)
            all_pairs.append(_format_pair(task, desc, hyp))
            pair_index.append((pi, ci))

    raw = _run_batched_inference(model, tokenizer, runtime, all_pairs, batch_size)

    result = {name: {c: uniform for c in choices} for name in patch_names}
    if len(raw) == len(pair_index):
        for (pi, ci), s in zip(pair_index, raw):
            result[scored[pi]][choices[ci]] = max(1e-6, min(1 - 1e-6, s))

    return result


def score_candidates(
    model,
    tokenizer,
    runtime: dict,
    candidate_patch_names: List[str],
    descriptions: Dict[str, str],
    question: str,
    choices: List[str],
    batch_size: int = 8,
) -> Dict[str, Dict[str, float]]:
    R = score_patch_descriptions(
        model, tokenizer, runtime,
        candidate_patch_names, descriptions, question, choices,
        batch_size=batch_size,
    )
    return R
