"""Stage 2: Belief Construction"""

from typing import Dict, List


def uniform_prior(choices: List[str]) -> Dict[str, float]:
    k = len(choices)
    return {c: 1.0 / k for c in choices} if k > 0 else {}


def normalize_R_rows(
    R: Dict[str, Dict[str, float]], choices: List[str], eps: float = 1e-6
) -> Dict[str, Dict[str, float]]:
    uniform = 1.0 / max(1, len(choices))
    normalized: Dict[str, Dict[str, float]] = {}
    for patch_name, scores in R.items():
        vals = [max(float(scores.get(c, 0.5)), 0.0) for c in choices]
        total = sum(vals)
        if total <= eps:
            normalized[patch_name] = {c: uniform for c in choices}
        else:
            normalized[patch_name] = {c: v / total for c, v in zip(choices, vals)}
    return normalized


def aggregate_prior_from_R(R: Dict[str, Dict[str, float]], choices: List[str]) -> Dict[str, float]:
    if not R:
        return uniform_prior(choices)
    totals = {c: 0.0 for c in choices}
    for patch_scores in R.values():
        for c in choices:
            totals[c] += float(patch_scores.get(c, 0.5))
    n = len(R)
    means = {c: totals[c] / n for c in choices}
    total = sum(means.values())
    if total <= 0:
        return uniform_prior(choices)
    return {c: means[c] / total for c in choices}
