"""Stage 3: Bayesian Evidence Acquisition"""

from typing import Dict, List

import numpy as np


def prior_vector(belief: Dict[str, float], choices: List[str]) -> np.ndarray:
    vals = np.array(
        [max(float(belief.get(c, 0.0)), 0.0) for c in choices],
        dtype=np.float64,
    )
    total = float(vals.sum())
    if total <= 1e-12:
        K = len(choices)
        return np.full(K, 1.0 / max(K, 1), dtype=np.float64)
    return vals / total


def build_support_matrix(
    R: Dict[str, Dict[str, float]],
    candidates: List[str],
    choices: List[str],
    eps: float = 1e-6,
) -> np.ndarray:
    M, K = len(candidates), len(choices)
    R_mat = np.full((M, K), 1.0 / max(K, 1), dtype=np.float64)
    for i, name in enumerate(candidates):
        scores = R.get(name, {})
        for j, c in enumerate(choices):
            R_mat[i, j] = float(scores.get(c, 1.0 / max(K, 1)))
    np.clip(R_mat, eps, 1.0 - eps, out=R_mat)
    return R_mat


def _entropy_rows(P: np.ndarray) -> np.ndarray:
    eps = 1e-12
    safeP = np.clip(P, eps, 1.0)
    return -np.sum(P * np.log(safeP), axis=-1)


def compute_eig(R: np.ndarray, pi: np.ndarray) -> np.ndarray:
    eps = 1e-12

    b = R @ pi
    H_prior = float(_entropy_rows(pi[None, :])[0])

    pos = R * pi[None, :]
    pos_sum = pos.sum(axis=1, keepdims=True)
    pos_norm = np.where(pos_sum > eps, pos / np.maximum(pos_sum, eps), pi[None, :])
    H_pos = _entropy_rows(pos_norm)

    neg = (1.0 - R) * pi[None, :]
    neg_sum = neg.sum(axis=1, keepdims=True)
    neg_norm = np.where(neg_sum > eps, neg / np.maximum(neg_sum, eps), pi[None, :])
    H_neg = _entropy_rows(neg_norm)

    eig = H_prior - (b * H_pos + (1.0 - b) * H_neg)
    return np.maximum(eig, 0.0)


def select_top_b_by_eig(
    candidates: List[str],
    R: Dict[str, Dict[str, float]],
    belief: Dict[str, float],
    choices: List[str],
    batch_size: int,
) -> List[str]:
    if not candidates or batch_size <= 0 or not choices:
        return []

    R_mat = build_support_matrix(R, candidates, choices)
    pi = prior_vector(belief, choices)
    eig = compute_eig(R_mat, pi)

    B = min(int(batch_size), len(candidates))
    order = np.argsort(-eig)[:B]

    return [candidates[i] for i in order]
