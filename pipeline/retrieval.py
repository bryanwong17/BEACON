"""Stage 1: Coarse Candidate Retrieval"""

from typing import Dict, List

import numpy as np


def coarse_candidate_retrieval(
    vlm_encoder,
    query: str,
    unqueried_patch_names: List[str],
    feature_cache: Dict[str, np.ndarray],
    top_m: int,
) -> List[str]:
    available = [n for n in unqueried_patch_names if n in feature_cache]
    if not available:
        return list(unqueried_patch_names)

    text_emb = vlm_encoder.encode_text([query], batch_size=1)
    text_emb = text_emb / np.linalg.norm(text_emb, axis=-1, keepdims=True)

    patch_embs = np.stack([feature_cache[n] for n in available])
    patch_embs = patch_embs / np.maximum(
        np.linalg.norm(patch_embs, axis=-1, keepdims=True), 1e-8
    )

    sims = (patch_embs @ text_emb.T).squeeze(-1)
    top_m = min(top_m, len(available))
    top_indices = np.argpartition(sims, -top_m)[-top_m:]
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

    selected = [available[i] for i in top_indices]
    return selected
