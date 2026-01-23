from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mrr(recommendations: NDArray[np.int64], relevant: NDArray[np.int64]) -> float:
    """Compute Mean Reciprocal Rank."""
    relevant_set = set(relevant)
    for rank, item in enumerate(recommendations, start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg(recommendations: NDArray[np.int64], relevant: NDArray[np.int64], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at k."""
    relevant_set = set(relevant)
    recs = recommendations[:k]

    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recs) if item in relevant_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))

    return dcg / idcg if idcg > 0 else 0.0


def ilad(item_ids: NDArray[np.int64], embeddings: NDArray[np.float32]) -> float:
    """Compute Intra-List Average Distance (average pairwise dissimilarity)."""
    if len(item_ids) < 2:
        return 0.0

    embs = embeddings[item_ids]
    sims = embs @ embs.T
    num_items = len(item_ids)
    mask = np.triu(np.ones((num_items, num_items), dtype=bool), k=1)

    return float(np.mean(1.0 - sims[mask]))


def ilmd(item_ids: NDArray[np.int64], embeddings: NDArray[np.float32]) -> float:
    """Compute Intra-List Minimum Distance (minimum pairwise dissimilarity)."""
    if len(item_ids) < 2:
        return 0.0

    embs = embeddings[item_ids]
    sims = embs @ embs.T
    num_items = len(item_ids)
    mask = np.triu(np.ones((num_items, num_items), dtype=bool), k=1)

    return float(np.min(1.0 - sims[mask]))
