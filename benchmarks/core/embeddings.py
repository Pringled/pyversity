from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from benchmarks.core.data import InteractionData

logger = logging.getLogger(__name__)


def generate_embeddings(data: InteractionData, dim: int = 64, seed: int = 42) -> NDArray[np.float32]:
    """Generate L2-normalized item embeddings using SVD on the user-item matrix."""
    ui_matrix = sparse.coo_matrix(
        (np.ones(len(data.user_ids)), (data.user_ids, data.item_ids)),
        shape=(data.n_users, data.n_items),
        dtype=np.float32,
    ).tocsr()

    effective_dim = min(dim, min(ui_matrix.T.shape) - 1)
    svd = TruncatedSVD(n_components=effective_dim, random_state=seed)
    embeddings = svd.fit_transform(ui_matrix.T)
    embeddings = normalize(embeddings, norm="l2", axis=1)

    logger.debug(f"Embeddings: {embeddings.shape}, explained variance: {svd.explained_variance_ratio_.sum():.3f}")
    return embeddings.astype(np.float32)


def compute_similarity_matrix(embeddings: NDArray[np.float32], top_k: int = 100) -> sparse.csr_matrix:
    """Compute sparse item-item similarity matrix (top-k per item)."""
    n_items = embeddings.shape[0]
    similarity = embeddings @ embeddings.T

    rows, cols, data = [], [], []
    for item_idx in range(n_items):
        sims = similarity[item_idx].copy()
        sims[item_idx] = -np.inf  # Exclude self

        if top_k < n_items - 1:
            top_indices = np.argpartition(sims, -top_k)[-top_k:]
        else:
            top_indices = np.arange(n_items)[np.arange(n_items) != item_idx]

        for neighbor_idx in top_indices:
            if sims[neighbor_idx] > 0:
                rows.append(item_idx)
                cols.append(neighbor_idx)
                data.append(sims[neighbor_idx])

    return sparse.coo_matrix((data, (rows, cols)), shape=(n_items, n_items), dtype=np.float32).tocsr()


def get_candidates(
    profile_items: NDArray[np.int64],
    similarity_matrix: sparse.csr_matrix,
    topk_per_item: int = 50,
    max_candidates: int = 1000,
    exclude_items: set[int] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """
    Generate candidate items for a user based on their profile.

    Args:
    ----
        profile_items: Items in user's profile (used to generate candidates)
        similarity_matrix: Item-item similarity matrix
        topk_per_item: Top-k similar items per profile item
        max_candidates: Maximum number of candidates to return
        exclude_items: Items to exclude from candidates (e.g., already-seen items)

    """
    # Exclude profile items and any additional exclusions
    exclude_set = set(profile_items)
    if exclude_items is not None:
        exclude_set |= exclude_items

    scores: dict[int, float] = {}

    for item in profile_items:
        row = similarity_matrix.getrow(item)
        indices, similarities = row.indices, row.data

        if len(indices) > topk_per_item:
            top_idx = np.argpartition(similarities, -topk_per_item)[-topk_per_item:]
            indices, similarities = indices[top_idx], similarities[top_idx]

        for idx, sim in zip(indices, similarities):
            if idx not in exclude_set:
                scores[idx] = scores.get(idx, 0.0) + sim

    if not scores:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    sorted_items = sorted(scores.items(), key=lambda x: -x[1])[:max_candidates]
    return (
        np.array([item[0] for item in sorted_items], dtype=np.int64),
        np.array([item[1] for item in sorted_items], dtype=np.float32),
    )
