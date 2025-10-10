from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray

from pyversity.datatypes import Metric
from pyversity.utils import normalize_rows, prepare_inputs, vector_similarity


def greedy_select(
    strategy: Literal["mmr", "msd"],
    relevances: np.ndarray,
    embeddings: np.ndarray,
    k: int,
    *,
    metric: Metric,
    normalize: bool,
    alpha: float,
) -> Tuple[NDArray[np.int32], NDArray[np.float32]]:
    """
    Greedy selection for MMR/MSD strategies.

    This function implements the greedy selection process for both the
    Maximal Marginal Relevance (MMR) and Maximal Sum of Distances (MSD)
    strategies. It iteratively selects items that optimize a trade-off
    between relevance and diversity based on the specified strategy.

    :param strategy: Either "mmr" (Maximal Marginal Relevance) or "msd" (Maximal Sum of Distances).
    :param relevances: 1D array of relevance scores for each item.
    :param embeddings: 2D array of shape (n_samples, n_features).
    :param k: Number of items to select.
    :param metric: Similarity metric to use. Default is Metric.COSINE.
    :param normalize: Whether to normalize embeddings before computing similarity.
    :param alpha: Trade-off parameter in [0, 1].
                  1.0 = pure relevance, 0.0 = pure diversity.
    :return: Tuple of selected indices and their marginal gains.
    :raises ValueError: If strategy is not "mmr" or "msd".
    :raises ValueError: If alpha is not in [0, 1].
    :raises ValueError: If input shapes are inconsistent.
    """
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    relevance_scores, feature_matrix, top_k, early_exit = prepare_inputs(relevances, embeddings, k)
    if early_exit:
        return np.empty(0, np.int32), np.empty(0, np.float32)

    if metric == Metric.COSINE and normalize:
        feature_matrix = normalize_rows(feature_matrix)

    num_items = feature_matrix.shape[0]
    selected_mask = np.zeros(num_items, dtype=bool)
    selected_indices = np.empty(top_k, dtype=np.int32)
    marginal_gains = np.empty(top_k, dtype=np.float32)

    if strategy == "mmr":
        max_similarity_to_selected = np.full(num_items, -np.inf, dtype=np.float32)
    else:
        cumulative_distance_to_selected = np.zeros(num_items, dtype=np.float32)

    last_selected_index = int(np.argmax(relevance_scores))
    selected_indices[0] = last_selected_index
    marginal_gains[0] = float(alpha * relevance_scores[last_selected_index])
    selected_mask[last_selected_index] = True

    for t in range(1, top_k):
        # For MMR we use non-negative similarity; for MSD we use raw cosine/dot.
        if strategy == "mmr":
            sim_for_penalty = vector_similarity(feature_matrix, feature_matrix[last_selected_index], metric=metric)
            np.maximum(max_similarity_to_selected, sim_for_penalty, out=max_similarity_to_selected)
            candidate_scores = alpha * relevance_scores - (1.0 - alpha) * max_similarity_to_selected
        else:
            raw_sim = feature_matrix @ feature_matrix[last_selected_index]
            if metric == Metric.COSINE:
                cosine = np.clip(raw_sim, -1.0, 1.0)
                distance = 1.0 - cosine
            else:
                distance = -raw_sim
            cumulative_distance_to_selected += distance
            candidate_scores = alpha * relevance_scores + (1.0 - alpha) * cumulative_distance_to_selected

        candidate_scores[selected_mask] = -np.inf
        last_selected_index = int(np.argmax(candidate_scores))
        selected_indices[t] = last_selected_index
        marginal_gains[t] = float(candidate_scores[last_selected_index])
        selected_mask[last_selected_index] = True

    return selected_indices, marginal_gains
