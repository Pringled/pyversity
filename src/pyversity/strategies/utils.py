from typing import Literal

import numpy as np

from pyversity.datatypes import DiversificationResult, Metric, Strategy
from pyversity.utils import normalize_rows, prepare_inputs, vector_similarity


def greedy_select(
    strategy: Literal["mmr", "msd"],
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    *,
    metric: Metric,
    normalize: bool,
    diversity: float = 0.5,
) -> DiversificationResult:
    """
    Greedy selection for MMR/MSD strategies.

    This function implements the greedy selection process for both the
    Maximal Marginal Relevance (MMR) and Maximal Sum of Distances (MSD)
    strategies. It iteratively selects items that optimize a trade-off
    between relevance and diversity based on the specified strategy.

    :param strategy: Either "mmr" (Maximal Marginal Relevance) or "msd" (Maximal Sum of Distances).
    :param embeddings: 2D array of shape (n_samples, n_features).
    :param scores: 1D array of relevance scores for each item.
    :param k: Number of items to select.
    :param metric: Similarity metric to use. Default is Metric.COSINE.
    :param normalize: Whether to normalize embeddings before computing similarity.
    :param diversity: Trade-off parameter in [0, 1]. Inverse of lambda parameter.
                  1.0 = pure diversity, 0.0 = pure relevance.
    :return: A DiversificationResult containing the selected item indices,
      their selection scores, the strategy used, and the parameters.
    :raises ValueError: If diversity is not in [0, 1].
    :raises ValueError: If input shapes are inconsistent.
    """
    # Validate parameters
    if not (0.0 <= float(diversity) <= 1.0):
        raise ValueError("diversity must be in [0, 1]")

    params = {
        "metric": metric,
    }

    # Lambda parameter for trade-off between relevance and diversity
    # This is 1 - diversity to align with common notation
    lambda_param = 1.0 - diversity

    # Prepare inputs
    feature_matrix, relevance_scores, top_k, early_exit = prepare_inputs(embeddings, scores, k)
    if early_exit:
        # Nothing to select: return empty arrays
        return DiversificationResult(
            indices=np.empty(0, np.int32),
            selection_scores=np.empty(0, np.float32),
            strategy=Strategy.MMR if strategy == "mmr" else Strategy.MSD,
            diversity=diversity,
            parameters=params,
        )

    if metric == Metric.COSINE and normalize:
        # Normalize feature vectors to unit length for cosine similarity
        feature_matrix = normalize_rows(feature_matrix)

    # Initialize selection state
    num_items = feature_matrix.shape[0]
    selected_mask = np.zeros(num_items, dtype=bool)
    selected_indices = np.empty(top_k, dtype=np.int32)
    marginal_gains = np.empty(top_k, dtype=np.float32)

    if strategy == "mmr":
        # For MMR we track the maximum similarity to any selected item
        max_similarity_to_selected = np.full(num_items, -np.inf, dtype=np.float32)
    else:
        # For MSD we track the cumulative distance to all selected items
        cumulative_distance_to_selected = np.zeros(num_items, dtype=np.float32)

    # Select the first item based on pure relevance
    best_index = int(np.argmax(relevance_scores))
    selected_indices[0] = best_index
    marginal_gains[0] = float(lambda_param * relevance_scores[best_index])
    selected_mask[best_index] = True

    for step in range(1, top_k):
        if strategy == "mmr":
            # Update maximum similarity to selected items
            sim_for_penalty = vector_similarity(feature_matrix, feature_matrix[best_index], metric=metric)
            np.maximum(max_similarity_to_selected, sim_for_penalty, out=max_similarity_to_selected)
            candidate_scores = lambda_param * relevance_scores - (1.0 - lambda_param) * max_similarity_to_selected
        else:
            # Update cumulative distance to selected items
            raw_sim = feature_matrix @ feature_matrix[best_index]
            if metric == Metric.COSINE:
                cosine = np.clip(raw_sim, -1.0, 1.0)
                distance = 1.0 - cosine
            else:
                distance = -raw_sim
            cumulative_distance_to_selected += distance
            candidate_scores = lambda_param * relevance_scores + (1.0 - lambda_param) * cumulative_distance_to_selected

        # Mask already selected items and select the best candidate
        candidate_scores[selected_mask] = -np.inf
        best_index = int(np.argmax(candidate_scores))
        selected_indices[step] = best_index
        marginal_gains[step] = float(candidate_scores[best_index])
        selected_mask[best_index] = True

    return DiversificationResult(
        indices=selected_indices,
        selection_scores=marginal_gains,
        strategy=Strategy.MMR if strategy == "mmr" else Strategy.MSD,
        diversity=diversity,
        parameters=params,
    )
