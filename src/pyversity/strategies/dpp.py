import numpy as np

from pyversity.datatypes import DiversificationResult, Strategy
from pyversity.utils import EPS32, normalize_rows, prepare_inputs


def _exp_zscore_weights(relevance: np.ndarray, beta: float) -> np.ndarray:
    """Compute exponential z-score weights for relevance scores."""
    mean = float(relevance.mean())
    std = float(relevance.std() + EPS32)
    weights = np.exp(beta * (relevance - mean) / std)
    return weights.astype(np.float32, copy=False)


def dpp(
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    diversity: float = 0.5,
    scale: float = 1.0,
) -> DiversificationResult:
    """
    Greedy determinantal point process (DPP) selection.

    This strategy selects a diverse and relevant subset of `k` items by
    maximizing the determinant of a kernel matrix that balances item relevance
    and pairwise similarity. Note that

    :param embeddings: 2D array of shape (n_samples, n_features).
    :param scores: 1D array of relevance scores for each item.
    :param k: Number of items to select.
    :param diversity: Controls the influence of relevance scores in the DPP kernel (inverse of beta parameter).
                      Higher values increase the emphasis on diversity.
    :param scale: Optional scaling factor for the beta parameter to adjust relevance influence.
    :return: A DiversificationResult containing the selected item indices,
      their selection scores, the strategy used, and the parameters.
    :raises ValueError: If diversity is not in [0, 1].
    """
    if not (0.0 <= float(diversity) <= 1.0):
        raise ValueError("diversity must be in [0, 1]")

    # Beta parameter to control relevance influence in DPP kernel.
    # This is the inverse of diversity to align with common notation.
    beta = (1 - diversity) * scale

    # Prepare inputs
    feature_matrix, relevance_scores, top_k, early_exit = prepare_inputs(embeddings, scores, k)
    if early_exit:
        # Nothing to select: return empty arrays
        return DiversificationResult(
            indices=np.empty(0, np.int32),
            selection_scores=np.empty(0, np.float32),
            strategy=Strategy.DPP,
            diversity=diversity,
            parameters={"scale": scale},
        )
    # Normalize feature vectors to unit length for cosine similarity
    feature_matrix = normalize_rows(feature_matrix)

    num_items = feature_matrix.shape[0]
    weights = _exp_zscore_weights(relevance_scores, beta)

    # Initial residual variance is the weighted self-similarity
    residual_variance = (weights * weights + float(EPS32)).astype(np.float32, copy=False)

    # Initialize selection state
    component_matrix = np.zeros((num_items, top_k), dtype=np.float32)
    selected_indices = np.empty(top_k, dtype=np.int32)
    marginal_gains = np.empty(top_k, dtype=np.float32)
    selected_mask = np.zeros(num_items, dtype=bool)

    step = 0
    for step in range(top_k):
        # Select item with highest residual variance
        residual_variance[selected_mask] = -np.inf
        best_index = int(np.argmax(residual_variance))
        best_score = float(residual_variance[best_index])

        selected_indices[step] = best_index
        marginal_gains[step] = best_score
        selected_mask[best_index] = True

        if step == top_k - 1:
            # No more items to select
            step += 1
            break

        # Update residual variance using the new component
        weighted_similarity_to_best = (weights * (feature_matrix @ feature_matrix[best_index])) * weights[best_index]

        if step > 0:
            # Project out the component in the span of previously selected items
            projected_component: np.ndarray = component_matrix[:, :step] @ component_matrix[best_index, :step]
        else:
            # No previous components, so projection is zero
            projected_component = np.zeros(num_items, dtype=np.float32)

        # Compute update component
        sqrt_best_score = np.float32(np.sqrt(best_score))
        update_component = (weighted_similarity_to_best - projected_component) / (sqrt_best_score + EPS32)

        # Update component matrix and residual variance
        component_matrix[:, step] = update_component
        residual_variance -= update_component * update_component
        np.maximum(residual_variance, 0.0, out=residual_variance)

    return DiversificationResult(
        indices=selected_indices[:step],
        selection_scores=marginal_gains[:step],
        strategy=Strategy.DPP,
        diversity=diversity,
        parameters={"scale": scale},
    )
