import numpy as np

from pyversity.utils import EPS32, normalize_rows, prepare_inputs


def _exp_zscore_weights(relevance: np.ndarray, beta: float) -> np.ndarray:
    """Compute exponential z-score weights for relevance scores."""
    mean = float(relevance.mean())
    std = float(relevance.std() + EPS32)
    weights = np.exp(beta * (relevance - mean) / std)
    return weights.astype(np.float32, copy=False)


def dpp(
    relevances: np.ndarray,
    embeddings: np.ndarray,
    k: int,
    beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy determinantal point process (DPP) selection.

    This strategy selects a diverse and relevant subset of `k` items by
    maximizing the determinant of a kernel matrix that balances item relevance
    and pairwise similarity.

    :param relevances: 1D array of relevance scores for each item.
    :param embeddings: 2D array of shape (n_samples, n_features).
    :param k: Number of items to select.
    :param beta: Controls the influence of relevance scores in the DPP kernel.
                 Higher values increase the emphasis on relevance.
    :return: Tuple of selected indices and their marginal gains.
    """
    relevance_scores, feature_matrix, top_k, early_exit = prepare_inputs(relevances, embeddings, k)
    if early_exit:
        return np.empty(0, np.int32), np.empty(0, np.float32)

    feature_matrix = normalize_rows(feature_matrix)

    num_items = feature_matrix.shape[0]
    weights = _exp_zscore_weights(relevance_scores, beta)

    # Diagonal of L plus jitter is the initial residual variance.
    residual_variance = (weights * weights + float(EPS32)).astype(np.float32, copy=False)

    # Columns will store orthogonalized update components.
    component_matrix = np.zeros((num_items, top_k), dtype=np.float32)

    selected_indices = np.empty(top_k, dtype=np.int32)
    marginal_gains = np.empty(top_k, dtype=np.float32)
    selected_mask = np.zeros(num_items, dtype=bool)

    t = 0
    for t in range(top_k):
        residual_variance[selected_mask] = -np.inf
        best_index = int(np.argmax(residual_variance))
        best_gain = float(residual_variance[best_index])

        selected_indices[t] = best_index
        marginal_gains[t] = best_gain
        selected_mask[best_index] = True

        if t == top_k - 1 or best_gain <= 0.0:
            t += 1
            break

        weighted_similarity_to_best = (weights * (feature_matrix @ feature_matrix[best_index])) * weights[best_index]

        if t > 0:
            projected_component: np.ndarray = component_matrix[:, :t] @ component_matrix[best_index, :t]
        else:
            projected_component = np.zeros(num_items, dtype=np.float32)

        sqrt_best_gain = np.float32(np.sqrt(best_gain))
        update_component = (weighted_similarity_to_best - projected_component) / (sqrt_best_gain + EPS32)

        component_matrix[:, t] = update_component
        residual_variance -= update_component * update_component
        np.maximum(residual_variance, 0.0, out=residual_variance)

    return selected_indices[:t], marginal_gains[:t]
