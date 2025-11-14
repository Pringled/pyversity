import numpy as np

from pyversity.datatypes import DiversificationResult, Strategy
from pyversity.utils import EPS32, normalize_rows, prepare_inputs


def ssd(  # noqa: C901
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    diversity: float = 0.5,
    window: int = 10,
    gamma: float = 1.0,
    normalize: bool = True,
    append_bias: bool = True,
    normalize_scores: bool = True,
) -> DiversificationResult:
    """
    Sliding Spectrum Decomposition (SSD) selection.

    This strategy performs greedy, sequence-aware diversification
    that maintains a sliding window of orthogonal bases (modified Gram-Schmidt).
    Each step picks the item that maximizes a combination of relevance and orthogonalized residual norm.

    :param embeddings: 2D array of shape (n_samples, n_features).
    :param scores: 1D array of relevance scores for each item.
    :param k: Number of items to select.
    :param diversity: Trade-off between relevance and coverage/diversity in [0, 1] (inverse of theta parameter).
                      1.0 = pure diversity, 0.0 = pure relevance.
    :param window: Sliding window size (≥1) for Gram-Schmidt bases.
    :param gamma: Diversity scale (>0).
    :param normalize: Whether to L2-normalize embeddings before computing similarity (cosine geometry).
    :param append_bias: Whether to append a constant-one bias dimension after normalization.
    :param normalize_scores: Whether to z-score normalize relevance scores per request (stabilizes gamma).
    :return: A DiversificationResult containing the selected item indices,
      their selection scores, the strategy used, and the parameters.
    :raises ValueError: If diversity is not in [0, 1].
    :raises ValueError: If window < 1.
    :raises ValueError: If gamma ≤ 0.
    """
    # Validate parameters
    if not (0.0 <= float(diversity) <= 1.0):
        raise ValueError("diversity must be in [0, 1]")
    if window < 1:
        raise ValueError("window must be >= 1")
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0")

    theta = 1.0 - float(diversity)
    window_size = int(window)

    # Prepare inputs
    feature_matrix, relevance_scores, top_k, early_exit = prepare_inputs(embeddings, scores, k)
    if early_exit:
        return DiversificationResult(
            indices=np.empty(0, np.int32),
            selection_scores=np.empty(0, np.float32),
            strategy=Strategy.SSD,
            diversity=diversity,
            parameters={
                "variant": "SSD*",
                "window": window_size,
                "gamma": float(gamma),
                "normalize": bool(normalize),
                "append_bias": bool(append_bias),
                "normalize_scores": bool(normalize_scores),
            },
        )

    if theta == 1.0:
        # Pure relevance: select top-k by relevance scores
        topk_indices = np.argsort(-relevance_scores)[:top_k].astype(np.int32)
        topk_scores = relevance_scores[topk_indices].astype(np.float32, copy=False)
        return DiversificationResult(
            indices=topk_indices,
            selection_scores=topk_scores,
            strategy=Strategy.SSD,
            diversity=diversity,
            parameters={
                "window": window_size,
                "gamma": float(gamma),
                "normalize": bool(normalize),
                "append_bias": bool(append_bias),
                "normalize_scores": bool(normalize_scores),
            },
        )

    # Normalize feature vectors to unit length
    if normalize:
        feature_matrix = normalize_rows(feature_matrix)

    # Append a constant-one dimension for bias
    if append_bias:
        last_col_is_ones = feature_matrix.shape[1] > 0 and np.allclose(feature_matrix[:, -1], 1.0, atol=1e-6, rtol=0.0)
        if not last_col_is_ones:
            ones = np.ones((feature_matrix.shape[0], 1), dtype=feature_matrix.dtype)
            feature_matrix = np.concatenate([feature_matrix, ones], axis=1)

    # Per-request z-score normalization of relevance to stabilize gamma
    if normalize_scores:
        mean = float(np.mean(relevance_scores))
        std = float(np.std(relevance_scores))
        relevance_scores = (relevance_scores - mean) / std if std > 0.0 else (relevance_scores - mean)

    num_items, _ = feature_matrix.shape

    # Initialize selection state
    selected_mask = np.zeros(num_items, dtype=bool)
    selected_indices = np.empty(top_k, dtype=np.int32)
    selection_scores = np.empty(top_k, dtype=np.float32)

    # Residuals of all candidates under the current sliding window
    residual_matrix = feature_matrix.astype(np.float32, copy=True)

    # Sliding window lists (oldest first)
    basis_vectors: list[np.ndarray] = []
    projection_coeffs_per_basis: list[np.ndarray] = []

    def push_new_basis(selected_index: int) -> None:
        """Update the sliding window with the newly selected basis."""
        if len(basis_vectors) == window_size:
            oldest_basis = basis_vectors.pop(0)
            oldest_coeffs = projection_coeffs_per_basis.pop(0)
            mask_unselected = ~selected_mask
            if np.any(mask_unselected):
                residual_matrix[mask_unselected] += oldest_coeffs[mask_unselected, None] * oldest_basis

        new_basis = residual_matrix[selected_index].copy()
        denom = float(new_basis @ new_basis) + EPS32
        basis_vectors.append(new_basis)

        mask_unselected = ~selected_mask
        coeffs = np.zeros(num_items, dtype=np.float32)
        if np.any(mask_unselected):
            proj = (residual_matrix[mask_unselected] @ new_basis) / denom
            coeffs[mask_unselected] = proj
            residual_matrix[mask_unselected] -= proj[:, None] * new_basis
        projection_coeffs_per_basis.append(coeffs)

    # First selection: pick item with highest relevance score
    first_index = int(np.argmax(relevance_scores))
    selected_mask[first_index] = True
    selected_indices[0] = first_index

    # Compute selection score for the first item
    first_norm = float(np.linalg.norm(feature_matrix[first_index]))
    selection_scores[0] = float(theta * relevance_scores[first_index] + (1.0 - theta) * gamma * first_norm)

    push_new_basis(first_index)

    for step in range(1, top_k):
        available_indices = np.where(~selected_mask)[0]
        if available_indices.size == 0:
            selected_indices = selected_indices[:step]
            selection_scores = selection_scores[:step]
            break

        residual_norms = np.linalg.norm(residual_matrix[available_indices], axis=1)
        combined_scores = theta * relevance_scores[available_indices] + (1.0 - theta) * gamma * residual_norms
        local_best = int(np.argmax(combined_scores))
        best_index = int(available_indices[local_best])
        best_score = float(combined_scores[local_best])

        selected_mask[best_index] = True
        selected_indices[step] = best_index
        selection_scores[step] = best_score
        push_new_basis(best_index)

    return DiversificationResult(
        indices=selected_indices,
        selection_scores=selection_scores.astype(np.float32, copy=False),
        strategy=Strategy.SSD,
        diversity=diversity,
        parameters={
            "window": window_size,
            "gamma": float(gamma),
            "normalize": bool(normalize),
            "append_bias": bool(append_bias),
            "normalize_scores": bool(normalize_scores),
        },
    )
