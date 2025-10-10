import numpy as np

from pyversity.datatypes import Metric
from pyversity.utils import normalize_rows, pairwise_similarity, prepare_inputs


def cover(
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    theta: float = 0.5,
    gamma: float = 0.5,
    metric: Metric = Metric.COSINE,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select a subset of items that balances relevance and coverage.

    This strategy chooses `k` items by combining pure relevance with
    diversity-driven coverage using a concave submodular formulation.

    :param embeddings: 2D array of shape (n_samples, n_features).
    :param scores: 1D array of relevance scores for each item.
    :param k: Number of items to select.
    :param theta: Trade-off between relevance and coverage in [0, 1].
                  1.0 = pure relevance, 0.0 = pure coverage.
    :param gamma: Concavity parameter in (0, 1]; lower values emphasize diversity.
    :param metric: Similarity metric to use. Default is Metric.COSINE.
    :param normalize: Whether to normalize embeddings before computing similarity.
    :return: Tuple of selected indices and their marginal gains.
    :raises ValueError: If theta is not in [0, 1].
    :raises ValueError: If gamma is not in (0, 1].
    """
    # Validate parameters
    if not (0.0 <= float(theta) <= 1.0):
        raise ValueError("theta must be in [0, 1]")
    if not (0.0 < float(gamma) <= 1.0):
        raise ValueError("gamma must be in (0, 1]")

    # Prepare inputs
    feature_matrix, relevance_scores, top_k, early_exit = prepare_inputs(embeddings, scores, k)
    if early_exit:
        # Nothing to select: return empty arrays
        return np.empty(0, np.int32), np.empty(0, np.float32)

    if metric == Metric.COSINE and normalize:
        # Normalize feature vectors to unit length for cosine similarity
        feature_matrix = normalize_rows(feature_matrix)

    if float(theta) == 1.0:
        # Pure relevance: select top-k by relevance scores
        topk = np.argsort(-relevance_scores)[:top_k].astype(np.int32)
        gains = relevance_scores[topk].astype(np.float32, copy=False)
        return topk, gains

    # Compute non-negative similarities for coverage to avoid concave-power NaNs
    similarity_matrix = pairwise_similarity(feature_matrix, metric)
    transposed_similarity_matrix = similarity_matrix.T

    # Initialize selection state
    accumulated_coverage = np.zeros(similarity_matrix.shape[0], dtype=np.float32)
    selected_mask = np.zeros(similarity_matrix.shape[0], dtype=bool)
    selected_indices = np.empty(top_k, dtype=np.int32)
    marginal_gains = np.empty(top_k, dtype=np.float32)

    for step in range(top_k):
        # Compute coverage gains using concave transformation
        concave_before = np.power(accumulated_coverage, gamma)
        concave_after = np.power(transposed_similarity_matrix + accumulated_coverage[None, :], gamma)
        coverage_gains = (concave_after - concave_before[None, :]).sum(axis=1)

        # Combine relevance and coverage gains
        candidate_scores = theta * relevance_scores + (1.0 - theta) * coverage_gains
        candidate_scores[selected_mask] = -np.inf

        # Select item with highest combined score
        best_index = int(np.argmax(candidate_scores))
        selected_indices[step] = best_index
        marginal_gains[step] = float(candidate_scores[best_index])
        selected_mask[best_index] = True

        # Update accumulated coverage
        accumulated_coverage += similarity_matrix[:, best_index]

    return selected_indices, marginal_gains
