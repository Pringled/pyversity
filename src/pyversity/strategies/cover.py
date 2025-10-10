import numpy as np

from pyversity.datatypes import Metric
from pyversity.utils import normalize_rows, pairwise_similarity, prepare_inputs


def cover(
    relevances: np.ndarray,
    embeddings: np.ndarray,
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

    :param relevances: 1D array of relevance scores for each item.
    :param embeddings: 2D array of shape (n_samples, n_features).
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
    if not (0.0 <= float(theta) <= 1.0):
        raise ValueError("theta must be in [0, 1]")
    if not (0.0 < float(gamma) <= 1.0):
        raise ValueError("gamma must be in (0, 1]")

    relevance_scores, feature_matrix, top_k, early_exit = prepare_inputs(relevances, embeddings, k)
    if early_exit:
        return np.empty(0, np.int32), np.empty(0, np.float32)

    if metric == Metric.COSINE and normalize:
        feature_matrix = normalize_rows(feature_matrix)

    # Pure relevance: short-circuit
    if float(theta) == 1.0:
        topk = np.argsort(-relevance_scores)[:top_k].astype(np.int32)
        gains = relevance_scores[topk].astype(np.float32, copy=False)
        return topk, gains

    # Nonnegative similarities for coverage to avoid concave-power NaNs
    similarity_matrix = pairwise_similarity(feature_matrix, metric)
    transposed_similarity = similarity_matrix.T

    n = similarity_matrix.shape[0]
    accumulated_coverage = np.zeros(n, dtype=np.float32)
    selected_mask = np.zeros(n, dtype=bool)
    selected_indices = np.empty(top_k, dtype=np.int32)
    marginal_gains = np.empty(top_k, dtype=np.float32)

    for t in range(top_k):
        concave_before = np.power(accumulated_coverage, gamma)
        concave_after = np.power(transposed_similarity + accumulated_coverage[None, :], gamma)
        coverage_gains = (concave_after - concave_before[None, :]).sum(axis=1)

        candidate_scores = theta * relevance_scores + (1.0 - theta) * coverage_gains
        candidate_scores[selected_mask] = -np.inf

        chosen = int(np.argmax(candidate_scores))
        selected_indices[t] = chosen
        marginal_gains[t] = float(candidate_scores[chosen])
        selected_mask[chosen] = True

        accumulated_coverage += similarity_matrix[:, chosen]

    return selected_indices, marginal_gains
