import numpy as np

from pyversity.datatypes import Metric
from pyversity.strategies.utils import greedy_select


def mmr(
    relevances: np.ndarray,
    embeddings: np.ndarray,
    k: int,
    alpha: float = 0.5,
    metric: Metric = Metric.COSINE,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Maximal Marginal Relevance (MMR) selection.

    This strategy selects `k` items that balance relevance and diversity by
    iteratively choosing items that maximize a combination of their relevance
    and their dissimilarity to already selected items.

    :param relevances: 1D array of relevance scores for each item.
    :param embeddings: 2D array of shape (n_samples, n_features).
    :param k: Number of items to select.
    :param alpha: Trade-off parameter in [0, 1].
                  1.0 = pure relevance, 0.0 = pure diversity.
    :param metric: Similarity metric to use. Default is Metric.COSINE.
    :param normalize: Whether to normalize embeddings before computing similarity.
    :return: Tuple of selected indices and their marginal gains.
    """
    return greedy_select(
        "mmr",
        relevances,
        embeddings,
        k,
        metric=metric,
        normalize=normalize,
        alpha=alpha,
    )
