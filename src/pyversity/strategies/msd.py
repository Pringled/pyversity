import numpy as np

from pyversity.datatypes import DiversificationResult, Metric
from pyversity.strategies.utils import greedy_select


def msd(
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    diversity: float = 0.5,
    metric: Metric = Metric.COSINE,
    normalize: bool = True,
) -> DiversificationResult:
    """
    Maximal Sum of Distances (MSD) selection.

    This strategy selects `k` items that balance relevance and diversity by
    iteratively choosing items that maximize a combination of their relevance
    and their total distance to already selected items.

    :param embeddings: 2D array of shape (n_samples, n_features).
    :param scores: 1D array of relevance scores for each item.
    :param k: Number of items to select.
    :param diversity: Trade-off parameter in [0, 1] (inverse of lambda parameter).
                  1.0 = pure diversity, 0.0 = pure relevance.
    :param metric: Similarity metric to use. Default is Metric.COSINE.
    :param normalize: Whether to normalize embeddings before computing similarity.
    :return: A DiversificationResult containing the selected item indices,
      their selection scores, the strategy used, and the parameters.
    """
    return greedy_select(
        "msd",
        embeddings=embeddings,
        scores=scores,
        k=k,
        metric=metric,
        normalize=normalize,
        diversity=diversity,
    )
