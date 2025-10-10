from typing import Literal, overload

import numpy as np

from pyversity.datatypes import Metric
from pyversity.strategies.utils import greedy_select


@overload
def mmr(
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    lambda_param: float = 0.5,
    return_gains: Literal[True] = True,
    metric: Metric = Metric.COSINE,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def mmr(
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    lambda_param: float = 0.5,
    return_gains: Literal[False] = False,
    metric: Metric = Metric.COSINE,
    normalize: bool = True,
) -> np.ndarray: ...


def mmr(
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    lambda_param: float = 0.5,
    return_gains: bool = False,
    metric: Metric = Metric.COSINE,
    normalize: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Maximal Marginal Relevance (MMR) selection.

    This strategy selects `k` items that balance relevance and diversity by
    iteratively choosing items that maximize a combination of their relevance
    and their dissimilarity to already selected items.

    :param embeddings: 2D array of shape (n_samples, n_features).
    :param scores: 1D array of relevance scores for each item.
    :param k: Number of items to select.
    :param lambda_param: Trade-off parameter in [0, 1].
                  1.0 = pure relevance, 0.0 = pure diversity.
    :param return_gains: Whether to return the marginal gains along with the indices.
    :param metric: Similarity metric to use. Default is Metric.COSINE.
    :param normalize: Whether to normalize embeddings before computing similarity.
    :return: selected indices, or a tuple of selected indices and their marginal gains.
    """
    return greedy_select(
        "mmr",
        embeddings=embeddings,
        scores=scores,
        k=k,
        return_gains=return_gains,
        metric=metric,
        normalize=normalize,
        lambda_param=lambda_param,
    )
