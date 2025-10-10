from typing import Any

import numpy as np

from pyversity.datatypes import Strategy
from pyversity.strategies import cover, dpp, mmr, msd


def diversify(
    strategy: Strategy,
    relevances: np.ndarray,
    embeddings: np.ndarray,
    k: int,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Diversify a retrieval result using a selected strategy.

    :param strategy: The diversification strategy to apply. Supported strategies are: MMR, MSD, COVER, and DPP.
    :param relevances: Array of relevance scores for the items.
    :param embeddings: Array of embeddings for the items.
    :param k: The number of items to select in the diversified result.
    :param **kwargs: Additional keyword arguments passed to the specific strategy function.
    :return: A tuple containing an array of indices of the selected items
      and an array of corresponding relevance scores for the selected items.
    :raises ValueError: If the provided strategy is not recognized.
    """
    if strategy == Strategy.MMR:
        return mmr(relevances, embeddings, k, **kwargs)
    if strategy == Strategy.MSD:
        return msd(relevances, embeddings, k, **kwargs)
    if strategy == Strategy.COVER:
        return cover(relevances, embeddings, k, **kwargs)
    if strategy == Strategy.DPP:
        return dpp(relevances, embeddings, k, **kwargs)
    raise ValueError(f"Unknown strategy: {strategy}")
