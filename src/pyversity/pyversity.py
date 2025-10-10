from typing import Any

import numpy as np

from pyversity.datatypes import DiversificationResult, Strategy
from pyversity.strategies import cover, dpp, mmr, msd


def diversify(
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    strategy: Strategy = Strategy.MMR,
    **kwargs: Any,
) -> DiversificationResult:
    """
    Diversify a retrieval result using a selected strategy.

    :param embeddings: Embeddings of the items to be diversified.
    :param scores: Scores (relevances) of the items to be diversified.
    :param k: The number of items to select for the diversified result.
    :param strategy: The diversification strategy to apply.
      Supported strategies are: 'mmr' (default), 'msd', 'cover', and 'dpp'.
    :param **kwargs: Additional keyword arguments passed to the specific strategy function.
    :return: A DiversificationResult containing the selected item indices,
      their marginal gains, the strategy used, and the parameters.
    :raises ValueError: If the provided strategy is not recognized.
    """
    if strategy == Strategy.MMR:
        return mmr(embeddings, scores, k, **kwargs)
    if strategy == Strategy.MSD:
        return msd(embeddings, scores, k, **kwargs)
    if strategy == Strategy.COVER:
        return cover(embeddings, scores, k, **kwargs)
    if strategy == Strategy.DPP:
        return dpp(embeddings, scores, k, **kwargs)
    raise ValueError(f"Unknown strategy: {strategy}")
