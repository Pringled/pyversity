from typing import Any

import numpy as np

from pyversity.datatypes import DiversificationResult, Strategy
from pyversity.strategies import cover, dpp, mmr, msd


def diversify(
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    strategy: Strategy = Strategy.MMR,
    diversity: float = 0.5,
    **kwargs: Any,
) -> DiversificationResult:
    """
    Diversify a retrieval result using a selected strategy.

    :param embeddings: Embeddings of the items to be diversified.
    :param scores: Scores (relevances) of the items to be diversified.
    :param k: The number of items to select for the diversified result.
    :param strategy: The diversification strategy to apply.
      Supported strategies are: 'mmr' (default), 'msd', 'cover', and 'dpp'.
    :param diversity: Diversity parameter (range of [0, 1]). Higher values prioritize diversity and lower values prioritize relevance.
    :param **kwargs: Additional keyword arguments passed to the specific strategy function.
    :return: A DiversificationResult containing the selected item indices,
      their selection scores, the strategy used, and the parameters.
    :raises ValueError: If the provided strategy is not recognized.
    """
    if strategy == Strategy.MMR:
        return mmr(embeddings, scores, k, diversity, **kwargs)
    if strategy == Strategy.MSD:
        return msd(embeddings, scores, k, diversity, **kwargs)
    if strategy == Strategy.COVER:
        return cover(embeddings, scores, k, diversity, **kwargs)
    if strategy == Strategy.DPP:
        return dpp(embeddings, scores, k, diversity, **kwargs)
    raise ValueError(f"Unknown strategy: {strategy}")
