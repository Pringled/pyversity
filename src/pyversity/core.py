from typing import Any, Literal, overload

import numpy as np

from pyversity.datatypes import Strategy
from pyversity.strategies import cover, dpp, mmr, msd


@overload
def diversify(
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    strategy: Strategy = Strategy.MMR,
    return_gains: Literal[True] = True,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def diversify(
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    strategy: Strategy = Strategy.MMR,
    return_gains: Literal[False] = False,
    **kwargs: Any,
) -> np.ndarray: ...


def diversify(
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    strategy: Strategy = Strategy.MMR,
    return_gains: bool = False,
    **kwargs: Any,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Diversify a retrieval result using a selected strategy.

    :param embeddings: Embeddings of the items to be diversified.
    :param scores: Scores (relevances) of the items to be diversified.
    :param k: The number of items to select for the diversified result.
    :param strategy: The diversification strategy to apply.
      Supported strategies are: 'mmr' (default), 'msd', 'cover', and 'dpp'.
    :param return_gains: Whether to return the marginal gains along with the indices.
    :param **kwargs: Additional keyword arguments passed to the specific strategy function.
    :return: The indicies of the selected items,
      or a tuple containing an array of indices of the selected items
      and an array of corresponding relevance scores for the selected items if `return_gains` is True.
    :raises ValueError: If the provided strategy is not recognized.
    """
    if strategy == Strategy.MMR:
        return mmr(embeddings=embeddings, scores=scores, k=k, return_gains=return_gains, **kwargs)  # type: ignore[call-overload]
    if strategy == Strategy.MSD:
        return msd(embeddings=embeddings, scores=scores, k=k, return_gains=return_gains, **kwargs)  # type: ignore[call-overload]
    if strategy == Strategy.COVER:
        return cover(embeddings=embeddings, scores=scores, k=k, return_gains=return_gains, **kwargs)  # type: ignore[call-overload]
    if strategy == Strategy.DPP:
        return dpp(embeddings=embeddings, scores=scores, k=k, return_gains=return_gains, **kwargs)  # type: ignore[call-overload]
    raise ValueError(f"Unknown strategy: {strategy}")
