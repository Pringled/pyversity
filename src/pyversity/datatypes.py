from dataclasses import dataclass
from enum import Enum

import numpy as np


class Strategy(str, Enum):
    """Supported diversification strategies."""

    MMR = "mmr"
    MSD = "msd"
    COVER = "cover"
    DPP = "dpp"


class Metric(str, Enum):
    """Supported similarity metrics."""

    COSINE = "cosine"
    DOT = "dot"


@dataclass
class DiversificationResult:
    """
    Result of a diversification operation.

    Attributes
    ----------
        indices: Diversified item indices.
        selection_scores: Selection scores for the diversified items.
        strategy: Diversification strategy used.
        diversity: Diversity parameter used in the strategy.
        parameters: Additional parameters used in the strategy.

    """

    indices: np.ndarray
    selection_scores: np.ndarray
    strategy: Strategy
    diversity: float
    parameters: dict | None = None
