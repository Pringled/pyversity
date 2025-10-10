from enum import Enum


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
