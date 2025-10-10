from enum import Enum


class Strategy(str, Enum):
    MMR = "mmr"
    MSD = "msd"
    COVER = "cover"
    DPP = "dpp"


class Metric(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
