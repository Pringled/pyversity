from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset as hf_load_dataset
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DatasetType(Enum):
    """Known dataset types."""

    MOVIELENS = "movielens"
    LASTFM = "lastfm"
    HUGGINGFACE = "huggingface"


@dataclass
class DatasetInfo:
    """Dataset configuration."""

    name: str
    path: str  # Local path or hf://... identifier
    dataset_type: DatasetType
    rating_threshold: float
    download_url: str | None = None


# Registry of known datasets
DATASET_REGISTRY: dict[str, DatasetInfo] = {
    "ml-32m": DatasetInfo(
        name="ml-32m",
        path="benchmarks/data/ml-32m",
        dataset_type=DatasetType.MOVIELENS,
        rating_threshold=4.0,
        download_url="https://files.grouplens.org/datasets/movielens/ml-32m.zip",
    ),
    "lastfm": DatasetInfo(
        name="lastfm",
        path="benchmarks/data/lastfm",
        dataset_type=DatasetType.LASTFM,
        rating_threshold=1.0,
        download_url="http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip",
    ),
    "amazon-video-games": DatasetInfo(
        name="amazon-video-games",
        path="hf://LoganKells/amazon_product_reviews_video_games",
        dataset_type=DatasetType.HUGGINGFACE,
        rating_threshold=4.0,
    ),
    "goodreads": DatasetInfo(
        name="goodreads",
        path="hf://qmaruf/goodreads-rating",
        dataset_type=DatasetType.HUGGINGFACE,
        rating_threshold=4.0,
    ),
}


@dataclass
class InteractionData:
    """Preprocessed interaction data."""

    user_ids: NDArray[np.int64]
    item_ids: NDArray[np.int64]
    n_users: int
    n_items: int


def load_dataset(
    dataset: str | DatasetInfo,
    min_interactions: int = 5,
    rating_threshold: float | None = None,
) -> InteractionData:
    """Load and preprocess a dataset from the registry."""
    # Resolve dataset info
    if isinstance(dataset, str):
        if dataset not in DATASET_REGISTRY:
            msg = f"Unknown dataset: {dataset}. Known: {list(DATASET_REGISTRY.keys())}"
            raise ValueError(msg)
        info = DATASET_REGISTRY[dataset]
    else:
        info = dataset

    threshold = rating_threshold if rating_threshold is not None else info.rating_threshold

    # Load based on type
    if info.dataset_type == DatasetType.HUGGINGFACE:
        df = _load_huggingface(info.path.replace("hf://", ""))
    elif info.dataset_type == DatasetType.MOVIELENS:
        df = _load_movielens(Path(info.path))
    elif info.dataset_type == DatasetType.LASTFM:
        df = _load_lastfm(Path(info.path))
    else:
        msg = f"Unknown dataset type: {info.dataset_type}"
        raise ValueError(msg)

    logger.debug(f"Loaded {len(df):,} raw interactions")

    # Binarize
    if threshold is not None:
        df = df[df["value"] >= threshold].copy()
        logger.debug(f"After binarization (>= {threshold}): {len(df):,}")

    # Filter by interaction count (iteratively until stable)
    prev_len = 0
    while len(df) != prev_len:
        prev_len = len(df)
        user_counts = df["user"].value_counts()
        item_counts = df["item"].value_counts()
        df = df[df["user"].isin(user_counts[user_counts >= min_interactions].index)]
        df = df[df["item"].isin(item_counts[item_counts >= min_interactions].index)]

    logger.debug(f"After filtering (min {min_interactions}): {len(df):,}")

    # Remap IDs to contiguous integers
    df["user"] = pd.factorize(df["user"])[0]
    df["item"] = pd.factorize(df["item"])[0]

    n_users = df["user"].nunique()
    n_items = df["item"].nunique()
    logger.debug(f"Final: {n_users:,} users, {n_items:,} items, {len(df):,} interactions")

    return InteractionData(
        user_ids=df["user"].values.astype(np.int64),
        item_ids=df["item"].values.astype(np.int64),
        n_users=n_users,
        n_items=n_items,
    )


def _load_movielens(path: Path) -> pd.DataFrame:
    """Load MovieLens dataset."""
    df = pd.read_csv(path / "ratings.csv")
    return df.rename(columns={"userId": "user", "movieId": "item", "rating": "value"})[["user", "item", "value"]]


def _load_lastfm(path: Path) -> pd.DataFrame:
    """Load Last.FM HetRec 2011 dataset."""
    df = pd.read_csv(path / "user_artists.dat", sep="\t")
    return df.rename(columns={"userID": "user", "artistID": "item", "weight": "value"})[["user", "item", "value"]]


def _load_huggingface(dataset_id: str) -> pd.DataFrame:
    """Load a dataset from HuggingFace Hub."""
    logger.debug(f"Loading from HuggingFace: {dataset_id}")
    ds = hf_load_dataset(dataset_id, split="train")
    df = ds.to_pandas()

    # Amazon Video Games format
    if "reviewerID" in df.columns:
        df = df.rename(columns={"reviewerID": "user", "asin": "item", "overall": "value"})
        df = df[df["value"] > 0]
        df["user"] = pd.factorize(df["user"])[0]
        df["item"] = pd.factorize(df["item"])[0]
        return df[["user", "item", "value"]]

    # Goodreads format
    if "user_id" in df.columns and "book_id" in df.columns:
        df = df.rename(columns={"user_id": "user", "book_id": "item", "rating": "value"})
        df = df[df["value"] > 0]
        df["user"] = pd.factorize(df["user"])[0]
        return df[["user", "item", "value"]]

    msg = f"Unknown HuggingFace dataset format: {list(df.columns)}"
    raise ValueError(msg)
