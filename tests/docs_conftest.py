"""
Fixtures for testing the README examples.

DO NOT USE ANY OF THESE IN OTHER TESTS!
"""

from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

_rng = np.random.default_rng(0)

_N = 20  # number of candidate items
_D = 16  # embedding dimension
_R = 5  # number of recent items


def _emb(n: int = _N) -> np.ndarray:
    return _rng.standard_normal((n, _D)).astype(np.float64)


def _scores(n: int = _N) -> np.ndarray:
    raw = _rng.random(n).astype(np.float64)
    return raw / raw.sum()


# --- benchmarks/README.md: Programmatic API ---


@pytest.fixture(scope="session")
def benchmark_data(tmp_path_factory: pytest.TempPathFactory) -> Generator[Path, None, None]:
    """
    Tiny synthetic MovieLens-format dataset for testing the benchmark API.

    Patches DATASET_REGISTRY["ml-32m"] to point at a temp directory containing
    a minimal ratings.csv, and returns a temp output directory to use as
    BenchmarkConfig.output_dir so the test never writes to the real results tree.
    """
    from benchmarks.core.data import DATASET_REGISTRY

    # Create fake ratings.csv: 100 users × 20 items each, rating=4.5 (above 4.0 threshold)
    rng = np.random.default_rng(0)
    n_users, n_items, per_user = 100, 50, 20
    user_ids = np.repeat(np.arange(1, n_users + 1), per_user)
    item_ids = np.concatenate([rng.choice(n_items, size=per_user, replace=False) + 1 for _ in range(n_users)])
    df = pd.DataFrame({"userId": user_ids, "movieId": item_ids, "rating": 4.5})

    data_dir: Path = tmp_path_factory.mktemp("ml-32m")
    df.to_csv(data_dir / "ratings.csv", index=False)

    out_dir: Path = tmp_path_factory.mktemp("benchmark_results")

    original_path = DATASET_REGISTRY["ml-32m"].path
    DATASET_REGISTRY["ml-32m"].path = str(data_dir)

    yield out_dir

    DATASET_REGISTRY["ml-32m"].path = original_path


# --- Product / Web Search (test_README_2) ---


@pytest.fixture
def item_embeddings() -> np.ndarray:
    """Item embeddings for testing the product/web search example."""
    return _emb()


@pytest.fixture
def item_scores() -> np.ndarray:
    """Item scores for testing the product/web search example."""
    return _scores()


# --- Literature Search (test_README_3) ---


@pytest.fixture
def paper_embeddings() -> np.ndarray:
    """Paper embeddings for testing the literature search example."""
    return _emb()


@pytest.fixture
def paper_scores() -> np.ndarray:
    """Paper scores for testing the literature search example."""
    return _scores()


# --- Conversational RAG (test_README_4) ---


@pytest.fixture
def chunk_embeddings() -> np.ndarray:
    """Chunk embeddings for testing the conversational RAG example."""
    return _emb()


@pytest.fixture
def chunk_scores() -> np.ndarray:
    """Chunk scores for testing the conversational RAG example."""
    return _scores()


@pytest.fixture
def recent_chunk_embeddings() -> np.ndarray:
    """Recent chunk embeddings for testing the conversational RAG example."""
    return _emb(_R)


# --- Infinite Scroll / Recommendation Feed (test_README_5) ---


@pytest.fixture
def feed_embeddings() -> np.ndarray:
    """Feed item embeddings for testing the infinite scroll / recommendation feed example."""
    return _emb()


@pytest.fixture
def feed_scores() -> np.ndarray:
    """Feed item scores for testing the infinite scroll / recommendation feed example."""
    return _scores()


@pytest.fixture
def recent_feed_embeddings() -> np.ndarray:
    """Recent feed item embeddings for testing the infinite scroll / recommendation feed example."""
    return _emb(_R)


# --- Single Long Document (test_README_6) ---


@pytest.fixture
def doc_chunk_embeddings() -> np.ndarray:
    """Document chunk embeddings for testing the single long document example."""
    return _emb()


@pytest.fixture
def doc_chunk_scores() -> np.ndarray:
    """Document chunk scores for testing the single long document example."""
    return _scores()


# --- Ambiguous Queries / xQuAD (test_README_7, test_README_8) ---

_A = 4  # number of aspects


@pytest.fixture
def article_embeddings() -> np.ndarray:
    """Article embeddings for testing the xQuAD / RxQuAD example."""
    return _emb()


@pytest.fixture
def article_scores() -> np.ndarray:
    """Article scores for testing the xQuAD / RxQuAD example."""
    return _scores()


@pytest.fixture
def aspect_embeddings() -> np.ndarray:
    """Aspect embeddings (one per user intent) for testing the xQuAD / RxQuAD example."""
    return _emb(_A)
