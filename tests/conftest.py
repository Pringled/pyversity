import numpy as np
import pytest


@pytest.fixture
def near_dups() -> tuple[np.ndarray, np.ndarray]:
    """Embeddings with near-duplicates and their scores."""
    emb = np.array(
        [
            [1.0, 0.0],
            [0.999, 0.001],  # ~same as 0
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([1.0, 0.99, 0.98], dtype=np.float32)
    return emb, scores


@pytest.fixture
def sim_data() -> tuple[np.ndarray, np.ndarray]:
    """Data for similarity tests: 3 samples and a query vector."""
    X = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    v = np.array([1.0, 0.5], dtype=np.float32)
    return X, v
