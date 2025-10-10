import numpy as np
import pytest
from pyversity import Metric
from pyversity.utils import (
    normalize_rows,
    pairwise_similarity,
    prepare_inputs,
    vector_similarity,
)


def test_normalize_rows() -> None:
    """Test row normalization."""
    X = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    Xn = normalize_rows(X)
    # Check that the non-zero row is normalized
    assert np.allclose(np.linalg.norm(Xn[0]), 1.0, atol=1e-6)
    # Check that the zero row remains zero
    assert np.allclose(Xn[1], [0.0, 0.0])


def test_prepare_inputs() -> None:
    """Test input preparation and validation."""
    scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    emb = np.eye(3, dtype=np.float32)
    embeddings, relevance_scores, k_clamped, early = prepare_inputs(emb, scores, k=5)
    assert relevance_scores.shape == (3,) and embeddings.shape == (3, 3) and k_clamped == 3 and early is False

    with pytest.raises(ValueError):
        prepare_inputs(emb, scores[:2], k=2)

    # Early exit case with k=0
    _, _, k0, early0 = prepare_inputs(emb, scores, k=0)
    assert k0 == 0 and early0 is True

    # Early exit case with empty input
    _, _, k1, early1 = prepare_inputs(np.empty((0, 3)), np.array([]), k=2)
    assert k1 == 0 and early1 is True


def test_vector_and_pairwise_similarity(sim_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test vector and pairwise similarity computations."""
    matrix, query_vector = sim_data

    # Vector-to-vector similarity
    sim_dot = vector_similarity(matrix, query_vector, Metric.DOT)
    assert np.all(sim_dot >= 0)

    sim_cos = vector_similarity(matrix, query_vector, Metric.COSINE)
    assert np.all(sim_cos >= 0) and np.all(sim_cos <= 1.0)

    # Pairwise similarity between rows
    pair_dot = pairwise_similarity(matrix, Metric.DOT)
    assert pair_dot.shape == (3, 3) and np.all(pair_dot >= 0)

    pair_cos = pairwise_similarity(matrix, Metric.COSINE)
    assert pair_cos.shape == (3, 3)
    assert np.all(pair_cos >= 0) and np.all(pair_cos <= 1.0 + 1e-6)
