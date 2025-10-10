import numpy as np
import pytest
from pyversity.datatypes import Metric
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
    r, E, k_clamped, early = prepare_inputs(scores, emb, k=5)
    assert r.shape == (3,) and E.shape == (3, 3) and k_clamped == 3 and early is False

    with pytest.raises(ValueError):
        prepare_inputs(scores[:2], emb, k=2)

    _, _, k0, early0 = prepare_inputs(scores, emb, k=0)
    assert k0 == 0 and early0 is True

    _, _, k1, early1 = prepare_inputs(np.array([]), np.empty((0, 3)), k=2)
    assert k1 == 0 and early1 is True


def test_vector_and_pairwise_similarity(sim_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test vector and pairwise similarity computations."""
    X, v = sim_data

    s_dot = vector_similarity(X, v, Metric.DOT)
    assert np.all(s_dot >= 0)

    s_cos = vector_similarity(X, v, Metric.COSINE)
    assert np.all(s_cos >= 0) and np.all(s_cos <= 1.0)

    P_dot = pairwise_similarity(X, Metric.DOT)
    assert P_dot.shape == (3, 3) and np.all(P_dot >= 0)

    P_cos = pairwise_similarity(X, Metric.COSINE)
    assert P_cos.shape == (3, 3) and np.all(P_cos <= 1.0 + 1e-6) and np.all(P_cos >= 0)
