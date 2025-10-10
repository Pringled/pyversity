from typing import Any, Callable

import numpy as np
import pytest
from pyversity import Metric, Strategy, cover, diversify, dpp, mmr, msd
from pyversity.datatypes import DiversificationResult


def test_mmr() -> None:
    """Test MMR strategy with various lambda settings."""
    # Pure relevance (lambda=1): picks top-k by scores
    emb = np.eye(5, dtype=np.float32)
    scores = np.array([0.1, 0.9, 0.3, 0.8, 0.2], dtype=np.float32)
    res = mmr(emb, scores, k=3, lambda_param=1.0, metric=Metric.COSINE, normalize=True)
    expected = np.array([1, 3, 2], dtype=np.int32)
    assert np.array_equal(res.indices, expected)
    assert np.allclose(res.marginal_gains, scores[expected])

    # Strong diversity (lambda=0): avoid near-duplicate
    emb = np.array([[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]], dtype=np.float32)
    scores = np.array([1.0, 0.99, 0.98], dtype=np.float32)
    res = mmr(emb, scores, k=2, lambda_param=0.0, metric=Metric.COSINE, normalize=True)
    assert res.indices[0] == 0 and res.indices[1] == 2

    # Balanced (lambda=0.5): picks mix of relevance and diversity
    res = mmr(emb, scores, k=2, lambda_param=0.5, metric=Metric.COSINE, normalize=True)
    assert res.indices[0] == 0 and res.indices[1] == 2

    # Bounds check
    with pytest.raises(ValueError):
        mmr(np.eye(2, dtype=np.float32), np.array([1.0, 0.5], dtype=np.float32), k=1, lambda_param=-0.1)


def test_msd() -> None:
    """Test MSD strategy with various lambda settings."""
    # Pure relevance (lambda=1): picks top-k by scores
    emb = np.eye(4, dtype=np.float32)
    scores = np.array([0.5, 0.2, 0.9, 0.1], dtype=np.float32)
    res = msd(emb, scores, k=2, lambda_param=1.0, metric=Metric.COSINE, normalize=True)
    assert np.array_equal(res.indices, np.array([2, 0], dtype=np.int32))

    # Strong diversity (lambda=0): picks most dissimilar
    emb = np.array([[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]], dtype=np.float32)
    scores = np.array([1.0, 0.99, 0.98], dtype=np.float32)
    res = msd(emb, scores, k=2, lambda_param=0.0, metric=Metric.COSINE, normalize=True)
    assert res.indices[0] == 0 and res.indices[1] == 2

    # Balanced (lambda=0.5): picks mix of relevance and diversity
    res = msd(emb, scores, k=2, lambda_param=0.5, metric=Metric.COSINE, normalize=True)
    assert res.indices[0] == 0 and res.indices[1] == 2

    # Bounds check
    with pytest.raises(ValueError):
        msd(np.eye(2, dtype=np.float32), np.array([1.0, 0.5], dtype=np.float32), k=1, lambda_param=1.1)


def test_cover() -> None:
    """Test COVER strategy with various theta and gamma settings."""
    emb = np.eye(3, dtype=np.float32)
    scores = np.array([0.1, 0.8, 0.3], dtype=np.float32)

    # Pure relevance (theta=1): picks top-k by scores
    res = cover(emb, scores, k=2, theta=1.0)
    expected = np.array([1, 2], dtype=np.int32)
    assert np.array_equal(res.indices, expected)
    assert np.allclose(res.marginal_gains, scores[expected])

    # Balanced coverage (theta=0.5, gamma=0.5): picks diverse set
    res = cover(emb, scores, k=2, theta=0.5, gamma=0.5)
    assert res.indices[0] == 1 and res.indices[1] in (0, 2)

    # Parameter validation
    with pytest.raises(ValueError):
        cover(emb, scores, k=2, theta=-0.01)
    with pytest.raises(ValueError):
        cover(emb, scores, k=2, theta=1.01)
    with pytest.raises(ValueError):
        cover(emb, scores, k=2, gamma=0.0)
    with pytest.raises(ValueError):
        cover(emb, scores, k=2, gamma=-0.5)


def test_dpp() -> None:
    """Test DPP strategy with various beta settings."""
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    scores = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    # Beta=0: ignore relevance, diversity-only kernel
    res = dpp(emb, scores, k=3, beta=0.0)
    assert 1 <= res.indices.size <= 3
    assert np.all(res.marginal_gains >= -1e-7)
    assert np.all(res.marginal_gains[:-1] + 1e-7 >= res.marginal_gains[1:])

    # Strong diversity (beta=1)
    res = dpp(emb, scores, k=2, beta=1.0)
    assert 1 <= res.indices.size <= 2
    assert np.all(res.marginal_gains >= -1e-7)
    assert np.all(res.marginal_gains[:-1] + 1e-7 >= res.marginal_gains[1:])

    # Balanced (beta=0.5)
    res = dpp(emb, scores, k=2, beta=0.5)
    assert 1 <= res.indices.size <= 2
    assert np.all(res.marginal_gains >= -1e-7)
    assert np.all(res.marginal_gains[:-1] + 1e-7 >= res.marginal_gains[1:])

    # Early exit on empty input
    res = dpp(np.empty((0, 3), dtype=np.float32), np.array([], dtype=np.float32), k=3)
    assert res.indices.size == 0 and res.marginal_gains.size == 0


@pytest.mark.parametrize(
    "strategy, fn, kwargs",
    [
        (Strategy.MMR, mmr, {"lambda_param": 0.5, "metric": Metric.COSINE, "normalize": True}),
        (Strategy.MSD, msd, {"lambda_param": 0.5, "metric": Metric.COSINE, "normalize": True}),
        (Strategy.COVER, cover, {"theta": 0.5, "gamma": 0.5}),
        (Strategy.DPP, dpp, {"beta": 0.5}),
    ],
)
def test_diversify(strategy: Strategy, fn: Callable[..., DiversificationResult], kwargs: Any) -> None:
    """Test the diversify function."""
    emb = np.eye(4, dtype=np.float32)
    scores = np.array([0.3, 0.7, 0.1, 0.5], dtype=np.float32)

    # direct call
    res_direct = fn(emb, scores, k=2, **kwargs)

    # dispatcher call
    res_disp = diversify(embeddings=emb, scores=scores, k=2, strategy=strategy, **kwargs)

    assert np.array_equal(res_direct.indices, res_disp.indices)
    assert np.allclose(res_direct.marginal_gains, res_disp.marginal_gains)
