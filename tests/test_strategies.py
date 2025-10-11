from typing import Any, Callable

import numpy as np
import pytest
from pyversity import Metric, Strategy, cover, diversify, dpp, mmr, msd
from pyversity.datatypes import DiversificationResult


def test_mmr() -> None:
    """Test MMR strategy with various diversity settings (1=diverse, 0=relevance)."""
    # Relevance-only (diversity=0): picks top-k by scores
    emb = np.eye(5, dtype=np.float32)
    scores = np.array([0.1, 0.9, 0.3, 0.8, 0.2], dtype=np.float32)
    res = mmr(emb, scores, k=3, diversity=0.0, metric=Metric.COSINE, normalize=True)
    expected = np.array([1, 3, 2], dtype=np.int32)
    assert np.array_equal(res.indices, expected)
    assert np.allclose(res.selection_scores, scores[expected])

    # Strong diversity (diversity=1): avoid near-duplicate
    emb = np.array([[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]], dtype=np.float32)
    scores = np.array([1.0, 0.99, 0.98], dtype=np.float32)
    res = mmr(emb, scores, k=2, diversity=1.0, metric=Metric.COSINE, normalize=True)
    assert res.indices[0] == 0 and res.indices[1] == 2

    # Balanced (diversity=0.5): picks mix of relevance and diversity
    res = mmr(emb, scores, k=2, diversity=0.5, metric=Metric.COSINE, normalize=True)
    assert res.indices[0] == 0 and res.indices[1] == 2

    # Bounds check
    with pytest.raises(ValueError):
        mmr(np.eye(2, dtype=np.float32), np.array([1.0, 0.5], dtype=np.float32), k=1, diversity=-0.1)


def test_msd() -> None:
    """Test MSD strategy with various diversity settings (1=diverse, 0=relevance)."""
    # Relevance-only (diversity=0): picks top-k by scores
    emb = np.eye(4, dtype=np.float32)
    scores = np.array([0.5, 0.2, 0.9, 0.1], dtype=np.float32)
    res = msd(emb, scores, k=2, diversity=0.0, metric=Metric.COSINE, normalize=True)
    assert np.array_equal(res.indices, np.array([2, 0], dtype=np.int32))

    # Strong diversity (diversity=1): picks most dissimilar
    emb = np.array([[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]], dtype=np.float32)
    scores = np.array([1.0, 0.99, 0.98], dtype=np.float32)
    res = msd(emb, scores, k=2, diversity=1.0, metric=Metric.COSINE, normalize=True)
    assert res.indices[0] == 0 and res.indices[1] == 2

    # Balanced (diversity=0.5): picks mix of relevance and diversity
    res = msd(emb, scores, k=2, diversity=0.5, metric=Metric.COSINE, normalize=True)
    assert res.indices[0] == 0 and res.indices[1] == 2

    # Bounds check
    with pytest.raises(ValueError):
        msd(np.eye(2, dtype=np.float32), np.array([1.0, 0.5], dtype=np.float32), k=1, diversity=1.1)


def test_cover() -> None:
    """Test COVER strategy with various diversity and gamma settings (1=diverse, 0=relevance)."""
    emb = np.eye(3, dtype=np.float32)
    scores = np.array([0.1, 0.8, 0.3], dtype=np.float32)

    # Relevance-only (diversity=0): picks top-k by scores
    res = cover(emb, scores, k=2, diversity=0.0)
    expected = np.array([1, 2], dtype=np.int32)
    assert np.array_equal(res.indices, expected)
    assert np.allclose(res.selection_scores, scores[expected])

    # Balanced coverage (diversity=0.5, gamma=0.5): picks diverse set
    res = cover(emb, scores, k=2, diversity=0.5, gamma=0.5)
    assert res.indices[0] == 1 and res.indices[1] in (0, 2)

    # Parameter validation
    with pytest.raises(ValueError):
        cover(emb, scores, k=2, diversity=-0.01)
    with pytest.raises(ValueError):
        cover(emb, scores, k=2, diversity=1.01)
    with pytest.raises(ValueError):
        cover(emb, scores, k=2, gamma=0.0)
    with pytest.raises(ValueError):
        cover(emb, scores, k=2, gamma=-0.5)


def test_dpp() -> None:
    """Test DPP strategy with various diversity settings (1=diverse, 0=relevance)."""
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    scores = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    # Strong diversity (diversity=1)
    res = dpp(emb, scores, k=2, diversity=1.0)
    assert 1 <= res.indices.size <= 2
    assert np.all(res.selection_scores >= -1e-7)
    assert np.all(res.selection_scores[:-1] + 1e-7 >= res.selection_scores[1:])

    # Balanced (diversity=0.5)
    res = dpp(emb, scores, k=2, diversity=0.5)
    assert 1 <= res.indices.size <= 2
    assert np.all(res.selection_scores >= -1e-7)
    assert np.all(res.selection_scores[:-1] + 1e-7 >= res.selection_scores[1:])

    # Low diversity (diversity=0.0): more relevance-driven
    res = dpp(emb, scores, k=2, diversity=0.0)
    assert 1 <= res.indices.size <= 2
    assert np.all(res.selection_scores >= -1e-7)
    assert np.all(res.selection_scores[:-1] + 1e-7 >= res.selection_scores[1:])

    # Early exit on empty input
    res = dpp(np.empty((0, 3), dtype=np.float32), np.array([], dtype=np.float32), k=3)
    assert res.indices.size == 0 and res.selection_scores.size == 0


@pytest.mark.parametrize(
    "strategy, fn, kwargs",
    [
        (Strategy.MMR, mmr, {"diversity": 0.5, "metric": Metric.COSINE, "normalize": True}),
        (Strategy.MSD, msd, {"diversity": 0.5, "metric": Metric.COSINE, "normalize": True}),
        (Strategy.COVER, cover, {"diversity": 0.5, "gamma": 0.5}),
        (Strategy.DPP, dpp, {"diversity": 0.5}),
    ],
)
def test_diversify(strategy: Strategy, fn: Callable[..., DiversificationResult], kwargs: Any) -> None:
    """Test the diversify dispatcher against direct calls (1=diverse, 0=relevance)."""
    emb = np.eye(4, dtype=np.float32)
    scores = np.array([0.3, 0.7, 0.1, 0.5], dtype=np.float32)

    # direct call
    res_direct = fn(emb, scores, k=2, **kwargs)

    # dispatcher call
    res_disp = diversify(embeddings=emb, scores=scores, k=2, strategy=strategy, **kwargs)

    assert np.array_equal(res_direct.indices, res_disp.indices)
    assert np.allclose(res_direct.selection_scores, res_disp.selection_scores)
