import numpy as np
import pytest
from pyversity import Strategy, diversify, rxquad, xquad
from pyversity.datatypes import DiversificationResult
from pyversity.utils import EPS32, normalize_rows

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_aspect_setup() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a minimal 2-aspect test fixture.

    4 items split across 2 orthogonal aspects.
    Items 0,1 match aspect A; items 2,3 match aspect B.
    Items 0,1 have higher relevance scores.
    """
    embeddings = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float32,
    )
    scores = np.array([1.0, 0.9, 0.3, 0.2], dtype=np.float32)
    aspects = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    return embeddings, scores, aspects


# ---------------------------------------------------------------------------
# xQuAD
# ---------------------------------------------------------------------------


def test_xquad_basic_shape() -> None:
    """Result has correct length and strategy tag."""
    emb, scores, aspects = _two_aspect_setup()
    res = xquad(emb, scores, k=3, diversity=0.5, aspect_embeddings=aspects)
    assert isinstance(res, DiversificationResult)
    assert res.indices.shape == (3,)
    assert res.selection_scores.shape == (3,)
    assert res.strategy == Strategy.XQUAD
    assert res.diversity == 0.5
    assert len(set(res.indices.tolist())) == 3  # no duplicates


def test_xquad_pure_relevance() -> None:
    """diversity=0.0 selects top-k by score, ignoring aspects."""
    emb, scores, aspects = _two_aspect_setup()
    res = xquad(emb, scores, k=2, diversity=0.0, aspect_embeddings=aspects)
    assert set(res.indices.tolist()) == {0, 1}


def test_xquad_pure_diversity_covers_both_aspects() -> None:
    """diversity=1.0 distributes selections across aspects."""
    emb, scores, aspects = _two_aspect_setup()
    res = xquad(emb, scores, k=2, diversity=1.0, aspect_embeddings=aspects)
    indices = set(res.indices.tolist())
    assert indices & {0, 1}  # at least one from aspect-A group
    assert indices & {2, 3}  # at least one from aspect-B group


def test_xquad_uniform_weights_equals_no_weights() -> None:
    """Explicit uniform weights produce the same result as omitting aspect_weights."""
    emb, scores, aspects = _two_aspect_setup()
    uniform = np.array([0.5, 0.5], dtype=np.float32)
    res_none = xquad(emb, scores, k=3, diversity=0.5, aspect_embeddings=aspects)
    res_uniform = xquad(emb, scores, k=3, diversity=0.5, aspect_embeddings=aspects, aspect_weights=uniform)
    assert np.array_equal(res_none.indices, res_uniform.indices)


def test_xquad_unnormalized_weights_same_result() -> None:
    """Aspect weights are normalized internally; scaling them should not change the result."""
    emb, scores, aspects = _two_aspect_setup()
    w1 = np.array([1.0, 3.0], dtype=np.float32)
    w2 = np.array([0.25, 0.75], dtype=np.float32)
    res1 = xquad(emb, scores, k=2, diversity=0.6, aspect_embeddings=aspects, aspect_weights=w1)
    res2 = xquad(emb, scores, k=2, diversity=0.6, aspect_embeddings=aspects, aspect_weights=w2)
    assert np.array_equal(res1.indices, res2.indices)


def test_xquad_aspect_weighting_biases_coverage() -> None:
    """High weight on one aspect biases selection toward that aspect's items."""
    emb, scores, aspects = _two_aspect_setup()
    # Strong bias toward aspect B (items 2,3) — even though items 0,1 are more relevant
    w_b = np.array([0.01, 0.99], dtype=np.float32)
    res = xquad(emb, scores, k=2, diversity=1.0, aspect_embeddings=aspects, aspect_weights=w_b)
    # At pure diversity both picks should come from the high-weight aspect B side
    assert set(res.indices.tolist()) & {2, 3}


def test_xquad_no_duplicate_indices() -> None:
    """Selected indices must be unique."""
    emb, scores, aspects = _two_aspect_setup()
    res = xquad(emb, scores, k=4, diversity=0.5, aspect_embeddings=aspects)
    assert len(res.indices) == len(set(res.indices.tolist()))


def test_xquad_k_larger_than_n_clamped() -> None:
    """K is clamped to n_items; no crash or out-of-bounds."""
    emb, scores, aspects = _two_aspect_setup()
    res = xquad(emb, scores, k=100, diversity=0.5, aspect_embeddings=aspects)
    assert res.indices.shape[0] == emb.shape[0]


def test_xquad_early_exit_empty_embeddings() -> None:
    """Empty candidate set returns empty arrays."""
    emb = np.empty((0, 4), dtype=np.float32)
    scores = np.array([], dtype=np.float32)
    aspects = np.ones((2, 4), dtype=np.float32)
    res = xquad(emb, scores, k=5, diversity=0.5, aspect_embeddings=aspects)
    assert res.indices.size == 0
    assert res.selection_scores.size == 0


def test_xquad_early_exit_k_zero() -> None:
    """k=0 returns empty arrays."""
    emb, scores, aspects = _two_aspect_setup()
    res = xquad(emb, scores, k=0, diversity=0.5, aspect_embeddings=aspects)
    assert res.indices.size == 0


def test_xquad_invalid_diversity() -> None:
    """Diversity outside [0, 1] raises ValueError."""
    emb, scores, aspects = _two_aspect_setup()
    with pytest.raises(ValueError):
        xquad(emb, scores, k=2, diversity=1.5, aspect_embeddings=aspects)
    with pytest.raises(ValueError):
        xquad(emb, scores, k=2, diversity=-0.1, aspect_embeddings=aspects)


def test_xquad_aspect_embeddings_shape_mismatch() -> None:
    """aspect_embeddings with wrong dim raises ValueError."""
    emb, scores, _ = _two_aspect_setup()
    bad_aspects = np.ones((2, 5), dtype=np.float32)  # dim=5, should be 2
    with pytest.raises(ValueError):
        xquad(emb, scores, k=2, diversity=0.5, aspect_embeddings=bad_aspects)


# ---------------------------------------------------------------------------
# RxQuAD
# ---------------------------------------------------------------------------


def test_rxquad_basic_shape() -> None:
    """Result has correct length and strategy tag."""
    emb, scores, aspects = _two_aspect_setup()
    res = rxquad(emb, scores, k=2, diversity=0.5, aspect_embeddings=aspects)
    assert res.indices.shape == (2,)
    assert res.strategy == Strategy.RXQUAD
    assert len(set(res.indices.tolist())) == 2


def test_rxquad_matches_xquad_with_inferred_weights() -> None:
    """
    Verify RxQuAD weight inference formula exactly.

    RxQuAD should produce the same result as xQuAD when the weights are manually
    computed via: w(c) ∝ scores @ doc_aspect_sim.
    """
    emb, scores, aspects = _two_aspect_setup()

    emb_norm = normalize_rows(np.asarray(emb, dtype=np.float32))
    asp_norm = normalize_rows(np.asarray(aspects, dtype=np.float32))
    doc_aspect_sim = np.clip(emb_norm @ asp_norm.T, 0.0, 1.0)
    raw_weights = scores @ doc_aspect_sim
    raw_weights = np.maximum(raw_weights, 0.0)
    raw_weights /= raw_weights.sum() + EPS32

    res_rx = rxquad(emb, scores, k=2, diversity=0.5, aspect_embeddings=aspects)
    res_x = xquad(emb, scores, k=2, diversity=0.5, aspect_embeddings=aspects, aspect_weights=raw_weights)
    assert np.array_equal(res_rx.indices, res_x.indices)


def test_rxquad_high_relevance_aspect_gets_more_weight() -> None:
    """
    Verify inferred weight skew when retrieval is dominated by one aspect.

    When most high-scoring docs match aspect A, RxQuAD should effectively
    weight aspect A higher, biasing coverage toward A.
    """
    # Items 0-2 match aspect A strongly; item 3 matches aspect B.
    # Scores drop off: item 0 is most relevant, item 3 least.
    emb = np.array(
        [[1.0, 0.0], [0.95, 0.05], [0.9, 0.1], [0.0, 1.0]],
        dtype=np.float32,
    )
    scores = np.array([1.0, 0.9, 0.8, 0.1], dtype=np.float32)
    aspects = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    emb_norm = normalize_rows(emb)
    asp_norm = normalize_rows(aspects)
    doc_aspect_sim = np.clip(emb_norm @ asp_norm.T, 0.0, 1.0)
    inferred = scores @ doc_aspect_sim
    inferred /= inferred.sum() + EPS32

    # Aspect A should receive a higher inferred weight than aspect B
    assert inferred[0] > inferred[1]


def test_rxquad_pure_relevance() -> None:
    """diversity=0.0 selects top-k by score regardless of aspects."""
    emb, scores, aspects = _two_aspect_setup()
    res = rxquad(emb, scores, k=2, diversity=0.0, aspect_embeddings=aspects)
    assert set(res.indices.tolist()) == {0, 1}


def test_rxquad_early_exit() -> None:
    """Empty candidate set returns empty arrays."""
    emb = np.empty((0, 4), dtype=np.float32)
    scores = np.array([], dtype=np.float32)
    aspects = np.ones((2, 4), dtype=np.float32)
    res = rxquad(emb, scores, k=5, diversity=0.5, aspect_embeddings=aspects)
    assert res.indices.size == 0


def test_rxquad_invalid_diversity() -> None:
    """Diversity outside [0, 1] raises ValueError."""
    emb, scores, aspects = _two_aspect_setup()
    with pytest.raises(ValueError):
        rxquad(emb, scores, k=2, diversity=2.0, aspect_embeddings=aspects)


# ---------------------------------------------------------------------------
# diversify() dispatcher
# ---------------------------------------------------------------------------


def test_diversify_xquad_missing_aspect_embeddings_raises() -> None:
    """diversify() with strategy=XQUAD and no aspect_embeddings raises a clear ValueError."""
    emb, scores, _ = _two_aspect_setup()
    with pytest.raises(ValueError, match="aspect_embeddings"):
        diversify(emb, scores, k=2, strategy=Strategy.XQUAD)


def test_diversify_rxquad_missing_aspect_embeddings_raises() -> None:
    """diversify() with strategy=RXQUAD and no aspect_embeddings raises a clear ValueError."""
    emb, scores, _ = _two_aspect_setup()
    with pytest.raises(ValueError, match="aspect_embeddings"):
        diversify(emb, scores, k=2, strategy=Strategy.RXQUAD)


def test_diversify_xquad_via_dispatcher() -> None:
    """diversify() correctly dispatches to xquad."""
    emb, scores, aspects = _two_aspect_setup()
    res = diversify(emb, scores, k=2, strategy=Strategy.XQUAD, aspect_embeddings=aspects)
    assert res.strategy == Strategy.XQUAD
    assert res.indices.shape == (2,)


def test_diversify_rxquad_via_dispatcher() -> None:
    """diversify() correctly dispatches to rxquad."""
    emb, scores, aspects = _two_aspect_setup()
    res = diversify(emb, scores, k=2, strategy=Strategy.RXQUAD, aspect_embeddings=aspects)
    assert res.strategy == Strategy.RXQUAD
    assert res.indices.shape == (2,)


def test_diversify_xquad_string_strategy() -> None:
    """diversify() accepts the string 'xquad' as strategy."""
    emb, scores, aspects = _two_aspect_setup()
    res = diversify(emb, scores, k=2, strategy="xquad", aspect_embeddings=aspects)
    assert res.strategy == Strategy.XQUAD
