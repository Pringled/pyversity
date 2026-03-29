import numpy as np

from pyversity.datatypes import DiversificationResult, Strategy
from pyversity.utils import EPS32, normalize_rows, prepare_inputs


def _xquad_select(
    scores: np.ndarray,
    k: int,
    diversity: float,
    doc_aspect_sim: np.ndarray,
    aspect_weights: np.ndarray,
    strategy: Strategy,
    normalize: bool,
) -> DiversificationResult:
    """
    Shared greedy selection loop for xQuAD and RxQuAD.

    :param scores: 1D array of relevance scores for each item.
    :param k: Number of items to select.
    :param diversity: Trade-off parameter in [0, 1]. 0.0 = pure relevance, 1.0 = pure intent coverage.
    :param doc_aspect_sim: 2D array of shape (n_items, n_aspects) with clipped cosine similarities.
    :param aspect_weights: 1D array of shape (n_aspects,) summing to 1.0.
    :param strategy: Strategy enum value to embed in the result.
    :param normalize: Whether normalization was applied (stored in parameters).
    :return: DiversificationResult with selected indices and selection scores.
    """
    n_items = scores.shape[0]
    n_aspects = doc_aspect_sim.shape[1]

    # coverage_remainder[c] = Π_{d' selected} (1 - sim(d', c)); starts at 1.0
    coverage_remainder = np.ones(n_aspects, dtype=np.float32)

    selected_mask = np.zeros(n_items, dtype=bool)
    selected_indices = np.empty(k, dtype=np.int32)
    selection_scores = np.empty(k, dtype=np.float32)

    for step in range(k):
        # diversity score for each candidate: Σ_c w(c) * sim(d, c) * coverage_remainder(c)
        diversity_scores = doc_aspect_sim @ (aspect_weights * coverage_remainder)
        combined = (1.0 - diversity) * scores + diversity * diversity_scores
        combined[selected_mask] = -np.inf

        best = int(np.argmax(combined))
        selected_indices[step] = best
        selection_scores[step] = float(combined[best])
        selected_mask[best] = True

        # Shrink coverage remainder: aspects already covered yield diminishing returns
        coverage_remainder *= 1.0 - doc_aspect_sim[best]

    return DiversificationResult(
        indices=selected_indices,
        selection_scores=selection_scores,
        strategy=strategy,
        diversity=diversity,
        parameters={"normalize": normalize},
    )


def xquad(
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    diversity: float = 0.5,
    *,
    aspect_embeddings: np.ndarray,
    aspect_weights: np.ndarray | None = None,
    normalize: bool = True,
) -> DiversificationResult:
    """
    Explicit Query Aspect Diversification (xQuAD).

    Greedily selects k items that collectively cover distinct user intents (aspects),
    proportional to each aspect's estimated importance. At each step, already-covered
    intents yield diminishing returns, so the result naturally distributes across all
    provided aspects.

    :param embeddings: 2D array of shape (n_items, d) — candidate item embeddings.
    :param scores: 1D array of shape (n_items,) — relevance scores.
    :param k: Number of items to select.
    :param diversity: Trade-off parameter in [0, 1]. 0.0 = pure relevance, 1.0 = pure intent coverage.
    :param aspect_embeddings: 2D array of shape (n_aspects, d) — one embedding per user intent. Required.
    :param aspect_weights: 1D array of shape (n_aspects,) — prior probability of each intent P(c|q).
                           If None, uniform weights are used.
    :param normalize: Whether to normalize embeddings before computing similarity.
    :return: A DiversificationResult containing the selected item indices,
      their selection scores, the strategy used, and the parameters.
    :raises ValueError: If diversity is not in [0, 1].
    :raises ValueError: If aspect_embeddings dimensionality does not match embeddings.
    """
    if not (0.0 <= float(diversity) <= 1.0):
        raise ValueError("diversity must be in [0, 1]")

    embeddings, scores, k, early_exit = prepare_inputs(embeddings, scores, k)
    if early_exit:
        return DiversificationResult(
            indices=np.empty(0, np.int32),
            selection_scores=np.empty(0, np.float32),
            strategy=Strategy.XQUAD,
            diversity=diversity,
            parameters={"normalize": normalize},
        )

    aspect_embeddings = np.asarray(aspect_embeddings, dtype=np.float32, order="C")
    if aspect_embeddings.ndim != 2 or aspect_embeddings.shape[1] != embeddings.shape[1]:
        raise ValueError(
            f"aspect_embeddings must be 2D with shape (n_aspects, {embeddings.shape[1]}), "
            f"got {aspect_embeddings.shape}"
        )

    if normalize:
        embeddings = normalize_rows(embeddings)
        aspect_embeddings = normalize_rows(aspect_embeddings)

    n_aspects = aspect_embeddings.shape[0]
    doc_aspect_sim = np.clip(embeddings @ aspect_embeddings.T, 0.0, 1.0)

    if aspect_weights is None:
        weights = np.full(n_aspects, 1.0 / n_aspects, dtype=np.float32)
    else:
        weights = np.asarray(aspect_weights, dtype=np.float32)
        total = weights.sum()
        weights = weights / (total + EPS32)

    return _xquad_select(scores, k, diversity, doc_aspect_sim, weights, Strategy.XQUAD, normalize)


def rxquad(
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    diversity: float = 0.5,
    *,
    aspect_embeddings: np.ndarray,
    normalize: bool = True,
) -> DiversificationResult:
    """
    Relevance-weighted Explicit Query Aspect Diversification (RxQuAD).

    Like xQuAD, but infers aspect importance directly from the retrieval scores
    and aspect-document similarities — no prior over aspect probabilities needed.
    Aspect weights are estimated as: w(c) ∝ Σ_d scores[d] · sim(d, c).

    :param embeddings: 2D array of shape (n_items, d) — candidate item embeddings.
    :param scores: 1D array of shape (n_items,) — relevance scores.
    :param k: Number of items to select.
    :param diversity: Trade-off parameter in [0, 1]. 0.0 = pure relevance, 1.0 = pure intent coverage.
    :param aspect_embeddings: 2D array of shape (n_aspects, d) — one embedding per user intent. Required.
    :param normalize: Whether to normalize embeddings before computing similarity.
    :return: A DiversificationResult containing the selected item indices,
      their selection scores, the strategy used, and the parameters.
    :raises ValueError: If diversity is not in [0, 1].
    :raises ValueError: If aspect_embeddings dimensionality does not match embeddings.
    """
    if not (0.0 <= float(diversity) <= 1.0):
        raise ValueError("diversity must be in [0, 1]")

    embeddings, scores, k, early_exit = prepare_inputs(embeddings, scores, k)
    if early_exit:
        return DiversificationResult(
            indices=np.empty(0, np.int32),
            selection_scores=np.empty(0, np.float32),
            strategy=Strategy.RXQUAD,
            diversity=diversity,
            parameters={"normalize": normalize},
        )

    aspect_embeddings = np.asarray(aspect_embeddings, dtype=np.float32, order="C")
    if aspect_embeddings.ndim != 2 or aspect_embeddings.shape[1] != embeddings.shape[1]:
        raise ValueError(
            f"aspect_embeddings must be 2D with shape (n_aspects, {embeddings.shape[1]}), "
            f"got {aspect_embeddings.shape}"
        )

    if normalize:
        embeddings = normalize_rows(embeddings)
        aspect_embeddings = normalize_rows(aspect_embeddings)

    doc_aspect_sim = np.clip(embeddings @ aspect_embeddings.T, 0.0, 1.0)

    # Infer aspect weights: aspects that high-scoring docs match well get more weight
    weights = scores @ doc_aspect_sim  # (n_aspects,)
    weights = np.maximum(weights, 0.0)
    weights = weights / (weights.sum() + EPS32)

    return _xquad_select(scores, k, diversity, doc_aspect_sim, weights, Strategy.RXQUAD, normalize)
