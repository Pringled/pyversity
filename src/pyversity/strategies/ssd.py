import numpy as np

from pyversity.datatypes import DiversificationResult, Strategy
from pyversity.utils import EPS32, normalize_rows, prepare_inputs


def ssd(  # noqa: C901
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int,
    diversity: float = 0.5,
    recent_embeddings: np.ndarray | None = None,
    window: int | None = None,
    gamma: float = 1.0,
    normalize: bool = True,
    append_bias: bool = True,
    normalize_scores: bool = True,
) -> DiversificationResult:
    """
    Sliding Spectrum Decomposition (SSD) selection.

    This strategy selects `k` items using a greedy, sequence-aware approach that maintains a sliding window
    of Gram-Schmidt bases to promote diversity while considering recent context.
    If `recent_embeddings` are provided (oldest → newest), the window is seeded so the very first pick is
    already novel relative to what the user just saw.

    Note: this follows the stabilized SSD variant (“SSD*”) described in Eq. (12) in the paper.

    :param embeddings: 2D array (n_items, n_dims) of candidate embeddings.
    :param scores: 1D array (n_items,) of relevance scores.
    :param k: Number of items to select.
    :param diversity: Trade-off between relevance and diversity in [0, 1] (inverse of theta parameter).
                      1.0 = pure diversity, 0.0 = pure relevance.
    :param recent_embeddings: Optional 2D array (m, n_dims), oldest → newest; seeds the sliding window so
                              selection is aware of what was recently shown.
    :param window: Window size (≥ 1) for Gram-Schmidt bases. If None, defaults to len(recent_embeddings) + k.
    :param gamma: Diversity scale (> 0).
    :param normalize: Whether to normalize embeddings before computing similarity.
    :param append_bias: Append a constant-one bias dimension after normalization.
    :param normalize_scores: Z-score the scores per request (stabilizes the gamma trade-off).
    :return: DiversificationResult with selected indices and their selection scores.
    :raises ValueError: If diversity ∉ [0, 1], or window < 1, or gamma ≤ 0.
    """
    # Validate parameters
    if not (0.0 <= float(diversity) <= 1.0):
        raise ValueError("diversity must be in [0, 1]")
    if window is not None and window < 1:
        raise ValueError("window must be >= 1")
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0")

    # Theta parameter for trade-off between relevance and diversity (1 - diversity)
    theta = 1.0 - float(diversity)

    # Prepare inputs
    feature_matrix, relevance_scores, top_k, early_exit = prepare_inputs(embeddings, scores, k)
    if early_exit:
        # Nothing to select: return empty arrays
        return DiversificationResult(
            indices=np.empty(0, np.int32),
            selection_scores=np.empty(0, np.float32),
            strategy=Strategy.SSD,
            diversity=diversity,
            parameters={"gamma": gamma, "window": window},
        )

    # Validate recent_embeddings
    n_recent = 0
    if recent_embeddings is not None and np.size(recent_embeddings) > 0:
        if recent_embeddings.ndim != 2:
            raise ValueError("recent_embeddings must be a 2D array of shape (n_items, n_dims).")
        if recent_embeddings.shape[1] != feature_matrix.shape[1]:
            raise ValueError(
                f"recent_embeddings has {recent_embeddings.shape[1]} dims; "
                f"expected {feature_matrix.shape[1]} to match `embeddings` columns."
            )
        n_recent = int(recent_embeddings.shape[0])

    # Determine effective window size
    window_size = (n_recent + top_k) if window is None else int(window)

    # Pure relevance: select top-k by relevance scores
    if float(theta) == 1.0:
        topk = np.argsort(-relevance_scores)[:top_k].astype(np.int32)
        selection_scores = relevance_scores[topk].astype(np.float32, copy=False)
        return DiversificationResult(
            indices=topk,
            selection_scores=selection_scores,
            strategy=Strategy.SSD,
            diversity=diversity,
            parameters={"gamma": gamma, "window": window_size},
        )

    def _prepare_vectors(matrix: np.ndarray) -> np.ndarray:
        """Prepare feature vectors with normalization and (optionally) a bias dimension."""
        prepared = matrix
        if normalize:
            prepared = normalize_rows(prepared)
        if append_bias:
            # Bias trick described in the paper (Section 5.3)
            last_col_is_ones = prepared.shape[1] > 0 and np.allclose(prepared[:, -1], 1.0, atol=1e-6, rtol=0.0)
            if not last_col_is_ones:
                ones = np.ones((prepared.shape[0], 1), dtype=prepared.dtype)
                prepared = np.concatenate([prepared, ones], axis=1)
        return prepared

    # Prepare feature vectors
    feature_matrix = _prepare_vectors(feature_matrix)

    # Per-request score normalization (stabilizes gamma)
    if normalize_scores:
        mean = float(np.mean(relevance_scores))
        std = float(np.std(relevance_scores))
        relevance_scores = (relevance_scores - mean) / std if std > 0.0 else (relevance_scores - mean)

    num_items, n_dims = feature_matrix.shape

    # Initialize selection state
    selected_mask = np.zeros(num_items, dtype=bool)
    selected_indices = np.empty(top_k, dtype=np.int32)
    selection_scores = np.empty(top_k, dtype=np.float32)

    # Current residuals under the sliding window
    residual_matrix = feature_matrix.astype(np.float32, copy=True)

    # Squared residual norms, maintained incrementally to avoid recomputing from the full matrix
    residual_sq_norms: np.ndarray = np.einsum("ij,ij->i", residual_matrix, residual_matrix)

    # Circular buffer for sliding window of basis vectors and their projection coefficients
    basis_matrix = np.zeros((window_size, n_dims), dtype=np.float32)
    coeff_matrix = np.zeros((window_size, num_items), dtype=np.float32)
    window_count = 0
    window_head = 0

    # Re-used buffer for rank-1 matrix updates (avoids allocating a new n×d array each step)
    update_buffer = np.empty((num_items, n_dims), dtype=np.float32)

    def _push_basis_vector(basis_vector: np.ndarray) -> None:
        """Add a new basis vector to the sliding window and update residuals/projections."""
        nonlocal window_count, window_head

        if window_count == window_size:
            # Evict the oldest basis and restore its contribution to residuals.
            # Skip selected items so their residuals stay untouched.
            oldest_slot = window_head
            coeff_matrix[oldest_slot][selected_mask] = 0.0
            evicted_coefficients = coeff_matrix[oldest_slot]
            evicted_basis = basis_matrix[oldest_slot]
            evicted_basis_sq_norm = float(evicted_basis @ evicted_basis)

            # Update squared norms to reflect the restored residuals
            projections = residual_matrix @ evicted_basis
            residual_sq_norms[:] += evicted_coefficients * (
                2.0 * projections + evicted_coefficients * evicted_basis_sq_norm
            )

            # Restore residuals by adding back the evicted projection
            np.outer(evicted_coefficients, evicted_basis, out=update_buffer)
            np.add(residual_matrix, update_buffer, out=residual_matrix)
        else:
            window_count += 1

        basis_sq_norm = float(basis_vector @ basis_vector)
        denominator = basis_sq_norm + EPS32
        basis_matrix[window_head] = basis_vector

        # Project each item's residual onto the new basis vector
        projections = residual_matrix @ basis_vector
        coefficients = projections / denominator
        coefficients[selected_mask] = 0.0
        coeff_matrix[window_head] = coefficients

        # Update squared norms to reflect the removed projection
        residual_sq_norms[:] -= coefficients * (2.0 * projections - coefficients * basis_sq_norm)
        np.maximum(residual_sq_norms, 0.0, out=residual_sq_norms)

        # Subtract projection from residuals
        np.outer(coefficients, basis_vector, out=update_buffer)
        np.subtract(residual_matrix, update_buffer, out=residual_matrix)
        window_head = (window_head + 1) % window_size

    # Seed with recent context (oldest → newest) if provided
    seeded_bases = 0
    if recent_embeddings is not None and np.size(recent_embeddings) > 0:
        context = _prepare_vectors(recent_embeddings.astype(feature_matrix.dtype, copy=False))
        context = context[-window_size:]  # keep only the latest `window_size` items
        for context_vector in context:
            residual_context = context_vector.copy()
            for slot_offset in range(window_count):
                slot_idx = (window_head - window_count + slot_offset) % window_size
                basis = basis_matrix[slot_idx]
                denominator = float(basis @ basis) + EPS32
                residual_context -= float(residual_context @ basis) / denominator * basis
            _push_basis_vector(residual_context)
            seeded_bases += 1

    # Decide what to select first
    if seeded_bases > 0:
        # Use combined scores with diversity from seeded context
        residual_norms = np.sqrt(residual_sq_norms)
        combined_scores = theta * relevance_scores + (1.0 - theta) * gamma * residual_norms
        combined_scores[selected_mask] = -np.inf
        first_index = int(np.argmax(combined_scores))
        first_score = float(combined_scores[first_index])
    else:
        # No context yet: pick by highest relevance, then start residualization
        first_index = int(np.argmax(relevance_scores))
        first_score = float(
            theta * relevance_scores[first_index]
            + (1.0 - theta) * gamma * float(np.linalg.norm(feature_matrix[first_index]))
        )

    # Select the first item
    selected_mask[first_index] = True
    selected_indices[0] = first_index
    selection_scores[0] = first_score
    _push_basis_vector(residual_matrix[first_index])

    # Main loop
    for step in range(1, top_k):
        # Residual norms measure novelty relative to the sliding window of recent selections/context
        residual_norms = np.sqrt(residual_sq_norms)
        combined_scores = theta * relevance_scores + (1.0 - theta) * gamma * residual_norms
        combined_scores[selected_mask] = -np.inf
        best_index = int(np.argmax(combined_scores))
        best_score = float(combined_scores[best_index])

        # Update selection state
        selected_mask[best_index] = True
        selected_indices[step] = best_index
        selection_scores[step] = best_score
        _push_basis_vector(residual_matrix[best_index])

    return DiversificationResult(
        indices=selected_indices,
        selection_scores=selection_scores.astype(np.float32, copy=False),
        strategy=Strategy.SSD,
        diversity=diversity,
        parameters={"gamma": gamma, "window": window_size},
    )
