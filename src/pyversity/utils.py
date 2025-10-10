import numpy as np

from pyversity.datatypes import Metric

# A small float32 constant to avoid division by zero
EPS32: float = float(np.finfo(np.float32).eps)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize each row vector to unit length.

    :param vectors: Vectors to normalize.
    :return: Row-normalized array where each vector has unit norm.
    """
    vectors = np.asarray(vectors, dtype=np.float32, order="C")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, EPS32)
    return vectors / safe_norms


def prepare_inputs(relevances: np.ndarray, embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """
    Prepare relevance scores and embeddings.

    :param relevances: Array of relevance scores.
    :param embeddings: Array of shape embeddings.
    :param k: Number of top elements to consider.
    :return: Tuple of relevances, embeddings, k_clamped, early_exit.
    :raises ValueError: If input shapes are inconsistent.
    """
    relevances = np.asarray(relevances, dtype=np.float32).reshape(-1)
    embeddings = np.asarray(embeddings, dtype=np.float32, order="C")

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")

    num_samples = embeddings.shape[0]
    if relevances.shape[0] != num_samples:
        raise ValueError(f"relevances length {relevances.shape[0]} != embeddings rows {num_samples}")

    k_clamped = int(max(0, min(int(k), num_samples)))
    early_exit = (num_samples == 0) or (k_clamped == 0)

    return relevances, embeddings, k_clamped, early_exit


def vector_similarity(
    matrix: np.ndarray,
    vector: np.ndarray,
    metric: Metric,
) -> np.ndarray:
    """
    Compute (non-negative) similarity between each row in a matrix and a single vector.

    :param matrix: 2D array of shape (n_samples, n_features).
    :param vector: 1D array of shape (n_features,).
    :param metric: Similarity metric to use.
    :return: 1D array of similarity scores.
    """
    matrix = np.asarray(matrix, dtype=np.float32, order="C")
    vector = np.asarray(vector, dtype=np.float32, order="C")

    similarity = matrix @ vector
    if metric == Metric.COSINE:
        return np.clip(similarity, 0.0, 1.0, out=similarity)
    return np.maximum(similarity, 0.0, out=similarity)


def pairwise_similarity(
    matrix: np.ndarray,
    metric: Metric,
) -> np.ndarray:
    """
    Compute (non-negative) pairwise similarities between all rows in a matrix.

    :param matrix: 2D array of shape (n_samples, n_features).
    :param metric: Similarity metric to use.
    :return: 2D array of shape (n_samples, n_samples) with pairwise similarities.
    """
    matrix = np.asarray(matrix, dtype=np.float32, order="C")

    similarity = matrix @ matrix.T
    if metric == Metric.COSINE:
        return np.clip(similarity, 0.0, 1.0, out=similarity)
    return np.maximum(similarity, 0.0, out=similarity)
