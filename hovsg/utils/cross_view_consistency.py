"""Cross-view agreement accumulation and confidence helpers."""

import numpy as np


_NORM_EPSILON = 1e-8


def accumulate_unit_embeddings(
    resultant_sum: np.ndarray,
    counter: np.ndarray,
    indices: np.ndarray,
    embeddings: np.ndarray,
    eps: float = _NORM_EPSILON,
) -> None:
    """Accumulate valid unit-normalized per-view embeddings by point index.

    Zero-norm and non-finite embeddings are not observations: they are skipped
    rather than being normalized into misleading evidence.
    """
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if indices.size == 0:
        return
    if embeddings.ndim != 2 or embeddings.shape[0] != indices.size:
        raise ValueError("indices and embeddings must describe the same observations")
    if resultant_sum.ndim != 2 or resultant_sum.shape[1] != embeddings.shape[1]:
        raise ValueError("resultant_sum and embeddings must have the same feature dimension")

    norms = np.linalg.norm(embeddings, axis=1)
    valid = np.isfinite(norms) & (norms > eps) & np.isfinite(embeddings).all(axis=1)
    if not np.any(valid):
        return
    unit_embeddings = embeddings[valid] / norms[valid, None]
    valid_indices = indices[valid]
    np.add.at(resultant_sum, valid_indices, unit_embeddings)
    np.add.at(np.asarray(counter).reshape(-1), valid_indices, 1)


def cross_view_values(
    resultant_sum: np.ndarray,
    count: int,
    min_observations: int = 2,
):
    """Return derived cross-view confidence and its low-support flag."""
    count = int(count)
    min_observations = int(min_observations)
    if count < 0:
        raise ValueError("cross_view_count must not be negative")
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1")
    if count == 0:
        return None, None, False

    resultant_sum = np.asarray(resultant_sum, dtype=np.float64).reshape(-1)
    c_view = float(np.linalg.norm(resultant_sum / count))
    c_view = float(np.clip(c_view, 0.0, 1.0))
    return c_view, float(1.0 - c_view), count >= min_observations
