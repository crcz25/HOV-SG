"""Cross-view agreement accumulation and confidence helpers.

Implements P^view_i = ||m_i||, where m_i is the running mean of the per-view
unit embeddings that the feature fusion averaged into the object embedding
v_i.  The signal is the disagreement the fusion discards: it equals one when
every view produced the same embedding and decreases toward zero as the views
cancel.

HOV-SG fuses features at point level, so the accumulator is keyed by point.
One call to :func:`accumulate_unit_embeddings` represents exactly one view
(one RGB-D frame), and each point contributes at most one unit observation per
view -- see the deduplication note in that function.
"""

import numpy as np


_NORM_EPSILON = 1e-8


def _as_writable_1d(array, name):
    """Return a 1-D view of ``array`` that in-place scatter-adds mutate.

    ``reshape(-1)`` silently returns a *copy* for non-contiguous input, which
    would turn ``np.add.at`` into a no-op. Returning an explicit view and
    checking it shares memory keeps the accumulation observable to the caller.
    """
    array = np.asarray(array)
    flat = array.reshape(-1)
    if not np.shares_memory(flat, array):
        raise ValueError(f"{name} must be contiguous so it can be updated in place")
    return flat


def accumulate_unit_embeddings(
    resultant_sum: np.ndarray,
    counter: np.ndarray,
    indices: np.ndarray,
    embeddings: np.ndarray,
    eps: float = _NORM_EPSILON,
) -> None:
    """Accumulate one view's unit-normalized per-point embeddings.

    Each call adds **one** observation per distinct point index, so ``counter``
    holds the number of *views* that saw each point rather than the number of
    pixels that projected onto it. Several pixels of a frame routinely land on
    the same voxel-downsampled point; counting each of them would inflate the
    running mean with repeated copies of a single view and would no longer be
    the mean the fusion computed.

    The retained observation for a repeated point index is the **last** valid
    one, which is the observation HOV-SG's fusion keeps: buffered fancy-index
    accumulation (``sum_features[idx] += F_2D`` in
    :meth:`hovsg.graph.graph.Graph.create_feature_map`) applies only the final
    write for a repeated index. Matching it keeps ``m_i`` the mean of exactly
    the per-view embeddings that produced ``v_i``.

    Zero-norm and non-finite embeddings are not observations -- pixels outside
    every SAM mask carry an all-zero feature -- so they are skipped rather than
    normalized into misleading evidence.
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

    valid_indices = indices[valid]
    valid_embeddings = embeddings[valid]
    valid_norms = norms[valid]

    # Keep the last valid observation per point: reverse, take the first
    # occurrence of each unique index, which is the last one in original order.
    reversed_order = np.arange(valid_indices.size - 1, -1, -1)
    _, first_in_reversed = np.unique(valid_indices[reversed_order], return_index=True)
    keep = reversed_order[first_in_reversed]

    unit_embeddings = valid_embeddings[keep] / valid_norms[keep, None]
    kept_indices = valid_indices[keep]
    np.add.at(resultant_sum, kept_indices, unit_embeddings)
    np.add.at(_as_writable_1d(counter, "counter"), kept_indices, 1)


def cross_view_values(
    resultant_sum: np.ndarray,
    count: int,
    min_observations: int = 2,
    point_count: int = None,
):
    """Return P^view, U^view and the low-support flag for one object.

    ``resultant_sum`` is the *unnormalized* accumulated sum of unit per-view
    embeddings and ``count`` the number of accumulated observations, so
    ``resultant_sum / count`` is the running mean ``m_i`` whose norm is the
    signal. Both are stored on the object; the normalized fused embedding
    ``v_i`` never replaces them.

    ``point_count`` is the number of distinct points that contributed. Because
    the accumulator is keyed by point, ``count`` totals point-view observations
    over the whole object, so ``count / point_count`` is the mean number of
    views per point -- the quantity ``min_observations`` is meant to threshold.
    When ``point_count`` is omitted the flag falls back to ``count``, which is
    equivalent only for a single-point object.

    Returns ``(None, None, False)`` when there is no evidence at all; the
    signal is undefined rather than assigned a fabricated probability.
    """
    count = int(count)
    min_observations = int(min_observations)
    if count < 0:
        raise ValueError("cross_view_count must not be negative")
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1")
    if count == 0:
        return None, None, False

    resultant_sum = np.asarray(resultant_sum, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(resultant_sum)):
        return None, None, False

    p_view = float(np.linalg.norm(resultant_sum / count))
    # The mean of unit vectors cannot exceed unit length; the clip only absorbs
    # floating-point drift at the perfect-agreement boundary.
    p_view = float(np.clip(p_view, 0.0, 1.0))

    if point_count is None:
        observations_per_point = float(count)
    else:
        point_count = int(point_count)
        if point_count <= 0:
            return p_view, float(1.0 - p_view), False
        observations_per_point = count / point_count

    return p_view, float(1.0 - p_view), observations_per_point >= min_observations
