"""Cross-view agreement accumulation and confidence.

Implements eq. (crossview) of the paper. Let m_i be the unweighted running mean
of the n per-view unit embeddings that the feature fusion averaged into the
object embedding v_i. The resultant-length identity for unit vectors gives

    ||m_i||^2 = 1/n + (1 - 1/n) * c_bar_i,

with c_bar_i the mean pairwise cosine similarity between the views. ||m_i||
therefore depends on n as well as on the agreement, so the signal is *not*
||m_i||: c_bar_i is first recovered from the identity and then mapped from
[-1, 1] onto [0, 1],

    P^view_i = (1 + c_bar_i) / 2,

which is one when every view produced the same embedding and decreases as the
views disagree, independently of how many views were observed. A single view
leaves c_bar_i undefined and P^view_i is set to 1.

HOV-SG fuses features at point level, so the accumulator is keyed by point and
one accumulated unit vector is one (point, view) observation; n is the number
of accumulated unit vectors, which is what the identity above requires.
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


def mean_pairwise_cosine_similarity(resultant_norm: float, view_count: int):
    """Recover c_bar_i from ||m_i|| and n via the resultant-length identity.

    Inverting ||m_i||^2 = 1/n + (1 - 1/n) c_bar_i gives

        c_bar_i = (||m_i||^2 - 1/n) / (1 - 1/n).

    Returns ``None`` for n < 2, where a mean pairwise similarity does not exist.
    The result is clipped to [-1, 1] to absorb floating-point drift only; the
    identity itself cannot leave that interval for unit observations.
    """
    view_count = int(view_count)
    if view_count < 2:
        return None
    inverse_count = 1.0 / view_count
    mean_cosine = (float(resultant_norm) ** 2 - inverse_count) / (1.0 - inverse_count)
    return float(np.clip(mean_cosine, -1.0, 1.0))


def compute_cross_view_consistency(resultant_sum: np.ndarray, view_count: int):
    """Return ``(P^view_i, U^view_i)`` for one object.

    ``resultant_sum`` is the *unnormalized* accumulated sum of unit per-view
    embeddings and ``view_count`` is n, the number of accumulated observations,
    so ``resultant_sum / view_count`` is the running mean ``m_i``. Both are
    stored on the object; the normalized fused embedding ``v_i`` never replaces
    them.

    Returns ``(None, None)`` when there is no evidence at all: the signal is
    undefined rather than assigned a fabricated probability.
    """
    view_count = int(view_count)
    if view_count < 0:
        raise ValueError("view_count must not be negative")
    if view_count == 0:
        return None, None

    resultant_sum = np.asarray(resultant_sum, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(resultant_sum)):
        return None, None

    if view_count == 1:
        # A single view leaves c_bar_i undefined; the paper sets P^view_i = 1.
        return 1.0, 0.0

    resultant_norm = float(np.linalg.norm(resultant_sum / view_count))
    mean_cosine = mean_pairwise_cosine_similarity(resultant_norm, view_count)
    # Affine map of [-1, 1] onto [0, 1]; no further clipping is needed because
    # mean_pairwise_cosine_similarity already returns a value in [-1, 1].
    p_view = (1.0 + mean_cosine) / 2.0
    return p_view, float(1.0 - p_view)
