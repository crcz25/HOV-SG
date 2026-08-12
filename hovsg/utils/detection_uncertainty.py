"""Utilities for object detection confidence and uncertainty."""

import warnings

import numpy as np


def _clamp_probability(value, name):
    """Return a finite probability, clamping small upstream drift."""
    value = float(value)
    if np.isnan(value):
        raise ValueError(f"{name} must not be NaN")

    clipped = float(np.clip(value, 0.0, 1.0))
    if clipped != value:
        warnings.warn(
            f"{name}={value} is outside [0, 1]; clamping to {clipped}",
            RuntimeWarning,
            stacklevel=2,
        )
    return clipped


def mask_predicted_iou(mask: dict) -> float:
    """Extract SAM ``predicted_iou`` as a clamped confidence.

    SAM scores can drift fractionally outside ``[0, 1]`` due to floating-point
    behavior, so finite values are clamped. NaN is treated as invalid evidence
    and raises ``ValueError`` instead of silently creating a misleading score.
    """
    return _clamp_probability(mask["predicted_iou"], "predicted_iou")


def accumulate_confidence(
    sum_conf: np.ndarray,
    counter: np.ndarray,
    indices: np.ndarray,
    value: float,
) -> None:
    """Accumulate one confidence value into point-indexed running sums."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return

    value = _clamp_probability(value, "confidence")
    np.add.at(sum_conf.reshape(-1), indices, value)
    np.add.at(counter.reshape(-1), indices, 1.0)


def finalize_confidence_array(
    sum_conf: np.ndarray,
    counter: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Safely divide confidence sums by counts using HOV-SG's epsilon guard."""
    safe_counter = np.asarray(counter, dtype=np.float64).copy()
    safe_counter[safe_counter == 0] = eps
    return np.asarray(sum_conf, dtype=np.float64) / safe_counter


def object_confidence_sum_from_points(
    full_conf_array: np.ndarray,
    tree_pcd,
    points: np.ndarray,
):
    """Return the point-confidence ``(sum, count)`` for one object point cloud.

    P_det is the mean SAM ``predicted_iou`` over the object's points. The sum
    and the count are returned separately -- rather than only their ratio --
    so the object node stores the raw evidence and P_det can be recomputed
    from it, in particular after a graph is reloaded from disk.
    """
    points = np.asarray(points)
    if points.size == 0:
        return 0.0, 0

    _, idx = tree_pcd.query(points, k=1, workers=-1)
    values = np.asarray(full_conf_array)[idx].reshape(-1)
    if values.size == 0:
        return 0.0, 0
    values = np.clip(np.nan_to_num(values), 0.0, 1.0)
    return float(values.sum()), int(values.size)


def confidence_from_sum(conf_sum, count, default=None):
    """Return the pooled mean confidence, or ``default`` without evidence.

    ``default=None`` reports P_det as undefined for an object with no
    confidence evidence rather than asserting a value for it.
    """
    if conf_sum is None or count is None:
        return default
    count = int(count)
    if count <= 0:
        return default
    return _clamp_probability(float(conf_sum) / count, "object confidence")


def uncertainty_from_confidence(p_det: float) -> float:
    """Return detection uncertainty as the complement of confidence."""
    p_det = _clamp_probability(p_det, "p_det")
    return 1.0 - p_det
