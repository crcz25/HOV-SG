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


def object_confidence_from_points(
    full_conf_array: np.ndarray,
    tree_pcd,
    points: np.ndarray,
    default: float = 0.0,
) -> float:
    """Average point-level confidence over one final object point cloud.

    ``default=0.0`` is the conservative direction for objects with no confidence
    evidence: its complement is ``u_det=1.0``, maximal detection uncertainty.
    """
    points = np.asarray(points)
    if points.size == 0:
        return _clamp_probability(default, "default")

    _, idx = tree_pcd.query(points, k=1, workers=-1)
    values = np.asarray(full_conf_array)[idx]
    if values.size == 0:
        return _clamp_probability(default, "default")
    return _clamp_probability(np.nan_to_num(values).mean(), "object confidence")


def uncertainty_from_confidence(c_det: float) -> float:
    """Return detection uncertainty as the complement of confidence."""
    c_det = _clamp_probability(c_det, "c_det")
    return 1.0 - c_det
