"""Side-effect-free utilities for semantic and containment uncertainty."""

import logging

import numpy as np
from scipy.special import expit


_NORM_EPSILON = 1e-8


def compute_cosine_similarities(embedding, text_feats):
    """Compute cosine similarities from one visual feature to text features.

    Both the visual feature and every text feature are normalized defensively.
    A zero visual feature produces an all-zero similarity vector.
    """
    embedding = np.asarray(embedding, dtype=np.float64).reshape(-1)
    text_feats = np.asarray(text_feats, dtype=np.float64)

    if text_feats.ndim != 2:
        raise ValueError("text_feats must be a two-dimensional array")
    if text_feats.shape[0] == 0:
        raise ValueError("text_feats must contain at least one class")
    if embedding.shape[0] != text_feats.shape[1]:
        raise ValueError("embedding and text_feats must have the same feature dimension")

    embedding_norm = np.linalg.norm(embedding)
    if embedding_norm < _NORM_EPSILON:
        return np.zeros(text_feats.shape[0], dtype=np.float64)

    text_norms = np.linalg.norm(text_feats, axis=1, keepdims=True)
    normalized_text_feats = np.divide(
        text_feats,
        text_norms,
        out=np.zeros_like(text_feats),
        where=text_norms >= _NORM_EPSILON,
    )
    return normalized_text_feats @ (embedding / embedding_norm)


def build_synonym_eligibility_mask(text_feats, tau):
    """Build the per-label mask of semantically distinct competitors.

    ``mask[i, j]`` is true exactly when the cosine similarity between text
    rows ``i`` and ``j`` is below ``tau``. Text rows are normalized
    defensively and the diagonal is always excluded.
    """
    text_feats = np.asarray(text_feats, dtype=np.float64)
    if text_feats.ndim != 2 or text_feats.shape[0] == 0:
        raise ValueError("text_feats must be a non-empty two-dimensional array")

    tau = float(tau)
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be strictly between 0 and 1")

    text_norms = np.linalg.norm(text_feats, axis=1, keepdims=True)
    normalized_text_feats = np.divide(
        text_feats,
        text_norms,
        out=np.zeros_like(text_feats),
        where=text_norms >= _NORM_EPSILON,
    )
    text_similarities = normalized_text_feats @ normalized_text_feats.T
    eligibility_mask = text_similarities < tau
    np.fill_diagonal(eligibility_mask, False)
    return eligibility_mask


def compute_semantic_margin_uncertainty(
    embedding,
    text_feats,
    label_idx,
    eligibility_mask,
    logit_scale=100.0,
    similarity=None,
):
    """Compute distinct-competitor margin confidence for one object.

    The optional ``similarity`` vector lets callers reuse the cosine
    similarities that assigned the label. The embedding is still normalized
    and checked internally so the zero-norm override remains authoritative.
    """
    embedding = np.asarray(embedding, dtype=np.float64).reshape(-1)
    text_feats = np.asarray(text_feats, dtype=np.float64)
    if text_feats.ndim != 2 or text_feats.shape[0] == 0:
        raise ValueError("text_feats must be a non-empty two-dimensional array")
    if embedding.shape[0] != text_feats.shape[1]:
        raise ValueError("embedding and text_feats must have the same feature dimension")

    num_classes = text_feats.shape[0]
    label_idx = _validate_label_idx(label_idx, num_classes)
    eligibility_mask = np.asarray(eligibility_mask, dtype=bool)
    if eligibility_mask.shape != (num_classes, num_classes):
        raise ValueError("eligibility_mask must have shape (num_classes, num_classes)")

    embedding_norm = np.linalg.norm(embedding)
    if embedding_norm < _NORM_EPSILON:
        return {
            "label_cos_sim": 0.0,
            "runner_up_idx": None,
            "runner_up_cos_sim": 0.0,
            "semantic_margin": 0.0,
            "c_sem": 0.0,
            "u_sem": 1.0,
        }
    normalized_embedding = embedding / embedding_norm

    if similarity is None:
        similarities = compute_cosine_similarities(normalized_embedding, text_feats)
    else:
        similarities = np.asarray(similarity, dtype=np.float64).reshape(-1)
        if similarities.shape != (num_classes,):
            raise ValueError("similarity must have one value per text feature")

    label_cos_sim = float(similarities[label_idx])
    candidates = np.flatnonzero(eligibility_mask[label_idx])
    if candidates.size == 0:
        logging.getLogger(__name__).warning(
            "No eligible distinct semantic competitor for label_idx=%d; "
            "using the empty-eligible-set fallback c_sem=1.0, u_sem=0.0",
            label_idx,
        )
        return {
            "label_cos_sim": label_cos_sim,
            "runner_up_idx": None,
            "runner_up_cos_sim": None,
            "semantic_margin": None,
            "c_sem": 1.0,
            "u_sem": 0.0,
        }

    runner_up_idx = int(candidates[np.argmax(similarities[candidates])])
    runner_up_cos_sim = float(similarities[runner_up_idx])
    semantic_margin = float(label_cos_sim - runner_up_cos_sim)
    c_sem = float(expit(float(logit_scale) * semantic_margin))
    return {
        "label_cos_sim": label_cos_sim,
        "runner_up_idx": runner_up_idx,
        "runner_up_cos_sim": runner_up_cos_sim,
        "semantic_margin": semantic_margin,
        "c_sem": c_sem,
        "u_sem": float(1.0 - c_sem),
    }


def compute_room_containment_probs(
    semantic_probs,
    detection_reliabilities=None,
    prior=0.0,
):
    """Aggregate object--class evidence into room--class containment beliefs."""
    semantic_probs = np.asarray(semantic_probs, dtype=np.float64)
    if semantic_probs.ndim != 2:
        raise ValueError("semantic_probs must have shape (num_objects, num_classes)")

    num_objects = semantic_probs.shape[0]
    if detection_reliabilities is None:
        detection_reliabilities = np.ones(num_objects, dtype=np.float64)
    else:
        detection_reliabilities = np.asarray(
            detection_reliabilities, dtype=np.float64
        ).reshape(-1)
        if detection_reliabilities.shape[0] != num_objects:
            raise ValueError("one detection reliability is required per object")

    epsilon = float(np.clip(prior, 0.0, 1.0))
    reliabilities = np.clip(detection_reliabilities, 0.0, 1.0)
    contributions = np.clip(semantic_probs * reliabilities[:, None], 0.0, 1.0)

    with np.errstate(divide="ignore"):
        log_not_contained = np.sum(np.log1p(-contributions), axis=0)
    not_contained = np.exp(log_not_contained)
    return np.clip(1.0 - (1.0 - epsilon) * not_contained, 0.0, 1.0)


def _validate_label_idx(label_idx, num_classes):
    """Return a Python integer after validating a vocabulary row index."""
    label_idx = int(label_idx)
    if not 0 <= label_idx < num_classes:
        raise IndexError("label_idx is outside the text feature vocabulary")
    return label_idx
