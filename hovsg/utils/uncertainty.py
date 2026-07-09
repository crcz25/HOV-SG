"""Side-effect-free utilities for hierarchical semantic uncertainty."""

import numpy as np


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


def compute_semantic_uncertainty(
    embedding,
    text_feats,
    label_idx=None,
    logit_scale=100.0,
    similarity=None,
):
    """Compute vocabulary compatibility and normalized softmax entropy.

    Args:
        embedding: One visual object embedding.
        text_feats: Text embeddings aligned row-by-row with the vocabulary.
        label_idx: Optional index of the already-assigned object label.
        logit_scale: Inverse temperature applied to cosine similarities.
        similarity: Optional cosine similarities already used for label selection.

    Returns:
        ``(similarity, semantic_uncertainty)`` when ``label_idx`` is omitted,
        otherwise ``(similarity, semantic_uncertainty, label_cos_sim)``.

    The softmax is a relative compatibility distribution over the supplied
    vocabulary; it is not a calibrated posterior probability.
    """
    similarities, _, semantic_uncertainty = compute_semantic_distribution(
        embedding,
        text_feats,
        logit_scale=logit_scale,
        similarity=similarity,
    )
    num_classes = similarities.shape[0]

    if label_idx is None:
        return similarities, semantic_uncertainty

    label_idx = _validate_label_idx(label_idx, num_classes)
    return similarities, semantic_uncertainty, float(similarities[label_idx])


def compute_semantic_distribution(
    embedding,
    text_feats,
    logit_scale=100.0,
    similarity=None,
):
    """Return cosine similarities, compatibility probabilities, and entropy.

    The probabilities form a relative compatibility distribution over the
    supplied vocabulary. They are not calibrated posterior probabilities.
    """
    embedding_array = np.asarray(embedding, dtype=np.float64).reshape(-1)
    text_feats_array = np.asarray(text_feats, dtype=np.float64)
    if text_feats_array.ndim != 2 or text_feats_array.shape[0] == 0:
        raise ValueError("text_feats must be a non-empty two-dimensional array")
    if embedding_array.shape[0] != text_feats_array.shape[1]:
        raise ValueError("embedding and text_feats must have the same feature dimension")

    num_classes = text_feats_array.shape[0]
    if np.linalg.norm(embedding_array) < _NORM_EPSILON:
        similarities = np.zeros(num_classes, dtype=np.float64)
        probabilities = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
        return similarities, probabilities, 1.0

    if similarity is None:
        similarities = compute_cosine_similarities(embedding_array, text_feats_array)
    else:
        similarities = np.asarray(similarity, dtype=np.float64).reshape(-1)
        if similarities.shape[0] != num_classes:
            raise ValueError("similarity must have one value per text feature")

    logits = float(logit_scale) * similarities
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities)

    if num_classes == 1:
        semantic_uncertainty = 0.0
    else:
        nonzero = probabilities > 0
        entropy = -np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))
        semantic_uncertainty = float(
            np.clip(entropy / np.log(num_classes), 0.0, 1.0)
        )

    return similarities, probabilities, semantic_uncertainty


def compute_room_containment_probs(
    semantic_probs,
    prior=0.0,
):
    """Aggregate object--class evidence into room--class containment beliefs."""
    semantic_probs = np.asarray(semantic_probs, dtype=np.float64)
    if semantic_probs.ndim != 2:
        raise ValueError("semantic_probs must have shape (num_objects, num_classes)")

    epsilon = float(np.clip(prior, 0.0, 1.0))
    contributions = np.clip(semantic_probs, 0.0, 1.0)

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
