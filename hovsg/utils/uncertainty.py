"""Side-effect-free utilities for object-level semantic uncertainty."""

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
    embedding_array = np.asarray(embedding, dtype=np.float64).reshape(-1)
    text_feats_array = np.asarray(text_feats, dtype=np.float64)
    if text_feats_array.ndim != 2 or text_feats_array.shape[0] == 0:
        raise ValueError("text_feats must be a non-empty two-dimensional array")
    if embedding_array.shape[0] != text_feats_array.shape[1]:
        raise ValueError("embedding and text_feats must have the same feature dimension")

    num_classes = text_feats_array.shape[0]
    embedding_norm = np.linalg.norm(embedding_array)
    if embedding_norm < _NORM_EPSILON:
        similarities = np.zeros(num_classes, dtype=np.float64)
        if label_idx is None:
            return similarities, 1.0
        _validate_label_idx(label_idx, num_classes)
        return similarities, 1.0, 0.0

    if similarity is None:
        similarities = compute_cosine_similarities(embedding_array, text_feats_array)
    else:
        similarities = np.asarray(similarity, dtype=np.float64).reshape(-1)
        if similarities.shape[0] != num_classes:
            raise ValueError("similarity must have one value per text feature")

    if num_classes == 1:
        semantic_uncertainty = 0.0
    else:
        logits = float(logit_scale) * similarities
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities)

        nonzero = probabilities > 0
        entropy = -np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))
        semantic_uncertainty = float(
            np.clip(entropy / np.log(num_classes), 0.0, 1.0)
        )

    if label_idx is None:
        return similarities, semantic_uncertainty

    label_idx = _validate_label_idx(label_idx, num_classes)
    return similarities, semantic_uncertainty, float(similarities[label_idx])


def _validate_label_idx(label_idx, num_classes):
    """Return a Python integer after validating a vocabulary row index."""
    label_idx = int(label_idx)
    if not 0 <= label_idx < num_classes:
        raise IndexError("label_idx is outside the text feature vocabulary")
    return label_idx
