"""Utilities for computing semantic-label uncertainty."""

import numpy as np


def compute_label_uncertainty(
    embedding,
    text_feats,
    temperature=100.0,
    similarity=None,
):
    """Return class cosine similarities and normalized softmax entropy.

    ``similarity`` may be supplied by a caller that already computed the class
    scores (for example, while selecting the label) to avoid a second matrix
    multiplication.
    """
    embedding = np.asarray(embedding)
    text_feats = np.asarray(text_feats)
    num_classes = text_feats.shape[0]

    embedding_norm = np.linalg.norm(embedding)
    if embedding_norm < 1e-8:
        return np.zeros(num_classes, dtype=np.float64), 1.0

    if similarity is None:
        normalized_embedding = embedding / embedding_norm
        similarity = np.dot(normalized_embedding, text_feats.T)
    else:
        similarity = np.asarray(similarity)

    if num_classes <= 1:
        return similarity, 0.0

    logits = float(temperature) * similarity
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities)

    nonzero = probabilities > 0
    entropy = -np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))
    entropy_normalized = entropy / np.log(num_classes)
    entropy_normalized = float(np.clip(entropy_normalized, 0.0, 1.0))
    return similarity, entropy_normalized
