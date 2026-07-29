"""Side-effect-free utilities for semantic and containment uncertainty."""

import logging
from typing import Optional

import numpy as np
from scipy.special import expit, logsumexp


_NORM_EPSILON = 1e-8

#: Field names written by :func:`compute_semantic_margin_uncertainty`.
SEMANTIC_FIELDS = (
    "label_cos_sim",
    "runner_up_idx",
    "runner_up_cos_sim",
    "semantic_margin",
    "p_sem",
    "u_sem",
)

#: Field names written by :func:`compute_vocabulary_membership`.
MEMBERSHIP_FIELDS = (
    "vocab_log_partition",
    "negative_log_partition",
    "p_mem",
    "u_mem",
)

#: Field names written by :func:`compute_label_coherence_uncertainty`.
COHERENCE_FIELDS = (
    "coherence_prototype_cos_sim",
    "coherence_runner_up_class",
    "coherence_runner_up_cos_sim",
    "label_coherence_margin",
    "p_coh",
    "u_coh",
)

# The fused object-level probability is kept as a probability/uncertainty
# pair, like each of its five input signals.
OBJECT_FIELDS = ("p_obj", "u_obj")


def compute_object_probability(
    p_det: Optional[float],
    p_view: Optional[float],
    p_mem: Optional[float],
    p_sem: Optional[float],
    p_coh: Optional[float],
    cross_view_implemented: bool,
) -> Optional[float]:
    """Fuse the object-level probability from the five signal providers.

    ``p_sem`` and ``p_coh`` estimate the same label-error event from
    different references, so the more conservative defined value is used.
    Missing required evidence propagates as ``None``.  The only neutral-value
    reduction is structural absence of the cross-view provider, where its
    factor is omitted from the product.
    """
    required = (p_det, p_mem, p_sem)
    if any(value is None for value in required):
        return None
    if cross_view_implemented and p_view is None:
        return None

    semantic_factor = p_sem if p_coh is None else min(p_sem, p_coh)
    factors = [p_det, p_mem, semantic_factor]
    if cross_view_implemented:
        factors.append(p_view)
    return float(np.prod(np.asarray(factors, dtype=np.float64)))


def _sigmoid_margin_confidence(margin, logit_scale):
    """Convert a cosine margin using the shared uncertainty logit scale."""
    return float(expit(float(logit_scale) * float(margin)))


def normalize_rows(features):
    """Return ``features`` with unit-norm rows; zero rows are left at zero.

    Cached CLIP text banks are stored unnormalized on disk, and both the
    vocabulary (1624 x 1024 for HM3D) and the negative bank (~20k x 1024) are
    reused for every object. Normalizing once through this helper and passing
    ``assume_normalized=True`` downstream avoids repeating that work per
    object; see :func:`compute_cosine_similarities`.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return np.divide(
        features,
        norms,
        out=np.zeros_like(features),
        where=norms >= _NORM_EPSILON,
    )


def compute_cosine_similarities(embedding, text_feats, assume_normalized=False):
    """Compute cosine similarities from one visual feature to text features.

    The visual feature is always normalized. Text features are normalized
    defensively unless ``assume_normalized`` states that the caller already
    passed unit rows (e.g. a bank prepared once with :func:`normalize_rows`).
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
    if not np.isfinite(embedding_norm) or embedding_norm < _NORM_EPSILON:
        return np.zeros(text_feats.shape[0], dtype=np.float64)

    normalized_text_feats = (
        text_feats if assume_normalized else normalize_rows(text_feats)
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

    normalized_text_feats = normalize_rows(text_feats)
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
    assume_normalized=False,
):
    """Compute distinct-competitor margin confidence for one object.

    Implements m_i = cos(v_i, t_l) - max over classes c with
    cos(t_c, t_l) < tau of cos(v_i, t_c), then P_sem = sigma(alpha * m_i) and
    U_sem = 1 - P_sem. The eligible-competitor set is supplied as
    ``eligibility_mask`` so the tau comparison is built once per vocabulary.

    The optional ``similarity`` vector lets callers reuse the cosine
    similarities that assigned the label. The embedding is still normalized
    and checked internally so the undefined-input result remains authoritative.
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
    if not np.isfinite(embedding_norm) or embedding_norm < _NORM_EPSILON:
        # cos(v_i, t_c) is undefined for a zero or non-finite embedding, so the
        # margin m_i is undefined and so is sigma(alpha * m_i). Reporting a
        # number here would be a fabricated probability: the previous
        # implementation returned margin 0.0 alongside P_sem 0.0, which are
        # mutually inconsistent (sigma(alpha * 0) = 0.5).
        logging.getLogger(__name__).warning(
            "Semantic Uncertainty is undefined for a zero-norm or non-finite "
            "embedding (label_idx=%d)",
            label_idx,
        )
        return dict.fromkeys(SEMANTIC_FIELDS)
    normalized_embedding = embedding / embedding_norm

    if similarity is None:
        similarities = compute_cosine_similarities(
            normalized_embedding, text_feats, assume_normalized=assume_normalized
        )
    else:
        similarities = np.asarray(similarity, dtype=np.float64).reshape(-1)
        if similarities.shape != (num_classes,):
            raise ValueError("similarity must have one value per text feature")

    label_cos_sim = float(similarities[label_idx])
    candidates = np.flatnonzero(eligibility_mask[label_idx])
    if candidates.size == 0:
        # Every other class is a near-synonym of l_i, so the maximum in the
        # margin is taken over an empty set and m_i is undefined. Label
        # Coherence reports the same situation the same way; neither signal
        # invents a probability for it.
        logging.getLogger(__name__).warning(
            "No eligible distinct semantic competitor for label_idx=%d; "
            "Semantic Uncertainty is undefined because every competitor is "
            "excluded as a synonym",
            label_idx,
        )
        undefined = dict.fromkeys(SEMANTIC_FIELDS)
        undefined["label_cos_sim"] = label_cos_sim
        return undefined

    runner_up_idx = int(candidates[np.argmax(similarities[candidates])])
    runner_up_cos_sim = float(similarities[runner_up_idx])
    semantic_margin = float(label_cos_sim - runner_up_cos_sim)
    p_sem = _sigmoid_margin_confidence(semantic_margin, logit_scale)
    return {
        "label_cos_sim": label_cos_sim,
        "runner_up_idx": runner_up_idx,
        "runner_up_cos_sim": runner_up_cos_sim,
        "semantic_margin": semantic_margin,
        "p_sem": p_sem,
        "u_sem": float(1.0 - p_sem),
    }


def compute_label_coherence_uncertainty(
    embedding,
    label_idx,
    class_embedding_sum,
    class_count,
    class_prototype_full,
    eligibility_mask,
    label_classes=None,
    logit_scale=100.0,
):
    """Compute the leave-one-out, visual-only class-coherence margin.

    The class sums and prototypes are supplied by the caller so they can be
    built once for the complete, post-merge object set.  Text features are
    deliberately absent from this function: ``eligibility_mask`` is derived
    from the already-cached text vocabulary and is used only to remove
    near-synonym competitor classes.  Label Coherence and Semantic
    Uncertainty estimate the same labeling-error event; they must not be
    multiplied together as independent factors by this signal.
    """
    undefined = dict.fromkeys(COHERENCE_FIELDS)

    try:
        label_idx = int(label_idx)
    except (TypeError, ValueError):
        return undefined

    embedding = np.asarray(embedding, dtype=np.float64).reshape(-1)
    embedding_norm = np.linalg.norm(embedding)
    if not np.isfinite(embedding_norm) or embedding_norm < _NORM_EPSILON:
        return undefined
    normalized_embedding = embedding / embedding_norm

    count = int(class_count.get(label_idx, 0))
    if count < 2 or label_idx not in class_embedding_sum:
        return undefined

    leave_one_out_sum = (
        np.asarray(class_embedding_sum[label_idx], dtype=np.float64).reshape(-1)
        - normalized_embedding
    )
    leave_one_out_norm = np.linalg.norm(leave_one_out_sum)
    if (
        not np.isfinite(leave_one_out_norm)
        or leave_one_out_norm < _NORM_EPSILON
    ):
        return undefined
    self_prototype = leave_one_out_sum / leave_one_out_norm
    self_cos_sim = float(np.dot(normalized_embedding, self_prototype))

    eligibility_mask = np.asarray(eligibility_mask, dtype=bool)
    if label_idx < 0 or label_idx >= eligibility_mask.shape[0]:
        return undefined
    instantiated_classes = sorted(
        int(class_idx)
        for class_idx in class_prototype_full
        if int(class_idx) != label_idx
    )
    eligible_classes = [
        class_idx
        for class_idx in instantiated_classes
        if class_idx < eligibility_mask.shape[1]
        and eligibility_mask[label_idx, class_idx]
    ]
    if not eligible_classes:
        if instantiated_classes:
            logging.getLogger(__name__).warning(
                "No eligible distinct visual competitor for label_idx=%d; "
                "Label Coherence is undefined because all competitors are "
                "excluded as synonyms",
                label_idx,
            )
        return undefined

    competitor_similarities = np.asarray(
        [
            np.dot(
                normalized_embedding,
                np.asarray(class_prototype_full[class_idx], dtype=np.float64),
            )
            for class_idx in eligible_classes
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(competitor_similarities)):
        return undefined
    runner_up_position = int(np.argmax(competitor_similarities))
    runner_up_idx = eligible_classes[runner_up_position]
    runner_up_cos_sim = float(competitor_similarities[runner_up_position])
    runner_up_class = (
        label_classes[runner_up_idx]
        if label_classes is not None and runner_up_idx < len(label_classes)
        else runner_up_idx
    )
    margin = float(self_cos_sim - runner_up_cos_sim)
    confidence = _sigmoid_margin_confidence(margin, logit_scale)
    return {
        "coherence_prototype_cos_sim": self_cos_sim,
        "coherence_runner_up_class": runner_up_class,
        "coherence_runner_up_cos_sim": runner_up_cos_sim,
        "label_coherence_margin": margin,
        "p_coh": confidence,
        "u_coh": float(1.0 - confidence),
    }


def compute_vocabulary_membership(
    embedding,
    text_feats,
    negative_text_feats,
    logit_scale=100.0,
    similarity=None,
    assume_normalized=False,
):
    """Compute confidence that an object belongs to the configured vocabulary.

    Implements P_mem = Z_C / (Z_C + Z_N) with
    Z_C = sum over c in C of exp(alpha cos(v_i, t_c)) and Z_N the same sum over
    the negative bank N. The ratio is evaluated as
    ``sigma(log Z_C - log Z_N)``, which is algebraically identical and keeps
    both partition sums in log space: at alpha = 100 the raw exponentials
    overflow float64 for cosines above ~7.1e-3, so log-sum-exp is required, not
    merely preferable.

    ``text_feats`` are the vocabulary classes and ``negative_text_feats`` are
    the fixed, scene-independent negative bank.  ``similarity`` can be supplied
    when the caller has already computed the vocabulary cosine vector for
    label assignment.
    """
    embedding = np.asarray(embedding, dtype=np.float64).reshape(-1)
    text_feats = np.asarray(text_feats, dtype=np.float64)
    negative_text_feats = np.asarray(negative_text_feats, dtype=np.float64)

    if text_feats.ndim != 2 or text_feats.shape[0] == 0:
        raise ValueError("text_feats must be a non-empty two-dimensional array")
    if negative_text_feats.ndim != 2 or negative_text_feats.shape[0] == 0:
        raise ValueError(
            "negative_text_feats must be a non-empty two-dimensional array"
        )
    if embedding.shape[0] != text_feats.shape[1]:
        raise ValueError("embedding and text_feats must have the same feature dimension")
    if negative_text_feats.shape[1] != embedding.shape[0]:
        raise ValueError(
            "embedding and negative_text_feats must have the same feature dimension"
        )

    embedding_norm = np.linalg.norm(embedding)
    if not np.isfinite(embedding_norm) or embedding_norm < _NORM_EPSILON:
        # Every cos(v_i, .) in eq. (membership) is undefined, so the two
        # partition sums and their ratio are undefined as well. The previous
        # implementation returned P_mem = 0.0, a fabricated probability that
        # claimed certain out-of-vocabulary status for an object that simply
        # has no usable embedding.
        logging.getLogger(__name__).warning(
            "Vocabulary Membership is undefined for a zero-norm or non-finite "
            "embedding"
        )
        return dict.fromkeys(MEMBERSHIP_FIELDS)

    if similarity is None:
        vocab_similarities = compute_cosine_similarities(
            embedding, text_feats, assume_normalized=assume_normalized
        )
    else:
        vocab_similarities = np.asarray(similarity, dtype=np.float64).reshape(-1)
        if vocab_similarities.shape != (text_feats.shape[0],):
            raise ValueError("similarity must have one value per text feature")
    negative_similarities = compute_cosine_similarities(
        embedding, negative_text_feats, assume_normalized=assume_normalized
    )

    alpha = float(logit_scale)
    vocab_log_partition = float(logsumexp(alpha * vocab_similarities))
    negative_log_partition = float(logsumexp(alpha * negative_similarities))
    p_mem = float(expit(vocab_log_partition - negative_log_partition))
    return {
        "vocab_log_partition": vocab_log_partition,
        "negative_log_partition": negative_log_partition,
        "p_mem": p_mem,
        "u_mem": float(1.0 - p_mem),
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
