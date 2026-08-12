"""Side-effect-free utilities for semantic and containment uncertainty."""

import logging
from typing import Iterable, Optional

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
    "vocab_log_likelihood",
    "negative_log_likelihood",
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

def combine_semantic_confidence(
    p_sem: Optional[float], p_coh: Optional[float]
) -> Optional[float]:
    """Combine the two label-error estimators, eq. (coherence).

    P_sem (against text references) and P_coh (against in-map visual
    prototypes) estimate the same labeling-error event, so they are combined as
    the minimum rather than multiplied as independent factors::

        P_sem_bar = min(P_sem, P_coh)

    and a low value from either lowers the confidence. When the object's class
    has no other instance in the map, P_coh is undefined and P_sem_bar reduces
    to P_sem. P_sem itself being undefined leaves the pair undefined: the paper
    defines no reduction for that direction.
    """
    if p_sem is None:
        return None
    if p_coh is None:
        return float(p_sem)
    return float(min(p_sem, p_coh))


def compute_object_probability(
    p_det: Optional[float],
    p_view: Optional[float],
    p_mem: Optional[float],
    p_sem_bar: Optional[float],
    cross_view_implemented: bool,
) -> Optional[float]:
    """Fuse the object-level probability from the four signal providers.

    ``p_sem_bar`` is the combined label-error estimator of eq. (coherence), so
    Semantic Uncertainty and Label Coherence enter as a single factor rather
    than as two independent ones. Missing required evidence propagates as
    ``None``.  The only neutral-value reduction is structural absence of the
    cross-view provider, where its factor is omitted from the product.
    """
    required = (p_det, p_mem, p_sem_bar)
    if any(value is None for value in required):
        return None
    if cross_view_implemented and p_view is None:
        return None

    factors = [p_det, p_mem, p_sem_bar]
    if cross_view_implemented:
        factors.append(p_view)
    return float(np.prod(np.asarray(factors, dtype=np.float64)))


def compute_class_containment_belief(
    object_probabilities: Iterable[float],
) -> Optional[float]:
    """Propagate object probabilities to one room-level class belief.

    This is eq. (noisyor) of the propagation section, evaluated exactly as
    written::

        b(r, c) = 1 - prod_{o_i in O(r, c)} (1 - q_i)

    ``object_probabilities`` are the fused object probabilities q_i of
    eq. (obj-prob) for O(r, c), the objects assigned to room r that the
    pipeline labeled c.  The result is the probability that at least one of
    them is a genuine instance of c, under the paper's assumption that the
    objects are independent, and lies in [0, 1] because every q_i does.

    One object reduces the product to a single factor and returns q_i itself.
    Each further object can only raise the belief, since every factor
    (1 - q_i) is at most one.  An empty O(r, c) returns ``None``: with no
    object labeled c the room carries no evidence about c, which is not the
    same claim as the belief 0 the empty product would produce.
    """
    probabilities = np.asarray(list(object_probabilities), dtype=np.float64)
    if probabilities.size == 0:
        return None
    return float(1.0 - np.prod(1.0 - probabilities))


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

    Implements m'_i = cos(v_i, mu_l) - max over classes c instantiated in the
    map with cos(t_c, t_l) < tau of cos(v_i, mu_c), then P_coh = sigma(alpha *
    m'_i) at the same logit scale and the same tau as eq. (semantic). The
    prototype mu_l of the object's own class excludes v_i itself, which is what
    the class sums and counts are passed in for; the signal is undefined when
    l_i has no other instance.

    The class sums and prototypes are supplied by the caller so they can be
    built once for the complete object set.  Text features are
    deliberately absent from this function: ``eligibility_mask`` is derived
    from the already-cached text vocabulary and is used only to remove
    near-synonym competitor classes.  Label Coherence and Semantic
    Uncertainty estimate the same labeling-error event and are combined by
    :func:`combine_semantic_confidence`, never multiplied.
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

    Implements eq. (membership), P_mem = L_C / (L_C + L_N) with the two
    *size-normalized* likelihoods

        L_C = (1 / |C|) sum over c in C of exp(alpha cos(v_i, t_c)),
        L_N = (1 / |N|) sum over t in N of exp(alpha cos(v_i, t)),

    the division by |C| and |N| being what removes the dependence of the
    posterior on the number of words in each set. The ratio is evaluated as
    ``sigma(log L_C - log L_N)``, which is algebraically identical and keeps
    both terms in log space: at alpha = 100 the raw exponentials overflow
    float64 for cosines above ~7.1e-3, so log-sum-exp is required, not merely
    preferable.

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
        # Every cos(v_i, .) in eq. (membership) is undefined, so both
        # likelihoods and their ratio are undefined as well. Reporting
        # P_mem = 0.0 would be a fabricated probability claiming certain
        # out-of-vocabulary status for an object that simply has no usable
        # embedding.
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
    # log of the size-normalized sums: log((1/K) sum exp(.)) = logsumexp(.) - log K.
    vocab_log_likelihood = float(
        logsumexp(alpha * vocab_similarities) - np.log(vocab_similarities.size)
    )
    negative_log_likelihood = float(
        logsumexp(alpha * negative_similarities) - np.log(negative_similarities.size)
    )
    p_mem = float(expit(vocab_log_likelihood - negative_log_likelihood))
    return {
        "vocab_log_likelihood": vocab_log_likelihood,
        "negative_log_likelihood": negative_log_likelihood,
        "p_mem": p_mem,
        "u_mem": float(1.0 - p_mem),
    }


def _validate_label_idx(label_idx, num_classes):
    """Return a Python integer after validating a vocabulary row index."""
    label_idx = int(label_idx)
    if not 0 <= label_idx < num_classes:
        raise IndexError("label_idx is outside the text feature vocabulary")
    return label_idx
