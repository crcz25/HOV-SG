"""Build and cache the scene-independent negative vocabulary bank."""

import hashlib
import json
import logging
import os

import numpy as np

from hovsg.utils.clip_utils import get_text_feats_multiple_templates


LOGGER = logging.getLogger(__name__)
DEFAULT_LEXICON = "oewn:2025+"
DEFAULT_MIN_NEGATIVE_LABELS = 200

#: Bumped whenever the mining rule changes, so caches built by an older rule
#: are rebuilt instead of being silently reused under new semantics.
MINING_VERSION = 2

#: Longest lemma kept. Open English Wordnet noun lemmas run to full titles of
#: works and events; anything beyond a short compound noun is not a plausible
#: label for a scanned object.
MAX_NEGATIVE_LABEL_WORDS = 3


def _normalise_rows(features):
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return np.divide(
        features,
        norms,
        out=np.zeros_like(features),
        where=norms >= 1e-8,
    )


def _feature_fingerprint(features):
    features = np.ascontiguousarray(np.asarray(features, dtype=np.float32))
    return hashlib.sha256(features.tobytes()).hexdigest()


def is_common_noun_lemma(lemma):
    """Return whether a Wordnet noun lemma is usable as a negative label.

    ``wn.Wordnet.words(pos="n")`` returns every noun in the lexicon, which for
    Open English Wordnet includes proper nouns and named entities: people
    ("1st Baron Beaverbrook"), places ("'s Gravenhage"), dates
    ("15 August 1945") and titles of works ("1 Maccabees"). Those are not
    object categories, so admitting them dilutes the negative bank with
    entries that are not plausible true classes for a scanned object.

    The rule keeps lemmas that are entirely lowercase (Wordnet capitalizes
    proper nouns), alphabetic apart from internal spaces and hyphens, and at
    most :data:`MAX_NEGATIVE_LABEL_WORDS` words long.
    """
    lemma = str(lemma).strip()
    if not lemma or lemma != lemma.lower():
        return False
    words = lemma.split()
    if not 1 <= len(words) <= MAX_NEGATIVE_LABEL_WORDS:
        return False
    return all(word.replace("-", "").isalpha() for word in words)


def _candidate_words(lexicon_name):
    """Return reproducibly ordered common-noun lemmas from Open English Wordnet.

    The result is sorted and deduplicated, so the mined bank depends only on
    the lexicon version and the filter rule -- never on scene content or on
    iteration order.
    """
    try:
        import wn
    except ImportError as exc:  # pragma: no cover - depends on deployment extras
        raise ImportError(
            "Negative-label mining requires the 'wn' package; install it with "
            "'pip install wn'."
        ) from exc

    oewn = wn.Wordnet(lexicon_name)
    lemmas = {word.lemma().replace("_", " ") for word in oewn.words(pos="n")}
    return sorted(lemma for lemma in lemmas if is_common_noun_lemma(lemma))


def _read_cached_bank(cache_path, words_path, metadata_path):
    if not (os.path.exists(cache_path) and os.path.exists(words_path)):
        return None

    try:
        features = np.load(cache_path, allow_pickle=False)
        with open(words_path, encoding="utf-8") as words_file:
            words_payload = json.load(words_file)
        if isinstance(words_payload, dict):
            words = words_payload["words"]
        else:
            words = words_payload
        if features.ndim != 2 or len(words) != features.shape[0]:
            raise ValueError("cached negative feature and word counts differ")
        return features, words
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        LOGGER.warning("Ignoring invalid negative-label cache at %s: %s", cache_path, exc)
        return None


def load_or_build_negative_label_feats(
    clip_model,
    clip_feat_dim,
    label_text_feats,
    max_class_similarity=0.5,
    negative_label_count=None,
    label_feat_path=None,
    lexicon=DEFAULT_LEXICON,
    min_negative_labels=DEFAULT_MIN_NEGATIVE_LABELS,
):
    """Load or mine the fixed negative-label text feature bank.

    Mining is scene-independent and reproducible: candidates are the common
    noun lemmas of ``lexicon`` (see :func:`is_common_noun_lemma`), sorted and
    deduplicated, encoded with the same CLIP text templates as the vocabulary,
    and kept when their maximum cosine similarity to any class in C is below
    ``max_class_similarity``. Nothing about the current scene enters the
    selection.

    The vocabulary features are supplied by the caller, so this utility never
    reloads or recomputes the positive vocabulary.  Cache metadata records the
    threshold, mining rule version and vocabulary fingerprint to avoid silently
    reusing a bank produced for a different configuration.

    Returns rows that are already unit-normalized, so callers pass
    ``assume_normalized=True`` into
    :func:`hovsg.utils.uncertainty.compute_vocabulary_membership`.

    ``negative_label_count`` optionally caps ``|N|``, keeping the candidates
    furthest from every vocabulary class. P_mem does not depend on ``|N|`` --
    eq. (membership) divides each sum by the size of its set -- so the cap only
    bounds the cost of encoding and storing the bank.
    """
    label_text_feats = np.asarray(label_text_feats, dtype=np.float64)
    if label_text_feats.ndim != 2 or label_text_feats.shape[0] == 0:
        raise ValueError("label_text_feats must be a non-empty two-dimensional array")

    max_class_similarity = float(max_class_similarity)
    if not np.isfinite(max_class_similarity):
        raise ValueError("max_class_similarity must be finite")
    if negative_label_count is not None:
        negative_label_count = int(negative_label_count)
        if negative_label_count <= 0:
            raise ValueError("negative_label_count must be positive or null")

    if label_feat_path is None:
        label_feat_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "labels")
        )
    os.makedirs(label_feat_path, exist_ok=True)
    cache_path = os.path.join(label_feat_path, "negative_label_feats.npy")
    words_path = os.path.join(label_feat_path, "negative_label_words.json")
    metadata_path = os.path.join(label_feat_path, "negative_label_metadata.json")

    cache_metadata = None
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, encoding="utf-8") as metadata_file:
                cache_metadata = json.load(metadata_file)
        except (OSError, json.JSONDecodeError):
            cache_metadata = None

    expected_metadata = {
        "lexicon": lexicon,
        "max_class_similarity": max_class_similarity,
        "negative_label_count": negative_label_count,
        "label_feature_fingerprint": _feature_fingerprint(label_text_feats),
        "mining_version": MINING_VERSION,
        "max_negative_label_words": MAX_NEGATIVE_LABEL_WORDS,
        "normalized": True,
    }
    # A missing metadata file means the bank predates this record and its
    # provenance is unknown, so it is rebuilt rather than trusted.
    cache_matches = cache_metadata is not None and all(
        cache_metadata.get(key) == value for key, value in expected_metadata.items()
    )
    if cache_matches:
        cached = _read_cached_bank(cache_path, words_path, metadata_path)
        if cached is not None:
            negative_text_feats, negative_words = cached
            LOGGER.info("Loaded %d cached negative labels from %s", len(negative_words), cache_path)
            if len(negative_words) < min_negative_labels:
                LOGGER.warning(
                    "Negative-label bank contains only %d labels; membership estimates "
                    "may be noisy (recommended minimum: %d)",
                    len(negative_words),
                    min_negative_labels,
                )
            return np.asarray(negative_text_feats)

    candidate_words = _candidate_words(lexicon)
    if not candidate_words:
        raise RuntimeError("Open English Wordnet returned no noun candidates")
    candidate_feats = np.asarray(
        get_text_feats_multiple_templates(candidate_words, clip_model, clip_feat_dim)
    )
    if candidate_feats.ndim != 2 or candidate_feats.shape != (
        len(candidate_words),
        label_text_feats.shape[1],
    ):
        raise ValueError(
            "negative candidate features must have shape "
            f"({len(candidate_words)}, {label_text_feats.shape[1]})"
        )

    normalized_candidates = _normalise_rows(candidate_feats)
    normalized_classes = _normalise_rows(label_text_feats)
    max_class_similarities = (normalized_candidates @ normalized_classes.T).max(axis=1)
    # A zero-norm or non-finite candidate encoding carries no direction, so it
    # would contribute exp(0) to Z_N regardless of the object: drop it rather
    # than let it act as a constant floor on the negative partition.
    usable = np.isfinite(normalized_candidates).all(axis=1) & (
        np.linalg.norm(normalized_candidates, axis=1) > 0.5
    )
    keep = usable & (max_class_similarities < max_class_similarity)
    negative_words = [word for word, include in zip(candidate_words, keep) if include]
    # Stored already normalized so eq. (membership) never re-normalizes ~20k
    # rows per object; callers pass assume_normalized=True.
    negative_text_feats = normalized_candidates[keep]
    if negative_label_count is not None and len(negative_words) > negative_label_count:
        # Truncating the alphabetically sorted list would keep whatever happens
        # to start with "a". Keep the labels furthest from every vocabulary
        # class instead, which is what the bank is selected for, then restore
        # alphabetical order so the stored bank stays deterministic.
        kept_similarities = max_class_similarities[keep]
        chosen = np.sort(np.argsort(kept_similarities, kind="stable")[:negative_label_count])
        negative_words = [negative_words[i] for i in chosen]
        negative_text_feats = negative_text_feats[chosen]
    if len(negative_words) == 0:
        raise RuntimeError("Negative-label filtering produced an empty bank")

    np.save(cache_path, negative_text_feats)
    with open(words_path, "w", encoding="utf-8") as words_file:
        json.dump({"words": negative_words}, words_file, indent=2)
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(expected_metadata, metadata_file, indent=2)

    LOGGER.info(
        "Built negative-label bank with %d/%d candidates at %s",
        len(negative_words),
        len(candidate_words),
        cache_path,
    )
    if len(negative_words) < min_negative_labels:
        LOGGER.warning(
            "Negative-label bank contains only %d labels; membership estimates may be "
            "noisy (recommended minimum: %d)",
            len(negative_words),
            min_negative_labels,
        )
    return negative_text_feats
