import inspect
import json

import numpy as np

from hovsg.utils import negative_labels


def test_negative_label_bank_filters_against_existing_vocabulary_and_caches(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        negative_labels,
        "_candidate_words",
        lambda _lexicon: ["near class", "unrelated thing"],
    )
    monkeypatch.setattr(
        negative_labels,
        "get_text_feats_multiple_templates",
        lambda words, _model, _dim: np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]][: len(words)]
        ),
    )

    bank = negative_labels.load_or_build_negative_label_feats(
        None,
        3,
        np.eye(3)[:2],
        max_class_similarity=0.5,
        label_feat_path=str(tmp_path),
        min_negative_labels=1,
    )

    np.testing.assert_array_equal(bank, np.array([[0.0, 0.0, 1.0]]))
    assert (tmp_path / "negative_label_feats.npy").exists()
    assert (tmp_path / "negative_label_words.json").exists()

    monkeypatch.setattr(
        negative_labels,
        "_candidate_words",
        lambda _lexicon: (_ for _ in ()).throw(AssertionError("cache was not used")),
    )
    cached = negative_labels.load_or_build_negative_label_feats(
        None,
        3,
        np.eye(3)[:2],
        max_class_similarity=0.5,
        label_feat_path=str(tmp_path),
        min_negative_labels=1,
    )
    np.testing.assert_array_equal(cached, bank)


def test_proper_nouns_and_long_titles_are_rejected_as_negative_labels():
    """Open English Wordnet nouns include named entities, dates and titles.

    Those are not object categories, and admitting ~20k of them both dilutes
    the bank and depresses P_mem for every object, because eq. (membership)
    compares partition *sums* over C and N.
    """
    rejected = [
        "'s Gravenhage",
        "1 Maccabees",
        "15 August 1945",
        "1st Baron Beaverbrook",
        "4.5-Inch Beach Barrage Rocket",
        "Apollo",
        "United States of America",
        "",
    ]
    accepted = ["telescope", "fire hydrant", "sewing machine", "x-ray"]

    for lemma in rejected:
        assert not negative_labels.is_common_noun_lemma(lemma), lemma
    for lemma in accepted:
        assert negative_labels.is_common_noun_lemma(lemma), lemma


def test_cached_bank_is_stored_normalized(monkeypatch, tmp_path):
    monkeypatch.setattr(
        negative_labels, "_candidate_words", lambda _lexicon: ["thing", "other"]
    )
    monkeypatch.setattr(
        negative_labels,
        "get_text_feats_multiple_templates",
        # Deliberately unnormalized encoder output.
        lambda words, _model, _dim: np.array([[0.0, 0.0, 3.0], [0.0, 4.0, 0.0]]),
    )

    bank = negative_labels.load_or_build_negative_label_feats(
        None,
        3,
        np.eye(3)[:1],
        max_class_similarity=0.5,
        label_feat_path=str(tmp_path),
        min_negative_labels=1,
    )

    np.testing.assert_allclose(np.linalg.norm(bank, axis=1), 1.0)


def test_a_stale_cache_is_rebuilt_when_the_mining_rule_changes(monkeypatch, tmp_path):
    calls = []

    def candidates(_lexicon):
        calls.append(1)
        return ["thing"]

    monkeypatch.setattr(negative_labels, "_candidate_words", candidates)
    monkeypatch.setattr(
        negative_labels,
        "get_text_feats_multiple_templates",
        lambda words, _model, _dim: np.array([[0.0, 0.0, 1.0]]),
    )
    kwargs = dict(
        max_class_similarity=0.5, label_feat_path=str(tmp_path), min_negative_labels=1
    )

    negative_labels.load_or_build_negative_label_feats(None, 3, np.eye(3)[:1], **kwargs)
    negative_labels.load_or_build_negative_label_feats(None, 3, np.eye(3)[:1], **kwargs)
    assert len(calls) == 1  # second call served from cache

    metadata_path = tmp_path / "negative_label_metadata.json"
    stale = json.loads(metadata_path.read_text(encoding="utf-8"))
    stale["mining_version"] = negative_labels.MINING_VERSION - 1
    metadata_path.write_text(json.dumps(stale), encoding="utf-8")

    negative_labels.load_or_build_negative_label_feats(None, 3, np.eye(3)[:1], **kwargs)
    assert len(calls) == 2  # rebuilt rather than silently reused


def test_a_bank_without_metadata_is_not_trusted(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        negative_labels,
        "_candidate_words",
        lambda _lexicon: (calls.append(1), ["thing"])[1],
    )
    monkeypatch.setattr(
        negative_labels,
        "get_text_feats_multiple_templates",
        lambda words, _model, _dim: np.array([[0.0, 0.0, 1.0]]),
    )
    kwargs = dict(
        max_class_similarity=0.5, label_feat_path=str(tmp_path), min_negative_labels=1
    )

    negative_labels.load_or_build_negative_label_feats(None, 3, np.eye(3)[:1], **kwargs)
    (tmp_path / "negative_label_metadata.json").unlink()
    negative_labels.load_or_build_negative_label_feats(None, 3, np.eye(3)[:1], **kwargs)

    assert len(calls) == 2


def test_count_limit_keeps_the_labels_furthest_from_the_vocabulary(monkeypatch, tmp_path):
    # Alphabetically first is the *closest* to the vocabulary; truncating the
    # sorted list would keep exactly the wrong one.
    monkeypatch.setattr(
        negative_labels, "_candidate_words", lambda _lexicon: ["aardvark", "zebra"]
    )
    monkeypatch.setattr(
        negative_labels,
        "get_text_feats_multiple_templates",
        lambda words, _model, _dim: np.array([[0.4, 0.917, 0.0], [0.0, 0.0, 1.0]]),
    )

    bank = negative_labels.load_or_build_negative_label_feats(
        None,
        3,
        np.eye(3)[:1],
        max_class_similarity=0.5,
        negative_label_count=1,
        label_feat_path=str(tmp_path),
        min_negative_labels=1,
    )
    words = json.loads(
        (tmp_path / "negative_label_words.json").read_text(encoding="utf-8")
    )["words"]

    assert words == ["zebra"]
    np.testing.assert_allclose(bank, [[0.0, 0.0, 1.0]])


def test_scene_independence_the_bank_depends_only_on_the_vocabulary(monkeypatch, tmp_path):
    """Nothing about the objects in a scene enters negative-label selection."""
    monkeypatch.setattr(
        negative_labels, "_candidate_words", lambda _lexicon: ["alpha", "beta"]
    )
    monkeypatch.setattr(
        negative_labels,
        "get_text_feats_multiple_templates",
        lambda words, _model, _dim: np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    signature = inspect.signature(negative_labels.load_or_build_negative_label_feats)

    # The API offers no way to pass scene or object data in.
    assert set(signature.parameters) == {
        "clip_model",
        "clip_feat_dim",
        "label_text_feats",
        "max_class_similarity",
        "negative_label_count",
        "label_feat_path",
        "lexicon",
        "min_negative_labels",
    }
