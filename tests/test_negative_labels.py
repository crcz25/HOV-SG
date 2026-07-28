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
