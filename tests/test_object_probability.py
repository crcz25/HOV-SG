import json
import pathlib

import numpy as np
import pytest

from hovsg.graph.object import Object
from hovsg.utils.uncertainty import compute_object_probability


def test_worked_object_probability_uses_the_conservative_semantic_factor():
    result = compute_object_probability(
        p_det=0.8,
        p_view=0.75,
        p_mem=0.9,
        p_sem=0.6,
        p_coh=0.5,
        cross_view_implemented=True,
    )

    assert result == pytest.approx(0.8 * 0.75 * 0.9 * 0.5)


def test_structural_absence_omits_cross_view_factor():
    result = compute_object_probability(
        0.8, None, 0.9, 0.6, 0.5, cross_view_implemented=False
    )

    assert result == pytest.approx(0.8 * 0.9 * 0.5)


def test_per_object_undefined_cross_view_propagates_when_provider_exists():
    result = compute_object_probability(
        0.8, None, 0.9, 0.6, 0.5, cross_view_implemented=True
    )

    assert result is None


def test_undefined_coherence_reduces_to_semantic_probability():
    result = compute_object_probability(
        0.8, 0.75, 0.9, 0.6, None, cross_view_implemented=True
    )

    assert result == pytest.approx(0.8 * 0.75 * 0.9 * 0.6)


@pytest.mark.parametrize("missing", ["p_det", "p_mem", "p_sem"])
def test_required_undefined_factor_propagates(missing):
    values = {
        "p_det": 0.8,
        "p_view": 0.75,
        "p_mem": 0.9,
        "p_sem": 0.6,
        "p_coh": 0.5,
    }
    values[missing] = None

    assert compute_object_probability(**values, cross_view_implemented=True) is None


def test_object_probability_stays_in_range_for_defined_inputs():
    rng = np.random.default_rng(23)
    for _ in range(1000):
        values = rng.random(5)
        result = compute_object_probability(*values, cross_view_implemented=True)
        assert 0.0 <= result <= 1.0


class FakeBoundingBox:
    def get_box_points(self):
        return np.zeros((8, 3))


class FakePointCloud:
    def is_empty(self):
        return False

    def __iadd__(self, other):
        return self

    def get_axis_aligned_bounding_box(self):
        return FakeBoundingBox()


def make_object(object_id, probability):
    obj = Object(object_id, "0_0", name="chair")
    obj.pcd = FakePointCloud()
    obj.embedding = np.array([1.0, 0.0])
    obj.detection_conf_sum = 0.8
    obj.detection_point_count = 1
    obj.p_det = 0.8
    obj.p_view = 0.75
    obj.p_mem = 0.9
    obj.p_sem = 0.6
    obj.p_coh = 0.5
    obj.p_obj = probability
    obj.u_obj = 1.0 - probability if probability is not None else None
    return obj


def test_merge_clears_object_probability_instead_of_combining_it():
    left = make_object("0_0_0", 0.12)
    right = make_object("0_0_1", 0.84)

    merged = left + right

    assert merged.p_obj is None
    assert merged.u_obj is None


def test_object_probability_metadata_round_trip_including_none(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("hovsg.graph.object.o3d.io.write_point_cloud", lambda *_: True)
    monkeypatch.setattr("hovsg.graph.object.o3d.io.read_point_cloud", lambda *_: object())

    source = make_object("0_0_0", 0.123)
    source.pcd = object()
    source.vertices = np.zeros((2, 3))
    source.save(tmp_path)

    saved = json.loads((tmp_path / "0_0_0.json").read_text(encoding="utf-8"))
    assert saved["p_obj"] == pytest.approx(0.123)
    assert saved["u_obj"] == pytest.approx(0.877)

    restored = Object("0_0_0", "0_0")
    restored.load(str(tmp_path))
    assert restored.p_obj == pytest.approx(0.123)
    assert restored.u_obj == pytest.approx(0.877)

    source.p_obj = None
    source.u_obj = None
    source.save(tmp_path)
    restored.load(str(tmp_path))
    assert restored.p_obj is None
    assert restored.u_obj is None


def test_implementation_does_not_use_endpoint_probabilities_for_missing_values():
    source = pathlib.Path(compute_object_probability.__code__.co_filename)
    text = source.read_text(encoding="utf-8")
    assert "p_obj = 0.0" not in text
    assert "p_obj = 1.0" not in text
