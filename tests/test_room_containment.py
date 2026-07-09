import numpy as np
import pytest

from hovsg.utils.uncertainty import compute_room_containment_probs


def test_empty_room_equals_prior():
    result = compute_room_containment_probs(np.empty((0, 3)), prior=0.05)

    np.testing.assert_allclose(result, np.full(3, 0.05))


def test_one_object_equals_its_compatibility_without_prior():
    result = compute_room_containment_probs(np.array([[0.8]]), prior=0.0)

    assert result[0] == pytest.approx(0.8)


def test_two_objects_follow_noisy_or():
    result = compute_room_containment_probs(
        np.array([[0.5], [0.4]]),
        prior=0.0,
    )

    assert result[0] == pytest.approx(0.7)


def test_more_support_is_monotonic():
    one_object = compute_room_containment_probs(np.array([[0.5, 0.1]]))
    two_objects = compute_room_containment_probs(
        np.array([[0.5, 0.1], [0.4, 0.2]])
    )

    assert np.all(two_objects >= one_object)
