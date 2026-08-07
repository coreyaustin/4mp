import numpy as np
import pytest

from fourmp.sensor_sim.geometry import (
    Transform,
    closest_points_between_rays,
    closest_points_between_rays_batch,
    rotation_y,
    unit,
)


def test_unit_normalizes_and_handles_zero():
    v = np.array([3.0, 0.0, 4.0])
    assert np.allclose(unit(v), [0.6, 0.0, 0.8])
    assert np.allclose(unit(np.zeros(3)), np.zeros(3))


def test_rotation_y_is_orthonormal():
    R = rotation_y(0.7)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_rotation_y_matches_theta_face_convention():
    # theta = atan2(-n_x, n_z) should rotate (n_x, 0, n_z) onto +Z.
    n = unit(np.array([1.0, 0.0, 0.3]))
    theta = np.arctan2(-n[0], n[2])
    rotated = rotation_y(theta) @ n
    assert np.allclose(rotated, [0.0, 0.0, 1.0], atol=1e-9)


def test_transform_compose_matches_manual_application():
    t_mount = Transform.translation(np.array([1.0, 2.0, 3.0]))
    rotate_and_place = Transform(rotation_y(np.pi / 2), np.array([10.0, 0.0, 0.0]))
    composed = t_mount.then(rotate_and_place)

    p = np.array([5.0, -1.0, 2.0])
    expected = rotate_and_place.apply_points(t_mount.apply_points(p))
    assert np.allclose(composed.apply_points(p), expected)


def test_closest_points_between_rays_exact_intersection():
    # Two rays that genuinely cross at (0, 0, 5).
    o1, d1 = np.array([-5.0, 0.0, 0.0]), unit(np.array([1.0, 0.0, 1.0]))
    o2, d2 = np.array([5.0, 0.0, 0.0]), unit(np.array([-1.0, 0.0, 1.0]))
    point, gap = closest_points_between_rays(o1, d1, o2, d2)
    assert gap < 1e-9
    assert np.allclose(point, [0.0, 0.0, 5.0], atol=1e-9)


def test_closest_points_between_rays_skew_gap_is_positive():
    o1, d1 = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    o2, d2 = np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    point, gap = closest_points_between_rays(o1, d1, o2, d2)
    assert gap == pytest.approx(1.0)


def test_closest_points_between_rays_batch_matches_scalar():
    o1 = np.array([-5.0, 0.0, 0.0])
    o2 = np.array([5.0, 0.0, 0.0])
    d1s = np.stack([unit(np.array([1.0, 0.0, 1.0])), unit(np.array([1.0, 0.0, 2.0]))])
    d2s = np.stack([unit(np.array([-1.0, 0.0, 1.0])), unit(np.array([-1.0, 0.1, 2.0]))])

    points, gaps = closest_points_between_rays_batch(o1, d1s, o2, d2s)
    for k in range(2):
        p, g = closest_points_between_rays(o1, d1s[k], o2, d2s[k])
        assert np.allclose(points[k], p)
        assert gaps[k] == pytest.approx(g)
