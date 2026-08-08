import numpy as np
import pytest

from fourmp.sensor_sim.geometry import (
    Transform,
    axis_alignment_rotation,
    closest_points_between_rays,
    closest_points_between_rays_batch,
    rotation_z,
    unit,
)


def test_unit_normalizes_and_handles_zero():
    v = np.array([3.0, 0.0, 4.0])
    assert np.allclose(unit(v), [0.6, 0.0, 0.8])
    assert np.allclose(unit(np.zeros(3)), np.zeros(3))


def test_rotation_z_is_orthonormal():
    R = rotation_z(0.7)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_rotation_z_matches_theta_face_convention():
    # theta = atan2(n_y, -n_x) should rotate (n_x, n_y, 0) onto (-r, 0, 0).
    n = unit(np.array([1.0, 0.3, 0.0]))
    theta = np.arctan2(n[1], -n[0])
    rotated = rotation_z(theta) @ n
    assert np.allclose(rotated, [-1.0, 0.0, 0.0], atol=1e-9)


def test_transform_compose_matches_manual_application():
    t_mount = Transform.translation(np.array([1.0, 2.0, 3.0]))
    rotate_and_place = Transform(rotation_z(np.pi / 2), np.array([10.0, 0.0, 0.0]))
    composed = t_mount.then(rotate_and_place)

    p = np.array([5.0, -1.0, 2.0])
    expected = rotate_and_place.apply_points(t_mount.apply_points(p))
    assert np.allclose(composed.apply_points(p), expected)


def test_transform_invert_round_trips():
    t = Transform(rotation_z(0.9), np.array([3.0, -2.0, 7.0]))
    p = np.array([1.5, 2.5, -4.0])
    assert np.allclose(t.invert().apply_points(t.apply_points(p)), p, atol=1e-10)


def test_transform_invert_matches_doc_formula():
    # T(B<-A) from T(A<-B): R(B<-A) = R^T, t(B<-A) = -R^T . t
    t = Transform(rotation_z(0.4), np.array([1.0, 2.0, 3.0]))
    inv = t.invert()
    assert np.allclose(inv.R, t.R.T)
    assert np.allclose(inv.t, -t.R.T @ t.t)


def test_to_quat_trans_reconstructs_the_rotation():
    from scipy.spatial.transform import Rotation

    t = Transform(rotation_z(1.1), np.array([4.0, 5.0, 6.0]))
    quat, translation = t.to_quat_trans()
    assert np.allclose(Rotation.from_quat(quat).as_matrix(), t.R, atol=1e-10)
    assert np.allclose(translation, t.t)


def test_axis_alignment_rotation_maps_source_onto_target():
    source = unit(np.array([0.3, 0.6, 0.1]))
    target = unit(np.array([-0.2, 0.1, 0.9]))
    R = axis_alignment_rotation(source, target)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ source, target, atol=1e-10)


def test_axis_alignment_rotation_default_up_axis_case():
    # V1.2's default: part file's +Z-up onto the internal "physical up"
    # direction, -Z in O_s -- an antiparallel (180 deg) degenerate case.
    R = axis_alignment_rotation((0.0, 0.0, 1.0), (0.0, 0.0, -1.0))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, -1.0], atol=1e-10)


def test_axis_alignment_rotation_identity_case():
    R = axis_alignment_rotation((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert np.allclose(R, np.eye(3))


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
