import numpy as np

from fourmp.sensor_sim.pinhole import PinholeModel


def test_looking_at_axis_points_to_target():
    center = np.array([-10.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 100.0])
    model = PinholeModel.looking_at(center, target, f_i=1000.0, f_j=1000.0, n_i=100, n_j=100)
    assert np.allclose(model.forward, (target - center) / np.linalg.norm(target - center))
    # right/up/forward form a right-handed orthonormal basis.
    assert np.isclose(np.dot(model.right, model.up), 0.0, atol=1e-12)
    assert np.isclose(np.dot(model.right, model.forward), 0.0, atol=1e-12)
    assert np.allclose(np.cross(model.right, model.up), model.forward, atol=1e-9)


def test_principal_index_maps_exactly_to_axis():
    center = np.array([-10.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 100.0])
    model = PinholeModel.looking_at(center, target, f_i=1000.0, f_j=1000.0, n_i=101, n_j=51)
    c_i, c_j = (101 - 1) / 2.0, (51 - 1) / 2.0
    direction = model.direction_for_index(c_i, c_j)
    assert np.allclose(direction, model.forward, atol=1e-12)


def test_project_and_unproject_round_trip():
    center = np.array([-10.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 100.0])
    model = PinholeModel.looking_at(center, target, f_i=800.0, f_j=800.0, n_i=200, n_j=200)

    i, j = 37.0, 162.0
    direction = model.direction_for_index(i, j)
    # A point 50 units out along that ray should project back to (i, j).
    point = center + 50.0 * direction
    pi, pj = model.project_point(point)
    assert np.isclose(pi, i, atol=1e-9)
    assert np.isclose(pj, j, atol=1e-9)


def test_points_behind_pinhole_are_rejected():
    center = np.zeros(3)
    target = np.array([0.0, 0.0, 1.0])
    model = PinholeModel.looking_at(center, target, f_i=100.0, f_j=100.0, n_i=50, n_j=50)
    behind_point = np.array([0.0, 0.0, -10.0])
    i, j = model.pixel_indices(behind_point[None, :])
    assert i[0] == -1 and j[0] == -1


def test_out_of_bounds_pixels_get_sentinel():
    center = np.zeros(3)
    target = np.array([0.0, 0.0, 1.0])
    model = PinholeModel.looking_at(center, target, f_i=10.0, f_j=10.0, n_i=20, n_j=20)
    # Far off to the side -> projects way outside [0, 20).
    far_point = np.array([1000.0, 0.0, 1.0])
    i, j = model.pixel_indices(far_point[None, :])
    assert i[0] == -1 and j[0] == -1
