import numpy as np
import pytest

from fourmp.sensor_sim.part import compute_theta_face, load_part


def test_theta_face_zero_for_normal_already_on_boresight():
    assert compute_theta_face(np.array([0.0, 0.0, 1.0])) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "normal,expected_deg",
    [
        ((1.0, 0.0, 0.0), -90.0),
        ((-1.0, 0.0, 0.0), 90.0),
        ((0.0, 0.0, -1.0), 180.0),
    ],
)
def test_theta_face_side_normals(normal, expected_deg):
    theta = compute_theta_face(np.array(normal))
    # +-180 deg are the same rotation; atan2's sign for the (nx=0, nz=-1)
    # case depends on the sign of zero, so compare angles modulo 360.
    delta = (np.degrees(theta) - expected_deg + 180.0) % 360.0 - 180.0
    assert delta == pytest.approx(0.0, abs=1e-9)


def test_theta_face_rejects_are_handled_by_face_selection(sensor_config, cube_stl_path):
    # The +Y (top) face is out of scope -- selecting it should fail, not
    # silently produce a bogus pose.
    with pytest.raises(ValueError):
        load_part(cube_stl_path, sensor_config, face_normal_hint=(0.0, 1.0, 0.0))


def test_load_part_cube_side_face(sensor_config, cube_stl_path):
    part = load_part(cube_stl_path, sensor_config, face_normal_hint=(1.0, 0.0, 0.0))

    assert np.allclose(part.face_frame.normal, [1.0, 0.0, 0.0])
    assert np.degrees(part.theta_face_rad) == pytest.approx(-90.0)

    # Posed face should be planar, perpendicular to boresight (constant Z),
    # at Z = working_distance + half the cube side (see part.py docstring).
    z_values = part.mesh.vertices[:, 2]
    assert np.allclose(z_values, z_values[0], atol=1e-9)
    assert z_values[0] == pytest.approx(sensor_config.working_distance_mm + 50.0, abs=1e-6)

    # Centered on the boresight/baseline axis.
    assert part.mesh.vertices[:, 0].mean() == pytest.approx(0.0, abs=1e-6)


def test_load_part_different_faces_give_different_rotations(sensor_config, cube_stl_path):
    part_a = load_part(cube_stl_path, sensor_config, face_normal_hint=(1.0, 0.0, 0.0))
    part_b = load_part(cube_stl_path, sensor_config, face_normal_hint=(0.0, 0.0, 1.0))
    assert part_a.theta_face_rad != pytest.approx(part_b.theta_face_rad)
    # Both should still end up facing the boresight after posing.
    for part in (part_a, part_b):
        z_values = part.mesh.vertices[:, 2]
        assert np.allclose(z_values, z_values[0], atol=1e-9)
