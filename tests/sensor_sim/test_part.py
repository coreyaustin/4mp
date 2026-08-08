import numpy as np
import pytest
import trimesh

from fourmp.sensor_sim.part import compute_theta_face, load_part


def _angle_delta_deg(a: float, b: float) -> float:
    """Smallest signed difference between two angles in degrees, handling
    the +-180 wraparound (atan2's sign at the branch cut depends on the sign
    of zero, which isn't worth pinning down in a test)."""
    return (a - b + 180.0) % 360.0 - 180.0


def test_theta_face_zero_for_normal_already_on_boresight():
    # Boresight is O_s's -X (see geometry.py's "depth = -X" convention), so
    # the in-plane normal that needs no rotation is (-1, 0, 0).
    assert compute_theta_face(np.array([-1.0, 0.0, 0.0])) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "normal,expected_deg",
    [
        ((1.0, 0.0, 0.0), 180.0),
        ((-1.0, 0.0, 0.0), 0.0),
        ((0.0, 1.0, 0.0), 90.0),
        ((0.0, -1.0, 0.0), -90.0),
    ],
)
def test_theta_face_in_plane_normals(normal, expected_deg):
    theta = compute_theta_face(np.array(normal))
    assert _angle_delta_deg(np.degrees(theta), expected_deg) == pytest.approx(0.0, abs=1e-9)


def test_theta_face_rejects_are_handled_by_face_selection(sensor_config, cube_stl_path):
    # The rotation axis is O_s's Z (V1.3) -- a face whose normal points that
    # way can't be rotated into view by a single-axis stage and must be
    # rejected, not silently posed.
    with pytest.raises(ValueError):
        load_part(cube_stl_path, sensor_config, face_normal_hint=(0.0, 0.0, 1.0))


def test_load_part_cube_side_face(sensor_config, cube_stl_path):
    part = load_part(cube_stl_path, sensor_config, face_normal_hint=(1.0, 0.0, 0.0))

    assert np.allclose(part.face_frame.normal, [1.0, 0.0, 0.0])
    assert _angle_delta_deg(np.degrees(part.theta_face_rad), 180.0) == pytest.approx(0.0, abs=1e-6)

    # Posed face should be planar, perpendicular to the boresight (constant
    # X in O_s), at X = -(working_distance + half the cube side) -- O_s's
    # depth convention is "depth = -X" (see geometry.py).
    x_values = part.mesh.vertices[:, 0]
    assert np.allclose(x_values, x_values[0], atol=1e-9)
    assert x_values[0] == pytest.approx(-(sensor_config.working_distance_mm + 50.0), abs=1e-6)

    # Centered on the boresight/baseline axis (O_s's Y).
    assert part.mesh.vertices[:, 1].mean() == pytest.approx(0.0, abs=1e-6)


def test_load_part_different_faces_give_different_rotations(sensor_config, cube_stl_path):
    part_a = load_part(cube_stl_path, sensor_config, face_normal_hint=(1.0, 0.0, 0.0))
    part_b = load_part(cube_stl_path, sensor_config, face_normal_hint=(0.0, 1.0, 0.0))
    assert part_a.theta_face_rad != pytest.approx(part_b.theta_face_rad)
    # Both should still end up facing the boresight after posing (constant X).
    for part in (part_a, part_b):
        x_values = part.mesh.vertices[:, 0]
        assert np.allclose(x_values, x_values[0], atol=1e-9)


def test_o_r_from_o_s_puts_height_on_y_and_extent_on_x_and_z(sensor_config, cube_stl_path):
    # The whole point of O_r (V1.3): for a flat face posed to face the
    # sensor, height (O_r's Y) should come out ~constant, and the two
    # in-plane axes (X, Z) should span the face's actual extent -- not the
    # other way around. This is the regression case for the bug where
    # O_r's rotation was accidentally tied to theta_face (see compute_pose's
    # docstring): that made X constant and Y/Z carry the in-plane spread.
    part = load_part(cube_stl_path, sensor_config, face_normal_hint=(1.0, 0.0, 0.0))
    points_or = part.o_r_from_o_s.apply_points(part.mesh.vertices)

    y_or = points_or[:, 1]
    assert np.allclose(y_or, y_or[0], atol=1e-9)
    # Known geometry (see test_load_part_cube_side_face): the face sits
    # 50mm beyond the reference plane.
    assert y_or[0] == pytest.approx(50.0, abs=1e-6)

    x_or, z_or = points_or[:, 0], points_or[:, 2]
    assert x_or.max() - x_or.min() == pytest.approx(100.0, abs=1e-6)
    assert z_or.max() - z_or.min() == pytest.approx(100.0, abs=1e-6)


# ---- V1.2: up-axis remap -------------------------------------------------
# The cube fixture is symmetric under axis relabeling, so it can't exercise
# a genuine up-axis bug (this is what the spec's V1.2 section notes --
# surfaced by testing a non-cube part). These tests use an asymmetric plate
# instead, so a wrong remap would actually change which faces are
# selectable/axial.


@pytest.fixture
def plate_stl_path(tmp_path):
    mesh = trimesh.creation.box(extents=(200.0, 100.0, 50.0))
    path = tmp_path / "plate.stl"
    mesh.export(str(path))
    return path


def test_up_axis_default_z_up_rejects_top_face(sensor_config, plate_stl_path):
    # Default up_axis=(0,0,1): the file's Z (its shortest dimension here) is
    # up, so its normal is axial and must be rejected.
    with pytest.raises(ValueError):
        load_part(plate_stl_path, sensor_config, face_normal_hint=(0.0, 0.0, 1.0))


def test_up_axis_default_z_up_accepts_side_face(sensor_config, plate_stl_path):
    part = load_part(plate_stl_path, sensor_config, face_normal_hint=(1.0, 0.0, 0.0))
    assert np.allclose(part.face_frame.normal, [1.0, 0.0, 0.0])


def test_up_axis_changes_pose_for_the_same_hint(sensor_config, plate_stl_path):
    # A cardinal hint against this axis-aligned, cardinal-up_axis plate
    # always resolves to *some* cardinal internal normal either way, so
    # accept/reject alone can't distinguish the two remaps here. T_mount can:
    # it depends on which file axis was declared "up" (bottom-flush centers
    # a different pair of bounding-box axes), and the plate's extents
    # (200, 100, 50) are all different, so the two remaps must disagree.
    part_default = load_part(plate_stl_path, sensor_config, face_normal_hint=(1.0, 0.0, 0.0))
    part_y_up = load_part(
        plate_stl_path, sensor_config, face_normal_hint=(1.0, 0.0, 0.0), up_axis=(0.0, 1.0, 0.0)
    )
    assert not np.allclose(part_default.T_mount.t, part_y_up.T_mount.t)
