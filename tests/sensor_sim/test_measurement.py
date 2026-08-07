import numpy as np

from fourmp.sensor_sim.measurement import run_measurement
from fourmp.sensor_sim.part import load_part


def test_run_measurement_hits_the_posed_face(sensor_config, cube_stl_path):
    part = load_part(cube_stl_path, sensor_config, face_normal_hint=(1.0, 0.0, 0.0))
    scan = run_measurement(part.mesh, sensor_config, step_stride=4, mirror_stride=4)

    assert len(scan) > 0
    assert scan.step.min() >= 0
    assert scan.step.max() < sensor_config.projector.n_i
    assert scan.mirror.min() >= 0
    assert scan.mirror.max() < sensor_config.projector.n_j
    assert scan.cam_i.min() >= 0
    assert scan.cam_i.max() < sensor_config.camera.n_i
    assert scan.cam_j.min() >= 0
    assert scan.cam_j.max() < sensor_config.camera.n_j


def test_run_measurement_no_hits_when_face_moved_out_of_range(sensor_config, cube_stl_path):
    part = load_part(cube_stl_path, sensor_config, face_normal_hint=(1.0, 0.0, 0.0))
    # Shift sideways (perpendicular to the boresight) far past the
    # projector's bounded angular sweep -- unlike shifting along the
    # boresight (Z), which the same finite ray fan would still reach.
    far_away = part.mesh.copy()
    far_away.vertices = far_away.vertices + np.array([5000.0, 0.0, 0.0])

    scan = run_measurement(far_away, sensor_config, step_stride=8, mirror_stride=8)
    assert len(scan) == 0
