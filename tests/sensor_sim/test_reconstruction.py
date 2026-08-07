import numpy as np

from fourmp.sensor_sim.measurement import ScanData, run_measurement
from fourmp.sensor_sim.part import load_part
from fourmp.sensor_sim.reconstruction import run_reconstruction


def _empty_scan_data() -> ScanData:
    empty_int = np.array([], dtype=int)
    empty_float = np.array([], dtype=float)
    return ScanData(
        step=empty_int,
        mirror=empty_int,
        cam_i=empty_int,
        cam_j=empty_int,
        cam_i_continuous=empty_float,
        cam_j_continuous=empty_float,
    )


def test_reconstruction_recovers_known_flat_face_height(sensor_config, cube_stl_path):
    part = load_part(cube_stl_path, sensor_config, face_normal_hint=(1.0, 0.0, 0.0))
    scan = run_measurement(part.mesh, sensor_config, step_stride=4, mirror_stride=4)
    result = run_reconstruction(scan, sensor_config)

    assert len(result.heights) > 0
    # Known geometry: face sits at Z = working_distance + 50mm (see
    # part.py/test_part.py). V1.1 triangulates from the camera's continuous
    # sub-pixel projection (not rounded to the nearest pixel), so the
    # forward/inverse round trip is geometrically exact again -- residual
    # should be floating-point noise, not sub-pixel quantization error.
    assert np.abs(result.heights.mean() - 50.0) < 1e-6
    assert np.abs(result.heights - 50.0).max() < 1e-6

    # Triangulation gap (projector ray vs. back-projected camera ray) should
    # likewise collapse to floating-point noise.
    assert result.gaps.max() < 1e-6

    # The camera-pixel-native diagnostic view should still agree.
    valid = ~np.isnan(result.height_map)
    assert valid.sum() > 0
    assert np.abs(result.height_map[valid] - 50.0).max() < 1e-6


def test_reconstruction_empty_scan_gives_all_nan(sensor_config):
    result = run_reconstruction(_empty_scan_data(), sensor_config)
    assert np.isnan(result.height_map).all()
    assert len(result.points) == 0
    assert len(result.heights) == 0
    assert result.collisions == 0


def test_centered_index_convention(sensor_config):
    result = run_reconstruction(_empty_scan_data(), sensor_config)
    n_i, n_j = result.height_map.shape

    # Corner pixel (0, 0) should report a large-negative centered index...
    row0, col0 = result.centered_index(0, 0)
    assert row0 < 0 and col0 < 0

    # ...and the opposite corner an equal-magnitude positive one.
    row_last, col_last = result.centered_index(n_i - 1, n_j - 1)
    assert row_last == -row0
    assert col_last == -col0
