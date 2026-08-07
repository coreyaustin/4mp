import numpy as np

from fourmp.sensor_sim.measurement import run_measurement
from fourmp.sensor_sim.part import load_part
from fourmp.sensor_sim.reconstruction import run_reconstruction


def test_reconstruction_recovers_known_flat_face_height(sensor_config, cube_stl_path):
    part = load_part(cube_stl_path, sensor_config, face_normal_hint=(1.0, 0.0, 0.0))
    scan = run_measurement(part.mesh, sensor_config, step_stride=4, mirror_stride=4)
    result = run_reconstruction(scan, sensor_config)

    valid = ~np.isnan(result.height_map)
    assert valid.sum() > 0
    # Known geometry: face sits at Z = working_distance + 50mm (see
    # part.py/test_part.py), so every reconstructed height should be ~+50mm,
    # up to camera-pixel quantization (sub-pixel-scale, well under 1mm here).
    heights = result.height_map[valid]
    assert np.abs(heights.mean() - 50.0) < 0.1
    assert np.abs(heights - 50.0).max() < 1.0

    # Triangulation gap (projector ray vs. back-projected camera ray) should
    # be small -- a large gap would indicate a calibration mismatch between
    # the two models rather than real quantization noise.
    gaps = result.triangulation_gap_mm[valid]
    assert gaps.max() < 1.0


def test_reconstruction_empty_scan_gives_all_nan(sensor_config):
    from fourmp.sensor_sim.measurement import ScanData

    empty = ScanData(
        step=np.array([], dtype=int),
        mirror=np.array([], dtype=int),
        cam_i=np.array([], dtype=int),
        cam_j=np.array([], dtype=int),
    )
    result = run_reconstruction(empty, sensor_config)
    assert np.isnan(result.height_map).all()
    assert result.collisions == 0


def test_centered_index_convention(sensor_config):
    from fourmp.sensor_sim.measurement import ScanData

    empty = ScanData(
        step=np.array([], dtype=int),
        mirror=np.array([], dtype=int),
        cam_i=np.array([], dtype=int),
        cam_j=np.array([], dtype=int),
    )
    result = run_reconstruction(empty, sensor_config)
    n_i, n_j = result.height_map.shape

    # Corner pixel (0, 0) should report a large-negative centered index...
    row0, col0 = result.centered_index(0, 0)
    assert row0 < 0 and col0 < 0

    # ...and the opposite corner an equal-magnitude positive one.
    row_last, col_last = result.centered_index(n_i - 1, n_j - 1)
    assert row_last == -row0
    assert col_last == -col0
