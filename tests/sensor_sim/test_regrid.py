import numpy as np
import pytest

from fourmp.sensor_sim.measurement import run_measurement
from fourmp.sensor_sim.part import load_part
from fourmp.sensor_sim.reconstruction import run_reconstruction
from fourmp.sensor_sim.regrid import (
    bin_values_to_grid,
    make_grid_spec,
    pixel_footprint_mm,
    regrid_reconstruction_and_truth,
    sample_true_height_on_grid,
)


def test_pixel_footprint_is_in_the_expected_60_to_70um_range(sensor_config):
    footprint = pixel_footprint_mm(sensor_config)
    assert 0.06 < footprint < 0.07


def test_make_grid_spec_covers_data_bounds_with_padding():
    x = np.array([-5.0, 5.0])
    y = np.array([-2.0, 2.0])
    grid = make_grid_spec(x, y, resolution_mm=1.0, pad_cells=1)
    assert grid.x_min <= -6.0
    assert grid.x_centers[0] < -5.0
    assert grid.x_centers[-1] > 5.0
    assert grid.y_centers[0] < -2.0
    assert grid.y_centers[-1] > 2.0


def test_bin_values_to_grid_averages_and_leaves_gaps_as_nan():
    x = np.array([0.5, 0.5, 5.5])  # first two land in the same cell
    y = np.array([0.5, 0.5, 0.5])
    values = np.array([10.0, 20.0, 99.0])
    grid = make_grid_spec(np.array([0.0, 6.0]), np.array([0.0, 1.0]), resolution_mm=1.0, pad_cells=0)

    result = bin_values_to_grid(x, y, values, grid)
    # cell containing (0.5, 0.5) should average the two co-located values.
    assert result[0, 0] == pytest.approx(15.0)
    # a cell with no points should be NaN, not 0.
    assert np.isnan(result[0, 3])


def test_sample_true_height_on_grid_matches_known_flat_face(sensor_config, cube_stl_path):
    part = load_part(cube_stl_path, sensor_config, face_normal_hint=(1.0, 0.0, 0.0))
    grid = make_grid_spec(np.array([-10.0, 10.0]), np.array([-10.0, 10.0]), resolution_mm=1.0)
    truth_grid = sample_true_height_on_grid(part.mesh, sensor_config, grid)
    valid = ~np.isnan(truth_grid)
    assert valid.any()
    assert np.allclose(truth_grid[valid], 50.0, atol=1e-6)


def test_regrid_reconstruction_and_truth_is_square_and_near_zero_residual(sensor_config, cube_stl_path):
    part = load_part(cube_stl_path, sensor_config, face_normal_hint=(1.0, 0.0, 0.0))
    scan = run_measurement(part.mesh, sensor_config, step_stride=2, mirror_stride=2)
    result = run_reconstruction(scan, sensor_config)

    grid, recon_grid, truth_grid = regrid_reconstruction_and_truth(result, part.mesh, sensor_config)

    # V1.1 fixes the trapezoid/keystone artifact: a square physical face
    # should regrid to an (approximately) square array, not a skewed one.
    n_rows, n_cols = grid.shape
    assert abs(n_rows - n_cols) / max(n_rows, n_cols) < 0.05

    both = ~np.isnan(recon_grid) & ~np.isnan(truth_grid)
    assert both.sum() > 0
    residual = recon_grid[both] - truth_grid[both]
    assert np.sqrt(np.mean(residual**2)) < 1e-6
    assert np.abs(residual).max() < 1e-5


def test_regrid_raises_on_empty_reconstruction(sensor_config):
    from fourmp.sensor_sim.measurement import ScanData
    from fourmp.sensor_sim.reconstruction import run_reconstruction

    empty_int = np.array([], dtype=int)
    empty_float = np.array([], dtype=float)
    empty_scan = ScanData(
        step=empty_int,
        mirror=empty_int,
        cam_i=empty_int,
        cam_j=empty_int,
        cam_i_continuous=empty_float,
        cam_j_continuous=empty_float,
    )
    empty_result = run_reconstruction(empty_scan, sensor_config)
    with pytest.raises(ValueError):
        regrid_reconstruction_and_truth(empty_result, None, sensor_config)
