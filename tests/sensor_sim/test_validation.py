import numpy as np

from fourmp.sensor_sim.measurement import run_measurement
from fourmp.sensor_sim.part import load_part
from fourmp.sensor_sim.reconstruction import run_reconstruction
from fourmp.sensor_sim.validation import (
    ground_truth_like,
    pointwise_residual,
    spectral_residual,
)


def test_ground_truth_matches_known_flat_face_height(sensor_config, cube_stl_path):
    part = load_part(cube_stl_path, sensor_config, face_normal_hint=(1.0, 0.0, 0.0))
    scan = run_measurement(part.mesh, sensor_config, step_stride=4, mirror_stride=4)
    result = run_reconstruction(scan, sensor_config)

    truth = ground_truth_like(part.mesh, sensor_config, result.height_map)
    valid = ~np.isnan(truth)
    assert valid.sum() > 0
    # Ground truth is sampled analytically (no scan-side quantization), so
    # it should sit essentially exactly at the known +50mm.
    assert np.allclose(truth[valid], 50.0, atol=1e-6)


def test_pointwise_residual_perfect_match_is_zero():
    a = np.array([[1.0, np.nan], [2.0, 3.0]])
    b = np.array([[1.0, 5.0], [2.0, 3.0]])
    result = pointwise_residual(a, b)
    assert result.rms_mm == 0.0
    assert result.max_mm == 0.0
    assert result.n_compared == 3
    assert result.n_truth_only == 1


def test_pointwise_residual_reports_mismatch():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 5.0])
    result = pointwise_residual(a, b)
    assert result.max_mm == 2.0
    assert result.rms_mm == np.sqrt((0 + 0 + 4) / 3)


def test_spectral_residual_of_perfect_match_has_no_non_dc_power():
    grid = np.zeros((16, 16))
    result = spectral_residual(grid, grid)
    assert result.total_power_mm2 == 0.0
    assert np.isnan(result.peak_to_mean_power_ratio)


def test_spectral_residual_flags_a_periodic_error():
    rng = np.random.default_rng(0)
    truth = np.zeros((32, 32))
    row_idx = np.arange(32)[:, None]
    # A strong single-frequency ripple error, well above float noise.
    periodic_error = 0.5 * np.sin(2 * np.pi * row_idx * 4 / 32) * np.ones((1, 32))
    noisy_error = rng.normal(scale=1e-6, size=(32, 32))
    reconstructed = truth + periodic_error + noisy_error

    result = spectral_residual(reconstructed, truth)
    assert result.peak_to_mean_power_ratio > 10.0
