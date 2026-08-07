"""End-to-end V1 milestone test (spec's "First concrete test case"):

one cube face, one pose, idealized sensor model (binary-intensity inverse-
pinhole projector + tilted-pinhole camera, full confirmed resolution -- 1600
scan steps x 2716 mirrors/line, 5328x3104 camera) -> measure -> reconstruct
-> regrid onto a physical XY grid (mm) -> compare against ground truth
sampled directly on that same grid, using both pointwise (RMS/max) and
spectral metrics.

V1.1 update: sub-pixel reconstruction (project_points(), not rounded
pixel_indices()) plus physical-grid regridding collapse the pointwise
residual to floating-point noise and remove the camera-pixel-native
trapezoid/keystone artifact -- both checked below.
"""

import numpy as np

from fourmp.sensor_sim.measurement import run_measurement
from fourmp.sensor_sim.part import load_part
from fourmp.sensor_sim.reconstruction import run_reconstruction
from fourmp.sensor_sim.regrid import regrid_reconstruction_and_truth
from fourmp.sensor_sim.validation import pointwise_residual, spectral_residual

# V1.1's only remaining error source is floating-point arithmetic -- the
# sub-pixel fix removes camera-pixel quantization, the regrid removes the
# keystone artifact. These thresholds are deliberately tight.
MAX_ALLOWED_RMS_MM = 1e-6
MAX_ALLOWED_PEAK_MM = 1e-5


def test_v1_cube_face_measure_reconstruct_validate(sensor_config, cube_stl_path):
    part = load_part(cube_stl_path, sensor_config, face_normal_hint=(1.0, 0.0, 0.0))

    scan = run_measurement(part.mesh, sensor_config)  # full 1600 x 2716 resolution
    assert len(scan) > 0, "measurement engine produced no hits at all"

    result = run_reconstruction(scan, sensor_config)
    assert len(result.heights) > 0

    grid, recon_grid, truth_grid = regrid_reconstruction_and_truth(result, part.mesh, sensor_config)

    # The V1.1 keystone fix: a square physical face should regrid to an
    # (approximately) square array on physical XY axes, not the skewed
    # trapezoid the camera-pixel-native grid produced.
    n_rows, n_cols = grid.shape
    assert abs(n_rows - n_cols) / max(n_rows, n_cols) < 0.05, grid.shape

    pointwise = pointwise_residual(recon_grid, truth_grid)
    assert pointwise.n_compared > 0, "no overlap between reconstruction and ground truth grids"
    assert pointwise.rms_mm < MAX_ALLOWED_RMS_MM, pointwise
    assert pointwise.max_mm < MAX_ALLOWED_PEAK_MM, pointwise

    spectral = spectral_residual(recon_grid, truth_grid)
    assert spectral.shape[0] > 0 and spectral.shape[1] > 0
    assert np.isfinite(spectral.total_power_mm2)

    print(
        f"\nV1.1 milestone: grid {grid.shape[1]}x{grid.shape[0]} @ {grid.resolution_mm * 1000:.1f}um, "
        f"{pointwise.n_compared} cells compared, "
        f"RMS={pointwise.rms_mm:.3e}mm, max={pointwise.max_mm:.3e}mm"
    )
