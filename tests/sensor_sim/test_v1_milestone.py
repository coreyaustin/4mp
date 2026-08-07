"""End-to-end V1 milestone test (spec's "First concrete test case"):

one cube face, one pose, idealized sensor model (binary-intensity inverse-
pinhole projector + tilted-pinhole camera, full confirmed resolution -- 1600
scan steps x 2716 mirrors/line, 5328x3104 camera) -> measure -> reconstruct
-> compare against ground truth sampled directly from the same posed mesh,
using both pointwise (RMS/max) and spectral metrics.
"""

import numpy as np

from fourmp.sensor_sim.measurement import run_measurement
from fourmp.sensor_sim.part import load_part
from fourmp.sensor_sim.reconstruction import run_reconstruction
from fourmp.sensor_sim.validation import ground_truth_like, pointwise_residual, spectral_residual

# The only error source in this idealized model is camera-pixel quantization
# (see reconstruction.py) -- sub-pixel-scale, so these are deliberately tight.
MAX_ALLOWED_RMS_MM = 0.2
MAX_ALLOWED_PEAK_MM = 1.0


def test_v1_cube_face_measure_reconstruct_validate(sensor_config, cube_stl_path):
    part = load_part(cube_stl_path, sensor_config, face_normal_hint=(1.0, 0.0, 0.0))

    scan = run_measurement(part.mesh, sensor_config)  # full 1600 x 2716 resolution
    assert len(scan) > 0, "measurement engine produced no hits at all"

    result = run_reconstruction(scan, sensor_config)
    n_reconstructed = int(np.sum(~np.isnan(result.height_map)))
    assert n_reconstructed > 0

    truth = ground_truth_like(part.mesh, sensor_config, result.height_map)

    pointwise = pointwise_residual(result.height_map, truth)
    assert pointwise.n_compared > 0.9 * n_reconstructed, (
        "most reconstructed pixels should have a ground-truth counterpart "
        "for a flat, fully-visible face"
    )
    assert pointwise.rms_mm < MAX_ALLOWED_RMS_MM, pointwise
    assert pointwise.max_mm < MAX_ALLOWED_PEAK_MM, pointwise

    spectral = spectral_residual(result.height_map, truth)
    # No hard threshold on the spectral shape metrics here (there's no
    # "correct" reference value to gate on for a single idealized flat-face
    # run) -- just assert the metric is well-formed, i.e. actually computed a
    # finite power spectrum over a sensibly-sized cropped region.
    assert spectral.shape[0] > 0 and spectral.shape[1] > 0
    assert np.isfinite(spectral.total_power_mm2)

    print(
        f"\nV1 milestone: {n_reconstructed} px reconstructed, "
        f"RMS={pointwise.rms_mm:.4f}mm, max={pointwise.max_mm:.4f}mm, "
        f"spectral peak/mean={spectral.peak_to_mean_power_ratio:.1f}, "
        f"low-freq fraction={spectral.low_frequency_power_fraction:.3f}"
    )
