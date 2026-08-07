"""Validation harness (test-harness concern, not a pipeline feature -- see
architecture-decisions.md "Validation approach").

Since the simulated part's true geometry is always exactly known, ground
truth is sampled directly from the same posed mesh the measurement engine
scanned -- no sensor simulation, no injected error: back-project each camera
pixel of interest to a ray (same camera calibration the reconstruction
engine used) and intersect it with the true mesh directly.

Two metrics, per spec ("no reason to pick one"):
- Pointwise residual (RMS/max) -- simplest, good pass/fail.
- Spectral comparison -- surfaces systematic/periodic error a pointwise
  summary would hide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from fourmp.sensor_sim.raytrace import nearest_intersection
from fourmp.sensor_sim.sensor_config import SensorConfig


def sample_ground_truth(
    true_mesh: trimesh.Trimesh,
    sensor_config: SensorConfig,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Ground-truth height at each given camera pixel (row, col), sampled
    directly from ``true_mesh`` (already posed into sensor space). NaN where
    the back-projected camera ray misses the mesh entirely.

    Restricted to caller-supplied pixels (typically: wherever the
    reconstruction produced a value) rather than the full camera array,
    since only those cells are ever compared -- enumerating the full
    5328 x 3104 array would ray-trace ~17M mostly-irrelevant rays.
    """
    camera = sensor_config.camera
    directions = camera.directions_for_indices(np.asarray(rows), np.asarray(cols))
    locations, hit = nearest_intersection(camera.center, directions, true_mesh)

    heights = np.full(len(rows), np.nan)
    heights[hit] = sensor_config.height_from_point(locations[hit])
    return heights


def ground_truth_like(
    true_mesh: trimesh.Trimesh, sensor_config: SensorConfig, height_map: np.ndarray
) -> np.ndarray:
    """Ground-truth height map matching ``height_map``'s shape and valid-cell
    footprint (same convention: mm, signed deviation, same camera-pixel grid)."""
    rows, cols = np.nonzero(~np.isnan(height_map))
    heights = sample_ground_truth(true_mesh, sensor_config, rows, cols)
    truth = np.full_like(height_map, np.nan)
    truth[rows, cols] = heights
    return truth


@dataclass
class PointwiseResidual:
    rms_mm: float
    max_mm: float
    n_compared: int
    n_reconstructed_only: int  # reconstructed a value but ground truth missed
    n_truth_only: int  # ground truth hit but reconstruction missed


def pointwise_residual(reconstructed: np.ndarray, truth: np.ndarray) -> PointwiseResidual:
    recon_valid = ~np.isnan(reconstructed)
    truth_valid = ~np.isnan(truth)
    both = recon_valid & truth_valid
    residual = reconstructed[both] - truth[both]
    rms = float(np.sqrt(np.mean(residual**2))) if residual.size else float("nan")
    peak = float(np.max(np.abs(residual))) if residual.size else float("nan")
    return PointwiseResidual(
        rms_mm=rms,
        max_mm=peak,
        n_compared=int(both.sum()),
        n_reconstructed_only=int((recon_valid & ~truth_valid).sum()),
        n_truth_only=int((~recon_valid & truth_valid).sum()),
    )


@dataclass
class SpectralResidual:
    """Power-spectrum summary of the residual field (reconstructed - truth).

    Deliberately built from the *residual's own* spectrum rather than a
    recon-vs-truth spectrum ratio: for a flat reference face (this V1 test
    case), truth's own spectrum is ~0 everywhere except DC, which makes any
    metric normalized by truth's spectrum blow up or go to 0/0 -- a
    degenerate case for a flat test face specifically, not a real signal.
    The residual's spectrum has no such degeneracy (it's exactly zero only
    for a perfect reconstruction) and it directly answers the motivating
    question -- is the error flat/white (sensor-noise-like), or does it
    concentrate at specific frequencies (a systematic or periodic artifact,
    e.g. matching the scan-step pitch)?
    """

    total_power_mm2: float  # Parseval-consistent total residual power
    peak_to_mean_power_ratio: float  # non-DC peak / non-DC mean; large => a periodic component
    low_frequency_power_fraction: float  # fraction of non-DC power in the lowest-decile frequencies
    shape: tuple[int, int]  # cropped region size actually compared


def _crop_to_valid_bbox(mask: np.ndarray) -> tuple[slice, slice]:
    rows, cols = np.nonzero(mask)
    return slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)


def _fill_and_window(patch: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill any remaining gaps in ``patch`` with the patch mean, remove the
    mean (drop the DC term -- the pointwise metric already covers overall
    bias), and apply a 2D Hann window to suppress edge-discontinuity
    artifacts in the FFT."""
    fill_value = patch[valid].mean() if valid.any() else 0.0
    filled = np.where(valid, patch, fill_value)
    filled = filled - filled.mean()
    win_r = np.hanning(filled.shape[0])
    win_c = np.hanning(filled.shape[1])
    window = np.outer(win_r, win_c)
    return filled * window


def spectral_residual(reconstructed: np.ndarray, truth: np.ndarray) -> SpectralResidual:
    both = ~np.isnan(reconstructed) & ~np.isnan(truth)
    if not both.any():
        return SpectralResidual(
            total_power_mm2=float("nan"),
            peak_to_mean_power_ratio=float("nan"),
            low_frequency_power_fraction=float("nan"),
            shape=(0, 0),
        )

    row_slice, col_slice = _crop_to_valid_bbox(both)
    residual_patch = reconstructed[row_slice, col_slice] - truth[row_slice, col_slice]
    valid_patch = both[row_slice, col_slice]

    windowed = _fill_and_window(residual_patch, valid_patch)
    power = np.abs(np.fft.fft2(windowed)) ** 2

    total_power = float(power.sum())

    non_dc = power.copy()
    non_dc[0, 0] = 0.0
    non_dc_total = float(non_dc.sum())

    if non_dc_total <= 0:
        # Residual is (numerically) constant -- no spectral content beyond
        # DC, i.e. as close to a perfect reconstruction as this test can show.
        peak_to_mean = float("nan")
        low_freq_fraction = float("nan")
    else:
        peak_to_mean = float(non_dc.max() / non_dc.mean())

        freq_r = np.fft.fftfreq(power.shape[0])
        freq_c = np.fft.fftfreq(power.shape[1])
        radius = np.sqrt(freq_r[:, None] ** 2 + freq_c[None, :] ** 2)
        low_freq_cutoff = np.quantile(radius, 0.10)
        low_freq_mask = (radius <= low_freq_cutoff) & (radius > 0)
        low_freq_fraction = float(non_dc[low_freq_mask].sum() / non_dc_total)

    return SpectralResidual(
        total_power_mm2=total_power,
        peak_to_mean_power_ratio=peak_to_mean,
        low_frequency_power_fraction=low_freq_fraction,
        shape=residual_patch.shape,
    )
