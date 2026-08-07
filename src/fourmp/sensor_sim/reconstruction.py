"""Reconstruction engine: the inverse model.

For each recorded hit, back-project the camera to a ray using the *same*
camera calibration, and intersect it with the *same* projector ray for that
step/mirror -- the identical PinholeModel.directions_for_indices used by the
forward model, since both projector and camera share one calibration object
via SensorConfig. In a perfectly noiseless world these two rays would
intersect exactly; camera pixel quantization means they generally only come
close, so geometry.closest_points_between_rays is used (midpoint of the
shortest connecting segment) rather than assuming an exact intersection.

**V1.1 sub-pixel fix:** triangulation back-projects the camera's continuous
sub-pixel (i, j) (``ScanData.cam_i_continuous``/``cam_j_continuous``), not
the rounded discrete pixel address -- see measurement.py's module docstring.
This was the dominant source of the V1.0 pointwise residual (~25.5um RMS):
rounding to the nearest integer pixel before triangulating threw away real
sub-pixel information the forward model already had, making the forward/
inverse round trip geometrically inexact even though the calibration itself
was correct. With the continuous value fed straight into triangulation, the
round trip is geometrically exact again (residual collapses to floating-
point noise for a still, flat surface).

Two outputs, per sensor-sim-v1-spec.md's V1.1 section:
- ``points``/``heights``/``gaps`` -- one entry per hit, full precision, no
  pixel binning. This is what the physical-XY-grid regridding (regrid.py)
  should consume for the "real" height map.
- ``height_map``/``triangulation_gap_mm`` -- the previous camera-pixel-
  native diagnostic view (discrete pixel address, collision-resolved),
  kept because some diagnostics are naturally per-camera-pixel. NaN where
  no data -- there's no such thing as a real "no data" reading of exactly
  0mm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fourmp.sensor_sim.geometry import closest_points_between_rays_batch
from fourmp.sensor_sim.measurement import ScanData
from fourmp.sensor_sim.sensor_config import SensorConfig


@dataclass
class HeightMapResult:
    """Full-precision triangulated points plus the camera-pixel-native
    diagnostic view derived from them."""

    points: np.ndarray  # (n_hits, 3), sensor frame, one row per hit
    heights: np.ndarray  # (n_hits,) mm, signed deviation from the reference plane
    gaps: np.ndarray  # (n_hits,) mm, triangulation gap (quantization-error diagnostic)
    pixel_i: np.ndarray  # (n_hits,) int, discrete/recorded camera pixel row
    pixel_j: np.ndarray  # (n_hits,) int, discrete/recorded camera pixel column

    height_map: np.ndarray  # (camera.n_i, camera.n_j), mm, NaN where no data
    triangulation_gap_mm: np.ndarray  # same shape as height_map, NaN where no data
    collisions: int  # count of hits that overwrote an earlier hit at the same discrete pixel

    def centered_index(self, row: int, col: int) -> tuple[float, float]:
        """Pixel (row, col) -> (row, col) relative to the grid center, per the
        "(0, 0) at grid center" height-map origin convention (placeholder,
        pending the real companion schema)."""
        n_i, n_j = self.height_map.shape
        return row - (n_i - 1) / 2.0, col - (n_j - 1) / 2.0


def _empty_result(camera_n_i: int, camera_n_j: int) -> HeightMapResult:
    empty_int = np.array([], dtype=int)
    empty_float = np.array([], dtype=float)
    empty_points = np.zeros((0, 3))
    return HeightMapResult(
        points=empty_points,
        heights=empty_float,
        gaps=empty_float,
        pixel_i=empty_int,
        pixel_j=empty_int,
        height_map=np.full((camera_n_i, camera_n_j), np.nan),
        triangulation_gap_mm=np.full((camera_n_i, camera_n_j), np.nan),
        collisions=0,
    )


def run_reconstruction(scan_data: ScanData, sensor_config: SensorConfig) -> HeightMapResult:
    projector = sensor_config.projector
    camera = sensor_config.camera

    if len(scan_data) == 0:
        return _empty_result(camera.n_i, camera.n_j)

    proj_dirs = projector.directions_for_indices(scan_data.step, scan_data.mirror)
    cam_dirs = camera.directions_for_indices(scan_data.cam_i_continuous, scan_data.cam_j_continuous)

    points, gaps = closest_points_between_rays_batch(
        projector.center, proj_dirs, camera.center, cam_dirs
    )
    heights = sensor_config.height_from_point(points)

    # Camera-pixel-native diagnostic view: deterministic collision handling,
    # if two hits (different scan steps/mirrors) round to the same discrete
    # camera pixel, keep the one with the smaller triangulation gap (the
    # more self-consistent of the two) rather than whichever happens to be
    # last in array order.
    height_map = np.full((camera.n_i, camera.n_j), np.nan)
    gap_map = np.full((camera.n_i, camera.n_j), np.nan)

    order = np.argsort(-gaps)  # worst-gap first, so the best gap is written last
    rows = scan_data.cam_i[order]
    cols = scan_data.cam_j[order]
    ordered_heights = heights[order]
    ordered_gaps = gaps[order]

    flat_index = rows.astype(np.int64) * camera.n_j + cols.astype(np.int64)
    _, first_seen_count = np.unique(flat_index, return_counts=True)
    collisions = int(np.sum(first_seen_count - 1))

    height_map.reshape(-1)[flat_index] = ordered_heights
    gap_map.reshape(-1)[flat_index] = ordered_gaps

    return HeightMapResult(
        points=points,
        heights=heights,
        gaps=gaps,
        pixel_i=scan_data.cam_i,
        pixel_j=scan_data.cam_j,
        height_map=height_map,
        triangulation_gap_mm=gap_map,
        collisions=collisions,
    )
