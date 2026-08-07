"""Reconstruction engine: the inverse model.

For each recorded hit (camera pixel, scan step), back-project the camera
pixel to a ray using the *same* camera calibration, and intersect it with
the *same* projector ray for that step/mirror -- the identical
PinholeModel.directions_for_indices used by the forward model, since both
projector and camera share one calibration object via SensorConfig. In a
perfectly noiseless world these two rays would intersect exactly; camera
pixel quantization means they generally only come close, so
geometry.closest_points_between_rays is used (midpoint of the shortest
connecting segment) rather than assuming an exact intersection.

Output: a height map on the camera's own pixel grid (mm, signed deviation
from the reference plane, per SensorConfig.height_from_point), matching the
"camera-pixel-native grid" contract. Cells with no recorded hit are NaN, not
zero -- there's no such thing as a real "no data" reading of exactly 0mm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fourmp.sensor_sim.geometry import closest_points_between_rays_batch
from fourmp.sensor_sim.measurement import ScanData
from fourmp.sensor_sim.sensor_config import SensorConfig


@dataclass
class HeightMapResult:
    """Reconstructed height map plus the per-hit triangulation gap (a
    quantization-error diagnostic -- see geometry.closest_points_between_rays)."""

    height_map: np.ndarray  # (camera.n_i, camera.n_j), mm, NaN where no data
    triangulation_gap_mm: np.ndarray  # same shape, NaN where no data
    collisions: int  # count of hits that overwrote an earlier hit at the same pixel

    def centered_index(self, row: int, col: int) -> tuple[float, float]:
        """Pixel (row, col) -> (row, col) relative to the grid center, per the
        "(0, 0) at grid center" height-map origin convention (placeholder,
        pending the real companion schema)."""
        n_i, n_j = self.height_map.shape
        return row - (n_i - 1) / 2.0, col - (n_j - 1) / 2.0


def run_reconstruction(scan_data: ScanData, sensor_config: SensorConfig) -> HeightMapResult:
    projector = sensor_config.projector
    camera = sensor_config.camera

    height_map = np.full((camera.n_i, camera.n_j), np.nan)
    gap_map = np.full((camera.n_i, camera.n_j), np.nan)
    collisions = 0

    if len(scan_data) == 0:
        return HeightMapResult(height_map, gap_map, collisions)

    proj_dirs = projector.directions_for_indices(scan_data.step, scan_data.mirror)
    cam_dirs = camera.directions_for_indices(scan_data.cam_i, scan_data.cam_j)

    points, gaps = closest_points_between_rays_batch(
        projector.center, proj_dirs, camera.center, cam_dirs
    )
    heights = sensor_config.height_from_point(points)

    # Deterministic collision handling: if two hits (different scan
    # steps/mirrors) land on the same camera pixel, keep the one with the
    # smaller triangulation gap (the more self-consistent of the two) rather
    # than whichever happens to be last in array order.
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

    return HeightMapResult(height_map, gap_map, collisions)
