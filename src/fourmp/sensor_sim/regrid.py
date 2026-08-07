"""V1.1: regrid the reconstruction onto a uniform physical XY grid (mm),
instead of reporting on raw camera-pixel indices.

The camera-pixel-native grid (reconstruction.py's ``height_map``) makes a
flat rectangular face render as a trapezoid -- an artifact of indexing by a
physically tilted camera's row/col, not a defect in the geometry (it shows
up identically in a directly-sampled ground truth). Binning the *triangulated*
points by their actual physical (X, Y) position removes that artifact and
gets mm-native axes for free.

Ground truth is resampled directly on this same grid too -- not by
back-projecting camera pixels (that's the camera-pixel-native view this
module is deliberately moving away from), but by casting a ray straight
along the boresight (+Z) through each grid cell's (X, Y) and intersecting
the true mesh. That keeps ground truth decoupled from any camera-specific
keystoning, which is the more literal reading of "sampled directly from the
true mesh" anyway.

Resolution is chosen to match the camera's own pixel footprint projected
onto the reference plane (~60-70um per sensor-sim-v1-spec.md's V1.1 note) --
a reasonable, documented default, not a precise per-point value (the real
footprint varies slightly across the keystoned FOV; this uses one
representative average, matching the spirit of other V1 placeholder
choices like the camera FOV margin).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from fourmp.sensor_sim.raytrace import nearest_intersection
from fourmp.sensor_sim.reconstruction import HeightMapResult
from fourmp.sensor_sim.sensor_config import SensorConfig


def pixel_footprint_mm(sensor_config: SensorConfig) -> float:
    """Representative camera pixel footprint (mm) at the reference plane --
    average of the two axes' footprints (distance-to-reference / focal
    length in pixel units), since the true footprint varies slightly across
    the keystoned FOV and this is a resolution choice, not a per-point value."""
    camera = sensor_config.camera
    distance = float(np.linalg.norm(camera.center - sensor_config.reference_point))
    footprint_i = distance / camera.f_i
    footprint_j = distance / camera.f_j
    return 0.5 * (footprint_i + footprint_j)


@dataclass(frozen=True)
class GridSpec:
    """A uniform grid over physical (X, Y), mm. Row axis = Y (line axis),
    column axis = X (baseline axis) -- matches typical image (row, col) =
    (y, x) convention for the plots in report.py."""

    x_min: float
    y_min: float
    resolution_mm: float
    shape: tuple[int, int]  # (n_rows, n_cols)

    @property
    def x_centers(self) -> np.ndarray:
        return self.x_min + (np.arange(self.shape[1]) + 0.5) * self.resolution_mm

    @property
    def y_centers(self) -> np.ndarray:
        return self.y_min + (np.arange(self.shape[0]) + 0.5) * self.resolution_mm

    @property
    def extent_mm(self) -> tuple[float, float, float, float]:
        """(left, right, bottom, top), for matplotlib's imshow(extent=...)."""
        n_rows, n_cols = self.shape
        return (
            self.x_min,
            self.x_min + n_cols * self.resolution_mm,
            self.y_min,
            self.y_min + n_rows * self.resolution_mm,
        )


def make_grid_spec(x: np.ndarray, y: np.ndarray, resolution_mm: float, pad_cells: int = 1) -> GridSpec:
    """A grid covering the bounding box of the given points, at
    ``resolution_mm``, padded by ``pad_cells`` on each side."""
    pad = pad_cells * resolution_mm
    x_min, x_max = float(x.min()) - pad, float(x.max()) + pad
    y_min, y_max = float(y.min()) - pad, float(y.max()) + pad
    n_cols = max(1, int(np.ceil((x_max - x_min) / resolution_mm)))
    n_rows = max(1, int(np.ceil((y_max - y_min) / resolution_mm)))
    return GridSpec(x_min=x_min, y_min=y_min, resolution_mm=resolution_mm, shape=(n_rows, n_cols))


def bin_values_to_grid(x: np.ndarray, y: np.ndarray, values: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Mean-aggregate ``values`` at (x, y) into ``grid``'s cells. NaN where
    no point lands in a cell (not 0 -- there's no real "no data" reading of
    exactly 0mm)."""
    n_rows, n_cols = grid.shape
    col = np.floor((x - grid.x_min) / grid.resolution_mm).astype(np.int64)
    row = np.floor((y - grid.y_min) / grid.resolution_mm).astype(np.int64)
    in_bounds = (row >= 0) & (row < n_rows) & (col >= 0) & (col < n_cols)

    flat = row[in_bounds] * n_cols + col[in_bounds]
    sums = np.bincount(flat, weights=values[in_bounds], minlength=n_rows * n_cols)
    counts = np.bincount(flat, minlength=n_rows * n_cols)

    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    means[counts == 0] = np.nan
    return means.reshape(n_rows, n_cols)


def sample_true_height_on_grid(
    true_mesh: trimesh.Trimesh, sensor_config: SensorConfig, grid: GridSpec
) -> np.ndarray:
    """Ground truth, sampled directly on ``grid``: for each cell center
    (x, y), cast a ray along the boresight (+Z) and intersect the true mesh.
    NaN where the ray misses."""
    x_centers = grid.x_centers
    y_centers = grid.y_centers
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)  # each (n_rows, n_cols)

    z_start = float(true_mesh.bounds[0][2]) - 1.0  # comfortably "before" the mesh along +Z
    origins = np.stack([x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, z_start)], axis=-1)
    direction = np.array([0.0, 0.0, 1.0])

    locations, hit = nearest_intersection(origins, direction, true_mesh)
    heights = np.full(x_grid.size, np.nan)
    heights[hit] = sensor_config.height_from_point(locations[hit])
    return heights.reshape(x_grid.shape)


def regrid_reconstruction_and_truth(
    result: HeightMapResult,
    true_mesh: trimesh.Trimesh,
    sensor_config: SensorConfig,
    resolution_mm: float | None = None,
) -> tuple[GridSpec, np.ndarray, np.ndarray]:
    """Reconstructed height map and ground truth, regridded onto the same
    physical XY grid (so they stay directly, cell-for-cell comparable).

    Returns (grid_spec, reconstructed_height_grid, ground_truth_height_grid).
    """
    if len(result.heights) == 0:
        raise ValueError("no reconstructed points to regrid")

    if resolution_mm is None:
        resolution_mm = pixel_footprint_mm(sensor_config)

    x, y = result.points[:, 0], result.points[:, 1]
    grid = make_grid_spec(x, y, resolution_mm)

    reconstructed = bin_values_to_grid(x, y, result.heights, grid)
    truth = sample_true_height_on_grid(true_mesh, sensor_config, grid)
    return grid, reconstructed, truth
