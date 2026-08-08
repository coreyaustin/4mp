"""V1.1: regrid the reconstruction onto a uniform physical grid (mm),
instead of reporting on raw camera-pixel indices. V1.3: that grid is now
O_r's own (X, Z) plane -- X horizontal, Z vertical -- per 4MP's cell-wide
convention, not an arbitrary O_s-frame XY plane.

The camera-pixel-native grid (reconstruction.py's ``height_map``) makes a
flat rectangular face render as a trapezoid -- an artifact of indexing by a
physically tilted camera's row/col, not a defect in the geometry (it shows
up identically in a directly-sampled ground truth). Binning the
*triangulated* points by their actual position removes that artifact and
gets mm-native axes for free; V1.3 places that position in O_r rather than
O_s so the grid axes and the height value (O_r's Y, see part.py's
``height_in_o_r``) all come from one standardized frame.

Ground truth is resampled directly on this same grid too -- not by
back-projecting camera pixels (that's the camera-pixel-native view this
module is deliberately moving away from), but by casting a ray along O_r's
Y axis through each grid cell's (X, Z) and intersecting the true mesh (the
ray itself is transformed into O_s to actually trace it there, since that's
the frame the mesh is posed in). That keeps ground truth decoupled from any
camera-specific keystoning, which is the more literal reading of "sampled
directly from the true mesh" anyway.

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

from fourmp.sensor_sim.geometry import Transform
from fourmp.sensor_sim.part import height_in_o_r
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
    """A uniform grid over O_r's (X, Z), mm -- X horizontal, Z vertical
    (V1.3). Row axis = Z, column axis = X, matching typical image
    (row, col) = (vertical, horizontal) convention for the plots in
    report.py."""

    x_min: float
    z_min: float
    resolution_mm: float
    shape: tuple[int, int]  # (n_rows, n_cols)

    @property
    def x_centers(self) -> np.ndarray:
        return self.x_min + (np.arange(self.shape[1]) + 0.5) * self.resolution_mm

    @property
    def z_centers(self) -> np.ndarray:
        return self.z_min + (np.arange(self.shape[0]) + 0.5) * self.resolution_mm

    @property
    def extent_mm(self) -> tuple[float, float, float, float]:
        """(left, right, bottom, top), for matplotlib's imshow(extent=...)."""
        n_rows, n_cols = self.shape
        return (
            self.x_min,
            self.x_min + n_cols * self.resolution_mm,
            self.z_min,
            self.z_min + n_rows * self.resolution_mm,
        )


def make_grid_spec(x: np.ndarray, z: np.ndarray, resolution_mm: float, pad_cells: int = 1) -> GridSpec:
    """A grid covering the bounding box of the given points, at
    ``resolution_mm``, padded by ``pad_cells`` on each side."""
    pad = pad_cells * resolution_mm
    x_min, x_max = float(x.min()) - pad, float(x.max()) + pad
    z_min, z_max = float(z.min()) - pad, float(z.max()) + pad
    n_cols = max(1, int(np.ceil((x_max - x_min) / resolution_mm)))
    n_rows = max(1, int(np.ceil((z_max - z_min) / resolution_mm)))
    return GridSpec(x_min=x_min, z_min=z_min, resolution_mm=resolution_mm, shape=(n_rows, n_cols))


def bin_values_to_grid(x: np.ndarray, z: np.ndarray, values: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Mean-aggregate ``values`` at (x, z) into ``grid``'s cells. NaN where
    no point lands in a cell (not 0 -- there's no real "no data" reading of
    exactly 0mm)."""
    n_rows, n_cols = grid.shape
    col = np.floor((x - grid.x_min) / grid.resolution_mm).astype(np.int64)
    row = np.floor((z - grid.z_min) / grid.resolution_mm).astype(np.int64)
    in_bounds = (row >= 0) & (row < n_rows) & (col >= 0) & (col < n_cols)

    flat = row[in_bounds] * n_cols + col[in_bounds]
    sums = np.bincount(flat, weights=values[in_bounds], minlength=n_rows * n_cols)
    counts = np.bincount(flat, minlength=n_rows * n_cols)

    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    means[counts == 0] = np.nan
    return means.reshape(n_rows, n_cols)


def sample_true_height_on_grid(
    true_mesh: trimesh.Trimesh, sensor_config: SensorConfig, o_r_from_o_s: Transform, grid: GridSpec
) -> np.ndarray:
    """Ground truth, sampled directly on ``grid``: for each cell center
    (x, z) in O_r, cast a ray along O_r's +Y and intersect the true mesh
    (transforming the ray into O_s, where the mesh is actually posed, to
    trace it). NaN where the ray misses."""
    o_s_from_o_r = o_r_from_o_s.invert()

    x_grid, z_grid = np.meshgrid(grid.x_centers, grid.z_centers)  # each (n_rows, n_cols)

    # Start comfortably "before" the mesh along O_r's Y, same pattern as the
    # V1.1 approach (start before the mesh's own bound, cast toward it).
    mesh_y_or = o_r_from_o_s.apply_points(true_mesh.vertices)[:, 1]
    y_start = float(mesh_y_or.min()) - 1.0

    origins_or = np.stack([x_grid.ravel(), np.full(x_grid.size, y_start), z_grid.ravel()], axis=-1)
    origins_os = o_s_from_o_r.apply_points(origins_or)
    direction_os = o_s_from_o_r.apply_vectors(np.array([0.0, 1.0, 0.0]))

    locations_os, hit = nearest_intersection(origins_os, direction_os, true_mesh)
    heights = np.full(x_grid.size, np.nan)
    heights[hit] = height_in_o_r(locations_os[hit], o_r_from_o_s)
    return heights.reshape(x_grid.shape)


def regrid_reconstruction_and_truth(
    result: HeightMapResult,
    true_mesh: trimesh.Trimesh,
    sensor_config: SensorConfig,
    o_r_from_o_s: Transform,
    resolution_mm: float | None = None,
) -> tuple[GridSpec, np.ndarray, np.ndarray]:
    """Reconstructed height map and ground truth, regridded onto the same
    O_r (X, Z) grid (so they stay directly, cell-for-cell comparable).

    Returns (grid_spec, reconstructed_height_grid, ground_truth_height_grid).
    """
    if len(result.heights) == 0:
        raise ValueError("no reconstructed points to regrid")

    if resolution_mm is None:
        resolution_mm = pixel_footprint_mm(sensor_config)

    points_or = o_r_from_o_s.apply_points(result.points)
    x, z = points_or[:, 0], points_or[:, 2]
    grid = make_grid_spec(x, z, resolution_mm)

    # result.heights is already O_r's Y component (see reconstruction.py) --
    # binning only changes which coordinates position each value.
    reconstructed = bin_values_to_grid(x, z, result.heights, grid)
    truth = sample_true_height_on_grid(true_mesh, sensor_config, o_r_from_o_s, grid)
    return grid, reconstructed, truth
