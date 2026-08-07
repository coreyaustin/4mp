"""Shared tilted-pinhole model for both the DMD projector and the camera.

Per the spec, the DMD/source/idealized-lens system collapses into an "idealized
inverse-pinhole projector" -- explicitly "the DMD-side equivalent of treating
the camera as an idealized pinhole." Both the projector and the camera are
therefore modeled with the *same* class: a single optical center, an axis
tilted by the triangulation half-angle off the sensor's boresight (see
geometry.py for the sensor-frame convention), and an intrinsics-style
(f_i, f_j, c_i, c_j) mapping between a discrete index grid (DMD mirror
row/col, or camera pixel row/col) and 3D ray directions -- exactly as
confirmed for the projector (own tilted axis, own focal length derived from
the DMD's physical pixel pitch) and standard for the camera.

The forward (index -> ray direction) and inverse (point -> index) directions
share one set of intrinsics/pose per instance, which is what lets the
measurement engine's forward model and the reconstruction engine's inverse
model stay calibration-consistent by construction (same object, same values)
rather than by convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fourmp.sensor_sim.geometry import unit


@dataclass(frozen=True)
class PinholeModel:
    """A tilted pinhole: optical center + local (right, up, forward) axes + intrinsics.

    ``forward`` is the optical axis (points from the center toward the scene).
    ``right``/``up`` span the local image/DMD plane, perpendicular to ``forward``.
    Local index i runs along ``right`` (with focal length/principal point
    f_i/c_i), local index j runs along ``up`` (f_j/c_j) -- e.g. for the
    projector, i = scan-step / mirror-row index (0..1599), j = along-line /
    mirror-column index (0..2715); for the camera, i = baseline-axis pixel
    row (0..3103), j = line-axis pixel column (0..5327).

    Index convention: integer index values are pixel/mirror *centers*, with
    the principal point c_i = (n_i - 1) / 2 sitting exactly between the two
    middle indices (matches the projector focal-length derivation in
    sensor_config.py). f_i/f_j are dimensionless ("focal length in index
    units" = physical focal length / physical pixel pitch), so no separate
    physical pitch needs to be tracked once these are computed.
    """

    center: np.ndarray  # (3,) optical center, sensor frame
    right: np.ndarray  # (3,) unit vector, local i axis
    up: np.ndarray  # (3,) unit vector, local j axis
    forward: np.ndarray  # (3,) unit vector, optical axis
    f_i: float
    f_j: float
    c_i: float
    c_j: float
    n_i: int
    n_j: int

    @staticmethod
    def looking_at(
        center: np.ndarray,
        target: np.ndarray,
        f_i: float,
        f_j: float,
        n_i: int,
        n_j: int,
        world_up: np.ndarray = (0.0, 1.0, 0.0),
    ) -> "PinholeModel":
        """Build a pinhole whose forward axis points from ``center`` at ``target``.

        ``world_up`` is used only to construct an orthonormal (right, up,
        forward) basis; for this sensor's baseline-tilted arms (see
        geometry.py) the true "up" (the line/Y axis) is unaffected by the
        tilt, so world_up = +Y reproduces it exactly.
        """
        center = np.asarray(center, dtype=float)
        forward = unit(np.asarray(target, dtype=float) - center)
        world_up = np.asarray(world_up, dtype=float)
        right = unit(np.cross(world_up, forward))
        # Re-orthogonalize "up" against forward (guards against non-unit/
        # non-orthogonal world_up; for our exact X-Z tilt this is a no-op).
        up = unit(np.cross(forward, right))
        c_i = (n_i - 1) / 2.0
        c_j = (n_j - 1) / 2.0
        return PinholeModel(
            center=center,
            right=right,
            up=up,
            forward=forward,
            f_i=f_i,
            f_j=f_j,
            c_i=c_i,
            c_j=c_j,
            n_i=n_i,
            n_j=n_j,
        )

    # ---- forward: index -> ray direction (world/sensor frame) ------------

    def directions_for_indices(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Vectorized index -> unit ray direction. i, j: broadcastable arrays."""
        i = np.asarray(i, dtype=float)
        j = np.asarray(j, dtype=float)
        local_i = (i - self.c_i) / self.f_i
        local_j = (j - self.c_j) / self.f_j
        local = np.stack([local_i, local_j, np.ones_like(local_i)], axis=-1)
        world = (
            local[..., 0:1] * self.right
            + local[..., 1:2] * self.up
            + local[..., 2:3] * self.forward
        )
        return unit(world)

    def direction_for_index(self, i: float, j: float) -> np.ndarray:
        return self.directions_for_indices(np.asarray(i), np.asarray(j))

    # ---- inverse: 3D point -> continuous index ----------------------------

    def project_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized point -> continuous (i, j). Points behind the pinhole
        (local z <= 0) get NaN in both outputs."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        rel = points - self.center
        local_x = rel @ self.right
        local_y = rel @ self.up
        local_z = rel @ self.forward
        with np.errstate(invalid="ignore", divide="ignore"):
            i = self.f_i * local_x / local_z + self.c_i
            j = self.f_j * local_y / local_z + self.c_j
        behind = local_z <= 0
        i = np.where(behind, np.nan, i)
        j = np.where(behind, np.nan, j)
        return i, j

    def project_point(self, point: np.ndarray) -> tuple[float, float]:
        i, j = self.project_points(np.asarray(point)[None, :])
        return float(i[0]), float(j[0])

    def project_and_pixel_indices(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Point -> continuous (i, j) *and* the nearest discrete pixel
        address, computed together in one pass (avoids reprojecting when a
        caller needs both -- see measurement.py, which triangulates from the
        continuous value but records the discrete one for bookkeeping only,
        per the V1.1 sub-pixel-reconstruction fix).

        Returns (continuous_i, continuous_j, discrete_i, discrete_j, valid).
        ``valid`` is False wherever the point projects behind the pinhole or
        the discrete address falls outside [0, n_i) x [0, n_j) -- continuous_i/j
        are NaN in the behind-the-pinhole case but may still be finite (just
        out of array bounds) when only the discrete/bounds check fails.
        """
        fi, fj = self.project_points(points)
        with np.errstate(invalid="ignore"):
            ii = np.rint(fi)
            jj = np.rint(fj)
            valid = (
                ~np.isnan(fi)
                & ~np.isnan(fj)
                & (ii >= 0)
                & (ii < self.n_i)
                & (jj >= 0)
                & (jj < self.n_j)
            )
            # NaN -> int cast below is a well-defined (if meaningless) value
            # for behind-the-pinhole points; `valid` is already False for
            # those, so callers never look at the result.
            ii_int = ii.astype(int)
            jj_int = jj.astype(int)
        return fi, fj, ii_int, jj_int, valid

    def pixel_indices(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized point -> nearest in-bounds integer (i, j), or (-1, -1)
        (sentinel, not a valid index) where the point projects behind the
        pinhole or outside the [0, n_i) x [0, n_j) array."""
        _fi, _fj, ii, jj, valid = self.project_and_pixel_indices(points)
        ii_out = np.where(valid, ii, -1)
        jj_out = np.where(valid, jj, -1)
        return ii_out, jj_out
