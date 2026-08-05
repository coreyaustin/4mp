"""Pinhole camera model: captures shaded source hits into an image."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Pose, look_at_pose
from .sut import SutWorld


@dataclass
class Camera:
    pose: Pose
    resolution: tuple[int, int]  # (rows, cols)
    fov_deg: tuple[float, float]  # (horizontal, vertical) full field of view
    cone_pixel_radius: int = 2  # half-width of the square BRDF-accumulation window, in pixels

    @classmethod
    def at(
        cls,
        position,
        target=(0.0, 0.0, 0.0),
        resolution=(60, 64),
        fov_deg=(24.0, 18.0),
        world_up=(0.0, 0.0, 1.0),
        cone_pixel_radius=2,
    ) -> "Camera":
        # See Source.at: matching world_up keeps the camera's axes consistent with the
        # source's, so triangulation against a scan line's row-plane stays well-conditioned.
        return cls(
            pose=look_at_pose(position, target, world_up=world_up),
            resolution=resolution,
            fov_deg=fov_deg,
            cone_pixel_radius=cone_pixel_radius,
        )

    def _tan_half_fov(self):
        fx, fy = self.fov_deg
        return np.tan(np.radians(fx) / 2), np.tan(np.radians(fy) / 2)

    def pixel_direction_world(self, row: float, col: float) -> np.ndarray:
        rows, cols = self.resolution
        tx, ty = self._tan_half_fov()
        u = ((col + 0.5) / cols * 2 - 1) * tx
        v = ((row + 0.5) / rows * 2 - 1) * ty
        d_local = np.array([u, -v, 1.0])
        d_local = d_local / np.linalg.norm(d_local)
        return self.pose.direction_to_world(d_local)

    def world_to_pixel(self, world_point: np.ndarray):
        local = self.pose.to_local(world_point)
        if local[2] <= 1e-6:
            return None
        tx, ty = self._tan_half_fov()
        u = (local[0] / local[2]) / tx
        v = -(local[1] / local[2]) / ty
        rows, cols = self.resolution
        col = (u + 1) / 2 * cols - 0.5
        row = (v + 1) / 2 * rows - 0.5
        if not (0 <= row < rows and 0 <= col < cols):
            return None
        return row, col

    def capture(self, source_hits, sut_world: SutWorld, source_position: np.ndarray) -> np.ndarray:
        """Accumulate BRDF-weighted radiance from every hit into a window of nearby pixels.

        Each hit's ideal projection centers a square `cone_pixel_radius`-wide
        window; every pixel in that window gets its own outgoing-direction
        evaluation of the SUT's BRDF, and contributions from every hit and every
        window pixel are summed (never overwritten) — this is what turns a
        geometrically crisp point into a "fuzzy" line for the centroid to find.
        No lens/aperture model or energy normalization: `cone_pixel_radius` is a
        bare prototype knob, tuned against real captured data later.
        """
        rows, cols = self.resolution
        frame = np.zeros((rows, cols))
        radius = self.cone_pixel_radius
        for hit in source_hits:
            ideal = self.world_to_pixel(hit.sut_hit.world_point)
            if ideal is None:
                continue
            ideal_row, ideal_col = ideal
            light_dir = source_position - hit.sut_hit.world_point
            light_dir = light_dir / np.linalg.norm(light_dir)

            row_lo = max(0, int(round(ideal_row)) - radius)
            row_hi = min(rows - 1, int(round(ideal_row)) + radius)
            col_lo = max(0, int(round(ideal_col)) - radius)
            col_hi = min(cols - 1, int(round(ideal_col)) + radius)
            for row in range(row_lo, row_hi + 1):
                for col in range(col_lo, col_hi + 1):
                    view_dir = -self.pixel_direction_world(row, col)
                    radiance = sut_world.shade(hit.sut_hit, hit.intensity, light_dir, view_dir)
                    frame[row, col] += radiance
        return frame
