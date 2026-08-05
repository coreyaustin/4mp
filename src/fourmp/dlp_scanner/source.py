"""DLP projector model: a pinhole 'inverse camera' that emits a scan-line pattern."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Pose, look_at_pose
from .patterns import SourcePattern
from .sut import SutHit, SutWorld


@dataclass
class Source:
    pose: Pose
    resolution: tuple[int, int]  # (rows, cols)
    fov_deg: tuple[float, float]  # (horizontal, vertical) full field of view
    pattern: SourcePattern
    working_distance_mm: float = 470.5
    magnification: float = 1.0
    focus_spot_diameter_mm: float = 0.0
    defocus_half_angle_deg: float = 0.1

    @classmethod
    def at(
        cls,
        position,
        target=(0.0, 0.0, 0.0),
        resolution=(1600, 2716),  # (rows, cols) -- DLP670S native
        chip_size_mm=(8.64, 14.67),  # (height, width) DMD active area
        focal_length_mm=23.0,
        working_distance_mm=470.5,
        defocus_half_angle_deg=0.1,
        pattern: SourcePattern | None = None,
        world_up=(0.0, 0.0, 1.0),
    ) -> "Source":
        # world_up = the table's Z spin axis (the natural "up" for how this rig gets
        # mounted). With a horizontal source/camera baseline, that keeps a scan line's
        # row-plane normal aligned with the baseline instead of perpendicular to it —
        # see `row_plane` — which is what keeps triangulation well-conditioned.
        pose = look_at_pose(position, target, world_up=world_up)

        # Thin-lens imaging equation 1/f = 1/s0 + 1/s_i, solved for the DMD-to-lens
        # distance s0 given a fixed focal length and working distance (lens to the
        # nominal SUT plane) -- see source_optics_spec.md for the locked hardware
        # numbers this is meant to reproduce (s0~=24.18mm, M~=19.5x, FOV~=(33.7,20.3)).
        s_i = working_distance_mm
        s0 = 1.0 / (1.0 / focal_length_mm - 1.0 / s_i)
        magnification = s_i / s0

        half_fov_h = np.arctan((chip_size_mm[1] / 2.0) / s0)
        half_fov_v = np.arctan((chip_size_mm[0] / 2.0) / s0)
        fov_deg = (np.degrees(2.0 * half_fov_h), np.degrees(2.0 * half_fov_v))

        # Pixel pitch derived from chip size / resolution (not hardcoded) so this
        # stays correct if either changes -- e.g. a coarse test resolution samples
        # bigger chunks of the same physical chip, so its "pixel" footprint grows
        # proportionally, which is the physically correct behavior.
        pixel_pitch_mm = chip_size_mm[1] / resolution[1]
        focus_spot_diameter_mm = pixel_pitch_mm * magnification

        return cls(
            pose=pose,
            resolution=resolution,
            fov_deg=fov_deg,
            pattern=pattern or SourcePattern.all_on(resolution),
            working_distance_mm=working_distance_mm,
            magnification=magnification,
            focus_spot_diameter_mm=focus_spot_diameter_mm,
            defocus_half_angle_deg=defocus_half_angle_deg,
        )

    def footprint_diameter_mm(self, defocus_distance_mm: float) -> float:
        """Illuminated spot diameter at a hit `defocus_distance_mm` from the nominal
        working distance (already `abs()`-ed by the caller).

        `defocus_half_angle_deg` is a placeholder, not derived from the locked optics
        table -- it depends on the lens's aperture/depth-of-focus, which isn't
        specified anywhere in the current numbers. Default doubles the at-focus spot
        diameter over ~30mm of defocus; tune against real captured data later, same
        spirit as `shininess`/`cone_pixel_radius` elsewhere in this codebase.
        """
        growth = 2.0 * defocus_distance_mm * np.tan(np.radians(self.defocus_half_angle_deg))
        return self.focus_spot_diameter_mm + growth

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

    def row_plane(self, row: float):
        """World-space plane swept by a single DMD row (all columns) — the plane a
        projected scan line traces through the projector's optical center."""
        _, cols = self.resolution
        d_left = self.pixel_direction_world(row, 0)
        d_right = self.pixel_direction_world(row, cols - 1)
        normal = np.cross(d_left, d_right)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            return None
        return self.pose.position, normal / norm

    def project(self, sut_world: SutWorld) -> list["SourceHit"]:
        """Cast a ray per active source pixel and find where it hits the SUT."""
        rows, cols = self.resolution
        hits = []
        for row in range(rows):
            for col in range(cols):
                intensity = self.pattern.image[row, col]
                if intensity <= 0.0:
                    continue
                direction = self.pixel_direction_world(row, col)
                hit = sut_world.intersect_ray(self.pose.position, direction)
                if hit is None:
                    continue
                distance_mm = float(np.linalg.norm(hit.world_point - self.pose.position))
                defocus_distance_mm = abs(distance_mm - self.working_distance_mm)
                footprint_diameter_mm = self.footprint_diameter_mm(defocus_distance_mm)
                hits.append(
                    SourceHit(
                        row=row,
                        col=col,
                        intensity=float(intensity),
                        sut_hit=hit,
                        footprint_diameter_mm=footprint_diameter_mm,
                    )
                )
        return hits


@dataclass
class SourceHit:
    row: int
    col: int
    intensity: float
    sut_hit: SutHit
    footprint_diameter_mm: float
