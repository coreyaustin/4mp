"""Surface-under-test model: a parametric height field posed on the rotary table.

BRDF is Phong (specular lobe + diffuse floor) via `k_specular`/`k_diffuse`/
`shininess`, tuned against real captured data once the rig exists.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .geometry import rotation_about_z


@dataclass
class SutHit:
    local_point: np.ndarray
    world_point: np.ndarray
    normal_world: np.ndarray


@dataclass
class SUT:
    """A machinable part, expressed as a height field z = f(x, y) in table-local coordinates.

    `albedo` is currently unused by `SutWorld.shade` — Phong (`k_specular`/
    `k_diffuse`/`shininess`) replaced the old Lambertian-only model, which was
    the sole consumer of `albedo`. Kept on the dataclass for backward
    compatibility rather than silently dropped; a future pass may fold it into
    `k_diffuse` or remove it once the Phong parameterization is settled.
    """

    x: np.ndarray  # (nx,) local grid coordinates, mm
    y: np.ndarray  # (ny,) local grid coordinates, mm
    heights: np.ndarray  # (ny, nx) local z, mm
    albedo: float = 0.8
    k_specular: float = 1.0
    k_diffuse: float = 0.0
    shininess: float = 20.0

    @classmethod
    def flat(
        cls,
        half_extent_mm: float = 40.0,
        resolution: int = 81,
        z0: float = 0.0,
        albedo: float = 0.8,
        k_specular: float = 1.0,
        k_diffuse: float = 0.0,
        shininess: float = 20.0,
    ) -> "SUT":
        axis = np.linspace(-half_extent_mm, half_extent_mm, resolution)
        return cls(
            x=axis,
            y=axis,
            heights=np.full((resolution, resolution), z0),
            albedo=albedo,
            k_specular=k_specular,
            k_diffuse=k_diffuse,
            shininess=shininess,
        )

    def with_heights(self, heights: np.ndarray) -> "SUT":
        return SUT(
            x=self.x,
            y=self.y,
            heights=heights,
            albedo=self.albedo,
            k_specular=self.k_specular,
            k_diffuse=self.k_diffuse,
            shininess=self.shininess,
        )

    def transform(self, table_angle_rad: float) -> "SutWorld":
        return SutWorld(self, table_angle_rad)


class SutWorld:
    """The SUT posed in the world frame at a given rotary-table angle."""

    def __init__(self, sut: SUT, table_angle_rad: float):
        self.sut = sut
        self.angle = table_angle_rad
        self._rotation = rotation_about_z(table_angle_rad)
        self._height_interp = RegularGridInterpolator(
            (sut.y, sut.x), sut.heights, bounds_error=False, fill_value=np.nan
        )
        dzdy, dzdx = np.gradient(sut.heights, sut.y, sut.x)
        self._dzdx_interp = RegularGridInterpolator((sut.y, sut.x), dzdx, bounds_error=False, fill_value=0.0)
        self._dzdy_interp = RegularGridInterpolator((sut.y, sut.x), dzdy, bounds_error=False, fill_value=0.0)

    def height_at_local(self, x: float, y: float) -> float:
        return float(self._height_interp((y, x)))

    def normal_at_local(self, x: float, y: float) -> np.ndarray:
        dzdx = float(self._dzdx_interp((y, x)))
        dzdy = float(self._dzdy_interp((y, x)))
        n = np.array([-dzdx, -dzdy, 1.0])
        return n / np.linalg.norm(n)

    def local_to_world(self, local_point: np.ndarray) -> np.ndarray:
        return self._rotation @ local_point

    def world_to_local(self, world_point: np.ndarray) -> np.ndarray:
        return self._rotation.T @ world_point

    def direction_world_to_local(self, world_dir: np.ndarray) -> np.ndarray:
        return self._rotation.T @ world_dir

    def intersect_ray(self, ray_origin_world: np.ndarray, ray_dir_world: np.ndarray, max_iter: int = 12):
        """Ray / height-field intersection via Newton iteration in table-local coordinates."""
        o = self.world_to_local(ray_origin_world)
        d = self.direction_world_to_local(ray_dir_world)
        if abs(d[2]) < 1e-9:
            return None

        t = (float(np.nanmean(self.sut.heights)) - o[2]) / d[2]
        if t <= 0:
            return None

        for _ in range(max_iter):
            p = o + t * d
            h = self.height_at_local(p[0], p[1])
            if np.isnan(h):
                return None
            residual = p[2] - h
            dzdx = float(self._dzdx_interp((p[1], p[0])))
            dzdy = float(self._dzdy_interp((p[1], p[0])))
            slope = d[2] - (dzdx * d[0] + dzdy * d[1])
            if abs(slope) < 1e-9:
                break
            t -= residual / slope
            if t <= 0:
                return None

        p = o + t * d
        h = self.height_at_local(p[0], p[1])
        if np.isnan(h) or abs(p[2] - h) > 1e-2:
            return None

        normal_local = self.normal_at_local(p[0], p[1])
        return SutHit(local_point=p, world_point=self.local_to_world(p), normal_world=self._rotation @ normal_local)

    def shade(
        self,
        hit: SutHit,
        pattern_intensity: float,
        light_dir_world: np.ndarray,
        view_dir_world: np.ndarray,
    ) -> float:
        """Phong radiance: a specular lobe around the mirror direction plus a diffuse floor."""
        cos_theta = max(float(hit.normal_world @ light_dir_world), 0.0)
        reflect_dir = 2.0 * (hit.normal_world @ light_dir_world) * hit.normal_world - light_dir_world
        spec = max(float(reflect_dir @ view_dir_world), 0.0) ** self.sut.shininess
        return pattern_intensity * (self.sut.k_specular * spec + self.sut.k_diffuse * cos_theta)

    def corrected_with(self, world_points: np.ndarray) -> SUT:
        """Return a new SUT with the height field overwritten at reconstructed points.

        Nearest-grid-cell scatter — a placeholder for a real gridding/meshing step
        (architecture doc, open question 4, covers what "converged" should mean here).
        """
        heights = self.sut.heights.copy()
        for world_point in world_points:
            local_point = self.world_to_local(world_point)
            xi = int(np.clip(np.searchsorted(self.sut.x, local_point[0]), 0, len(self.sut.x) - 1))
            yi = int(np.clip(np.searchsorted(self.sut.y, local_point[1]), 0, len(self.sut.y) - 1))
            heights[yi, xi] = local_point[2]
        return self.sut.with_heights(heights)
