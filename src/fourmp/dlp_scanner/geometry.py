"""Pose and ray-geometry helpers shared across the DLP scanner simulation.

World frame convention: the rotary table's rotation axis is world +Z, and the
table's center is the world origin (architecture doc, section 0). Source and
camera poses are fixed rigid bodies expressed in this frame; the SUT pose is
the only thing that moves, driven by the table rotation angle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose:
    """A rigid-body pose: world position plus an orthonormal (right, up, forward) frame."""

    position: np.ndarray  # (3,)
    right: np.ndarray  # (3,)
    up: np.ndarray  # (3,)
    forward: np.ndarray  # (3,)

    def to_world(self, local_point: np.ndarray) -> np.ndarray:
        return (
            self.position
            + local_point[0] * self.right
            + local_point[1] * self.up
            + local_point[2] * self.forward
        )

    def direction_to_world(self, local_dir: np.ndarray) -> np.ndarray:
        return local_dir[0] * self.right + local_dir[1] * self.up + local_dir[2] * self.forward

    def to_local(self, world_point: np.ndarray) -> np.ndarray:
        rel = world_point - self.position
        return np.array([rel @ self.right, rel @ self.up, rel @ self.forward])


def look_at_pose(position, target=(0.0, 0.0, 0.0), world_up=(0.0, 0.0, 1.0)) -> Pose:
    """Build a Pose whose forward axis points from `position` at `target` (the table origin)."""
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    world_up = np.asarray(world_up, dtype=float)

    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return Pose(position=position, right=right, up=up, forward=forward)


def rotation_about_z(angle_rad: float) -> np.ndarray:
    """Rotation matrix for the rotary table's single degree of freedom (about world +Z)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ray_plane_intersection(ray_origin, ray_dir, plane_point, plane_normal):
    """First forward intersection of a ray with a plane, or None if parallel/behind."""
    denom = ray_dir @ plane_normal
    if abs(denom) < 1e-9:
        return None
    t = (plane_point - ray_origin) @ plane_normal / denom
    if t <= 0:
        return None
    return ray_origin + t * ray_dir
