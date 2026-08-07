"""Rigid transforms and ray math shared by the measurement and reconstruction engines.

Sensor-frame convention (fixed for the whole package -- see architecture-decisions.md
"Part/face pose" and the projector-model correction relayed 2026-08-07):

    X -- baseline / scan-step axis (188mm short measurement-area dimension).
    Y -- line axis (319mm long measurement-area dimension); also the rotary
         stage's rotation axis (the DMD's 2716-mirror line is treated as
         "vertical" for pose purposes -- see part.py).
    Z -- boresight/bisector axis (depth, positive from the sensor toward the part).

The projector's optical center sits at X = -baseline/2, the camera's at
X = +baseline/2, both at Y = 0, Z = 0, each aimed at the reference point
(0, 0, working_distance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalize vector(s) along ``axis``. Zero vectors are returned unchanged."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    safe_norm = np.where(norm == 0, 1.0, norm)
    return v / safe_norm


def rotation_y(theta: float) -> np.ndarray:
    """3x3 rotation matrix about the Y axis (right-handed, active rotation).

    R_y(theta) @ (x, y, z) rotates (x, z) toward +Z as theta increases from
    atan2(-x, z) toward 0 -- see part.py's theta_face derivation.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


@dataclass(frozen=True)
class Transform:
    """A rigid transform: point' = R @ point + t."""

    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)

    @staticmethod
    def identity() -> "Transform":
        return Transform(np.eye(3), np.zeros(3))

    @staticmethod
    def translation(t: np.ndarray) -> "Transform":
        return Transform(np.eye(3), np.asarray(t, dtype=float))

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        """Transform an (N, 3) or (3,) array of points."""
        points = np.asarray(points, dtype=float)
        return points @ self.R.T + self.t

    def apply_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Rotate (but don't translate) an (N, 3) or (3,) array of direction vectors."""
        vectors = np.asarray(vectors, dtype=float)
        return vectors @ self.R.T

    def then(self, outer: "Transform") -> "Transform":
        """Compose: apply self first, then ``outer``. Equivalent to outer ∘ self."""
        return Transform(outer.R @ self.R, outer.R @ self.t + outer.t)


def closest_points_between_rays(
    o1: np.ndarray, d1: np.ndarray, o2: np.ndarray, d2: np.ndarray
) -> tuple[np.ndarray, float]:
    """Midpoint of the shortest segment connecting two (generally skew) 3D rays.

    Used by the reconstruction engine's triangulation step: in a perfectly
    noiseless world the back-projected camera ray and the projector ray for the
    same scan step would intersect exactly, but camera-pixel quantization means
    they generally miss by a small amount. Returns (midpoint, gap_distance)
    where gap_distance is the residual separation -- a useful quantization-error
    diagnostic in its own right.

    ``d1``/``d2`` are assumed unit vectors. Degenerate (near-parallel) rays
    return the midpoint of the two origins with an infinite gap rather than
    raising, since the sensor geometry here (finite triangulation half-angle)
    never actually produces parallel projector/camera rays.
    """
    o1 = np.asarray(o1, dtype=float)
    o2 = np.asarray(o2, dtype=float)
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)

    r = o1 - o2
    b = float(np.dot(d1, d2))
    d = float(np.dot(d1, r))
    e = float(np.dot(d2, r))
    denom = 1.0 - b * b

    if abs(denom) < 1e-12:
        midpoint = 0.5 * (o1 + o2)
        return midpoint, float("inf")

    t1 = (b * e - d) / denom
    t2 = (e - b * d) / denom
    p1 = o1 + t1 * d1
    p2 = o2 + t2 * d2
    return 0.5 * (p1 + p2), float(np.linalg.norm(p1 - p2))


def closest_points_between_rays_batch(
    o1: np.ndarray, d1: np.ndarray, o2: np.ndarray, d2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`closest_points_between_rays`.

    ``o1``/``o2`` are single (3,) origins (this sensor's projector/camera
    optical centers are fixed -- only ray direction varies per hit).
    ``d1``/``d2`` are (N, 3) unit direction arrays. Returns (points (N, 3),
    gaps (N,)).
    """
    o1 = np.asarray(o1, dtype=float)
    o2 = np.asarray(o2, dtype=float)
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)

    r = o1 - o2  # (3,), constant across hits
    b = np.sum(d1 * d2, axis=-1)  # (N,)
    d = d1 @ r  # (N,)
    e = d2 @ r  # (N,)
    denom = 1.0 - b * b

    safe_denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    t1 = (b * e - d) / safe_denom
    t2 = (e - b * d) / safe_denom

    p1 = o1 + t1[:, None] * d1
    p2 = o2 + t2[:, None] * d2
    points = 0.5 * (p1 + p2)
    gaps = np.linalg.norm(p1 - p2, axis=-1)
    return points, gaps
