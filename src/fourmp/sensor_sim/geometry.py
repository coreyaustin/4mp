"""Rigid transforms and ray math shared by the measurement and reconstruction engines.

Sensor-frame convention (V1.3 -- aligned to 4MP's cell-wide reference doc,
``coordinate_transforms_equations_v3.md`` / ``transform_point_cloud_v3.py``;
see architecture-decisions.md's "Coordinate frame convention" section for
the full derivation). This frame is named **O_s** (scanner) in that doc:

    X -- depth / boresight axis. Physical range from the sensor increases as
         X becomes more *negative* (the doc's "scanner depth = -X", an
         artifact of the projector/camera optics being drawn from the
         image-sensor side). Positive X points back toward the sensor.
    Y -- lateral axis: the projector/camera baseline. Projector optical
         center at Y = -baseline/2, camera at Y = +baseline/2.
    Z -- the DMD's line axis (2716-mirror axis) *and* the rotary stage's
         rotation axis. Per the doc's "Z down" convention for O_s, physical
         "up" is -Z.

The projector's optical center sits at (X, Y, Z) = (0, -baseline/2, 0), the
camera's at (0, +baseline/2, 0), each aimed at the reference point
(-working_distance, 0, 0).

Before V1.3 this package used its own ad hoc frame (X = baseline, Y = line/
rotation axis, Z = boresight, positive toward the part) which had drifted
from 4MP's cell-wide convention. The relabeling is a pure change of basis --
a proper rotation, not a mirror -- given by:

    X_(O_s) = -Z_ours     Y_(O_s) = +X_ours     Z_(O_s) = -Y_ours

None of the ray-tracing/triangulation math below depends on which axis is
called what; only the hardcoded geometry in sensor_config.py (and the
default ``world_up`` in pinhole.py) needed re-deriving under the new labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalize vector(s) along ``axis``. Zero vectors are returned unchanged."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    safe_norm = np.where(norm == 0, 1.0, norm)
    return v / safe_norm


def axis_alignment_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """3x3 rotation mapping unit vector ``source`` onto unit vector ``target``
    (Rodrigues' rotation formula). General-purpose -- works for any pair of
    directions, not just cardinal axes.

    Used by part.py's V1.2 up-axis remap (an arbitrary part file's own "up"
    direction onto this package's internal rotation axis) rather than
    special-casing cardinal directions.
    """
    a = unit(np.asarray(source, dtype=float))
    b = unit(np.asarray(target, dtype=float))
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))

    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # Antiparallel: no unique rotation axis from the cross product: pick
        # any axis perpendicular to `a` and rotate 180 deg about it.
        candidates = np.eye(3)
        alignment = np.abs(candidates @ a)
        k = candidates[int(np.argmin(alignment))]
        k = unit(k - np.dot(k, a) * a)
        return 2.0 * np.outer(k, k) - np.eye(3)

    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def rotation_z(yaw: float) -> np.ndarray:
    """3x3 rotation matrix for yaw -- rotation about +Z (right-handed, active).

    Matches the cell-wide doc's intrinsic Z->X->Y (yaw, pitch, roll) Euler
    convention with pitch = roll = 0 (our rotary stage has one rotational
    DOF): R(yaw, 0, 0) = R_z(yaw). Z is the stage's rotation axis in the
    O_s/O_r frames (see module docstring) -- this replaces the pre-V1.3
    ``rotation_y``, which rotated about this package's old Y axis before the
    relabel.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


@dataclass(frozen=True)
class Transform:
    """A rigid transform: point' = R @ point + t.

    Matches the cell-wide doc's convention exactly: ``p_A = R(A<-B) @ p_B +
    t(A<-B)``, stacked as a 4x4 homogeneous matrix for composition/inversion
    purposes (``.then()`` / ``.invert()`` below implement that 4x4 algebra
    directly on the (R, t) pair rather than materializing the matrix).
    """

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
        """Compose: apply self first, then ``outer``. Equivalent to outer ∘ self.

        If self = T(B<-C) and outer = T(A<-B), the result is T(A<-C) -- the
        cell-wide doc's ``compose: T(A<-C) = T(A<-B) . T(B<-C)``.
        """
        return Transform(outer.R @ self.R, outer.R @ self.t + outer.t)

    def invert(self) -> "Transform":
        """T(B<-A) from T(A<-B): R(B<-A) = R^T, t(B<-A) = -R^T @ t.

        Used to derive T(O_r<-O_s) from the T(O_s<-O_r) this package builds
        directly (see part.py) -- the cell-wide doc's ``invert`` rule,
        reusing this one class rather than a parallel implementation.
        """
        Rt = self.R.T
        return Transform(Rt, -Rt @ self.t)

    def to_quat_trans(self) -> tuple[np.ndarray, np.ndarray]:
        """Split into a scalar-last quaternion ``[x, y, z, w]`` and a
        translation, matching ``transform_point_cloud_v3.py``'s
        ``transform_to_quat_trans`` / ``transform_point_cloud`` signature
        (scalar-last is also SciPy's own default ``as_quat()`` order, so no
        reordering is needed at this boundary)."""
        quat = Rotation.from_matrix(self.R).as_quat()  # [x, y, z, w]
        return quat, self.t.copy()


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
