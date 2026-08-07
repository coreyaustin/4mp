"""Vectorized ray-triangle intersection (Moller-Trumbore), looped over the
mesh's (typically very few) triangles and vectorized over rays.

trimesh's own RayMeshIntersector needs an rtree-based broad-phase tree for
its ray_triangle backend, which pulls in the native ``rtree``/libspatialindex
dependency. Our meshes are single flat faces (2 triangles for the cube test
part) -- a broad-phase spatial index buys nothing at that scale, so this
implements the intersection test directly instead of taking on that
dependency for no benefit.
"""

from __future__ import annotations

import numpy as np
import trimesh


def nearest_intersection(
    origin: np.ndarray, directions: np.ndarray, mesh: trimesh.Trimesh, eps: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest ray-mesh intersection.

    ``origin`` and ``directions`` broadcast against each other: either may be
    a single (3,) vector (shared by every ray) or an (N, 3) array (one per
    ray). The projector/camera fire many rays from one shared optical center
    (constant origin, varying direction); ground-truth grid sampling fires
    one ray per grid cell in a shared direction from a per-cell origin
    (varying origin, constant direction). At least one of the two must be
    (N, 3) so N is determined.

    Parameters
    ----------
    origin : (3,) or (N, 3) float
    directions : (3,) or (N, 3) float -- unit ray direction(s)
    mesh : trimesh.Trimesh

    Returns
    -------
    locations : (N, 3) float, NaN rows where the ray misses every triangle.
    hit : (N,) bool
    """
    origin = np.asarray(origin, dtype=float)
    directions = np.asarray(directions, dtype=float)

    if origin.ndim == 2:
        n_rays = origin.shape[0]
    elif directions.ndim == 2:
        n_rays = directions.shape[0]
    else:
        raise ValueError("at least one of origin/directions must be an (N, 3) array")

    def _broadcast(a: np.ndarray) -> np.ndarray:
        return np.broadcast_to(a, (n_rays,))

    best_t = np.full(n_rays, np.inf)
    hit = np.zeros(n_rays, dtype=bool)

    triangles = mesh.triangles  # (T, 3, 3)
    for v0, v1, v2 in triangles:
        e1 = v1 - v0  # (3,)
        e2 = v2 - v0  # (3,)

        p = np.cross(directions, e2)  # (3,) or (N, 3)
        det = _broadcast(np.sum(p * e1, axis=-1))  # (N,)
        parallel = np.abs(det) < eps
        inv_det = np.full(n_rays, np.nan)
        np.divide(1.0, det, out=inv_det, where=~parallel)

        t_vec = origin - v0  # (3,) or (N, 3)
        u = _broadcast(np.sum(p * t_vec, axis=-1)) * inv_det

        q = np.cross(t_vec, e1)  # (3,) or (N, 3)
        v = _broadcast(np.sum(directions * q, axis=-1)) * inv_det
        t = _broadcast(np.sum(e2 * q, axis=-1)) * inv_det

        valid = (
            ~parallel & (u >= -eps) & (u <= 1 + eps) & (v >= -eps) & (u + v <= 1 + eps) & (t > eps)
        )
        better = valid & (t < best_t)
        best_t = np.where(better, t, best_t)
        hit = hit | better

    with np.errstate(invalid="ignore"):
        # best_t is +inf for missed rays -- inf * direction is a harmless NaN/inf
        # that np.where immediately discards in favor of the np.nan branch below.
        hit_points = np.broadcast_to(origin, (n_rays, 3)) + best_t[:, None] * np.broadcast_to(
            directions, (n_rays, 3)
        )
    locations = np.where(hit[:, None], hit_points, np.nan)
    return locations, hit
