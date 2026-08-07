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
    """Nearest ray-mesh intersection for a shared ray origin.

    Parameters
    ----------
    origin : (3,) float -- single ray origin, shared by every ray (true for
        both the projector and the camera in this sensor model).
    directions : (N, 3) float -- unit ray directions.
    mesh : trimesh.Trimesh

    Returns
    -------
    locations : (N, 3) float, NaN rows where the ray misses every triangle.
    hit : (N,) bool
    """
    origin = np.asarray(origin, dtype=float)
    directions = np.asarray(directions, dtype=float)
    n_rays = directions.shape[0]

    best_t = np.full(n_rays, np.inf)
    hit = np.zeros(n_rays, dtype=bool)

    triangles = mesh.triangles  # (T, 3, 3)
    for v0, v1, v2 in triangles:
        e1 = v1 - v0
        e2 = v2 - v0

        p = np.cross(directions, e2)  # (N, 3)
        det = p @ e1  # (N,)
        parallel = np.abs(det) < eps
        inv_det = np.divide(1.0, det, out=np.full_like(det, np.nan), where=~parallel)

        t_vec = origin - v0  # (3,)
        u = (p @ t_vec) * inv_det

        q = np.cross(t_vec, e1)  # (3,)
        v = (directions @ q) * inv_det
        t = (e2 @ q) * inv_det

        valid = (
            ~parallel & (u >= -eps) & (u <= 1 + eps) & (v >= -eps) & (u + v <= 1 + eps) & (t > eps)
        )
        better = valid & (t < best_t)
        best_t = np.where(better, t, best_t)
        hit = hit | better

    locations = np.where(hit[:, None], origin[None, :] + best_t[:, None] * directions, np.nan)
    return locations, hit
