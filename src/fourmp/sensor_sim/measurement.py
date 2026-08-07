"""Measurement engine: the forward model.

Per scan step (one active DMD row = one value of the scan-step index i,
0..1599), every mirror in that row (j, 0..2715) that's "on" sends a ray from
the idealized inverse-pinhole projector. Per spec, this implementation:

1. Ray-traces each ray against the posed face mesh -> nearest intersection.
2. Forward-projects that 3D hit point through the camera's pinhole model to
   get a (row, col) pixel address.
3. Records a binary hit (pixel address, step index, mirror index) or nothing.

"Reflects off the mesh and heads toward the camera" (spec's phrasing) is
implemented as a direct forward-projection of the 3D hit point through the
camera model, rather than literal specular ray-reflection physics -- see the
architecture note in this module's tests. This matches the spec's own
"no BRDF, no occlusion" simplifications (a real specular reflection would
almost never land in the camera aperture except at one exact geometry, which
would make most of the face invisible -- inconsistent with wanting a useful,
mostly-fully-sampled V1 test face) and matches the reconstruction engine's
inverse model, which triangulates by intersecting a projector ray with a
back-projected camera ray -- exactly the geometric picture this produces.

Rays are all fired from the same projector optical center (only direction
varies by mirror), so the whole scan is one vectorized ray-mesh intersection
call rather than 1600 x 2716 individual calls.

**V1.1 sub-pixel fix:** the camera records *both* the continuous projected
(i, j) and the nearest discrete pixel address. The real sensor locates the
imaged line to sub-pixel precision via centroid/peak-finding over the line's
intensity profile; V1 has no photometric/PSF model yet to run a real
estimator over, so the continuous projection stands in as a perfect,
zero-error sub-pixel estimate (see sensor-sim-v1-spec.md's V1.1 section) --
a deliberate placeholder, the same idealization pattern as binary intensity
standing in for a real BRDF. The discrete address is kept alongside purely
for bookkeeping (e.g. collision detection between scan steps landing on the
same physical camera pixel); the continuous value is what actually feeds
triangulation (see reconstruction.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from fourmp.sensor_sim.raytrace import nearest_intersection
from fourmp.sensor_sim.sensor_config import SensorConfig


@dataclass
class ScanData:
    """Sparse raw scan data: one entry per recorded hit.

    The spec describes raw scan data as "a stack of line images, one per
    scan step" -- represented here sparsely (per-hit records) rather than as
    dense per-step images, since at most n_line_mirrors hits are possible
    per step and the vast majority of (mirror, camera-pixel) pairs never
    fire; a dense stack at full sensor resolution (1600 x 3104 x 5328
    booleans) would be many GB for no benefit.
    """

    step: np.ndarray  # (n_hits,) int, scan-step / projector i-index, 0..1599
    mirror: np.ndarray  # (n_hits,) int, projector j-index, 0..2715
    cam_i: np.ndarray  # (n_hits,) int, discrete/recorded camera pixel row -- bookkeeping only
    cam_j: np.ndarray  # (n_hits,) int, discrete/recorded camera pixel column -- bookkeeping only
    cam_i_continuous: np.ndarray  # (n_hits,) float, sub-pixel camera row -- feeds triangulation
    cam_j_continuous: np.ndarray  # (n_hits,) float, sub-pixel camera column -- feeds triangulation

    def __len__(self) -> int:
        return len(self.step)


def run_measurement(
    part_mesh: trimesh.Trimesh,
    sensor_config: SensorConfig,
    step_stride: int = 1,
    mirror_stride: int = 1,
) -> ScanData:
    """Forward-model a full scan of ``part_mesh`` (already posed into sensor
    space -- see part.py).

    ``step_stride``/``mirror_stride`` subsample the 1600 x 2716 mirror grid
    (e.g. for fast tests); both default to 1, i.e. the full confirmed
    1600-scan-step resolution.
    """
    projector = sensor_config.projector
    camera = sensor_config.camera

    steps = np.arange(0, projector.n_i, step_stride)
    mirrors = np.arange(0, projector.n_j, mirror_stride)
    ii, jj = np.meshgrid(steps, mirrors, indexing="ij")
    ii_flat = ii.ravel()
    jj_flat = jj.ravel()

    directions = projector.directions_for_indices(ii_flat, jj_flat)
    locations, hit = nearest_intersection(projector.center, directions, part_mesh)

    if not hit.any():
        empty_int = np.array([], dtype=int)
        empty_float = np.array([], dtype=float)
        return ScanData(
            step=empty_int,
            mirror=empty_int,
            cam_i=empty_int,
            cam_j=empty_int,
            cam_i_continuous=empty_float,
            cam_j_continuous=empty_float,
        )

    fi, fj, ii_disc, jj_disc, valid = camera.project_and_pixel_indices(locations[hit])

    return ScanData(
        step=ii_flat[hit][valid],
        mirror=jj_flat[hit][valid],
        cam_i=ii_disc[valid],
        cam_j=jj_disc[valid],
        cam_i_continuous=fi[valid],
        cam_j_continuous=fj[valid],
    )
