"""STL ingestion, face-frame computation, and the T_mount/theta_face pose math.

Frame convention (see geometry.py for the full sensor-frame writeup):

- **Part frame** -- native to the ingested STL. This package assumes part
  files are authored with +Y as "up" (the rotary stage's rotation axis,
  which is also the sensor's line axis -- see below); there's no STL
  metadata to confirm this from, and the spec's companion part schema that
  will eventually pin down such conventions doesn't exist yet, so this is a
  documented placeholder assumption, not a derived fact. The test cube
  fixture (fixtures.py) is authored to satisfy it by construction.
- **Face frame** -- per face: centroid at origin, outward normal, one
  in-plane axis. Computed once per selected face, in part-frame coordinates
  (i.e. before T_mount/theta_face are applied).
- **Sensor frame** -- see geometry.py.

Pose math (``pose(face) = R_z(theta_face) . T_mount`` in the spec's
notation; here "z" names the stage's own rotation axis, which corresponds to
this package's sensor-frame Y -- see geometry.py):

T_mount is a pure translation (no rotation -- "idealized: part's bottom
flush on the stage top, centered on the rotation axis") that (a) centers the
part's horizontal (X, Z) bounding-box on the rotation axis and (b) sets the
part's bottom (min Y) to the stage's mounting-frame origin. rotation_y
(theta_face) then spins the now-centered part *in place* about that same
origin -- a true in-place rotation about the physical rotation axis, which
is what "rotate about the stage's rotation axis" has to mean physically.

The stage's rotation axis is itself fixed at sensor-frame (X=0, Z=working
distance) -- i.e. centered on the baseline and sitting at the working
distance -- since that's the only placement consistent with "centered on the
rotation axis" (X=0) and a rotary table positioned at the sensor's working
distance. This isn't literally handed to us as a third term in the spec's
one-line pose formula, but it's the necessary way to make that formula's two
pieces (a translation with no rotation, then a rotation about the frame
origin) add up to "the part sits in front of the sensor and spins in place,"
so it's applied here as part of getting a face's pose into sensor space
(``Part.pose``), on top of the spec's own T_mount and theta_face pieces
(both retained on the object, unchanged, for inspection/debugging).

The stage's *vertical* (Y) placement relative to the sensor's optical axis
isn't specified at all (same category as the camera FOV margin -- a
non-blocking open parameter per the spec) -- chosen here to vertically
center the mounted part in the sensor's line-axis FOV.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from fourmp.sensor_sim.geometry import Transform, rotation_y, unit
from fourmp.sensor_sim.sensor_config import SensorConfig


@dataclass(frozen=True)
class FaceFrame:
    """Per-face local frame, in part-frame coordinates (pre-pose)."""

    centroid: np.ndarray  # (3,)
    normal: np.ndarray  # (3,), unit, outward
    in_plane_axis: np.ndarray  # (3,), unit, perpendicular to normal


@dataclass
class Part:
    """Minimal placeholder Part object (spec: "no final schema yet").

    ``mesh`` is the selected face's geometry only (not the whole part),
    already posed into sensor space -- ready to hand directly to the
    measurement engine's ray tracer and to the validation harness's
    ground-truth sampler (both should use this exact mesh, per spec).
    """

    mesh: trimesh.Trimesh  # posed into sensor space
    face_frame: FaceFrame  # part-frame coordinates
    T_mount: Transform
    theta_face_rad: float
    pose: Transform  # part frame -> sensor frame
    brdf: None = None  # unused in V1 -- binary intensity has no photometric model
    height_map: np.ndarray | None = None  # filled in by the reconstruction engine


def _validate_and_repair(mesh: trimesh.Trimesh, source: str) -> trimesh.Trimesh:
    if not mesh.is_watertight:
        mesh.fill_holes()
        trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight:
        warnings.warn(
            f"{source}: mesh is not watertight after repair attempt; "
            "proceeding anyway (V1 does no occlusion/self-intersection handling)."
        )
    if not mesh.is_winding_consistent:
        trimesh.repair.fix_winding(mesh)
    return mesh


def load_mesh(stl_path: str | Path) -> trimesh.Trimesh:
    """Import + validate/repair an STL. Units are assumed millimeters
    (STL carries no unit metadata -- see architecture-decisions.md's CAD
    ingestion notes; there's no companion schema yet to confirm this from)."""
    mesh = trimesh.load(str(stl_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{stl_path}: expected a single mesh, got {type(mesh)}")
    return _validate_and_repair(mesh, source=str(stl_path))


def _select_facet(
    mesh: trimesh.Trimesh, normal_hint: np.ndarray, rotation_axis: np.ndarray, max_angle_deg: float
) -> int:
    """Find the coplanar facet whose outward normal best matches ``normal_hint``.

    Rejects any candidate whose normal is nearly parallel to ``rotation_axis``
    (the stage's rotation axis) -- a face like that can't be rotated into
    view by a single-axis stage and is explicitly out of scope (the part's
    top/bottom faces).
    """
    normal_hint = unit(np.asarray(normal_hint, dtype=float))
    facets_normal = mesh.facets_normal
    if len(facets_normal) == 0:
        raise ValueError("mesh has no coplanar facets to select a face from")

    alignment = facets_normal @ normal_hint
    best = int(np.argmax(alignment))
    if alignment[best] < math.cos(math.radians(max_angle_deg)):
        raise ValueError(
            f"no facet found within {max_angle_deg} deg of normal hint {normal_hint}; "
            f"best alignment was {alignment[best]:.4f}"
        )

    axial_component = abs(float(facets_normal[best] @ unit(np.asarray(rotation_axis, dtype=float))))
    if axial_component > math.cos(math.radians(5.0)):
        raise ValueError(
            "selected facet's normal is nearly parallel to the stage rotation axis "
            "(a top/bottom-style face) -- out of scope for V1, per spec."
        )
    return best


def _facet_centroid(mesh: trimesh.Trimesh, facet_idx: int) -> np.ndarray:
    face_indices = mesh.facets[facet_idx]
    vertex_indices = np.unique(mesh.faces[face_indices].reshape(-1))
    return mesh.vertices[vertex_indices].mean(axis=0)


def _in_plane_axis(normal: np.ndarray) -> np.ndarray:
    """A deterministic in-plane axis: project the part-frame world axis least
    aligned with ``normal`` onto the face plane."""
    candidates = np.eye(3)
    alignment = np.abs(candidates @ normal)
    reference = candidates[int(np.argmin(alignment))]
    projected = reference - np.dot(reference, normal) * normal
    return unit(projected)


def build_face_frame(mesh: trimesh.Trimesh, facet_idx: int) -> FaceFrame:
    centroid = _facet_centroid(mesh, facet_idx)
    normal = unit(np.asarray(mesh.facets_normal[facet_idx], dtype=float))
    return FaceFrame(centroid=centroid, normal=normal, in_plane_axis=_in_plane_axis(normal))


ROTATION_AXIS = np.array([0.0, 1.0, 0.0])  # sensor-frame Y == part-frame Y (see module docstring)


def compute_t_mount(mesh: trimesh.Trimesh) -> tuple[Transform, float]:
    """T_mount: center the part's (X, Z) bounding box on the rotation axis and
    set its bottom (min Y) to the mount frame's origin. Also returns the
    part's Y-extent (height), used only to choose the stage's vertical
    placement in ``compute_pose`` -- not part of T_mount itself."""
    bounds = mesh.bounds  # (2, 3): [min, max]
    cx = 0.5 * (bounds[0, 0] + bounds[1, 0])
    cz = 0.5 * (bounds[0, 2] + bounds[1, 2])
    min_y = bounds[0, 1]
    t = np.array([-cx, -min_y, -cz])
    height = float(bounds[1, 1] - bounds[0, 1])
    return Transform.translation(t), height


def compute_theta_face(normal: np.ndarray) -> float:
    """Solve the stage rotation angle that aligns ``normal``'s in-plane
    (rotation-plane) component with the sensor's boresight (+Z).

    Decomposition: normal = (n_x, n_y, n_z) in part frame == sensor-frame
    axes (T_mount has no rotation). n_y is the component along the rotation
    axis (unused here -- by construction, out-of-scope faces with |n_y| ~ 1
    are rejected in face selection). (n_x, n_z) is the in-plane component;
    theta = atan2(-n_x, n_z) is the unique angle with
    R_y(theta) @ (n_x, 0, n_z) = (0, 0, sqrt(n_x^2 + n_z^2)), i.e. rotated
    onto +Z.
    """
    normal = unit(np.asarray(normal, dtype=float))
    return math.atan2(-normal[0], normal[2])


def compute_pose(
    mesh: trimesh.Trimesh, normal: np.ndarray, sensor_config: SensorConfig
) -> tuple[Transform, Transform, float]:
    """Full part-frame -> sensor-frame pose for the face with outward
    ``normal``. Returns (pose, T_mount, theta_face_rad)."""
    t_mount, part_height = compute_t_mount(mesh)
    theta = compute_theta_face(normal)
    stage_vertical_offset = -0.5 * part_height  # center the part in the line-axis FOV
    stage_origin = np.array([0.0, stage_vertical_offset, sensor_config.working_distance_mm])
    rotate_and_place = Transform(rotation_y(theta), stage_origin)
    pose = t_mount.then(rotate_and_place)
    return pose, t_mount, theta


def load_part(
    stl_path: str | Path,
    sensor_config: SensorConfig,
    face_normal_hint: np.ndarray = (1.0, 0.0, 0.0),
    face_normal_tol_deg: float = 1.0,
) -> Part:
    """Ingest an STL, select the face whose outward normal best matches
    ``face_normal_hint``, and pose it into sensor space."""
    mesh = load_mesh(stl_path)
    facet_idx = _select_facet(
        mesh, np.asarray(face_normal_hint), rotation_axis=ROTATION_AXIS, max_angle_deg=face_normal_tol_deg
    )
    face_frame = build_face_frame(mesh, facet_idx)

    pose, t_mount, theta = compute_pose(mesh, face_frame.normal, sensor_config)

    face_indices = mesh.facets[facet_idx]
    face_mesh = mesh.submesh([face_indices], append=True)
    face_mesh.vertices = pose.apply_points(face_mesh.vertices)

    return Part(
        mesh=face_mesh,
        face_frame=face_frame,
        T_mount=t_mount,
        theta_face_rad=theta,
        pose=pose,
    )
