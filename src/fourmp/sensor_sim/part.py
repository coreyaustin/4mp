"""STL ingestion, face-frame computation, and the T_mount/theta_face pose math.

Frame convention (see geometry.py for the full O_s writeup):

- **Part frame (native)** -- whatever coordinate system the STL file itself
  uses. V1.2: since real part files are authored in CNC/CAM convention
  (Z as the vertical/spindle axis), the default assumed up-axis is +Z, not
  this package's earlier +Y placeholder -- override with ``up_axis=`` (the
  CLI's ``--up-axis X Y Z``) for files using a different convention.
- **Part frame (internal)** -- the native frame after a single fixed
  change-of-basis remap (``_remap_up_axis``) sends the file's up-axis onto
  this package's internal "physical up" direction (-Z in O_s, see below).
  Everything past ingestion (face selection, ``FaceFrame``, T_mount,
  theta_face) operates in this internal convention, not the file's native one.
- **Face frame** -- per face: centroid at origin, outward normal, one
  in-plane axis. Computed in the internal part frame (post-remap, pre-pose).
- **O_s** (scanner/optics) and **O_r** (rotary table) -- see geometry.py and
  this module's ``compute_pose``.

Pose math (spec's ``pose(face) = R_z(theta_face) . T_mount``; V1.3 exposes
this as two composable, named 4x4 transforms matching 4MP's cell-wide
convention, rather than one opaque part->sensor composition):

- **T_mount**: ``T(O_r <- part_frame)``, fixed per part (same for every
  face), pure translation -- "idealized: part's bottom flush on the stage
  top, centered on the rotation axis." O_r's rotation axis is Z (down-
  positive, see geometry.py), so "bottom flush" means the part's *maximum*
  Z (its lowest physical point) sits at the mount frame's Z = 0, and
  "centered on the rotation axis" means the (X, Y) bounding-box center sits
  at (0, 0).
- **T(O_s <- O_r) at theta_face**: rotation by theta_face (yaw, about O_r's
  shared Z axis) composed with a *fixed* translation placing O_r's origin at
  its known physical location in O_s -- (X, Y, Z) = (-working_distance, 0,
  stage_vertical_offset). That placement isn't literally a third term in the
  spec's one-line pose formula, but it's the necessary way to make T_mount
  (a translation) and the theta rotation (about the frame origin) add up to
  "the part sits in front of the sensor and spins in place about the real
  rotation axis" rather than orbiting the sensor's own origin.
- **T(O_r <- O_s)**: ``T(O_s <- O_r).invert()`` -- the doc's
  ``T(rotary<-scanner)``, exposed on ``Part`` so height (O_r's Y component,
  see regrid.py) and any future multi-pose fusion can use it directly rather
  than re-deriving it.

The stage's *vertical* placement (``stage_vertical_offset``, along O_r's Z)
relative to the sensor's optical axis isn't specified at all (same category
as the camera FOV margin -- non-blocking) -- chosen here to vertically
center the mounted part in the sensor's line-axis FOV.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from fourmp.sensor_sim.geometry import Transform, axis_alignment_rotation, rotation_z, unit
from fourmp.sensor_sim.sensor_config import SensorConfig


@dataclass(frozen=True)
class FaceFrame:
    """Per-face local frame, in internal part-frame coordinates (post-remap,
    pre-pose)."""

    centroid: np.ndarray  # (3,)
    normal: np.ndarray  # (3,), unit, outward
    in_plane_axis: np.ndarray  # (3,), unit, perpendicular to normal


@dataclass
class Part:
    """Minimal placeholder Part object (spec: "no final schema yet").

    ``mesh`` is the selected face's geometry only (not the whole part),
    already posed into O_s -- ready to hand directly to the measurement
    engine's ray tracer and to the validation harness's ground-truth
    sampler (both should use this exact mesh, per spec).
    """

    mesh: trimesh.Trimesh  # posed into O_s
    face_frame: FaceFrame  # internal part-frame coordinates
    T_mount: Transform  # T(O_r <- part_frame)
    theta_face_rad: float
    pose: Transform  # T(O_s <- part_frame)
    o_r_from_o_s: Transform  # T(O_r <- O_s) at this face's theta_face
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


# The rotation axis, in O_s labels (see geometry.py): Z. Physical "up" is
# -Z (O_s's "Z down" convention) -- see module docstring.
ROTATION_AXIS_OS = np.array([0.0, 0.0, 1.0])
PHYSICAL_UP_OS = -ROTATION_AXIS_OS


def _remap_up_axis(mesh: trimesh.Trimesh, up_axis: np.ndarray) -> trimesh.Trimesh:
    """V1.2: a single fixed change-of-basis, applied once at ingestion,
    sending the part file's own up-axis onto this package's internal
    "physical up" direction. General axis-to-axis rotation (works for any
    up_axis, not just cardinal directions) -- everything downstream
    (face selection, T_mount, theta_face) is written against the internal
    convention and needs no further change."""
    R = axis_alignment_rotation(up_axis, PHYSICAL_UP_OS)
    remapped = mesh.copy()
    remapped.vertices = remapped.vertices @ R.T
    return remapped


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


def compute_t_mount(mesh: trimesh.Trimesh) -> tuple[Transform, float]:
    """T(O_r <- part_frame): center the part's (X, Y) bounding box on the
    rotation axis and set its bottom to the mount frame's Z = 0. "Bottom"
    means *maximum* Z, since O_r's Z increases downward (see module
    docstring) -- the part's lowest physical point is its largest Z value.
    Also returns the part's Z-extent (vertical height), used only to choose
    the stage's vertical placement in ``compute_pose`` -- not part of
    T_mount itself."""
    bounds = mesh.bounds  # (2, 3): [min, max]
    cx = 0.5 * (bounds[0, 0] + bounds[1, 0])
    cy = 0.5 * (bounds[0, 1] + bounds[1, 1])
    max_z = bounds[1, 2]
    t = np.array([-cx, -cy, -max_z])
    height = float(bounds[1, 2] - bounds[0, 2])
    return Transform.translation(t), height


def compute_theta_face(normal: np.ndarray) -> float:
    """Solve the stage yaw that aligns ``normal``'s in-plane (rotation-plane)
    component with the boresight, expressed in O_s labels.

    Decomposition: normal = (n_x, n_y, n_z) in the internal part frame ==
    O_r axes (T_mount has no rotation). n_z is the component along the
    rotation axis (unused here -- by construction, out-of-scope faces with
    |n_z| ~ 1 are rejected in face selection). (n_x, n_y) is the in-plane
    component; theta = atan2(n_y, -n_x) is the unique yaw with
    R_z(theta) @ (n_x, n_y, 0) landing on (-r, 0, 0), r = sqrt(n_x^2+n_y^2)
    -- i.e. rotated onto the boresight (O_s's -X, since O_s's depth
    convention is "depth = -X": the face ends up facing away from the
    sensor along increasing physical range, matching the pre-V1.3 behavior
    under the axis relabel).
    """
    normal = unit(np.asarray(normal, dtype=float))
    return math.atan2(normal[1], -normal[0])


def height_in_o_r(points: np.ndarray, o_r_from_o_s: Transform) -> np.ndarray:
    """Signed height/deviation (mm) for O_s-frame ``points``: O_r's Y
    component after applying ``T(O_r <- O_s)`` (V1.3 -- see module
    docstring's "Pose math" section and regrid.py). Height is only
    meaningful relative to a specific face/pose's O_r, hence a function of
    that pose's transform rather than a fixed SensorConfig method."""
    return o_r_from_o_s.apply_points(points)[..., 1]


def compute_pose(
    mesh: trimesh.Trimesh, normal: np.ndarray, sensor_config: SensorConfig
) -> tuple[Transform, Transform, float, Transform]:
    """Full internal-part-frame -> O_s pose for the face with outward
    ``normal``. Returns (pose, T_mount, theta_face_rad, o_r_from_o_s).

    Two *different* rotations are involved here, easy to conflate since both
    are yaws about the same shared Z axis:

    - ``rotation_z(theta)`` rotates *this face's normal* onto the boresight
      -- the rotation the mesh itself needs for ray-tracing (``pose``).
    - O_r's own axes, relative to the mounted (T_mount-applied) frame, are
      independent of which face is selected -- and work out to a *fixed*
      quarter turn, not ``theta``. Proof sketch: define O_r's Y axis (in
      mounted-frame coordinates) as the face normal itself and Z as the
      shared rotation axis; X completes a right-handed basis via
      ``cross(Y, Z)``. Composing that per-face basis change with
      ``rotation_z(theta)`` gives, for *any* in-plane normal (n_x, n_y, 0),
      exactly ``rotation_z(pi/2)`` -- the theta-dependence cancels
      algebraically (verified numerically against several normals while
      this was debugged: a naive ``rotation_z(theta)`` here put the
      *in-plane* face extent onto O_r's Y and the near-constant height onto
      O_r's X -- backwards, and the tell was a regridded height map with a
      ~100mm "height" spread instead of ~66um of quantization noise).
    """
    t_mount, part_height = compute_t_mount(mesh)
    theta = compute_theta_face(normal)
    stage_vertical_offset = 0.5 * part_height  # center the part in the line-axis FOV
    stage_origin_os = np.array([-sensor_config.working_distance_mm, 0.0, stage_vertical_offset])
    pose = t_mount.then(Transform(rotation_z(theta), stage_origin_os))
    o_s_from_o_r = Transform(rotation_z(math.pi / 2.0), stage_origin_os)
    return pose, t_mount, theta, o_s_from_o_r.invert()


def load_part(
    stl_path: str | Path,
    sensor_config: SensorConfig,
    face_normal_hint: np.ndarray = (1.0, 0.0, 0.0),
    face_normal_tol_deg: float = 1.0,
    up_axis: np.ndarray = (0.0, 0.0, 1.0),
) -> Part:
    """Ingest an STL, select the face whose outward normal best matches
    ``face_normal_hint`` (evaluated in the internal, post-remap frame), and
    pose it into O_s.

    ``up_axis`` (V1.2, default +Z -- CNC/CAM convention) is the part file's
    own vertical direction, remapped once at ingestion onto this package's
    internal rotation axis; override for files authored with a different
    native up-axis (the CLI's ``--up-axis X Y Z``).
    """
    mesh = load_mesh(stl_path)
    mesh = _remap_up_axis(mesh, np.asarray(up_axis, dtype=float))

    facet_idx = _select_facet(
        mesh, np.asarray(face_normal_hint), rotation_axis=ROTATION_AXIS_OS, max_angle_deg=face_normal_tol_deg
    )
    face_frame = build_face_frame(mesh, facet_idx)

    pose, t_mount, theta, o_r_from_o_s = compute_pose(mesh, face_frame.normal, sensor_config)

    face_indices = mesh.facets[facet_idx]
    face_mesh = mesh.submesh([face_indices], append=True)
    face_mesh.vertices = pose.apply_points(face_mesh.vertices)

    return Part(
        mesh=face_mesh,
        face_frame=face_frame,
        T_mount=t_mount,
        theta_face_rad=theta,
        pose=pose,
        o_r_from_o_s=o_r_from_o_s,
    )
