"""Confirmed V1 sensor parameters, and the calibration (projector + camera
pinhole models) derived from them.

Values below come directly from the "Confirmed parameters" table in
sensor-sim-v1-spec.md and the "Real hardware constraints" section of
architecture-decisions.md. Two things are *derived* rather than given
outright, per the 2026-08-07 projector-model correction:

- Baseline (projector/camera separation): the working distance and
  triangulation half-angle alone determine it, given the confirmed geometry
  (illumination and camera sit symmetrically about the boresight, converging
  on the reference point at the working distance):
      baseline = 2 * working_distance * tan(half_angle)
- Projector focal length: chosen so the DMD's physical half-extents (mirror
  count/2 * pixel pitch), viewed through the projector's own tilted axis at
  its own (oblique) distance to the reference point, subtend a half-angle
  that fills the confirmed 188mm x 319mm measurement area. Both axes
  independently land on ~24.2mm for a single non-anamorphic lens -- a
  self-consistency check, not a free parameter.

Genuinely open/non-blocking items (per spec's "Known open parameters"): the
camera's FOV margin over the measurement area, and the vertical placement of
the rotary stage relative to the sensor's optical axis. Both are given
explicit, documented placeholder values here (``CAMERA_FOV_MARGIN``,
``stage_vertical_offset_mm`` in part.py) rather than left as magic numbers,
since the spec says not to gate V1 on pinning them down exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fourmp.sensor_sim.pinhole import PinholeModel

# ---- Confirmed parameters (sensor-sim-v1-spec.md) --------------------------

WORKING_DISTANCE_MM = 470.0
HALF_ANGLE_DEG = 27.0
MEASUREMENT_AREA_SHORT_MM = 188.0  # baseline / scan-step axis (X)
MEASUREMENT_AREA_LONG_MM = 319.0  # line axis (Y)

DMD_SCAN_STEPS = 1600  # short-axis mirror count == number of scan steps
DMD_LINE_MIRRORS = 2716  # long-axis mirror count, fired simultaneously per step
DMD_PIXEL_PITCH_MM = 5.4e-3  # 5.4 micron

CAMERA_BASELINE_AXIS_PX = 3104  # short/188mm axis (matches DMD_SCAN_STEPS axis)
CAMERA_LINE_AXIS_PX = 5328  # long/319mm axis (matches DMD_LINE_MIRRORS axis)

# Non-blocking placeholder (spec: "Exact camera FOV margin ... doesn't block
# V1"). 10% linear margin over the measurement area half-extents.
CAMERA_FOV_MARGIN = 1.10


@dataclass(frozen=True)
class SensorConfig:
    """Confirmed sensor parameters plus the derived projector/camera calibration.

    ``projector`` and ``camera`` are the *shared* calibration referenced
    throughout the spec ("the forward and inverse models must share the
    identical projector-ray model and camera calibration") -- the measurement
    engine and reconstruction engine should each be handed the same
    SensorConfig instance rather than constructing their own.
    """

    working_distance_mm: float
    half_angle_deg: float
    baseline_mm: float
    reference_point: np.ndarray  # (3,), sensor frame: (0, 0, working_distance_mm)
    projector: PinholeModel
    camera: PinholeModel

    @property
    def half_angle_rad(self) -> float:
        return math.radians(self.half_angle_deg)

    def height_from_point(self, point: np.ndarray) -> np.ndarray:
        """Signed deviation from the reference plane: height = Z - working_distance.

        The reference plane is perpendicular to the boresight (sensor-frame Z)
        at the working distance; Z is the same boresight coordinate for any
        point already expressed in sensor frame, so this is a direct
        subtraction, not a per-point projection.
        """
        point = np.asarray(point, dtype=float)
        return point[..., 2] - self.working_distance_mm


def build_sensor_config(
    working_distance_mm: float = WORKING_DISTANCE_MM,
    half_angle_deg: float = HALF_ANGLE_DEG,
    measurement_area_short_mm: float = MEASUREMENT_AREA_SHORT_MM,
    measurement_area_long_mm: float = MEASUREMENT_AREA_LONG_MM,
    dmd_scan_steps: int = DMD_SCAN_STEPS,
    dmd_line_mirrors: int = DMD_LINE_MIRRORS,
    dmd_pixel_pitch_mm: float = DMD_PIXEL_PITCH_MM,
    camera_baseline_axis_px: int = CAMERA_BASELINE_AXIS_PX,
    camera_line_axis_px: int = CAMERA_LINE_AXIS_PX,
    camera_fov_margin: float = CAMERA_FOV_MARGIN,
) -> SensorConfig:
    """Build a SensorConfig from confirmed parameters (all overridable, e.g.
    for fast, reduced-resolution test configurations)."""
    half_angle_rad = math.radians(half_angle_deg)
    baseline_mm = 2.0 * working_distance_mm * math.tan(half_angle_rad)
    half_baseline_mm = baseline_mm / 2.0

    projector_center = np.array([-half_baseline_mm, 0.0, 0.0])
    camera_center = np.array([half_baseline_mm, 0.0, 0.0])
    reference_point = np.array([0.0, 0.0, working_distance_mm])

    distance_to_reference = math.hypot(half_baseline_mm, working_distance_mm)

    # Projector focal length: fills the measurement area at the reference
    # point, measured perpendicular to the projector's own (tilted) axis.
    half_fov_short = math.atan((measurement_area_short_mm / 2.0) / distance_to_reference)
    half_fov_long = math.atan((measurement_area_long_mm / 2.0) / distance_to_reference)
    f_short_mm = (dmd_scan_steps / 2.0 * dmd_pixel_pitch_mm) / math.tan(half_fov_short)
    f_long_mm = (dmd_line_mirrors / 2.0 * dmd_pixel_pitch_mm) / math.tan(half_fov_long)
    # Single (non-anamorphic) lens: average the two independent estimates.
    f_projector_mm = 0.5 * (f_short_mm + f_long_mm)
    f_projector_px = f_projector_mm / dmd_pixel_pitch_mm

    projector = PinholeModel.looking_at(
        center=projector_center,
        target=reference_point,
        f_i=f_projector_px,
        f_j=f_projector_px,
        n_i=dmd_scan_steps,
        n_j=dmd_line_mirrors,
    )

    # Camera focal length (in pixel units directly -- physical camera pixel
    # pitch isn't a confirmed parameter, but only the focal-length/pitch
    # *ratio* is needed for projection, so it's never separately required).
    half_fov_short_cam = math.atan(
        (measurement_area_short_mm / 2.0 * camera_fov_margin) / distance_to_reference
    )
    half_fov_long_cam = math.atan(
        (measurement_area_long_mm / 2.0 * camera_fov_margin) / distance_to_reference
    )
    f_short_px = (camera_baseline_axis_px / 2.0) / math.tan(half_fov_short_cam)
    f_long_px = (camera_line_axis_px / 2.0) / math.tan(half_fov_long_cam)

    camera = PinholeModel.looking_at(
        center=camera_center,
        target=reference_point,
        f_i=f_short_px,
        f_j=f_long_px,
        n_i=camera_baseline_axis_px,
        n_j=camera_line_axis_px,
    )

    return SensorConfig(
        working_distance_mm=working_distance_mm,
        half_angle_deg=half_angle_deg,
        baseline_mm=baseline_mm,
        reference_point=reference_point,
        projector=projector,
        camera=camera,
    )
