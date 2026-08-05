"""Forward and reverse simulation pipelines, tying the hardware models together.

Mirrors the architecture doc's section 4 sketch: forward maps source -> SUT ->
camera; reverse maps camera -> SUT -> source. The scan is a swept single-row
line (architecture doc open question 2 resolved as line-scan, not a
multiplexed/fringe code), so the reverse direction never needs to decode
anything — each shot's row index is known directly, and every lit camera
pixel triangulates against that row's known light plane (ray/plane
intersection). That's a deterministic replacement for the doc's
`while not converged` sketch: there's no iterative fit to converge, so
`validation_loop` runs the scan once and reconstructs directly.
"""
from __future__ import annotations

import numpy as np

from .camera import Camera
from .centroid import centroid_rows
from .geometry import ray_plane_intersection
from .patterns import SourcePattern
from .source import Source
from .sut import SUT


def forward_pipeline(
    source: Source,
    sut: SUT,
    table_angle_rad: float,
    camera: Camera,
    pattern: SourcePattern | None = None,
) -> np.ndarray:
    """Source -> SUT -> Camera. Predicts the frame the camera would capture."""
    active_source = source if pattern is None else Source(
        pose=source.pose,
        resolution=source.resolution,
        fov_deg=source.fov_deg,
        pattern=pattern,
        working_distance_mm=source.working_distance_mm,
        magnification=source.magnification,
        focus_spot_diameter_mm=source.focus_spot_diameter_mm,
        defocus_half_angle_deg=source.defocus_half_angle_deg,
    )
    sut_world = sut.transform(table_angle_rad)
    hits = active_source.project(sut_world)
    return camera.capture(hits, sut_world, source.pose.position)


def acquire_line_scan(
    source: Source,
    sut: SUT,
    table_angle_rad: float,
    camera: Camera,
    rows=None,
    line_width_rows: int = 1,
) -> list[tuple[int, np.ndarray]]:
    """Sweep single-row patterns across the source, capturing a frame per line position."""
    rows = range(source.resolution[0]) if rows is None else rows
    return [
        (
            row,
            forward_pipeline(
                source,
                sut,
                table_angle_rad,
                camera,
                pattern=SourcePattern.single_row(source.resolution, row, line_width_rows),
            ),
        )
        for row in rows
    ]


def reverse_pipeline(
    line_frames: list[tuple[int, np.ndarray]],
    source: Source,
    camera: Camera,
    intensity_threshold: float = 1e-3,
) -> np.ndarray:
    """Camera -> Source. Triangulate a swept line scan into 3D points.

    No decoding step: each frame already tells us which row lit it. Instead of
    triangulating every individual lit pixel, each column's sub-pixel line
    center (its intensity-weighted centroid row) triangulates against that
    row's plane — one point per column per shot, at sub-pixel accuracy.
    """
    world_points = []
    for row, frame in line_frames:
        plane = source.row_plane(row)
        if plane is None:
            continue
        centroids = centroid_rows(frame, intensity_threshold)
        for col, centroid_row in enumerate(centroids):
            if np.isnan(centroid_row):
                continue
            camera_ray_dir = camera.pixel_direction_world(centroid_row, col)
            point = ray_plane_intersection(camera.pose.position, camera_ray_dir, *plane)
            if point is not None:
                world_points.append(point)
    return np.array(world_points)


def validation_loop(
    source: Source,
    sut_true: SUT,
    sut_init: SUT,
    table_angle_rad: float,
    camera: Camera,
    rows=None,
    line_width_rows: int = 1,
):
    """Scan `sut_true`, reconstruct a SUT from the scan, and check it against a fresh render.

    Mirrors architecture doc section 3: (1)-(2) reverse-solve the surface from
    a captured line scan (a direct triangulation, not an iterative fit),
    (3) forward-render the reconstructed SUT, (4) compare that render to a
    fully-lit reference capture of the true SUT.
    """
    line_frames = acquire_line_scan(
        source, sut_true, table_angle_rad, camera, rows=rows, line_width_rows=line_width_rows
    )
    world_points = reverse_pipeline(line_frames, source, camera)

    sut_world = sut_init.transform(table_angle_rad)
    reconstructed_sut = sut_world.corrected_with(world_points) if world_points.size else sut_init

    all_on = SourcePattern.all_on(source.resolution)
    reference_frame = forward_pipeline(source, sut_true, table_angle_rad, camera, pattern=all_on)
    predicted_frame = forward_pipeline(source, reconstructed_sut, table_angle_rad, camera, pattern=all_on)
    frame_rmse = float(np.sqrt(np.mean((predicted_frame - reference_frame) ** 2)))

    return reconstructed_sut, predicted_frame, reference_frame, frame_rmse
