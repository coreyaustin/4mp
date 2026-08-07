"""Per-run report artifacts for the CLI: height-map/residual plots, raw
arrays, and a JSON metrics summary, all written under one output directory.

Not part of the measurement/reconstruction/validation pipeline itself --
this module just packages already-computed results into files a person can
open, per the "output directory... reports, height maps, and any plots"
request.

**V1.1:** the primary height_map/ground_truth/residual outputs are now the
physical-XY-grid arrays from regrid.py (mm-native axes, no camera-pixel
keystone artifact) rather than the camera-pixel-native ``HeightMapResult``
arrays. The camera-pixel-native triangulation-gap map is still saved
separately, since that diagnostic is naturally per-camera-pixel (per
sensor-sim-v1-spec.md's V1.1 note to keep it available).
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe; the CLI never needs an interactive backend
import matplotlib.pyplot as plt
import numpy as np

from fourmp.sensor_sim.measurement import ScanData
from fourmp.sensor_sim.part import Part
from fourmp.sensor_sim.reconstruction import HeightMapResult
from fourmp.sensor_sim.regrid import GridSpec
from fourmp.sensor_sim.sensor_config import SensorConfig
from fourmp.sensor_sim.validation import PointwiseResidual, SpectralResidual


def _valid_bbox(mask: np.ndarray) -> tuple[slice, slice]:
    rows, cols = np.nonzero(mask)
    return slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)


def _cropped_extent_mm(grid: GridSpec, row_slice: slice, col_slice: slice) -> tuple[float, float, float, float]:
    res = grid.resolution_mm
    return (
        grid.x_min + col_slice.start * res,
        grid.x_min + col_slice.stop * res,
        grid.y_min + row_slice.start * res,
        grid.y_min + row_slice.stop * res,
    )


def _plot_grid_field(
    data: np.ndarray,
    grid: GridSpec,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    center_zero: bool = False,
) -> None:
    """Plot a physical-XY-grid field (mm axes, per regrid.GridSpec)."""
    valid = ~np.isnan(data)
    if not valid.any():
        return
    row_slice, col_slice = _valid_bbox(valid)
    cropped = np.ma.masked_invalid(data[row_slice, col_slice])
    extent = _cropped_extent_mm(grid, row_slice, col_slice)

    kwargs: dict[str, Any] = {}
    if center_zero:
        limit = float(np.abs(cropped).max())
        kwargs = {"vmin": -limit, "vmax": limit}

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cropped, cmap=cmap, origin="lower", aspect="equal", extent=extent, **kwargs)
    ax.set_xlabel("X, baseline axis (mm)")
    ax.set_ylabel("Y, line axis (mm)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="height (mm)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pixel_native_field(data: np.ndarray, title: str, out_path: Path, cmap: str = "magma") -> None:
    """Plot a camera-pixel-native field (pixel-index axes) -- used only for
    the triangulation-gap diagnostic, which is naturally per-camera-pixel."""
    valid = ~np.isnan(data)
    if not valid.any():
        return
    row_slice, col_slice = _valid_bbox(valid)
    cropped = np.ma.masked_invalid(data[row_slice, col_slice])

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(cropped, cmap=cmap, origin="lower", aspect="equal")
    ax.set_xlabel("line-axis camera pixel (cropped)")
    ax.set_ylabel("baseline-axis camera pixel (cropped)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="triangulation gap (mm)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def save_report(
    output_dir: str | Path,
    part_name: str,
    stl_path: str | Path,
    part: Part,
    scan: ScanData,
    result: HeightMapResult,
    grid: GridSpec,
    recon_grid: np.ndarray,
    truth_grid: np.ndarray,
    pointwise: PointwiseResidual,
    spectral: SpectralResidual,
    sensor_config: SensorConfig,
) -> Path:
    """Write height_map/ground_truth/residual (physical XY grid, mm) +
    camera-pixel-native gap-map diagnostic + report.json into
    ``output_dir/part_name/``. Returns that directory."""
    run_dir = Path(output_dir) / part_name
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "height_map.npy", recon_grid)
    np.save(run_dir / "ground_truth.npy", truth_grid)

    residual = recon_grid - truth_grid

    _plot_grid_field(recon_grid, grid, f"{part_name}: reconstructed height", run_dir / "height_map.png")
    _plot_grid_field(truth_grid, grid, f"{part_name}: ground truth height", run_dir / "ground_truth.png")
    _plot_grid_field(
        residual,
        grid,
        f"{part_name}: residual (reconstructed - truth)",
        run_dir / "residual.png",
        cmap="RdBu_r",
        center_zero=True,
    )
    _plot_pixel_native_field(
        result.triangulation_gap_mm,
        f"{part_name}: triangulation gap (camera-pixel-native)",
        run_dir / "triangulation_gap_map.png",
    )

    n_reconstructed_grid = int(np.sum(~np.isnan(recon_grid)))
    n_reconstructed_pixels = int(np.sum(~np.isnan(result.height_map)))
    report = {
        "part_name": part_name,
        "stl_path": str(stl_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "face_frame": {
            "normal_part_frame": _jsonable(part.face_frame.normal),
            "centroid_part_frame": _jsonable(part.face_frame.centroid),
        },
        "theta_face_deg": float(np.degrees(part.theta_face_rad)),
        "sensor_config": {
            "working_distance_mm": sensor_config.working_distance_mm,
            "half_angle_deg": sensor_config.half_angle_deg,
            "baseline_mm": sensor_config.baseline_mm,
        },
        "scan": {
            "n_hits": len(scan),
            "n_scan_steps": int(sensor_config.projector.n_i),
            "n_mirrors_per_line": int(sensor_config.projector.n_j),
        },
        "reconstruction": {
            "n_reconstructed_pixels_camera_native": n_reconstructed_pixels,
            "collisions_camera_native": result.collisions,
            "max_triangulation_gap_mm": float(np.nanmax(result.triangulation_gap_mm))
            if n_reconstructed_pixels
            else None,
        },
        "physical_grid": {
            "resolution_mm": grid.resolution_mm,
            "shape": list(grid.shape),
            "extent_mm": list(grid.extent_mm),
            "n_reconstructed_cells": n_reconstructed_grid,
        },
        "validation": {
            "pointwise": _jsonable(pointwise),
            "spectral": _jsonable(spectral),
        },
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    return run_dir
