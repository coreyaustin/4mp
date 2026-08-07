"""``4mp`` CLI: scan a part through the V1 sensor simulation pipeline.

V1: one positional argument (the part to scan, resolved against
``input/`` by default -- see the module docstring in
``fourmp.sensor_sim.part`` for what "scan" means). More subcommands/options
are expected to land here as the pipeline grows (multi-face/multi-pose,
occlusion, etc. -- see architecture-decisions.md's roadmap); this is
intentionally a flat argparse command for now rather than a subcommand tree,
since there's only one thing to do yet.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from fourmp.sensor_sim.measurement import run_measurement
from fourmp.sensor_sim.part import load_part
from fourmp.sensor_sim.reconstruction import run_reconstruction
from fourmp.sensor_sim.regrid import regrid_reconstruction_and_truth
from fourmp.sensor_sim.report import save_report
from fourmp.sensor_sim.sensor_config import build_sensor_config
from fourmp.sensor_sim.validation import pointwise_residual, spectral_residual

DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"


def _resolve_stl_path(part: str, input_dir: Path) -> Path:
    """``part`` may be a bare name (looked up in ``input_dir``, ``.stl``
    appended if missing) or a path (used as-is, relative or absolute)."""
    candidate = Path(part)
    if candidate.suffix.lower() == ".stl" and candidate.exists():
        return candidate
    if candidate.exists() and candidate.is_file():
        return candidate

    name = candidate.name if candidate.suffix.lower() == ".stl" else candidate.name + ".stl"
    resolved = input_dir / name
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="4mp", description="Scan an STL part through the V1 sensor simulation pipeline."
    )
    parser.add_argument(
        "part",
        help=f"part to scan: a bare name looked up in --input-dir (default {DEFAULT_INPUT_DIR}/), "
        "or a path to an STL file",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help=f"directory to resolve a bare part name against (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"directory to write the report/height maps/plots into (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--face-normal",
        type=float,
        nargs=3,
        default=(1.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="outward normal (part frame) of the face to select and scan (default: 1 0 0)",
    )
    parser.add_argument(
        "--step-stride",
        type=int,
        default=1,
        help="subsample the projector's scan-step axis by this factor (default: 1, full resolution)",
    )
    parser.add_argument(
        "--mirror-stride",
        type=int,
        default=1,
        help="subsample the projector's along-line axis by this factor (default: 1, full resolution)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    stl_path = _resolve_stl_path(args.part, args.input_dir)
    if not stl_path.exists():
        print(f"error: no such STL file: {stl_path}", file=sys.stderr)
        return 1

    sensor_config = build_sensor_config()

    print(f"loading {stl_path} ...")
    part = load_part(stl_path, sensor_config, face_normal_hint=tuple(args.face_normal))
    print(f"  face normal (part frame): {part.face_frame.normal}")
    print(f"  theta_face: {math.degrees(part.theta_face_rad):.3f} deg")

    print("running measurement engine (forward model) ...")
    scan = run_measurement(
        part.mesh, sensor_config, step_stride=args.step_stride, mirror_stride=args.mirror_stride
    )
    print(f"  {len(scan)} hits")

    if len(scan) == 0:
        print("error: measurement produced no hits -- part is likely out of the sensor's range", file=sys.stderr)
        return 1

    print("running reconstruction engine (inverse model) ...")
    result = run_reconstruction(scan, sensor_config)
    n_reconstructed = int((~np.isnan(result.height_map)).sum())
    print(f"  {n_reconstructed} pixels reconstructed (camera-native), {result.collisions} pixel collisions")

    print("regridding onto a physical XY grid (mm) + sampling ground truth on that grid ...")
    grid, recon_grid, truth_grid = regrid_reconstruction_and_truth(result, part.mesh, sensor_config)
    print(f"  grid: {grid.shape[1]}x{grid.shape[0]} cells at {grid.resolution_mm * 1000:.1f}um/cell")

    print("computing validation metrics ...")
    pointwise = pointwise_residual(recon_grid, truth_grid)
    spectral = spectral_residual(recon_grid, truth_grid)
    print(f"  pointwise RMS: {pointwise.rms_mm:.6f} mm, max: {pointwise.max_mm:.6f} mm")
    print(
        f"  spectral: peak/mean power ratio {spectral.peak_to_mean_power_ratio:.2f}, "
        f"low-freq fraction {spectral.low_frequency_power_fraction:.3f}"
    )

    run_dir = save_report(
        args.output_dir,
        stl_path.stem,
        stl_path,
        part,
        scan,
        result,
        grid,
        recon_grid,
        truth_grid,
        pointwise,
        spectral,
        sensor_config,
    )
    print(f"wrote report to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
