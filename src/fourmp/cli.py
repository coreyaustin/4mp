"""``4mp`` CLI: scan one or more parts through the V1 sensor simulation pipeline.

Three ways to invoke it (see the module docstring in ``fourmp.sensor_sim.part``
for what "scan" means):

- ``4mp <name>`` -- scan one part, resolved against ``--input-dir`` (default
  ``input/``) or given as a direct path.
- ``4mp --batch`` -- scan every STL in ``--input-dir``, writing each part's
  report under ``--output-dir/<part-name>/`` (default ``output/``), same as
  the single-part layout.
- ``4mp <directory>`` -- scan every STL in the given directory. Output goes
  to a new ``output/`` subfolder created *inside that directory*, not the
  top-level ``--output-dir`` -- this is for scanning an arbitrary folder of
  parts elsewhere on disk, not just the project's own ``input/``.

More subcommands/options are expected to land here as the pipeline grows
(multi-face/multi-pose, occlusion, etc. -- see architecture-decisions.md's
roadmap); this is intentionally a flat argparse command for now rather than
a subcommand tree.
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
from fourmp.sensor_sim.sensor_config import SensorConfig, build_sensor_config
from fourmp.sensor_sim.validation import pointwise_residual, spectral_residual

DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"
BATCH_OUTPUT_SUBDIR = "output"  # subfolder name created inside a batch-scanned directory


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


def _find_stl_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".stl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="4mp", description="Scan STL part(s) through the V1 sensor simulation pipeline."
    )
    parser.add_argument(
        "part",
        nargs="?",
        default=None,
        help=(
            "part to scan: a bare name looked up in --input-dir "
            f"(default {DEFAULT_INPUT_DIR}/), a path to an STL file, or a path to a "
            "directory to batch-scan (writes an 'output/' subfolder inside it). "
            "Omit together with --batch to batch-scan --input-dir instead."
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "batch-scan every STL in --input-dir, writing each part's report to "
            "--output-dir/<part-name>/ (same layout as scanning one part)"
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help=f"directory to resolve a bare part name against, or to --batch (default: {DEFAULT_INPUT_DIR})",
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


def _scan_one(
    stl_path: Path,
    output_dir: Path,
    sensor_config: SensorConfig,
    face_normal_hint: tuple[float, float, float],
    step_stride: int,
    mirror_stride: int,
) -> bool:
    """Run the full pipeline for one STL and write its report under
    ``output_dir/<stl_path.stem>/``. Returns True on success."""
    print(f"loading {stl_path} ...")
    try:
        part = load_part(stl_path, sensor_config, face_normal_hint=face_normal_hint)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return False
    print(f"  face normal (part frame): {part.face_frame.normal}")
    print(f"  theta_face: {math.degrees(part.theta_face_rad):.3f} deg")

    print("running measurement engine (forward model) ...")
    scan = run_measurement(part.mesh, sensor_config, step_stride=step_stride, mirror_stride=mirror_stride)
    print(f"  {len(scan)} hits")

    if len(scan) == 0:
        print(
            "error: measurement produced no hits -- part is likely out of the sensor's range",
            file=sys.stderr,
        )
        return False

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
        output_dir,
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
    return True


def _run_batch(
    batch_dir: Path,
    output_dir: Path,
    sensor_config: SensorConfig,
    face_normal_hint: tuple[float, float, float],
    step_stride: int,
    mirror_stride: int,
) -> int:
    if not batch_dir.is_dir():
        print(f"error: no such directory: {batch_dir}", file=sys.stderr)
        return 1

    stl_paths = _find_stl_files(batch_dir)
    if not stl_paths:
        print(f"error: no STL files found in {batch_dir}", file=sys.stderr)
        return 1

    print(f"batch-scanning {len(stl_paths)} part(s) from {batch_dir} -> {output_dir}")
    succeeded: list[str] = []
    failed: list[str] = []
    for i, stl_path in enumerate(stl_paths, start=1):
        print(f"\n[{i}/{len(stl_paths)}] {stl_path.name}")
        try:
            ok = _scan_one(stl_path, output_dir, sensor_config, face_normal_hint, step_stride, mirror_stride)
        except Exception as exc:  # one bad part shouldn't abort the whole batch
            print(f"error: unhandled exception scanning {stl_path.name}: {exc}", file=sys.stderr)
            ok = False
        (succeeded if ok else failed).append(stl_path.name)

    print(f"\nbatch complete: {len(succeeded)} succeeded, {len(failed)} failed")
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
    return 0 if not failed else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    face_normal_hint = tuple(args.face_normal)

    sensor_config = build_sensor_config()

    part_is_dir = args.part is not None and Path(args.part).is_dir()

    if args.batch and args.part is not None and not part_is_dir:
        parser.error("--batch doesn't take a part name; omit it to batch --input-dir, or pass a directory")

    if part_is_dir:
        batch_dir = Path(args.part)
        return _run_batch(
            batch_dir,
            batch_dir / BATCH_OUTPUT_SUBDIR,
            sensor_config,
            face_normal_hint,
            args.step_stride,
            args.mirror_stride,
        )

    if args.batch:
        return _run_batch(
            args.input_dir,
            args.output_dir,
            sensor_config,
            face_normal_hint,
            args.step_stride,
            args.mirror_stride,
        )

    if args.part is None:
        parser.error("the following arguments are required: part (or use --batch)")

    stl_path = _resolve_stl_path(args.part, args.input_dir)
    if not stl_path.exists():
        print(f"error: no such STL file: {stl_path}", file=sys.stderr)
        return 1

    ok = _scan_one(
        stl_path, args.output_dir, sensor_config, face_normal_hint, args.step_stride, args.mirror_stride
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
