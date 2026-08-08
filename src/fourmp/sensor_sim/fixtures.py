"""Test-part generation.

The V1 test part is a cube, "provided as an STL file" per the spec -- no
such file exists yet (4MP's companion part schema/sample parts don't exist
yet either), so this module generates one deterministically rather than
requiring a hand-supplied asset.

Size: 100mm per side. Comfortably inside the 188mm x 319mm measurement area
(leaving margin on both axes), and small relative to the 470mm working
distance, while still being large enough to sample with real coverage.
A cube is symmetric under axis relabeling, so no special reorientation is
needed to satisfy part.py's default "+Z is up" convention (V1.2) -- it
holds by construction. That symmetry is also why a non-cube part (a tapered
frustum) was what actually surfaced the V1.2 up-axis bug in the first
place -- see the dedicated up-axis-remap tests in test_part.py for coverage
this fixture's symmetry can't provide.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh

DEFAULT_CUBE_SIDE_MM = 100.0


def make_test_cube(side_mm: float = DEFAULT_CUBE_SIDE_MM) -> trimesh.Trimesh:
    """An axis-aligned cube centered at the part-frame origin."""
    return trimesh.creation.box(extents=(side_mm, side_mm, side_mm))


def write_test_cube(path: str | Path, side_mm: float = DEFAULT_CUBE_SIDE_MM) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_test_cube(side_mm).export(str(path))
    return path


def _default_out_path() -> Path:
    return Path(__file__).resolve().parents[3] / "tests" / "data" / "cube_100mm.stl"


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate a cube STL test/example part.")
    parser.add_argument(
        "--out", type=Path, default=_default_out_path(), help="output STL path (default: %(default)s)"
    )
    parser.add_argument(
        "--side-mm", type=float, default=DEFAULT_CUBE_SIDE_MM, help="cube side length in mm"
    )
    args = parser.parse_args()
    out = write_test_cube(args.out, side_mm=args.side_mm)
    print(f"wrote {out}")


if __name__ == "__main__":
    _main()
