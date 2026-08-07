import json

import numpy as np
import pytest

from fourmp.cli import main


@pytest.fixture
def cube_stl(tmp_path):
    from fourmp.sensor_sim.fixtures import write_test_cube

    return write_test_cube(tmp_path / "input" / "cube_100mm.stl")


def test_cli_end_to_end(tmp_path, cube_stl, capsys):
    output_dir = tmp_path / "output"
    exit_code = main(
        [
            "cube_100mm",
            "--input-dir",
            str(cube_stl.parent),
            "--output-dir",
            str(output_dir),
            "--step-stride",
            "8",
            "--mirror-stride",
            "8",
        ]
    )
    assert exit_code == 0

    run_dir = output_dir / "cube_100mm"
    assert (run_dir / "report.json").exists()
    assert (run_dir / "height_map.npy").exists()
    assert (run_dir / "height_map.png").exists()
    assert (run_dir / "ground_truth.png").exists()
    assert (run_dir / "residual.png").exists()
    assert (run_dir / "triangulation_gap_map.png").exists()

    height_map = np.load(run_dir / "height_map.npy")
    valid = ~np.isnan(height_map)
    assert valid.any()
    # V1.1: sub-pixel reconstruction + physical-grid regridding should put
    # this at floating-point noise, not sub-pixel quantization error.
    assert np.abs(height_map[valid].mean() - 50.0) < 1e-3

    report = json.loads((run_dir / "report.json").read_text())
    assert "physical_grid" in report
    assert report["physical_grid"]["resolution_mm"] == pytest.approx(0.066, abs=0.01)

    out = capsys.readouterr().out
    assert "pointwise RMS" in out


def test_cli_accepts_direct_stl_path(tmp_path, cube_stl, capsys):
    output_dir = tmp_path / "output"
    exit_code = main(
        [
            str(cube_stl),
            "--output-dir",
            str(output_dir),
            "--step-stride",
            "16",
            "--mirror-stride",
            "16",
        ]
    )
    assert exit_code == 0
    assert (output_dir / "cube_100mm" / "report.json").exists()


def test_cli_missing_part_reports_error(tmp_path, capsys):
    exit_code = main(["does_not_exist", "--input-dir", str(tmp_path)])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "no such STL file" in err
