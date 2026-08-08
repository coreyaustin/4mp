import json

import numpy as np
import pytest
import trimesh

from fourmp.cli import main


@pytest.fixture
def cube_stl(tmp_path):
    from fourmp.sensor_sim.fixtures import write_test_cube

    return write_test_cube(tmp_path / "input" / "cube_100mm.stl")


def _write_unscannable_stl(path):
    """A single flat quad whose only normal is (0, 1, 0) -- the "top face"
    direction, which face selection always rejects (out of scope, can't be
    rotated into view by the single-axis stage). Used to exercise batch
    mode's per-part failure handling deterministically, without relying on
    scan geometry happening to miss."""
    mesh = trimesh.Trimesh(
        vertices=[[-10, 0, -10], [10, 0, -10], [10, 0, 10], [-10, 0, 10]],
        faces=[[0, 1, 2], [0, 2, 3]],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    return path


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
    assert "o_r_grid" in report
    assert report["o_r_grid"]["resolution_mm"] == pytest.approx(0.066, abs=0.01)

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


def test_cli_batch_flag_scans_input_dir(tmp_path, cube_stl, capsys):
    from fourmp.sensor_sim.fixtures import write_test_cube

    write_test_cube(cube_stl.parent / "cube_80mm.stl", side_mm=80.0)
    output_dir = tmp_path / "output"

    exit_code = main(
        [
            "--batch",
            "--input-dir",
            str(cube_stl.parent),
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
    assert (output_dir / "cube_80mm" / "report.json").exists()

    out = capsys.readouterr().out
    assert "batch complete: 2 succeeded, 0 failed" in out


def test_cli_directory_argument_batches_with_nested_output_folder(tmp_path, capsys):
    from fourmp.sensor_sim.fixtures import write_test_cube

    parts_dir = tmp_path / "some_folder_of_parts"
    write_test_cube(parts_dir / "cube_a.stl", side_mm=90.0)
    write_test_cube(parts_dir / "cube_b.stl", side_mm=110.0)

    # No --output-dir given -- output should land inside parts_dir itself.
    exit_code = main([str(parts_dir), "--step-stride", "16", "--mirror-stride", "16"])
    assert exit_code == 0

    assert (parts_dir / "output" / "cube_a" / "report.json").exists()
    assert (parts_dir / "output" / "cube_b" / "report.json").exists()

    out = capsys.readouterr().out
    assert f"{parts_dir}\\output" in out or f"{parts_dir}/output" in out


def test_cli_batch_continues_after_one_part_fails(tmp_path, cube_stl, capsys):
    _write_unscannable_stl(cube_stl.parent / "bad_part.stl")
    output_dir = tmp_path / "output"

    exit_code = main(
        [
            "--batch",
            "--input-dir",
            str(cube_stl.parent),
            "--output-dir",
            str(output_dir),
            "--step-stride",
            "16",
            "--mirror-stride",
            "16",
        ]
    )
    # One part failed -> nonzero exit, but the good part should still have
    # been processed rather than the whole batch aborting.
    assert exit_code == 1
    assert (output_dir / "cube_100mm" / "report.json").exists()
    assert not (output_dir / "bad_part").exists()

    captured = capsys.readouterr()
    assert "batch complete: 1 succeeded, 1 failed" in captured.out
    assert "bad_part" in (captured.out + captured.err)


def test_cli_batch_with_empty_input_dir_reports_error(tmp_path, capsys):
    empty_dir = tmp_path / "empty_input"
    empty_dir.mkdir()
    exit_code = main(["--batch", "--input-dir", str(empty_dir)])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "no STL files found" in err


def test_cli_batch_rejects_part_name_alongside_flag(tmp_path, cube_stl, capsys):
    with pytest.raises(SystemExit):
        main(["--batch", "cube_100mm", "--input-dir", str(cube_stl.parent)])
