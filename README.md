# fourmp

4MP's simulation of its DMD-based line-scan triangulation sensor: a
measurement engine (forward model: posed part mesh -> simulated scan) and a
reconstruction engine (inverse model: scan -> height map), sharing one
sensor calibration so the two stay consistent by construction.

See [`sensor-sim-v1-spec.md`](sensor-sim-v1-spec.md) for the V1 build spec
and [`architecture-decisions.md`](architecture-decisions.md) for the full
design history, including how a few things the spec left implicit (baseline
geometry, projector focal length, stage placement) were resolved.

## Layout

- `src/fourmp/sensor_sim/` -- the V1 pipeline (see module docstrings for the
  sensor-frame convention and per-module design notes):
  - `sensor_config.py` -- confirmed sensor parameters + derived projector/
    camera calibration.
  - `part.py` -- STL ingestion, face-frame computation, pose math.
  - `measurement.py` -- forward model.
  - `reconstruction.py` -- inverse model.
  - `validation.py` -- ground-truth sampling + pointwise/spectral metrics.
  - `report.py` -- plots + JSON report writer for the CLI.
- `src/fourmp/cli.py` -- the `4mp` command.
- `input/` -- put the STL parts you want to scan here (not tracked in git --
  see `input/README.md`).
- `output/` -- per-run reports/height maps/plots land here, one subdirectory
  per part (not tracked in git -- see `output/README.md`).
- `tests/sensor_sim/` -- unit tests plus the end-to-end V1 milestone test
  (`test_v1_milestone.py`).
- `tests/data/cube_100mm.stl` -- generated test fixture (see `fixtures.py`),
  separate from `input/`'s user-facing example copy.

## Setup

```bash
poetry install
```

## Scanning a part

```bash
# generate an example part to scan, if you don't have one yet
poetry run python -m fourmp.sensor_sim.fixtures --out input/cube_100mm.stl

# scan it -- looks up input/cube_100mm.stl by name, or pass a path directly
poetry run 4mp cube_100mm
```

Writes `output/cube_100mm/`: `height_map.npy`/`.png`, `ground_truth.npy`/
`.png`, `residual.png`, and `report.json` (metrics + run metadata). Run
`poetry run 4mp --help` for the full option list (face-normal selection,
scan-resolution striding, custom input/output directories).

## Tests

```bash
poetry run pytest
```
