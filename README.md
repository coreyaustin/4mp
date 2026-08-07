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
- `tests/sensor_sim/` -- unit tests plus the end-to-end V1 milestone test
  (`test_v1_milestone.py`).
- `tests/data/cube_100mm.stl` -- generated test part (see `fixtures.py`).

## Setup

```bash
poetry install
```

## Tests

```bash
poetry run pytest
```
