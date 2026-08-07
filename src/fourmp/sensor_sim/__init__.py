"""V1 sensor simulation pipeline for 4MP's DMD-based line-scan triangulation sensor.

See ``sensor-sim-v1-spec.md`` (repo root) and ``architecture-decisions.md`` for the
build spec and the reasoning behind the modeling choices made here. Two engines,
sharing a common sensor calibration (:class:`fourmp.sensor_sim.sensor_config.SensorConfig`)
and a common :class:`fourmp.sensor_sim.part.Part` object:

- :mod:`fourmp.sensor_sim.measurement` -- forward model (posed mesh -> raw scan hits).
- :mod:`fourmp.sensor_sim.reconstruction` -- inverse model (raw scan hits -> height map).
- :mod:`fourmp.sensor_sim.validation` -- ground-truth sampling + pointwise/spectral metrics.

Everything under "Explicitly out of scope for V1" in the spec (BRDF/photometric
shading, Zemax chief rays/PSFs, camera lens distortion, occlusion handling,
multi-face/multi-pose stitching, cutting/correction engines, error injection,
the final companion part schema) is intentionally not implemented here.
"""
