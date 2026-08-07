from pathlib import Path

import pytest

from fourmp.sensor_sim.sensor_config import build_sensor_config

CUBE_STL = Path(__file__).resolve().parents[1] / "data" / "cube_100mm.stl"


@pytest.fixture
def sensor_config():
    """Full confirmed-parameters sensor config (1600 scan steps, 2716
    mirrors/line, 5328x3104 camera). Full-resolution runs against the tiny
    test cube complete in ~1-2s, so there's no need for a separate reduced
    fixture."""
    return build_sensor_config()


@pytest.fixture
def cube_stl_path():
    assert CUBE_STL.exists(), f"missing test fixture: {CUBE_STL}"
    return CUBE_STL
