"""Forward/reverse simulation of a DLP-projector triangulation scanner.

Implements the architecture from the design artifact: a rotary-table-centric
world frame, four hardware models (source, table, SUT, camera), and a forward
+ reverse pipeline pair used to validate a reconstructed SUT against a
captured scan, before any of it exists in hardware.
"""

from .camera import Camera
from .centroid import centroid_rows
from .patterns import SourcePattern
from .pipeline import acquire_line_scan, forward_pipeline, reverse_pipeline, validation_loop
from .source import Source
from .sut import SUT
from .table import RotaryTable

__all__ = [
    "SUT",
    "Camera",
    "RotaryTable",
    "Source",
    "SourcePattern",
    "acquire_line_scan",
    "centroid_rows",
    "forward_pipeline",
    "reverse_pipeline",
    "validation_loop",
]
