"""Line-scan pattern generation.

Per project decision, the source only ever projects a single scan line (a
band of active DMD rows) swept across the row range — not a multiplexed
fringe/structured-light code. Each shot's row index is known directly, so the
reverse pipeline never has to decode anything: every illuminated camera pixel
triangulates straight against that row's known light plane.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SourcePattern:
    image: np.ndarray  # (rows, cols) intensity in [0, 1]

    @classmethod
    def all_on(cls, resolution) -> "SourcePattern":
        rows, cols = resolution
        return cls(image=np.ones((rows, cols)))

    @classmethod
    def single_row(cls, resolution, row: int, line_width_rows: int = 1) -> "SourcePattern":
        """A single scan line: `line_width_rows` active rows centered on `row`."""
        rows, cols = resolution
        image = np.zeros((rows, cols))
        lo = max(0, row - line_width_rows // 2)
        hi = min(rows, lo + line_width_rows)
        image[lo:hi, :] = 1.0
        return cls(image=image)
