"""The rotary table: defines the world frame's origin and the SUT's only per-scan degree of freedom."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RotaryTable:
    """Holds the table's current angle.

    Source and camera poses never reference this directly — only the SUT's
    `transform()` call takes the angle, per the architecture doc's world-setup
    section (re-machining or re-posing the SUT never touches optics geometry).
    """

    angle_rad: float = 0.0

    def rotated_to(self, angle_rad: float) -> "RotaryTable":
        return RotaryTable(angle_rad=angle_rad)
