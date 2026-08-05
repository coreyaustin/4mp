"""End-to-end smoke test for the DLP triangulation scanner simulation.

Builds a synthetic SUT with a raised boss, scans it by sweeping a single-row
line across the source (no structured-light/fringe patterns), reconstructs a
height map from the scan, and checks that a forward render of the
reconstruction matches a fully-lit reference capture.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fourmp.dlp_scanner import SUT, Camera, Source
from fourmp.dlp_scanner.pipeline import validation_loop

RESOLUTION = (48, 56)
BASELINE_MM = 260.0
STANDOFF_MM = 320.0
TABLE_ANGLE_RAD = 0.0
ROW_STRIDE = 2  # skip every other row for a faster demo scan


def make_true_sut() -> SUT:
    sut = SUT.flat(half_extent_mm=40.0, resolution=81, z0=0.0)
    xx, yy = np.meshgrid(sut.x, sut.y)
    boss = np.exp(-((xx**2 + yy**2) / (2 * 12.0**2))) * 6.0
    return sut.with_heights(sut.heights + boss)


def main() -> int:
    source = Source.at(position=(-BASELINE_MM / 2, 0.0, STANDOFF_MM), resolution=RESOLUTION)
    camera = Camera.at(position=(BASELINE_MM / 2, 0.0, STANDOFF_MM), resolution=RESOLUTION)

    sut_true = make_true_sut()
    sut_guess = SUT.flat(half_extent_mm=40.0, resolution=81, z0=0.0)

    scan_rows = range(0, RESOLUTION[0], ROW_STRIDE)
    reconstructed, predicted_frame, reference_frame, rmse = validation_loop(
        source, sut_true, sut_guess, TABLE_ANGLE_RAD, camera, rows=scan_rows
    )
    print(f"forward-vs-captured RMSE: {rmse:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes[0, 0].imshow(reference_frame, cmap="gray")
    axes[0, 0].set_title("Captured reference frame (true SUT, all rows lit)")
    axes[0, 1].imshow(predicted_frame, cmap="gray")
    axes[0, 1].set_title("Forward render of reconstructed SUT")
    axes[1, 0].imshow(sut_true.heights, cmap="viridis")
    axes[1, 0].set_title("True height field")
    axes[1, 1].imshow(reconstructed.heights, cmap="viridis")
    axes[1, 1].set_title("Reconstructed height field (from line scan)")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
