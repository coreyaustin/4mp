# output/

Per-run artifacts from the CLI (`4mp <part-name>`), one subdirectory per
part scanned: height map (`.npy` + `.png`), ground-truth/residual plots, and
a `report.json` with the pointwise/spectral validation metrics.

**Not tracked in git** (see `.gitignore`): generated from whatever's in
`input/`, which may itself be confidential. This README is the exception,
kept tracked so the directory itself exists on a fresh clone.
