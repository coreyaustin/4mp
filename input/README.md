# input/

Drop the STL files you want to scan here. The CLI (`4mp <part-name>`) looks
in this directory by default.

**Not tracked in git** (see `.gitignore`): part geometry may be real
product/customer designs, so nothing under here is committed unless you
explicitly `git add -f` a specific file. This README is the exception, kept
tracked so the directory itself exists on a fresh clone.

For a quick local example to scan immediately, generate the test cube:

```bash
poetry run python -m fourmp.sensor_sim.fixtures --out input/cube_100mm.stl
```
