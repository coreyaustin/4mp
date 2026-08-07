# Sensor Simulation — V1 Build Spec

Simulate 4MP's DMD-based line-scan triangulation sensor measuring a single
face of a part, and reconstruct a height map from the simulated scan —
matching the format used by the correction-engine team's Unity pipeline.

Two engines, sharing a part object:

- **Measurement engine** — simulates the sensor against a posed face mesh →
  outputs raw scan data (a stack of line images, one per scan step).
- **Reconstruction engine** — triangulates that scan data into a height map
  (mm, camera-pixel-native grid).

## Confirmed parameters

| Parameter | Value |
|---|---|
| Working distance | 470mm |
| Triangulation half-angle (boresight to each arm) | 27° |
| Measurement area | 188mm (short) × 319mm (long) |
| DMD array | 2716 × 1600 mirrors, 5.4µm pixel pitch |
| DMD-axis → measurement-area mapping | 2716-axis ↔ 319mm long dimension; 1600-axis ↔ 188mm short dimension |
| Camera array | 5328 × 3104 pixels |
| Scan structure | 1600 scan steps (one DMD row at a time, stepping across the 1600-axis); each step fires all 2716 mirrors in that row simultaneously (line spans the 319mm dimension) |
| Intensity | Binary (hit/no-hit), no BRDF |
| Camera model | Ideal pinhole, no distortion, FOV slightly larger than measurement area (margin TBD) |
| Height convention | Signed deviation from a reference plane at the working distance (`height = distance_from_camera − reference_distance`) |
| Height map grid origin | (0, 0) at grid center (placeholder, will change once the team's schema lands) |

## Coordinate frames & pose

Three frames:
- **Sensor frame** — fixed; boresight/bisector axis is the primary axis.
- **Part frame** — native to the ingested STL.
- **Face frame** — per face: centroid at origin, outward normal on one axis,
  a consistent in-plane axis for the other two. Computed during ingestion.

Fixture is a single-axis rotary stage. Pose of a face in sensor space:

```
pose(face) = R_z(θ_face) · T_mount
```

- `T_mount` — fixed, idealized: part's bottom flush on the stage top,
  centered on the rotation axis. Same for every face on a given part.
- `θ_face` — per face: decompose the face's outward normal into a component
  along the stage's rotation axis and a component in the rotation plane;
  solve the angle that rotates the in-plane component onto the sensor's
  boresight/bisector axis (atan2-style).

Transform the face mesh **into** sensor space before running the measurement
engine (standard machine-vision extrinsics convention) — pose is a
preprocessing step, not something the ray-tracing/triangulation math needs to
know about.

The part's top face is out of scope — the rotary stage can't bring it into
view, and it isn't a face we plan to measure.

## Sensor model (V1 — idealized, no Zemax dependency)

1. Collimated light source sends one ray to each DMD mirror.
2. DMD is addressable — "off" mirrors terminate their ray; "on" mirrors
   reflect it onward.
3. An idealized projection lens magnifies the ray bundle to fill the
   188×319mm measurement area.
4. The ray reflects off the posed face mesh and heads toward the camera.
5. The camera records which pixel the ray lands on, with binary intensity.
6. No-hit (mirror off, ray misses the part, or reflected ray misses the
   camera) → the scan step simply produces nothing. No special encoding.

**Simplification:** binary intensity + collimated source + idealized lens
collapses the whole DMD/source/lens system into an **idealized
inverse-pinhole projector** (mirror index → ray direction) — the DMD-side
equivalent of treating the camera as an idealized pinhole. No Zemax data is
needed for V1.

**Forward model (measurement engine), per scan step:**
1. For each "on" mirror in the active row, get its ray direction from the
   inverse-pinhole projector model.
2. Ray-trace against the posed face mesh → nearest intersection, if any.
3. Reflect toward the camera and forward-project using the camera's known
   calibration (intrinsics + pose) → which pixel it lands on, if any.
4. Record a binary hit (pixel address, step index) or nothing.

**Inverse model (reconstruction engine):** for each hit pixel, back-project
using the *same* camera calibration to get a camera ray, and intersect it
with the *same* projector ray for that step → triangulated 3D point in
sensor space. **The forward and inverse models must share the identical
projector-ray model and camera calibration** — any mismatch shows up as a
phantom reconstruction error, not a real algorithm issue.

Bin/grid the triangulated points into the height map (mm, camera-pixel-native
grid, signed deviation from the reference plane, origin at grid center).

## Part object (minimal placeholder shape — no final schema yet)

- `mesh` — the face geometry (from CAD ingestion).
- `face_frame` — centroid + normal + in-plane axis.
- `T_mount`, `θ_face`, and the composed `pose`.
- `brdf` — placeholder field, unused in V1 (binary intensity has no
  photometric model).
- `height_map` — reconstruction engine's output.

Don't wait on 4MP's companion schema (units, tessellation tolerance,
material, etc.) — it doesn't exist yet. Use this minimal shape and expect to
re-derive it from the real schema later.

## CAD ingestion

- Source format: STL only.
- Pipeline: import → validate/repair (watertight, manifold — trimesh or
  similar) → confirm units → compute face frame + `θ_face` for the face(s)
  of interest.

## V1 test case

- Test part: **a cube**, provided as an STL file.
- Test face: **one side face** (not the top).
- Measure it (idealized sensor model above) → reconstruct → height map.
- Compare against ground truth sampled directly/analytically from the same
  face's true mesh (same convention: mm, signed deviation, same grid).
- Use **both** metrics: pointwise residual (RMS/max) and a spectral
  comparison. No injected error needed — the true geometry is already known.

## Explicitly out of scope for V1

- BRDF / photometric shading (Lambertian or specular) — intensity is binary.
- Zemax-derived chief rays, PSFs, or any per-mirror interpolation.
- Camera lens distortion.
- Occlusion handling.
- Multi-face / multi-pose stitching.
- Cutting engine, correction engine, known-error injection.
- Final companion part schema (use the placeholder shape above).

## Known open parameters (non-blocking — don't gate V1 on these)

- Full Z measurement range — depends on optical modeling/testing not done
  yet. V1's idealized model doesn't need a hard range limit.
- Exact camera FOV margin over the measurement area.
- Final height-map grid origin/coordinate convention (pending the team's
  schema — using grid-center origin for now).

## V1.1 — calibration fix (first-run findings, 2026-08-07)

First run against the cube fixture (100mm cube, one face) reconstructed
correctly geometrically (ground truth is flat to ~10 significant figures,
confirming pose/triangulation math) but surfaced two follow-ups, both
scoped as part of finishing V1 rather than deferred to Phase 2:

1. **Regrid the height map onto a uniform physical XY grid (mm), not raw
   camera-pixel indices.** The current camera-pixel-native grid makes a flat
   rectangular face render as a trapezoid (an artifact of indexing by a
   physically tilted camera's row/col, not a defect — it appears identically
   in `ground_truth.png`), and it also means plots are labeled in pixels
   instead of mm. Bin/interpolate the triangulated `(X, Y, Z)` points from
   `reconstruction.py` onto a regular XY grid (e.g. resolution matched to the
   ~60-70µm/pixel footprint at the reference plane) before producing
   `height_map.npy`/`.png` and `ground_truth.npy`/`.png`. This single change
   fixes both the trapezoid-shape artifact and the pixels-vs-mm plotting
   request. Keep the current camera-pixel-native array available too (or
   easily reproducible) since some diagnostics (e.g. the triangulation gap
   map) are naturally expressed per-camera-pixel.

2. **Reconstruct at sub-pixel resolution — drop nearest-pixel rounding as
   the V1 camera model, not just a diagnostic.** The real sensor locates the
   imaged line to sub-pixel resolution via a centroid/peak-finding algorithm
   over the line's intensity profile; nearest-integer-pixel rounding
   (`PinholeModel.pixel_indices()`'s `np.rint()`) was never a faithful model
   of that — it was an incidental side effect, not a deliberate fidelity
   choice, and it's the dominant driver of the ~25.5µm RMS / 46.7µm max
   pointwise residual (same order of magnitude as `max_triangulation_gap_mm`
   and the camera's pixel footprint at the reference plane). Since V1 has no
   photometric/PSF model yet to run a real centroid estimator over, the
   faithful V1 stand-in is a **perfect, zero-error sub-pixel estimate**:
   reconstruct from the camera's continuous projected `(i, j)`
   (`project_points()`) instead of the rounded integer output. This should
   make the forward/inverse round trip geometrically exact again (residual
   should collapse to floating-point noise).
   - The camera's photosensitive array is still physically a discrete pixel
     grid — sub-pixel centroiding doesn't eliminate that. Recommend keeping
     a rounded/discrete "recorded pixel" address alongside the continuous
     value, for bookkeeping only (collision detection between scan steps
     landing on the same physical pixel, indexing into any camera-pixel-
     native intermediate array) — but feed the **continuous** value into
     `directions_for_indices()`/triangulation, not the rounded one.
   - This is a placeholder the same way binary intensity is a placeholder
     for a real BRDF model — a later fidelity phase should replace "perfect
     centroid" with a realistic centroid-estimation error model (a function
     of line width/PSF and per-pixel SNR) once the photometric side of the
     sensor model exists. Not required for V1.1.
