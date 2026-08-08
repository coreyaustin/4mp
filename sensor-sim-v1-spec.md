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

## V1.2 — part-frame up-axis convention (surfaced testing a non-cube part, 2026-08-08)

Testing a tapered-frustum STL (Z-up) surfaced that `part.py` hardcoded the
assumption that a part file's own frame is +Y-up, "satisfied by
construction" only for the cube fixture and never actually checked. For a
Z-up file this broke two things silently: the "reject faces nearly parallel
to the rotation axis" check (meant to exclude the part's real top/bottom
faces) was keyed to the wrong axis and would not have caught the frustum's
actual top/bottom caps; and the mounting transform centered the wrong pair
of axes, since it assumed Y was vertical when the file's vertical axis was
Z.

**Two changes, together:**

1. **Default part-frame up-axis becomes +Z, not +Y** — CNC/CAM convention
   (Z as the vertical/spindle axis), which is what real part files will
   actually be authored in.
2. **New CLI argument `--up-axis X Y Z`** (3-vector, default `0 0 1`, same
   style as the existing `--face-normal`) to override for files authored
   with a different native convention.

**Implementation approach:** don't rewrite `compute_theta_face`/
`compute_t_mount` to work generically about an arbitrary rotation axis.
Instead, apply a single fixed change-of-basis to the mesh at ingestion time
that remaps the specified up-axis onto the sensor frame's own rotation axis
(stays **Y internally, unchanged** — the sensor's fixed hardware convention
in `geometry.py`, independent of how any part file is authored), then run
the existing pose math unchanged downstream.
- Default case (part +Z up → internal +Y): the same -90°-about-X remap
  already verified by hand on the pyramid test file:
  `(x, y, z) -> (x, z, -y)`.
- General case (arbitrary `--up-axis` vector, not just a cardinal axis):
  build the remap via a general axis-to-axis rotation (e.g. Rodrigues'
  rotation formula rotating `up_axis` onto `+Y`) rather than special-casing
  cardinal axes.
- `compute_theta_face`'s existing decomposition logic (normal's component
  along the rotation axis ignored, remaining in-plane component rotated
  onto the sensor's boresight bisector) needs **no change** — it's already
  general enough once the remap above runs first. The fix is entirely at
  the ingestion boundary, not the pose/triangulation math.
- `_select_facet`'s axial-rejection check already takes `rotation_axis` as
  a parameter — it's currently just always called with the hardcoded Y
  constant. No change needed inside the function; fix is upstream.

**Also required:** re-author `fixtures.py`'s test cube to be Z-up (matching
the new default), since it currently only satisfies the old Y-up assumption
by construction. Re-run the V1 milestone test after this change to confirm
the cube still reconstructs correctly under the new default convention.

## V1.3 — align to 4MP's cell-wide coordinate-frame doc (2026-08-08)

4MP has a cell-wide reference doc standardizing frame conventions across the
scanner, rotary table, and cutting tool (internal:
`coordinate_transforms_equations_v3.md` / `transform_point_cloud_v3.py`).
Adopt it — see "Coordinate frame convention" in `architecture-decisions.md`
for the full derivation. Summary of what changes:

1. **Relabel the internal sensor/optics frame** from our ad hoc convention
   (`X`=baseline, `Y`=line/rotation axis, `Z`=boresight) to the doc's `O_s`
   convention (`X`=depth/boresight, `Y`=lateral/baseline,
   `Z`=turntable-sweep axis), per this exact mapping:
   ```
   X_(O_s) = -Z_ours     Y_(O_s) = +X_ours     Z_(O_s) = -Y_ours
   ```
   This is a pure relabel (permutation + sign flips) — the actual
   projector/camera optics math and triangulation logic don't change, only
   which axis is called what. Touches `geometry.py`'s documented convention,
   `pinhole.py`, `sensor_config.py`, `measurement.py`, `reconstruction.py`
   wherever axes are referenced by letter/position.

2. **Introduce an explicit `O_r` (rotary table) frame** with a fixed "home"
   relationship to the scanner frame at table angle θ=0 (rotation axis at
   `X=0, Z=working_distance` in the old sensor-frame terms — already known
   exactly in sim, no calibration needed). `T(O_r ← O_s)` at any θ is that
   home relationship composed with the rotation by θ about the shared
   vertical axis — mechanically close to what `part.py`'s `compute_pose`
   already does, just exposed as its own named, composable transform
   (matching the doc's `T(rotary←scanner)`) instead of folded invisibly into
   one part→sensor transform.

3. **Generate height maps and plots in `O_r`, not raw sensor/camera
   coordinates:** `X` (horizontal) and `Z` (vertical/up) as the two in-plane
   spatial axes, with the signed height/deviation value being `O_r`'s `Y`
   component (into the page) after applying `T(O_r ← O_s)`. Fold this into
   the V1.1 regridding work above — same fix, now targeting the correct
   standardized frame instead of an arbitrary physical-XY plane.

4. **Express the stage's one rotational DOF as yaw** (about the shared
   vertical axis, matching the doc's intrinsic `Z→X→Y` yaw/pitch/roll
   convention) rather than a bespoke `rotation_y`, so it composes cleanly if
   `transform_point_cloud_v3.py`'s helpers are ever called directly against
   our sim's transforms. Pitch/roll aren't needed (single-DOF stage).

5. **Don't reimplement compose/invert/quaternion machinery from scratch** —
   `geometry.py`'s `Transform` already does rotation+translation and
   composition (`.then()`); add an `invert()` method and a
   quaternion-conversion helper to be drop-in compatible with
   `transform_point_cloud_v3.py`'s conventions, rather than a parallel
   implementation.

`O_t` (cutting tool) is not needed — no cutting engine in our scope.
