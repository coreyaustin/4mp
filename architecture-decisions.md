# Sensor Simulation Pipeline — Architecture Decisions (living doc)

Last updated: 2026-08-07

## Purpose

Simulate 4MP's sensor to generate/validate measurement and reconstruction of a
given part, producing a height map in the same format used by the
correction-engine team's Unity pipeline.

## SCOPE NARROWED (2026-08-06, three rounds)

**Round 1:** The correction-engine team generates their own synthetic training
data via Unity (see "Other team's pipeline" section below) — output is a
height map (2D array, one height per cell), captured with the part on a
rotary stage and a camera at the approximate real camera pose, one height map
per pose. That data feeds their correction engine directly. As a result, our
simulation dropped the **cutting engine** and **correction engine** entirely —
scope became measurement + reconstruction only, producing a height map in a
format compatible with their Unity output.

**Round 2:** We also don't need **known-error injection**. The part to measure
is simply *given* — we don't need to deliberately deform a nominal mesh to
create a validation target. Since this is a simulation, we always know the
exact true geometry of whatever part we're given, so validation still works:
sample the given mesh directly (analytically) into a height map as ground
truth, and compare it to whatever the measurement + reconstruction pipeline
produces from simulated sensor data. This is a **test-harness concern**, not a
pipeline feature.

**Round 3:** Scope narrows to **measuring individual faces**, not full
multi-face/multi-pose parts, for now. Height output is in **mm**. The height
map array is sized to match the **camera sensor's actual pixel arrangement**
(exact pixel dimensions TBD) rather than an independently chosen grid — the
grid is native to the sensor, not an arbitrary sampling choice.

**Current scope, in full:** given a single face of a part (mesh + material),
simulate the sensor measuring it at a given pose → produce a reconstructed
height map (mm, camera-pixel-native grid), in the shared Unity-compatible
format. Two engines, no injection, no cutting, no correction, single face to
start. The part's top face is out of scope for measurement (see "Part/face
pose" below).

Sections below marked **[OUT OF SCOPE]** are kept for reference (decisions
already made, in case scope changes again) but are not part of the current
build target.

## Tech stack

- **Language: Python.** Fits the mesh/CAD tooling already in play (trimesh /
  PyMeshLab-type validation, numpy, scikit-image for marching cubes, etc.).
- **[OUT OF SCOPE]** Toolpath format (G-code, Haas dialect) — was relevant to
  the cutting/correction engines; not needed now.

## Pipeline structure (current scope)

Two engines, sharing a part object:

- **Measurement engine** — simulates the sensor (see "Measurement engine —
  sensor model" below) against the given face's mesh surface, producing raw
  scan data: a **stack of images, one per scanned line, per pose**.
- **Reconstruction engine** — converts that image stack into a **reconstructed
  height map per pose** (mm, camera-pixel-native grid), via triangulation, in
  the same format as the Unity pipeline's output.

**[OUT OF SCOPE]** Correction engine, cutting engine, and known-error
injection — see "Scope narrowed" above. Prior decisions preserved further
down for reference.

## Validation approach (test harness, not a pipeline feature)

Since the part is simulated, its true geometry is always known exactly.
Validation = sample the given mesh directly into a height map (ground truth,
computed analytically, no sensor simulation involved) and compare it to the
reconstructed height map the measurement+reconstruction pipeline produces.
No deliberate error injection needed — any test face works, including
arbitrary/representative geometry, not just parts we've deformed ourselves.

**Comparison metric: both.** Run both a pointwise residual (reconstructed −
truth, summarized as RMS/max error — simplest to implement, good for a quick
pass/fail) and a spectral comparison (more diagnostic — surfaces systematic/
periodic error patterns a pointwise summary would hide) for every V1 test.
No reason to pick one; they answer different questions and both are cheap to
compute once the height maps exist.

## Height map output contract (interop with the other team's pipeline)

Our reconstruction engine's output needs to match the Unity pipeline's height
map format so it's directly comparable/usable alongside their synthetic
training data.

### Height convention: signed deviation from a reference plane, not raw range

Confirmed via the other team's Unity setup, and worth adopting for our own
output too. Laser-triangulation displacement/profile sensors (the class of
sensor both the Keyence unit they're mimicking, and our own custom DMD sensor,
belong to) report height as a **signed deviation from a reference plane**
positioned at the sensor's **reference distance** — not as raw distance from
the sensor.

Verified against Keyence's published LJ-S320 specs (the unit their Unity setup
mimics), which match what the colleague described:
- **Reference distance: 470.5 mm** (colleague said "~470 mm" — confirmed).
- **Z-axis measurement range: −120 mm to +100 mm** (220 mm full-scale) —
  matches "−120 to 100" the colleague described. The asymmetry reflects the
  triangulation optics' near/far field-of-view geometry (243 mm near-side FOV
  vs. 320 mm far-side FOV), not an arbitrary choice.
- [Sensor head - LJ-S320 | KEYENCE America](https://www.keyence.com/products/measure/laser-2d/lj-s8000/models/lj-s320/)

**Why "subtract the reference distance":** a raw Unity raycast returns the
straight-line distance from the virtual camera to the surface point along the
optical axis — call it ~470.5 mm ± the actual surface height. That raw number
isn't what the real sensor reports. Subtracting the reference distance
(`height = raw_distance_from_camera − reference_distance`) converts it into
the sensor's native convention: 0 at the reference plane, negative when
closer to the sensor, positive when farther. That's "height referenced to a
plane" — the colleague's read on it, and the correct one.

**Implication for our sensor:** our reconstructed height maps should use the
same convention — signed deviation from a reference plane at *our* sensor's
own reference/working distance — for consistency with the Keyence-style
output the other team is matching. The reference plane is perpendicular to
the sensor's boresight/bisector axis (see "Part/face pose" below) at the
reference distance. Working distance is confirmed (470mm), and the
triangulation half-angle is confirmed (27° — see "Real hardware constraints"
below); the full Z measurement range is genuinely open (see below).

**Grid origin: (0,0) at the center of the grid** (confirmed, temporary). This
will change once the companion schema/grid-convention spec comes from the
rest of the team — noted here so it's easy to find and swap out later rather
than buried in code.

### Still need to obtain / decide

- **Full Z measurement range** — working distance (470mm) and triangulation
  half-angle (27°) are confirmed, but the full Z range **can't be pinned down
  yet** — it depends on quite a bit more optical modeling and bench testing
  that hasn't happened. Not a blocker for V1 (the idealized sensor model
  doesn't need a hard range limit yet), but flagged so it isn't assumed to
  be a simple placeholder-away problem.
- **Camera pixel array dimensions** — confirmed: **5328×3104** (candidate
  sensor: Sony IMX532-AAMJ, monochrome, global shutter). Height map grid
  should be sized to this array.
- **Pose scheme** — which rotary-stage angles/how many poses (likely the same
  "24 rotary viewpoints" mentioned in the other team's spec) — though with
  scope narrowed to single-face measurement, may not be needed for V1.
- **Open question**: does our reconstruction engine's scope end at one height
  map per pose/face, or does it also need to stitch multiple into one combined
  map? Deferred given the single-face V1 scope, but will resurface once
  multi-face/multi-pose parts come back into scope.
- Likely source for the eventual real grid/pose spec: the other team's
  referenced "source-agnostic record contract"
  (`wam-stage2-design/sample_schema.md`) — worth requesting directly, or
  getting a sample Unity height map file. Until then, V1 uses the
  center-origin convention above.

## Part/face pose — orienting the part relative to the sensor

How to represent "where the face is relative to the sensor," and how
"rotating to measure a different face" works.

**Fixture: confirmed as a single-axis rotation stage.** The part is placed on
the stage and rotated to bring different faces into the sensor's view. This
resolves the earlier open question (single-axis vs. arbitrary orientation) in
favor of single-axis, and simplifies the pose math considerably from the
general 6-DOF case.

**The part's top face is out of scope for measurement** (confirmed) — no
flip/secondary-fixture mechanism needed. This also resolves the single-axis
stage's physical limitation (a face whose normal points straight along the
rotation axis can't be rotated into view) — it simply isn't a face we plan to
measure.

**The sensor's boresight is a bisector, not a single ray.** The system design
has the *nominal* face normal bisecting the angle between the illumination
source and the camera — i.e. the illumination source and camera sit
symmetrically at equal-and-opposite angles about this boresight axis. This
precisely defines what "pointing the face at the sensor" means: aligning the
face normal with this bisector direction, not with the camera axis or the
illumination axis individually. The reference plane (see "Height map output
contract" above) is perpendicular to this same bisector axis, at the
reference distance. This describes the nominal/central configuration — as the
DMD sweeps to off-center mirrors, those individual chief rays have their own
distinct angles relative to the camera (that asymmetry across the sweep is
exactly why the sparse-sample-plus-interpolation plan exists, rather than
assuming symmetry holds everywhere). Once the Zemax chief-ray table exists,
its central/reference-mirror entry should reproduce this bisector geometry —
worth using as a sanity check. **Half-angle confirmed at 27°** from the
boresight/bisector to each arm (illumination and camera) — see "Real
hardware constraints" below.

**Three frames, not two:**
- **Sensor frame** — fixed. Everything in the sensor model (chief ray table,
  camera calibration, reference plane) is already defined relative to this,
  with its primary axis being the boresight/bisector direction above.
- **Part frame** — native to the ingested mesh (whatever the STL's own
  coordinate system is).
- **Face frame** — a local frame per face of interest, derived from that
  face's own geometry (centroid at origin, outward normal along one axis, a
  consistent in-plane direction for the other two). Computed once per face
  during CAD ingestion / face selection, so every face has a self-consistent
  local frame independent of how it sits on the part.

**Convention:** transform the face's mesh *into* sensor space before running
the measurement engine, rather than transforming the sensor around the part
(standard machine-vision convention — extrinsics move the object into camera
space). This means pose handling is a preprocessing step; none of the
ray-tracing/triangulation math needs to know poses exist.

**Representation, now decomposed for the single-axis stage:** each face's
pose is still a rigid 6-DOF transform (`pose(face) = R_z(θ_face) · T_mount`),
but now generated from two simple pieces instead of specified arbitrarily:
- **`T_mount`** — the fixed transform from part frame to stage frame (how the
  part sits on the stage), the same for every face.
- **`θ_face`** — the one number that differs per face: the stage rotation
  angle that brings that face's normal into alignment with the sensor's
  boresight/bisector axis.

**Mounting convention: idealize it (aligned).** Assume the part's bottom sits
flush on the stage's top surface, centered on the rotation axis — i.e.
`T_mount` is a simple fixed vertical offset with no arbitrary rotation or XY
translation. This removes unknowns for no real cost: with alignment, every
face's rotation angle is directly computable from the part's own CAD geometry;
without it, `T_mount` picks up extra unknowns (an arbitrary offset
rotation/translation) that would need to be measured, guessed, or calibrated
for no benefit at this stage. Consistent with idealizing other parts of the
system first (sensor optics, occlusion) and adding realism later — if
mounting/fixturing tolerance ever needs to be stress-tested, that becomes a
deliberate later perturbation, not a default assumption. **This idealized
choice is validated by the real system's design**: the rotary stage is also
the system's calibrated coordinate origin (see "Real hardware constraints"
below, specular-reflection mitigation), which is exactly the role `T_mount`
plays here.

**Computing `θ_face`:** decompose the face's outward normal (known from its
face frame) into a component along the stage's rotation axis and a component
in the rotation plane; solve for the angle that rotates the in-plane
component to align with the sensor's boresight/bisector direction (a simple
atan2-style calculation). Done once per face during ingestion/face-selection.

## Real hardware constraints (from optical design memo, 2026-08-05)

4MP's internal optical-design memo (responses to a partner's design
questions) locks down several real values our simulation should track. This
is the bridge between "how we're simplifying the sim" (next section) and
"what the real sensor will actually be" — kept here as reference, alongside
the specific choices made for our simulation.

**Confirmed:**
- **Working distance: 470mm** — matches the Keyence LJ-S320 baseline already
  used for our height-map convention (see "Height map output contract").
- **Measurement area: 188mm × 319mm** (confirmed for our simulation). The
  real optical design has a second candidate (295mm × 500mm, DMD's short axis
  matched to Keyence's short axis, ~34× magnification) and may ultimately
  support both via interchangeable lens sets, but our sim targets 188×319mm
  specifically — DMD's long axis matched to Keyence's long axis, ~22×
  magnification, ~119µm projected line width.
- **Triangulation half-angle: 27°** from the boresight/bisector axis to each
  arm (confirmed for our simulation). Falls within the 45°–65° full-angle
  range (i.e. ~22°–32° half-angle) typical of commercial laser-triangulation
  sensors, per published sensor design guidance — so it's a reasonable
  nominal value, though our own optical layout may eventually fix a different
  one once more modeling/testing (see Z-range note above) is done.
  [Principles Of Measurement Used By Laser Sensors And Scanners - Acuity Laser](https://www.acuitylaser.com/sensor-resources/measurement-principles/)
- **DMD: 2716×1600 mirror array, 5.4µm pixel pitch**, no pixel-shifting
  (native resolution only), max incident illumination angle 55° from array
  normal.
- **Camera: 5328×3104** (monochrome, global shutter candidate) — final
  sensor choice still pending quantum-efficiency verification at the source
  wavelength.
- **Scan order (flipped, 2026-08-07): one DMD row at a time, stepping across
  rows.** The **1600-pixel axis is the scan-step axis (1600 total scan
  steps** — down from 2716), and the **2716-pixel axis is the along-line
  axis** (all 2716 mirrors in the active row fire simultaneously per step).
  Chosen deliberately for **fewer scan steps → fewer captured line-images per
  face** (1600 vs. 2716), at the cost of each line now spanning the DMD's
  long axis instead of its short axis. This doesn't change the physical
  DMD-axis-to-measurement-area mapping (2716-pixel axis ↔ the 319mm long
  dimension, 1600-pixel axis ↔ the 188mm short dimension — fixed by the
  projection optics, see measurement-area bullet above) — only which axis is
  stepped vs. fired together. So the projected line now spans the full
  319mm dimension at each step, sweeping across the 188mm dimension in 1600
  steps.
- **Camera FOV design intent:** camera FOV will be slightly larger than the
  measurement area, to guarantee full coverage. Margin amount TBD — doesn't
  block V1 (the idealized camera model has no FOV clipping yet).
- **No-hit handling:** if a ray from an "on" mirror misses the part, or the
  reflected ray misses the camera, it simply terminates — no special
  encoding, just no recorded hit. (Distinct from occlusion's deliberate
  non-illumination tagging, which is a separate, still-deferred concept.)

**Later-fidelity items surfaced by the memo** (not needed for V1; folded into
the fidelity roadmap in the next section):
- **Scheimpflug-driven, sweep-position-dependent defocus** — the receiver
  lens uses a *fixed* Scheimpflug tilt (no dynamic adjustment as the line
  sweeps, consistent with no moving optical elements). Since the
  triangulation angle changes continuously across the sweep, a fixed tilt
  only exactly satisfies the Scheimpflug condition at one sweep position;
  elsewhere, defocus varies. This degrades line-centroid precision (width /
  contrast / uncertainty) — it is *not* a static geometric bias, so it
  wouldn't show up as a calibration error, only as increased scatter.
- **Specular reflection — two distinct failure modes**, each needing its own
  mitigation: **saturation** (the specular lobe reflects straight into the
  receiver and clips the sensor even at minimum exposure) and **directional
  signal loss** (other surface orientations reflect light away from the
  camera, starving the line of signal). One relevant mitigation for the sim:
  **rotary repositioning** — since the part sits on the same rotary stage
  that serves as the system's calibrated coordinate origin, a modest tilt
  moves the part off the exact specular condition, and the reflected ray
  moves at **twice the rate** of any change in surface orientation (standard
  mirror-reflection physics), so only a small additional rotation is needed.
  Directional-loss has no single-camera fix; a dual-camera differential
  architecture is a deferred Phase-2 idea, not needed for the current
  single-camera scope. Together, these confirm real metal parts have a
  meaningful specular BRDF component — Lambertian-only (our near-term
  assumption) is a genuine simplification to revisit later, not a final
  model.
- **Calibration model:** the existing artifact-based calibration method needs
  extension to account for two field angles (angle varying with sweep
  position, and angle varying along the length of each projected line). The
  planned correction is a **low-order polynomial surface fit over the two
  field angles** (not a dense per-mirror lookup table) — cheap to evaluate on
  an embedded platform, and consistent with standard lens-distortion
  modeling convention. Worth knowing now: when Zemax data eventually feeds
  the sim, it's more likely to arrive as polynomial field-angle correction
  coefficients + intensity profiles than as a literal discrete per-mirror
  chief-ray table.

## Measurement engine — sensor model (DMD line scanner)

The sensor is a new, custom design: functionally similar to a laser
triangulation line scanner, but uses a DMD to steer/scan the illumination
(selectively activating individual micromirrors) instead of a moving galvo
mirror. At the nominal/central configuration, the illumination source and
camera sit symmetrically about the boresight axis at the confirmed 27°
half-angle (the nominal face normal bisects the angle between them — see
"Part/face pose" above).

**V1 optical model — deliberately simplified, no Zemax dependency:**

1. **Collimated light source** sends one ray to each DMD mirror.
2. **DMD is addressable**: "off" mirrors terminate their ray immediately (no
   reflection); "on" mirrors reflect it onward.
3. An idealized **projection lens** magnifies the bundle of rays leaving the
   DMD to fill the **188mm × 319mm** measurement area (confirmed — see "Real
   hardware constraints" above).
4. The ray reflects off the part face's mesh (already posed into sensor
   space) and heads toward the camera.
5. The camera records **which pixel** the ray lands on, with **binary
   intensity** (1 or 0) — hit or no hit. No BRDF/photometric calculation is
   needed for V1, since intensity carries no information yet (this is the
   simplification that drops Lambertian shading from V1 — see fidelity
   roadmap below).
6. **No-hit termination:** if the mirror is off, or the reflected ray misses
   the part, or misses the camera, the scan step simply produces no hit — no
   special encoding (distinct from occlusion's deliberate tagging, still
   deferred).

**Key simplification:** with binary intensity + a collimated source + an
idealized projection lens, the whole "DMD + source + lens" system collapses
into an **idealized inverse-pinhole projector** — mirror index → ray
direction — directly analogous to treating the camera as an idealized
pinhole. This means **V1 has no Zemax dependency at all**; Zemax-derived
chief rays/PSFs become a later fidelity increment (see roadmap), not a V1
input.

**Scan structure (flipped for fewer images, 2026-08-07):** one DMD row at a
time, stepping across rows — **1600 scan steps total** (down from 2716),
each step firing all **2716 mirrors** in that row simultaneously. The
projected line now spans the DMD's long/2716 axis (the 319mm dimension);
scan-stepping happens across the short/1600 axis (the 188mm dimension) — see
"Real hardware constraints" above.

**Forward model (measurement engine, per scan step):**
1. For each "on" mirror in the active row, get its idealized ray direction
   (inverse-pinhole projector model above).
2. Ray-trace it against the given face's mesh (already posed into sensor
   space) → nearest intersection, if any.
3. Reflect the ray toward the camera side of the system and forward-project
   using the camera's known calibration (intrinsics + pose) → which pixel it
   lands on, if it lands on the sensor at all.
4. Record a binary hit (pixel address, step index) or nothing.

**Inverse model (reconstruction engine):** given a hit pixel per scan step,
back-project using the *same* camera calibration to get a camera ray, and
intersect it with the *same* idealized projector ray for the mirror/step
active at that time → triangulated 3D point (in sensor space). **The
measurement engine's forward model and the reconstruction engine's inverse
model must share the identical projector-ray model and camera calibration** —
any mismatch would show up as a phantom reconstruction error that's actually
just a calibration inconsistency in the sim, not a real algorithm limitation.

Reconstruction then bins/grids the triangulated 3D points into the height map
format described in "Height map output contract" above (camera-pixel-native
grid, mm, signed deviation from the sensor's reference plane, origin at grid
center).

**Camera model:** ideal pinhole (known intrinsics/pose), no lens distortion
for now, FOV slightly larger than the measurement area (margin TBD, doesn't
affect V1's idealized/unclipped model).

**V1 fidelity (idealized, matches the V1 milestone below):**
- Single pixel hit per scan step (no PSF blur, no per-mirror interpolation).
- Binary intensity only — no BRDF/photometric model.
- Idealized inverse-pinhole projector — no Zemax dependency.
- Ideal pinhole camera, no distortion.
- 1600 scan steps per face (flipped orientation — see above).
- No occlusion handling (see below — deferred).

**Fidelity roadmap (later phases, roughly in order):**
1. Zemax-derived chief ray angles + PSFs for a sparse sample of DMD mirrors,
   interpolated across the full array — replaces the idealized inverse-
   pinhole projector with the real optical system's geometry.
2. Lambertian BRDF, then a specular BRDF component (real metal parts have
   meaningful specular reflectance — see "Real hardware constraints" above).
   Includes modeling the two specular failure modes (saturation,
   directional loss) and their mitigations (rotary repositioning,
   multi-exposure HDR, etc.).
3. Camera-side optical realism: lens distortion, then Scheimpflug-driven,
   sweep-position-dependent defocus (a fixed receiver-lens tilt means focus
   quality varies across the sweep — affects line-centroid precision, not a
   static geometric bias).
4. Calibration model: polynomial field-angle correction (two field angles —
   sweep position and along-line position) rather than a dense per-mirror
   lookup table, matching the real system's planned calibration approach.

Building full optical fidelity into the very first milestone would conflate
"is the optical model right" with "is the reconstruction algorithm right" —
better to validate reconstruction against clean idealized data first.

## Occlusion (deferred, design captured for later)

Not built for V1 — full visibility is assumed for now. Design intent for when
it is built:

- Because the DMD can selectively disable individual mirrors (unlike a
  mechanically-scanned mirror), occlusion doesn't have to be discovered
  reactively during a real scan. Instead: **simulate the measurement against
  the nominal (as-designed) part first**, identify which chief rays would be
  occluded (self-shadowing / camera can't see the illuminated point), and
  **don't illuminate those mirrors** during the real/actual scan.
- Those skipped scan positions get explicitly tagged for the reconstruction
  engine as **known, intentional non-illumination** (occlusion), not as
  missing/ambiguous data — this is a different failure mode than a real height
  discontinuity, and reconstruction needs to be able to tell the two apart
  rather than guessing at a null value.
- Known limitation to reconcile later (not now): the occlusion mask is
  predicted from the *nominal* part, so an actual part deviation could
  introduce occlusion the nominal geometry didn't have (or remove occlusion it
  did have). Not a blocker — just a gap to revisit when this is actually built.
- When it is eventually simulated: nearest-intersection-along-the-ray logic
  plus a visibility check (second ray from the illuminated point to the
  camera) determines whether a scan step is even valid.

## [OUT OF SCOPE] Correction engine — prior decisions, preserved for reference

- Corrections were planned as a **delta/overlay** applied on top of a base
  G-code program, not full re-emitted G-code (rationale: re-emitting risks
  introducing new errors independent of the ones being corrected).
- The real correction algorithm is owned by a separate team (see "Other
  team's pipeline" below) — was planned as a pluggable stub interface in our
  pipeline. No longer applicable since correction is fully out of our scope.

## Other team's correction-engine pipeline (context, from their V2 spec + colleague conversation)

Reviewed the other team's "G-Code Engine V2 — Learned Correction Layer" spec
(2026-08-05), plus a direct conversation with the colleague doing the Unity
modeling (2026-08-06). Notes, for reference and for matching our output format
to theirs:

- **Their Unity setup:** part on a rotary stage, camera placed at the
  approximate real camera pose (mimicking a Keyence LJ-S320 — see "Height map
  output contract" for verified specs). Output is a **height map (2D array,
  height per cell, mm, signed deviation from the sensor's reference plane)** —
  one per pose — mimicking real scan output at the same poses. This is the
  data used to refine their correction engine.
- **Their error model:** a "ladder" of error causes, each owned by the
  simplest tool that can solve it — similarity transform (7-DOF) → affine
  (12-DOF) → static volumetric map, all deterministic/closed-form, covering
  anything that's a function of *position*. A separate learned model handles
  only the residual that's a function of *motion* (feed/engagement-dependent
  non-affine path bow, their "banana" error).
- **Their simulator split:** geometry/material-removal is a separate "exact
  CAD/kernel layer" (not Unity). Unity's role is narrowly synthetic scan
  capture — depth-camera/structured-light capture from rotary viewpoints (with
  occlusion), raycast sampling into point clouds via Unity Perception, plus
  domain randomization. Point clouds are sampled **analytically off the STEP
  B-rep — explicitly no voxels**, to avoid aliasing away micron-scale
  features. Errors are injected as parameterized machine transforms on
  nominal geometry.
- **Explicitly rejected** a generative-video simulator (e.g. Cosmos) for the
  same reason we would: it would hallucinate parts and corrupt the labels.
- **Model:** scan registered to nominal → signed deviation field d(x) → a
  supervised 3D deviation-field transformer with two heads (dense per-point
  deviation/error segmentation; parametric "banana" regression). Correction
  output is a counter-bow pre-distortion applied to the remaining G-code.
- **Sim-to-real plan:** identical record schema for sim and real; tune
  simulated sensor noise to match their real characterized sensor once
  machines arrive (Fall 2026); switch from exact injected labels to
  measured-deviation labels on real parts.
- **Interop implication for us:** their "sensor" here is a generic Unity
  depth-camera/structured-light abstraction, not physically modeled (no
  chief-ray/PSF/Zemax fidelity) — different purpose (bulk training data) than
  our higher-fidelity sensor simulation. Our reconstructed height maps are
  meant to land in the **same format and convention** as their Unity height
  maps (see "Height map output contract" above) so the two are
  comparable/interchangeable as data sources for their correction engine.

## [OUT OF SCOPE] Cutting engine — prior decisions, preserved for reference

- Planned to consume base G-code + correction overlay, interpret it (incl.
  arc interpolation), and sweep tool geometry through a voxel/dexel volumetric
  representation to simulate material removal.
- Voxel/dexel was chosen over mesh boolean subtraction because repeated
  booleans degrade over many iterations (self-intersections, non-manifold
  artifacts) — this is why commercial CNC simulators (Vericut, NCSimul) use
  similar volumetric approaches. No longer needed without a cutting engine.

## [OUT OF SCOPE] Known-error injection — prior decisions, preserved for reference

- Was planned as: apply a known deviation directly to the nominal mesh's
  surface (displacement along the normal by a parameterized function —
  uniform offset, tilt/taper, dent, ripple, step, etc.) to produce a synthetic
  "actual" part, then compare reconstructed deviation to the known injected
  ground truth.
- No longer needed — see "Validation approach" above. The part is simply
  given, and ground truth comes for free from knowing the simulated part's
  true geometry, without needing to deliberately deform anything.

## Part object — representation (current scope)

Per face/pose, the part object carries:

- **Mesh** — the given face's geometry (from CAD ingestion), whatever it is.
  No "nominal vs. actual" distinction needed — it's just the face being
  measured.
- **Face frame** — the face's own local coordinate frame (centroid + normal +
  in-plane axis), computed during ingestion/face-selection (see "Part/face
  pose" above).
- **Mounting transform (`T_mount`)** — fixed, shared across all faces of a
  given part: how the part sits on the rotary stage (idealized/aligned — see
  "Part/face pose" above).
- **Rotation angle (`θ_face`)** — per face: the stage angle that aligns that
  face's normal with the sensor's boresight/bisector axis.
- **Pose** — the composed rigid transform `R_z(θ_face) · T_mount`, from face
  frame to sensor frame, for this measurement.
- **Surface material/BRDF** — property of the part (or its companion schema),
  consumed by the measurement engine's forward model.
- **Reconstructed height map** — the reconstruction engine's output (mm,
  camera-pixel-native grid, signed deviation from the sensor's reference
  plane, origin at grid center), in the shared height-map format (see
  "Height map output contract").

**Companion schema: starting without it.** 4MP's part-level companion schema
(units, tessellation tolerance, BRDF/material, etc.) doesn't exist yet, and
V1 doesn't wait on it — the part object above is a minimal, ad hoc
representation (mesh, face frame, `T_mount`, `θ_face`, pose, a placeholder
BRDF field) built directly in code. Once the real schema exists, the part
object's fields should be re-derived from it rather than the other way
around — treat this as a placeholder shape, not a preempted schema.

**[OUT OF SCOPE, no longer needed]** Volumetric working representation
(voxel/dexel — was for cutting only); ground-truth-deviation field and
actual/deformed mesh distinction (was for known-error injection only — see
above).

### First concrete test case (V1 milestone) — simplified further

- **Test part: a cube**, provided as an STL file. Confirmed — this resolves
  the flat-plate-vs-representative-face question: a cube face *is* the
  simple flat plate we wanted for hand-checkable debugging, and the cube
  gives us multiple equivalent side faces to test rotation/pose logic
  against without needing a more complex mesh yet.
- **Test face: one side face of the cube** (not the top — out of scope, see
  "Part/face pose").
- Define its face frame, the mounting transform, and the resulting single
  pose (face frame → sensor frame) aligning it with the sensor's
  boresight/bisector axis at the confirmed 470mm working distance / 27°
  half-angle.
- Measure it (idealized inverse-pinhole projector + binary-intensity camera,
  1600 scan steps per the flipped scan orientation — see "Measurement
  engine" above) → reconstruct (triangulation → height map, mm,
  camera-pixel-native grid, signed deviation from reference plane, origin at
  grid center).
- Compare the reconstructed height map to a ground-truth height map sampled
  directly/analytically from the same test face's true mesh (same convention)
  — no injection, no deformation, just direct comparison, using **both**
  pointwise (RMS/max residual) and spectral comparison (see "Validation
  approach" above).
- This defines a clean, bounded V1 scope: one cube face, one pose, idealized
  sensor model (binary-intensity inverse-pinhole projector, 1600 scan steps,
  no BRDF, no PSF, no occlusion, no distortion), measure → reconstruct →
  dual-metric residual against known truth, output in mm on a
  camera-pixel-native grid. Good first Claude Code milestone.

## CAD ingestion

- Source format: **STL** (confirmed — this is what's used in practice; the
  V1 test part will arrive as an STL cube).
- Ingestion pipeline: import → validate/repair (watertight, manifold check;
  tools like trimesh/PyMeshLab) → confirm units → compute face frame(s) and
  rotation angle(s) for face(s) of interest (see "Part/face pose"). (No
  voxelize/dexel-ize step — not needed without a cutting engine.)
- STL is lossy relative to STEP/B-rep: no analytic surface, no units metadata, no
  GD&T/PMI. Tessellation tolerance on export should be meaningfully finer
  (rule of thumb: 5–10x) than the deviation magnitudes of interest, since
  facet error is otherwise indistinguishable from real part deviation. (Low
  risk for a cube's flat faces specifically, but the rule still matters once
  more representative geometry is used.)
- 4MP is standardizing a **companion schema** that will accompany each part file,
  capturing this kind of metadata (units, tessellation tolerance, and other
  part-level info, including surface material/BRDF). Schema details TBD —
  V1's part object uses a minimal placeholder shape in the meantime (see
  "Part object" above).

## Roadmap / phasing

- **Phase 1 (current focus):** Measurement + reconstruction engines only, no
  injection, single face at a time (not the top face). Test part is a cube
  (STL), one side face, one pose. Validate via direct comparison to
  analytically-known ground truth from that face, using both pointwise and
  spectral metrics, under the idealized sensor model (binary-intensity
  inverse-pinhole projector, 188×319mm measurement area, 27° half-angle,
  470mm working distance, 1600-scan-step orientation, no Zemax dependency).
  Output in mm, on a camera-pixel-native grid, signed deviation from a
  reference plane, origin at grid center, matching the Unity pipeline's
  height map contract.
- **Phase 2:** Multi-face/multi-pose parts (pose stitching question
  resurfaces); add sensor fidelity in the order captured in "Measurement
  engine" above (Zemax chief-ray/PSF interpolation → BRDF → camera optics/
  Scheimpflug defocus → polynomial field-angle calibration), plus occlusion
  handling; adopt the real companion schema once it exists; possibly revisit
  cutting/correction engines or error injection if scope changes again.
- Details still open (to refine incrementally, including once in Claude Code):
  - **Full Z measurement range** — genuinely unresolved pending further
    optical modeling and bench testing; not a V1 blocker.
  - **Pose-stitching ownership** — deferred given single-face V1 scope.
  - Concrete part object schema/serialization (types, file formats per
    field) — using a minimal placeholder now, will be replaced once the real
    companion schema lands.
  - Occlusion implementation (design captured above, not yet built).
  - Camera FOV margin over the measurement area — design intent confirmed,
    exact margin TBD.
  - Fidelity-roadmap implementations, in order: Zemax chief-ray/PSF +
    interpolation, BRDF (Lambertian then specular), camera distortion +
    Scheimpflug defocus, polynomial field-angle calibration model (all
    design captured above, none built yet).

## Engine interface contracts (current scope)

- Measurement engine: face (mesh + BRDF), already posed into sensor space
  (see "Part/face pose") + sensor config (188×319mm measurement area, 27°
  half-angle, 1600-scan-step orientation, ray/projector model, camera
  calibration) → raw scan data (stack of simulated line images for that pose
  — which pixels lit up, per scan step; binary intensity in V1).
- Reconstruction engine: raw scan data + the same ray/projector model and
  camera calibration → height map for that pose (mm, camera-pixel-native 2D
  array, signed deviation from the sensor's reference plane, origin at grid
  center, matching the Unity output format/convention).

**[OUT OF SCOPE]** Correction engine and cutting engine interface contracts —
preserved above under their respective "OUT OF SCOPE" sections.

## V1.2 implementation notes (2026-08-08)

Part-frame up-axis convention: default assumed up-axis for an ingested STL
changed from +Y (an undocumented placeholder that only the symmetric cube
fixture happened to satisfy) to +Z, matching CNC/CAM convention. A
`--up-axis X Y Z` CLI flag (`load_part`'s `up_axis` parameter) overrides it
for files authored differently. Implemented as one fixed change-of-basis
(`part.py`'s `_remap_up_axis`, via `geometry.py`'s general
`axis_alignment_rotation` -- Rodrigues' formula, handling the antiparallel
degenerate case explicitly since the default case (+Z file-up onto internal
"physical up") is exactly antiparallel) applied to the mesh once at
ingestion; everything downstream (face selection, T_mount, theta_face) is
unchanged and written against the internal convention.

The cube fixture is symmetric under axis relabeling, so it can't exercise
this at all (this is literally how the bug went unnoticed -- surfaced only
by testing a non-cube part). `test_part.py` covers the remap with a
deliberately asymmetric plate fixture instead.

## V1.3 implementation notes: Coordinate frame convention (2026-08-08)

Adopted 4MP's cell-wide `coordinate_transforms_equations_v3.md` /
`transform_point_cloud_v3.py` convention in place of this package's own ad
hoc sensor frame (see "Part/face pose" above), which had drifted from it.
Three changes, all implemented:

**1. Relabeled the internal sensor/optics frame to O_s.** Pure change of
basis (a proper rotation -- permutation + sign flips, not a mirror), applied
throughout `geometry.py`/`pinhole.py`/`sensor_config.py`/`measurement.py`/
`reconstruction.py`:

```
X_(O_s) = -Z_ours     Y_(O_s) = +X_ours     Z_(O_s) = -Y_ours
```

O_s: X = depth/boresight (depth = -X, an artifact of the doc's image-sensor-
side drawing convention), Y = baseline/lateral, Z = the DMD line axis *and*
the rotary stage's rotation axis (physical "up" = -Z, matching the doc's
"O_s: Z down"). Concretely: projector at (0, -baseline/2, 0), camera at
(0, +baseline/2, 0), reference point at (-working_distance, 0, 0).

None of the ray-tracing/triangulation math needed to change -- it was
already generic vector algebra with no hardcoded axis assumptions, *except*
one: `PinholeModel.looking_at`'s default `world_up` (used only to build an
orthonormal local basis) had to change from `(0,1,0)` to `(0,0,-1)`, the
same physical direction carried through the axis mapping. Verified
algebraically (both by transforming the old basis vectors through the
mapping and by recomputing the projector's local basis from scratch in O_s)
before touching the code.

**2. Introduced an explicit O_r (rotary table) frame.** `part.py`'s
`compute_pose` now returns a named `o_r_from_o_s` (`T(O_r <- O_s)`)
alongside `pose`, rather than folding everything into one opaque part-to-
sensor composition. `Transform` gained `.invert()` and `.to_quat_trans()`
(scalar-last quaternion, matching `transform_point_cloud_v3.py`'s
convention) rather than a parallel implementation.

Getting this right took a wrong turn worth recording. **The rotation used
to derive `T(O_s <- O_r)` is a *fixed* quarter turn (`rotation_z(pi/2)`),
not `rotation_z(theta_face)`** -- easy to conflate since both are yaws about
the same shared Z axis, and reusing `theta_face` (the rotation that aligns
*this face's normal* with the boresight, needed for `pose`) is the natural
first guess. It's wrong: `theta_face` depends on which face is selected,
but O_r's own axes relative to the mounted (T_mount-applied) frame don't.
Proof sketch: define O_r's Y axis (in mounted-frame coordinates) as the face
normal, Z as the shared rotation axis, X completing a right-handed basis;
composing that per-face basis change with `rotation_z(theta_face)` gives,
for *any* in-plane normal, exactly `rotation_z(pi/2)` -- the theta-
dependence cancels out algebraically. Verified numerically against several
normals during debugging.

The tell was the regridded output: the naive (`theta_face`-based) version
put the flat test face's actual ~100mm in-plane extent onto what was
supposed to be the near-constant height axis, and the near-constant depth
onto one of the in-plane axes -- a `3 x 1512`-cell grid instead of a sane
`~1512 x 1512` square, and a height "map" ranging over 100mm instead of
sitting at ~50mm +/- float noise. Both are now covered by regression tests
(`test_part.py`'s `test_o_r_from_o_s_puts_height_on_y_and_extent_on_x_and_z`,
`test_v1_milestone.py`'s square-grid assertion).

**3. Height maps/plots now live in O_r, folded into V1.1's regridding.**
`regrid.py`'s grid is O_r's (X, Z) plane (X horizontal, Z vertical); the
binned value is O_r's Y component (`part.py`'s `height_in_o_r`, replacing
the earlier ad hoc `SensorConfig.height_from_point`'s "Z - working_distance"
shortcut, which stopped being meaningful once the frame was relabeled and
was removed outright rather than left as dead/misleading code). Ground
truth is sampled directly on the same O_r grid by casting a ray along O_r's
+Y through each cell's (X, Z), transformed into O_s to actually trace it
against the posed mesh -- not by back-projecting camera pixels, which would
reintroduce the keystone this is meant to avoid. Height is therefore a
function of a specific face/pose's `o_r_from_o_s`, not a fixed `SensorConfig`
property -- `run_reconstruction`, `regrid_reconstruction_and_truth`, and
`validation.py`'s pixel-native ground-truth helpers all take it as a
parameter now.

**Not changed:** `O_t` (cutting tool) -- no cutting engine in scope. The
camera-pixel-native diagnostic view (`HeightMapResult.height_map`/
`triangulation_gap_mm`) is unchanged in spirit (still per-camera-pixel,
still kept per the V1.1 note) other than sourcing its heights through the
same `height_in_o_r` call.
