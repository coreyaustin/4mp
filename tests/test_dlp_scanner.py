import numpy as np

from fourmp.dlp_scanner import Camera, Source, SUT
from fourmp.dlp_scanner.centroid import centroid_rows
from fourmp.dlp_scanner.geometry import look_at_pose, ray_plane_intersection, rotation_about_z
from fourmp.dlp_scanner.pipeline import acquire_line_scan, reverse_pipeline


def test_look_at_pose_is_orthonormal_and_faces_target():
    pose = look_at_pose(position=(10.0, 0.0, 5.0), target=(0.0, 0.0, 0.0))
    basis = np.stack([pose.right, pose.up, pose.forward])
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-9)

    to_target = -pose.position / np.linalg.norm(pose.position)
    np.testing.assert_allclose(pose.forward, to_target, atol=1e-9)


def test_rotation_about_z_preserves_z_and_rotates_xy():
    rotated = rotation_about_z(np.pi / 2) @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)


def test_ray_plane_intersection_hits_expected_point():
    point = ray_plane_intersection(
        np.array([0.0, 0.0, 10.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, 2.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(point, [0.0, 0.0, 2.0], atol=1e-9)


def test_flat_sut_intersect_ray_recovers_known_height():
    sut = SUT.flat(half_extent_mm=20.0, resolution=41, z0=3.5)
    sut_world = sut.transform(table_angle_rad=0.0)

    hit = sut_world.intersect_ray(
        ray_origin_world=np.array([1.0, -2.0, 50.0]),
        ray_dir_world=np.array([0.0, 0.0, -1.0]),
    )

    assert hit is not None
    np.testing.assert_allclose(hit.world_point, [1.0, -2.0, 3.5], atol=1e-2)


def test_line_scan_reconstructs_flat_height_from_a_single_row():
    # Resolution/baseline/standoff need to give a reasonably steep triangulation
    # angle and fine-enough angular pixel pitch: depth is recovered from where the
    # per-column intensity centroid falls, so a coarse or shallow rig still limits
    # precision. At the SUT's default Phong parameters (shininess=20) and this
    # rig's narrow 18-degree vertical FOV over 60 rows, the specular lobe is very
    # broad — its peak sits close to the camera's own optical-center row rather
    # than sharply tracking each hit's true reflection point, so centroiding here
    # doesn't yet meaningfully beat the old integer-pixel-quantization precision
    # (verified: max error stays ~0.8mm at these defaults; cranking shininess into
    # the thousands sharpens the lobe around that same biased peak and makes
    # triangulation *worse*, not better — a property of the `view_dir_world ≈
    # -pixel_direction_world(row, col)` approximation, not a bug). These numbers
    # match the demo script's working configuration.
    resolution = (60, 64)
    source = Source.at(position=(-130.0, 0.0, 320.0), resolution=resolution)
    camera = Camera.at(position=(130.0, 0.0, 320.0), resolution=resolution)
    sut = SUT.flat(half_extent_mm=20.0, resolution=41, z0=4.0)

    frames = acquire_line_scan(source, sut, table_angle_rad=0.0, camera=camera, rows=[resolution[0] // 2])
    points = reverse_pipeline(frames, source, camera)

    assert points.size > 0
    np.testing.assert_allclose(points[:, 2], 4.0, atol=1.0)


def test_centroid_rows_recovers_known_analytic_centroid():
    frame = np.zeros((10, 3))
    frame[3, 1] = 1.0
    frame[4, 1] = 3.0
    frame[5, 1] = 1.0
    # weighted mean of rows [3, 4, 5] with weights [1, 3, 1] -> (3+12+5)/5 = 4.0
    centroids = centroid_rows(frame)
    assert np.isnan(centroids[0])
    np.testing.assert_allclose(centroids[1], 4.0, atol=1e-9)
    assert np.isnan(centroids[2])


def test_point_sampling_limit_matches_pre_windowing_behavior():
    # k_specular=0, k_diffuse=1 reduces the Phong lobe to plain Lambertian; with
    # cone_pixel_radius=0 the accumulation window collapses to a single pixel per
    # hit — together this should reproduce the pipeline's original point-sampling
    # behavior (one shaded pixel per source ray, no spreading).
    resolution = (60, 64)
    source = Source.at(position=(-130.0, 0.0, 320.0), resolution=resolution)
    camera = Camera.at(position=(130.0, 0.0, 320.0), resolution=resolution, cone_pixel_radius=0)
    sut = SUT.flat(half_extent_mm=20.0, resolution=41, z0=4.0, k_specular=0.0, k_diffuse=1.0)

    row, frame = acquire_line_scan(source, sut, table_angle_rad=0.0, camera=camera, rows=[resolution[0] // 2])[0]

    assert np.count_nonzero(frame) > 0
    # every lit pixel sits on its own row -- no window spread means neighbors stay dark
    lit_rows, lit_cols = np.nonzero(frame)
    for r, c in zip(lit_rows, lit_cols):
        if r > 0:
            assert frame[r - 1, c] == 0.0
        if r < frame.shape[0] - 1:
            assert frame[r + 1, c] == 0.0


def test_source_optics_derive_fov_magnification_and_focus_spot():
    # Locked hardware numbers from source_optics_spec.md: DLP670S, 23mm lens,
    # 470.5mm working distance -> fov~=(33.7, 20.3) deg, M~=19.5x, spot~=0.105mm.
    source = Source.at(position=(-130.0, 0.0, 470.5))
    np.testing.assert_allclose(source.fov_deg, (33.7, 20.3), atol=0.1)
    assert abs(source.magnification - 19.5) < 0.1
    assert abs(source.focus_spot_diameter_mm - 0.105) < 0.001


def test_source_footprint_diameter_symmetric_and_decoupled():
    source = Source.at(position=(-130.0, 0.0, 470.5))

    # At the nominal working distance, footprint == focus_spot_diameter_mm exactly.
    np.testing.assert_allclose(source.footprint_diameter_mm(0.0), source.focus_spot_diameter_mm, atol=1e-12)

    # Growth is symmetric in |defocus_distance_mm| by construction (caller abs()'s it).
    assert source.footprint_diameter_mm(30.0) > source.footprint_diameter_mm(0.0)

    # Changing defocus_half_angle_deg changes growth but not the at-focus value.
    steeper = Source.at(position=(-130.0, 0.0, 470.5), defocus_half_angle_deg=0.5)
    np.testing.assert_allclose(steeper.footprint_diameter_mm(0.0), source.footprint_diameter_mm(0.0), atol=1e-12)
    assert steeper.footprint_diameter_mm(30.0) > source.footprint_diameter_mm(30.0)

    # Changing focal_length_mm changes magnification/focus_spot_diameter_mm but not
    # defocus_half_angle_deg -- the two knobs stay decoupled.
    different_lens = Source.at(position=(-130.0, 0.0, 470.5), focal_length_mm=35.0)
    assert different_lens.magnification != source.magnification
    assert different_lens.defocus_half_angle_deg == source.defocus_half_angle_deg


def test_windowed_accumulation_produces_roughly_constant_width_fuzzy_line():
    resolution = (60, 64)
    source = Source.at(position=(-130.0, 0.0, 320.0), resolution=resolution)
    camera = Camera.at(position=(130.0, 0.0, 320.0), resolution=resolution, cone_pixel_radius=2)
    sut = SUT.flat(half_extent_mm=20.0, resolution=41, z0=4.0)

    row, frame = acquire_line_scan(source, sut, table_angle_rad=0.0, camera=camera, rows=[resolution[0] // 2])[0]

    widths = [np.count_nonzero(frame[:, col]) for col in range(frame.shape[1]) if np.any(frame[:, col])]
    assert widths, "expected at least one illuminated column"
    assert max(widths) - min(widths) <= 1  # roughly constant width across columns

    centroids = centroid_rows(frame)
    assert np.count_nonzero(~np.isnan(centroids)) == len(widths)
