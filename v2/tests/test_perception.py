"""Perception: what a frame says before anything reasons about it.

Camera motion and frame quality were both absent from v1 and v2, so every test
here covers behaviour the product did not have. The scenes are synthetic,
which bounds what they prove: they show that a known translation is recovered
and that a deliberately ruined frame is refused, not that a real mast in real
wind behaves the way the estimator expects.
"""

import cv2
import numpy as np
import pytest

from vigil.domain.appearance import (
    MAX_APPEARANCE_DISTANCE, MAX_REIDENTIFY_DISTANCE, Appearance, Gallery,
)
from vigil.perception.appearance import describe
from vigil.perception.motion import CameraMotion, CameraMotionEstimator
from vigil.perception.quality import BLUR_FLOOR, FrameQualityMonitor


def _textured(size=(360, 640), seed=1):
    """A frame with real texture in it, which is what optical flow needs."""
    rng = np.random.default_rng(seed)
    base = rng.integers(20, 235, size=(size[0] // 8, size[1] // 8, 3), dtype=np.uint8)
    return cv2.resize(base, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)


def _shift(image, dx_px, dy_px):
    matrix = np.array([[1.0, 0.0, dx_px], [0.0, 1.0, dy_px]], dtype=np.float32)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                          borderMode=cv2.BORDER_REFLECT)


# ------------------------------------------------------------ camera motion


def test_the_first_frame_of_a_run_reports_no_measurement_rather_than_no_motion():
    estimator = CameraMotionEstimator()
    motion = estimator.estimate(_textured())
    assert not motion.measured and motion.fault == "no previous frame"
    # The identity is there so a caller composing transforms need not special
    # case it — never so that "unknown" can be read as "did not move".
    assert np.allclose(motion.warp, [[1, 0, 0], [0, 1, 0]])
    assert not motion.still


def test_a_known_pan_is_recovered_to_within_a_pixel():
    image = _textured()
    height, width = image.shape[:2]
    estimator = CameraMotionEstimator()
    estimator.estimate(image)
    for dx_px, dy_px in ((12, 0), (0, -9), (7, 5), (-20, 3)):
        estimator.reset()
        estimator.estimate(image)
        motion = estimator.estimate(_shift(image, dx_px, dy_px))
        assert motion.measured, motion.fault
        got_x, got_y = motion.shift
        assert abs(got_x * width - dx_px) < 1.5, f"{got_x * width:.2f} px vs {dx_px}"
        assert abs(got_y * height - dy_px) < 1.5, f"{got_y * height:.2f} px vs {dy_px}"
        assert motion.inlier_ratio > 0.8
        assert not motion.still


def test_the_warp_is_normalised_so_a_resolution_change_does_not_break_it():
    # A warp in pixels silently stops being right when a stream reconnects at
    # a different size. Two frames of the same scene at two sizes, shifted by
    # the same *fraction*, must produce the same warp.
    tall = _textured((360, 640))
    small = cv2.resize(tall, (320, 180))
    warps = []
    for image in (tall, small):
        estimator = CameraMotionEstimator()
        estimator.estimate(image)
        motion = estimator.estimate(_shift(image, image.shape[1] * 0.02, 0))
        assert motion.measured, motion.fault
        warps.append(motion.shift[0])
    assert abs(warps[0] - warps[1]) < 0.005, f"{warps[0]:.4f} vs {warps[1]:.4f}"


def test_a_still_camera_is_reported_as_still_rather_than_as_drifting():
    image = _textured()
    estimator = CameraMotionEstimator()
    estimator.estimate(image)
    motion = estimator.estimate(image.copy())
    assert motion.measured and motion.still
    assert motion.magnitude < 0.0025


def test_a_rotation_is_measured_as_a_rotation():
    image = _textured()
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 3.0, 1.0)
    turned = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    estimator = CameraMotionEstimator()
    estimator.estimate(image)
    motion = estimator.estimate(turned)
    assert motion.measured, motion.fault
    assert abs(abs(motion.rotation_degrees) - 3.0) < 0.5, motion.rotation_degrees
    assert abs(motion.scale - 1.0) < 0.02


def test_a_frame_with_nothing_to_track_refuses_rather_than_inventing_a_transform():
    blank = np.full((240, 320, 3), 128, dtype=np.uint8)
    estimator = CameraMotionEstimator()
    estimator.estimate(blank)
    motion = estimator.estimate(blank.copy())
    assert not motion.measured
    assert "corners" in motion.fault or "round trip" in motion.fault


def test_one_lorry_crossing_the_frame_is_not_the_camera_moving():
    """The failure that matters. A large object moving across a static scene
    produces a consistent transform for a minority of the frame; applying it
    would drag every track sideways after the lorry."""
    image = _textured()
    height, width = image.shape[:2]
    moved = image.copy()
    # A textured block occupying about a fifth of the frame, translating.
    block = _textured((height // 3, width // 3), seed=9)
    y0, x0 = height // 3, width // 8
    image[y0:y0 + block.shape[0], x0:x0 + block.shape[1]] = block
    moved[y0:y0 + block.shape[0], x0 + 30:x0 + 30 + block.shape[1]] = block
    estimator = CameraMotionEstimator()
    estimator.estimate(image)
    motion = estimator.estimate(moved)
    # Either it refuses, or it measures a shift far smaller than the block's.
    if motion.measured:
        assert abs(motion.shift[0] * width) < 8.0, (
            f"the lorry was read as the camera moving {motion.shift[0] * width:.1f} px"
        )


def test_a_cut_or_a_slew_is_refused_rather_than_applied():
    image = _textured()
    estimator = CameraMotionEstimator()
    estimator.estimate(image)
    motion = estimator.estimate(_shift(image, image.shape[1] * 0.4, 0))
    assert not motion.measured, "warping every track by 40% of a frame is worse than admitting defeat"
    assert motion.fault, "a refusal has to say why"


def test_the_plausibility_ceiling_refuses_a_shift_no_shake_produces():
    from vigil.perception.motion import MAX_PLAUSIBLE_SHIFT

    # Directly, because getting optical flow to succeed *and* report an
    # implausible shift needs a contrived scene, and the guard is the point.
    huge = CameraMotion(np.array([[1.0, 0.0, MAX_PLAUSIBLE_SHIFT * 2], [0.0, 1.0, 0.0]]),
                        True, 50, 50, None, 16 / 9)
    assert huge.magnitude > MAX_PLAUSIBLE_SHIFT
    small = CameraMotion(np.array([[1.0, 0.0, 0.01], [0.0, 1.0, 0.0]]), True, 50, 50, None, 16 / 9)
    assert small.magnitude < MAX_PLAUSIBLE_SHIFT and not small.still


def test_a_resolution_change_mid_stream_is_not_compared_across():
    estimator = CameraMotionEstimator()
    estimator.estimate(_textured((360, 640)))
    motion = estimator.estimate(_textured((240, 320)))
    # Either the size guard fires, or the working resize made them the same
    # size and the scene genuinely does not match; both refuse.
    assert not motion.measured


# ------------------------------------------------------------ frame quality


def test_a_good_frame_is_usable_and_says_what_it_measured():
    monitor = FrameQualityMonitor()
    quality = monitor.measure(_textured())
    assert quality.usable, quality.describe()
    assert quality.sharpness > BLUR_FLOOR
    assert quality.score > 0.3
    assert "sharpness" in quality.describe()


def test_an_out_of_focus_frame_is_refused_and_named():
    monitor = FrameQualityMonitor()
    blurred = cv2.GaussianBlur(_textured(), (0, 0), 12)
    quality = monitor.measure(blurred)
    assert not quality.usable
    assert any("focus" in fault for fault in quality.faults), quality.faults
    assert quality.score == 0.0


def test_a_blown_out_frame_is_refused_and_named():
    monitor = FrameQualityMonitor()
    glare = _textured()
    glare[:, : glare.shape[1] // 2] = 255
    quality = monitor.measure(glare)
    assert not quality.usable
    assert any("blown out" in fault for fault in quality.faults), quality.faults


def test_a_lens_cap_is_refused_and_named():
    monitor = FrameQualityMonitor()
    quality = monitor.measure(np.zeros((240, 320, 3), dtype=np.uint8))
    assert not quality.usable
    assert any("nothing in the frame" in f or "crushed" in f for f in quality.faults), quality.faults


def test_a_frozen_stream_is_noticed_where_a_frame_counter_would_not_be():
    """Some IP cameras hold the last frame rather than dropping the
    connection, so the decoder is happy, the frame counter climbs, the fps
    looks healthy, and the picture is an hour old."""
    monitor = FrameQualityMonitor()
    image = _textured()
    first = monitor.measure(image)
    assert first.change is None, "there is nothing to compare the first frame to"
    second = monitor.measure(image.copy())
    # The *image* is fine — it is a picture, just an old one — so a detector
    # should still run over it. What it is not is a frame worth sampling into
    # a map, and what the camera is is degraded.
    assert second.usable and second.faults == ()
    assert second.stale and not second.worth_sampling
    assert second.degraded and "repeating itself" in second.degraded
    assert second.score == 0.0


def test_a_static_scene_from_a_real_sensor_is_not_called_frozen():
    # The distinction the threshold exists for: read noise is what separates a
    # still scene from a stalled decoder.
    monitor = FrameQualityMonitor()
    rng = np.random.default_rng(3)
    image = _textured()
    monitor.measure(image)
    noisy = np.clip(image.astype(np.int16) + rng.integers(-4, 5, image.shape), 0, 255).astype(np.uint8)
    quality = monitor.measure(noisy)
    assert quality.usable and not quality.stale, quality.describe()
    assert quality.worth_sampling


def test_the_rolling_score_is_what_a_health_check_should_read():
    monitor = FrameQualityMonitor()
    assert monitor.recent_score is None
    for _ in range(5):
        monitor.measure(_textured(seed=np.random.randint(1000)))
    good = monitor.recent_score
    assert good is not None and good > 0.0
    monitor.reset()
    assert monitor.recent_score is None


# --------------------------------------------------------------- appearance


def test_two_crops_of_the_same_coat_are_close_and_two_coats_are_not():
    frame = np.full((240, 320, 3), 30, dtype=np.uint8)
    frame[100:200, 40:80] = (200, 40, 40)     # a red coat
    frame[100:200, 200:240] = (40, 40, 200)   # a blue one
    red = describe(frame, (40 / 320, 100 / 240, 40 / 320, 100 / 240))
    red_again = describe(frame, (41 / 320, 101 / 240, 40 / 320, 100 / 240))
    blue = describe(frame, (200 / 320, 100 / 240, 40 / 320, 100 / 240))
    assert red.usable and blue.usable
    assert red.distance(red_again) < 0.05
    # Two coats have to be separable by the gate that *decides* — the one
    # re-identification uses, where appearance is all the evidence there is.
    # The association gate is deliberately looser than this: inside a frame
    # geometry decides and appearance only shades, and a gate tight enough to
    # separate two coats there vetoed correct pairs and cost identity
    # switches. Measured separation on this scene: 0.56.
    separation = red.distance(blue)
    assert separation > MAX_REIDENTIFY_DISTANCE, separation
    assert MAX_REIDENTIFY_DISTANCE < MAX_APPEARANCE_DISTANCE, (
        "the two gates do different jobs and the re-identification one is the tight one"
    )


def test_a_crop_too_small_to_describe_says_so_rather_than_describing_noise():
    frame = np.full((240, 320, 3), 30, dtype=np.uint8)
    tiny = describe(frame, (0.5, 0.5, 0.004, 0.004))
    assert not tiny.usable
    assert tiny.distance(tiny) == float("inf"), "an unusable descriptor matches nothing"


def test_a_mask_keeps_the_background_out_of_the_descriptor():
    """A box around a person is mostly not the person. Two people in front of
    the same wall must not look alike because of the wall.

    The person is deliberately *narrow* inside the box. With a wide subject
    the centre-band fallback already excludes the background and the mask
    changes nothing — which is why the fallback exists, and why this test has
    to construct the case where it is not enough.
    """
    wall = np.full((240, 320, 3), 200, dtype=np.uint8)
    a, b = wall.copy(), wall.copy()
    a[130:170, 112:122] = (30, 30, 160)
    b[130:170, 112:122] = (30, 160, 30)
    box = (95 / 320, 110 / 240, 40 / 320, 80 / 240)
    mask = np.zeros((80, 40), dtype=np.uint8)
    mask[20:60, 17:27] = 1
    unmasked = describe(a, box).distance(describe(b, box))
    masked = describe(a, box, mask).distance(describe(b, box, mask))
    assert masked > unmasked * 1.5, (
        f"the wall dominated the descriptor: masked {masked:.3f} vs unmasked {unmasked:.3f}"
    )


def test_a_gallery_keeps_several_looks_and_matches_the_closest():
    """A track's gallery holds a person's front and their back, and the mean
    of those two is a person who does not exist and matches neither."""
    front = Appearance(np.array([1.0, 0.0, 0.0], dtype=np.float32), 500)
    back = Appearance(np.array([0.0, 1.0, 0.0], dtype=np.float32), 500)
    gallery = Gallery()
    gallery.observe(front)
    gallery.observe(back)
    assert len(gallery) == 2
    assert gallery.distance(front) < 1e-6
    assert gallery.distance(back) < 1e-6
    assert gallery.distance(Appearance(np.array([0.0, 0.0, 1.0], dtype=np.float32), 500)) > 0.9


def test_a_gallery_is_bounded_and_refuses_a_descriptor_of_nothing():
    gallery = Gallery(depth=3)
    for i in range(10):
        vector = np.zeros(3, dtype=np.float32)
        vector[i % 3] = 1.0
        gallery.observe(Appearance(vector, 500))
    assert len(gallery) == 3
    gallery.observe(Appearance(np.zeros(3, dtype=np.float32), 2))
    assert len(gallery) == 3, "40 pixels of noise does not become useful by being stored"
    assert not Gallery().usable
