"""Bounding-box validation and crop geometry."""
import numpy as np
import pytest

from src.video.geometry import BBox, BBoxError, clamp_bbox, crop_target, iou, validate_bbox, visible_fraction

W, H = 160, 120


class TestValidateBBox:
    def test_valid_box_returned_unchanged(self):
        assert validate_bbox(10, 20, 30, 40, W, H) == BBox(10, 20, 30, 40)

    def test_box_touching_frame_edge_is_valid(self):
        assert validate_bbox(W - 20, H - 20, 20, 20, W, H) == BBox(W - 20, H - 20, 20, 20)

    @pytest.mark.parametrize("args,needle", [
        ((-1, 10, 30, 30), "negative"),
        ((10, -5, 30, 30), "negative"),
        ((10, 10, 0, 30), "too small"),
        ((10, 10, 30, 7), "too small"),
        ((W, 10, 20, 20), "outside"),
        ((10, H, 20, 20), "outside"),
        ((W - 10, 10, 20, 20), "exceeds frame width"),
        ((10, H - 10, 20, 20), "exceeds frame height"),
        ((10, 10, 500, 30), "exceeds frame width"),
    ])
    def test_invalid_boxes_rejected_with_specific_reason(self, args, needle):
        with pytest.raises(BBoxError, match=needle):
            validate_bbox(*args, W, H)

    def test_box_is_never_clamped_or_repaired(self):
        with pytest.raises(BBoxError):
            validate_bbox(150, 10, 30, 30, W, H)   # would be "fixable" by clamping; must not be

    @pytest.mark.parametrize("bad", [1.5, "10", None, True])
    def test_non_integer_coordinates_rejected(self, bad):
        with pytest.raises(BBoxError, match="integer"):
            validate_bbox(bad, 10, 30, 30, W, H)

    def test_invalid_frame_size_rejected(self):
        with pytest.raises(BBoxError):
            validate_bbox(0, 0, 10, 10, 0, 100)


class TestCrops:
    def test_crop_is_letterboxed_to_requested_size_and_keeps_aspect(self):
        frame = np.zeros((H, W, 3), np.uint8)
        frame[40:80, 50:60] = (0, 0, 255)                      # tall, narrow 10x40 red target
        crop = crop_target(frame, BBox(50, 40, 10, 40), (112, 112), context_pad=0.0)
        assert crop.shape == (112, 112, 3)
        red = (crop[..., 2] > 200)
        ys, xs = np.where(red)
        # aspect preserved: red region is ~4x taller than wide, not stretched square
        assert (ys.max() - ys.min() + 1) / (xs.max() - xs.min() + 1) == pytest.approx(4.0, rel=0.1)
        assert crop[:, :20].sum() == 0 and crop[:, -20:].sum() == 0   # letterbox bars are black

    def test_crop_contains_only_padded_box_pixels(self):
        frame = np.zeros((H, W, 3), np.uint8)
        frame[:, :] = (0, 255, 0)                              # everything green...
        frame[50:70, 50:70] = (255, 0, 0)                      # ...except the target (blue)
        crop = crop_target(frame, BBox(50, 50, 20, 20), (64, 64), context_pad=0.1)
        blue_frac = float(((crop[..., 0] > 200) & (crop[..., 1] < 50)).mean())
        # 20px box + 2px pad each side => (20/24)^2 ~ 0.69 of the crop is target
        assert blue_frac == pytest.approx((20 / 24) ** 2, abs=0.05)

    def test_crop_of_box_partly_outside_frame_uses_clamped_region(self):
        frame = np.full((H, W, 3), 128, np.uint8)
        crop = crop_target(frame, BBox(W - 10, 10, 30, 30), (32, 32), 0.0)
        assert crop.shape == (32, 32, 3) and crop.any()

    def test_crop_with_no_overlap_raises(self):
        with pytest.raises(BBoxError):
            crop_target(np.zeros((H, W, 3), np.uint8), BBox(W + 10, 10, 20, 20), (32, 32))


class TestVisibility:
    def test_fully_inside(self):
        assert visible_fraction(BBox(10, 10, 20, 20), W, H) == 1.0

    def test_half_outside(self):
        assert visible_fraction(BBox(W - 10, 10, 20, 20), W, H) == pytest.approx(0.5)

    def test_fully_outside(self):
        assert visible_fraction(BBox(W + 5, 10, 20, 20), W, H) == 0.0

    def test_clamp_and_iou(self):
        assert clamp_bbox(BBox(-5, -5, 20, 20), W, H) == BBox(0, 0, 15, 15)
        assert clamp_bbox(BBox(W + 1, 0, 5, 5), W, H) is None
        assert iou(BBox(0, 0, 10, 10), BBox(0, 0, 10, 10)) == 1.0
        assert iou(BBox(0, 0, 10, 10), BBox(20, 20, 10, 10)) == 0.0
