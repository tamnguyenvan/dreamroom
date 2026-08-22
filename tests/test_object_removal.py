"""Tests for square selected-object crop and patch compositing."""

from __future__ import annotations

import numpy as np
import pytest

from dreamroom.object_removal import (
    annotate_selected_object,
    crop_selected_object,
    resize_patch,
    stitch_patch,
)


def test_crop_is_unpadded_square_of_shorter_image_side():
    image = np.arange(5 * 8 * 3, dtype=np.uint8).reshape(5, 8, 3)
    mask = np.zeros((5, 8), dtype=bool)
    mask[1:3, 4:6] = True

    crop = crop_selected_object(image, mask)

    assert crop.size == 5
    assert crop.image_bgr.shape == (5, 5, 3)
    assert (crop.x, crop.y) == (2, 0)
    np.testing.assert_array_equal(crop.image_bgr, image[0:5, 2:7])
    assert crop.to_dict()["object_bbox_xyxy"] == [4, 1, 6, 3]


def test_crop_rejects_empty_or_too_large_selection():
    image = np.zeros((5, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="mask is empty"):
        crop_selected_object(image, np.zeros((5, 8), dtype=bool))

    mask = np.zeros((5, 8), dtype=bool)
    mask[:, :8] = True
    with pytest.raises(ValueError, match="does not fit any supported crop aspect ratio"):
        crop_selected_object(image, mask)


def test_crop_falls_back_to_best_supported_ratio_for_wide_selection():
    image = np.zeros((9, 16, 3), dtype=np.uint8)
    mask = np.zeros((9, 16), dtype=bool)
    mask[1:8, 2:14] = True

    crop = crop_selected_object(image, mask)

    assert crop.aspect_ratio == "16:9"
    assert crop.image_bgr.shape == (9, 16, 3)
    assert crop.to_dict()["size_hw"] == [9, 16]


def test_resize_and_stitch_patch():
    image = np.zeros((5, 8, 3), dtype=np.uint8)
    mask = np.zeros((5, 8), dtype=bool)
    mask[1:3, 4:6] = True
    crop = crop_selected_object(image, mask)

    resized = resize_patch(
        np.full((2, 2, 3), 9, dtype=np.uint8),
        (crop.width, crop.height),
    )
    stitched = stitch_patch(image, resized, crop)

    assert resized.shape == (5, 5, 3)
    np.testing.assert_array_equal(stitched[0:5, 2:7], 9)
    np.testing.assert_array_equal(stitched[:, :2], 0)
    np.testing.assert_array_equal(stitched[:, 7:], 0)


def test_annotate_selected_object_draws_red_mask_and_outline():
    image = np.zeros((9, 9, 3), dtype=np.uint8)
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:7] = True

    annotated = annotate_selected_object(image, mask)

    assert annotated.shape == image.shape
    assert annotated[4, 4, 2] > 80
    assert annotated[4, 4, 1] < 50
    assert np.all(annotated[2, 4] == [0, 0, 255])
    assert np.all(annotated[0, 0] == 0)
