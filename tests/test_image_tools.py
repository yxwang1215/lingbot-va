"""Regression tests for resize-with-padding inference image preprocessing."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_IMAGE_TOOLS = (
    Path(__file__).resolve().parents[1]
    / "wan_va/utils/Simple_Remote_Infer/deploy/image_tools.py"
)
_spec = importlib.util.spec_from_file_location("image_tools", _IMAGE_TOOLS)
image_tools = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(image_tools)


@pytest.mark.parametrize("image_height,image_width", [(1, 512), (512, 1)])
def test_extremely_narrow_image_keeps_nonzero_resized_dimension(
    image_height, image_width
):
    image = np.full((1, image_height, image_width, 3), 173, dtype=np.uint8)
    output = image_tools.resize_with_pad(image, height=256, width=256)

    assert output.shape == (1, 256, 256, 3)
    assert output.dtype == np.uint8
    assert np.any(output == 173)
    assert np.any(output == 0)


def test_regular_aspect_ratio_resizes_and_pads():
    image = np.full((1, 32, 64, 3), 173, dtype=np.uint8)
    output = image_tools.resize_with_pad(image, height=64, width=64)

    assert output.shape == (1, 64, 64, 3)
    np.testing.assert_array_equal(output[0, 16:48], 173)
    np.testing.assert_array_equal(output[0, 0], 0)


def test_matching_size_returns_input_unchanged():
    image = np.full((2, 16, 16, 3), 173, dtype=np.uint8)
    output = image_tools.resize_with_pad(image, height=16, width=16)

    assert output is image
