from __future__ import annotations

import numpy as np
import pytest

from garbage_auto_annotation.conversion import (
    format_yolo_segment,
    mask_to_polygons,
    normalize_polygon,
)


def test_normalize_polygon_clamps_coordinates() -> None:
    result = normalize_polygon(
        [(-1.0, 0.0), (10.0, 0.0), (10.0, 20.0)], width=10, height=20
    )

    assert result == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]


def test_normalize_polygon_rejects_invalid_image_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        normalize_polygon([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], 0, 10)


def test_mask_to_polygons_preserves_small_valid_component() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:10] = 1

    polygons = mask_to_polygons(mask, min_area=16)

    assert len(polygons) == 1
    assert len(set(polygons[0])) >= 3


def test_mask_to_polygons_drops_component_below_threshold() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:7, 5:7] = 1

    assert mask_to_polygons(mask, min_area=16) == []


def test_format_rejects_degenerate_polygon() -> None:
    with pytest.raises(ValueError, match="three"):
        format_yolo_segment(0, [(0.0, 0.0), (1.0, 1.0)])


def test_format_yolo_segment() -> None:
    line = format_yolo_segment(0, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])

    assert line == "0 0.000000 0.000000 1.000000 0.000000 1.000000 1.000000"
