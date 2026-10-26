from __future__ import annotations

import math
from typing import Any, Mapping


def _as_list(value: Any) -> list:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _normalized_polygon(value: Any) -> tuple[list[list[float]] | None, str | None]:
    try:
        points = [[float(x), float(y)] for x, y in _as_list(value)]
    except (TypeError, ValueError):
        return None, "polygon_is_not_an_xy_sequence"
    if any(not math.isfinite(value) for point in points for value in point):
        return None, "polygon_contains_non_finite_coordinate"
    if any(value < 0.0 or value > 1.0 for point in points for value in point):
        return None, "polygon_coordinate_out_of_range"
    if len({(x, y) for x, y in points}) < 3:
        return None, "polygon_has_fewer_than_3_unique_points"
    return points, None


def result_to_manifest_entry(
    image_name: str,
    result: Any,
    class_names: Mapping[int, str],
) -> dict:
    """Turn one Ultralytics segmentation result into auditable JSON data."""
    entry = {"image": image_name, "predictions": [], "rejections": []}
    if result.boxes is None or result.masks is None:
        return entry

    class_ids = [int(value) for value in _as_list(result.boxes.cls)]
    confidences = [float(value) for value in _as_list(result.boxes.conf)]
    polygons = list(result.masks.xyn)
    count = max(len(class_ids), len(confidences), len(polygons))
    for index in range(count):
        if index >= len(class_ids) or index >= len(confidences) or index >= len(polygons):
            entry["rejections"].append({"index": index, "reason": "result_component_count_mismatch"})
            continue
        polygon, rejection = _normalized_polygon(polygons[index])
        if rejection:
            entry["rejections"].append({"index": index, "reason": rejection})
            continue
        class_id = class_ids[index]
        entry["predictions"].append(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, str(class_id)),
                "confidence": confidences[index],
                "polygon": polygon,
            }
        )
    return entry
