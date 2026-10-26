from __future__ import annotations

import math

import cv2
import numpy as np


Point = tuple[float, float]


def normalize_polygon(points: list[Point], width: int, height: int) -> list[Point]:
    """Normalize pixel coordinates and clamp them to the image boundary."""
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")

    normalized = []
    for x, y in points:
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Polygon coordinates must be finite")
        normalized.append(
            (
                min(1.0, max(0.0, float(x) / width)),
                min(1.0, max(0.0, float(y) / height)),
            )
        )
    return normalized


def mask_to_polygons(mask: np.ndarray, min_area: int = 16) -> list[list[Point]]:
    """Extract external polygon contours from a binary foreground mask."""
    if mask.ndim != 2:
        raise ValueError("Mask must be a two-dimensional array")
    if min_area < 0:
        raise ValueError("Minimum area cannot be negative")

    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[Point]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        points = [(float(point[0][0]), float(point[0][1])) for point in contour]
        if len(set(points)) >= 3:
            polygons.append(points)
    return polygons


def format_yolo_segment(class_id: int, polygon: list[Point]) -> str:
    """Format one normalized polygon as a YOLO segmentation label row."""
    if class_id < 0:
        raise ValueError("Class ID cannot be negative")
    if len(set(polygon)) < 3:
        raise ValueError("A polygon needs at least three unique points")

    coordinates = []
    for x, y in polygon:
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Polygon coordinates must be finite")
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError("Normalized polygon coordinates must be within [0, 1]")
        coordinates.extend((f"{x:.6f}", f"{y:.6f}"))
    return " ".join((str(class_id), *coordinates))
