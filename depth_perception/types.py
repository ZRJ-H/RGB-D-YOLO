"""Validated public data contracts for depth perception."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics for the depth image coordinate system."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("intrinsics width and height must be positive")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("intrinsics fx and fy must be positive")


@dataclass(frozen=True)
class FrameMetadata:
    """Transport-neutral identity copied from input to output."""

    sequence_id: int
    timestamp_ns: int
    source_id: str
    frame_id: str

    def __post_init__(self) -> None:
        if self.sequence_id < 0 or self.timestamp_ns < 0:
            raise ValueError("frame sequence and timestamp must be non-negative")
        if not self.source_id or not self.frame_id:
            raise ValueError("frame source_id and frame_id must be non-empty")


@dataclass(frozen=True)
class DepthDetectorConfig:
    """Thresholds for the standalone floor-relative depth detector."""

    depth_scale: float = 0.001
    min_depth_m: float = 0.1
    max_depth_m: float = 5.0
    floor_roi_y_min: float = 0.5
    plane_sample_stride: int = 4
    plane_ransac_iterations: int = 160
    plane_distance_threshold_m: float = 0.015
    min_plane_points: int = 250
    min_plane_inlier_ratio: float = 0.45
    min_object_height_m: float = 0.02
    max_object_height_m: float = 0.5
    min_component_pixels: int = 60
    morphology_kernel: int = 3
    min_mask_valid_ratio: float = 0.35
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.depth_scale <= 0:
            raise ValueError("depth_scale must be positive")
        if self.min_depth_m < 0 or self.max_depth_m <= self.min_depth_m:
            raise ValueError("depth range is invalid")
        if not 0 <= self.floor_roi_y_min < 1:
            raise ValueError("floor_roi_y_min must be in [0, 1)")
        if self.plane_sample_stride < 1:
            raise ValueError("plane_sample_stride must be at least one")
        if self.plane_ransac_iterations < 1:
            raise ValueError("plane_ransac_iterations must be positive")
        if self.plane_distance_threshold_m <= 0:
            raise ValueError("plane_distance_threshold_m must be positive")
        if self.min_plane_points < 3:
            raise ValueError("min_plane_points must be at least three")
        if not 0 < self.min_plane_inlier_ratio <= 1:
            raise ValueError("min_plane_inlier_ratio must be in (0, 1]")
        if self.min_object_height_m <= 0 or self.max_object_height_m <= self.min_object_height_m:
            raise ValueError("object height range is invalid")
        if self.min_component_pixels < 1:
            raise ValueError("min_component_pixels must be positive")
        if self.morphology_kernel < 1 or self.morphology_kernel % 2 == 0:
            raise ValueError("morphology_kernel must be a positive odd integer")
        if not 0 < self.min_mask_valid_ratio <= 1:
            raise ValueError("min_mask_valid_ratio must be in (0, 1]")


class QualityFlag(IntFlag):
    """Machine-readable reasons why a depth result needs caution."""

    NONE = 0
    INSUFFICIENT_VALID_DEPTH = 1 << 0
    FLOOR_PLANE_UNAVAILABLE = 1 << 1
    LOW_PLANE_CONFIDENCE = 1 << 2
    NO_OBJECTS = 1 << 3


@dataclass(frozen=True)
class PlaneModel:
    """Normalized plane ``normal dot point + offset_m = 0``."""

    normal: tuple[float, float, float]
    offset_m: float
    inlier_ratio: float
    rmse_m: float

    def signed_distance(self, points: object) -> np.ndarray:
        array = np.asarray(points, dtype=np.float64)
        return array @ np.asarray(self.normal, dtype=np.float64) + self.offset_m


@dataclass(frozen=True)
class DepthObservation:
    """Objective floor-relative geometry for one connected depth component."""

    component_id: int
    bbox_xyxy: tuple[int, int, int, int]
    center_camera_m: tuple[float, float, float]
    footprint_camera_m: tuple[tuple[float, float, float], ...]
    area_m2: float
    height_m: float
    valid_pixel_count: int
    geometry_confidence: float
    quality_flags: QualityFlag = QualityFlag.NONE


@dataclass(frozen=True)
class DepthFrameResult:
    """Versioned output for one depth frame."""

    algorithm_version: str
    width: int
    height: int
    valid_depth_ratio: float
    plane: PlaneModel | None
    observations: tuple[DepthObservation, ...]
    quality_flags: QualityFlag
    detector_config: DepthDetectorConfig
    metadata: FrameMetadata | None
    object_mask: np.ndarray = field(repr=False, compare=False)

    @property
    def plane_valid(self) -> bool:
        return self.plane is not None and not bool(
            self.quality_flags & (QualityFlag.FLOOR_PLANE_UNAVAILABLE | QualityFlag.LOW_PLANE_CONFIDENCE)
        )
