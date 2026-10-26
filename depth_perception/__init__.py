"""Protocol-agnostic RGB-D geometry primitives for the sweeper project."""

from .detector import DepthObstacleDetector
from .config import load_config, load_intrinsics
from .serialization import result_to_dict
from .geometry import rotate_depth_180
from .types import (
    CameraIntrinsics,
    DepthDetectorConfig,
    DepthFrameResult,
    DepthObservation,
    FrameMetadata,
    QualityFlag,
)

__all__ = [
    "CameraIntrinsics",
    "DepthDetectorConfig",
    "DepthFrameResult",
    "DepthObservation",
    "DepthObstacleDetector",
    "FrameMetadata",
    "QualityFlag",
    "load_config",
    "load_intrinsics",
    "result_to_dict",
    "rotate_depth_180",
]
