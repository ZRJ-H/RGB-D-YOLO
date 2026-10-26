"""Stable JSON-safe serialization for depth perception results."""

from __future__ import annotations

from dataclasses import asdict

from .types import DepthFrameResult, QualityFlag


def _flags_payload(flags: QualityFlag) -> dict[str, object]:
    names = [flag.name for flag in QualityFlag if flag is not QualityFlag.NONE and flags & flag]
    return {"value": int(flags), "names": names}


def result_to_dict(result: DepthFrameResult) -> dict[str, object]:
    plane = None
    if result.plane is not None:
        plane = {
            "normal": list(result.plane.normal),
            "offset_m": result.plane.offset_m,
            "inlier_ratio": result.plane.inlier_ratio,
            "rmse_m": result.plane.rmse_m,
        }
    observations = []
    for observation in result.observations:
        observations.append(
            {
                "component_id": observation.component_id,
                "bbox_xyxy": list(observation.bbox_xyxy),
                "center_camera_m": list(observation.center_camera_m),
                "footprint_camera_m": [list(point) for point in observation.footprint_camera_m],
                "area_m2": observation.area_m2,
                "height_m": observation.height_m,
                "valid_pixel_count": observation.valid_pixel_count,
                "geometry_confidence": observation.geometry_confidence,
                "quality_flags": _flags_payload(observation.quality_flags),
            }
        )
    frame = asdict(result.metadata) if result.metadata is not None else None
    return {
        "algorithm_version": result.algorithm_version,
        "frame": frame,
        "image": {"width": result.width, "height": result.height},
        "valid_depth_ratio": result.valid_depth_ratio,
        "plane_valid": result.plane_valid,
        "plane": plane,
        "quality_flags": _flags_payload(result.quality_flags),
        "detector_config": asdict(result.detector_config),
        "observations": observations,
    }
