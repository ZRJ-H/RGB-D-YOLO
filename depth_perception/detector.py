"""Standalone floor-relative detector for depth frames."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import decode_depth, deproject_depth, fit_plane_ransac, footprint_from_points
from .types import (
    CameraIntrinsics,
    DepthDetectorConfig,
    DepthFrameResult,
    DepthObservation,
    FrameMetadata,
    PlaneModel,
    QualityFlag,
)


ALGORITHM_VERSION = "depth-baseline-v1"


class DepthObstacleDetector:
    """Detect connected protrusions relative to the dominant floor plane."""

    def __init__(self, config: DepthDetectorConfig | None = None) -> None:
        self.config = config or DepthDetectorConfig()

    def process(
        self,
        depth_image: np.ndarray,
        intrinsics: CameraIntrinsics,
        *,
        metadata: FrameMetadata | None = None,
    ) -> DepthFrameResult:
        # 输入是已有深度图：先统一为米，再拟合地面；此处不运行双目估计网络。
        depth_m = decode_depth(
            depth_image,
            depth_scale=self.config.depth_scale,
            min_depth_m=self.config.min_depth_m,
            max_depth_m=self.config.max_depth_m,
        )
        height, width = depth_m.shape
        if (height, width) != (intrinsics.height, intrinsics.width):
            raise ValueError("depth image shape does not match camera intrinsics")
        valid_ratio = float(np.mean(np.isfinite(depth_m)))
        empty_mask = np.zeros((height, width), dtype=np.uint8)

        floor_roi = np.zeros((height, width), dtype=bool)
        # 假设地面主要出现在画面下方，只从该ROI抽样拟合，降低其他物体的干扰。
        floor_start = int(np.floor(height * self.config.floor_roi_y_min))
        floor_roi[floor_start:, :] = True
        _, floor_points = deproject_depth(
            depth_m,
            intrinsics,
            floor_roi,
            stride=self.config.plane_sample_stride,
        )
        if len(floor_points) < self.config.min_plane_points:
            return self._empty_result(
                width,
                height,
                valid_ratio,
                QualityFlag.INSUFFICIENT_VALID_DEPTH | QualityFlag.FLOOR_PLANE_UNAVAILABLE,
                empty_mask,
                metadata,
            )

        try:
            plane = fit_plane_ransac(floor_points, self.config)
        except ValueError:
            return self._empty_result(
                width,
                height,
                valid_ratio,
                QualityFlag.FLOOR_PLANE_UNAVAILABLE,
                empty_mask,
                metadata,
            )
        if plane.inlier_ratio < self.config.min_plane_inlier_ratio:
            # 主平面支持点不足时保留质量信息，不继续输出不可靠的物体测量。
            return DepthFrameResult(
                algorithm_version=ALGORITHM_VERSION,
                width=width,
                height=height,
                valid_depth_ratio=valid_ratio,
                plane=plane,
                observations=(),
                quality_flags=QualityFlag.LOW_PLANE_CONFIDENCE,
                detector_config=self.config,
                metadata=metadata,
                object_mask=empty_mask,
            )

        pixels, points = deproject_depth(depth_m, intrinsics)
        signed_height = plane.signed_distance(points)
        # 高度是到拟合地面的有符号距离，并非相机Z方向的距离；只提取正向突出物。
        selected = (
            (signed_height >= self.config.min_object_height_m)
            & (signed_height <= self.config.max_object_height_m)
        )
        object_mask = np.zeros((height, width), dtype=np.uint8)
        selected_pixels = pixels[selected]
        object_mask[selected_pixels[:, 1], selected_pixels[:, 0]] = 255
        if self.config.morphology_kernel > 1:
            # 开运算去掉孤立噪点，闭运算填补小间隙，再按连通域归并候选物体。
            kernel = np.ones(
                (self.config.morphology_kernel, self.config.morphology_kernel),
                dtype=np.uint8,
            )
            object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
            object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)

        observations = self._extract_components(depth_m, intrinsics, object_mask, plane)
        flags = QualityFlag.NONE if observations else QualityFlag.NO_OBJECTS
        return DepthFrameResult(
            algorithm_version=ALGORITHM_VERSION,
            width=width,
            height=height,
            valid_depth_ratio=valid_ratio,
            plane=plane,
            observations=observations,
            quality_flags=flags,
            detector_config=self.config,
            metadata=metadata,
            object_mask=object_mask,
        )

    def measure_mask(
        self,
        depth_image: np.ndarray,
        intrinsics: CameraIntrinsics,
        instance_mask: np.ndarray,
        *,
        component_id: int = 0,
        plane: PlaneModel | None = None,
    ) -> DepthObservation:
        """Measure one external instance mask, such as a future YOLO mask."""

        mask = np.asarray(instance_mask, dtype=bool)
        # 外部掩膜必须事先配准到深度图；尺寸一致只是必要条件，不代表已经配准。
        if mask.shape != (intrinsics.height, intrinsics.width):
            raise ValueError("instance mask shape does not match camera intrinsics")
        total_pixels = int(np.count_nonzero(mask))
        if total_pixels < 3:
            raise ValueError("instance mask must contain at least three pixels")

        depth_m = decode_depth(
            depth_image,
            depth_scale=self.config.depth_scale,
            min_depth_m=self.config.min_depth_m,
            max_depth_m=self.config.max_depth_m,
        )
        if depth_m.shape != mask.shape:
            raise ValueError("depth image shape does not match instance mask")

        valid_mask = mask & np.isfinite(depth_m)
        valid_pixels = int(np.count_nonzero(valid_mask))
        valid_ratio = valid_pixels / float(total_pixels)
        if valid_ratio < self.config.min_mask_valid_ratio or valid_pixels < 3:
            raise ValueError("instance mask does not contain enough valid depth")

        if plane is None:
            frame_result = self.process(depth_image, intrinsics)
            if not frame_result.plane_valid or frame_result.plane is None:
                raise ValueError("a valid floor plane is required for mask measurement")
            plane = frame_result.plane

        _, points = deproject_depth(depth_m, intrinsics, valid_mask)
        signed_height = plane.signed_distance(points)
        footprint, area = footprint_from_points(points, plane)
        centre = np.median(points, axis=0)
        rows, columns = np.nonzero(mask)
        plane_quality = max(
            0.0,
            1.0 - plane.rmse_m / self.config.plane_distance_threshold_m,
        )
        confidence = float(np.clip(plane.inlier_ratio * plane_quality * valid_ratio, 0.0, 1.0))
        # 该分数仅描述几何数据质量，不是YOLO的类别置信度或校准后的概率。
        return DepthObservation(
            component_id=component_id,
            bbox_xyxy=(
                int(np.min(columns)),
                int(np.min(rows)),
                int(np.max(columns)),
                int(np.max(rows)),
            ),
            center_camera_m=tuple(float(value) for value in centre),
            footprint_camera_m=footprint,
            area_m2=area,
            height_m=max(0.0, float(np.percentile(signed_height, 95))),
            valid_pixel_count=valid_pixels,
            geometry_confidence=confidence,
        )

    def _extract_components(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        object_mask: np.ndarray,
        plane: PlaneModel,
    ) -> tuple[DepthObservation, ...]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(object_mask, connectivity=8)
        observations: list[DepthObservation] = []
        for label in range(1, count):
            # 标签0为背景；其余区域先按像素面积过滤，再计算三维几何量。
            pixel_count = int(stats[label, cv2.CC_STAT_AREA])
            if pixel_count < self.config.min_component_pixels:
                continue
            component_mask = labels == label
            _, points = deproject_depth(depth_m, intrinsics, component_mask)
            if len(points) < 3:
                continue
            signed_height = plane.signed_distance(points)
            footprint, area = footprint_from_points(points, plane)
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            centre = np.median(points, axis=0)
            # 用中位数中心和95分位高度，减少个别异常深度点对结果的影响。
            height_m = float(np.percentile(signed_height, 95))
            plane_quality = max(
                0.0,
                1.0 - plane.rmse_m / self.config.plane_distance_threshold_m,
            )
            size_quality = min(1.0, pixel_count / float(self.config.min_component_pixels * 2))
            confidence = float(np.clip(plane.inlier_ratio * plane_quality * size_quality, 0.0, 1.0))
            observations.append(
                DepthObservation(
                    component_id=len(observations) + 1,
                    bbox_xyxy=(x, y, x + w - 1, y + h - 1),
                    center_camera_m=tuple(float(value) for value in centre),
                    footprint_camera_m=footprint,
                    area_m2=area,
                    height_m=height_m,
                    valid_pixel_count=pixel_count,
                    geometry_confidence=confidence,
                )
            )
        return tuple(observations)

    def _empty_result(
        self,
        width: int,
        height: int,
        valid_ratio: float,
        flags: QualityFlag,
        object_mask: np.ndarray,
        metadata: FrameMetadata | None,
    ) -> DepthFrameResult:
        return DepthFrameResult(
            algorithm_version=ALGORITHM_VERSION,
            width=width,
            height=height,
            valid_depth_ratio=valid_ratio,
            plane=None,
            observations=(),
            quality_flags=flags,
            detector_config=self.config,
            metadata=metadata,
            object_mask=object_mask,
        )
