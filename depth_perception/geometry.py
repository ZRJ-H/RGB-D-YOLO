"""Pure geometry operations used by the depth detector."""

from __future__ import annotations

import numpy as np

from .types import CameraIntrinsics
from .types import DepthDetectorConfig, PlaneModel


def rotate_depth_180(
    depth_image: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, CameraIntrinsics]:
    """Rotate a depth frame and its pinhole principal point by 180 degrees."""

    array = np.asarray(depth_image)
    if array.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    if array.shape != (intrinsics.height, intrinsics.width):
        raise ValueError("depth image shape does not match camera intrinsics")
    rotated = np.ascontiguousarray(array[::-1, ::-1])
    # 像素坐标从0起算，180度翻转后主点变为(width-1-cx, height-1-cy)。
    rotated_intrinsics = CameraIntrinsics(
        width=intrinsics.width,
        height=intrinsics.height,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=(intrinsics.width - 1) - intrinsics.cx,
        cy=(intrinsics.height - 1) - intrinsics.cy,
    )
    return rotated, rotated_intrinsics


def decode_depth(
    depth_image: np.ndarray,
    *,
    depth_scale: float,
    min_depth_m: float = 0.1,
    max_depth_m: float = 10.0,
) -> np.ndarray:
    """Convert a two-dimensional depth image to metres with NaN invalids."""

    array = np.asarray(depth_image)
    if array.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive")
    if min_depth_m < 0 or max_depth_m <= min_depth_m:
        raise ValueError("depth range must satisfy 0 <= min_depth_m < max_depth_m")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("depth image must contain numeric values")

    depth_m = array.astype(np.float32, copy=True) * float(depth_scale)
    # 超范围或非有限深度统一为NaN，后续反投影会忽略这些点。
    valid = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    depth_m[~valid] = np.nan
    return depth_m


def deproject_depth(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: np.ndarray | None = None,
    *,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid ``(u, v)`` pixels and their optical-frame XYZ points."""

    depth = np.asarray(depth_m)
    if depth.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError("depth image shape does not match camera intrinsics")
    if stride < 1:
        raise ValueError("stride must be at least one")

    selected = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != depth.shape:
            raise ValueError("mask shape does not match depth image")
        selected &= mask_array
    if stride > 1:
        sampling = np.zeros(depth.shape, dtype=bool)
        sampling[::stride, ::stride] = True
        selected &= sampling

    v, u = np.nonzero(selected)
    # 针孔反投影：Z为深度，X=(u-cx)Z/fx，Y=(v-cy)Z/fy；坐标轴右、下、前。
    z = depth[v, u].astype(np.float64)
    x = (u.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (v.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    pixels = np.column_stack((u, v)).astype(np.int32, copy=False)
    points = np.column_stack((x, y, z))
    return pixels, points


def fit_plane_ransac(points: np.ndarray, config: DepthDetectorConfig) -> PlaneModel:
    """Fit a dominant plane and orient its positive side toward the camera."""

    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError("plane points must have shape (N, 3)")
    finite = cloud[np.all(np.isfinite(cloud), axis=1)]
    if len(finite) < config.min_plane_points:
        raise ValueError("insufficient points for floor-plane fitting")

    rng = np.random.default_rng(config.random_seed)
    best_mask: np.ndarray | None = None
    best_count = 0
    best_error = float("inf")
    for _ in range(config.plane_ransac_iterations):
        # 随机取三个点建立候选平面；优先保留内点最多的解，数量相同时比较残差。
        sample = finite[rng.choice(len(finite), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        distances = np.abs(finite @ normal + offset)
        inliers = distances <= config.plane_distance_threshold_m
        count = int(np.count_nonzero(inliers))
        error = float(np.mean(distances[inliers])) if count else float("inf")
        if count > best_count or (count == best_count and error < best_error):
            best_mask = inliers
            best_count = count
            best_error = error

    if best_mask is None or best_count < 3:
        raise ValueError("could not fit a floor plane")

    inlier_points = finite[best_mask]
    # 用最佳候选的全部内点做SVD精拟合，最小奇异值对应的方向为平面法向。
    centroid = np.mean(inlier_points, axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ centroid)
    if offset < 0:
        # 平面式为n·X+d=0；使相机原点处d非负，统一突出物高度的正方向。
        normal = -normal
        offset = -offset

    distances = np.abs(finite @ normal + offset)
    final_mask = distances <= config.plane_distance_threshold_m
    final_distances = distances[final_mask]
    ratio = float(np.mean(final_mask))
    rmse = float(np.sqrt(np.mean(np.square(final_distances)))) if final_distances.size else float("inf")
    return PlaneModel(
        normal=tuple(float(value) for value in normal),
        offset_m=offset,
        inlier_ratio=ratio,
        rmse_m=rmse,
    )


def plane_basis(plane: PlaneModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a plane origin and orthonormal 2D basis in camera coordinates."""

    normal = np.asarray(plane.normal, dtype=np.float64)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(normal @ reference)) > 0.9:
        # 避免参考轴与法向近乎平行，否则叉积太小，平面内基底不稳定。
        reference = np.array([0.0, 1.0, 0.0])
    axis_u = np.cross(normal, reference)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    axis_v /= np.linalg.norm(axis_v)
    origin = -plane.offset_m * normal
    return origin, axis_u, axis_v


def footprint_from_points(
    points: np.ndarray,
    plane: PlaneModel,
) -> tuple[tuple[tuple[float, float, float], ...], float]:
    """Project points onto a plane and return their convex 3D footprint and area."""

    import cv2

    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 3:
        return (), 0.0
    normal = np.asarray(plane.normal, dtype=np.float64)
    signed = plane.signed_distance(cloud)
    projected = cloud - signed[:, None] * normal[None, :]
    origin, axis_u, axis_v = plane_basis(plane)
    relative = projected - origin
    uv = np.column_stack((relative @ axis_u, relative @ axis_v)).astype(np.float32)
    # 在地面二维坐标中求凸包和平方米面积，再还原为相机坐标中的三维顶点。
    hull = cv2.convexHull(uv).reshape(-1, 2)
    if len(hull) < 3:
        return (), 0.0
    area = float(abs(cv2.contourArea(hull)))
    footprint = origin + hull[:, :1] * axis_u + hull[:, 1:] * axis_v
    return tuple(tuple(float(value) for value in row) for row in footprint), area
