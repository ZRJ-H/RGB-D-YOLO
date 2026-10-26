"""Export provisional XYZ on an unchanged stereo image grid and a matching PLY."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depth_perception.geometry import deproject_depth
from depth_perception.types import CameraIntrinsics


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.frame.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("Use a new output directory")
    metadata = json.loads((source / "result.json").read_text(encoding="utf-8"))
    if metadata["status"] != "conditional_on_rectified_input":
        raise ValueError("This frame has no applicable metric stereo calibration")
    intr = metadata["intrinsics_on_output_grid"]
    camera = CameraIntrinsics(**{k: int(intr[k]) if k in ("width", "height") else float(intr[k])
                                for k in ("width", "height", "fx", "fy", "cx", "cy")})
    depth_path = source / metadata["depth_file"]
    depth = np.load(depth_path, allow_pickle=False)
    gray = cv2.imread(str(source / "left_native.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(source / "validity_mask.png"), cv2.IMREAD_GRAYSCALE)
    if gray is None or mask is None or gray.shape != depth.shape or mask.shape != depth.shape:
        raise ValueError("Depth, original IR and validity mask must share a grid")
    if list(reversed(depth.shape)) != metadata["source_size"]:
        raise ValueError("Never change source image dimensions")
    pixels, points = deproject_depth(depth, camera, mask=mask > 0)
    if not len(points):
        raise ValueError("No valid points")
    u, v = pixels.T
    xyz = np.full((*depth.shape, 3), np.nan, dtype=np.float32)
    xyz[v, u] = points.astype(np.float32)
    valid = np.isfinite(xyz).all(axis=2)
    np.testing.assert_array_equal(valid, np.isfinite(depth) & (depth > 0) & (mask > 0))
    np.testing.assert_array_equal(xyz[..., 2][valid], depth[valid])
    reconstructed = xyz[v, u].astype(np.float64)
    reprojection = np.column_stack((camera.fx * reconstructed[:, 0] / reconstructed[:, 2] + camera.cx,
                                   camera.fy * reconstructed[:, 1] / reconstructed[:, 2] + camera.cy))
    max_error = float(np.max(np.abs(reprojection - pixels)))
    if max_error > 0.001:
        raise ValueError(f"Unexpected numerical roundtrip error: {max_error}")
    output.mkdir(parents=True)
    np.save(output / "xyz_m_provisional.npy", xyz)
    np.testing.assert_array_equal(np.load(output / "xyz_m_provisional.npy"), xyz)
    # Row-major valid pixels only. Colour is left-IR intensity, not registered RGB.
    vertex_type = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                            ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertices = np.empty(len(points), dtype=vertex_type)
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[v, u, index]
    for name in ("red", "green", "blue"):
        vertices[name] = gray[v, u]
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment provisional left IR optical frame, metres, X right Y down Z forward\n"
              "comment half-resolution inference restored to original grid; no RGB registration\n"
              f"element vertex {len(vertices)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode("ascii")
    ply = output / "cloud_ir_gray_provisional.ply"
    with ply.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)
    with ply.open("rb") as stream:
        if stream.read(len(header)) != header:
            raise ValueError("PLY header mismatch")
        reloaded = np.fromfile(stream, dtype=vertex_type)
    np.testing.assert_array_equal(reloaded, vertices)
    samples = []
    for x, y in ((640, 400), (400, 300), (900, 600)):
        if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1] and valid[y, x]:
            samples.append({"u": x, "v": y, "xyz_m": xyz[y, x].astype(float).tolist(),
                            "range_m": float(np.linalg.norm(xyz[y, x].astype(float)))})
    report = {
        "status": "conditional_on_rectified_input", "metric_accuracy_verified": False,
        "source_frame": str(source), "source_number": metadata["number"], "doe": metadata["doe"],
        "source_depth_sha256": hashlib.sha256(depth_path.read_bytes()).hexdigest(),
        "calibration_sha256": metadata["calibration_sha256"], "intrinsics": intr,
        "coordinate_frame": "left_IR_optical_assuming_supplied_rectified_geometry",
        "axes": {"X": "right", "Y": "down", "Z": "forward_optical_axis"}, "unit": "metre",
        "pixel_origin": "top_left_zero_based", "indexing": "xyz[v, u] == [X,Y,Z]",
        "formula": "X=(u-cx)*Z/fx; Y=(v-cy)*Z/fy; Z=depth_m[v,u]",
        "xyz_shape": list(xyz.shape), "dtype": "float32", "invalid_value": "NaN in all 3 channels",
        "total_grid_pixels": int(depth.size), "valid_points": len(points), "valid_fraction": float(valid.mean()),
        "native_resolution_inference": False, "original_inference_size": metadata["input_size"],
        "xyz_file": "xyz_m_provisional.npy", "pointcloud_file": ply.name,
        "pointcloud_colours": "left_IR_grayscale_not_RGB", "pointcloud_order": "valid pixels in row-major order",
        "max_reprojection_error_pixels": max_error,
        "verification_scope": "numerical conversion and file roundtrip only, not physical accuracy",
        "world_pose_known": False, "depth_to_RGB_registration_verified": False,
        "examples_zero_based": samples,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        f"# 第{metadata['number']}组逐像素三维坐标样例\n\n"
        "`xyz_m_provisional.npy`为800×1280×3 float32数组；`xyz[v,u]`返回该像素的[X,Y,Z]，单位米，无效位置三个通道均为NaN。\n\n"
        "坐标原点为左IR相机光心，X向右、Y向下、Z沿光轴向前。Z不是欧氏距离；到光心的距离为sqrt(X²+Y²+Z²)。\n\n"
        "`cloud_ir_gray_provisional.ply`包含全部有效像素的XYZ，按行优先排列；颜色来自左IR灰度，未做RGB配准。\n\n"
        "当前深度仍为640×400推理后最近邻恢复至1280×800，逐像素坐标并非同等数量的独立深度观测。"
        "历史IR校正与米制精度尚待确认，没有世界/地面位姿，未提供遮挡背面的坐标；几何有效掩膜不是准确性证明。\n\n"
        "report.json保存参数、示例坐标、来源哈希与投影回像素的数值检查。该检查只验证计算正确，不验证实际测距精度。\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
