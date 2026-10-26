"""Export verified, conditional metric geometry from a native IR batch.

This is CPU-only postprocessing. It never resizes metric arrays, applies RGB
registration, fills holes, or treats geometric validity as model confidence.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from depth_perception.geometry import deproject_depth
from depth_perception.types import CameraIntrinsics
from tools.apply_stereo_calibration_trial import (
    depth_colour, font, legend, read_image, require, save_image,
    validate_calibration, write_json,
)

VERTEX_TYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
PERCENTILES = [0, 2, 25, 50, 75, 98, 100]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path, base):
    return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()


def validate_pair_selection(rows, numbers):
    if numbers is None:
        require([row["number"] for row in rows] == list(range(1, 48)), "Require complete 47-pair batch")
        return
    require(numbers and len(numbers) == len(set(numbers)), "Requested pair numbers must be nonempty and unique")
    require(all(1 <= number <= 47 for number in numbers), "Requested pair numbers must be within 1..47")
    require([row["number"] for row in rows] == sorted(numbers),
            "Input report pair numbers must exactly match --numbers")


def geometric_mask(disparity):
    require(disparity.ndim == 2, "Expected two-dimensional disparity")
    xx = np.arange(disparity.shape[1])[None, :]
    return (np.isfinite(disparity) & (disparity > 0)
            & (xx - disparity >= 0) & (xx - disparity <= disparity.shape[1] - 1))


def metric_arrays(disparity, camera, baseline_m):
    require(disparity.shape == (camera.height, camera.width), "Calibration grid mismatch")
    require(np.isfinite(baseline_m) and baseline_m > 0, "Invalid baseline")
    valid = geometric_mask(disparity)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = camera.fx * baseline_m / disparity[valid].astype(np.float64)
    require(np.isfinite(depth[valid]).all() and (depth[valid] > 0).all(), "Invalid converted depth")
    pixels, points = deproject_depth(depth, camera, mask=valid)
    xyz = np.full((*depth.shape, 3), np.nan, dtype=np.float32)
    u, v = pixels.T
    xyz[v, u] = points.astype(np.float32)
    return depth, xyz, valid


def save_ply(path, xyz, valid, gray):
    vertices = np.empty(int(valid.sum()), dtype=VERTEX_TYPE)
    for i, field in enumerate(("x", "y", "z")):
        vertices[field] = xyz[..., i][valid]
    for field in ("red", "green", "blue"):
        vertices[field] = gray[valid]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment conditional left IR optical coordinates, metres, X right Y down Z forward\n"
        "comment native-resolution inference, IR grayscale colours, no RGB registration\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)


def verify_metric_files(folder, disparity, camera, baseline_m, gray):
    """Independently read all persisted numeric values, not just sampled pixels."""
    depth = np.load(folder / "depth_m_provisional.npy", allow_pickle=False)
    xyz = np.load(folder / "xyz_m_provisional.npy", allow_pickle=False)
    mask_png = read_image(folder / "geometric_validity_mask.png", cv2.IMREAD_GRAYSCALE)
    valid = geometric_mask(disparity)
    require(depth.dtype == np.float32 and xyz.dtype == np.float32, "NPY dtype mismatch")
    require(depth.shape == disparity.shape and xyz.shape == (*disparity.shape, 3), "NPY shape mismatch")
    np.testing.assert_array_equal(mask_png, valid.astype(np.uint8) * 255)
    np.testing.assert_array_equal(np.isfinite(depth), valid)
    np.testing.assert_array_equal(np.isnan(depth), ~valid)
    np.testing.assert_array_equal(np.isfinite(xyz), np.repeat(valid[..., None], 3, axis=2))
    np.testing.assert_array_equal(np.isnan(xyz), np.repeat((~valid)[..., None], 3, axis=2))
    expected_z = camera.fx * baseline_m / disparity[valid].astype(np.float64)
    np.testing.assert_array_equal(depth[valid], expected_z.astype(np.float32))
    np.testing.assert_array_equal(xyz[..., 2][valid], depth[valid])
    v, u = np.nonzero(valid)
    z = depth[valid].astype(np.float64)
    np.testing.assert_array_equal(xyz[..., 0][valid], ((u - camera.cx) * z / camera.fx).astype(np.float32))
    np.testing.assert_array_equal(xyz[..., 1][valid], ((v - camera.cy) * z / camera.fy).astype(np.float32))
    x, y = xyz[..., 0][valid].astype(np.float64), xyz[..., 1][valid].astype(np.float64)
    max_reprojection = max(float(np.max(np.abs(camera.fx * x / z + camera.cx - u))),
                           float(np.max(np.abs(camera.fy * y / z + camera.cy - v))))
    require(max_reprojection < 0.001, "Numerical reprojection error exceeds tolerance")
    ply = folder / "cloud_ir_gray_provisional.ply"
    with ply.open("rb") as stream:
        header = []
        for _ in range(30):
            line = stream.readline()
            header.append(line.decode("ascii").strip())
            if line == b"end_header\n":
                break
        require(header[0] == "ply" and "format binary_little_endian 1.0" in header, "Invalid PLY format")
        require(f"element vertex {int(valid.sum())}" in header and header[-1] == "end_header", "Invalid PLY count/header")
        vertices = np.fromfile(stream, dtype=VERTEX_TYPE)
    require(len(vertices) == int(valid.sum()), "PLY vertex count mismatch")
    for i, field in enumerate(("x", "y", "z")):
        np.testing.assert_array_equal(vertices[field], xyz[..., i][valid])
    for field in ("red", "green", "blue"):
        np.testing.assert_array_equal(vertices[field], gray[valid])
    decoded = {}
    for name in ("geometric_validity_mask.png", "depth_colour_0p3_5m.png", "preview.jpg"):
        decoded[name] = list(read_image(folder / name).shape)
    require(decoded["depth_colour_0p3_5m.png"][:2] == list(disparity.shape), "Depth colour grid mismatch")
    return {
        "status": "passed", "depth_formula_float32_exact": True, "invalid_all_nan": True,
        "xyz_z_bitwise_equal_to_depth": True, "xyz_formula_float32_exact": True,
        "ply_all_vertices_and_ir_colours_exact": True,
        "max_depth_rounding_error_m": float(np.max(np.abs(z - expected_z))),
        "max_reprojection_error_pixels": max_reprojection, "decoded_image_shapes": decoded,
    }


def preview(left, right, colour, row, metric, path):
    width = 416
    height = round(left.shape[0] * width / left.shape[1])
    canvas = Image.new("RGB", (1280, height + 198), "#f4f6f2")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), f"{row['number']:02d}  DOE={row['doe']}  Native inference {left.shape[1]} x {left.shape[0]}",
              font=font(23), fill="#173b3e")
    labels = ("LEFT IR", "RIGHT IR", "CONDITIONAL DEPTH (m)" if metric else "DISPARITY ONLY (pixels)")
    for i, (im, label) in enumerate(zip((left, right, colour), labels)):
        if im.ndim == 2:
            rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        x = 8 + i * 424
        draw.text((x, 43), label, font=font(17), fill="#415d60")
        canvas.paste(Image.fromarray(rgb), (x, 68))
    if metric:
        canvas.paste(legend(640), (632, height + 77))
    else:
        draw.text((12, height + 89), "848 x 480 calibration unavailable; metric depth and XYZ withheld.", font=font(18), fill="#995625")
    draw.text((12, height + 88 if metric else height + 121),
              "Native arrays; panels reduced for viewing. No RGB registration.", font=font(16), fill="#415d60")
    draw.text((12, height + 114 if metric else height + 146),
              "Validity = geometric overlap, not confidence or accuracy.", font=font(16), fill="#415d60")
    draw.text((12, height + 147 if metric else height + 173),
              f"PAIR {row['number']:02d}: timestamp delta {row['timestamp_delta_raw']}, unit/tolerance unverified" if row["timestamp_delta_raw"] else
              "Historical rectification / physical accuracy remain unverified.",
              font=font(16), fill="#995625")
    save_image(path, cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))


def make_overviews(output, results):
    paths = []
    for page, offset in enumerate(range(0, len(results), 12), start=1):
        page_results = results[offset:offset + 12]
        canvas = Image.new("RGB", (1484, 1108), "#edf2ee")
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 10), f"Native stereo batch / {len(results)} pairs / page {page}", font=font(28), fill="#173b3e")
        caveats = []
        uncertain = [f"{row['number']:02d}" for row in page_results if row["pairing_uncertain"]]
        nonmetric = [f"{row['number']:02d}" for row in page_results if not row["metric_exported"]]
        if uncertain:
            caveats.append(f"{', '.join(uncertain)} pairing unverified")
        if nonmetric:
            caveats.append(f"{', '.join(nonmetric)} disparity only")
        subtitle = "LEFT IR | conditional depth, common 0.3-5 m scale"
        if caveats:
            subtitle += f" ({'; '.join(caveats)})"
        draw.text((20, 48), subtitle, font=font(18), fill="#995625")
        for i, row in enumerate(page_results):
            x, y = 16 + (i % 3) * 490, 83 + (i // 3) * 252
            draw.rounded_rectangle((x, y, x + 478, y + 238), radius=7, fill="white")
            metric = row["metric_exported"]
            suffix = "PAIR +/-1" if row["pairing_uncertain"] else ("NO METRIC" if not metric else "conditional")
            draw.text((x + 10, y + 8), f"{row['number']:02d} / DOE {row['doe']} / {suffix}", font=font(20), fill="#173b3e")
            left = read_image(output / row["files"]["left_input"], cv2.IMREAD_GRAYSCALE)
            colour = read_image(output / row["files"]["depth_colour" if metric else "disparity_colour"])
            for k, img in enumerate((cv2.cvtColor(left, cv2.COLOR_GRAY2RGB), cv2.cvtColor(colour, cv2.COLOR_BGR2RGB))):
                canvas.paste(Image.fromarray(cv2.resize(img, (224, 140), interpolation=cv2.INTER_AREA)), (x + 10 + k * 234, y + 42))
            text = (f"valid {row['geometric_valid_fraction'] * 100:.1f}% | median Z {row['depth_percentiles_m']['50']:.3f} m" if metric
                    else "848 x 480 / per-frame disparity colours")
            draw.text((x + 10, y + 194), text, font=font(17), fill="#415d60")
            draw.text((x + 10, y + 216), "Native grid; display previews reduced", font=font(14), fill="#627774")
        path = output / f"overview_{page}.jpg"
        save_image(path, cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
        paths.append(path.name)
    return paths


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numbers", type=int, nargs="+", metavar="N",
                        help="Explicit unique pair numbers (1..47); must exactly match the input report. Default requires all 47 pairs.")
    args = parser.parse_args()
    source, calibration_path, output = args.input.resolve(), args.calibration.resolve(), args.output.resolve()
    require(not output.exists(), "Output already exists; refusing to overwrite")
    report_path = source / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = sorted(report["results"], key=lambda row: row["number"])
    validate_pair_selection(rows, args.numbers)
    require(report["scale"] == 1, "Only native-resolution inference is accepted")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    intrinsics, baseline_m = validate_calibration(calibration)
    camera = CameraIntrinsics(**{k: int(intrinsics[0][k]) if k in ("width", "height") else float(intrinsics[0][k])
                                 for k in ("width", "height", "fx", "fy", "cx", "cy")})
    camera_dict = {k: getattr(camera, k) for k in ("width", "height", "fx", "fy", "cx", "cy")}
    calibration_hash = sha256(calibration_path)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(calibration_path, output / "calibration.json")
    require(sha256(output / "calibration.json") == calibration_hash, "Calibration copy hash differs")
    results, checks = [], []
    for row in rows:
        folder = f"left_{row['number']:02d}_doe{row['doe']}"
        original = source / folder
        dest = output / folder
        dest.mkdir(exist_ok=False)
        source_row = json.loads((original / "result.json").read_text(encoding="utf-8"))
        require(source_row == row, "Source result/report mismatch")
        left = read_image(original / "left_input.png", cv2.IMREAD_GRAYSCALE)
        right = read_image(original / "right_input.png", cv2.IMREAD_GRAYSCALE)
        native_left = read_image(Path(row["file"]), cv2.IMREAD_GRAYSCALE)
        native_right = read_image(Path(row["right_file"]), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(left, native_left)
        np.testing.assert_array_equal(right, native_right)
        require(left.shape == right.shape, "Left/right grid mismatch")
        require(row["size"] == row["input_size"] == [left.shape[1], left.shape[0]], "Native grid was not preserved")
        disparity = np.load(original / "disparity_pixels.npy", allow_pickle=False)
        require(disparity.dtype == np.float32 and disparity.shape == left.shape, "Disparity dtype/grid mismatch")
        valid = geometric_mask(disparity)
        positive = np.isfinite(disparity) & (disparity > 0)
        np.testing.assert_array_equal(read_image(original / "horizontal_overlap_mask.png", cv2.IMREAD_GRAYSCALE), valid.astype(np.uint8) * 255)
        np.testing.assert_array_equal(read_image(original / "positive_finite_mask.png", cv2.IMREAD_GRAYSCALE), positive.astype(np.uint8) * 255)
        inverse = np.load(original / "inverse_disparity_nonmetric.npy", allow_pickle=False)
        np.testing.assert_array_equal(inverse[positive], 1.0 / disparity[positive])
        np.testing.assert_array_equal(np.isnan(inverse), ~positive)
        metric = row["size"] == [camera.width, camera.height]
        require(metric or (row["number"] == 24 and row["size"] == [848, 480]), "Unexpected calibration mismatch")
        require(abs(row["timestamp_delta_raw"]) <= 1, "Unsupported stereo pairing")
        save_image(dest / "geometric_validity_mask.png", valid.astype(np.uint8) * 255)
        files = {key: relative(original / filename, output) for key, filename in {
            "disparity": "disparity_pixels.npy", "left_input": "left_input.png", "right_input": "right_input.png",
            "disparity_colour": "disparity_colour.png", "source_result": "result.json"}.items()}
        files.update(geometric_validity_mask=f"{folder}/geometric_validity_mask.png", preview=f"{folder}/preview.jpg")
        stats = None
        if metric:
            depth, xyz, valid = metric_arrays(disparity, camera, baseline_m)
            np.save(dest / "depth_m_provisional.npy", depth)
            np.save(dest / "xyz_m_provisional.npy", xyz)
            save_ply(dest / "cloud_ir_gray_provisional.ply", xyz, valid, left)
            colour = depth_colour(depth, valid)
            save_image(dest / "depth_colour_0p3_5m.png", colour)
            stats = dict(zip(map(str, PERCENTILES), np.percentile(depth[valid], PERCENTILES).astype(float).tolist()))
            files.update({k: f"{folder}/{v}" for k, v in {
                "depth": "depth_m_provisional.npy", "xyz": "xyz_m_provisional.npy",
                "pointcloud": "cloud_ir_gray_provisional.ply", "depth_colour": "depth_colour_0p3_5m.png"}.items()})
        else:
            colour = read_image(original / "disparity_colour.png")
        preview(left, right, colour, row, metric, dest / "preview.jpg")
        check = verify_metric_files(dest, disparity, camera, baseline_m, left) if metric else {
            "status": "passed", "metric_files_absent": not any(dest.glob("*.npy")) and not any(dest.glob("*.ply")),
            "decoded_image_shapes": {name: list(read_image(dest / name).shape) for name in ("preview.jpg", "geometric_validity_mask.png")}}
        require(check.get("metric_files_absent", True), "Mismatched grid must have no metric output")
        source_hashes = {"original_left": sha256(row["file"]), "original_right": sha256(row["right_file"])}
        for key, rel_path in files.items():
            if key in ("disparity", "left_input", "right_input", "source_result", "disparity_colour"):
                source_hashes[key] = sha256(output / rel_path)
        result = {
            "number": row["number"], "doe": row["doe"], "folder": folder,
            "status": "conditional_on_rectified_input" if metric else "metric_skipped_calibration_size_mismatch",
            "metric_exported": metric, "metric_accuracy_verified": False,
            "source_size": row["size"], "inference_size": row["input_size"], "output_size": row["size"],
            "native_resolution_inference": True, "grid_transform": "none",
            "inference_seconds": row["inference_seconds"], "source_left": row["file"], "source_right": row["right_file"],
            "source_hashes_sha256": source_hashes, "calibration_sha256": calibration_hash,
            "left_device_timestamp": row["device_timestamp"], "right_device_timestamp": row["right_device_timestamp"],
            "timestamp_delta_raw": row["timestamp_delta_raw"], "pairing": row["pairing"],
            "pairing_uncertain": row["timestamp_delta_raw"] != 0, "timestamp_unit_verified": False,
            "grid_pixels": int(disparity.size), "geometric_valid_pixels": int(valid.sum()),
            "pointcloud_points": int(valid.sum()) if metric else 0,
            "geometric_valid_fraction": float(valid.mean()), "masks_are_confidence": False,
            "depth_percentiles_m": stats, "depth_formula": "Z_m=fx_px*baseline_m/disparity_px" if metric else None,
            "fx_times_baseline_m_px": camera.fx * baseline_m if metric else None,
            "intrinsics": camera_dict if metric else None, "baseline_m": baseline_m if metric else None,
            "unit": "metre" if metric else "disparity_pixel",
            "coordinate_frame": "left_IR_optical_assuming_supplied_rectified_geometry" if metric else None,
            "axes": {"X": "right", "Y": "down", "Z": "forward_optical_axis"} if metric else None,
            "xyz_indexing": "xyz[v,u]=[X,Y,Z], zero-based; invalid=NaN in all channels" if metric else None,
            "pointcloud_colours": "left_IR_grayscale_not_RGB" if metric else None,
            "depth_to_RGB_registration_verified": False, "files_base": "geometry output directory", "files": files,
            "verification": check,
        }
        write_json(dest / "result.json", result)
        results.append(result)
        checks.append({"number": row["number"], "source_left_right_pixels_equal_native": True,
                       "source_positive_mask_overlap_mask_and_inverse_verified": True, **check})
        print(f"Verified {row['number']:02d}: {left.shape[1]}x{left.shape[0]}, metric={metric}, valid={int(valid.sum())}", flush=True)
    overviews = make_overviews(output, results)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "completed",
        "input_report": relative(report_path, output), "input_report_sha256": sha256(report_path),
        "calibration_file": "calibration.json", "calibration_source": str(calibration_path),
        "calibration_sha256": calibration_hash, "intrinsics": camera_dict, "baseline_m": baseline_m,
        "fx_times_baseline_m_px": camera.fx * baseline_m,
        "total_pairs": len(results), "metric_pairs": sum(r["metric_exported"] for r in results),
        "metric_skipped_numbers": [r["number"] for r in results if not r["metric_exported"]],
        "uncertain_pair_numbers": [r["number"] for r in results if r["pairing_uncertain"]],
        "total_pointcloud_points": sum(r["pointcloud_points"] for r in results),
        "inference_seconds_sum": sum(r["inference_seconds"] for r in results),
        "inference_seconds_median": float(np.median([r["inference_seconds"] for r in results])),
        "native_resolution_inference": True, "display_depth_range_m": [0.3, 5.0],
        "display_values_outside_range": "saturated colours, original values retained in NPY",
        "geometric_validity_definition": "finite positive disparity and 0 <= u-disparity <= width-1",
        "masks_are_confidence": False, "metric_accuracy_verified": False,
        "historical_input_rectification_verified": False, "depth_to_RGB_registration_verified": False,
        "unit_note": "Depth/XYZ/PLY already metres. Do not multiply by device depth_unit_mm.",
        "overviews": overviews, "results": results,
    }
    write_json(output / "summary.json", summary)
    fields = ["number", "doe", "status", "width", "height", "native_resolution_inference", "timestamp_delta_raw",
              "pairing_uncertain", "geometric_valid_pixels", "geometric_valid_fraction", "pointcloud_points",
              "depth_p02_m", "depth_p50_m", "depth_p98_m", "inference_seconds", "preview", "disparity", "depth", "xyz", "pointcloud"]
    with (output / "frames.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            flat = {key: row[key] for key in fields if key in row}
            flat.update(width=row["output_size"][0], height=row["output_size"][1])
            flat.update({key: row["files"].get(key, "") for key in ("preview", "disparity", "depth", "xyz", "pointcloud")})
            flat.update({f"depth_p{key}_m": row["depth_percentiles_m"][str(int(key))] if row["metric_exported"] else "" for key in ("02", "50", "98")})
            writer.writerow(flat)
    verification = {
        "status": "passed", "verified_pairs": len(checks), "metric_pairs": summary["metric_pairs"],
        "calibration_copy_sha256_equal": True, "native_grid_preserved_all_pairs": True,
        "scope": "All numeric pixels and persisted PLY vertices checked; implementation/file integrity only, not physical accuracy",
        "max_reprojection_error_pixels": max(c.get("max_reprojection_error_pixels", 0) for c in checks),
        "overview_decoded_shapes": {name: list(read_image(output / name).shape) for name in overviews},
        "frames": checks,
    }
    write_json(output / "verification.json", verification)
    caveats = "".join(
        f"{row['number']} 号为 {row['source_size'][0]}×{row['source_size'][1]}，缺少对应标定，仅引用原视差，不输出米制结果。"
        for row in results if not row["metric_exported"])
    caveats += "".join(
        f"{row['number']} 号左右时间戳差 {row['timestamp_delta_raw']}，配对仍待确认。"
        for row in results if row["pairing_uncertain"])
    (output / "README.md").write_text(
        "# 原生双目深度与逐像素三维输出\n\n"
        f"本次导出的 {len(results)} 组左右 IR 全部在原尺寸进行 FoundationStereo 推理。"
        f"这里对其中 {summary['metric_pairs']} 组 {camera.width}×{camera.height} 视差换算条件性光轴深度和 XYZ。"
        f"{caveats}\n\n"
        "深度和 XYZ 不缩放、不插值、不填孔，像素网格与左 IR 一致；无效深度为 NaN，XYZ 无效位置三个通道全为 NaN。"
        "坐标系原点为左 IR 光心，X 右、Y 下、Z 前。Z 为光轴深度，不是到光心的欧氏距离。"
        "深度、XYZ、PLY 已经使用米，不得再乘标定中的设备 depth_unit_mm。\n\n"
        "计算：Z=fx×B/d；X=(u−cx)×Z/fx；Y=(v−cy)×Z/fy。"
        "几何有效掩膜仅表示视差有限、为正且右目对应点落在图内，不是模型置信度或准确性保证。"
        "PLY 按有效像素行优先排列，颜色来自左 IR 灰度，未做 RGB 配准。\n\n"
        "伪彩统一显示 0.3～5 m，范围外颜色饱和但 NPY 保留原始数值。preview.jpg 和 overview_*.jpg 仅为缩小排版预览。"
        "原生推理没有提高标定的可信度：历史 IR 校正状态与实际米制精度仍待独立实测确认。\n\n"
        "summary.json 含全部相对输出路径、原输入位置与 SHA-256、参数和逐组结果；frames.csv 方便表格查看；"
        "verification.json 记录逐像素公式、NaN、XYZ-Z、数值回投、PLY 全部顶点/灰度读回与图像解码检查。"
        "这些检查验证计算和落盘完整性，不等同实物测量精度。\n",
        encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "total_pairs", "metric_pairs", "total_pointcloud_points", "inference_seconds_sum")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
