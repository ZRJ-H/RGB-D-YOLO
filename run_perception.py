"""Portable preview entry point; raw depth is never treated as registered RGBD."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
VERSION = "0.1.0-preview"


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Cannot write image: {path}")


def read_image(path, flags):
    # Ultralytics 8.3.179 monkey-patches cv2.imread to append a channel axis
    # even for uint16 depth. imdecode preserves the native depth shape.
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def raw_depth_summary(depth):
    # 未标定深度只统计原始值和非零比例，不能把这些数值解释成米。
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("raw depth must be a single-channel uint16 image")
    valid = depth > 0
    values = depth[valid]
    quantiles = np.percentile(values, [10, 50, 90]).tolist() if values.size else [None] * 3
    return {
        "width": int(depth.shape[1]), "height": int(depth.shape[0]),
        "nonzero_ratio": float(valid.mean()), "raw_percentiles_10_50_90": quantiles,
        "unit": "unknown_raw_unit", "metric_geometry": None,
        "quality_flags": ["UNCALIBRATED", "NOT_REGISTERED_TO_RGB"] + ([] if values.size else ["NO_VALID_DEPTH"]),
    }


def depth_colour(depth):
    # 每帧按2%～98%分位拉伸颜色以突出轮廓；不同帧的同一颜色不代表相同距离。
    valid = depth > 0
    gray = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        gray[valid] = np.clip(255 * (hi - depth[valid].astype(float)) / max(hi - lo, 1), 0, 255).astype(np.uint8)
    result = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    result[~valid] = 0
    return result


def panel(image, title):
    # Presentation only. Padding does not establish any spatial registration.
    canvas = np.full((530, 640, 3), 24, dtype=np.uint8)
    scale = min(640 / image.shape[1], 480 / image.shape[0])
    view = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    x, y = (640 - view.shape[1]) // 2, 50 + (480 - view.shape[0]) // 2
    canvas[y:y + view.shape[0], x:x + view.shape[1]] = view
    cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def synthetic_geometry():
    # 人工构造2米背景平面和1.8米前景，用于验证0.2米突出物的几何计算。
    from depth_perception import DepthObstacleDetector
    from depth_perception.types import CameraIntrinsics, DepthDetectorConfig
    raw = np.full((64, 96), 2000, dtype=np.uint16)
    raw[32:47, 30:50] = 1800
    intrinsics = CameraIntrinsics(96, 64, 120., 120., 47.5, 31.5)
    config = DepthDetectorConfig(depth_scale=.001, plane_sample_stride=2,
        min_plane_points=100, min_component_pixels=100, morphology_kernel=1,
        min_plane_inlier_ratio=.7, min_object_height_m=.05, max_object_height_m=.4)
    result = DepthObstacleDetector(config).process(raw, intrinsics)
    return raw, result


def parse_args(argv=None):
    # rgb仅分割；paired-preview并排预览；另两种模式分别验证合成/标定深度。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["rgb", "paired-preview", "depth-synthetic", "depth-calibrated"])
    parser.add_argument("--rgb", type=Path)
    parser.add_argument("--depth", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "best.pt")
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--depth-scale", type=float, help="verified raw-depth unit in metres; no default")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "depth_baseline.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--iou", type=float, default=.35)
    parser.add_argument("--rotate", type=int, choices=[0, 180], default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output directory already exists: {args.output}")
    if args.mode in ("rgb", "paired-preview"):
        if args.rgb is None or not args.rgb.is_file():
            parser.error("--rgb must be an existing image")
        if not args.weights.is_file():
            parser.error("--weights must be an existing local model (no auto-download)")
    if args.mode in ("paired-preview", "depth-calibrated") and (args.depth is None or not args.depth.is_file()):
        parser.error("--depth must be an existing image")
    if args.mode == "depth-calibrated":
        # 米制计算必须显式提供深度单位，避免把相机原始整数直接当作米。
        if args.intrinsics is None or not args.intrinsics.is_file() or args.depth_scale is None:
            parser.error("depth-calibrated requires --intrinsics and explicit --depth-scale")
        if not np.isfinite(args.depth_scale) or args.depth_scale <= 0:
            parser.error("--depth-scale must be finite and positive")
    elif args.intrinsics is not None or args.depth_scale is not None:
        parser.error("calibration arguments are only valid for depth-calibrated")
    if args.imgsz <= 0 or not 0 < args.conf < 1 or not 0 < args.iou <= 1:
        parser.error("invalid inference thresholds or image size")
    return args


def main(argv=None):
    args = parse_args(argv)
    start = time.perf_counter()
    payload = {"release_version": VERSION, "mode": args.mode, "spatial_fusion": False,
        "predictions_are_ground_truth": False, "rotation_degrees": args.rotate,
        "rgb": None, "depth": None}
    images = {}
    # RGB实例掩膜与深度仍是独立输出，尚未建立像素配准关系。
    if args.mode in ("rgb", "paired-preview"):
        # Keep runtime settings local so another account does not need the
        # previous account's AppData configuration to use this release.
        settings = Path(os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics")))
        settings.mkdir(parents=True, exist_ok=True)
        from ultralytics import YOLO
        rgb = read_image(args.rgb, cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"Cannot decode RGB: {args.rgb}")
        if args.rotate:
            rgb = cv2.rotate(rgb, cv2.ROTATE_180)
        model = YOLO(str(args.weights))
        if model.task != "segment" or model.names != {0: "sweepable_litter"}:
            raise ValueError("Expected single-class sweepable_litter segmentation model")
        prediction = model.predict(rgb, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            device=args.device, retina_masks=True, verbose=False, save=False)[0]
        instances = []
        boxes = prediction.boxes
        if len(boxes) and (prediction.masks is None or len(prediction.masks.data) != len(boxes)):
            raise RuntimeError("Missing segmentation masks")
        for index in range(len(boxes)):
            # 将掩膜转成原图尺寸的0/255单通道图，便于查看及后续人工复核。
            mask = (prediction.masks.data[index].cpu().numpy() > .5).astype(np.uint8) * 255
            if mask.shape != rgb.shape[:2]:
                raise RuntimeError("Instance mask must be in original RGB coordinates")
            name = f"mask_{index:03d}.png"
            images[name] = mask
            instances.append({"instance_id": index, "class_id": int(boxes.cls[index].item()),
                "class_name": "sweepable_litter", "semantic_confidence": float(boxes.conf[index].item()),
                "bbox_xyxy": boxes.xyxy[index].cpu().tolist(), "mask_file": name,
                "geometry": None})
        images["rgb_annotated.jpg"] = prediction.plot()
        payload["rgb"] = {"source": args.rgb.name, "width": rgb.shape[1], "height": rgb.shape[0],
            "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            "parameters": {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou,
                "device": args.device, "batch": 1, "retina_masks": True},
            "instances": instances, "ultralytics_speed_ms": prediction.speed}
    if args.mode == "paired-preview":
        # 调用者提供的图像对不等于已时间同步或空间对齐的RGBD数据。
        depth = read_image(args.depth, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Cannot decode depth: {args.depth}")
        if args.rotate:
            depth = cv2.rotate(depth, cv2.ROTATE_180)
        payload["depth"] = raw_depth_summary(depth)
        payload["depth"]["source"] = args.depth.name
        payload["pairing"] = "caller_supplied_pair; synchronization_not_verified_by_this_command"
        images["depth_colour.png"] = depth_colour(depth)
        images["depth_nonzero.png"] = (depth > 0).astype(np.uint8) * 255
        images["preview.jpg"] = np.hstack([panel(images["rgb_annotated.jpg"], "RGB segmentation | pre-annotations only"),
            panel(images["depth_colour.png"], "RAW depth | NOT registered | NOT metres")])
    elif args.mode in ("depth-synthetic", "depth-calibrated"):
        from depth_perception.serialization import result_to_dict
        if args.mode == "depth-synthetic":
            raw, result = synthetic_geometry()
            images["synthetic_depth.png"] = raw
            payload["synthetic_only"] = True
        else:
            from depth_perception.config import load_config, load_intrinsics
            from depth_perception.detector import DepthObstacleDetector
            from depth_perception.geometry import rotate_depth_180
            from run_depth_baseline import load_depth_image
            raw = load_depth_image(args.depth)
            intrinsics = load_intrinsics(args.intrinsics)
            if not all(np.isfinite(v) for v in (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)):
                raise ValueError("Intrinsics must be finite")
            config = replace(load_config(args.config), depth_scale=args.depth_scale)
            if args.rotate:
                raw, intrinsics = rotate_depth_180(raw, intrinsics)
            result = DepthObstacleDetector(config).process(raw, intrinsics)
            payload["calibration_status"] = "caller_supplied_not_independently_verified"
        payload["depth"] = result_to_dict(result)
        images["object_mask.png"] = result.object_mask
    payload["elapsed_seconds_including_load_and_inference"] = time.perf_counter() - start
    # 使用新目录集中写入图像和JSON；拒绝覆盖已有实验结果。
    args.output.mkdir(parents=True, exist_ok=False)
    for name, image in images.items():
        save_image(args.output / name, image)
    write_json(args.output / "result.json", payload)
    print(json.dumps({"output": str(args.output), "mode": args.mode,
        "instances": len(payload["rgb"]["instances"]) if payload["rgb"] else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
