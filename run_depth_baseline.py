"""Run the standalone depth baseline on a 16-bit PNG/TIFF or NumPy array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from depth_perception.config import load_config, load_intrinsics
from depth_perception.detector import DepthObstacleDetector
from depth_perception.geometry import rotate_depth_180
from depth_perception.serialization import result_to_dict


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect floor-relative objects in one depth frame.")
    parser.add_argument("--depth", type=Path, required=True, help="16-bit PNG/TIFF or .npy depth image")
    parser.add_argument("--intrinsics", type=Path, required=True, help="calibrated depth-camera YAML")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "depth_baseline.yaml",
        help="depth detector YAML",
    )
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument(
        "--rotate",
        type=int,
        choices=(0, 180),
        default=0,
        help="rotate depth and calibrated intrinsics together before processing",
    )
    return parser.parse_args(argv)


def load_depth_image(path: Path) -> np.ndarray:
    # NPY保留数值数组；PNG/TIFF使用原始位深读取，避免16位距离值被压成8位。
    if path.suffix.lower() == ".npy":
        image = np.load(path, allow_pickle=False)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"could not read depth image: {path}")
    if image.ndim != 2:
        raise ValueError("depth input must be a single-channel image")
    return image


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"output directory already exists: {args.output}")
    config = load_config(args.config)
    intrinsics = load_intrinsics(args.intrinsics)
    depth_image = load_depth_image(args.depth)
    if args.rotate == 180:
        # 图像翻转时同步更新主点，否则反投影后的三维坐标会发生偏移。
        depth_image, intrinsics = rotate_depth_180(depth_image, intrinsics)
    result = DepthObstacleDetector(config).process(depth_image, intrinsics)

    args.output.mkdir(parents=True)
    # JSON保存几何量和质量标记，掩膜保存候选区域；这里不判定垃圾语义类别。
    payload = result_to_dict(result)
    (args.output / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if not cv2.imwrite(str(args.output / "object_mask.png"), result.object_mask):
        raise RuntimeError("could not write object mask")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "plane_valid": result.plane_valid,
                "observations": len(result.observations),
                "quality_flags": int(result.quality_flags),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
