"""Benchmark the current RGB YOLO model on one CPU core and the current GPU."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import statistics
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}
DEFAULT_WEIGHTS = Path(
    "outputs/training/garbage_teacher_yolo11s_seg_reviewed20_r10_lr5e5_e10/weights/best.pt"
)
DEFAULT_SOURCE = Path("work/extracted_frames/test_camera1_1fps")
DEFAULT_OUTPUT = Path("outputs/inference/benchmark_current_best_cpu_gpu.json")


def percentile(values: list[float], percentage: float) -> float:
    """Return a linearly interpolated percentile (the common type-7 method)."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_ms(measurements: Iterable[float]) -> dict[str, float | int]:
    """Summarize per-image latency measurements expressed in milliseconds."""
    values = [float(value) for value in measurements]
    if not values:
        raise ValueError("at least one latency measurement is required")
    mean_ms = statistics.fmean(values)
    return {
        "runs": len(values),
        "mean_ms": round(mean_ms, 4),
        "p50_ms": round(percentile(values, 50), 4),
        "p95_ms": round(percentile(values, 95), 4),
        "min_ms": round(min(values), 4),
        "max_ms": round(max(values), 4),
        "fps": round(1000.0 / mean_ms, 4),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--gpu", default="0", help="CUDA device index")
    parser.add_argument("--half", action="store_true", help="Use FP16 on the GPU")
    return parser.parse_args(argv)


def find_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source.resolve()]
    if source.is_dir():
        return sorted(
            path.resolve()
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return []


def run_device(
    weights: Path,
    images: list[Path],
    device: str,
    imgsz: int,
    warmup: int,
    runs: int,
    half: bool,
) -> dict[str, object]:
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights))
    first_image = str(images[0])
    # The first call initializes Ultralytics' predictor and target device. It is
    # always excluded, even when --warmup=0, so setup never enters measurements.
    for warmup_index in range(max(1, warmup)):
        model.predict(
            source=first_image,
            imgsz=imgsz,
            batch=1,
            device=device,
            half=half,
            verbose=False,
        )
        if device == "cpu" and warmup_index == 0:
            # Ultralytics' CPU device setup resets PyTorch's intra-op threads.
            # Re-apply the one-thread limit after that setup and before timing.
            torch.set_num_threads(1)

    timings = {"preprocess": [], "inference": [], "postprocess": [], "pipeline": []}
    for index in range(runs):
        result = model.predict(
            source=str(images[index % len(images)]),
            imgsz=imgsz,
            batch=1,
            device=device,
            half=half,
            verbose=False,
        )[0]
        speed = result.speed
        preprocess = float(speed["preprocess"])
        inference = float(speed["inference"])
        postprocess = float(speed["postprocess"])
        timings["preprocess"].append(preprocess)
        timings["inference"].append(inference)
        timings["postprocess"].append(postprocess)
        timings["pipeline"].append(preprocess + inference + postprocess)

    return {
        "device": device,
        "precision": "FP16" if half else "FP32",
        "torch_threads": torch.get_num_threads(),
        "pure_inference": summarize_ms(timings["inference"]),
        "pipeline_excluding_io": summarize_ms(timings["pipeline"]),
        "preprocess": summarize_ms(timings["preprocess"]),
        "postprocess": summarize_ms(timings["postprocess"]),
    }


def configure_single_core(torch_module: object) -> None:
    torch_module.set_num_threads(1)
    torch_module.set_num_interop_threads(1)


def benchmark_device_order(gpu_available: bool, gpu: str) -> list[str]:
    """Run CUDA first because Ultralytics' CPU selection hides CUDA afterwards."""
    return [gpu, "cpu"] if gpu_available else ["cpu"]


@contextlib.contextmanager
def single_core_affinity(process: object):
    """Pin this process to one allowed logical CPU, then restore its affinity."""
    original_affinity = process.cpu_affinity()
    if not original_affinity:
        raise RuntimeError("the process has no available CPU cores")
    selected_core = original_affinity[0]
    process.cpu_affinity([selected_core])
    try:
        yield selected_core
    finally:
        process.cpu_affinity(original_affinity)


def main() -> None:
    args = parse_args()
    if args.runs < 1 or args.warmup < 0:
        raise SystemExit("--runs must be >= 1 and --warmup must be >= 0")
    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")
    images = find_images(args.source)
    if not images:
        raise SystemExit(f"No RGB images found under: {args.source}")

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))

    import cv2
    import psutil
    import torch
    import ultralytics  # noqa: F401 - import before setting PyTorch thread counts

    configure_single_core(torch)
    cv2.setNumThreads(1)
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    cpu_result = None
    gpu_result = None
    for device in benchmark_device_order(gpu_available, args.gpu):
        if device == "cpu":
            with single_core_affinity(psutil.Process()) as cpu_core:
                cpu_result = run_device(
                    args.weights, images, "cpu", args.imgsz, args.warmup, args.runs, False
                )
            cpu_result["logical_core"] = cpu_core
        else:
            gpu_result = run_device(
                args.weights, images, device, args.imgsz, args.warmup, args.runs, args.half
            )

    if cpu_result is None:
        raise RuntimeError("CPU benchmark did not run")

    result: dict[str, object] = {
        "weights": str(args.weights.resolve()),
        "source": str(args.source.resolve()),
        "image_count": len(images),
        "imgsz": args.imgsz,
        "batch": 1,
        "warmup": args.warmup,
        "runs": args.runs,
        "model_load_included": False,
        "cpu_threads": cpu_result["torch_threads"],
        "environment": {
            "platform": platform.platform(),
            "cpu": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": gpu_name,
        },
        "cpu": cpu_result,
        "gpu": gpu_result,
    }
    if gpu_result is not None:
        cpu_mean = cpu_result["pure_inference"]["mean_ms"]
        gpu_mean = gpu_result["pure_inference"]["mean_ms"]
        result["gpu_speedup_vs_single_core_cpu"] = round(cpu_mean / gpu_mean, 4)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
