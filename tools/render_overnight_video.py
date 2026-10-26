"""Render chronological baseline/candidate segmentation at the source frame rate.

This is qualitative comparison on the same factory bag, with training exposure and
nearby frames. Candidate counts are not accuracy, and this is not an independent
test. Only final MP4, three screenshots and an audit report are saved. Source
images and model weights are read-only inputs; frames are processed one at a time.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import traceback


DEFAULT_SOURCE = Path(r"C:\Users\leg\Documents\Codex\2026-09-03\yolo-e-26-9-3-yolo\work\extracted_frames\test_camera1_full_10fps")
HEADER_HEIGHT = 54
CONFIDENCE = .25
NMS_IOU = .35
QUALITATIVE_NOTE = (
    "Same-bag chronological qualitative comparison only. The bag contains training "
    "images and temporal neighbours. Prediction/candidate counts and confidence "
    "scores are NOT accuracy, and this is NOT an independent test."
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def load_sources(source: Path, expected_frames: int) -> tuple[list[Path], dict]:
    if not source.is_dir():
        raise NotADirectoryError(source)
    images = sorted((path for path in source.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}), key=natural_key)
    if not images:
        raise ValueError(f"No image frames found: {source}")
    if expected_frames and len(images) != expected_frames:
        raise ValueError(f"Expected {expected_frames} source frames, found {len(images)}")
    timestamps = {}
    csv_path = source / "frames.csv"
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("frame") and row.get("timestamp_ns"):
                    timestamps[row["frame"]] = {"timestamp_ns": int(row["timestamp_ns"]), "source_topic": row.get("source_topic")}
        if any(image.name not in timestamps for image in images):
            raise ValueError("frames.csv does not cover every source image")
        ordered_times = [timestamps[image.name]["timestamp_ns"] for image in images]
        if any(second <= first for first, second in zip(ordered_times, ordered_times[1:])):
            raise ValueError("Source timestamps do not increase in filename order")
    return images, timestamps


def read_image(path: Path):
    import cv2
    import numpy as np
    blob = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Source frame cannot be decoded: {path}")
    return image, hashlib.sha256(blob).hexdigest()


def render_result(result, source_image):
    """Render native-resolution masks and concise scores without resizing frames."""
    import cv2
    import numpy as np
    image = source_image.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return image, {"candidate_count": 0, "mask_count": 0, "confidences": [], "boxes_xyxy": []}
    if result.masks is None:
        raise ValueError("Segmentation checkpoint returned boxes without masks")
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    polygons = result.masks.xy
    if len(polygons) != len(boxes) or np.any(classes != 0):
        raise ValueError("Unexpected mask/box count or class other than sweepable_litter")
    height, width = image.shape[:2]
    pixel_polygons = []
    for polygon in polygons:
        points = np.rint(polygon).astype(np.int32)
        if len(points) < 3:
            continue
        points[:, 0] = points[:, 0].clip(0, width - 1)
        points[:, 1] = points[:, 1].clip(0, height - 1)
        pixel_polygons.append(points)
    overlay = image.copy()
    if pixel_polygons:
        # Draw instances separately: fillPoly's even-odd rule can punch holes in
        # overlaps when independent instance polygons are passed in one call.
        for polygon in pixel_polygons:
            cv2.fillPoly(overlay, [polygon], (35, 210, 70))
        image = cv2.addWeighted(image, .72, overlay, .28, 0)
        cv2.polylines(image, pixel_polygons, True, (30, 245, 80), 1, cv2.LINE_AA)
    for box, score in zip(boxes, scores):
        label = f"{float(score):.2f}"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .42, 1)
        x = max(0, min(width - text_size[0] - 6, int(box[0])))
        y = max(text_size[1] + 5, min(height - 3, int(box[1]) - 3))
        cv2.rectangle(image, (x, y - text_size[1] - 4), (x + text_size[0] + 4, y + 2), (15, 45, 22), -1)
        cv2.putText(image, label, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, .42, (240, 255, 240), 1, cv2.LINE_AA)
    return image, {"candidate_count": len(boxes), "mask_count": len(polygons),
                   "confidences": [float(score) for score in scores], "boxes_xyxy": boxes.tolist()}


def verify_video(path: Path, expected_count: int, expected_fps: float, expected_size: tuple[int, int]) -> dict:
    """Decode every frame, not only the container frame-count metadata."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open written video: {path}")
    reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != expected_size:
                raise RuntimeError(f"Video frame {decoded} has an unexpected resolution")
            decoded += 1
    finally:
        capture.release()
    if decoded != expected_count or reported_count != expected_count:
        raise RuntimeError(f"Video frame-count verification failed: decoded={decoded}, reported={reported_count}, expected={expected_count}")
    if not math.isfinite(reported_fps) or abs(reported_fps - expected_fps) > .001:
        raise RuntimeError(f"Video FPS mismatch: {reported_fps} versus {expected_fps}")
    return {"status": "passed", "all_frames_decoded": decoded, "reported_frames": reported_count,
            "fps": reported_fps, "width": expected_size[0], "height": expected_size[1],
            "duration_seconds": decoded / reported_fps}


def run(args: argparse.Namespace) -> dict:
    baseline_path, candidate_path = args.baseline.resolve(), args.candidate.resolve()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing video output: {output}")
    for path in (baseline_path, candidate_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    images, timestamps = load_sources(source, args.expected_frames)
    # All runtime files are confined to this new output directory. Import only after
    # the environment is configured, since Ultralytics initializes settings at import.
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "ultralytics_runtime"
    runtime.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(runtime)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    started = time.perf_counter()
    writer = None
    report = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_directory": str(source), "output_directory": str(output),
              "baseline": {"path": str(baseline_path), "sha256": sha256_file(baseline_path)},
              "candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
              "qualitative_only": True, "is_accuracy_measurement": False, "independent_test": False,
              "scope_note": QUALITATIVE_NOTE, "parameters": {"imgsz": args.imgsz, "device": args.device,
                  "conf": CONFIDENCE, "iou": NMS_IOU, "fps": args.fps, "half": False, "retina_masks": True,
                  "batch": 1, "max_det": 300}, "source_frame_count": len(images), "frames": []}
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
        baseline, candidate = YOLO(str(baseline_path)), YOLO(str(candidate_path))
        if baseline.task != "segment" or candidate.task != "segment":
            raise ValueError("Both checkpoints must be segmentation models")
        first_image, _ = read_image(images[0])
        height, width = first_image.shape[:2]
        video_size = (width * 2, height + HEADER_HEIGHT)
        # mp4v requires even dimensions; real source frames are 640 x 480.
        if video_size[0] % 2 or video_size[1] % 2:
            raise ValueError(f"MP4 requires even canvas dimensions: {video_size}")
        report["source_resolution"] = {"width": width, "height": height}
        report["output_resolution"] = {"width": video_size[0], "height": video_size[1]}
        video_path = output / "comparison.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, video_size)
        if not writer.isOpened():
            raise RuntimeError("Cannot initialize mp4v video writer")
        snapshot_indices = {0: "first.jpg", len(images) // 2: "middle.jpg", len(images) - 1: "last.jpg"}
        counts = {"baseline": 0, "candidate": 0}
        predict_kwargs = {"imgsz": args.imgsz, "device": args.device, "conf": CONFIDENCE, "iou": NMS_IOU,
                          "max_det": 300, "half": False, "retina_masks": True, "save": False, "verbose": False,
                          "stream": False}
        for index, image_path in enumerate(images):
            image, source_sha = read_image(image_path)
            if image.shape[:2] != (height, width):
                raise ValueError(f"Source dimensions changed at {image_path}")
            # Each call processes exactly one frame; predictions are discarded before
            # the next image. No 909-frame array or rendered-frame list is retained.
            result = baseline.predict(source=image, **predict_kwargs)[0]
            left, left_log = render_result(result, image)
            del result
            result = candidate.predict(source=image, **predict_kwargs)[0]
            right, right_log = render_result(result, image)
            del result
            canvas = np.zeros((video_size[1], video_size[0], 3), np.uint8)
            canvas[:] = (23, 19, 14)
            canvas[HEADER_HEIGHT:, :width] = left
            canvas[HEADER_HEIGHT:, width:] = right
            for offset, label, log in ((0, "BASELINE", left_log), (width, "CANDIDATE", right_log)):
                cv2.putText(canvas, f"{label} | {log['candidate_count']} proposals | conf=.25 IoU=.35", (offset + 10, 21), cv2.FONT_HERSHEY_SIMPLEX, .51, (245, 245, 245), 1, cv2.LINE_AA)
                cv2.putText(canvas, f"{image_path.name} | {index/args.fps:06.1f}s | same bag; qualitative only", (offset + 10, 43), cv2.FONT_HERSHEY_SIMPLEX, .43, (130, 220, 250), 1, cv2.LINE_AA)
            writer.write(canvas)
            if index in snapshot_indices:
                success, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
                if not success:
                    raise RuntimeError("Screenshot JPEG encoding failed")
                (output / snapshot_indices[index]).write_bytes(encoded.tobytes())
            counts["baseline"] += left_log["candidate_count"]
            counts["candidate"] += right_log["candidate_count"]
            report["frames"].append({"frame_index": index, "source_name": image_path.name, "source_sha256": source_sha,
                                     "playback_time_seconds": index / args.fps, **timestamps.get(image_path.name, {}),
                                     "baseline": left_log, "candidate": right_log})
            if (index + 1) % 100 == 0 or index + 1 == len(images):
                print(f"Video: {index+1}/{len(images)} frames | baseline={counts['baseline']} candidate={counts['candidate']}", flush=True)
        writer.release()
        writer = None
        verification = verify_video(video_path, len(images), args.fps, video_size)
        if sha256_file(baseline_path) != report["baseline"]["sha256"] or sha256_file(candidate_path) != report["candidate"]["sha256"]:
            raise RuntimeError("A source weight file changed during video rendering")
        report.update({"status": "complete", "video": video_path.name, "video_sha256": sha256_file(video_path),
                       "frame_count": len(images), "fps": args.fps, "duration_seconds": len(images) / args.fps,
                       "total_candidate_counts": counts, "candidate_counts_are_not_accuracy": True,
                       "screenshots": snapshot_indices, "verification": verification,
                       "wall_seconds": time.perf_counter() - started})
        if timestamps:
            report["source_timestamp_span_seconds"] = (timestamps[images[-1].name]["timestamp_ns"] - timestamps[images[0].name]["timestamp_ns"]) / 1e9
            report["timing_note"] = "Chronological extracted 10 FPS frames played at the requested constant FPS; source timestamp jitter is not reproduced."
        write_json(output / "video_report.json", report)
        print(json.dumps({key: report[key] for key in ("status", "video", "frame_count", "fps", "duration_seconds", "total_candidate_counts", "verification")}, ensure_ascii=False, indent=2), flush=True)
        return report
    except Exception as error:
        report.update({"status": "failed", "error": str(error), "traceback": traceback.format_exc(), "wall_seconds": time.perf_counter() - started})
        write_json(output / "video_report.json", report)
        raise
    finally:
        if writer is not None:
            writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are rejected")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--fps", type=float, default=10.)
    parser.add_argument("--expected-frames", type=int, default=909, help="Assert source count; 0 disables the expected-count check")
    args = parser.parse_args()
    if args.imgsz < 32 or not math.isfinite(args.fps) or args.fps <= 0 or args.expected_frames < 0:
        parser.error("imgsz >= 32, finite fps > 0 and expected-frames >= 0 required")
    run(args)


if __name__ == "__main__":
    main()
