from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics import YOLOWorld


DEFAULT_CLASSES = [
    "plastic bottle",
    "paper scrap",
    "cardboard",
    "plastic bag",
    "aluminum can",
    "paper cup",
    "food package",
    "litter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small YOLO-World pseudo-label pilot.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Open-vocabulary prompts. Quote prompts that contain spaces.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = sorted(
        path
        for path in args.source.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not images:
        raise SystemExit(f"No images found under {args.source}")

    sample_count = min(args.samples, len(images))
    indices = np.linspace(0, len(images) - 1, sample_count, dtype=int)
    selected = [images[index] for index in indices]

    model = YOLOWorld(str(args.weights))
    model.set_classes(args.classes)
    results = model.predict(
        source=[str(path) for path in selected],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(args.output.parent),
        name=args.output.name,
        exist_ok=True,
        verbose=False,
    )

    counts: Counter[str] = Counter()
    detections = []
    for source, result in zip(selected, results):
        image_detections = []
        if result.boxes is not None:
            for class_id, confidence in zip(
                result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()
            ):
                class_name = args.classes[int(class_id)]
                counts[class_name] += 1
                image_detections.append(
                    {"class": class_name, "confidence": round(float(confidence), 4)}
                )
        detections.append({"image": source.name, "detections": image_detections})

    report = {
        "weights": str(args.weights),
        "source": str(args.source),
        "sample_count": sample_count,
        "sample_indices": indices.tolist(),
        "classes": args.classes,
        "settings": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "device": args.device,
        },
        "total_detections": sum(counts.values()),
        "detections_by_class": dict(counts),
        "images": detections,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "pilot_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
