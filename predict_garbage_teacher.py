from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

from garbage_auto_annotation.hard_negatives import IMAGE_SUFFIXES, make_contact_sheet
from garbage_auto_annotation.prediction import result_to_manifest_entry


DEFAULT_TARGET = Path(__file__).resolve().parent
PILOT_INDICES = [0, 4, 9, 14, 18, 23, 28, 33, 37, 42, 47, 52, 56, 61, 66, 71, 75, 80, 85, 90]


def _select_images(source: Path, all_after_pilot: bool, sample_count: int) -> list[Path]:
    images = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No images found under {source}")
    if all_after_pilot:
        return images
    if sample_count <= 0:
        raise ValueError("sample-count must be positive")
    if sample_count == 1:
        return [images[0]]
    if sample_count == 20 and len(images) > PILOT_INDICES[-1]:
        return [images[index] for index in PILOT_INDICES]
    if sample_count >= len(images):
        return images
    indices = [round(index * (len(images) - 1) / (sample_count - 1)) for index in range(sample_count)]
    return [images[index] for index in indices]


def _label_line(prediction: dict, include_confidence: bool) -> str:
    coordinates = " ".join(f"{value:.6f}" for point in prediction["polygon"] for value in point)
    line = f'{prediction["class_id"]} {coordinates}'
    return f'{line} {prediction["confidence"]:.6f}' if include_confidence else line


def _write_cvat_archive(output: Path, images: list[Path], include_images: bool) -> Path:
    archive = output / "cvat_yolo_seg_proposals.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("data.yaml", "path: .\ntrain: train.txt\nnames:\n  0: sweepable_litter\n")
        handle.writestr(
            "train.txt",
            "".join(f"images/train/{image.name}\n" for image in images),
        )
        for label in sorted((output / "labels").glob("*.txt")):
            handle.write(label, f"labels/train/{label.name}")
        if include_images:
            for image in images:
                handle.write(image, f"images/train/{image.name}")
    return archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the garbage teacher and export reviewable proposals.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--all-after-pilot", action="store_true")
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model does not exist: {args.model}")
    if not args.source.is_dir():
        raise FileNotFoundError(f"Image source does not exist: {args.source}")
    images = _select_images(args.source, args.all_after_pilot, args.sample_count)
    if args.dry_run:
        print(json.dumps({"model": str(args.model.resolve()), "images": [str(path.resolve()) for path in images]}, indent=2))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    annotated_dir = args.output / "annotated"
    labels_dir = args.output / "labels"
    audit_labels_dir = args.output / "labels_with_confidence"
    annotated_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    audit_labels_dir.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(DEFAULT_TARGET / ".ultralytics"))
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(args.model.resolve()))
    manifest = {"model": str(args.model.resolve()), "source": str(args.source.resolve()), "images": []}
    results = model.predict(
        source=[str(path) for path in images], conf=args.conf, iou=args.iou,
        imgsz=args.imgsz, device=args.device, stream=True, verbose=False,
    )
    for source_image, result in zip(images, results):
        entry = result_to_manifest_entry(source_image.name, result, model.names)
        manifest["images"].append(entry)
        cvat_text = "\n".join(_label_line(item, include_confidence=False) for item in entry["predictions"])
        audit_text = "\n".join(_label_line(item, include_confidence=True) for item in entry["predictions"])
        if cvat_text:
            cvat_text += "\n"
            audit_text += "\n"
        (labels_dir / f"{source_image.stem}.txt").write_text(cvat_text, encoding="utf-8")
        (audit_labels_dir / f"{source_image.stem}.txt").write_text(audit_text, encoding="utf-8")
        cv2.imwrite(str(annotated_dir / source_image.name), result.plot())

    (args.output / "predictions.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_contact_sheet(annotated_dir, args.output / "contact_sheet.jpg", columns=4)
    archive = _write_cvat_archive(args.output, images, args.include_images)
    proposal_count = sum(len(entry["predictions"]) for entry in manifest["images"])
    print(f"Processed {len(images)} images with {proposal_count} proposals")
    print(f"CVAT proposal archive: {archive.resolve()}")


if __name__ == "__main__":
    main()
