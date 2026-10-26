from __future__ import annotations

import json
import math
from pathlib import Path

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _dataset_root(dataset_yaml: Path, payload: dict) -> Path:
    configured = Path(payload.get("path", dataset_yaml.parent))
    return configured if configured.is_absolute() else (dataset_yaml.parent / configured).resolve()


def validate_yolo_seg_dataset(dataset_yaml: Path) -> dict:
    """Validate file pairing and binary YOLO segmentation label semantics."""
    dataset_yaml = dataset_yaml.resolve()
    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = _dataset_root(dataset_yaml, payload)
    errors: list[dict] = []
    image_count = 0
    instance_count = 0
    split_counts: dict[str, dict[str, int]] = {}

    names = payload.get("names", {})
    normalized_names = {int(key): value for key, value in names.items()}
    if normalized_names != {0: "sweepable_litter"}:
        errors.append({"path": str(dataset_yaml), "reason": "unexpected_names_mapping"})

    for split in ("train", "val"):
        image_dir = root / payload[split]
        label_dir = root / "labels" / split
        images = sorted(path for path in image_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        image_stems = {path.stem for path in images}
        split_instances = 0
        for image_path in images:
            image_count += 1
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as error:
                errors.append({"path": str(image_path), "reason": "unreadable_image", "detail": str(error)})
            label_path = labels.get(image_path.stem)
            if label_path is None:
                errors.append({"path": str(image_path), "reason": "missing_label"})
                continue
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                parts = line.split()
                location = {"path": str(label_path), "line": line_number}
                if len(parts) < 7 or (len(parts) - 1) % 2:
                    errors.append({**location, "reason": "invalid_segment_field_count"})
                    continue
                try:
                    class_id = int(parts[0])
                    coordinates = [float(value) for value in parts[1:]]
                except ValueError:
                    errors.append({**location, "reason": "non_numeric_label_value"})
                    continue
                if class_id != 0:
                    errors.append({**location, "reason": "unexpected_class_id"})
                if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in coordinates):
                    errors.append({**location, "reason": "coordinate_out_of_range"})
                split_instances += 1
                instance_count += 1
        for stem, label_path in labels.items():
            if stem not in image_stems:
                errors.append({"path": str(label_path), "reason": "orphan_label"})
        split_counts[split] = {"images": len(images), "instances": split_instances}

    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_ids = {item["source_id"] for item in manifest.get("items", []) if item.get("split") == "train"}
        val_ids = {item["source_id"] for item in manifest.get("items", []) if item.get("split") == "val"}
        for source_id in sorted(train_ids & val_ids):
            errors.append({"path": str(manifest_path), "reason": "source_split_leakage", "source_id": source_id})

    return {
        "valid": not errors,
        "dataset_yaml": str(dataset_yaml),
        "root": str(root),
        "images": image_count,
        "instances": instance_count,
        "splits": split_counts,
        "errors": errors,
    }
