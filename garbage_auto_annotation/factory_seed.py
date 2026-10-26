from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path

import yaml

from .conversion import format_yolo_segment


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_yolo_box_row(row: str) -> str:
    """Convert one normalized YOLO detection box into a rectangle segment."""
    parts = row.split()
    if len(parts) != 5:
        raise ValueError("A YOLO detection row must contain exactly five fields")
    try:
        class_id = int(parts[0])
        center_x, center_y, width, height = (float(value) for value in parts[1:])
    except ValueError as error:
        raise ValueError("YOLO detection fields must be numeric") from error
    if class_id != 0:
        raise ValueError("Factory seed labels must use class 0")
    values = (center_x, center_y, width, height)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("YOLO detection coordinates must be finite")
    if not 0.0 <= center_x <= 1.0 or not 0.0 <= center_y <= 1.0:
        raise ValueError("YOLO detection center must be within [0, 1]")
    if width <= 0.0 or height <= 0.0 or width > 1.0 or height > 1.0:
        raise ValueError("YOLO detection width and height must be within (0, 1]")

    left = max(0.0, center_x - width / 2.0)
    right = min(1.0, center_x + width / 2.0)
    top = max(0.0, center_y - height / 2.0)
    bottom = min(1.0, center_y + height / 2.0)
    if left >= right or top >= bottom:
        raise ValueError("YOLO detection box has no area after clamping")
    return format_yolo_segment(
        class_id,
        [(left, top), (right, top), (right, bottom), (left, bottom)],
    )


def _count_dataset(root: Path) -> tuple[int, int]:
    images = 0
    annotations = 0
    for split in ("train", "val"):
        images += sum(
            1
            for path in (root / "images" / split).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for label in (root / "labels" / split).rglob("*.txt"):
            annotations += sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())
    return images, annotations


def _output_stem(relative_image: Path) -> str:
    return "factory_seed_" + "__".join(relative_image.with_suffix("").parts)


def build_factory_seed_dataset(base_dataset: Path, source: Path, output: Path) -> dict:
    """Clone a segmentation dataset and add CVAT box labels as train-only rectangles."""
    base_dataset = base_dataset.resolve()
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output dataset already exists: {output}")
    if not (base_dataset / "dataset.yaml").is_file():
        raise FileNotFoundError(f"Base dataset.yaml does not exist: {base_dataset}")
    source_images = source / "images" / "train"
    source_labels = source / "labels" / "train"
    images = sorted(
        path
        for path in source_images.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No factory images found under {source_images}")

    prepared: list[tuple[Path, Path, str, list[str]]] = []
    for image in images:
        relative = image.relative_to(source_images)
        source_label = source_labels / relative.with_suffix(".txt")
        rows = source_label.read_text(encoding="utf-8").splitlines() if source_label.is_file() else []
        converted: list[str] = []
        for line_number, row in enumerate(rows, start=1):
            if not row.strip():
                continue
            try:
                converted.append(convert_yolo_box_row(row))
            except ValueError as error:
                raise ValueError(f"Invalid label {source_label}:{line_number}: {error}") from error
        prepared.append((image, relative, _output_stem(relative), converted))

    shutil.copytree(base_dataset, output)
    records: list[dict] = []
    for image, relative, stem, converted in prepared:
        destination = output / "images" / "train" / f"{stem}{image.suffix.lower()}"
        label = output / "labels" / "train" / f"{stem}.txt"
        if destination.exists() or label.exists():
            raise FileExistsError(f"Factory seed output name collides: {stem}")
        shutil.copy2(image, destination)
        label.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
        records.append(
            {
                "dataset": "factory_seed",
                "source_id": f"factory_seed:{relative.as_posix()}",
                "source_image": str(image.resolve()),
                "output_image": str(destination.resolve()),
                "output_label": str(label.resolve()),
                "split": "train",
                "annotations": len(converted),
                "sha256": _sha256(destination),
            }
        )

    dataset_yaml_path = output / "dataset.yaml"
    dataset_yaml = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8"))
    dataset_yaml["path"] = str(output)
    dataset_yaml_path.write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    image_count, annotation_count = _count_dataset(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {
        "version": 1,
        "class_names": {"0": "sweepable_litter"},
        "summary": {},
        "items": [],
        "rejected": [],
    }
    manifest_items = manifest.setdefault("items", [])
    known_source_ids = {item.get("source_id") for item in manifest_items}
    hard_negatives_path = output / "hard_negatives.json"
    if hard_negatives_path.is_file():
        hard_negatives = json.loads(hard_negatives_path.read_text(encoding="utf-8"))
        for record in hard_negatives.get("records", []):
            if record.get("source_id") not in known_source_ids:
                manifest_items.append(record)
                known_source_ids.add(record.get("source_id"))
    manifest_items.extend(records)
    manifest["summary"]["images_written"] = image_count
    manifest["summary"]["annotations_written"] = annotation_count
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    factory_annotations = sum(record["annotations"] for record in records)
    summary = {
        "factory_images": len(records),
        "factory_positive_images": sum(record["annotations"] > 0 for record in records),
        "factory_negative_images": sum(record["annotations"] == 0 for record in records),
        "factory_annotations": factory_annotations,
        "images_written": image_count,
        "annotations_written": annotation_count,
    }
    (output / "factory_seed.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def write_factory_focus_view(dataset_root: Path) -> dict:
    """Create a train-only domain view while retaining the public validation split."""
    dataset_root = dataset_root.resolve()
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.stem.startswith(("factory_seed_", "factory_negative_"))
    )
    if not images:
        raise FileNotFoundError(f"No factory seed or hard-negative images found under {image_dir}")

    annotation_count = 0
    lines: list[str] = []
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Factory focus image has no label: {image}")
        annotation_count += sum(
            1 for row in label.read_text(encoding="utf-8").splitlines() if row.strip()
        )
        lines.append(f"./images/train/{image.name}")

    (dataset_root / "factory_focus_train.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    payload = {
        "path": str(dataset_root),
        "train": "factory_focus_train.txt",
        "val": "images/val",
        "names": {0: "sweepable_litter"},
    }
    (dataset_root / "factory_focus.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return {
        "factory_focus_images": len(images),
        "factory_focus_annotations": annotation_count,
    }


def _validate_reviewed_segment_row(row: str) -> None:
    parts = row.split()
    if len(parts) < 7 or len(parts) % 2 == 0:
        raise ValueError("A reviewed segmentation row must contain a class and at least three points")
    try:
        class_id = int(parts[0])
        coordinates = [float(value) for value in parts[1:]]
    except ValueError as error:
        raise ValueError("Reviewed segmentation fields must be numeric") from error
    if class_id != 0:
        raise ValueError("Reviewed factory labels must use class 0")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in coordinates):
        raise ValueError("Reviewed segmentation coordinates must be finite and within [0, 1]")
    points = list(zip(coordinates[::2], coordinates[1::2]))
    area = abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
    ) / 2.0
    if area == 0.0:
        raise ValueError("Reviewed segmentation polygon must have a positive area")


def build_reviewed_factory_dataset(
    base_dataset: Path, archive: Path, output: Path, repeat: int = 1
) -> dict:
    """Clone a dataset and add human-reviewed Ultralytics segmentation data."""
    base_dataset = base_dataset.resolve()
    archive = archive.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output dataset already exists: {output}")
    if not (base_dataset / "dataset.yaml").is_file():
        raise FileNotFoundError(f"Base dataset.yaml does not exist: {base_dataset}")
    if not archive.is_file():
        raise FileNotFoundError(f"Reviewed archive does not exist: {archive}")
    if repeat < 1:
        raise ValueError("Reviewed factory repeat must be at least 1")

    with zipfile.ZipFile(archive) as source:
        if source.testzip() is not None:
            raise ValueError(f"Reviewed archive contains a corrupt entry: {archive}")
        names = set(source.namelist())
        if "data.yaml" not in names:
            raise ValueError("Reviewed archive does not contain data.yaml")
        config = yaml.safe_load(source.read("data.yaml"))
        class_names = config.get("names", {}) if isinstance(config, dict) else {}
        if class_names not in ({0: "sweepable_litter"}, {"0": "sweepable_litter"}):
            raise ValueError("Reviewed archive must define only class 0 sweepable_litter")
        image_names = sorted(
            name
            for name in names
            if name.startswith("images/train/")
            and Path(name).suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_names:
            raise ValueError("Reviewed archive contains no images under images/train")

        prepared: list[tuple[str, Path, str, list[str]]] = []
        for image_name in image_names:
            relative = Path(image_name).relative_to(Path("images/train"))
            stem = "factory_reviewed_" + "__".join(relative.with_suffix("").parts)
            label_name = (Path("labels/train") / relative.with_suffix(".txt")).as_posix()
            rows = source.read(label_name).decode("utf-8-sig").splitlines() if label_name in names else []
            reviewed_rows: list[str] = []
            for line_number, row in enumerate(rows, start=1):
                if not row.strip():
                    continue
                try:
                    _validate_reviewed_segment_row(row)
                except ValueError as error:
                    raise ValueError(f"Invalid reviewed label {label_name}:{line_number}: {error}") from error
                reviewed_rows.append(row.strip())
            prepared.append((image_name, relative, stem, reviewed_rows))

        shutil.copytree(base_dataset, output)
        records: list[dict] = []
        for image_name, relative, stem, rows in prepared:
            image_bytes = source.read(image_name)
            for repeat_index in range(1, repeat + 1):
                output_stem = (
                    stem
                    if repeat == 1
                    else stem.replace("factory_reviewed_", f"factory_reviewed_r{repeat_index:02d}_", 1)
                )
                destination = (
                    output / "images" / "train" / f"{output_stem}{relative.suffix.lower()}"
                )
                label = output / "labels" / "train" / f"{output_stem}.txt"
                if destination.exists() or label.exists():
                    raise FileExistsError(f"Reviewed factory output name collides: {output_stem}")
                destination.write_bytes(image_bytes)
                label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
                source_id = f"factory_reviewed:{relative.as_posix()}"
                if repeat > 1:
                    source_id += f"#repeat={repeat_index}"
                records.append(
                    {
                        "dataset": "factory_reviewed",
                        "source_id": source_id,
                        "source_archive": str(archive),
                        "source_image": image_name,
                        "repeat_index": repeat_index,
                        "output_image": str(destination.resolve()),
                        "output_label": str(label.resolve()),
                        "split": "train",
                        "annotations": len(rows),
                        "sha256": _sha256(destination),
                    }
                )

    dataset_yaml_path = output / "dataset.yaml"
    dataset_yaml = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8"))
    dataset_yaml["path"] = str(output)
    dataset_yaml_path.write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    image_count, annotation_count = _count_dataset(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("items", []).extend(records)
    manifest.setdefault("summary", {})["images_written"] = image_count
    manifest["summary"]["annotations_written"] = annotation_count
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    reviewed_annotations = sum(record["annotations"] for record in records)
    summary = {
        "reviewed_source_images": len(prepared),
        "reviewed_images": len(records),
        "reviewed_positive_images": sum(record["annotations"] > 0 for record in records),
        "reviewed_negative_images": sum(record["annotations"] == 0 for record in records),
        "reviewed_annotations": reviewed_annotations,
        "images_written": image_count,
        "annotations_written": annotation_count,
    }
    (output / "factory_reviewed.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
