from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .conversion import format_yolo_segment, mask_to_polygons, normalize_polygon


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _split_for_key(key: str, seed: int, validation_fraction: float = 0.2) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    return "val" if fraction < validation_fraction else "train"


def _ensure_layout(output: Path) -> None:
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def _find_taco_annotations(source: Path) -> Path:
    candidates = sorted(source.rglob("annotations.json"))
    if not candidates:
        raise FileNotFoundError(f"TACO annotations.json not found under {source}")
    return candidates[0]


def _prepare_taco(source: Path, output: Path, seed: int) -> tuple[list[dict], list[dict]]:
    annotation_path = _find_taco_annotations(source)
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    items: list[dict] = []
    rejected: list[dict] = []
    for image in coco.get("images", []):
        relative_name = Path(image["file_name"])
        source_image = annotation_path.parent / relative_name
        if not source_image.is_file():
            source_image = source / relative_name
        if not source_image.is_file():
            rejected.append({"dataset": "taco", "source": str(relative_name), "reason": "missing_image"})
            continue

        width = int(image["width"])
        height = int(image["height"])
        lines: list[str] = []
        accepted_ids: list[int] = []
        for annotation in annotations_by_image.get(int(image["id"]), []):
            segmentation = annotation.get("segmentation")
            if not isinstance(segmentation, list):
                rejected.append({"dataset": "taco", "annotation_id": annotation.get("id"), "reason": "unsupported_rle"})
                continue
            for flattened in segmentation:
                if not isinstance(flattened, list) or len(flattened) < 6 or len(flattened) % 2:
                    rejected.append({"dataset": "taco", "annotation_id": annotation.get("id"), "reason": "malformed_polygon"})
                    continue
                points = [(float(flattened[index]), float(flattened[index + 1])) for index in range(0, len(flattened), 2)]
                normalized = normalize_polygon(points, width, height)
                try:
                    lines.append(format_yolo_segment(0, normalized))
                except ValueError as error:
                    rejected.append({"dataset": "taco", "annotation_id": annotation.get("id"), "reason": str(error)})
                    continue
                accepted_ids.append(int(annotation["id"]))

        split = _split_for_key(relative_name.as_posix(), seed)
        output_stem = "taco_" + relative_name.with_suffix("").as_posix().replace("/", "__")
        output_image = output / "images" / split / f"{output_stem}{source_image.suffix.lower()}"
        output_label = output / "labels" / split / f"{output_stem}.txt"
        shutil.copy2(source_image, output_image)
        output_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        items.append(
            {
                "dataset": "taco",
                "source_id": f"taco:{relative_name.as_posix()}",
                "source_image": str(source_image.resolve()),
                "output_image": str(output_image.resolve()),
                "output_label": str(output_label.resolve()),
                "split": split,
                "annotations": len(lines),
                "source_annotation_ids": accepted_ids,
            }
        )
    return items, rejected


def _discover_dwsd_pairs(source: Path) -> list[tuple[str, Path, Path]]:
    images: dict[tuple[str, str], Path] = {}
    masks: dict[tuple[str, str], Path] = {}
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        lowered_parts = [part.lower() for part in path.parts]
        split = "test" if "test" in lowered_parts else "train" if "train" in lowered_parts else ""
        if not split:
            continue
        is_mask = any("mask" in part for part in lowered_parts)
        stem = path.stem.lower()
        prefixes = ("mask_",) if is_mask else ("image_", "img_")
        for prefix in prefixes:
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        key = (split, stem)
        if is_mask:
            masks[key] = path
        elif any(part in {"image", "images"} for part in lowered_parts):
            images[key] = path
    return [(split, image, masks[(split, stem)]) for (split, stem), image in sorted(images.items()) if (split, stem) in masks]


def _prepare_dwsd(source: Path, output: Path, min_mask_area: int) -> tuple[list[dict], list[dict]]:
    items: list[dict] = []
    rejected: list[dict] = []
    pairs = _discover_dwsd_pairs(source)
    if not pairs:
        raise FileNotFoundError(f"No DWSD image/mask pairs found under {source}")

    for source_split, source_image, source_mask in pairs:
        with Image.open(source_mask) as mask_image:
            mask = np.asarray(mask_image)
        if mask.ndim == 3:
            mask = np.any(mask != 0, axis=2).astype(np.uint8)
        height, width = mask.shape
        polygons = mask_to_polygons(mask, min_area=min_mask_area)
        lines = [format_yolo_segment(0, normalize_polygon(polygon, width, height)) for polygon in polygons]

        split = "val" if source_split == "test" else "train"
        output_stem = f"dwsd_{source_split}_{source_image.stem}"
        output_image = output / "images" / split / f"{output_stem}{source_image.suffix.lower()}"
        output_label = output / "labels" / split / f"{output_stem}.txt"
        shutil.copy2(source_image, output_image)
        output_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        items.append(
            {
                "dataset": "dwsd",
                "source_id": f"dwsd:{source_split}:{source_image.stem}",
                "source_image": str(source_image.resolve()),
                "source_mask": str(source_mask.resolve()),
                "output_image": str(output_image.resolve()),
                "output_label": str(output_label.resolve()),
                "split": split,
                "annotations": len(lines),
            }
        )
    return items, rejected


def _write_dataset_metadata(output: Path, items: list[dict], rejected: list[dict]) -> dict:
    summary = {
        "images_written": len(items),
        "annotations_written": sum(item["annotations"] for item in items),
        "rejected_annotations": len(rejected),
    }
    manifest = {
        "version": 1,
        "class_names": {"0": "sweepable_litter"},
        "summary": summary,
        "items": items,
        "rejected": rejected,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    dataset_yaml = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "sweepable_litter"},
    }
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return summary


def prepare_dwsd_dataset(source: Path, output: Path, min_mask_area: int = 16) -> dict:
    """Build a DWSD-only interim dataset while the larger TACO archive downloads."""
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Prepared dataset already exists: {output}")
    _ensure_layout(output)
    items, rejected = _prepare_dwsd(source, output, min_mask_area)
    return _write_dataset_metadata(output, items, rejected)


def prepare_combined_dataset(
    taco_source: Path,
    dwsd_source: Path,
    output: Path,
    seed: int = 42,
    min_mask_area: int = 16,
) -> dict:
    """Convert TACO and DWSD into one binary YOLO segmentation dataset."""
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Prepared dataset already exists: {output}")
    _ensure_layout(output)

    taco_items, taco_rejected = _prepare_taco(taco_source, output, seed)
    dwsd_items, dwsd_rejected = _prepare_dwsd(dwsd_source, output, min_mask_area)
    items = taco_items + dwsd_items
    rejected = taco_rejected + dwsd_rejected
    return _write_dataset_metadata(output, items, rejected)
