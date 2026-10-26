"""Build auditable, isolated YOLO segmentation views from existing reviewed truth.

No model prediction is consumed. Public original labels and CVAT-reviewed polygons
are the only label sources. Copy/paste moves an already reviewed object and its mask
together; generated images remain explicitly tagged synthetic. Source data are never
changed. Images are hardlinked where possible, labels are independent copies.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import zipfile

import cv2
import numpy as np
import yaml


DEFAULT_PROJECT = Path(r"C:\Users\leg\Documents\Codex\2026-09-03\yolo-e-26-9-3-yolo")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
REVIEWED_VIEWS = {"round2": "garbage_seg_factory_reviewed20_r10", "round3": "garbage_seg_factory_round3_r3"}
# Conservative clear-floor rectangles manually inspected in the 17 approved images.
# Coordinates are relative to the complete image. They avoid robots, people, shelves
# and cartons, rather than assuming that every pixel in the lower half is floor.
FLOOR_RECTS = {
    14: (.05, .69, .94, .86), 15: (.04, .72, .97, .96),
    16: (.04, .75, .97, .96), 17: (.04, .75, .97, .96),
    18: (.03, .79, .97, .97), 19: (.03, .83, .97, .97),
    20: (.51, .65, .97, .97), 21: (.27, .61, .97, .97),
    22: (.08, .67, .97, .97), 23: (.08, .65, .97, .97),
    24: (.13, .59, .94, .96), 25: (.13, .59, .94, .96),
    26: (.13, .62, .94, .96), 27: (.13, .65, .94, .96),
    33: (.14, .58, .96, .94), 35: (.06, .54, .95, .94),
    36: (.06, .54, .95, .91),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or min(image.shape[:2]) < 2:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def parse_rows(text: str, source: str) -> list[np.ndarray]:
    polygons = []
    for number, row in enumerate(text.splitlines(), 1):
        if not row.strip():
            continue
        fields = row.split()
        if len(fields) < 7 or len(fields) % 2 != 1 or fields[0] != "0":
            raise ValueError(f"Invalid class/polygon at {source}:{number}")
        points = np.asarray([float(x) for x in fields[1:]], dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(points).all() or points.min() < 0 or points.max() > 1:
            raise ValueError(f"Nonfinite or out of bounds label: {source}:{number}")
        area = abs(np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1))) / 2
        if area <= 1e-12:
            raise ValueError(f"Zero area polygon: {source}:{number}")
        polygons.append(points)
    return polygons


def inspect_pair(item: dict) -> dict:
    image_path, label_path = Path(item["source_image"]), Path(item["source_label"])
    blob = image_path.read_bytes()
    image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {image_path}")
    label_blob = label_path.read_bytes()
    rows = parse_rows(label_blob.decode("utf-8-sig"), str(label_path))
    return {**item, "sha256": sha256_bytes(blob), "label_sha256": sha256_bytes(label_blob),
            "width": int(image.shape[1]), "height": int(image.shape[0]), "annotations": len(rows)}


def verify_materialized(item: dict) -> None:
    checked = inspect_pair({"source_image": item["output_image"], "source_label": item["output_label"]})
    for field in ("sha256", "label_sha256", "width", "height", "annotations"):
        if checked[field] != item[field]:
            raise ValueError(f"Materialized {field} mismatch: {item['output_image']}")


def load_base(project: Path) -> tuple[list[dict], dict]:
    root = project / "datasets" / "garbage_seg"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    negatives = json.loads((root / "hard_negatives.json").read_text(encoding="utf-8"))
    lookup = {(x["split"], Path(x["output_image"]).name): x for x in manifest["items"] + negatives["records"]}
    items = []
    for split in ("train", "val"):
        for image in sorted((root / "images" / split).rglob("*")):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            original = lookup.get((split, image.name))
            if not original or original["dataset"] not in {"taco", "dwsd", "factory_negative"}:
                raise ValueError(f"Unapproved source or missing provenance: {image}")
            if image.stem.startswith("factory_seed_"):
                raise ValueError(f"Rectangle hint must not be included: {image}")
            if original["dataset"] == "factory_negative" and image.name.replace("factory_negative_", "", 1) not in negatives["approved"]:
                raise ValueError(f"Negative not explicitly approved: {image}")
            label = root / "labels" / split / image.relative_to(root / "images" / split).with_suffix(".txt")
            items.append({"source_image": str(image), "source_label": str(label), "source_split": split,
                          "source_id": original["source_id"], "dataset": original["dataset"],
                          "label_authority": "explicitly_approved_empty" if original["dataset"] == "factory_negative" else "public_original_annotation"})
    with ThreadPoolExecutor(max_workers=6) as pool:
        checked = list(pool.map(inspect_pair, items))
    for item in checked:
        if item["dataset"] == "factory_negative" and item["annotations"]:
            raise ValueError("Approved negative unexpectedly has objects")
    return checked, {"manifest": str(root / "manifest.json"), "manifest_sha256": sha256_file(root / "manifest.json"),
                     "hard_negatives": str(root / "hard_negatives.json"), "hard_negatives_sha256": sha256_file(root / "hard_negatives.json")}


def load_reviewed(project: Path) -> tuple[list[dict], list[dict]]:
    reviewed, archives = [], []
    for round_name, directory in REVIEWED_VIEWS.items():
        meta_path = project / "datasets" / directory / "factory_reviewed.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        records = [x for x in meta["records"] if x.get("repeat_index", 1) == 1]
        if len(records) != {"round2": 20, "round3": 26}[round_name]:
            raise ValueError(f"Unexpected count for {round_name}: {len(records)}")
        source_archive = Path(records[0]["source_archive"])
        with zipfile.ZipFile(source_archive) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"Corrupt CVAT entry: {bad}")
            for record in records:
                item = inspect_pair({"source_image": record["output_image"], "source_label": record["output_label"],
                                     "source_split": "train", "source_id": f"{round_name}:{record['source_image']}",
                                     "dataset": round_name, "original_name": Path(record["source_image"]).name,
                                     "source_archive": str(source_archive), "archive_image": record["source_image"],
                                     "label_authority": "cvat_human_reviewed_segmentation"})
                if item["sha256"] != sha256_bytes(archive.read(record["source_image"])):
                    raise ValueError(f"Image no longer matches reviewed archive: {record['output_image']}")
                archive_label = record["source_image"].replace("images/", "labels/", 1).rsplit(".", 1)[0] + ".txt"
                archive_text = archive.read(archive_label).decode("utf-8-sig") if archive_label in archive.namelist() else ""
                actual_rows = Path(item["source_label"]).read_text(encoding="utf-8-sig").split()
                if archive_text.split() != actual_rows:
                    raise ValueError(f"Labels no longer match reviewed archive: {item['source_label']}")
                reviewed.append(item)
        archives.append({"round": round_name, "archive": str(source_archive), "sha256": sha256_file(source_archive),
                         "unique_reviewed_images": len(records), "metadata": str(meta_path)})
    return reviewed, archives


def parse_holdout(argument: str | None) -> set[str]:
    if not argument:
        return set()
    path = Path(argument)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("holdout_reviewed_names", data.get("names", data.get("holdout_names")))
        if not isinstance(data, list):
            raise ValueError("Holdout JSON must be a list or contain holdout_reviewed_names/names/holdout_names")
        return {Path(str(name)).stem for name in data}
    return {Path(name.strip()).stem for name in argument.split(",") if name.strip()}


def link_or_copy(source: Path, destination: Path) -> str:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def materialize(item: dict, root: Path, split: str, stem: str, purpose: str, repeat_index: int = 1) -> dict:
    image = root / "images" / split / (stem + Path(item["source_image"]).suffix.lower())
    label = root / "labels" / split / (stem + ".txt")
    method = link_or_copy(Path(item["source_image"]), image)
    label.parent.mkdir(parents=True, exist_ok=True)
    if label.exists():
        raise FileExistsError(label)
    shutil.copyfile(item["source_label"], label)
    # Source and destination labels must agree byte-for-byte; source labels are never hardlinked.
    if sha256_file(label) != item["label_sha256"]:
        raise ValueError(f"Label copy mismatch: {label}")
    return {**item, "output_image": str(image), "output_label": str(label), "output_split": split,
            "purpose": purpose, "repeat_index": repeat_index, "image_storage": method}


def build_assets(reviewed: list[dict]) -> tuple[list[dict], list[dict]]:
    assets, rejected = [], []
    for source in reviewed:
        image = read_image(Path(source["source_image"]))
        h, w = image.shape[:2]
        polygons = parse_rows(Path(source["source_label"]).read_text(encoding="utf-8-sig"), source["source_label"])
        for index, polygon in enumerate(polygons):
            asset_id = f"{source['source_id']}#polygon={index}"
            if len(polygon) == 4 and len(np.unique(polygon[:, 0])) == 2 and len(np.unique(polygon[:, 1])) == 2:
                rejected.append({"source_id": asset_id, "reason": "axis_aligned_rectangle_cannot_provide_object_cutout"})
                continue
            mask = np.zeros((h, w), np.uint8)
            pixel_poly = np.rint(polygon * [w, h]).astype(np.int32)
            pixel_poly[:, 0] = pixel_poly[:, 0].clip(0, w - 1)
            pixel_poly[:, 1] = pixel_poly[:, 1].clip(0, h - 1)
            cv2.fillPoly(mask, [pixel_poly], 255)
            x, y, width, height = cv2.boundingRect(pixel_poly)
            area = int(np.count_nonzero(mask))
            if area < 40 or width < 4 or height < 4 or width > .55 * w or height > .45 * h or area > .16 * h * w:
                rejected.append({"source_id": asset_id, "reason": "tiny_or_large_unreliable_cutout", "area_px": area})
                continue
            x0, y0, x1, y1 = max(0, x - 2), max(0, y - 2), min(w, x + width + 2), min(h, y + height + 2)
            assets.append({"id": asset_id, "source": source, "image": image[y0:y1, x0:x1].copy(),
                           "mask": mask[y0:y1, x0:x1].copy(), "bbox_xyxy": [x0, y0, x1, y1], "area_px": area})
    if not assets:
        raise ValueError("No usable reviewed objects for copy/paste")
    return assets, rejected


def transform_asset(asset: dict, rng: random.Random, max_width: int, max_height: int) -> tuple[np.ndarray, np.ndarray, dict] | None:
    image, mask = asset["image"], asset["mask"]
    flip = rng.random() < .5
    if flip:
        image, mask = np.ascontiguousarray(image[:, ::-1]), np.ascontiguousarray(mask[:, ::-1])
    angle, scale = rng.uniform(-24, 24), rng.uniform(.65, 1.35)
    height, width = mask.shape
    angle_r = math.radians(angle)
    target_width = int(math.ceil((abs(width * math.cos(angle_r)) + abs(height * math.sin(angle_r))) * scale)) + 4
    target_height = int(math.ceil((abs(width * math.sin(angle_r)) + abs(height * math.cos(angle_r))) * scale)) + 4
    limit = min(1., max_width / target_width, max_height / target_height)
    scale *= limit
    target_width, target_height = max(4, int(target_width * limit)), max(4, int(target_height * limit))
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, scale)
    matrix[:, 2] += [(target_width - width) / 2, (target_height - height) / 2]
    transformed_image = cv2.warpAffine(image, matrix, (target_width, target_height), flags=cv2.INTER_LINEAR)
    transformed_mask = cv2.warpAffine(mask, matrix, (target_width, target_height), flags=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(transformed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 28:
        return None
    # Keep only the largest connected part so a YOLO polygon exactly describes it.
    transformed_mask[:] = 0
    cv2.fillPoly(transformed_mask, [largest], 255)
    gain = rng.uniform(.88, 1.10)
    transformed_image = np.clip(transformed_image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return transformed_image, transformed_mask, {"flip_horizontal": flip, "angle_degrees": angle, "scale": scale, "brightness_gain": gain}


def synthetic_sample(index: int, root: Path, backgrounds: list[dict], assets: list[dict], rng: random.Random) -> dict:
    background = rng.choice(backgrounds)
    image = read_image(Path(background["source_image"]))
    h, w = image.shape[:2]
    number = int(Path(background["source_image"]).stem.rsplit("_", 1)[1])
    rect = FLOOR_RECTS[number]
    left, top, right, bottom = [int(round(value * (w if axis % 2 == 0 else h))) for axis, value in enumerate(rect)]
    occupied = np.zeros((h, w), np.uint8)
    rows, placements = [], []
    desired = rng.choices([1, 2, 3, 4], [.20, .35, .30, .15])[0]
    for attempt in range(100):
        if len(placements) >= desired:
            break
        asset = rng.choice(assets)
        transformed = transform_asset(asset, rng, min(int(.20 * w), right - left - 2), min(int(.17 * h), bottom - top - 2))
        if transformed is None:
            continue
        cutout, mask, transform = transformed
        mh, mw = mask.shape
        if right - left < mw + 2 or bottom - top < mh + 2:
            continue
        x, y = rng.randint(left, right - mw), rng.randint(top, bottom - mh)
        padded = cv2.dilate(mask, np.ones((7, 7), np.uint8))
        if np.any((padded > 0) & (occupied[y:y+mh, x:x+mw] > 0)):
            continue
        # One-pixel antialiasing inside the mask avoids opaque copied floor fringes.
        alpha = np.minimum(cv2.distanceTransform(mask, cv2.DIST_L2, 3), 1.0)[..., None]
        region = image[y:y+mh, x:x+mw].astype(np.float32)
        image[y:y+mh, x:x+mw] = np.rint(alpha * cutout + (1 - alpha) * region).clip(0, 255).astype(np.uint8)
        occupied[y:y+mh, x:x+mw] = np.maximum(occupied[y:y+mh, x:x+mw], padded)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        contour = cv2.approxPolyDP(contour, .45, True).reshape(-1, 2).astype(np.float64)
        if len(contour) < 3:
            raise ValueError("Transformed object unexpectedly lost its contour")
        contour += [x, y]
        normalized = contour / [w, h]
        rows.append("0 " + " ".join(f"{value:.8f}" for value in normalized.ravel()))
        placements.append({"asset_id": asset["id"], "source_image_sha256": asset["source"]["sha256"],
                           "source_label_sha256": asset["source"]["label_sha256"], "source_bbox_xyxy": asset["bbox_xyxy"],
                           "destination_bbox_xyxy": [x, y, x+mw, y+mh], "mask_area_pixels": int(np.count_nonzero(mask)), **transform})
    if not placements:
        raise ValueError(f"Could not place any reviewed objects for synthetic image {index}")
    image_path, label_path = root / "images" / "synthetic" / f"copy_paste_{index:05d}.jpg", root / "labels" / "synthetic" / f"copy_paste_{index:05d}.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 96])
    if not success:
        raise ValueError("JPEG encoding failed")
    image_path.write_bytes(encoded.tobytes())
    label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    checked = inspect_pair({"source_image": str(image_path), "source_label": str(label_path)})
    return {"dataset": "synthetic_copy_paste", "label_authority": "deterministic_transform_of_reviewed_polygons",
            "source_split": "train", "output_split": "synthetic", "output_image": str(image_path), "output_label": str(label_path),
            "sha256": checked["sha256"], "label_sha256": checked["label_sha256"], "width": w, "height": h,
            "annotations": len(rows), "background_source_id": background["source_id"], "background_sha256": background["sha256"],
            "background_source_image": background["source_image"], "floor_roi_normalized": list(rect),
            "floor_roi_method": "conservative_rectangle_visually_checked_20260923", "placements": placements,
            "purpose": "synthetic_train_only", "image_storage": "new_generated"}


def contact_sheet(items: list[dict], target: Path, maximum: int = 24) -> None:
    if not items:
        return
    indices = np.linspace(0, len(items) - 1, min(maximum, len(items))).round().astype(int)
    tiles = []
    for index in indices:
        item = items[int(index)]
        image = read_image(Path(item["output_image"]))
        h, w = image.shape[:2]
        polygons = parse_rows(Path(item["output_label"]).read_text(encoding="utf-8-sig"), item["output_label"])
        overlay = image.copy()
        pixel_polys = [np.rint(poly * [w, h]).astype(np.int32) for poly in polygons]
        cv2.fillPoly(overlay, pixel_polys, (20, 215, 90))
        image = cv2.addWeighted(image, .78, overlay, .22, 0)
        cv2.polylines(image, pixel_polys, True, (0, 245, 90), 1)
        image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
        image = cv2.copyMakeBorder(image, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
        cv2.putText(image, f"{Path(item['output_image']).stem} | {len(polygons)} masks", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .43, (235, 235, 235), 1)
        tiles.append(image)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[i:i+4], axis=1) for i in range(0, len(tiles), 4)])
    success, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not success:
        raise ValueError("Contact sheet encoding failed")
    target.write_bytes(encoded.tobytes())


def write_view(root: Path, name: str, train: list[dict], val: str = "images/val") -> None:
    text_path = root / (name + "_train.txt")
    text_path.write_text("\n".join(Path(item["output_image"]).as_posix() for item in train) + "\n", encoding="utf-8")
    config = {"path": str(root), "train": text_path.name, "val": val, "names": {0: "sweepable_litter"}}
    (root / (name + ".yaml")).write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def build(args: argparse.Namespace) -> dict:
    project, output = args.source_project.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if args.reviewed_repeat < 1 or args.synthetic_count < 0:
        raise ValueError("reviewed-repeat must be positive; synthetic-count must be nonnegative")
    print("Validating original public images and CVAT-reviewed polygons...", flush=True)
    base, base_provenance = load_base(project)
    reviewed, archives = load_reviewed(project)
    holdout_names = parse_holdout(args.holdout_reviewed_names)
    known_names = {Path(item["original_name"]).stem for item in reviewed}
    if holdout_names - known_names:
        raise ValueError(f"Unknown reviewed holdout names: {sorted(holdout_names - known_names)}")
    holdout = [item for item in reviewed if Path(item["original_name"]).stem in holdout_names]
    holdout_hashes = {item["sha256"] for item in holdout}
    public_val = [item for item in base if item["source_split"] == "val"]
    val_hashes = {item["sha256"] for item in public_val}
    if holdout_hashes & val_hashes:
        raise ValueError("Factory holdout unexpectedly duplicates public validation")
    excluded, base_train, reviewed_train = [], [], []
    for item in base:
        if item["source_split"] == "val":
            continue
        if item["sha256"] in holdout_hashes | val_hashes:
            excluded.append({**item, "reason": "content_hash_duplicate_of_holdout_or_public_val"})
        else:
            base_train.append(item)
    reviewed_hashes = set()
    for item in reviewed:
        if item["sha256"] in holdout_hashes | val_hashes:
            continue
        if item["sha256"] in reviewed_hashes:
            raise ValueError("Reviewed source images duplicate each other; resolve label precedence first")
        reviewed_hashes.add(item["sha256"])
        reviewed_train.append(item)
    if not reviewed_train:
        raise ValueError("No reviewed images remain for training")
    # Replace duplicate base negative copies with their newer reviewed annotation.
    retained_base = []
    for item in base_train:
        if item["sha256"] in reviewed_hashes:
            excluded.append({**item, "reason": "prefer_newer_cvat_review_for_identical_image"})
        else:
            retained_base.append(item)
    base_train = retained_base
    backgrounds = [item for item in base_train if item["dataset"] == "factory_negative"]
    # An identical older approved background is safe only if CVAT also confirms empty.
    backgrounds += [item for item in reviewed_train if not item["annotations"] and any(
        original["dataset"] == "factory_negative" and original["sha256"] == item["sha256"] for original in base)]
    for item in backgrounds:
        number = int(Path(item["source_image"]).stem.rsplit("_", 1)[1])
        if number not in FLOOR_RECTS:
            raise ValueError(f"No inspected floor ROI for approved background {item['source_image']}")
    if args.synthetic_count and not backgrounds:
        raise ValueError("No reviewed factory negative backgrounds remain")
    assets, rejected_assets = build_assets(reviewed_train) if args.synthetic_count else ([], [])
    output.mkdir(parents=True, exist_ok=False)
    (output / "BUILDING.json").write_text(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed}), encoding="utf-8")
    print(f"Materializing {len(base_train)} base train, {len(public_val)} public val, {len(reviewed_train)} reviewed, {len(holdout)} factory holdout images...", flush=True)
    items, common = [], []
    for item in base_train + public_val:
        split = item["source_split"]
        result = materialize(item, output, split, Path(item["source_image"]).stem, "public_or_approved_negative")
        items.append(result)
        if split == "train":
            common.append(result)
    for item in reviewed_train:
        for repeat in range(1, args.reviewed_repeat + 1):
            result = materialize(item, output, "train", f"reviewed_{item['dataset']}_r{repeat:02d}_{Path(item['original_name']).stem}", "reviewed_weighted_train", repeat)
            items.append(result)
            common.append(result)
    holdout_items = []
    for item in holdout:
        result = materialize(item, output, "holdout", f"{item['dataset']}_{Path(item['original_name']).stem}", "factory_holdout_only")
        holdout_items.append(result)
        items.append(result)
    rng = random.Random(args.seed)
    synthetic = []
    for index in range(args.synthetic_count):
        synthetic.append(synthetic_sample(index, output, backgrounds, assets, rng))
        if (index + 1) % 50 == 0:
            print(f"Copy/paste: {index+1}/{args.synthetic_count}", flush=True)
    items.extend(synthetic)
    # Equal total examples and minibatch count: control gets resampled reviewed real
    # images in place of synthetic ones. This is sampling, not new independent data.
    control = []
    control_pool = reviewed_train.copy()
    control_rng = random.Random(args.seed + 1)
    control_rng.shuffle(control_pool)
    for index in range(args.synthetic_count):
        item = control_pool[index % len(control_pool)]
        result = materialize(item, output, "control", f"matched_real_{index:05d}_{item['dataset']}_{Path(item['original_name']).stem}", "matched_budget_real_resample", 1 + index // len(control_pool))
        control.append(result)
    items.extend(control)
    print(f"Rechecking hashes, decoding, and labels for all {len(items)} materialized image/label pairs...", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(verify_materialized, items))
    write_view(output, "dataset_baseline", common + control)
    write_view(output, "dataset_synthetic", common + synthetic)
    if holdout_items:
        write_view(output, "factory_holdout", common, "images/holdout")
    contact_sheet(synthetic, output / "contact_sheet.jpg")
    contact_sheet(holdout_items, output / "factory_holdout_contact_sheet.jpg", 32)
    summary = {"base_train_images": len(base_train), "public_val_images": len(public_val), "reviewed_unique_train_images": len(reviewed_train),
               "reviewed_unique_train_instances": sum(item["annotations"] for item in reviewed_train), "reviewed_repeat": args.reviewed_repeat,
               "factory_holdout_images": len(holdout_items), "factory_holdout_instances": sum(item["annotations"] for item in holdout),
               "synthetic_images": len(synthetic), "synthetic_instances": sum(item["annotations"] for item in synthetic),
               "synthetic_assets": len(assets), "backgrounds": len(backgrounds), "matched_real_resamples": len(control),
               "baseline_train_samples": len(common) + len(control), "synthetic_train_samples": len(common) + len(synthetic),
               "unique_real_training_images": len({item["sha256"] for item in base_train + reviewed_train}),
               "image_hardlinks": sum(item.get("image_storage") == "hardlink" for item in items), "excluded_duplicate_base_images": len(excluded)}
    used_train_hashes = {item["sha256"] for item in base_train + reviewed_train}
    synthetic_source_hashes = {entry["source_image_sha256"] for item in synthetic for entry in item["placements"]} | {item["background_sha256"] for item in synthetic}
    if (used_train_hashes | synthetic_source_hashes) & (holdout_hashes | val_hashes):
        raise AssertionError("Train/holdout provenance leakage detected")
    validation = {"status": "passed", "all_unique_source_images_decoded": len(base) + len(reviewed), "all_generated_images_decoded": len(synthetic),
                  "all_materialized_images_decoded_and_hash_verified": len(items),
                  "all_source_and_generated_labels_validated": True, "public_val_preserved_byte_for_byte": True,
                  "public_val_images": len(public_val), "all_materialized_label_hashes_verified": True,
                  "train_public_val_exact_content_overlap": 0, "train_factory_holdout_exact_content_overlap": 0,
                  "synthetic_source_holdout_overlap": 0, "baseline_and_synthetic_sample_counts_match": len(control) == len(synthetic),
                  "no_pseudo_labels_consumed": True, "no_seed_rectangle_labels_consumed": True,
                  "temporal_scene_independence_guaranteed": False,
                  "caveats": ["Factory holdout is from the same short video and is a small diagnostic, not independent deployment proof.",
                              "Copy/paste quality and floor rectangles require contact-sheet inspection; labels are deterministic transformed reviewed polygons.",
                              "Hardlinked images must remain immutable; labels are separate copies and source caches are not linked."]}
    manifest = {"version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed, "source_project": str(project),
                "output": str(output), "class_names": {"0": "sweepable_litter"}, "summary": summary, "validation": validation,
                "label_policy": "Only original public annotations, explicitly approved negatives, and CVAT-reviewed polygons; no predicted label is training truth.",
                "base_provenance": base_provenance, "reviewed_archives": archives, "holdout_reviewed_names": sorted(holdout_names),
                "floor_rectangles": {str(key): value for key, value in FLOOR_RECTS.items()}, "rejected_copy_paste_assets": rejected_assets,
                "excluded_base_images": excluded, "items": items}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "READY.json").write_text(json.dumps({"status": "ready", "summary": summary, "visual_acceptance_pending": bool(synthetic)}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "BUILDING.json").unlink()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewed-repeat", type=int, default=6)
    parser.add_argument("--synthetic-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--holdout-reviewed-names", help="Comma-separated frame basenames or JSON list/dict path; excluded from all train backgrounds and patches")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
