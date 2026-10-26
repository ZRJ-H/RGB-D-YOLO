"""Compare two segmentation checkpoints and write an honest local acceptance report.

Validation data are staged inside the new output directory, so Ultralytics caches
and plots never modify the source datasets. Unlabelled factory images are only
qualitative evidence; prediction counts are not accuracy measurements.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import traceback


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
METRIC_NAMES = ("precision", "recall", "map50", "map50_95")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "data", "factory-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--factory-labeled-data", type=Path,
                        help="Optional YOLO dataset YAML with a reviewed val split")
    parser.add_argument("--training-data", type=Path, action="append", default=[],
                        help="Additional known training YAML; repeat for historical datasets")
    parser.add_argument("--factory-note", default="",
                        help="Dataset provenance/temporal-neighbour caveat included in JSON and HTML")
    parser.add_argument("--reuse-local-data", action="store_true",
                        help="Use validation data directly only when YAML, images and labels are inside this workspace's datasets directory")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Factory qualitative prediction confidence, not AP cutoff")
    parser.add_argument("--iou", type=float, default=0.35,
                        help="Factory qualitative NMS IoU")
    parser.add_argument("--val-conf", type=float, default=0.001)
    parser.add_argument("--val-iou", type=float, default=0.7)
    parser.add_argument("--max-factory-images", type=int, default=0,
                        help="0 means all; otherwise select evenly spaced files")
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Audit files and hashes without loading models or using the GPU")
    args = parser.parse_args(argv)
    if args.imgsz < 32 or args.batch < 1 or args.panel_width < 160:
        parser.error("imgsz >= 32, batch >= 1, panel-width >= 160 required")
    if args.max_factory_images < 0 or args.video_fps <= 0:
        parser.error("max-factory-images >= 0 and video-fps > 0 required")
    if not all(0 <= x <= 1 for x in (args.conf, args.iou, args.val_conf, args.val_iou)):
        parser.error("confidence and IoU values must be in [0, 1]")
    return args


def read_yaml(path):
    import yaml
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Not a YAML mapping: {path}")
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def resolve_relative(path, base):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def image_files(source):
    source = Path(source).resolve()
    if source.is_dir():
        return sorted(p.resolve() for p in source.rglob("*")
                      if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    if source.is_file() and source.suffix.lower() == ".txt":
        paths = []
        for line in source.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                paths.extend(image_files(resolve_relative(line.strip(), source.parent)))
        return paths
    raise ValueError(f"Expected an existing image, directory, or image-list TXT: {source}")


def split_images(data_path, split):
    data_path = Path(data_path).resolve()
    config = read_yaml(data_path)
    root = resolve_relative(config.get("path", "."), data_path.parent)
    entries = config.get(split)
    if entries is None:
        raise ValueError(f"Missing {split!r} split in {data_path}")
    if not isinstance(entries, list):
        entries = [entries]
    paths = []
    for entry in entries:
        paths.extend(image_files(resolve_relative(entry, root)))
    # Repetition is useful in training but must not inflate validation metrics.
    return list(dict.fromkeys(paths)), config


def label_for_image(image):
    parts = list(image.parts)
    matches = [i for i, value in enumerate(parts) if value.lower() == "images"]
    if not matches:
        raise ValueError(f"Cannot find YOLO images/labels layout for {image}")
    parts[matches[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value):
    result = float(value)
    return result if math.isfinite(result) else None


def checkpoint_training_yaml(weight):
    args_path = Path(weight).resolve().parent.parent / "args.yaml"
    if args_path.is_file():
        config = read_yaml(args_path)
        raw = config.get("data")
        if raw:
            data_path = Path(raw)
            if data_path.is_absolute() and data_path.is_file():
                return data_path.resolve()
    return None


def audit_inputs(args):
    for path in (args.baseline, args.candidate, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    public_images, public_config = split_images(args.data, "val")
    if not public_images:
        raise ValueError("Public val split is empty")
    labelled_images, labelled_config = [], None
    if args.factory_labeled_data:
        labelled_images, labelled_config = split_images(args.factory_labeled_data, "val")
        if not labelled_images:
            raise ValueError("Factory labelled val split is empty")
    for path in public_images + labelled_images:
        label = label_for_image(path)
        if not label.is_file():
            raise FileNotFoundError(f"Validation needs an explicit label, even for negatives: {label}")
    factory_images = list(dict.fromkeys(image_files(args.factory_source)))
    if not factory_images:
        raise ValueError("Factory source contains no images")
    if 0 < args.max_factory_images < len(factory_images):
        count = args.max_factory_images
        indices = [round(i * (len(factory_images) - 1) / max(1, count - 1)) for i in range(count)]
        factory_images = [factory_images[i] for i in indices]

    training_yamls = [args.data.resolve(), *(p.resolve() for p in args.training_data)]
    inferred = {}
    for role in ("baseline", "candidate"):
        found = checkpoint_training_yaml(getattr(args, role))
        inferred[role] = str(found) if found else None
        if found:
            training_yamls.append(found)
    training_yamls = list(dict.fromkeys(training_yamls))
    training_images = []
    warnings = []
    for path in training_yamls:
        # A labelled evaluation-only YAML may omit train, but supplied training
        # YAML files must be real and readable rather than silently ignored.
        found, _ = split_images(path, "train")
        training_images.extend(found)
    training_images = list(dict.fromkeys(training_images))
    hashes = {}
    by_hash = {}
    train_frames = {}
    for path in training_images:
        digest = sha256(path)
        hashes[str(path)] = digest
        by_hash.setdefault(digest, []).append(str(path))
        frame = re.search(r"frame_\d+", path.stem)
        if frame:
            train_frames.setdefault(frame.group(), []).append(str(path))

    def exposure(paths):
        rows = []
        for path in paths:
            digest = hashes.get(str(path)) or sha256(path)
            frame = re.search(r"frame_\d+", path.stem)
            exact = by_hash.get(digest, [])
            # Frame IDs can collide across bags; this is a warning, not proof.
            possible = train_frames.get(frame.group(), []) if frame and not exact else []
            nearest_frame_gap = None
            if frame and train_frames:
                number = int(frame.group().split("_")[-1])
                nearest_frame_gap = min(abs(number - int(key.split("_")[-1])) for key in train_frames)
            rows.append({"image": str(path), "sha256": digest,
                         "exact_training_matches": exact,
                         "possible_same_frame_id_training_matches": possible,
                         "nearest_known_training_frame_id_gap": nearest_frame_gap,
                         "frame_gap_note": "Numeric filename gap only; meaningful temporally only when the capture source and sampling rate match." if frame else None,
                         "exposure": "seen_exact" if exact else
                         "possible_frame_id_match" if possible else "no_exact_match_known"})
        return {"images": len(rows),
                "exact_training_overlap": sum(bool(r["exact_training_matches"]) for r in rows),
                "possible_frame_id_overlap": sum(bool(r["possible_same_frame_id_training_matches"]) for r in rows),
                "items": rows}

    warnings.extend([
        "Training audit covers only the supplied/inferred datasets, not every pretraining or historical checkpoint ancestor.",
        "No exact hash overlap does not establish independent acquisition; neighbouring frames and reused backgrounds can be correlated.",
        "Unlabelled factory predictions are qualitative only. Fewer predictions is not evidence of fewer false positives.",
        "Public validation metrics are for version comparison; repeated model selection uses this split and it is not a final untouched test.",
        "Factory frames from test_camera1 share a capture sequence with existing training data; report same-bag regression, not independent-site generalization.",
    ])
    if args.factory_note:
        warnings.append(args.factory_note)
    audit = {"training_yaml_files": [str(p) for p in training_yamls],
             "inferred_checkpoint_training_yaml": inferred,
             "known_unique_training_files": len(training_images),
             "public_validation": exposure(public_images),
             "factory_qualitative": exposure(factory_images),
             "factory_labeled": exposure(labelled_images) if labelled_images else None,
             "factory_provenance_note": args.factory_note or None,
             "limitations": warnings}
    return audit, public_images, public_config, factory_images, labelled_images, labelled_config


def stage_validation(images, config, target):
    """Hardlink/copy validation inputs into an evaluation-only dataset."""
    import yaml
    image_dir = target / "images" / "val"
    label_dir = target / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (target / "images" / "train").mkdir()
    mappings = []
    for index, image in enumerate(images):
        name = f"{index:05d}_{image.name}"
        staged_image = image_dir / name
        try:
            os.link(image, staged_image)
        except OSError:
            shutil.copyfile(image, staged_image)
        staged_label = label_dir / Path(name).with_suffix(".txt")
        shutil.copyfile(label_for_image(image), staged_label)
        mappings.append({"source_image": str(image), "source_label": str(label_for_image(image)),
                         "staged_image": str(staged_image)})
    staged_config = {"path": str(target.resolve()), "train": "images/train", "val": "images/val",
                     "names": config["names"]}
    yaml_path = target / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(staged_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(target / "provenance.json", {"purpose": "evaluation_only_empty_train_split", "images": mappings})
    return yaml_path


def validation_data_path(source, images, config, target, reuse_local):
    if not reuse_local:
        return stage_validation(images, config, target)
    allowed = Path(__file__).resolve().parents[1] / "datasets"
    paths = [Path(source).resolve(), *images, *(label_for_image(image) for image in images)]
    for path in paths:
        if allowed != path and allowed not in path.parents:
            raise ValueError(f"--reuse-local-data requires all validation files inside {allowed}: {path}")
    return Path(source).resolve()


def metric_summary(metrics):
    result = {}
    for name, attr in (("box", "box"), ("mask", "seg")):
        metric = getattr(metrics, attr)
        result[name] = dict(zip(METRIC_NAMES, map(finite_float, metric.mean_results())))
    result["raw_results"] = {key: finite_float(value) for key, value in metrics.results_dict.items()}
    result["speed_ms_per_image"] = {key: finite_float(value) for key, value in metrics.speed.items()}
    return result


def compare_metrics(baseline, candidate):
    return {group: {key: None if baseline[group][key] is None or candidate[group][key] is None
                    else candidate[group][key] - baseline[group][key] for key in METRIC_NAMES}
            for group in ("box", "mask")}


def match_binary_masks(predicted_masks, confidences, predicted_classes, truth_masks, truth_classes,
                       iou_threshold=0.5):
    """Greedily match confidence-ranked detections to unmatched same-class GT."""
    import numpy as np
    if not (len(predicted_masks) == len(confidences) == len(predicted_classes)):
        raise ValueError("Prediction masks, confidence and classes must have equal lengths")
    if len(truth_masks) != len(truth_classes):
        raise ValueError("Ground-truth masks and classes must have equal lengths")
    unmatched = set(range(len(truth_masks)))
    matches, false_positive_indices = [], []
    for pred_index in sorted(range(len(confidences)), key=lambda i: (-float(confidences[i]), i)):
        best_iou, best_truth = -1.0, None
        predicted = np.asarray(predicted_masks[pred_index], dtype=bool)
        for truth_index in sorted(unmatched):
            if int(predicted_classes[pred_index]) != int(truth_classes[truth_index]):
                continue
            truth = np.asarray(truth_masks[truth_index], dtype=bool)
            if predicted.shape != truth.shape:
                raise ValueError("Prediction and GT masks must share the original image shape")
            union = int(np.count_nonzero(predicted | truth))
            iou = float(np.count_nonzero(predicted & truth) / union) if union else 0.0
            if iou > best_iou:
                best_iou, best_truth = iou, truth_index
        if best_truth is not None and best_iou >= iou_threshold:
            unmatched.remove(best_truth)
            matches.append({"prediction_index": pred_index, "ground_truth_index": best_truth,
                            "confidence": float(confidences[pred_index]), "mask_iou": best_iou})
        else:
            false_positive_indices.append(pred_index)
    return {"tp": len(matches), "fp": len(false_positive_indices), "fn": len(unmatched),
            "matches": matches, "false_positive_indices": false_positive_indices,
            "false_negative_indices": sorted(unmatched)}


def ground_truth_masks(image_path, shape):
    import cv2
    import numpy as np
    height, width = shape
    masks, classes = [], []
    label = label_for_image(image_path)
    for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) < 7 or len(values) % 2 != 1:
            raise ValueError(f"Expected segmentation polygon: {label}:{line_number}")
        polygon = np.asarray(values[1:], dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(polygon).all() or (polygon < 0).any() or (polygon > 1).any():
            raise ValueError(f"Invalid normalized polygon: {label}:{line_number}")
        polygon *= np.asarray([width, height])
        polygon = np.rint(polygon).astype(np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        if not mask.any():
            raise ValueError(f"Ground-truth polygon rasterized to empty: {label}:{line_number}")
        masks.append(mask.astype(bool))
        classes.append(int(values[0]))
    return masks, classes


def factory_operating_point(model, images, args):
    """Measure a fixed deployment-like threshold on the reviewed diagnostic set."""
    rows = []
    for image_path in images:
        result = model.predict(source=str(image_path), imgsz=args.imgsz, device=args.device,
                               conf=args.conf, iou=args.iou, max_det=300, half=False,
                               retina_masks=True, verbose=False, save=False)[0]
        truths, truth_classes = ground_truth_masks(image_path, result.orig_shape)
        boxes = result.boxes
        confs = boxes.conf.cpu().tolist() if boxes is not None else []
        classes = boxes.cls.cpu().tolist() if boxes is not None else []
        predictions = result.masks.data.cpu().numpy() > 0.5 if result.masks is not None else []
        if len(predictions) and tuple(predictions.shape[1:]) != tuple(result.orig_shape):
            raise ValueError("retina_masks did not return original-image-sized masks")
        matched = match_binary_masks(predictions, confs, classes, truths, truth_classes)
        rows.append({"image": str(image_path), "label": str(label_for_image(image_path)),
                     "shape_hw": list(result.orig_shape), "ground_truth_instances": len(truths),
                     "predicted_instances": len(confs), "negative_image": not truths, **matched})
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    negatives = [row for row in rows if row["negative_image"]]
    return {"scope": "same_video_near_neighbor_diagnostic_not_independent_test",
            "conf": args.conf, "nms_iou": args.iou, "mask_iou_threshold": 0.5,
            "retina_masks": True, "coordinates": "original_image_pixels",
            "matching": "confidence_descending_one_to_one_same_class_mask_iou_ge_0.5",
            "undefined_ratio_policy": "null_if_denominator_is_zero",
            "images": len(rows), "ground_truth_instances": tp + fn,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "negative_images": len(negatives), "negative_image_fp": sum(row["fp"] for row in negatives),
            "negative_image_details": [{"image": row["image"], "fp": row["fp"]} for row in negatives],
            "items": rows, "used_for_model_selection": False}


def predict_factory(model, images, role, output, args):
    from PIL import Image
    target = output / "factory" / role
    target.mkdir(parents=True)
    records = []
    for index, image_path in enumerate(images):
        result = model.predict(source=str(image_path), imgsz=args.imgsz, device=args.device,
                               conf=args.conf, iou=args.iou, max_det=300, half=False,
                               retina_masks=False, verbose=False, save=False)[0]
        rendered = result.plot(labels=True, boxes=True, masks=True, line_width=2)
        annotated = target / f"{index:04d}_{image_path.stem}.jpg"
        Image.fromarray(rendered[:, :, ::-1]).save(annotated, quality=92)
        boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
        confs = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
        classes = result.boxes.cls.cpu().tolist() if result.boxes is not None else []
        polygons = result.masks.xyn if result.masks is not None else []
        instances = [{"class_id": int(classes[i]), "confidence": float(confs[i]),
                      "box_xyxy": boxes[i],
                      "polygon_normalized": polygons[i].tolist() if i < len(polygons) else []}
                     for i in range(len(boxes))]
        records.append({"image": str(image_path), "annotated": str(annotated.relative_to(output)).replace("\\", "/"),
                        "count": len(instances), "predictions": instances})
        print(f"[{role}] factory {index + 1}/{len(images)}: {len(instances)} candidates", flush=True)
    write_json(target / "predictions.json", {"role": role, "qualitative_only": True, "images": records})
    return records


def build_comparison_images(images, predictions, audit, output, args):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    paired_dir = output / "paired"
    paired_dir.mkdir()
    pair_paths, thumbnails, video = [], [], None
    canvas_size = (args.panel_width * 3, round(args.panel_width * 0.75) + 60)
    try:
        font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
        title_font = ImageFont.truetype(str(font_path), 22)
        foot_font = ImageFont.truetype(str(font_path), 14)
    except OSError:
        title_font = foot_font = ImageFont.load_default()
    video_path = output / "factory_comparison.mp4"
    try:
        for index, source in enumerate(images):
            canvas = Image.new("RGB", canvas_size, "#101825")
            draw = ImageDraw.Draw(canvas)
            left, right = predictions["baseline"][index], predictions["candidate"][index]
            paths = [source, output / left["annotated"], output / right["annotated"]]
            captions = ["Original", f"Baseline | proposals={left['count']}", f"Candidate | proposals={right['count']}"]
            for panel, (path, caption) in enumerate(zip(paths, captions)):
                with Image.open(path) as raw:
                    view = ImageOps.contain(raw.convert("RGB"), (args.panel_width, canvas_size[1] - 60))
                x = panel * args.panel_width + (args.panel_width - view.width) // 2
                canvas.paste(view, (x, 38 + (canvas_size[1] - 60 - view.height) // 2))
                draw.text((panel * args.panel_width + 10, 6), caption, fill="white", font=title_font)
            exposure = audit["factory_qualitative"]["items"][index]["exposure"]
            draw.text((10, canvas_size[1] - 19), f"{source.name} | {exposure} | qualitative only; slideshow, not original timing", fill="#ffe38b", font=foot_font)
            paired = paired_dir / f"{index:04d}_{source.stem}.jpg"
            canvas.save(paired, quality=90)
            pair_paths.append(str(paired.relative_to(output)).replace("\\", "/"))
            thumbnail = ImageOps.contain(canvas, (960, 290))
            thumbnails.append(thumbnail.copy())
            if video is None:
                video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps, canvas_size)
                if not video.isOpened():
                    raise RuntimeError("Could not open comparison MP4 writer")
            video.write(np.asarray(canvas)[:, :, ::-1].copy())
    finally:
        if video is not None:
            video.release()
    sheets = []
    for start in range(0, len(thumbnails), 8):
        batch = thumbnails[start:start + 8]
        sheet = Image.new("RGB", (960, sum(p.height for p in batch)), "#101825")
        offset = 0
        for thumbnail in batch:
            sheet.paste(thumbnail, (0, offset))
            offset += thumbnail.height
        name = f"overview_{start // 8 + 1:02d}.jpg"
        sheet.save(output / name, quality=88)
        sheets.append(name)
    capture = cv2.VideoCapture(str(video_path))
    good, _ = capture.read()
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if not good or frame_count != len(images):
        raise RuntimeError(f"Comparison video verification failed: {frame_count} versus {len(images)}")
    return {"paired_images": pair_paths, "contact_sheets": sheets,
            "video": video_path.name, "video_frames": frame_count, "video_fps": args.video_fps,
            "video_kind": "visual_slideshow_not_original_timing"}


def report_html(report, output):
    esc = html.escape
    def metric_table(key, title):
        if key not in report["metrics"]["baseline"]:
            return ""
        rows = []
        for group in ("box", "mask"):
            for metric in METRIC_NAMES:
                b = report["metrics"]["baseline"][key][group][metric]
                c = report["metrics"]["candidate"][key][group][metric]
                fmt = lambda value: "N/A" if value is None else f"{value:.4f}"
                delta = "N/A" if b is None or c is None else f"{c - b:+.4f}"
                rows.append(f"<tr><td>{group} {metric}</td><td>{fmt(b)}</td><td>{fmt(c)}</td><td>{delta}</td></tr>")
        return f"<h2>{title}</h2><table><thead><tr><th>指标</th><th>原基线</th><th>新候选</th><th>变化</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"

    audit = report["audit"]
    cards = []
    for index, pair in enumerate(report["artifacts"]["paired_images"]):
        provenance = audit["factory_qualitative"]["items"][index]
        source = provenance["image"]
        exact = len(provenance["exact_training_matches"])
        status = f"已见训练图：{exact}个精确匹配" if exact else "未发现已知训练文件的精确哈希匹配；不等于独立测试"
        possible = provenance["possible_same_frame_id_training_matches"]
        if possible:
            status += f"；{len(possible)}个同帧编号待核对"
        cards.append(f'<article><h3>{index + 1}. {esc(Path(source).name)}</h3><p>{esc(status)}</p><a href="{esc(pair)}"><img loading="lazy" src="{esc(pair)}"></a><details><summary>来源与训练匹配</summary><pre>{esc(json.dumps(provenance, ensure_ascii=False, indent=2))}</pre></details></article>')
    label_note = "没有提供审核过的现场验证真值，因此没有计算现场准确率、召回率或AP。"
    if audit["factory_labeled"] is not None:
        labelled = audit["factory_labeled"]
        label_note = f"已提供{labelled['images']}张审核过的现场验证图；已知训练精确匹配{labelled['exact_training_overlap']}张。若来自同一bag，仅作为有限同场景回归证据，不代表独立场景泛化。"
    if audit.get("factory_provenance_note"):
        label_note += esc(audit["factory_provenance_note"])
    public = audit["public_validation"]
    operating_table = ""
    if report["metrics"]["baseline"].get("factory_operating_point"):
        rows = []
        for role, label in (("baseline", "原基线"), ("candidate", "新候选")):
            values = report["metrics"][role]["factory_operating_point"]
            precision = "N/A" if values["precision"] is None else f"{values['precision']:.4f}"
            recall = "N/A" if values["recall"] is None else f"{values['recall']:.4f}"
            rows.append(f"<tr><th>{label}</th><td>{values['tp']}</td><td>{values['fp']}</td><td>{values['fn']}</td><td>{precision}</td><td>{recall}</td><td>{values['negative_image_fp']}</td></tr>")
        operating_table = f"<h2>现场固定阈值诊断</h2><p>conf={report['settings']['conf']}、NMS IoU={report['settings']['iou']}；原图像素掩膜 IoU ≥ 0.5，按置信度降序一对一匹配审核真值。同视频近邻诊断，不参与选模。</p><table><tr><th>模型</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>负图FP</th></tr>{''.join(rows)}</table>"
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>垃圾分割模型 · 验收对照</title><style>
body{{max-width:1440px;margin:32px auto;padding:0 24px;background:#111827;color:#e5e7eb;font:16px/1.65 system-ui,sans-serif}}
h1,h2,h3{{color:#fff}}h1{{margin-bottom:8px}}a{{color:#7dd3fc}}.notice{{padding:18px;border:1px solid #dca842;background:#332c20;border-radius:10px}}
table{{border-collapse:collapse;width:100%;max-width:820px}}td,th{{padding:9px 16px;border-bottom:1px solid #374151;text-align:left}}th{{color:#93c5fd}}
img,video{{width:100%;border-radius:8px}}article{{margin:24px 0;padding:18px;background:#1f2937;border-radius:10px}}pre{{overflow-wrap:anywhere;white-space:pre-wrap;font-size:12px}}
.small{{font-size:14px;color:#bbc5d2}}details{{margin-top:10px}}code{{overflow-wrap:anywhere}}
</style><h1>垃圾分割模型：原基线 / 新候选</h1><p class="small">{esc(report['created_at'])} · 单类 sweepable_litter · 自动评估完成</p>
<div class="notice">候选模型等待验收，未自动替换现用模型。未标注现场画面只做目视对照：候选数量下降不能证明误检下降。已见训练图不能作为独立测试。{label_note}</div>
<p>原基线：<code>{esc(report['weights']['baseline']['path'])}</code><br>新候选：<code>{esc(report['weights']['candidate']['path'])}</code></p>
{metric_table('public_validation', '公共验证集（用于版本比较）')}
<p class="small">共{public['images']}张图，已知训练精确匹配{public['exact_training_overlap']}张。AP使用conf={report['settings']['val_conf']}、NMS IoU={report['settings']['val_iou']}；P/R为Ultralytics曲线选取的工作点，并非下方现场conf阈值下的P/R。该验证集已用于多轮模型选择，不是未触碰的最终测试集。</p>
{metric_table('factory_labeled', '有限现场真值回归（请结合来源与污染检查）')}
{operating_table}
<h2>现场图像对照</h2><p>从左到右：原图、原基线、新候选。相同conf={report['settings']['conf']}、NMS IoU={report['settings']['iou']}、imgsz={report['settings']['imgsz']}。重点检查鞋子/纸箱/亮斑误检，以及小碎纸、薄膜漏检。</p>
<p><a href="factory_comparison.mp4">对照图幻灯预览（{report['artifacts']['video_fps']}帧/秒，非原始视频时序）</a> · <a href="metrics.json">完整指标与来源JSON</a> · <a href="audit.json">训练重叠审计</a></p>
<video controls preload="metadata" src="factory_comparison.mp4"></video>
{''.join(cards)}<h2>评估边界</h2><ul>{''.join('<li>' + esc(line) + '</li>' for line in audit['limitations'])}</ul>
<p class="small">自动报告只描述测量结果，不保证模型已达到车端部署或现场安全要求。</p></html>'''
    (output / "report.html").write_text(content, encoding="utf-8")


def main(argv=None):
    args = arguments(argv)
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    print("Checking datasets and exact training-image overlaps...", flush=True)
    audit, public, public_config, factory, labelled, labelled_config = audit_inputs(args)
    args.output.mkdir(parents=True)
    write_json(args.output / "audit.json", audit)
    settings = {key: str(value) if isinstance(value, Path) else
                [str(p) for p in value] if key == "training_data" else value
                for key, value in vars(args).items()}
    report = {"schema_version": 1, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "status": "audited" if args.dry_run else "running", "settings": settings,
              "weights": {role: {"path": str(getattr(args, role).resolve()),
                                    "sha256": sha256(getattr(args, role))}
                          for role in ("baseline", "candidate")}, "audit": audit,
              "metrics": {}, "artifacts": {}, "promotion": "not_automatically_promoted",
              "evaluation_scope": {
                  "public_validation": "reused_version_comparison_benchmark_not_final_untouched_test",
                  "factory_labeled": "same_capture_sequence_near_neighbor_regression_not_independent_test" if labelled else "not_provided",
                  "factory_qualitative": "unlabelled_visual_comparison_no_accuracy_claim"}}
    write_json(args.output / "metrics.json", report)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "output": str(args.output),
                          "public_images": len(public), "factory_images": len(factory),
                          "factory_labelled_images": len(labelled)}, ensure_ascii=False), flush=True)
        return
    try:
        runtime = args.output / "runtime"
        runtime.mkdir()
        os.environ["YOLO_CONFIG_DIR"] = str(runtime)
        from ultralytics import YOLO, __version__
        import torch
        report["runtime"] = {"ultralytics": __version__, "torch": torch.__version__, "python": sys.version}
        staged = {"public_validation": validation_data_path(args.data, public, public_config,
                  args.output / "eval_data" / "public", args.reuse_local_data)}
        if labelled:
            staged["factory_labeled"] = validation_data_path(args.factory_labeled_data, labelled, labelled_config,
                args.output / "eval_data" / "factory_labeled", args.reuse_local_data)
        predictions = {}
        for role in ("baseline", "candidate"):
            model = YOLO(str(getattr(args, role).resolve()))
            if model.task != "segment":
                raise ValueError(f"Expected segmentation checkpoint for {role}, got {model.task}")
            report["metrics"][role] = {}
            for split, data in staged.items():
                metrics = model.val(data=str(data), split="val", imgsz=args.imgsz, batch=args.batch,
                                    device=args.device, workers=0, conf=args.val_conf, iou=args.val_iou,
                                    half=False, max_det=300, plots=True, save_json=False, verbose=False,
                                    cache=False, seed=42, deterministic=True,
                                    project=str(args.output / "validation"), name=f"{role}_{split}", exist_ok=False)
                report["metrics"][role][split] = metric_summary(metrics)
                write_json(args.output / "metrics.json", report)
            if labelled:
                report["metrics"][role]["factory_operating_point"] = factory_operating_point(model, labelled, args)
                write_json(args.output / "metrics.json", report)
            predictions[role] = predict_factory(model, factory, role, args.output, args)
            report["metrics"][role]["factory_qualitative"] = {
                "images": len(factory), "total_proposals": sum(row["count"] for row in predictions[role]),
                "per_image_counts": [row["count"] for row in predictions[role]], "accuracy_claim": False}
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report["deltas_candidate_minus_baseline"] = {
            split: compare_metrics(report["metrics"]["baseline"][split], report["metrics"]["candidate"][split])
            for split in staged}
        report["artifacts"] = build_comparison_images(factory, predictions, audit, args.output, args)
        report["status"] = "complete_acceptance_pending"
        write_json(args.output / "metrics.json", report)
        report_html(report, args.output)
        print(f"Acceptance report: {args.output / 'report.html'}", flush=True)
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        write_json(args.output / "metrics.json", report)
        raise


if __name__ == "__main__":
    main()
