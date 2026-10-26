from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image

from garbage_auto_annotation.validation import validate_yolo_seg_dataset


def _make_dataset(root: Path) -> None:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (10, 10), "white").save(root / "images" / split / f"{split}.jpg")
        (root / "labels" / split / f"{split}.txt").write_text(
            "0 0.1 0.1 0.9 0.1 0.5 0.9\n", encoding="utf-8"
        )
    (root / "dataset.yaml").write_text(
        yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val", "names": {0: "sweepable_litter"}}),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps({"items": [{"source_id": "a", "split": "train"}, {"source_id": "b", "split": "val"}]}),
        encoding="utf-8",
    )


def test_valid_dataset_reports_counts(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    report = validate_yolo_seg_dataset(tmp_path / "dataset.yaml")
    assert report["valid"] is True
    assert report["images"] == 2
    assert report["instances"] == 2
    assert report["errors"] == []


def test_invalid_class_and_missing_label_are_reported(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    (tmp_path / "labels" / "train" / "train.txt").write_text(
        "1 0.1 0.1 0.9 0.1 0.5 0.9\n", encoding="utf-8"
    )
    (tmp_path / "labels" / "val" / "val.txt").unlink()
    report = validate_yolo_seg_dataset(tmp_path / "dataset.yaml")
    reasons = {error["reason"] for error in report["errors"]}
    assert report["valid"] is False
    assert "unexpected_class_id" in reasons
    assert "missing_label" in reasons
