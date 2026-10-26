from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import zipfile

import numpy as np

from garbage_auto_annotation.prediction import result_to_manifest_entry
from predict_garbage_teacher import _label_line, _write_cvat_archive


def test_result_becomes_prediction_manifest_entry() -> None:
    result = SimpleNamespace(
        boxes=SimpleNamespace(cls=np.array([0.0]), conf=np.array([0.73])),
        masks=SimpleNamespace(
            xyn=[np.array([[0.1, 0.2], [0.4, 0.2], [0.3, 0.5]], dtype=float)]
        ),
    )

    entry = result_to_manifest_entry("frame.jpg", result, {0: "sweepable_litter"})

    assert entry["image"] == "frame.jpg"
    assert entry["predictions"] == [
        {
            "class_id": 0,
            "class_name": "sweepable_litter",
            "confidence": 0.73,
            "polygon": [[0.1, 0.2], [0.4, 0.2], [0.3, 0.5]],
        }
    ]
    assert entry["rejections"] == []


def test_invalid_polygon_is_excluded_with_reason() -> None:
    result = SimpleNamespace(
        boxes=SimpleNamespace(cls=np.array([0.0]), conf=np.array([0.4])),
        masks=SimpleNamespace(xyn=[np.array([[0.1, 0.1], [0.2, 0.2]], dtype=float)]),
    )

    entry = result_to_manifest_entry("bad.jpg", result, {0: "sweepable_litter"})

    assert entry["predictions"] == []
    assert entry["rejections"][0]["reason"] == "polygon_has_fewer_than_3_unique_points"


def test_cvat_label_omits_confidence_but_audit_label_keeps_it() -> None:
    prediction = {
        "class_id": 0,
        "confidence": 0.73,
        "polygon": [[0.1, 0.2], [0.4, 0.2], [0.3, 0.5]],
    }
    assert len(_label_line(prediction, include_confidence=False).split()) == 7
    assert len(_label_line(prediction, include_confidence=True).split()) == 8


def test_cvat_archive_contains_file_list_and_optional_images(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg-placeholder")
    labels = tmp_path / "output" / "labels"
    labels.mkdir(parents=True)
    (labels / "frame.txt").write_text("0 0.1 0.1 0.9 0.1 0.5 0.9\n", encoding="utf-8")

    archive = _write_cvat_archive(tmp_path / "output", [image], include_images=False)

    with zipfile.ZipFile(archive) as handle:
        assert "train.txt" in handle.namelist()
        assert handle.read("train.txt").decode() == "images/train/frame.jpg\n"
        assert "labels/train/frame.txt" in handle.namelist()
        assert "images/train/frame.jpg" not in handle.namelist()
