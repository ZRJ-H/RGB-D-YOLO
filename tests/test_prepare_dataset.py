from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from garbage_auto_annotation.prepare import prepare_combined_dataset, prepare_dwsd_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color).save(path)


def _make_taco_fixture(root: Path) -> None:
    _write_rgb(root / "batch_1" / "one.jpg", (255, 0, 0))
    _write_rgb(root / "batch_1" / "two.jpg", (0, 255, 0))
    annotations = {
        "images": [
            {"id": 1, "file_name": "batch_1/one.jpg", "width": 20, "height": 20},
            {"id": 2, "file_name": "batch_1/two.jpg", "width": 20, "height": 20},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 4, "segmentation": [[2, 2, 10, 2, 10, 10, 2, 10]]},
            {"id": 11, "image_id": 2, "category_id": 33, "segmentation": [[5, 5, 15, 5, 15, 15, 5, 15]]},
        ],
        "categories": [
            {"id": 4, "name": "Other plastic bottle"},
            {"id": 33, "name": "Normal paper"},
        ],
    }
    (root / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")


def _make_dwsd_fixture(root: Path) -> None:
    for split, stem in (("train", "alpha"), ("test", "beta")):
        _write_rgb(root / split / "images" / f"img_{stem}.jpg", (0, 0, 255))
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[4:12, 4:12] = 9
        mask_path = root / split / "masks" / f"mask_{stem}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(mask_path)


def test_prepare_combined_dataset_writes_valid_yolo_layout(tmp_path: Path) -> None:
    taco = tmp_path / "taco"
    dwsd = tmp_path / "dwsd"
    output = tmp_path / "output"
    taco.mkdir()
    dwsd.mkdir()
    _make_taco_fixture(taco)
    _make_dwsd_fixture(dwsd)

    summary = prepare_combined_dataset(
        taco_source=taco,
        dwsd_source=dwsd,
        output=output,
        seed=42,
        min_mask_area=16,
    )

    assert summary["images_written"] == 4
    assert summary["annotations_written"] == 4
    dataset = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert dataset["names"] == {0: "sweepable_litter"}

    image_stems = {
        path.stem
        for split in ("train", "val")
        for path in (output / "images" / split).glob("*")
    }
    label_stems = {
        path.stem
        for split in ("train", "val")
        for path in (output / "labels" / split).glob("*.txt")
    }
    assert image_stems == label_stems


def test_prepare_combined_dataset_has_no_source_split_leakage(tmp_path: Path) -> None:
    taco = tmp_path / "taco"
    dwsd = tmp_path / "dwsd"
    output = tmp_path / "output"
    taco.mkdir()
    dwsd.mkdir()
    _make_taco_fixture(taco)
    _make_dwsd_fixture(dwsd)

    prepare_combined_dataset(taco, dwsd, output, seed=42, min_mask_area=16)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    train_sources = {
        item["source_id"] for item in manifest["items"] if item["split"] == "train"
    }
    val_sources = {
        item["source_id"] for item in manifest["items"] if item["split"] == "val"
    }

    assert train_sources.isdisjoint(val_sources)
    assert {item["dataset"] for item in manifest["items"]} == {"taco", "dwsd"}


def test_prepare_cli_shows_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "prepare_garbage_dataset.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--taco-source" in completed.stdout
    assert "--dwsd-source" in completed.stdout
    assert "--dry-run" in completed.stdout


def test_prepare_dwsd_dataset_can_build_interim_dataset(tmp_path: Path) -> None:
    dwsd = tmp_path / "dwsd"
    output = tmp_path / "dwsd_output"
    dwsd.mkdir()
    _make_dwsd_fixture(dwsd)

    summary = prepare_dwsd_dataset(dwsd, output, min_mask_area=16)

    assert summary == {
        "images_written": 2,
        "annotations_written": 2,
        "rejected_annotations": 0,
    }
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert {item["dataset"] for item in manifest["items"]} == {"dwsd"}
