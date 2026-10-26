from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml
from PIL import Image

from garbage_auto_annotation import factory_seed
from garbage_auto_annotation.factory_seed import (
    build_factory_seed_dataset,
    convert_yolo_box_row,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


def _make_base_dataset(root: Path) -> None:
    _write_image(root / "images" / "train" / "public_train.jpg", (255, 0, 0))
    _write_image(root / "images" / "val" / "public_val.jpg", (0, 255, 0))
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "labels" / "train" / "public_train.txt").write_text(
        "0 0.100000 0.100000 0.200000 0.100000 0.200000 0.200000\n",
        encoding="utf-8",
    )
    (root / "labels" / "val" / "public_val.txt").write_bytes(b"")
    (root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "sweepable_litter"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "class_names": {"0": "sweepable_litter"},
                "summary": {
                    "images_written": 2,
                    "annotations_written": 1,
                    "rejected_annotations": 0,
                },
                "items": [],
                "rejected": [],
            }
        ),
        encoding="utf-8",
    )


def test_convert_yolo_box_row_outputs_hand_checked_rectangle() -> None:
    # Catches swapped axes, half-width mistakes, or accidentally retaining box format.
    assert convert_yolo_box_row("0 0.500000 0.500000 0.400000 0.200000") == (
        "0 0.300000 0.400000 0.700000 0.400000 "
        "0.700000 0.600000 0.300000 0.600000"
    )


def test_convert_yolo_box_row_clamps_rectangle_to_image_boundary() -> None:
    # Catches out-of-range polygons when a source box touches an image edge.
    assert convert_yolo_box_row("0 0.050000 0.950000 0.200000 0.200000") == (
        "0 0.000000 0.850000 0.150000 0.850000 "
        "0.150000 1.000000 0.000000 1.000000"
    )


@pytest.mark.parametrize(
    "row",
    [
        "1 0.5 0.5 0.2 0.2",
        "0 0.5 0.5 -0.2 0.2",
        "0 0.5 0.5 0.2",
        "0 nan 0.5 0.2 0.2",
    ],
)
def test_convert_yolo_box_row_rejects_unsafe_source_labels(row: str) -> None:
    # Catches silent ingestion of another class or malformed coordinates.
    with pytest.raises(ValueError):
        convert_yolo_box_row(row)


def test_build_factory_seed_dataset_copies_base_and_adds_positive_and_negative(
    tmp_path: Path,
) -> None:
    # Catches edits to the base dataset, validation leakage, and missing empty labels.
    base = tmp_path / "base"
    source = tmp_path / "cvat_export"
    output = tmp_path / "seeded"
    _make_base_dataset(base)
    _write_image(source / "images" / "train" / "garbage-pilot" / "positive.jpg", (1, 2, 3))
    _write_image(source / "images" / "train" / "garbage-pilot" / "negative.jpg", (4, 5, 6))
    label = source / "labels" / "train" / "garbage-pilot" / "positive.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")

    summary = build_factory_seed_dataset(base, source, output)

    assert summary == {
        "factory_images": 2,
        "factory_positive_images": 1,
        "factory_negative_images": 1,
        "factory_annotations": 1,
        "images_written": 4,
        "annotations_written": 2,
    }
    assert (base / "images" / "train" / "factory_seed_garbage-pilot__positive.jpg").exists() is False
    assert (output / "images" / "val" / "public_val.jpg").is_file()
    positive = output / "labels" / "train" / "factory_seed_garbage-pilot__positive.txt"
    negative = output / "labels" / "train" / "factory_seed_garbage-pilot__negative.txt"
    assert positive.read_text(encoding="utf-8") == (
        "0 0.300000 0.400000 0.700000 0.400000 "
        "0.700000 0.600000 0.300000 0.600000\n"
    )
    assert negative.read_bytes() == b""
    dataset = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert Path(dataset["path"]) == output.resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert {item["source_id"] for item in manifest["items"]} == {
        "factory_seed:garbage-pilot/negative.jpg",
        "factory_seed:garbage-pilot/positive.jpg",
    }


def test_build_factory_seed_dataset_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    # Catches accidental destruction of a prior training dataset or run.
    base = tmp_path / "base"
    source = tmp_path / "source"
    output = tmp_path / "already_exists"
    _make_base_dataset(base)
    output.mkdir()

    with pytest.raises(FileExistsError):
        build_factory_seed_dataset(base, source, output)


def test_build_factory_seed_dataset_merges_existing_hard_negative_provenance(
    tmp_path: Path,
) -> None:
    # Catches a manifest that silently omits hard negatives already present in the base copy.
    base = tmp_path / "base"
    source = tmp_path / "source"
    output = tmp_path / "seeded"
    _make_base_dataset(base)
    (base / "hard_negatives.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "dataset": "factory_negative",
                        "source_id": "factory_negative:clean.jpg",
                        "split": "train",
                        "annotations": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_image(source / "images" / "train" / "positive.jpg", (1, 2, 3))
    label = source / "labels" / "train" / "positive.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")

    build_factory_seed_dataset(base, source, output)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert {item["source_id"] for item in manifest["items"]} == {
        "factory_negative:clean.jpg",
        "factory_seed:positive.jpg",
    }


def test_write_factory_focus_view_selects_only_domain_images(tmp_path: Path) -> None:
    # Catches public images leaking into the domain-weighted training view.
    dataset = tmp_path / "dataset"
    _make_base_dataset(dataset)
    _write_image(dataset / "images" / "train" / "factory_seed_positive.jpg", (1, 2, 3))
    _write_image(dataset / "images" / "train" / "factory_negative_clean.jpg", (4, 5, 6))
    (dataset / "labels" / "train" / "factory_seed_positive.txt").write_text(
        "0 0.3 0.4 0.7 0.4 0.7 0.6 0.3 0.6\n",
        encoding="utf-8",
    )
    (dataset / "labels" / "train" / "factory_negative_clean.txt").write_bytes(b"")

    summary = factory_seed.write_factory_focus_view(dataset)

    assert summary == {"factory_focus_images": 2, "factory_focus_annotations": 1}
    assert (dataset / "factory_focus_train.txt").read_text(encoding="utf-8") == (
        "./images/train/factory_negative_clean.jpg\n"
        "./images/train/factory_seed_positive.jpg\n"
    )
    payload = yaml.safe_load((dataset / "factory_focus.yaml").read_text(encoding="utf-8"))
    assert payload == {
        "path": str(dataset.resolve()),
        "train": "factory_focus_train.txt",
        "val": "images/val",
        "names": {0: "sweepable_litter"},
    }


def test_factory_seed_cli_shows_required_inputs() -> None:
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "add_factory_seed.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--base-dataset" in completed.stdout
    assert "--source" in completed.stdout
    assert "--output" in completed.stdout


def test_factory_focus_cli_shows_dataset_input() -> None:
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "make_factory_focus_view.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--dataset-root" in completed.stdout


def test_build_reviewed_factory_dataset_preserves_polygons_and_missing_labels_as_negative(
    tmp_path: Path,
) -> None:
    # Catches accidental box conversion and dropping reviewed empty factory scenes.
    base = tmp_path / "base"
    archive = tmp_path / "reviewed.zip"
    output = tmp_path / "reviewed_dataset"
    _make_base_dataset(base)
    positive = tmp_path / "positive.jpg"
    negative = tmp_path / "negative.jpg"
    _write_image(positive, (11, 12, 13))
    _write_image(negative, (21, 22, 23))
    polygon = "0 0.100000 0.200000 0.300000 0.200000 0.200000 0.400000\n"
    with zipfile.ZipFile(archive, "w") as destination:
        destination.writestr(
            "data.yaml",
            "names:\n  0: sweepable_litter\npath: .\ntrain: train.txt\n",
        )
        destination.write(positive, "images/train/positive.jpg")
        destination.write(negative, "images/train/negative.jpg")
        destination.writestr("labels/train/positive.txt", polygon)

    summary = factory_seed.build_reviewed_factory_dataset(base, archive, output)

    assert summary == {
        "reviewed_source_images": 2,
        "reviewed_images": 2,
        "reviewed_positive_images": 1,
        "reviewed_negative_images": 1,
        "reviewed_annotations": 1,
        "images_written": 4,
        "annotations_written": 2,
    }
    positive_label = output / "labels" / "train" / "factory_reviewed_positive.txt"
    negative_label = output / "labels" / "train" / "factory_reviewed_negative.txt"
    assert positive_label.read_text(encoding="utf-8") == polygon
    assert negative_label.read_bytes() == b""
    assert not (base / "images" / "train" / "factory_reviewed_positive.jpg").exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    reviewed = [item for item in manifest["items"] if item["dataset"] == "factory_reviewed"]
    assert {item["source_id"] for item in reviewed} == {
        "factory_reviewed:negative.jpg",
        "factory_reviewed:positive.jpg",
    }


def test_reviewed_factory_cli_shows_required_inputs() -> None:
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "add_reviewed_factory_dataset.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--base-dataset" in completed.stdout
    assert "--archive" in completed.stdout
    assert "--output" in completed.stdout
    assert "--repeat" in completed.stdout


def test_build_reviewed_factory_dataset_repeats_only_reviewed_training_images(
    tmp_path: Path,
) -> None:
    # Catches a repeat option that duplicates public/validation data or loses empty negatives.
    base = tmp_path / "base"
    archive = tmp_path / "reviewed.zip"
    output = tmp_path / "repeated"
    _make_base_dataset(base)
    positive = tmp_path / "positive.jpg"
    negative = tmp_path / "negative.jpg"
    _write_image(positive, (31, 32, 33))
    _write_image(negative, (41, 42, 43))
    with zipfile.ZipFile(archive, "w") as destination:
        destination.writestr(
            "data.yaml",
            "names:\n  0: sweepable_litter\npath: .\ntrain: train.txt\n",
        )
        destination.write(positive, "images/train/positive.jpg")
        destination.write(negative, "images/train/negative.jpg")
        destination.writestr(
            "labels/train/positive.txt",
            "0 0.100000 0.200000 0.300000 0.200000 0.200000 0.400000\n",
        )

    summary = factory_seed.build_reviewed_factory_dataset(base, archive, output, repeat=3)

    assert summary["reviewed_source_images"] == 2
    assert summary["reviewed_images"] == 6
    assert summary["reviewed_positive_images"] == 3
    assert summary["reviewed_negative_images"] == 3
    assert summary["images_written"] == 8
    assert len(list((output / "images" / "val").glob("*.jpg"))) == 1
    assert len(list((output / "images" / "train").glob("factory_reviewed_*.jpg"))) == 6
