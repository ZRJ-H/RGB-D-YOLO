from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from garbage_auto_annotation.hard_negatives import (
    copy_approved_negatives,
    make_contact_sheet,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


def test_copy_approved_negative_creates_identical_image_and_empty_label(tmp_path: Path) -> None:
    source = tmp_path / "frames"
    dataset = tmp_path / "dataset"
    _write_image(source / "clean.jpg", (20, 40, 60))

    records = copy_approved_negatives(source, ["clean.jpg"], dataset)

    copied = dataset / "images" / "train" / "factory_negative_clean.jpg"
    label = dataset / "labels" / "train" / "factory_negative_clean.txt"
    assert hashlib.sha256(copied.read_bytes()).digest() == hashlib.sha256(
        (source / "clean.jpg").read_bytes()
    ).digest()
    assert label.read_bytes() == b""
    assert records[0]["source_id"] == "factory_negative:clean.jpg"


def test_copy_approved_negatives_rejects_path_traversal(tmp_path: Path) -> None:
    source = tmp_path / "frames"
    source.mkdir()

    with pytest.raises(ValueError, match="filename"):
        copy_approved_negatives(source, ["../escape.jpg"], tmp_path / "dataset")


def test_make_contact_sheet_includes_all_images(tmp_path: Path) -> None:
    source = tmp_path / "frames"
    _write_image(source / "one.jpg", (255, 0, 0))
    _write_image(source / "two.jpg", (0, 255, 0))
    output = tmp_path / "sheet.jpg"

    names = make_contact_sheet(source, output, columns=2)

    assert names == ["one.jpg", "two.jpg"]
    with Image.open(output) as sheet:
        assert sheet.width == 640
        assert sheet.height > 240


def test_hard_negative_cli_shows_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "select_hard_negatives.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--make-contact-sheet" in completed.stdout
    assert "--approved-json" in completed.stdout
