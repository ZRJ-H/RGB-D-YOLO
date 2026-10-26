from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_approved_negatives(
    source_dir: Path,
    approved_filenames: list[str],
    dataset_root: Path,
) -> list[dict]:
    """Copy only explicitly approved clean frames and create empty labels."""
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for filename in approved_filenames:
        relative = Path(filename)
        if relative.name != filename or relative.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Approved entry must be one image filename: {filename}")
        source = source_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Approved negative does not exist: {source}")

        output_stem = f"factory_negative_{source.stem}"
        destination = image_dir / f"{output_stem}{source.suffix.lower()}"
        label = label_dir / f"{output_stem}.txt"
        shutil.copy2(source, destination)
        label.write_bytes(b"")
        records.append(
            {
                "dataset": "factory_negative",
                "source_id": f"factory_negative:{filename}",
                "source_image": str(source.resolve()),
                "output_image": str(destination.resolve()),
                "output_label": str(label.resolve()),
                "split": "train",
                "annotations": 0,
                "sha256": _sha256(destination),
            }
        )
    return records


def make_contact_sheet(
    source_dir: Path,
    output: Path,
    columns: int = 4,
    tile_width: int = 320,
    tile_height: int = 240,
    caption_height: int = 26,
) -> list[str]:
    """Render every source image into a labeled review contact sheet."""
    if columns <= 0:
        raise ValueError("Columns must be positive")
    images = sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No images found under {source_dir}")

    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + caption_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, image_path in enumerate(images):
        with Image.open(image_path) as image:
            tile = ImageOps.contain(image.convert("RGB"), (tile_width, tile_height))
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + caption_height)
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + tile_height + 4), image_path.name, fill="black")

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)
    return [path.name for path in images]
