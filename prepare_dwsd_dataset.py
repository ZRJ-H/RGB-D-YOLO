from __future__ import annotations

import argparse
import json
from pathlib import Path

from garbage_auto_annotation.prepare import prepare_dwsd_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an interim DWSD-only YOLO segmentation dataset.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-mask-area", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.source.is_dir():
        raise FileNotFoundError(f"DWSD source directory does not exist: {args.source}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    if args.dry_run:
        print(f"DWSD preflight passed. source={args.source.resolve()}, output={args.output.resolve()}")
        return
    print(json.dumps(prepare_dwsd_dataset(args.source, args.output, args.min_mask_area), indent=2))


if __name__ == "__main__":
    main()
