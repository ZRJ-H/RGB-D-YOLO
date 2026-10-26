from __future__ import annotations

import argparse
import json
from pathlib import Path

from garbage_auto_annotation.prepare import prepare_combined_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert official TACO and DWSD sources to binary YOLO segmentation."
    )
    parser.add_argument("--taco-source", type=Path, required=True)
    parser.add_argument("--dwsd-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-mask-area", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for name, source in (("TACO", args.taco_source), ("DWSD", args.dwsd_source)):
        if not source.is_dir():
            raise FileNotFoundError(f"{name} source directory does not exist: {source}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    if args.dry_run:
        print(
            f"Dataset preparation preflight passed. taco={args.taco_source.resolve()}, "
            f"dwsd={args.dwsd_source.resolve()}, output={args.output.resolve()}"
        )
        return

    summary = prepare_combined_dataset(
        taco_source=args.taco_source,
        dwsd_source=args.dwsd_source,
        output=args.output,
        seed=args.seed,
        min_mask_area=args.min_mask_area,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
