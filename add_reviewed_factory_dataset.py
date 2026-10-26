from __future__ import annotations

import argparse
import json
from pathlib import Path

from garbage_auto_annotation.factory_seed import build_reviewed_factory_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone a YOLO segmentation dataset and add human-reviewed factory polygons."
    )
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    summary = build_reviewed_factory_dataset(
        args.base_dataset, args.archive, args.output, repeat=args.repeat
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
