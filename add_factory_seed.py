from __future__ import annotations

import argparse
import json
from pathlib import Path

from garbage_auto_annotation.factory_seed import build_factory_seed_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone a YOLO segmentation dataset and add CVAT box labels as rectangle masks."
    )
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = build_factory_seed_dataset(args.base_dataset, args.source, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
