from __future__ import annotations

import argparse
import json
from pathlib import Path

from garbage_auto_annotation.validation import validate_yolo_seg_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a binary YOLO segmentation dataset.")
    parser.add_argument("--data", required=True, type=str)
    args = parser.parse_args()
    report = validate_yolo_seg_dataset(Path(args.data))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
