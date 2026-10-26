from __future__ import annotations

import argparse
import json
from pathlib import Path

from garbage_auto_annotation.hard_negatives import copy_approved_negatives, make_contact_sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Review and add factory hard-negative frames.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-contact-sheet", type=Path, metavar="OUTPUT_JPG")
    mode.add_argument("--approved-json", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()

    if not args.source.is_dir():
        raise FileNotFoundError(f"Frame source directory does not exist: {args.source}")

    if args.make_contact_sheet:
        names = make_contact_sheet(args.source, args.make_contact_sheet, columns=args.columns)
        print(f"Contact sheet contains {len(names)} frames: {args.make_contact_sheet.resolve()}")
        return

    if args.dataset_root is None:
        parser.error("--dataset-root is required with --approved-json")
    approved_payload = json.loads(args.approved_json.read_text(encoding="utf-8"))
    approved = approved_payload.get("approved", approved_payload) if isinstance(approved_payload, dict) else approved_payload
    if not isinstance(approved, list) or not all(isinstance(item, str) for item in approved):
        raise ValueError("Approved JSON must be a list of image filenames or an object with an approved list")
    records = copy_approved_negatives(args.source, approved, args.dataset_root)
    result = {"approved": approved, "records": records}
    output = args.dataset_root / "hard_negatives.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Added {len(records)} hard-negative frames: {output.resolve()}")


if __name__ == "__main__":
    main()
