from __future__ import annotations

import argparse
import json
from pathlib import Path

from garbage_auto_annotation.provenance import build_source_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Write verified dataset source provenance as JSON.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--doi", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--license", required=True, dest="license_name")
    parser.add_argument("--file", action="append", type=Path, required=True, dest="files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    missing = [str(path) for path in args.files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Source files do not exist: {missing}")
    record = build_source_record(args.name, args.doi, args.url, args.files, args.license_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
