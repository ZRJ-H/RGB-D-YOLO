"""Check the release payload without importing inference dependencies."""
import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    errors = []
    for relative, expected in manifest["sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            errors.append(relative + ": missing or invalid path")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(relative + ": checksum mismatch")
    return errors, len(manifest["sha256"])


if __name__ == "__main__":
    errors, count = verify(Path(__file__).resolve().parent)
    print(json.dumps({"checked": count, "errors": errors}, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)
