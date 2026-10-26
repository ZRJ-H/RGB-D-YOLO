"""Prepare a verified build context and deterministic application layer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deployment/jetson"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
    base = json.loads((SOURCE / "base-image.json").read_text(encoding="utf-8"))
    release = ROOT / "releases" / base["source_release"]
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["sha256"].items():
        path = (release / name).resolve()
        if not path.is_relative_to(release.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Release integrity failure: " + name)
    context = target / "context"
    context.mkdir()
    for name in (*manifest["sha256"], "manifest.json"):
        destination = context / "app" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(release / name, destination)
    for name in ("Dockerfile", ".dockerignore", "base-image.json", "run_jetson.sh", "README.md"):
        shutil.copy2(SOURCE / name, context / name)
    shutil.copytree(SOURCE / "container", context / "container", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    layer = target / "application-layer.tar"
    with tarfile.open(layer, "x", format=tarfile.PAX_FORMAT) as archive:
        for directory, prefix in ((context / "app", "opt/yolo-depth"), (context / "container", "opt/yolo-depth/container")):
            for path in sorted(directory.rglob("*")):
                if not path.is_file():
                    continue
                info = archive.gettarinfo(str(path), (Path(prefix) / path.relative_to(directory)).as_posix())
                info.uid = info.gid = 0
                info.uname = info.gname = "root"
                info.mtime = 0
                info.mode = 0o644
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
    record = {"base": base, "app_layer_sha256": hashlib.sha256(layer.read_bytes()).hexdigest(),
        "verified_release_files": len(manifest["sha256"]), "build_method": "pinned base + application layer; no dependency replacement"}
    (target / "preparation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
