"""Assemble a Docker-loadable archive from verified base blobs and app layer."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def add_bytes(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-dir", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    prep = json.loads((args.prepared / "preparation.json").read_text(encoding="utf-8"))
    base = prep["base"]
    completed = json.loads((args.download_dir / "download-complete.json").read_text(encoding="utf-8"))
    if completed["manifest_digest"] != base["digest"]:
        raise RuntimeError("Download digest differs from pinned base")
    manifest_bytes = (args.download_dir / "base-manifest.json").read_bytes()
    if "sha256:" + hashlib.sha256(manifest_bytes).hexdigest() != base["digest"]:
        raise RuntimeError("Base manifest integrity failure")
    manifest = json.loads(manifest_bytes)
    config_bytes = (args.download_dir / "base-config.json").read_bytes()
    if "sha256:" + hashlib.sha256(config_bytes).hexdigest() != manifest["config"]["digest"]:
        raise RuntimeError("Base config integrity failure")
    config = json.loads(config_bytes)
    if config["os"] != "linux" or config["architecture"] != "arm64":
        raise RuntimeError("Wrong base platform")
    app_layer = args.prepared / "application-layer.tar"
    if hashlib.sha256(app_layer.read_bytes()).hexdigest() != prep["app_layer_sha256"]:
        raise RuntimeError("Application layer changed since preparation")
    config["created"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    config["rootfs"]["diff_ids"].append("sha256:" + prep["app_layer_sha256"])
    config["history"].append({"created": config["created"], "created_by": "COPY verified YOLO-depth release; configure entrypoint"})
    runtime = config["config"]
    updates = {"YOLO_CONFIG_DIR": "/tmp/yolo-depth-settings", "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": "/tmp/yolo-depth-matplotlib"}
    env = dict(item.split("=", 1) for item in runtime.get("Env", []))
    env.update(updates)
    runtime["Env"] = [key + "=" + value for key, value in env.items()]
    runtime["Entrypoint"] = ["python3", "/opt/yolo-depth/container/entrypoint.py"]
    runtime["Cmd"] = ["--help"]
    runtime["WorkingDir"] = "/opt/yolo-depth"
    runtime.setdefault("Labels", {}).update({"org.opencontainers.image.title": "YOLO + Depth preview for Jetson Orin Nano",
        "org.opencontainers.image.version": "0.1.0-preview-jp6.2.1-arm64", "ai.yolo-depth.host-jetpack": "6.2.1",
        "ai.yolo-depth.board-validation": "pending", "org.opencontainers.image.base.digest": base["digest"]})
    payload = json.dumps(config, separators=(",", ":")).encode()
    config_name = hashlib.sha256(payload).hexdigest() + ".json"
    names = [layer["digest"].split(":")[1] + ".tar.gz" for layer in manifest["layers"]]
    output = args.destination / "yolo-depth_0.1.0-preview-jp6.2.1-arm64.tar"
    partial = args.destination / "yolo-depth_0.1.0-preview-jp6.2.1-arm64.assembling.tar"
    if output.exists() or partial.exists():
        raise RuntimeError("Refusing to overwrite archive")
    with tarfile.open(partial, "x", format=tarfile.PAX_FORMAT) as archive:
        add_bytes(archive, config_name, payload)
        add_bytes(archive, "manifest.json", json.dumps([{"Config": config_name,
            "RepoTags": [base["target_image_tag"]], "Layers": names + ["application-layer.tar"]}]).encode())
        for name in names:
            archive.add(args.download_dir / name, arcname=name, recursive=False)
        archive.add(app_layer, arcname="application-layer.tar", recursive=False)
    validation = args.destination / "image-validation.json"
    subprocess.run([sys.executable, str(ROOT / "tools/check_jetson_archive.py"), str(partial),
        "--preparation", str(args.prepared / "preparation.json"), "--output", str(validation)], check=True)
    report = json.loads(validation.read_text(encoding="utf-8"))
    partial.rename(output)
    (args.destination / (output.name + ".sha256")).write_text(report["archive_sha256"] + "  " + output.name + "\n", encoding="utf-8", newline="\n")
    for name in ("README.md", "run_jetson.sh", "base-image.json"):
        shutil.copy2(ROOT / "deployment/jetson" / name, args.destination / name)
    status = {"status": "archive_verified_board_acceptance_pending", "image": output.name,
        "base_digest": base["digest"], "tag": base["target_image_tag"], "validation": report,
        "build_method": "verified registry layers plus COPY-only application layer; Docker image archive format",
        "is_flashable_system_image": False, "host_jetpack": "6.2.1", "board_runtime_tested": False}
    for destination in (args.destination, args.prepared):
        (destination / "BUILD_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status))


if __name__ == "__main__":
    main()
