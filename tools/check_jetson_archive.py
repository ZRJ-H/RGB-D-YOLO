"""Validate image metadata and application content without running ARM64 code."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest_stream(handle):
    digest = hashlib.sha256()
    while chunk := handle.read(4 * 1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite validation report")
    preparation = json.loads(args.preparation.read_text(encoding="utf-8"))
    expected = preparation["base"]
    report = {"archive_integrity_passed": False, "board_runtime_tested": False,
        "jetpack": expected["target_host_jetpack"], "image_tag": expected["target_image_tag"]}
    with tarfile.open(args.archive, "r:*") as archive:
        manifests = json.load(archive.extractfile("manifest.json"))
        if len(manifests) != 1:
            raise RuntimeError("Expected exactly one image")
        manifest = manifests[0]
        if expected["target_image_tag"] not in manifest["RepoTags"]:
            raise RuntimeError("Unexpected image tag")
        config_bytes = archive.extractfile(manifest["Config"]).read()
        config = json.loads(config_bytes)
        if (config["os"], config["architecture"]) != ("linux", "arm64"):
            raise RuntimeError("Expected Linux ARM64")
        if config["config"]["Entrypoint"] != ["python3", "/opt/yolo-depth/container/entrypoint.py"]:
            raise RuntimeError("Unexpected entrypoint")
        if config["config"]["Cmd"] != ["--help"] or config["config"]["WorkingDir"] != "/opt/yolo-depth":
            raise RuntimeError("Unexpected default command or workdir")
        if "YOLO_CONFIG_DIR=/tmp/yolo-depth-settings" not in config["config"]["Env"]:
            raise RuntimeError("Missing runtime settings isolation")
        diff_ids = config["rootfs"]["diff_ids"]
        if len(diff_ids) != len(manifest["Layers"]):
            raise RuntimeError("Layer count mismatch")
        if diff_ids[-1] != "sha256:" + preparation["app_layer_sha256"]:
            raise RuntimeError("Application layer differs from prepared payload")
        # Check every layer's uncompressed digest against the image config.
        for index, (name, expected_digest) in enumerate(zip(manifest["Layers"], diff_ids)):
            with archive.extractfile(name) as stream:
                compressed = stream.read(2) == b"\x1f\x8b"
                stream.seek(0)
                reader = gzip.GzipFile(fileobj=stream) if compressed else stream
                actual = "sha256:" + digest_stream(reader)
                if actual != expected_digest:
                    raise RuntimeError("Layer digest mismatch: " + name)
            print(f"Validated layer {index + 1}/{len(diff_ids)}", flush=True)
        with archive.extractfile(manifest["Layers"][-1]) as stream:
            with tarfile.open(fileobj=stream, mode="r:*") as layer:
                app_manifest = json.load(layer.extractfile("opt/yolo-depth/manifest.json"))
                for name, sha in app_manifest["sha256"].items():
                    if digest_stream(layer.extractfile("opt/yolo-depth/" + name)) != sha:
                        raise RuntimeError("Application file checksum mismatch: " + name)
                for name in ("container/doctor.py", "container/entrypoint.py"):
                    compile(layer.extractfile("opt/yolo-depth/" + name).read(), name, "exec")
        report.update({"archive_integrity_passed": True, "architecture": "linux/arm64", "layers": len(diff_ids),
            "image_id": "sha256:" + hashlib.sha256(config_bytes).hexdigest(),
            "verified_app_files": len(app_manifest["sha256"]), "bytes": args.archive.stat().st_size})
    with args.archive.open("rb") as stream:
        report["archive_sha256"] = digest_stream(stream)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
