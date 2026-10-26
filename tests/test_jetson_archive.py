"""Exercise offline image assembly with a small, explicitly non-runtime fixture."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def tar_bytes(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class ArchiveAssemblyTests(unittest.TestCase):
    def test_verified_layers_produce_expected_config_and_reject_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, prepared, output = (root / name for name in ("cache", "prepared", "output"))
            for path in (cache, prepared, output):
                path.mkdir()
            raw = tar_bytes({"fixture.txt": b"format-only fixture; not a runnable base"})
            compressed = gzip.compress(raw)
            digest = hashlib.sha256(compressed).hexdigest()
            (cache / (digest + ".tar.gz")).write_bytes(compressed)
            config = json.dumps({"architecture": "arm64", "os": "linux", "config": {"Env": ["EXISTING=keep"]},
                "history": [{}], "rootfs": {"type": "layers", "diff_ids": ["sha256:" + hashlib.sha256(raw).hexdigest()]}}).encode()
            manifest = json.dumps({"config": {"digest": "sha256:" + hashlib.sha256(config).hexdigest()},
                "layers": [{"digest": "sha256:" + digest, "size": len(compressed)}]}).encode()
            base_digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
            (cache / "base-config.json").write_bytes(config)
            (cache / "base-manifest.json").write_bytes(manifest)
            (cache / "download-complete.json").write_text(json.dumps({"manifest_digest": base_digest}))
            app = tar_bytes({"opt/yolo-depth/manifest.json": b'{"sha256": {}}',
                "opt/yolo-depth/container/doctor.py": b"# fixture\n",
                "opt/yolo-depth/container/entrypoint.py": b"# fixture\n"})
            (prepared / "application-layer.tar").write_bytes(app)
            prep = {"base": {"digest": base_digest, "target_image_tag": "archive-format-fixture:local",
                "target_host_jetpack": "6.2.1"}, "app_layer_sha256": hashlib.sha256(app).hexdigest()}
            (prepared / "preparation.json").write_text(json.dumps(prep))
            result = subprocess.run([sys.executable, str(ROOT / "tools/assemble_jetson_image.py"),
                "--download-dir", str(cache), "--prepared", str(prepared), "--destination", str(output)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            status = json.loads((output / "BUILD_STATUS.json").read_text())
            self.assertTrue(status["validation"]["archive_integrity_passed"])
            self.assertFalse(status["board_runtime_tested"])
            self.assertEqual(status["validation"]["layers"], 2)
            archive = output / status["image"]
            with tarfile.open(archive) as handle:
                exported = json.load(handle.extractfile("manifest.json"))[0]
                actual = json.load(handle.extractfile(exported["Config"]))
                self.assertIn("EXISTING=keep", actual["config"]["Env"])
                self.assertEqual(actual["config"]["Entrypoint"], ["python3", "/opt/yolo-depth/container/entrypoint.py"])
            # Verify that the checker detects a payload changed after preparation.
            prep["app_layer_sha256"] = "0" * 64
            (prepared / "preparation.json").write_text(json.dumps(prep))
            failed = subprocess.run([sys.executable, str(ROOT / "tools/check_jetson_archive.py"), str(archive),
                "--preparation", str(prepared / "preparation.json"), "--output", str(output / "must-not-exist.json")],
                capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((output / "must-not-exist.json").exists())


if __name__ == "__main__":
    unittest.main()
