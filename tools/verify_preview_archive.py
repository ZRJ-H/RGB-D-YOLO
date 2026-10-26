"""Relocate a built ZIP and exercise all entry modes with local dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    report = {"passed": False, "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(), "checks": checks}
    try:
        with zipfile.ZipFile(args.archive) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("ZIP CRC failure")
            for name in archive.namelist():
                if not (output / name).resolve().is_relative_to(output):
                    raise RuntimeError("invalid archive member")
            archive.extractall(output)
        root = next(p for p in output.iterdir() if p.is_dir())

        def run(name, arguments):
            proc = subprocess.run([sys.executable, *map(str, arguments)], cwd=output,
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            (output / (name + ".log")).write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
            checks.append({"name": name, "exit_code": proc.returncode})
            if proc.returncode:
                raise RuntimeError(name + " failed")

        run("checksums", [root / "verify_release.py"])
        run("relocated_rgb_cpu", [root / "run_perception.py", "rgb", "--rgb", root / "samples/sample_02_rgb.jpg",
            "--output", output / "rgb_cpu"])
        rgb = json.loads((output / "rgb_cpu/result.json").read_text(encoding="utf-8"))
        if len(rgb["rgb"]["instances"]) != 4 or rgb["depth"] is not None:
            raise RuntimeError("unexpected relocated RGB output")
        run("relocated_synthetic", [root / "run_perception.py", "depth-synthetic", "--output", output / "synthetic"])
        intrinsics = output / "synthetic_only_intrinsics.yaml"
        intrinsics.write_text("width: 96\nheight: 64\nfx: 120.0\nfy: 120.0\ncx: 47.5\ncy: 31.5\n", encoding="utf-8")
        # Small synthetic images need their matching sampling thresholds,
        # not the 640x400 real-camera defaults.
        synthetic = json.loads((output / "synthetic/result.json").read_text(encoding="utf-8"))
        config = output / "synthetic_only_config.json"
        config.write_text(json.dumps(synthetic["depth"]["detector_config"]), encoding="utf-8")
        run("calibrated_mode_synthetic_input", [root / "run_perception.py", "depth-calibrated", "--depth",
            output / "synthetic/synthetic_depth.png", "--intrinsics", intrinsics, "--depth-scale", ".001",
            "--config", config, "--output", output / "calibrated_synthetic"])
        result = json.loads((output / "calibrated_synthetic/result.json").read_text(encoding="utf-8"))
        objects = result["depth"]["observations"]
        if len(objects) != 1 or abs(objects[0]["height_m"] - .2) > .001:
            raise RuntimeError("calibrated entry synthetic geometry mismatch")
        run("checksums_after_execution", [root / "verify_release.py"])
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (output / "archive_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
