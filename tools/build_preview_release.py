"""Build a self-contained preview directory; seal only after smoke validation."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    dest = args.destination.resolve()
    if args.seal:
        report = json.loads((dest / "validation/report.json").read_text(encoding="utf-8"))
        if report.get("passed") is not True:
            raise SystemExit("Refusing to seal: validation did not pass")
        archive = dest.parent / (dest.name + ".zip")
        if archive.exists() or (dest / "manifest.json").exists():
            raise SystemExit("Refusing to overwrite an existing sealed release")
        files = sorted(p for p in dest.rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".runtime" not in p.parts and p.suffix != ".pyc")
        dump(dest / "manifest.json", {"version": "0.1.0-preview", "spatial_fusion": False,
            "sha256": {p.relative_to(dest).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as output:
            for path in files + [dest / "manifest.json"]:
                output.write(path, (Path(dest.name) / path.relative_to(dest)).as_posix())
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
        print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": digest}))
        return
    if args.weights is None or not args.weights.is_file() or args.samples is None or not args.samples.is_dir():
        parser.error("--weights and --samples are required for build")
    dest.mkdir(parents=True, exist_ok=False)
    mapping = {
        "run_perception.py": "run_perception.py",
        "run_depth_baseline.py": "run_depth_baseline.py",
        "benchmark_rgb.py": "benchmark_rgb.py",
        "tools/preview_release_README.md": "README.md",
        "tools/preview_smoke_test.py": "smoke_test.py",
        "tools/verify_release.py": "verify_release.py",
        "config/depth_baseline.yaml": "config/depth_baseline.yaml",
        "schemas/depth_perception_result.schema.json": "schemas/depth_perception_result.schema.json",
        "tests/test_depth_perception.py": "tests/test_depth_perception.py",
        "tests/test_benchmark_rgb.py": "tests/test_benchmark_rgb.py",
        "tests/test_preview_release.py": "tests/test_preview_release.py",
    }
    for source, target in mapping.items():
        path = dest / target
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, path)
    shutil.copytree(ROOT / "depth_perception", dest / "depth_perception", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (dest / "models").mkdir()
    shutil.copy2(args.weights, dest / "models/best.pt")
    (dest / "samples").mkdir()
    for i in range(6):
        for suffix in ("rgb.jpg", "depth_raw.png"):
            name = f"sample_{i:02d}_{suffix}"
            shutil.copy2(args.samples / name, dest / "samples" / name)
    records = json.loads((args.samples / "summary.json").read_text(encoding="utf-8"))
    dump(dest / "samples/provenance.json", {"source_bag": "test_camera1", "rotation_already_applied": 180,
        "spatial_registration": "unknown", "pairing": "nearest recorded bag timestamp", "records": records})
    (dest / "config/intrinsics.template.yaml").write_text(
        "# Fill with actual DEPTH camera calibration. null is intentionally unusable.\nwidth: null\nheight: null\nfx: null\nfy: null\ncx: null\ncy: null\n", encoding="utf-8")
    (dest / "requirements.txt").write_text("# Install platform-matched torch/torchvision first.\nultralytics==8.3.179\nnumpy>=1.24,<3\nopencv-python>=4.8,<5\nPyYAML==6.0.2\npsutil>=5.9,<8\n", encoding="utf-8")
    packages = {name: importlib.metadata.version(name) for name in
        ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "PyYAML", "psutil")}
    dump(dest / "environment.json", {"python": sys.version, "platform": platform.platform(), "packages": packages,
        "scope": "observed development environment, not a portable lockfile"})
    dump(dest / "model.json", {"architecture": "YOLO11s-seg", "classes": {"0": "sweepable_litter"},
        "experiment": "garbage_teacher_yolo11s_seg_reviewed20_r10_lr5e5_e10",
        "sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "model_role": "review-required pre-annotation", "board_validated": False})
    print(dest)


if __name__ == "__main__":
    main()
