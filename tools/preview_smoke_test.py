"""Run packaged tests and real inference; writes machine-readable evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "device": args.device, "checks": [], "accuracy_evaluation": False}

    def run(name, arguments, expected=0):
        start = time.perf_counter()
        proc = subprocess.run([sys.executable, *map(str, arguments)], cwd=output,
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        (output / f"{name}.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        ok = proc.returncode == expected if expected == 0 else proc.returncode != 0
        report["checks"].append({"name": name, "passed": ok, "exit_code": proc.returncode,
            "elapsed_seconds": time.perf_counter() - start})
        check(ok, f"{name} failed; inspect its log")

    try:
        run("unit_tests", ["-m", "unittest", "discover", "-s", ROOT / "tests", "-v"])
        run("synthetic", [ROOT / "run_perception.py", "depth-synthetic", "--output", output / "synthetic"])
        synthetic = json.loads((output / "synthetic/result.json").read_text(encoding="utf-8"))
        observations = synthetic["depth"]["observations"]
        check(len(observations) == 1 and abs(observations[0]["height_m"] - .2) < .001, "synthetic geometry incorrect")
        for index in range(6):
            name = f"sample_{index:02d}"
            destination = output / name
            run(name, [ROOT / "run_perception.py", "paired-preview", "--device", args.device,
                "--rgb", ROOT / "samples" / f"{name}_rgb.jpg",
                "--depth", ROOT / "samples" / f"{name}_depth_raw.png", "--output", destination])
            result = json.loads((destination / "result.json").read_text(encoding="utf-8"))
            check(result["spatial_fusion"] is False and result["depth"]["metric_geometry"] is None, "uncalibrated geometry exposed")
            depth = cv2.imread(str(ROOT / "samples" / f"{name}_depth_raw.png"), cv2.IMREAD_UNCHANGED)
            check(abs(result["depth"]["nonzero_ratio"] - float((depth > 0).mean())) < 1e-12, "depth statistics mismatch")
            for instance in result["rgb"]["instances"]:
                mask = cv2.imread(str(destination / instance["mask_file"]), cv2.IMREAD_UNCHANGED)
                check(mask is not None and mask.shape == (480, 640), "mask missing or wrong size")
                check(set(np.unique(mask)).issubset({0, 255}) and np.any(mask), "invalid binary instance mask")
                check(instance["geometry"] is None, "invented RGB-depth association")
            check(cv2.imread(str(destination / "preview.jpg")).shape == (530, 1280, 3), "preview missing or wrong size")
            report["checks"][-1]["instances"] = len(result["rgb"]["instances"])
        # Guard checks are executed against the actual packaged CLI, not just parse_args.
        run("missing_calibration_rejected", [ROOT / "run_perception.py", "depth-calibrated",
            "--depth", ROOT / "samples/sample_00_depth_raw.png", "--output", output / "must_not_exist"], expected=1)
        check(not (output / "must_not_exist").exists(), "failure created output")
        before = (output / "synthetic/result.json").read_bytes()
        run("overwrite_rejected", [ROOT / "run_perception.py", "depth-synthetic", "--output", output / "synthetic"], expected=1)
        check((output / "synthetic/result.json").read_bytes() == before, "output overwritten")
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
