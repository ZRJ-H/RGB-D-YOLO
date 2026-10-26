"""Check the actual container stack and optionally Jetson CUDA kernels."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = {"passed": False, "architecture": platform.machine(), "python": sys.version,
        "cuda_required": args.require_cuda, "packages": {}, "errors": []}
    try:
        import cv2
        import numpy as np
        import torch
        import torchvision
        import ultralytics
        import yaml
        import psutil
        for package in ("ultralytics", "torch", "torchvision", "numpy", "PyYAML", "psutil"):
            report["packages"][package] = metadata.version(package)
        report["packages"]["opencv_runtime"] = cv2.__version__
        if ultralytics.__version__ != "8.3.179":
            raise RuntimeError("Expected Ultralytics 8.3.179")
        # Probe uint16 decoding without the global Ultralytics imread patch.
        raw = np.array([[0, 2000]], dtype=np.uint16)
        ok, encoded = cv2.imencode(".png", raw)
        if not ok or not np.array_equal(cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED), raw):
            raise RuntimeError("uint16 OpenCV PNG round-trip failed")
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda"] = torch.version.cuda
        if args.require_cuda:
            if platform.machine() not in ("aarch64", "arm64"):
                raise RuntimeError("Jetson image requires ARM64 for target acceptance")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; check host JetPack and NVIDIA container runtime")
            report["gpu"] = torch.cuda.get_device_name(0)
            tensor = torch.ones((64, 64), device="cuda")
            if float((tensor @ tensor).sum().item()) != 262144.:
                raise RuntimeError("CUDA matrix operation failed")
            keep = torchvision.ops.nms(torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]], device="cuda"),
                torch.tensor([.9, .8], device="cuda"), .5)
            if keep.cpu().tolist() != [0]:
                raise RuntimeError("torchvision CUDA NMS failed")
            torch.cuda.synchronize()
        report["passed"] = True
    except Exception as error:
        report["errors"].append(str(error))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
