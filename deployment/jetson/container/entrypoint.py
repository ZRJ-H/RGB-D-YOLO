"""Dispatch explicit diagnostics or the packaged perception CLI."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    args = sys.argv[1:]
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/yolo-depth-settings")
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    if args and args[0] == "doctor":
        os.execv(sys.executable, [sys.executable, str(ROOT / "container/doctor.py"), *args[1:]])
    if args and args[0] == "self-test":
        output = Path(args[1] if len(args) == 2 else "/outputs/acceptance")
        if len(args) > 2 or output.exists():
            raise SystemExit("Usage: self-test [NEW_OUTPUT_DIRECTORY]")
        subprocess.run([sys.executable, str(ROOT / "verify_release.py")], check=True)
        subprocess.run([sys.executable, str(ROOT / "container/doctor.py"), "--require-cuda"], check=True)
        os.execv(sys.executable, [sys.executable, str(ROOT / "smoke_test.py"), "--device", "0", "--output", str(output)])
    os.execv(sys.executable, [sys.executable, str(ROOT / "run_perception.py"), *(args or ["--help"])])


if __name__ == "__main__":
    main()
