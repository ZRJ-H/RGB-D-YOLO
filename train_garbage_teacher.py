from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_TARGET = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the offline YOLO garbage segmentation teacher.")
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics dataset YAML")
    parser.add_argument("--model", default="yolo11s-seg.pt")
    parser.add_argument("--project", type=Path, default=DEFAULT_TARGET / "outputs" / "training")
    parser.add_argument("--name", default="garbage_teacher_yolo11s_seg")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = args.data.resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data}")
    project = args.project.resolve()
    run_args = {
        "model": args.model,
        "data": str(data),
        "project": str(project),
        "name": args.name,
        "batch": args.batch,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "warmup_epochs": args.warmup_epochs,
        "warmup_bias_lr": args.warmup_bias_lr,
    }
    if args.dry_run:
        print(json.dumps(run_args, ensure_ascii=False, indent=2))
        return

    config_dir = DEFAULT_TARGET / ".ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(**{key: value for key, value in run_args.items() if key != "model"})


if __name__ == "__main__":
    main()
