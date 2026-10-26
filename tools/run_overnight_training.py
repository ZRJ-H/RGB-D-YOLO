"""Run isolated, auditable YOLO experiments and produce a morning acceptance bundle.

All source datasets and incumbent weights are read-only inputs. Model-generated
pseudo labels are intentionally not part of this workflow.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def configure_environment(output):
    os.environ["YOLO_CONFIG_DIR"] = str(Path(output) / "runtime" / "ultralytics")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["MPLCONFIGDIR"] = str(Path(output) / "matplotlib")
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)


def train_stage(config, stage):
    output = Path(config["output"])
    configure_environment(output)
    import torch
    from ultralytics import YOLO
    from ultralytics import settings

    settings.update({"sync": False, "wandb": False, "comet": False, "mlflow": False,
                     "clearml": False, "neptune": False, "dvc": False, "hub": False})

    if not torch.cuda.is_available() and str(config.get("device", "0")) != "cpu":
        raise RuntimeError("GPU training requested but CUDA is unavailable")
    torch.set_num_threads(4)
    run = next(item for item in config["experiments"] if item["name"] == stage)
    project = output / "training"
    if (project / stage).exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {project / stage}")
    progress_path = output / f"progress_{stage}.json"
    started = time.monotonic()
    model = YOLO(config["baseline"])

    def progress(trainer):
        values = {}
        for key, value in (getattr(trainer, "metrics", None) or {}).items():
            try:
                converted = float(value)
                values[key] = converted if math.isfinite(converted) else None
            except (ValueError, TypeError):
                pass
        payload = {
            "stage": stage, "epoch": int(trainer.epoch) + 1,
            "epochs": int(trainer.epochs), "elapsed_seconds": time.monotonic() - started,
            "updated_at": stamp(), "metrics": values,
        }
        write_json(progress_path, payload)
        print("EPOCH_PROGRESS " + json.dumps(payload), flush=True)

    model.add_callback("on_fit_epoch_end", progress)
    train_args = {
        "data": run["data"], "project": str(project), "name": stage,
        "epochs": config.get("epochs", 18), "patience": config.get("patience", 8),
        "imgsz": config.get("imgsz", 960), "batch": config.get("batch", 8),
        "device": config.get("device", "0"), "workers": 0, "cache": False,
        "optimizer": "AdamW", "lr0": config.get("lr0", 0.00008), "lrf": 0.1,
        "warmup_epochs": 0.0, "warmup_bias_lr": 0.0,
        "seed": config.get("seed", 20260923), "deterministic": True,
        "cos_lr": True, "mosaic": 0.15, "close_mosaic": 6,
        "mixup": 0.0, "copy_paste": 0.0, "cutmix": 0.0,
        "scale": 0.25, "translate": 0.08, "degrees": 3.0,
        "hsv_h": 0.015, "hsv_s": 0.35, "hsv_v": 0.25,
        "fliplr": 0.5, "flipud": 0.0, "amp": True,
        "mask_ratio": 4, "overlap_mask": True, "plots": True,
        "save": True, "save_period": -1, "exist_ok": False,
    }
    train_args.update(run.get("overrides", {}))
    write_json(output / f"train_args_{stage}.json", train_args)
    model.train(**train_args)
    best = project / stage / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("Training finished without best.pt")
    rows = list(csv.DictReader((project / stage / "results.csv").open(encoding="utf-8")))
    best_row = max(rows, key=lambda row: float(row["metrics/mAP50-95(M)"]))
    write_json(output / f"result_{stage}.json", {
        "stage": stage, "status": "trained", "best": str(best), "sha256": sha256(best),
        "epochs_completed": len(rows), "elapsed_seconds": time.monotonic() - started,
        "peak_public_mask_map": float(best_row["metrics/mAP50-95(M)"]),
        "note": "CSV peak is informational; best.pt is independently re-evaluated before comparison.",
        "completed_at": stamp(),
    })


def run_child(command, output, name, state, deadline):
    log = output / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    state.update(stage=name, stage_started_at=stamp(), log=str(log), updated_at=stamp())
    write_json(output / "STATUS.json", state)
    print(f"[{stamp()}] Starting {name}; log={log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 env=os.environ.copy(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        state["child_pid"] = child.pid
        try:
            while child.poll() is None:
                state["updated_at"] = stamp()
                state["disk_free_bytes"] = shutil.disk_usage(output).free
                write_json(output / "STATUS.json", state)
                if time.monotonic() > deadline or state["disk_free_bytes"] < 500 * 1024**2:
                    raise RuntimeError("Runtime limit or less than 500 MB disk space")
                time.sleep(10)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
    state.pop("child_pid", None)
    state.setdefault("finished_stages", []).append({"stage": name, "exit_code": child.returncode, "log": str(log)})
    write_json(output / "STATUS.json", state)
    if child.returncode:
        raise RuntimeError(f"Stage {name} failed with exit code {child.returncode}; see {log}")


def workflow(config_path, dry_run=False):
    config = read_json(config_path)
    output = Path(config["output"]).resolve()
    configure_environment(output)
    if dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    # Validate critical inputs before creating the exclusive run marker.
    initial_hash = sha256(config["baseline"])
    for experiment in config["experiments"]:
        if not Path(experiment["data"]).is_file():
            raise FileNotFoundError(experiment["data"])
    lock = output / "RUN_STARTED.json"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": stamp(), "config": str(config_path)}, stream)
    state = {"status": "running", "started_at": stamp(), "pid": os.getpid(),
             "incumbent_sha256": initial_hash, "finished_stages": [], "failures": []}
    deadline = time.monotonic() + config.get("max_hours", 6) * 3600
    successful = []
    keep_awake = False
    try:
        write_json(output / "config_snapshot.json", config)
        if sys.platform == "win32":
            import ctypes
            # Request system availability only for this process; allow the screen to sleep.
            keep_awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        state["temporary_system_awake_request"] = keep_awake
        for experiment in config["experiments"]:
            name = experiment["name"]
            try:
                run_child([sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
                           "--train-stage", name], output, f"train_{name}", state, deadline)
                result = read_json(output / f"result_{name}.json")
                evaluation_output = output / "evaluation" / name
                command = [sys.executable, str(ROOT / "tools" / "evaluate_overnight_candidate.py"),
                           "--baseline", config["baseline"], "--candidate", result["best"],
                           "--data", experiment["data"], "--factory-source", config["factory_source"],
                           "--output", str(evaluation_output), "--device", str(config.get("device", "0")),
                           "--imgsz", str(config.get("imgsz", 960)), "--batch", str(config.get("batch", 8)),
                           "--max-factory-images", "200", "--training-data", config["baseline_data"],
                           "--training-data", experiment["data"], "--reuse-local-data"]
                if config.get("factory_labeled_data"):
                    command.extend(["--factory-labeled-data", config["factory_labeled_data"]])
                command.extend(["--factory-note", "现场8张来自test_camera1，与基线训练图相邻约1–2秒；只作同视频未见图回归，不属于独立场景测试。"])
                run_child(command, output, f"evaluate_{name}", state, deadline)
                result["evaluation"] = str(evaluation_output)
                successful.append(result)
            except Exception as error:
                state["failures"].append({"experiment": name, "error": str(error)})
                write_json(output / "STATUS.json", state)
                print(traceback.format_exc(), flush=True)
        if not successful:
            raise RuntimeError("No experiment completed training and evaluation")
        if sha256(config["baseline"]) != initial_hash:
            raise RuntimeError("Incumbent weight checksum changed unexpectedly")
        # The final comparison script reads independently measured evaluation metrics.
        # Never replace the deployed/current-best model automatically.
        write_json(output / "completed_experiments.json", successful)
        summary_command = [sys.executable, str(ROOT / "tools" / "summarize_overnight_training.py"),
                           "--config", str(config_path)]
        run_child(summary_command, output, "acceptance_bundle", state, deadline)
        if config.get("video_source"):
            video_command = [sys.executable, str(ROOT / "tools" / "render_overnight_video.py"),
                             "--baseline", config["baseline"], "--candidate", str(output / "selected_candidate.pt"),
                             "--source", config["video_source"], "--output", str(output / "full_video"),
                             "--imgsz", str(config.get("imgsz", 960)), "--device", str(config.get("device", "0")),
                             "--fps", "10"]
            run_child(video_command, output, "full_video_acceptance", state, deadline)
        state.update(status="completed_with_failures" if state["failures"] else "completed",
                     all_experiments_completed=len(successful) == len(config["experiments"]),
                     stage="ready_for_user_review", completed_at=stamp(),
                     report=str(output / "acceptance.html"), experiments_completed=len(successful),
                     incumbent_preserved=True)
    except Exception as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc(), failed_at=stamp())
        print(traceback.format_exc(), flush=True)
    finally:
        if keep_awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        state.update(updated_at=stamp())
        write_json(output / "STATUS.json", state)
    if state["status"] not in {"completed", "completed_with_failures"}:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-stage")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.train_stage:
        train_stage(read_json(config_path), args.train_stage)
    else:
        workflow(config_path, args.dry_run)


if __name__ == "__main__":
    main()
