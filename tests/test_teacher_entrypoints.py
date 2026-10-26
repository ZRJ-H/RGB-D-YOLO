from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_teacher_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "train_garbage_teacher.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--data" in completed.stdout


def test_teacher_dry_run_rejects_missing_data(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "train_garbage_teacher.py"),
            "--data",
            str(tmp_path / "missing.yaml"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "does not exist" in (completed.stdout + completed.stderr)


def test_teacher_dry_run_exposes_fine_tuning_optimizer_settings(tmp_path: Path) -> None:
    data = tmp_path / "dataset.yaml"
    data.write_text("names:\n  0: sweepable_litter\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "train_garbage_teacher.py"),
            "--data",
            str(data),
            "--optimizer",
            "AdamW",
            "--lr0",
            "0.0002",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert '"optimizer": "AdamW"' in completed.stdout
    assert '"lr0": 0.0002' in completed.stdout


def test_teacher_dry_run_exposes_small_dataset_warmup_controls(tmp_path: Path) -> None:
    # Catches an unsafe hidden 0.1 bias warmup during tiny-domain fine-tuning.
    data = tmp_path / "dataset.yaml"
    data.write_text("names:\n  0: sweepable_litter\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "train_garbage_teacher.py"),
            "--data",
            str(data),
            "--warmup-epochs",
            "0",
            "--warmup-bias-lr",
            "0",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"warmup_epochs": 0.0' in completed.stdout
    assert '"warmup_bias_lr": 0.0' in completed.stdout
