"""Tests for the RGB inference benchmark."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class SummaryTests(unittest.TestCase):
    def test_summary_reports_latency_percentiles_and_fps(self) -> None:
        from benchmark_rgb import summarize_ms

        summary = summarize_ms([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(summary["runs"], 4)
        self.assertEqual(summary["mean_ms"], 2.5)
        self.assertEqual(summary["p50_ms"], 2.5)
        self.assertEqual(summary["p95_ms"], 3.85)
        self.assertEqual(summary["min_ms"], 1.0)
        self.assertEqual(summary["max_ms"], 4.0)
        self.assertTrue(math.isclose(summary["fps"], 400.0))

    def test_summary_rejects_empty_measurements(self) -> None:
        from benchmark_rgb import summarize_ms

        with self.assertRaisesRegex(ValueError, "at least one"):
            summarize_ms([])


class BenchmarkConfigurationTests(unittest.TestCase):
    def test_defaults_target_current_segmentation_model_at_960(self) -> None:
        from benchmark_rgb import parse_args

        args = parse_args([])

        self.assertEqual(args.imgsz, 960)
        self.assertEqual(
            args.weights.as_posix(),
            "outputs/training/garbage_teacher_yolo11s_seg_reviewed20_r10_lr5e5_e10/weights/best.pt",
        )
        self.assertEqual(args.source.as_posix(), "work/extracted_frames/test_camera1_1fps")

    def test_gpu_runs_before_cpu_can_hide_cuda(self) -> None:
        from benchmark_rgb import benchmark_device_order

        self.assertEqual(benchmark_device_order(True, "0"), ["0", "cpu"])
        self.assertEqual(benchmark_device_order(False, "0"), ["cpu"])

    def test_single_core_affinity_is_restored_after_cpu_benchmark(self) -> None:
        import psutil

        from benchmark_rgb import single_core_affinity

        process = psutil.Process()
        original = process.cpu_affinity()
        self.assertGreaterEqual(len(original), 1)

        with single_core_affinity(process) as selected_core:
            self.assertEqual(process.cpu_affinity(), [selected_core])

        self.assertEqual(process.cpu_affinity(), original)


class EntrypointTests(unittest.TestCase):
    def test_help_does_not_require_ultralytics(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        completed = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "benchmark_rgb.py"), "--help"],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--weights", completed.stdout)
        self.assertIn("--source", completed.stdout)
        self.assertIn("--runs", completed.stdout)


if __name__ == "__main__":
    unittest.main()
