"""Release safety and uncalibrated-depth behavior tests."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_perception import raw_depth_summary, parse_args, synthetic_geometry, read_image


class PreviewTests(unittest.TestCase):
    def test_depth_decoder_is_independent_of_ultralytics_imread_patch(self):
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.png"
            raw = np.array([[0, 1800], [2000, 4756]], dtype=np.uint16)
            cv2.imwrite(str(path), raw)
            with patch.object(cv2, "imread", side_effect=RuntimeError("global imread replaced")):
                decoded = read_image(path, cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(decoded, raw)
            self.assertEqual(decoded.dtype, np.uint16)

    def test_unknown_scale_never_produces_metric_geometry(self):
        result = raw_depth_summary(np.array([[0, 1000], [2000, 3000]], dtype=np.uint16))
        self.assertEqual(result["nonzero_ratio"], .75)
        self.assertIsNone(result["metric_geometry"])
        self.assertIn("NOT_REGISTERED_TO_RGB", result["quality_flags"])

    def test_all_missing_depth_is_explicit(self):
        result = raw_depth_summary(np.zeros((4, 5), dtype=np.uint16))
        self.assertEqual(result["raw_percentiles_10_50_90"], [None] * 3)
        self.assertIn("NO_VALID_DEPTH", result["quality_flags"])

    def test_wrong_depth_encoding_is_rejected(self):
        with self.assertRaises(ValueError):
            raw_depth_summary(np.ones((4, 5), dtype=np.uint8))

    def test_metric_mode_requires_calibration_and_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            depth = root / "depth.png"
            depth.touch()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["depth-calibrated", "--depth", str(depth), "--output", str(root / "out")])
            self.assertFalse((root / "out").exists())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["depth-synthetic", "--output", directory])
            self.assertEqual(sentinel.read_text(), "keep")

    def test_synthetic_object_height_and_count(self):
        _, result = synthetic_geometry()
        self.assertTrue(result.plane_valid)
        self.assertEqual(len(result.observations), 1)
        self.assertAlmostEqual(result.observations[0].height_m, .2, places=3)


if __name__ == "__main__":
    unittest.main()
