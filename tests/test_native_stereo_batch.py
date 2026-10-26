"""Small geometry regressions for native stereo export (no model/GPU)."""
import unittest

import numpy as np

from depth_perception.types import CameraIntrinsics
from tools.export_native_stereo_batch import geometric_mask, metric_arrays


class NativeStereoBatchTest(unittest.TestCase):
    def test_native_disparity_is_not_doubled_or_depth_scaled(self):
        camera = CameraIntrinsics(width=6, height=3, fx=10.0, fy=12.0, cx=2.5, cy=1.0)
        disparity = np.full((3, 6), 2.0, dtype=np.float32)
        depth, xyz, valid = metric_arrays(disparity, camera, baseline_m=0.1)
        self.assertEqual(depth.shape, (3, 6))
        self.assertEqual(xyz.shape, (3, 6, 3))
        np.testing.assert_array_equal(depth[:, 2:], np.full((3, 4), 0.5, dtype=np.float32))
        np.testing.assert_array_equal(xyz[..., 2][valid], depth[valid])
        np.testing.assert_allclose(xyz[1, 3], [0.025, 0.0, 0.5], rtol=1e-6)
        self.assertTrue(np.isnan(depth[:, :2]).all())
        self.assertTrue(np.isnan(xyz[:, :2]).all())

    def test_mask_excludes_nonfinite_nonpositive_and_out_of_overlap(self):
        disparity = np.array([[1, 0, -1, np.nan, np.inf, 2]], dtype=np.float32)
        np.testing.assert_array_equal(geometric_mask(disparity), [[False, False, False, False, False, True]])

    def test_refuses_using_intrinsics_on_another_grid(self):
        camera = CameraIntrinsics(width=6, height=3, fx=10.0, fy=12.0, cx=2.5, cy=1.0)
        with self.assertRaisesRegex(ValueError, "Calibration grid mismatch"):
            metric_arrays(np.ones((2, 3), dtype=np.float32), camera, 0.1)


if __name__ == "__main__":
    unittest.main()
