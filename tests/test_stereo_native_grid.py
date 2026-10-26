"""Geometry regressions for returning stereo depth on the original image grid."""
import unittest

import numpy as np

from tools.apply_stereo_calibration_trial import restore_native_disparity


class NativeGridTests(unittest.TestCase):
    def test_metric_values_and_pixel_coordinates_survive_twofold_restore(self):
        disparity = np.array([[0.5, 0.5, 1.0, 1.5], [0.5, 0.75, 1.25, 1.75]], dtype=np.float32)
        before = disparity.copy()
        native, valid = restore_native_disparity(disparity, [8, 4])
        expected = np.repeat(np.repeat(disparity, 2, axis=0), 2, axis=1)
        np.testing.assert_array_equal(native, expected * 2)
        np.testing.assert_array_equal(disparity, before)
        # Same physical plane/edges: depth must not be doubled with image size.
        half_fx, baseline = 310.069397, 0.0952326965
        np.testing.assert_allclose((half_fx * 2) * baseline / native[valid],
                                   half_fx * baseline / expected[valid], rtol=1e-6)
        # Source pixel (2, 1) expands into rows 2:4, columns 4:6 without a crop.
        np.testing.assert_array_equal(native[2:4, 4:6], np.full((2, 2), 2.5))

    def test_invalid_holes_and_out_of_view_pixels_are_not_filled(self):
        disparity = np.array([[2, np.nan, 0, 0.5, np.inf, -1]], dtype=np.float32)
        native, valid = restore_native_disparity(disparity, [12, 2])
        expected = np.repeat(np.repeat(np.array([[False, False, False, True, False, False]]), 2, 0), 2, 1)
        np.testing.assert_array_equal(valid, expected)
        depth = np.full(native.shape, np.nan, dtype=np.float32)
        depth[valid] = 1 / native[valid]
        np.testing.assert_array_equal(np.isnan(depth), ~expected)

    def test_mixed_resolution_is_restored_to_its_own_grid(self):
        native, valid = restore_native_disparity(np.full((240, 424), 8, dtype=np.float32), [848, 480])
        self.assertEqual(native.shape, (480, 848))
        self.assertEqual(valid.shape, (480, 848))
        np.testing.assert_array_equal(native, np.full((480, 848), 16, dtype=np.float32))
        self.assertFalse(valid[:, :16].any())
        self.assertTrue(valid[:, 16:].all())

    def test_identity_and_invalid_shapes(self):
        source = np.ones((3, 4), dtype=np.float32)
        native, _ = restore_native_disparity(source, [4, 3])
        np.testing.assert_array_equal(native, source)
        with self.assertRaises(ValueError):
            restore_native_disparity(source, [2, 3])
        with self.assertRaises(ValueError):
            restore_native_disparity(source, [4, 0])


if __name__ == "__main__":
    unittest.main()
