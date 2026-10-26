"""CPU-only regression checks for qualitative video rendering."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.render_overnight_video import render_result


class _CpuArray:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _TwoBoxes:
    xyxy = _CpuArray([[10, 10, 50, 50], [30, 30, 70, 70]])
    conf = _CpuArray([.8, .7])
    cls = _CpuArray([0, 0])

    def __len__(self):
        return 2


class VideoMaskRenderingTests(unittest.TestCase):
    def test_overlapping_instances_keep_union_fill_in_intersection(self):
        result = SimpleNamespace(
            boxes=_TwoBoxes(),
            masks=SimpleNamespace(xy=[
                np.array([[10, 10], [50, 10], [50, 50], [10, 50]]),
                np.array([[30, 30], [70, 30], [70, 70], [30, 70]]),
            ]),
        )
        image, record = render_result(result, np.zeros((90, 90, 3), dtype=np.uint8))
        # These interior pixels avoid contours and score labels. A single fillPoly
        # call with both polygons would leave the intersection black (even-odd).
        self.assertGreater(int(image[40, 40].sum()), 0)
        np.testing.assert_array_equal(image[40, 40], image[40, 20])
        np.testing.assert_array_equal(image[40, 40], image[60, 60])
        self.assertEqual(record["candidate_count"], 2)
        self.assertEqual(record["mask_count"], 2)


if __name__ == "__main__":
    unittest.main()
