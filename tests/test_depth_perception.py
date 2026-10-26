"""Behavior tests for the protocol-agnostic depth perception core."""

from __future__ import annotations

import sys
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class DepthDecodingTests(unittest.TestCase):
    def test_uint16_depth_uses_explicit_scale_and_invalidates_zero(self) -> None:
        from depth_perception.geometry import decode_depth

        raw = np.array([[0, 500, 1_250, 7_000]], dtype=np.uint16)

        result = decode_depth(raw, depth_scale=0.001, min_depth_m=0.2, max_depth_m=5.0)

        self.assertTrue(np.isnan(result[0, 0]))
        self.assertAlmostEqual(float(result[0, 1]), 0.5)
        self.assertAlmostEqual(float(result[0, 2]), 1.25)
        self.assertTrue(np.isnan(result[0, 3]))

    def test_depth_decoder_rejects_non_image_input(self) -> None:
        from depth_perception.geometry import decode_depth

        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            decode_depth(np.zeros((2, 3, 1), dtype=np.uint16), depth_scale=0.001)

    def test_depth_decoder_rejects_non_positive_scale(self) -> None:
        from depth_perception.geometry import decode_depth

        with self.assertRaisesRegex(ValueError, "depth_scale"):
            decode_depth(np.ones((2, 2), dtype=np.uint16), depth_scale=0.0)


class PinholeGeometryTests(unittest.TestCase):
    def test_deprojection_matches_hand_derived_camera_coordinates(self) -> None:
        from depth_perception.geometry import deproject_depth
        from depth_perception.types import CameraIntrinsics

        depth_m = np.array([[2.0, np.nan], [np.nan, 2.0]], dtype=np.float32)
        intrinsics = CameraIntrinsics(width=2, height=2, fx=2.0, fy=4.0, cx=0.0, cy=0.0)

        pixels, points = deproject_depth(depth_m, intrinsics)

        np.testing.assert_array_equal(pixels, np.array([[0, 0], [1, 1]]))
        np.testing.assert_allclose(points, np.array([[0.0, 0.0, 2.0], [1.0, 0.5, 2.0]]))

    def test_deprojection_rejects_intrinsics_for_a_different_image_shape(self) -> None:
        from depth_perception.geometry import deproject_depth
        from depth_perception.types import CameraIntrinsics

        intrinsics = CameraIntrinsics(width=3, height=2, fx=2.0, fy=2.0, cx=1.0, cy=0.5)

        with self.assertRaisesRegex(ValueError, "intrinsics"):
            deproject_depth(np.ones((2, 2), dtype=np.float32), intrinsics)

    def test_rotating_depth_180_updates_pixels_and_principal_point(self) -> None:
        from depth_perception.geometry import rotate_depth_180
        from depth_perception.types import CameraIntrinsics

        depth = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
        intrinsics = CameraIntrinsics(width=3, height=2, fx=10.0, fy=11.0, cx=0.25, cy=0.4)

        rotated, rotated_intrinsics = rotate_depth_180(depth, intrinsics)

        np.testing.assert_array_equal(rotated, np.array([[6, 5, 4], [3, 2, 1]], dtype=np.uint16))
        self.assertEqual(rotated_intrinsics.width, 3)
        self.assertEqual(rotated_intrinsics.height, 2)
        self.assertEqual(rotated_intrinsics.fx, 10.0)
        self.assertEqual(rotated_intrinsics.fy, 11.0)
        self.assertAlmostEqual(rotated_intrinsics.cx, 1.75)
        self.assertAlmostEqual(rotated_intrinsics.cy, 0.6)


def synthetic_floor_with_one_object() -> tuple[np.ndarray, "CameraIntrinsics"]:
    from depth_perception.types import CameraIntrinsics

    depth_m = np.full((64, 96), 2.0, dtype=np.float32)
    depth_m[32:47, 30:50] = 1.8
    intrinsics = CameraIntrinsics(
        width=96,
        height=64,
        fx=120.0,
        fy=120.0,
        cx=47.5,
        cy=31.5,
    )
    return depth_m, intrinsics


class FloorPlaneTests(unittest.TestCase):
    def test_ransac_plane_orients_positive_distance_toward_camera(self) -> None:
        from depth_perception.geometry import deproject_depth, fit_plane_ransac
        from depth_perception.types import DepthDetectorConfig

        depth_m, intrinsics = synthetic_floor_with_one_object()
        floor_mask = np.ones(depth_m.shape, dtype=bool)
        floor_mask[32:47, 30:50] = False
        _, floor_points = deproject_depth(depth_m, intrinsics, floor_mask, stride=2)

        plane = fit_plane_ransac(floor_points, DepthDetectorConfig())

        self.assertGreater(plane.offset_m, 0.0)
        self.assertAlmostEqual(float(plane.signed_distance([[0.0, 0.0, 1.8]])[0]), 0.2, places=3)
        self.assertGreater(plane.inlier_ratio, 0.99)
        self.assertLess(plane.rmse_m, 1e-6)


class StandaloneDepthDetectorTests(unittest.TestCase):
    def test_detector_extracts_raised_component_with_metric_geometry(self) -> None:
        from depth_perception.detector import DepthObstacleDetector
        from depth_perception.types import DepthDetectorConfig

        depth_m, intrinsics = synthetic_floor_with_one_object()
        config = DepthDetectorConfig(
            depth_scale=1.0,
            min_depth_m=0.1,
            max_depth_m=5.0,
            floor_roi_y_min=0.5,
            plane_sample_stride=2,
            min_plane_points=100,
            min_plane_inlier_ratio=0.7,
            min_object_height_m=0.05,
            max_object_height_m=0.4,
            min_component_pixels=100,
            morphology_kernel=1,
        )

        result = DepthObstacleDetector(config).process(depth_m, intrinsics)

        self.assertTrue(result.plane_valid)
        self.assertEqual(result.quality_flags, 0)
        self.assertEqual(len(result.observations), 1)
        observation = result.observations[0]
        self.assertEqual(observation.bbox_xyxy, (30, 32, 49, 46))
        self.assertAlmostEqual(observation.height_m, 0.2, places=2)
        self.assertGreater(observation.area_m2, 0.05)
        self.assertLess(observation.area_m2, 0.07)
        self.assertGreater(observation.geometry_confidence, 0.5)

    def test_flat_floor_reports_no_objects_without_fabricating_one(self) -> None:
        from depth_perception.detector import DepthObstacleDetector
        from depth_perception.types import CameraIntrinsics, DepthDetectorConfig, QualityFlag

        depth_m = np.full((48, 64), 2.0, dtype=np.float32)
        intrinsics = CameraIntrinsics(64, 48, 100.0, 100.0, 31.5, 23.5)
        config = DepthDetectorConfig(
            depth_scale=1.0,
            floor_roi_y_min=0.5,
            plane_sample_stride=2,
            min_plane_points=100,
            min_plane_inlier_ratio=0.9,
            min_object_height_m=0.05,
            morphology_kernel=1,
        )

        result = DepthObstacleDetector(config).process(depth_m, intrinsics)

        self.assertEqual(result.observations, ())
        self.assertTrue(result.quality_flags & QualityFlag.NO_OBJECTS)

    def test_insufficient_valid_depth_returns_explicit_quality_flags(self) -> None:
        from depth_perception.detector import DepthObstacleDetector
        from depth_perception.types import CameraIntrinsics, DepthDetectorConfig, QualityFlag

        depth_m = np.full((24, 32), np.nan, dtype=np.float32)
        depth_m[-2:, :2] = 1.0
        intrinsics = CameraIntrinsics(32, 24, 80.0, 80.0, 15.5, 11.5)
        config = DepthDetectorConfig(
            depth_scale=1.0,
            min_plane_points=20,
            plane_sample_stride=1,
        )

        result = DepthObstacleDetector(config).process(depth_m, intrinsics)

        self.assertFalse(result.plane_valid)
        self.assertTrue(result.quality_flags & QualityFlag.INSUFFICIENT_VALID_DEPTH)
        self.assertTrue(result.quality_flags & QualityFlag.FLOOR_PLANE_UNAVAILABLE)


class MaskMeasurementTests(unittest.TestCase):
    def test_yolo_style_instance_mask_reuses_depth_geometry(self) -> None:
        from depth_perception.detector import DepthObstacleDetector
        from depth_perception.types import DepthDetectorConfig

        depth_m, intrinsics = synthetic_floor_with_one_object()
        instance_mask = np.zeros(depth_m.shape, dtype=bool)
        instance_mask[32:47, 30:50] = True
        detector = DepthObstacleDetector(
            DepthDetectorConfig(
                depth_scale=1.0,
                floor_roi_y_min=0.5,
                plane_sample_stride=2,
                min_plane_points=100,
                min_plane_inlier_ratio=0.7,
                min_mask_valid_ratio=0.8,
                morphology_kernel=1,
            )
        )

        observation = detector.measure_mask(depth_m, intrinsics, instance_mask)

        self.assertEqual(observation.bbox_xyxy, (30, 32, 49, 46))
        self.assertEqual(observation.valid_pixel_count, 300)
        self.assertAlmostEqual(observation.height_m, 0.2, places=2)
        self.assertGreater(observation.geometry_confidence, 0.7)

    def test_mask_measurement_rejects_too_few_valid_depth_pixels(self) -> None:
        from depth_perception.detector import DepthObstacleDetector
        from depth_perception.types import DepthDetectorConfig

        depth_m, intrinsics = synthetic_floor_with_one_object()
        instance_mask = np.zeros(depth_m.shape, dtype=bool)
        instance_mask[32:47, 30:50] = True
        depth_m[instance_mask] = np.nan
        detector = DepthObstacleDetector(
            DepthDetectorConfig(
                depth_scale=1.0,
                plane_sample_stride=2,
                min_plane_points=100,
                min_mask_valid_ratio=0.5,
            )
        )

        with self.assertRaisesRegex(ValueError, "valid depth"):
            detector.measure_mask(depth_m, intrinsics, instance_mask)


class ConfigurationAndSerializationTests(unittest.TestCase):
    def test_config_loader_rejects_unknown_keys(self) -> None:
        from depth_perception.config import load_config

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("depth_scale: 0.001\nmade_up_key: 42\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "made_up_key"):
                load_config(path)

    def test_intrinsics_loader_lists_missing_calibration_fields(self) -> None:
        from depth_perception.config import load_intrinsics

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intrinsics.yaml"
            path.write_text("width: 640\nheight: 400\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "fx"):
                load_intrinsics(path)

    def test_frame_result_serializes_without_numpy_or_enum_objects(self) -> None:
        from depth_perception.detector import DepthObstacleDetector
        from depth_perception.serialization import result_to_dict
        from depth_perception.types import DepthDetectorConfig

        depth_m, intrinsics = synthetic_floor_with_one_object()
        result = DepthObstacleDetector(
            DepthDetectorConfig(
                depth_scale=1.0,
                floor_roi_y_min=0.5,
                plane_sample_stride=2,
                min_plane_points=100,
                min_component_pixels=100,
                min_object_height_m=0.05,
                morphology_kernel=1,
            )
        ).process(depth_m, intrinsics)

        payload = result_to_dict(result)
        encoded = json.dumps(payload)

        self.assertIn('"algorithm_version": "depth-baseline-v1"', encoded)
        self.assertEqual(payload["quality_flags"]["value"], 0)
        self.assertNotIn("object_mask", payload)
        self.assertEqual(len(payload["observations"]), 1)
        self.assertIn("geometry_confidence", payload["observations"][0])
        self.assertNotIn("confidence", payload["observations"][0])

    def test_frame_metadata_round_trips_to_serialized_output(self) -> None:
        from depth_perception.detector import DepthObstacleDetector
        from depth_perception.serialization import result_to_dict
        from depth_perception.types import DepthDetectorConfig, FrameMetadata

        depth_m, intrinsics = synthetic_floor_with_one_object()
        metadata = FrameMetadata(
            sequence_id=27,
            timestamp_ns=1_788_242_492_308_984_064,
            source_id="camera_03",
            frame_id="camera_03_depth_optical_frame",
        )
        result = DepthObstacleDetector(
            DepthDetectorConfig(
                depth_scale=1.0,
                plane_sample_stride=2,
                min_plane_points=100,
                min_component_pixels=100,
                morphology_kernel=1,
            )
        ).process(depth_m, intrinsics, metadata=metadata)

        payload = result_to_dict(result)

        self.assertEqual(
            payload["frame"],
            {
                "sequence_id": 27,
                "timestamp_ns": 1_788_242_492_308_984_064,
                "source_id": "camera_03",
                "frame_id": "camera_03_depth_optical_frame",
            },
        )


class DepthCliTests(unittest.TestCase):
    def test_cli_writes_json_and_object_mask_for_uint16_png(self) -> None:
        import cv2
        import yaml

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            depth_path = root / "depth.png"
            intrinsics_path = root / "intrinsics.yaml"
            config_path = root / "config.yaml"
            output = root / "result"
            raw = np.full((64, 96), 2_000, dtype=np.uint16)
            raw[32:47, 30:50] = 1_800
            self.assertTrue(cv2.imwrite(str(depth_path), raw))
            intrinsics_path.write_text(
                yaml.safe_dump(
                    {"width": 96, "height": 64, "fx": 120.0, "fy": 120.0, "cx": 47.5, "cy": 31.5}
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "depth_scale": 0.001,
                        "floor_roi_y_min": 0.5,
                        "plane_sample_stride": 2,
                        "min_plane_points": 100,
                        "min_plane_inlier_ratio": 0.7,
                        "min_object_height_m": 0.05,
                        "max_object_height_m": 0.4,
                        "min_component_pixels": 100,
                        "morphology_kernel": 1,
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "run_depth_baseline.py"),
                    "--depth",
                    str(depth_path),
                    "--intrinsics",
                    str(intrinsics_path),
                    "--config",
                    str(config_path),
                    "--output",
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["observations"]), 1)
            self.assertTrue((output / "object_mask.png").is_file())


if __name__ == "__main__":
    unittest.main()
