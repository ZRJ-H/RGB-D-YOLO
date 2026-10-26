"""Prepare a new rectified Raw1 batch; preserve originals and unknown metric scale."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def save_image(path, value):
    ok, data = cv2.imencode(".png", value)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    data.tofile(path)
    if not np.array_equal(read_image(path), value):
        raise ValueError(f"PNG readback mismatch: {path}")


def rooted_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Input reference escapes its batch: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def distribution(values):
    if not len(values):
        return None
    return {"p50": float(np.median(values)), "p90": float(np.percentile(values, 90)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def transform_points(points, K, D, R, P):
    return cv2.undistortPointsIter(
        np.asarray(points, np.float64).reshape(-1, 1, 2), K, D, R, P,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10),
    ).reshape(-1, 2)


def paired_diagnostic(number, left, right, KL, DL, KR, DR, R1, R2, P1, P2):
    # Select matches once in the original images, then transform these exact locations.
    cv2.setRNGSeed(20260928 + number)
    sift = cv2.SIFT_create(nfeatures=6000)
    kl, dl = sift.detectAndCompute(left, None)
    kr, dr = sift.detectAndCompute(right, None)
    row = {"number": number, "keypoints_left": len(kl), "keypoints_right": len(kr)}
    if dl is None or dr is None:
        return dict(row, status="insufficient_features", fixed_match_count=0)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = [m for ms in matcher.knnMatch(dl, dr, k=2) if len(ms) == 2
               for m, n in [ms] if m.distance < 0.75 * n.distance]
    reverse = {(m.trainIdx, m.queryIdx) for ms in matcher.knnMatch(dr, dl, k=2) if len(ms) == 2
               for m, n in [ms] if m.distance < 0.75 * n.distance}
    matches = [m for m in forward if (m.queryIdx, m.trainIdx) in reverse]
    row.update(lowe_forward_count=len(forward), reciprocal_lowe_count=len(matches))
    if len(matches) < 8:
        return dict(row, status="insufficient_matches", fixed_match_count=0)
    pl = np.asarray([kl[m.queryIdx].pt for m in matches], np.float32)
    pr = np.asarray([kr[m.trainIdx].pt for m in matches], np.float32)
    F, mask = cv2.findFundamentalMat(pl, pr, cv2.FM_RANSAC, 1.5, 0.999, 10000)
    if F is None or mask is None:
        return dict(row, status="fundamental_matrix_failed", fixed_match_count=0)
    keep = mask.ravel().astype(bool)
    pl, pr = pl[keep], pr[keep]
    rl = transform_points(pl, KL, DL, R1, P1)
    rr = transform_points(pr, KR, DR, R2, P2)
    if not np.isfinite(rl).all() or not np.isfinite(rr).all():
        raise ValueError(f"Non-finite rectified feature points in pair {number}")
    original_dy, corrected_dy = pl[:, 1] - pr[:, 1], rl[:, 1] - rr[:, 1]
    row.update(
        status="computed_fixed_original_matches", fixed_match_count=len(pl),
        original_vertical_abs_pixels=distribution(np.abs(original_dy)),
        rectified_vertical_abs_pixels=distribution(np.abs(corrected_dy)),
        original_vertical_signed_pixels=distribution(original_dy),
        rectified_vertical_signed_pixels=distribution(corrected_dy),
        original_horizontal_positive_fraction=float(np.mean(pl[:, 0] - pr[:, 0] > 0)),
        rectified_horizontal_positive_fraction=float(np.mean(rl[:, 0] - rr[:, 0] > 0)),
        original_points_left=pl.tolist(), original_points_right=pr.tolist(),
        rectified_points_left=rl.tolist(), rectified_points_right=rr.tolist(),
    )
    return row


def map_diagnostic(K, D, R, P, mx, my, support, roi, size):
    width, height = size
    yy, xx = np.mgrid[0:height:19, 0:width:19]
    good = support[yy, xx]
    xy = np.column_stack([xx[good], yy[good]])
    source = np.column_stack([mx[yy[good], xx[good]], my[yy[good], xx[good]]])
    restored = transform_points(source, K, D, R, P)
    error = np.linalg.norm(restored - xy, axis=1)
    if not len(error) or not np.isfinite(error).all() or error.max() > 0.01:
        raise ValueError("Rectified-to-source map roundtrip failed (expected <=0.01 pixels)")
    # Original full-image ROI boundary projected into the new rectified pixel grid.
    x = np.linspace(0, width - 1, 33)
    y = np.linspace(0, height - 1, 33)
    border = np.concatenate([np.column_stack([x, 0*x]), np.column_stack([x, 0*x+height-1]),
                             np.column_stack([0*y, y]), np.column_stack([0*y+width-1, y])])
    transformed_border = transform_points(border, K, D, R, P)
    if not np.isfinite(transformed_border).all():
        raise ValueError("Original image ROI maps to non-finite rectified coordinates")
    rx, ry, rw, rh = map(int, roi)
    if not (0 <= rx <= width and 0 <= ry <= height and rx+rw <= width and ry+rh <= height):
        raise ValueError("Invalid stereoRectify valid ROI")
    roi_support = support[ry:ry+rh, rx:rx+rw]
    return {
        "support_fraction": float(support.mean()), "map_roundtrip_sample_count": len(error),
        "rectified_to_source_to_rectified_error_pixels": distribution(error),
        "original_full_image_roi_inclusive_xyxy": [0, 0, width-1, height-1],
        "original_roi_boundary_in_rectified_grid_bounds_xyxy": [
            *transformed_border.min(axis=0).tolist(), *transformed_border.max(axis=0).tolist()],
        "opencv_valid_roi_xywh": list(map(int, roi)),
        "strict_support_fraction_inside_opencv_roi": float(roi_support.mean()) if roi_support.size else None,
    }


def prepare(root, calibration_path):
    root, calibration_path = root.resolve(), calibration_path.resolve()
    output = root / "rectified"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    old = json.loads((root / "report.json").read_text(encoding="utf-8"))
    rows = copy.deepcopy(old.get("prepared_pairs") or old.get("results"))
    if not rows or sorted(r["number"] for r in rows) != list(range(1, 16)):
        raise ValueError("Expected exactly 15 prepared or completed source pairs")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    size = tuple(map(int, calibration["image_size"]))
    if size != (1280, 720):
        raise ValueError("This Raw1 batch requires supplied 1280x720 calibration")
    KL, KR, R = [np.asarray(calibration[k], np.float64) for k in ("cameraMatrixL", "cameraMatrixR", "R")]
    DL, DR = [np.asarray(calibration[k], np.float64) for k in ("distCoeffL", "distCoeffR")]
    T = np.asarray(calibration["T"], np.float64).reshape(3, 1)
    if any(a.shape != (3, 3) for a in (KL, KR, R)) or any(a.size != 5 for a in (DL, DR)):
        raise ValueError("Unexpected calibration matrix dimensions")
    if any(not np.isfinite(a).all() for a in (KL, KR, R, DL, DR, T)) or np.linalg.norm(T) <= 0:
        raise ValueError("Invalid or non-finite calibration")
    if abs(np.linalg.det(R)-1) > 1e-3 or np.max(np.abs(R.T @ R-np.eye(3))) > 1e-3:
        raise ValueError("Supplied R is not approximately a rotation")
    if any(np.linalg.det(a) == 0 for a in (KL, KR, R)):
        raise ValueError("Singular calibration matrix")
    if not calibration.get("source", {}).get("decoded_text"):
        raise ValueError("Recovered decoded calibration text is required for provenance")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        KL, DL, KR, DR, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=1, newImageSize=size)
    lm = cv2.initUndistortRectifyMap(KL, DL, R1, P1, size, cv2.CV_32FC1)
    rm = cv2.initUndistortRectifyMap(KR, DR, R2, P2, size, cv2.CV_32FC1)
    maps = dict(left_map_x=lm[0], left_map_y=lm[1], right_map_x=rm[0], right_map_y=rm[1])
    if any(m.dtype != np.float32 or m.shape != (720, 1280) or not np.isfinite(m).all() for m in maps.values()):
        raise ValueError("Invalid rectification maps")
    support = [(mx >= 0) & (mx < 1279) & (my >= 0) & (my < 719) for mx, my in (lm, rm)]
    mapping_checks = {
        "left": map_diagnostic(KL, DL, R1, P1, *lm, support[0], roi1, size),
        "right": map_diagnostic(KR, DR, R2, P2, *rm, support[1], roi2, size),
    }
    scaled = cv2.stereoRectify(KL, DL, KR, DR, size, R, T / 1000,
                              flags=cv2.CALIB_ZERO_DISPARITY, alpha=1, newImageSize=size)
    scaled_lm = cv2.initUndistortRectifyMap(KL, DL, scaled[0], scaled[2], size, cv2.CV_32FC1)
    scaled_rm = cv2.initUndistortRectifyMap(KR, DR, scaled[1], scaled[3], size, cv2.CV_32FC1)
    scale_errors = {
        "R1": float(np.max(np.abs(R1-scaled[0]))), "R2": float(np.max(np.abs(R2-scaled[1]))),
        "P1_left_3x3": float(np.max(np.abs(P1[:, :3]-scaled[2][:, :3]))),
        "P2_left_3x3": float(np.max(np.abs(P2[:, :3]-scaled[3][:, :3]))),
        **{name: float(np.max(np.abs(original-rescaled))) for name, original, rescaled in
           zip(maps, maps.values(), (*scaled_lm, *scaled_rm))},
    }
    if max(scale_errors.values()) > 1e-5:
        raise ValueError("Image rectification unexpectedly depends on translation scale")
    scale_check = {
        "test": "Compare supplied T against T/1000, same stereoRectify settings",
        "image_transform_max_abs_differences": scale_errors,
        "projection_fourth_column_T": [P1[:, 3].tolist(), P2[:, 3].tolist()],
        "projection_fourth_column_T_div_1000": [scaled[2][:, 3].tolist(), scaled[3][:, 3].tolist()],
        "Q_with_T_div_1000": scaled[4].tolist(),
        "interpretation": "Image rectification is invariant to this positive uniform translation scaling; "
                          "P fourth columns and Q retain scale dependence. This numerical check "
                          "does not establish the physical translation unit or depth accuracy.",
    }
    inverses = {"KL": KL, "KR": KR, "R": R, "R1": R1, "R2": R2, "P1_intrinsics": P1[:, :3],
                "P2_intrinsics": P2[:, :3], "Q": Q}
    inverse_errors = {name: float(np.max(np.abs(np.linalg.inv(a) @ a - np.eye(a.shape[0]))))
                      for name, a in inverses.items()}
    if max(inverse_errors.values()) > 1e-8:
        raise ValueError("Calibration matrix inverse check failed")

    source_images = []
    for row in rows:
        if row["folder"] != f"pair_{row['number']:02d}" or row["input_size"] != [1280, 720]:
            raise ValueError("Unexpected source folder or image size")
        images = [read_image(rooted_file(root, row["files"][k])) for k in ("left_input", "right_input")]
        if any(im.shape != (720, 1280) or im.dtype != np.uint8 for im in images):
            raise ValueError("Original pair inputs must be 1280x720 uint8 grayscale")
        for key in ("source_color", "device_raw", "device_display"):
            path = rooted_file(root, row["files"][key])
            if not path.is_relative_to(root / "source"):
                raise ValueError(f"Expected source/ reference: {key}")
        source_images.append(images)

    output.mkdir()  # All input and mathematical preflight checks pass before writing.
    shutil.copyfile(calibration_path, output / "calibration.json")
    (output / "calibration_recovered.txt").write_bytes(calibration["source"]["decoded_text"].encode("utf-8"))
    np.savez_compressed(output / "rectification.npz", **maps)
    with np.load(output / "rectification.npz", allow_pickle=False) as stored:
        if set(stored.files) != set(maps) or any(not np.array_equal(stored[k], v) for k, v in maps.items()):
            raise ValueError("Saved rectification maps readback failed")
    for side, mask in zip(("left", "right"), support):
        save_image(output / f"{side}_source_support_mask.png", mask.astype(np.uint8) * 255)

    diagnostic_rows = []
    for row, (left, right) in zip(rows, source_images):
        dest = output / row["folder"]
        dest.mkdir()
        for key, source, remap in zip(("left_input", "right_input"), (left, right), (lm, rm)):
            value = cv2.remap(source, *remap, interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            row["files"][key] = f"{row['folder']}/{key}.png"
            save_image(output / row["files"][key], value)
        for key in ("source_color", "device_raw", "device_display"):
            row["files"][key] = "../" + Path(row["files"][key]).as_posix()
        row["files"].update(left_source_support_mask="left_source_support_mask.png",
                            right_source_support_mask="right_source_support_mask.png")
        for key in ("inference_seconds", "positive_finite_ratio", "horizontal_overlap_ratio", "disparity_percentiles",
                    "masks_are_confidence"):
            row.pop(key, None)
        row.update(rectification_applied=True, output_grid="rectified_left_camera_grid_not_original_pixels",
                   input_rectification_parameters="rectification.json", rectification_maps="rectification.npz")
        diagnostic_rows.append(paired_diagnostic(row["number"], left, right, KL, DL, KR, DR, R1, R2, P1, P2))
        print(f"Rectified pair {row['number']:02d}/15; retained full 1280x720 canvas", flush=True)

    diagnostic = {
        "status": "diagnostic_only_calibration_applicability_not_validated", "opencv_version": cv2.__version__,
        "algorithm": "SIFT nfeatures=6000, grayscale original images at native size; reciprocal BF L2 Lowe<0.75; "
                     "original F-matrix RANSAC threshold 1.5px/confidence .999/maxIters10000; seed 20260928+pair; "
                     "exact same original inlier point locations transformed via undistortPointsIter(K,D,R,P), "
                     "100 iterations/epsilon1e-10; no point selection after rectification, no support/ROI filtering",
        "metric_scale_verified": False, "calibration_sha256": digest(calibration_path),
        "interpretation": "Compare fixed sparse correspondences only. An increase in vertical residual flags "
                          "unverified calibration applicability; user confirmed original images are unrectified. "
                          "These residuals neither certify calibration nor measure depth accuracy.",
        "rows": diagnostic_rows,
    }
    complete = [r for r in diagnostic_rows if r["status"] == "computed_fixed_original_matches"]
    diagnostic["summary"] = {
        "pairs_computed": len(complete), "fixed_match_count": sum(r["fixed_match_count"] for r in complete),
        "pairs_with_larger_rectified_median_abs_vertical_error": [r["number"] for r in complete if
            r["rectified_vertical_abs_pixels"]["p50"] > r["original_vertical_abs_pixels"]["p50"]],
    }
    write_json(output / "rectification_diagnostic.json", diagnostic)
    metadata = {
        "image_size": list(size), "rectified_canvas_size": list(size), "cropping_applied": False,
        "original_pixel_grid_preserved": False, "output_grid": "rectified_camera_grid",
        "rectification": "rectified_with_supplied_calibration_scale_unknown",
        "translation_unit": calibration["translation_unit"], "metric_scale_verified": False,
        "input_rectification": calibration["input_rectification"],
        "left_half_as_left_and_right_half_as_right": "inferred_from_positive_feature_disparities",
        "calibration_sha256": digest(calibration_path), "recovered_parameter_source": calibration["source"],
        "method": "OpenCV stereoRectify CALIB_ZERO_DISPARITY alpha=1, newImageSize=1280x720; "
                  "CV_32FC1 inverse maps; remap INTER_LINEAR BORDER_CONSTANT=0; no cropping",
        "R1": R1.tolist(), "R2": R2.tolist(), "P1": P1.tolist(), "P2": P2.tolist(), "Q": Q.tolist(),
        "valid_roi_left_xywh": list(map(int, roi1)), "valid_roi_right_xywh": list(map(int, roi2)),
        "map_direction": "rectified destination pixel to original source-image pixel",
        "support_mask_definition": "finite map; 0<=source_x<1279 and 0<=source_y<719; "
                                   "strict complete bilinear source support, not correspondence confidence",
        "files": {"maps": "rectification.npz", "calibration": "calibration.json",
                  "recovered_text": "calibration_recovered.txt", "diagnostic": "rectification_diagnostic.json",
                  "left_source_support_mask": "left_source_support_mask.png",
                  "right_source_support_mask": "right_source_support_mask.png"},
        "verification": {"mapping": mapping_checks, "matrix_inverse_max_abs_errors": inverse_errors,
                         "translation_scale_invariance": scale_check,
                         "R_determinant": float(np.linalg.det(R)), "image_and_map_readback": "passed",
                         "prepared_pairs": len(rows), "physical_accuracy_verified": False},
    }
    write_json(output / "rectification.json", metadata)
    report = copy.deepcopy(old)
    report.pop("completed_utc", None)
    report.update(status="prepared", results=[], prepared_pairs=rows, source_root_relative="..",
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  rectification="rectified_with_supplied_calibration_scale_unknown",
                  output_grid="rectified_left_camera_grid_not_original_pixels",
                  input_preprocessing="split BGR color at column1280; BGR to grayscale; calibrated remap "
                                      "on full1280x720 canvas; repeat grayscale to three channels for network",
                  rectification_metadata="rectification.json", rectification_diagnostic="rectification_diagnostic.json",
                  rectification_maps="rectification.npz", calibration="calibration.json",
                  translation_unit=calibration["translation_unit"], metric_depth=False, metric_depth_verified=False)
    (output / "provenance").mkdir()
    shutil.copy2(Path(__file__), output / "provenance" / Path(__file__).name)
    write_json(output / "report.json", report)  # Prepared status appears only after successful full completion.
    print(f"Prepared all {len(rows)} rectified pairs in {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing prepared/completed Raw1 batch root")
    parser.add_argument("--calibration", type=Path, required=True, help="Recovered1280x720 calibration JSON")
    args = parser.parse_args()
    cv2.setNumThreads(4)
    prepare(args.input, args.calibration)


if __name__ == "__main__":
    main()
