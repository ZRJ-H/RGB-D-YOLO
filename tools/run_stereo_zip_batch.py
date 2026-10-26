"""Prepare and infer the 15 side-by-side pairs in Image_Depth_Raw1.zip."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from foundationstereo_runtime import load_model, sha256

sys.dont_write_bytecode = True


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_image(path, flags=cv2.IMREAD_UNCHANGED):
    value = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if value is None:
        raise ValueError(f"Cannot decode {path}")
    return value


def save_image(path, value):
    ok, data = cv2.imencode(path.suffix, value)
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    data.tofile(path)


def prepare(source_zip, output):
    if output.exists():
        raise FileExistsError("Use a new output directory")
    expected = {f"Image_{kind}{n}.png" for n in range(1, 16)
                for kind in ("Color", "Depth_Raw", "Depth_Dispaly")}
    with zipfile.ZipFile(source_zip) as archive:
        names = archive.namelist()
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("Expected exactly Color/Depth_Raw/Depth_Dispaly for 15 numbered groups")
        decoded, manifest = {}, []
        # Validate before creating any output; only known flat filenames are written.
        for name in sorted(names):
            data = archive.read(name)
            value = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if value is None:
                raise ValueError(f"Cannot decode {name}")
            shape = (720, 2560, 3) if "Color" in name else ((720, 1280) if "Raw" in name else (720, 1280, 3))
            dtype = np.uint16 if "Raw" in name else np.uint8
            assert value.shape == shape and value.dtype == dtype, name
            decoded[name] = value
            manifest.append({"name": name, "sha256": hashlib.sha256(data).hexdigest(),
                             "bytes": len(data), "shape": list(value.shape), "dtype": str(value.dtype)})
        output.mkdir(parents=True)
        (output / "source").mkdir()
        for name in names:
            (output / "source" / name).write_bytes(archive.read(name))
    rows = []
    for number in range(1, 16):
        names = {"color": f"Image_Color{number}.png", "raw": f"Image_Depth_Raw{number}.png",
                 "display": f"Image_Depth_Dispaly{number}.png"}
        original = decoded[names["color"]]
        left = cv2.cvtColor(original[:, :1280], cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(original[:, 1280:], cv2.COLOR_BGR2GRAY)
        folder = f"pair_{number:02d}"
        dest = output / folder
        dest.mkdir()
        save_image(dest / "left_input.png", left)
        save_image(dest / "right_input.png", right)
        raw = decoded[names["raw"]]
        files = {key: f"{folder}/{value}" for key, value in {
            "left_input": "left_input.png", "right_input": "right_input.png",
            "disparity": "disparity_pixels.npy", "inverse_disparity": "inverse_disparity_nonmetric.npy",
            "positive_mask": "positive_finite_mask.png", "overlap_mask": "horizontal_overlap_mask.png",
            "disparity_colour": "disparity_colour.png", "preview": "preview.jpg"}.items()}
        files.update(source_color=f"source/{names['color']}", device_raw=f"source/{names['raw']}",
                     device_display=f"source/{names['display']}")
        rows.append(dict(number=number, folder=folder, input_size=[1280, 720], padded_size=[1280, 736],
                         source_members=names, raw_nonzero_ratio=float((raw > 0).mean()),
                         raw_value_range=[int(raw.min()), int(raw.max())], files=files))
    report = {
        "status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_zip": str(source_zip), "source_zip_sha256": sha256(source_zip),
        "iterations": 32, "scale": 1.0, "native_resolution_inference": True,
        "metric_depth": False, "metric_depth_verified": False,
        "rectification": "unrectified_user_confirmed_no_rectification_applied",
        "pair_order": "left_half_as_left_right_half_as_right_unverified",
        "input_preprocessing": "split at column 1280; BGR to grayscale; repeat three channels; no resizing",
        "device_raw_encoding_and_units": "unverified", "results": [], "prepared_pairs": rows,
    }
    write_json(output / "source_manifest.json", manifest)
    write_json(output / "report.json", report)
    print(f"Prepared all 15 pairs / 45 original PNGs in {output}", flush=True)


def preview(output, row, colour):
    width, height = 600, 338
    canvas = Image.new("RGB", (1236, 820), "#f5f6f4")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    draw.text((12, 8), f"Pair {row['number']:02d} | native 1280 x 720 | 32 iterations | disparity only", fill="#14363a", font=font)
    arrays = [read_image(output / row["files"]["left_input"]),
              read_image(output / row["files"]["right_input"]), colour,
              read_image(output / row["files"]["device_display"])]
    labels = ["LEFT HALF / MODEL LEFT INPUT (grayscale)", "RIGHT HALF / MODEL RIGHT INPUT (grayscale)",
              f"MODEL DISPARITY (px), colour P2-P98 {row['disparity_percentiles'][0]:.2f}-{row['disparity_percentiles'][2]:.2f}",
              "DEVICE DISPLAY / supplied image, encoding unverified"]
    for i, (value, label) in enumerate(zip(arrays, labels)):
        x, y = 12 + (i % 2) * 612, 48 + (i // 2) * 365
        draw.text((x, y), label, fill="#34484a", font=font)
        value = cv2.resize(value, (width, height), interpolation=cv2.INTER_AREA)
        if value.ndim == 3:
            value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
        canvas.paste(Image.fromarray(value).convert("RGB"), (x, y + 24))
    footer = ("Rectified image grid, not original pixels. Translation unit unknown; no metric depth."
              if row.get("rectification_applied") else
              "Unrectified diagnostic reference. No metric depth, XYZ or physical size claims.")
    draw.text((12, 785), footer, fill="#875126", font=font)
    canvas.save(output / row["files"]["preview"], quality=92)


def infer(output):
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "prepared" or report["results"]:
        raise ValueError("Inference requires a fresh prepared batch; existing predictions are preserved")
    assert sha256(Path(report["source_zip"])) == report["source_zip_sha256"]
    source_root = (output / report.get("source_root_relative", ".")).resolve()
    for item in json.loads((source_root / "source_manifest.json").read_text(encoding="utf-8")):
        assert sha256(source_root / "source" / item["name"]) == item["sha256"]
    model, InputPadder, torch, metadata = load_model()
    report.update(metadata, status="running")
    write_json(output / "report.json", report)
    (output / "provenance").mkdir(exist_ok=True)
    for name in ("run_stereo_zip_batch.py", "foundationstereo_runtime.py"):
        shutil.copy2(Path(__file__).with_name(name), output / "provenance" / name)
    shutil.copy2(Path(report["checkpoint"]).with_name("cfg.yaml"), output / "provenance/cfg.yaml")
    for row in report["prepared_pairs"]:
        inputs = [read_image(output / row["files"][key]) for key in ("left_input", "right_input")]
        tensors = [torch.from_numpy(np.repeat(a[..., None], 3, axis=2)).permute(2, 0, 1)[None].float().cuda() for a in inputs]
        padder = InputPadder(tensors[0].shape, divis_by=32, force_square=False)
        tensors = padder.pad(*tensors)
        assert list(reversed(tensors[0].shape[-2:])) == row["padded_size"]
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            prediction = model(*tensors, iters=32, test_mode=True)
        torch.cuda.synchronize()
        row["inference_seconds"] = time.perf_counter() - start
        disparity = padder.unpad(prediction.float()).squeeze().cpu().numpy()
        assert disparity.shape == (720, 1280) and disparity.dtype == np.float32
        valid = np.isfinite(disparity) & (disparity > 0)
        if not valid.any():
            raise RuntimeError("No finite positive disparities")
        overlap = valid & (np.arange(1280)[None] - disparity >= 0) & (np.arange(1280)[None] - disparity <= 1279)
        inverse = np.full(disparity.shape, np.nan, dtype=np.float32)
        inverse[valid] = 1 / disparity[valid]
        np.save(output / row["files"]["disparity"], disparity)
        np.save(output / row["files"]["inverse_disparity"], inverse)
        save_image(output / row["files"]["positive_mask"], valid.astype(np.uint8) * 255)
        save_image(output / row["files"]["overlap_mask"], overlap.astype(np.uint8) * 255)
        percentiles = np.percentile(disparity[valid], [2, 50, 98]).tolist()
        lo, _, hi = percentiles
        scaled = np.nan_to_num(np.clip((disparity - lo) / max(hi - lo, 1e-6) * 255, 0, 255)).astype(np.uint8)
        colour = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        colour[~valid] = 0
        save_image(output / row["files"]["disparity_colour"], colour)
        row.update(positive_finite_ratio=float(valid.mean()), horizontal_overlap_ratio=float(overlap.mean()),
                   disparity_percentiles=percentiles, masks_are_confidence=False)
        preview(output, row, colour)
        write_json(output / row["folder"] / "result.json", row)
        report["results"].append(row)
        write_json(output / "report.json", report)
        print(f"Pair {row['number']:02d}/15: {row['inference_seconds']:.3f}s, disparity median {percentiles[1]:.3f}px", flush=True)
    report.pop("prepared_pairs")
    report.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "report.json", report)
    print("DONE: 15/15 native pairs; no metric conversion", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-zip", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "infer"), required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.mode == "prepare":
        if args.source_zip is None:
            parser.error("--source-zip is required for prepare")
        prepare(args.source_zip.resolve(), output)
    else:
        infer(output)


if __name__ == "__main__":
    main()
