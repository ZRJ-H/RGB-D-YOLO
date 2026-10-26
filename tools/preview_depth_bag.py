"""Create calibration-independent RGB/depth previews from a ROS 2 bag."""

from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader


RGB_TOPIC = "/camera_03/color/image_raw"
DEPTH_TOPIC = "/camera_03/depth/image_raw"


def decode_rgb(message) -> np.ndarray:
    encoding = message.encoding.lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported RGB encoding: {message.encoding}")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    rows = raw[: message.height * message.step].reshape(message.height, message.step)
    image = rows[:, : message.width * channels].reshape(message.height, message.width, channels)
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def decode_depth(message) -> np.ndarray:
    encoding = message.encoding.lower()
    if encoding not in {"16uc1", "mono16"}:
        raise ValueError(f"unsupported depth encoding: {message.encoding}")
    dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
    row_values = message.step // dtype.itemsize
    raw = np.frombuffer(message.data, dtype=dtype)
    rows = raw[: message.height * row_values].reshape(message.height, row_values)
    return rows[:, : message.width].astype(np.uint16, copy=True)


def depth_colour(depth_raw: np.ndarray, minimum_raw: int = 300, maximum_raw: int = 5000) -> np.ndarray:
    """Colour raw depth values without assuming that one unit equals one millimetre."""
    valid = (depth_raw >= minimum_raw) & (depth_raw <= maximum_raw)
    normalised = np.zeros(depth_raw.shape, dtype=np.uint8)
    clipped = np.clip(depth_raw.astype(np.float32), minimum_raw, maximum_raw)
    normalised[valid] = np.round(
        255.0 * (maximum_raw - clipped[valid]) / (maximum_raw - minimum_raw)
    ).astype(np.uint8)
    colour = cv2.applyColorMap(normalised, cv2.COLORMAP_TURBO)
    colour[~valid] = (24, 24, 24)
    return colour


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def label(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 58), (18, 18, 18), -1)
    cv2.putText(result, title, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, subtitle, (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    return result


def nearest(items: list[tuple[int, object]], timestamp: int) -> tuple[int, object]:
    times = [item[0] for item in items]
    index = bisect_left(times, timestamp)
    candidates = [min(index, len(items) - 1), max(0, index - 1)]
    return min((items[item] for item in candidates), key=lambda value: abs(value[0] - timestamp))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    rgb_messages: list[tuple[int, object]] = []
    depth_messages: list[tuple[int, object]] = []
    with AnyReader([args.bag]) as reader:
        wanted = [item for item in reader.connections if item.topic in {RGB_TOPIC, DEPTH_TOPIC}]
        for connection, timestamp, rawdata in reader.messages(connections=wanted):
            message = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == RGB_TOPIC:
                rgb_messages.append((timestamp, message))
            else:
                depth_messages.append((timestamp, message))

    indices = np.linspace(0, len(depth_messages) - 1, args.samples + 2, dtype=int)[1:-1]
    rows = []
    summary = []
    for sample_number, index in enumerate(indices):
        depth_time, depth_message = depth_messages[int(index)]
        rgb_time, rgb_message = nearest(rgb_messages, depth_time)
        rgb = cv2.rotate(decode_rgb(rgb_message), cv2.ROTATE_180)
        depth = cv2.rotate(decode_depth(depth_message), cv2.ROTATE_180)
        valid = depth > 0
        colour = depth_colour(depth)
        valid_view = np.zeros((*depth.shape, 3), dtype=np.uint8)
        valid_view[valid] = (235, 235, 235)
        raw_name = f"sample_{sample_number:02d}_depth_raw.png"
        cv2.imwrite(str(args.output / raw_name), depth)
        cv2.imwrite(str(args.output / f"sample_{sample_number:02d}_rgb.jpg"), rgb)
        cv2.imwrite(str(args.output / f"sample_{sample_number:02d}_depth_colour.png"), colour)
        seconds = (depth_time - depth_messages[0][0]) / 1e9
        valid_values = depth[valid]
        valid_ratio = float(np.mean(valid))
        median_raw = float(np.median(valid_values)) if valid_values.size else None
        p10_raw = float(np.percentile(valid_values, 10)) if valid_values.size else None
        p90_raw = float(np.percentile(valid_values, 90)) if valid_values.size else None
        delta_ms = abs(rgb_time - depth_time) / 1e6
        subtitle = f"t={seconds:.1f}s | RGB-depth dt={delta_ms:.1f} ms"
        stats = f"valid={valid_ratio:.1%} | raw P10={p10_raw:.0f} median={median_raw:.0f} P90={p90_raw:.0f}"
        row = np.hstack(
            [
                label(letterbox(rgb, 640, 480), "RGB (rotated 180 deg)", subtitle),
                label(letterbox(colour, 640, 480), "Depth colour: near red, far blue", stats),
                label(letterbox(valid_view, 640, 480), "Valid depth mask", "white=measured, dark=missing/out of range"),
            ]
        )
        rows.append(row)
        summary.append(
            {
                "sample": sample_number,
                "depth_index": int(index),
                "seconds_from_start": seconds,
                "rgb_depth_delta_ms": delta_ms,
                "valid_depth_ratio": valid_ratio,
                "depth_unit": "unknown_16UC1_raw_unit",
                "depth_p10_raw": p10_raw,
                "depth_median_raw": median_raw,
                "depth_p90_raw": p90_raw,
                "raw_depth_file": raw_name,
            }
        )

    contact = np.vstack(rows)
    cv2.imwrite(str(args.output / "contact_sheet.png"), contact)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps({"rgb_frames": len(rgb_messages), "depth_frames": len(depth_messages), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
