"""Package selected stereo/XYZ samples for offline review, one per DOE setting."""
import argparse
import hashlib
from html import escape
from html.parser import HTMLParser
import io
import json
from pathlib import Path, PureWindowsPath
import shutil
import zipfile

import cv2
import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def portable(value):
    """Keep metadata, omitting external absolute paths and upstream relative links."""
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()
                if not (isinstance(item, str) and
                        (Path(item).is_absolute() or PureWindowsPath(item).is_absolute()
                         or item.replace("\\", "/").startswith("../")))
                and key not in {"output", "source_trial", "source_frame", "source_result_path",
                                "source_preview_path", "raw_disparity_href", "source_preview_href"}}
    if isinstance(value, list):
        return [portable(item) for item in value]
    return value


def source_path(value, parent):
    path = Path(value)
    return path if path.is_absolute() else parent / path


def copy_file(source, bundle, relative, origins):
    target = bundle / relative
    target.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy2(source, target)
    file_hash = digest(target)
    if digest(source) != file_hash:
        raise ValueError(f"Copy hash mismatch: {relative}")
    origins.append({"package_file": relative, "original_filename": source.name,
                    "sha256": file_hash, "bytes": target.stat().st_size})


READ_XYZ = '''"""Query XYZ or a two-point distance; requires Python and NumPy only."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--doe", type=int, choices=(0, 6), required=True)
parser.add_argument("--u", type=int, required=True)
parser.add_argument("--v", type=int, required=True)
parser.add_argument("--u2", type=int)
parser.add_argument("--v2", type=int)
args = parser.parse_args()
if (args.u2 is None) != (args.v2 is None):
    parser.error("--u2 and --v2 must be given together")
root = Path(__file__).resolve().parents[1]
folder = root / ("DOE" + str(args.doe))
if not folder.is_dir():
    parser.error("This package does not contain the selected DOE setting")
report = json.loads((folder / "metadata/xyz_report.json").read_text(encoding="utf-8"))
xyz = np.load(folder / "data/xyz_m_provisional.npy", allow_pickle=False, mmap_mode="r")

def point(u, v):
    if not (0 <= v < xyz.shape[0] and 0 <= u < xyz.shape[1]):
        parser.error("Pixel outside image")
    p = xyz[v, u].astype(float)
    valid = bool(np.isfinite(p).all())
    return p, {"u": u, "v": v, "valid": valid,
               "xyz_m": p.tolist() if valid else None,
               "range_m": float(np.linalg.norm(p)) if valid else None}

a, first = point(args.u, args.v)
result = {"status": "provisional_metric_accuracy_unverified", "doe": args.doe,
          "source_number": report["source_number"],
          "frame": "left_IR_optical_Xright_Ydown_Zforward", "point1": first}
if args.u2 is not None:
    b, second = point(args.u2, args.v2)
    result["point2"] = second
    result["point_distance_m"] = float(np.linalg.norm(a-b)) if first["valid"] and second["valid"] else None
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
'''


VERIFY = '''"""Verify packaged file hashes using standard Python only."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "SHA256.json").read_text(encoding="utf-8"))
errors = []
for name, expected in manifest.items():
    path = (root / name).resolve()
    if root not in path.parents or not path.is_file():
        errors.append(name)
        continue
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    if result.hexdigest() != expected:
        errors.append(name)
if errors:
    raise SystemExit("FAILED: " + ", ".join(errors))
print("PASS: " + str(len(manifest)) + " files verified; physical accuracy is unverified")
'''


def documents(bundle, frames, batch, scene_kind):
    scene_title = "无货物参考场景" if scene_kind == "empty" else "货物场景"
    case_label = " / ".join(f"DOE={item['doe']}（第{item['number']}组）" for item in frames)
    if scene_kind == "empty":
        scene_note = "所选图像的中央白板上未摆放前面样例中的包装盒；白板、设备和前景结构仍保留。无货物指没有本次待测包装盒，不表示整幅画面没有物体。"
    else:
        scene_note = "所选图像可见完整包装盒。多组样例的包装盒位置和拍摄角度可能不同，不作为仅改变DOE的受控精度对比。"
    folder_note = "、".join(f"`DOE{item['doe']}/`" for item in frames)
    commands = "\n".join(f"python tools/read_xyz.py --doe {item['doe']} --u {item['output_size'][0] // 2} --v {item['output_size'][1] // 2}" for item in frames)
    first_doe = frames[0]["doe"]
    rows, cards = [], []
    for item in frames:
        doe, number = item["doe"], item["number"]
        w, h = item["output_size"]
        iw, ih = item["input_size"]
        count, original_count = item["valid_points"], item["valid_inference_positions"]
        base = f"DOE{doe}"
        rows.append(f"| DOE={doe} / 第{number}组 | {w}×{h} | {iw}×{ih} | {count:,} | {original_count:,} |")
        cards.append(f'''<section><h2>DOE={doe} · 第{number}组</h2>
<img src="{base}/images/preview.jpg" alt="第{number}组左红外与暂定深度对照">
<p>输出 {w}×{h}；推理 {iw}×{ih}；可计算点 {count:,} 个。原始推理网格中满足有限正视差及水平重叠条件的位置 {original_count:,} 个；有效数量不代表准确率。</p>
<ul><li><a href="{base}/data/cloud_ir_gray_provisional.ply">PLY点云</a> · <a href="{base}/data/xyz_m_provisional.npy">逐像素XYZ</a> · <a href="{base}/data/depth_m_provisional.npy">原尺寸深度</a></li>
<li><a href="{base}/images/left_ir.png">原始左IR</a> · <a href="{base}/images/right_ir.png">原始右IR</a> · <a href="{base}/images/depth_colour.png">深度伪彩色</a> · <a href="{base}/images/validity_mask.png">有效掩膜</a></li>
<li><a href="{base}/metadata/xyz_report.json">坐标报告</a> · <a href="{base}/metadata/frame_report.json">深度报告</a> · <a href="{base}/metadata/provenance.json">来源与计数</a></li></ul>
<p><code>python tools/read_xyz.py --doe {doe} --u {w // 2} --v {h // 2}</code></p></section>''')
    colour = batch.get("shared_colour_range_m")
    colour_note = (f"深度伪彩色统一显示范围为{colour[0]}～{colour[1]}米，深度数值本身未按色标截断。"
                   if colour else "深度伪彩色用于预览，实际数值以深度NPY为准。")
    readme = f'''# 双目{scene_title}深度 / XYZ / 点云查看包

本包包含{case_label}。{scene_note}解压后双击 `START_HERE.html` 查看样例，PLY可导入点云查看软件。

| 条件 / 原组号 | 输出宽×高 | 模型推理宽×高 | PLY有效点数 | 推理网格几何有效位置数 |
|---|---|---|---:|---:|
{chr(10).join(rows)}

## 文件与单位

样例目录为{folder_note}；所有JSON中的数据文件路径均以整个包根目录为基准。

| 每组文件 | 内容 |
|---|---|
| data/xyz_m_provisional.npy | H×W×3、float32，`xyz[v,u]=[X,Y,Z]`，单位米 |
| data/depth_m_provisional.npy | H×W、float32光轴Z，单位米 |
| data/cloud_ir_gray_provisional.ply | 有效XYZ，左IR灰度着色，按图像行优先排列 |
| data/disparity_pixels_native.npy | 原图网格上的像素视差 |
| data/disparity_pixels_inference.npy | 模型推理网格上的像素视差 |
| images/left_ir.png、right_ir.png | 原始左右红外图，复制时未重编码 |
| images/depth_colour.png | 与原图同尺寸的深度伪彩色 |
| images/validity_mask.png | 255可计算、0无效；不是精度或置信度 |
| images/preview.jpg | 原图与深度对照排版；不能当作深度数据 |
| metadata/ | 逐组参数、真实点数、来源文件名及哈希 |

共享 `calibration/calibration.json` 是标定原字节副本；顶层 `metadata/metric_report.json` 与 `metadata/inference_report.json` 保存本次处理和模型推理参数，模型权重不随包提供。{colour_note}

坐标原点为左IR相机光心，X向右、Y向下、Z沿光轴向前，单位米。像素(u,v)左上角为(0,0)，数组用[v,u]索引；Z不是欧氏距离，到光心距离为sqrt(X²+Y²+Z²)。无效像素的XYZ三个通道均为NaN，PLY不含无效点。

**深度NPY、XYZ和PLY已经是米，读取时不要再乘标定中的 `depth_unit_mm`。** 模型深度使用双目焦距、基线和视差换算，`depth_unit_mm`是设备原始深度采集设置。

## 查询坐标与两点距离

Python安装NumPy后执行（u/v使用待测端点的原图像素坐标）：

```text
{commands}
```

也可 `xyz = numpy.load('DOE{first_doe}/data/xyz_m_provisional.npy')`，再读 `xyz[v,u]`。加`--u2 900 --v2 600`可查两点距离；仅在端点可见、深度可靠且端点选择正确时有测量意义。本包没有标注实物边界或尺量值。

文件完整性检查无需NumPy：`python tools/verify_package.py`。顶层 `SHA256.json` 记录包内文件哈希；ZIP旁 `.zip.sha256` 记录整个压缩包哈希。

## 当前结果边界

- 输出与原始左IR尺寸、方向及像素网格一致。半分辨率推理后以最近邻恢复尺寸，恢复时视差按宽度比例换算，深度米值不额外缩放；更多输出点不代表新增独立观测细节。
- 米制结果以历史IR已正确校正、提供的标定适用为前提；这两项仍待确认。有效掩膜与数值回投检查仅说明计算/文件一致性，不是实物精度验收。
- 尚无尺量真值，不能声称达到毫米级、厘米级或比较两种DOE的尺寸精度。需要指定可见边或平面并提供实测尺寸，再计算绝对及相对误差。
- 当前为左IR相机坐标系，未完成RGB配准、地面/世界坐标转换；未重建遮挡背面。DEPTH→COLOR外参不能直接证明IR_LEFT与DEPTH坐标系相同，历史RGB与标定分辨率的对应也待确认，因此本包不含RGB图。
'''
    (bundle / "README_先读我.md").write_text(readme, encoding="utf-8")
    serial = escape(str(batch.get("calibration_serial", "相机序列号见标定")))
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>双目{scene_title} · {escape(case_label)}</title>
<style>body{{font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif;background:#f4f6f3;color:#183b3f;max-width:1120px;margin:35px auto;padding:0 22px}}h1,h2{{line-height:1.3}}img{{width:100%;border-radius:10px}}a{{color:#006c65}}section{{background:white;padding:18px 24px;border-radius:10px;margin:18px 0}}code{{background:#edf2f0;padding:3px 6px}}li{{margin:6px 0}}.note{{border-left:4px solid #b67b27}}</style>
<h1>双目{scene_title} · 深度与三维坐标</h1><p>{serial} · FoundationStereo · {escape(case_label)}</p><p>{escape(scene_note)}</p>
<p>解压后可离线查看。XYZ与深度单位均为米，原点在左IR光心，X右、Y下、Z前；无效像素为NaN。{escape(colour_note)}</p>
{''.join(cards)}
<section class="note"><h2>条件性三维估计，实物精度尚未验证</h2><p>半分辨率推理后恢复原始尺寸，没有新增独立深度细节。历史IR校正状态和标定适用性仍待确认；没有尺量真值，不能据此判定尺寸精度或DOE优劣。</p><p>尚未做RGB配准、地面/世界坐标转换或遮挡背面重建。读取NPY/PLY时不要再次乘标定中的depth_unit_mm。</p></section>
<p><a href="README_先读我.md">完整中文说明</a> · <a href="calibration/calibration.json">相机参数</a> · <a href="metadata/inference_report.json">模型推理参数</a> · <a href="metadata/metric_report.json">米制处理参数</a> · <a href="SHA256.json">文件哈希</a></p>
<p>文件完整性检查：<code>python tools/verify_package.py</code>。</p></html>'''
    (bundle / "START_HERE.html").write_text(page, encoding="utf-8")


def verify_archive(archive, bundle, manifest, frames):
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
        prefix = bundle.name + "/"
        for name, expected in manifest.items():
            if hashlib.sha256(zipped.read(prefix + name)).hexdigest() != expected:
                raise ValueError(f"Archive hash mismatch: {name}")
        if json.loads(zipped.read(prefix + "SHA256.json")) != manifest:
            raise ValueError("Archive manifest differs")
        for frame in frames:
            base = f"DOE{frame['doe']}"
            get = lambda name: zipped.read(prefix + base + "/" + name)
            xyz = np.load(io.BytesIO(get("data/xyz_m_provisional.npy")), allow_pickle=False)
            depth = np.load(io.BytesIO(get("data/depth_m_provisional.npy")), allow_pickle=False)
            disparity = np.load(io.BytesIO(get("data/disparity_pixels_native.npy")), allow_pickle=False)
            raw = np.load(io.BytesIO(get("data/disparity_pixels_inference.npy")), allow_pickle=False)
            shape = tuple(reversed(frame["output_size"]))
            if xyz.shape != (*shape, 3) or depth.shape != shape or disparity.shape != shape:
                raise ValueError(f"Output dimensions differ: {base}")
            if raw.shape != tuple(reversed(frame["input_size"])):
                raise ValueError(f"Inference dimensions differ: {base}")
            np.testing.assert_array_equal(xyz[..., 2], depth)
            valid = np.isfinite(xyz).all(axis=2)
            if int(valid.sum()) != frame["valid_points"] or not np.isnan(xyz[~valid]).all():
                raise ValueError(f"XYZ validity differs: {base}")
            for name in (n for n in manifest if n.startswith(base + "/images/")):
                pixels = cv2.imdecode(np.frombuffer(zipped.read(prefix + name), np.uint8), cv2.IMREAD_UNCHANGED)
                if pixels is None or (not name.endswith("preview.jpg") and pixels.shape[:2] != shape):
                    raise ValueError(f"Image decode/dimensions failed: {name}")
                if name.endswith("validity_mask.png"):
                    np.testing.assert_array_equal(pixels > 0, valid)
            ply = get("data/cloud_ir_gray_provisional.ply")
            marker = b"end_header\n"
            offset = ply.index(marker) + len(marker)
            header = ply[:offset].decode("ascii")
            if (f"element vertex {frame['valid_points']}\n" not in header or
                    "format binary_little_endian 1.0" not in header):
                raise ValueError(f"PLY header differs: {base}")
            dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                              ("red", "u1"), ("green", "u1"), ("blue", "u1")])
            vertices = np.frombuffer(ply[offset:], dtype=dtype)
            if len(vertices) != frame["valid_points"]:
                raise ValueError(f"PLY point count differs: {base}")
            for index, name in enumerate(("x", "y", "z")):
                np.testing.assert_array_equal(vertices[name], xyz[..., index][valid])
            gray = cv2.imdecode(np.frombuffer(get("images/left_ir.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
            for name in ("red", "green", "blue"):
                np.testing.assert_array_equal(vertices[name], gray[valid])

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ("src", "href"):
                    target = (bundle / value).resolve()
                    if bundle not in target.parents or not target.is_file():
                        raise ValueError(f"Nonportable HTML link: {value}")
    Links().feed((bundle / "START_HERE.html").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--metric", required=True, type=Path, help="Metric batch with report.json and one or two frame folders")
    parser.add_argument("--xyz", required=True, type=Path, help="XYZ root with matching frame folders")
    parser.add_argument("--output", required=True, type=Path, help="New package directory; a same-name ZIP is also created")
    parser.add_argument("--scene-kind", choices=("cargo", "empty"), default="cargo", help="Visually verified source scene description")
    args = parser.parse_args()
    metric, xyz_root, bundle = args.metric.resolve(), args.xyz.resolve(), args.output.resolve()
    archive = bundle.with_name(bundle.name + ".zip")
    checksum = archive.with_suffix(".zip.sha256")
    verification_path = archive.with_suffix(".verification.json")
    if any(path.exists() for path in (bundle, archive, checksum, verification_path)):
        raise ValueError("Use a distinct new package name; an output or sidecar already exists")
    batch = read_json(metric / "report.json")
    selections = batch["results"]
    does = [int(row["doe"]) for row in selections]
    if not 1 <= len(selections) <= 2 or len(set(does)) != len(does) or any(doe not in (0, 6) for doe in does):
        raise ValueError("Require one or two results, at most one per DOE=6/DOE=0 setting")
    calibration = metric / "calibration.json"
    if not calibration.is_file():
        raise FileNotFoundError(calibration)
    jobs = []
    for row in sorted(selections, key=lambda item: -int(item["doe"])):
        folder = row["folder"]
        if Path(folder).name != folder or PureWindowsPath(folder).name != folder:
            raise ValueError("Frame folder must be a single directory name")
        frame, sample = metric / folder, xyz_root / folder
        metadata, report = read_json(frame / "result.json"), read_json(sample / "report.json")
        if (int(metadata["number"]) != int(row["number"]) or int(metadata["doe"]) != int(row["doe"])
                or int(report["source_number"]) != int(row["number"]) or int(report["doe"]) != int(row["doe"])):
            raise ValueError(f"Source identity mismatch: {folder}")
        if metadata["status"] != "conditional_on_rectified_input" or report["unit"] != "metre":
            raise ValueError(f"Expected conditional metre XYZ: {folder}")
        files = {
            "data/xyz_m_provisional.npy": sample / report["xyz_file"],
            "data/cloud_ir_gray_provisional.ply": sample / report["pointcloud_file"],
            "data/depth_m_provisional.npy": frame / metadata["depth_file"],
            "data/disparity_pixels_native.npy": frame / metadata["disparity_native_file"],
            "data/disparity_pixels_inference.npy": source_path(metadata["raw_disparity_path"], frame),
            "images/left_ir.png": source_path(metadata["source_left"], frame),
            "images/right_ir.png": source_path(metadata["source_right"], frame),
            "images/depth_colour.png": frame / "depth_colour.png",
            "images/validity_mask.png": frame / metadata["validity_mask"],
            "images/preview.jpg": frame / "preview.jpg",
        }
        for source in files.values():
            if not source.is_file():
                raise FileNotFoundError(source)
        if (report["source_depth_sha256"] != digest(files["data/depth_m_provisional.npy"])
                or report["calibration_sha256"] != digest(calibration)
                or metadata["calibration_sha256"] != digest(calibration)):
            raise ValueError(f"Source hash mismatch: {folder}")
        jobs.append((metadata, report, files))
    bundle.mkdir(parents=True)
    origins = []
    copy_file(calibration, bundle, "calibration/calibration.json", origins)
    frames = []
    for metadata, report, files in jobs:
        doe, number = int(metadata["doe"]), int(metadata["number"])
        base = f"DOE{doe}"
        frame_origins = []
        for relative, source in files.items():
            copy_file(source, bundle, base + "/" + relative, frame_origins)
        origins.extend(frame_origins)
        raw = np.load(files["data/disparity_pixels_inference.npy"], allow_pickle=False)
        u = np.arange(raw.shape[1], dtype=np.float32)[None, :]
        raw_valid = np.isfinite(raw) & (raw > 0) & (u - raw >= 0) & (u - raw <= raw.shape[1] - 1)
        frame_info = {"number": number, "doe": doe, "source_folder": metadata["folder"],
                      "input_size": metadata["input_size"], "output_size": metadata["output_size"],
                      "valid_points": int(report["valid_points"]),
                      "valid_inference_positions": int(raw_valid.sum()),
                      "inference_count_definition": "finite positive disparity with projected right x inside inference image",
                      "model_weights_included": False, "metric_accuracy_verified": False,
                      "files": frame_origins}
        frames.append(frame_info)
        clean_report = portable(report)
        clean_report.update(source_frame=metadata["folder"], paths_relative_to="package_root",
                            xyz_file=base + "/data/xyz_m_provisional.npy",
                            pointcloud_file=base + "/data/cloud_ir_gray_provisional.ply",
                            depth_file=base + "/data/depth_m_provisional.npy",
                            calibration_file="calibration/calibration.json")
        clean_metadata = portable(metadata)
        clean_metadata.update(paths_relative_to="package_root", source_left=base + "/images/left_ir.png",
                              source_right=base + "/images/right_ir.png",
                              raw_disparity_path=base + "/data/disparity_pixels_inference.npy",
                              disparity_native_file=base + "/data/disparity_pixels_native.npy",
                              depth_file=base + "/data/depth_m_provisional.npy",
                              validity_mask=base + "/images/validity_mask.png")
        write_json(bundle / base / "metadata/xyz_report.json", clean_report)
        write_json(bundle / base / "metadata/frame_report.json", clean_metadata)
        write_json(bundle / base / "metadata/provenance.json", frame_info)
        inference_result = metadata.get("source_result_path")
        if inference_result and source_path(inference_result, metric).is_file():
            write_json(bundle / base / "metadata/inference_frame.json", portable(read_json(source_path(inference_result, metric))))
    clean_batch = portable({key: value for key, value in batch.items() if key != "results"})
    clean_batch.update(paths_relative_to="package_root", calibration_copy="calibration/calibration.json",
                       scene_kind=args.scene_kind,
                       results=[{"number": f["number"], "doe": f["doe"], "folder": f"DOE{f['doe']}"} for f in frames])
    source_trial = source_path(batch["source_trial"], metric) if batch.get("source_trial") else None
    inference = (read_json(source_trial / "report.json") if source_trial and (source_trial / "report.json").is_file()
                 else {key: batch[key] for key in ("checkpoint", "attention_backend", "iterations", "scale") if key in batch})
    inference = {key: value for key, value in inference.items() if key != "results"}
    clean_inference = portable(inference)
    checkpoint = inference.get("checkpoint", batch.get("checkpoint"))
    if checkpoint:
        checkpoint_parts = str(checkpoint).replace("\\", "/").split("/")
        clean_inference["checkpoint_identifier"] = "/".join(checkpoint_parts[-2:])
    clean_inference.update(model_weights_included=False, selected_source_numbers=[f["number"] for f in frames])
    write_json(bundle / "metadata/metric_report.json", clean_batch)
    write_json(bundle / "metadata/inference_report.json", clean_inference)
    write_json(bundle / "metadata/provenance.json", {"frames": frames, "copied_files": origins})
    (bundle / "tools").mkdir()
    (bundle / "tools/read_xyz.py").write_text(READ_XYZ, encoding="utf-8")
    (bundle / "tools/verify_package.py").write_text(VERIFY, encoding="utf-8")
    documents(bundle, frames, batch, args.scene_kind)
    manifest = {p.relative_to(bundle).as_posix(): digest(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
    write_json(bundle / "SHA256.json", manifest)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for file in sorted(bundle.rglob("*")):
            if file.is_file():
                zipped.write(file, f"{bundle.name}/{file.relative_to(bundle).as_posix()}")
    verify_archive(archive, bundle, manifest, frames)
    archive_hash = digest(archive)
    checksum.write_text(archive_hash + "  " + archive.name + "\n", encoding="utf-8")
    verification = {"status": "passed", "archive": str(archive), "archive_bytes": archive.stat().st_size,
                    "archive_sha256": archive_hash, "verified_content_files": len(manifest),
                    "frames": [{key: f[key] for key in ("number", "doe", "input_size", "output_size", "valid_points")} for f in frames],
                    "zip_crc_verified": True, "manifest_verified_inside_zip": True,
                    "xyz_depth_and_ply_verified_inside_zip": True, "images_decoded_inside_zip": True,
                    "html_links_portable": True, "metric_accuracy_verified": False}
    write_json(verification_path, verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
