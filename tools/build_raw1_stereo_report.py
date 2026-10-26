"""Verify the 15-pair Raw1 run, write a Chinese report, and archive it on E:."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from urllib.parse import unquote
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = Path(r"E:\gpt\_release")
CHUNK = 4 * 1024 * 1024
FILE_KEYS = (
    "left_input", "right_input", "disparity", "inverse_disparity",
    "positive_mask", "overlap_mask", "disparity_colour", "preview",
    "source_color", "device_raw", "device_display",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def local_file(base, relative, allowed_root=None):
    relative = str(relative).replace("\\", "/")
    require(not re.match(r"^[A-Za-z]:", relative), f"Absolute path: {relative}")
    candidate = (base / relative).resolve()
    require((allowed_root or base) in candidate.parents, f"Path leaves package: {relative}")
    require(candidate.is_file() and not candidate.is_symlink(), f"Missing file: {candidate}")
    return candidate


def read_image(path, mode=cv2.IMREAD_UNCHANGED):
    result = cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)
    require(result is not None, f"Cannot decode image: {path}")
    return result


def near(actual, expected, label, tolerance=1e-6):
    require(math.isfinite(float(actual)), f"Nonfinite {label}")
    require(abs(float(actual) - float(expected)) <= tolerance, f"Mismatch {label}: {actual} != {expected}")


def verify_run(base, report, package_root=None):
    package_root = package_root or base
    rectified = base != package_root
    require(report["status"] == "completed", "Inference has not completed")
    require(report["iterations"] == 32 and report["scale"] == 1.0, "Unexpected inference configuration")
    require(report["metric_depth"] is False, "This report supports nonmetric output only")
    require(bool(report["rectification"]), "Missing rectification status")
    if not rectified:
        require(report["rectification"] == "unrectified_user_confirmed_no_rectification_applied", "Expected user-confirmed unrectified baseline")
    else:
        require(report["rectification"] == "rectified_with_supplied_calibration_scale_unknown", "Unexpected rectified branch status")
    require(report["pair_order"] == "left_half_as_left_right_half_as_right_unverified", "Unexpected pair order")
    rows = sorted(report["results"], key=lambda r: r["number"])
    require(len(rows) == 15 and [r["number"] for r in rows] == list(range(1, 16)), "Expected exactly pairs 1–15")
    source_zip = Path(report["source_zip"])
    require(source_zip.is_file(), f"Missing source ZIP: {source_zip}")
    require(digest(source_zip) == report["source_zip_sha256"], "Source ZIP SHA-256 mismatch")
    source_manifest = read_json(package_root / "source_manifest.json")
    require(isinstance(source_manifest, list) and len(source_manifest) == 45, "Expected 45 source PNG manifest entries")
    manifest_names = set()
    for entry in source_manifest:
        name = entry["name"]
        require(name not in manifest_names, f"Duplicate source manifest: {name}")
        manifest_names.add(name)
        path = local_file(package_root / "source", name)
        require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], f"Source digest mismatch: {name}")
        decoded = read_image(path)
        require(list(decoded.shape) == entry["shape"] and str(decoded.dtype) == entry["dtype"], f"Source metadata mismatch: {name}")
    source_files = {p.relative_to(package_root / "source").as_posix() for p in (package_root / "source").rglob("*") if p.is_file()}
    require(source_files == manifest_names, "Source directory and manifest differ")
    maps = None
    if rectified:
        maps = np.load(base / "rectification.npz", allow_pickle=False)
        for key in ("left_map_x", "left_map_y", "right_map_x", "right_map_y"):
            require(maps[key].shape == (720, 1280) and maps[key].dtype == np.float32 and np.isfinite(maps[key]).all(), f"Invalid rectification map: {key}")
        rectification_metadata = read_json(base / "rectification.json")
        require(rectification_metadata["rectified_canvas_size"] == [1280, 720], "Unexpected rectification canvas")
        require(rectification_metadata["metric_scale_verified"] is False, "Unexpected verified metric scale")
        require(digest(base / "calibration.json") == rectification_metadata["calibration_sha256"], "Rectification calibration digest mismatch")
        for side in ("left", "right"):
            mx, my = maps[f"{side}_map_x"], maps[f"{side}_map_y"]
            support = read_image(base / f"{side}_source_support_mask.png")
            require(support.shape == (720, 1280) and support.dtype == np.uint8, f"Support mask shape/dtype: {side}")
            expected = (mx >= 0) & (mx < 1279) & (my >= 0) & (my < 719)
            require(np.array_equal(support > 0, expected), f"Support mask values: {side}")
    checks, csv_rows, used_sources = [], [], set()
    for row in rows:
        number = row["number"]
        require(row["folder"] == f"pair_{number:02d}", f"Unexpected pair folder: {number}")
        require(row["input_size"] == [1280, 720] and row["padded_size"] == [1280, 736], f"Unexpected size: {number}")
        files = row["files"]
        paths = {key: local_file(base, files[key], package_root) for key in FILE_KEYS}
        source = read_image(paths["source_color"])
        raw = read_image(paths["device_raw"])
        device_display = read_image(paths["device_display"])
        require(source.shape == (720, 2560, 3) and source.dtype == np.uint8, f"Unexpected source color: {number}")
        require(raw.shape == (720, 1280) and raw.dtype == np.uint16, f"Unexpected device Raw: {number}")
        require(device_display.shape == (720, 1280, 3) and device_display.dtype == np.uint8, f"Unexpected device display: {number}")
        left = read_image(paths["left_input"])
        right = read_image(paths["right_input"])
        require(left.shape == right.shape == (720, 1280), f"Inputs must be single-channel gray: {number}")
        require(left.dtype == right.dtype == np.uint8, f"Input dtype: {number}")
        expected_left = cv2.cvtColor(source[:, :1280], cv2.COLOR_BGR2GRAY)
        expected_right = cv2.cvtColor(source[:, 1280:], cv2.COLOR_BGR2GRAY)
        if rectified:
            expected_left = cv2.remap(expected_left, maps["left_map_x"], maps["left_map_y"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            expected_right = cv2.remap(expected_right, maps["right_map_x"], maps["right_map_y"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        require(np.array_equal(left, expected_left), f"Left input pixels changed: {number}")
        require(np.array_equal(right, expected_right), f"Right input pixels changed: {number}")
        d = np.load(paths["disparity"], allow_pickle=False)
        inv = np.load(paths["inverse_disparity"], allow_pickle=False)
        require(d.shape == inv.shape == (720, 1280) and d.dtype == inv.dtype == np.float32, f"Array shape/dtype: {number}")
        positive = np.isfinite(d) & (d > 0)
        overlap = positive & (np.arange(1280)[None, :] - d >= 0) & (np.arange(1280)[None, :] - d <= 1279)
        require(positive.any(), f"No positive finite disparities: {number}")
        for key, expected in (("positive_mask", positive), ("overlap_mask", overlap)):
            mask = np.load(paths[key], allow_pickle=False) if paths[key].suffix == ".npy" else read_image(paths[key])
            require(mask.shape == (720, 1280) and mask.dtype in (np.dtype("bool"), np.dtype("uint8")), f"Mask shape/dtype: {number}/{key}")
            require(np.array_equal(mask > 0, expected), f"Mask values: {number}/{key}")
        require(np.allclose(inv[positive], 1.0 / d[positive], rtol=1e-6, atol=1e-7), f"Inverse disparity relation: {number}")
        require(np.isnan(inv[~positive]).all(), f"Invalid inverse pixels must be NaN: {number}")
        near(row["positive_finite_ratio"], positive.mean(), f"Pair {number} positive ratio")
        near(row["horizontal_overlap_ratio"], overlap.mean(), f"Pair {number} overlap ratio")
        near(row["raw_nonzero_ratio"], (raw > 0).mean(), f"Pair {number} Raw nonzero ratio")
        require(list(row["raw_value_range"]) == [int(raw.min()), int(raw.max())], f"Raw range: {number}")
        percentiles = np.percentile(d[positive], [2, 50, 98])
        require(np.allclose(row["disparity_percentiles"], percentiles, rtol=1e-6, atol=1e-5), f"Disparity percentiles: {number}")
        require(math.isfinite(row["inference_seconds"]) and row["inference_seconds"] > 0, f"Invalid inference time: {number}")
        for key in ("disparity_colour", "preview"):
            image = read_image(paths[key])
            require(image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8, f"Preview dtype/shape: {number}/{key}")
        for source_key, file_key in (("color", "source_color"), ("raw", "device_raw"), ("display", "device_display")):
            member = row["source_members"][source_key]
            require(member in manifest_names, f"Unknown source member: {member}")
            require(paths[file_key] == local_file(package_root / "source", member), f"Source path mismatch: {member}")
            require(member not in used_sources, f"Source reused: {member}")
            used_sources.add(member)
        checks.append({"number": number, "coordinate_grid": "rectified_left" if rectified else "original_left_half", "array_shape": list(d.shape), "array_dtype": str(d.dtype),
                       "input_pixels_exact": True, "inverse_disparity_relation": True,
                       "positive_finite_mask_exact": True, "horizontal_overlap_mask_exact": True,
                       "source_images_decoded": True, "reported_statistics_match": True})
        csv_rows.append({"number": number, "width": 1280, "height": 720,
                         "inference_seconds": row["inference_seconds"],
                         "positive_finite_ratio": float(positive.mean()), "horizontal_overlap_ratio": float(overlap.mean()),
                         "disparity_p2_px": float(percentiles[0]), "disparity_p50_px": float(percentiles[1]),
                         "disparity_p98_px": float(percentiles[2]), "device_raw_nonzero_ratio": float((raw > 0).mean()),
                         "device_raw_min": int(raw.min()), "device_raw_max": int(raw.max()),
                         "disparity_file": files["disparity"], "inverse_disparity_file": files["inverse_disparity"],
                         "preview_file": files["preview"], "source_color": files["source_color"],
                         "source_raw": files["device_raw"], "source_display": files["device_display"]})
    require(used_sources == manifest_names, "Not all 45 source files covered exactly once")
    require(not list(base.rglob("*.ply")), "Unexpected metric point cloud in nonmetric release")
    if maps is not None:
        maps.close()
    return rows, csv_rows, checks


def make_overview(base, rows):
    width, height, margin, label = 560, 340, 12, 32
    canvas = np.full((5 * (height + label + margin) + margin, 3 * (width + margin) + margin, 3), 244, dtype=np.uint8)
    for index, row in enumerate(rows):
        preview = read_image(local_file(base, row["files"]["preview"]), cv2.IMREAD_COLOR)
        scale = min(width / preview.shape[1], height / preview.shape[0])
        image = cv2.resize(preview, (round(preview.shape[1] * scale), round(preview.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        y = margin + (index // 3) * (height + label + margin)
        x = margin + (index % 3) * (width + margin)
        cv2.putText(canvas, f"Pair {row['number']:02d} | disparity P50 {row['disparity_percentiles'][1]:.1f} px", (x + 8, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)
        y += label + (height - image.shape[0]) // 2
        x += (width - image.shape[1]) // 2
        canvas[y:y + image.shape[0], x:x + image.shape[1]] = image
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 91])
    require(ok, "Overview encoding failed")
    with (base / "overview_all15.jpg").open("xb") as stream:
        stream.write(encoded.tobytes())
    read_image(base / "overview_all15.jpg")


def copy_provenance(base, runtime_source):
    sources = [Path(__file__).resolve(), ROOT / "tools" / "run_stereo_zip_batch.py", runtime_source.resolve()]
    if (base / "rectified" / "report.json").is_file():
        sources.append(ROOT / "tools" / "prepare_raw1_rectified.py")
    require(len({p.name for p in sources}) == len(sources), "Provenance filename collision")
    target = base / "provenance"
    target.mkdir(exist_ok=True)
    records = []
    for source in sources:
        require(source.is_file(), f"Missing provenance source: {source}")
        dest = target / source.name
        if dest.exists():
            require(digest(dest) == digest(source), f"Existing provenance differs: {dest}")
        else:
            with source.open("rb") as inp, dest.open("xb") as out:
                shutil.copyfileobj(inp, out, CHUNK)
        records.append({"original_path": str(source), "file": dest.relative_to(base).as_posix(), "sha256": digest(dest)})
    return records


def write_documents(base, report, rows, csv_rows, provenance, filename="REPORT.md"):
    with (base / "frames.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    times = [r["inference_seconds"] for r in rows]
    table = []
    for row in rows:
        p2, p50, p98 = row["disparity_percentiles"]
        table.append(f"| {row['number']:02d} | {row['inference_seconds']:.3f} | {row['positive_finite_ratio']:.2%} | {row['horizontal_overlap_ratio']:.2%} | {p2:.2f} / {p50:.2f} / {p98:.2f} | {row['raw_nonzero_ratio']:.2%} | [{row['raw_value_range'][0]}～{row['raw_value_range'][1]}]({row['files']['device_raw']}) | [预览]({row['files']['preview']}) · [NPY]({row['files']['disparity']}) |")
    sample = rows[7]
    files = sample["files"]
    environment = json.dumps(report.get("environment", {}), ensure_ascii=False, indent=2)
    provenance_links = "、".join(f"[{Path(p['file']).name}]({p['file']})" for p in provenance)
    diagnostic = "\n附加配对诊断见 [pair_diagnostic.json](pair_diagnostic.json)。诊断为稀疏匹配参考，不能据此确认全部像素的左右顺序、校正状态或深度精度。\n" if (base / "pair_diagnostic.json").is_file() else ""
    text = f"""# _Raw1：15组未校正输入诊断对照报告

已对本次 ZIP 中编号 **1～15 的全部15组图像**运行标准 FoundationStereo，每组使用原生 **1280×720** 输入、**32次内部迭代**。本包保留45张来源PNG，保存15组模型视差、逆视差、掩膜和对照预览。

用户已确认来源图没有进行双目校正。本页记录未校正输入的诊断对照；它不满足标准水平双目匹配的校正前提，不能作为正式几何结果。本次网络直接输出像素视差，设备Raw编码/单位未确认，因此不生成米制深度、XYZ或PLY，也不把运行完成当作货物长宽高或实物尺寸精度验收完成。若包内包含校正分支，请以 [主报告](REPORT.md) 的校正后流程为主。

## 1. 模型输入是什么

来源压缩包：`{report['source_zip']}`。用户说明这些材料使用同一台相机；文件名或采集模式不同不作为另一台物理相机的证据。本批次的图像格式与之前1280×800采集模式不同，历史标定的适用关系不能由“同一相机”直接推得。

| 每组来源 | 实际格式 | 用途 |
|---|---|---|
| `Image_ColorN.png` | 2560×720、uint8、三通道左右拼接图 | 从中切分模型左右输入 |
| `Image_Depth_RawN.png` | 1280×720、uint16 | 原样保留的设备数值；编码/单位未确认，不输入网络 |
| `Image_Depth_DispalyN.png` | 1280×720、uint8、三通道 | 设备已有显示图，仅供外观对照，不输入网络 |

按文件编号配对。暂按拼接图左半幅为左图、右半幅为右图，每半幅1280×720；这一物理顺序仍待采集记录确认。两半幅分别通过 OpenCV `BGR2GRAY` 转为uint8灰度，并将灰度复制到三个通道送入模型。这与之前47组运行使用的灰度三通道预处理一致。没有外部缩小、插值恢复或重新校正。

1280已可被32整除；高度720上下各复制补齐8像素，网络实际张量为 `[1,3,736,1280]`，最终去补边恢复 `[720,1280]`。保存的左右输入PNG为补边前灰度图。

原始文件哈希、字节数、shape与dtype见 [source_manifest.json](source_manifest.json)，实际运行记录见 [report.json](report.json)。所有45张来源图均恰好归入一组。
{diagnostic}
## 2. 中间模型怎样处理

```mermaid
flowchart TD
    A[2560×720 拼接图] --> B[左右切分：各1280×720]
    B --> C[分别转灰度、复制三通道、上下补边]
    C --> D[标准化与共享特征提取]
    D --> E[水平匹配代价体、3D聚合及注意力]
    E --> F[初始化视差与多尺度上下文]
    F --> G[GRU内部迭代细化32次]
    G --> H[学习上采样、裁掉补边]
    H --> I[1280×720 float32像素视差]
    I --> J[逆视差、数值掩膜、可视化]
    K[设备Raw和Display] --> L[原样保留并对照]
```

| 阶段 | 本次处理与含义 |
|---|---|
| 归一化 | 像素除以255，再按每通道mean `[0.485,0.456,0.406]`、std `[0.229,0.224,0.225]` 标准化。灰度复制的三个通道在标准化后数值并不相同。 |
| 特征提取 | 共享 EdgeNeXt 局部特征与基于DINOv2的 DepthAnything 特征，在模型内部融合；辅助单目结果不作为最终深度。 |
| 水平匹配 | 在主要1/4网格上沿同一行比较左右特征，建立分组相关和特征拼接代价体；这依赖输入符合双目校正几何的前提。 |
| 聚合与初始化 | 3D卷积hourglass及沿视差方向的注意力聚合，再由候选概率加权初始化视差。 |
| 32次细化 | 三尺度上下文、当前视差及邻域相关送入GRU，逐次加上视差修正量。32次是单次推理内部迭代，没有训练。 |
| 回到原网格 | 依据学习的邻域权重上采样，并去掉上下补边。网络内部降采样不改变本次外部输入原生分辨率。 |

本次注意力后端记录为 `{report['attention_backend']}`；使用既有Windows兼容运行入口。完整权重按strict模式加载，源码快照见本报告末尾。兼容层运行成功不等于已经完成与另一FlashAttention环境的逐位数值一致性验收。

## 3. 输出是什么

| 输出（每组） | 数值/作用 | 第08组示例 |
|---|---|---|
| 视差NPY | `[720,1280]`、float32、单位像素；网络直接输出 | [disparity]({files['disparity']}) |
| 逆视差NPY | 正有限位置为 `1/d`，其余NaN；单位1/像素，非米制深度 | [inverse disparity]({files['inverse_disparity']}) |
| 正有限掩膜 | `isfinite(d) and d>0` | [positive mask]({files['positive_mask']}) |
| 水平重叠掩膜 | 再要求 `0 <= u-d <= 1279`，右侧对应横坐标仍在图内 | [overlap mask]({files['overlap_mask']}) |
| 灰度输入与伪彩预览 | 追溯实际输入与观察视差；精确数值以NPY为准 | [左输入]({files['left_input']}) · [右输入]({files['right_input']}) · [视差伪彩]({files['disparity_colour']}) |
| 设备原始输出 | 保留Raw与已有Display，未转换单位或当作真值 | [Raw]({files['device_raw']}) · [Display]({files['device_display']}) |

视差约定为 `d(u,v)=u_left-u_right`，对应右图横坐标 `u_right=u_left-d`。两个掩膜是数值与横向坐标筛选，不能当作网络置信度、遮挡完全剔除或正确匹配比例。流程没有执行逐像素左右一致性验收。

仅在相应校正内参、基线及单位明确时，才可用 `Z=fx×baseline/[d-(cx_left-cx_right)]` 进一步求米制光轴深度。本页是未校正对照；本包已另用恢复的1280×720参数执行校正，见 [主报告](REPORT.md)，米制换算仍受T单位和参数适用关系限制。未套用之前1280×800/约95mm基线的标定。设备Raw的非零比例与取值范围只描述保存的数据，不证明其单位或真实性，也未将它与模型像素视差直接相减计算误差。

## 4. 全部15组结果

网络调用累计 **{sum(times):.3f}秒**，单组中位 **{float(np.median(times)):.3f}秒**。这是本批记录的网络调用耗时，不含权重加载、文件读写与报告打包，不是独立测速或车端性能。

| 编号 | 网络秒 | 正有限比例 | 水平重叠比例 | 视差P2/P50/P98（px） | 设备Raw非零比例 | 设备Raw取值范围 | 结果 |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(table)}

完整机器可读表格：[frames.csv](frames.csv)。分位数统计正有限视差。各图的显示色标可能单独缩放，不能通过不同图的颜色直接比较数值。

![全部15组总览](overview_all15.jpg)

上图按编号1～15排列，每格沿用该组带标注的左右灰度、模型视差和设备显示图对照。设备图的显示颜色与模型视差颜色不是同一数值标尺。

第08组放大查看：

![第08组代表预览]({files['preview']})

## 5. 已做核验与结论边界

- 45张源PNG全部可解码，来源ZIP与逐文件SHA-256一致，15个编号完整覆盖且没有重复使用来源图。
- 15组左右灰度输入与原始拼接图切分后灰度转换逐像素一致；视差与逆视差都是720×1280 float32。
- 全像素检查逆视差关系、NaN规则、正有限掩膜和水平重叠掩膜；报告统计与保存数组一致。
- 报告的包内相对链接已检查；打包时逐文件流式计算SHA-256，完成ZIP解压/CRC与哈希读回核验。

证据见 [verification.json](verification.json)；包内 [SHA256.json](SHA256.json)记录除自身以外的文件哈希。压缩包旁的`.archive_verification.json`记录归档检查，`.zip.sha256`记录整个ZIP哈希。这些检查证明交付完整性和实现一致性，不能替代物理测量精度验收。

本页完成全部15组未校正输入的诊断运行和报告。本包已另用恢复参数执行校正，见 [主报告](REPORT.md)；米制结果仍受T单位和参数适用关系限制。后续还需确认物理左右顺序和设备Raw定义；测尺寸需要具体物体与测量边的独立实测真值。

## 6. 来源与复现

- 来源ZIP SHA-256：`{report['source_zip_sha256']}`
- 模型权重：`{report['checkpoint']}`
- 权重SHA-256：`{report['checkpoint_sha256']}`
- 本次脚本快照：{provenance_links}
- 推理原始记录：[report.json](report.json)；源数据清单：[source_manifest.json](source_manifest.json)。

```json
{environment}
```

本包包含来源图、数值结果、可视化、运行记录与脚本快照；未包含外部模型权重和完整Python/GPU依赖环境。JSON和源码中的本机绝对路径为追溯记录，移机查看报告不依赖这些路径。

要在已有环境重新生成报告与包，先将已完成的推理结果放入 `E:\\gpt\\_release` 下的新目录，再运行：

```powershell
python tools/build_raw1_stereo_report.py --input "E:\\gpt\\_release\\<新的结果目录>" --runtime-source "<实际兼容运行入口.py>"
```

该入口会核验所有15组文件、生成文档并打包，拒绝覆盖已生成报告或同名归档。重新推理请运行随包的 `run_stereo_zip_batch.py --help` 查看参数，另指定新的E盘输出目录。
"""
    with (base / filename).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    branch_note = "主流程位于`rectified/`：按提供的参数校正后运行模型；根目录`pair_01/`至`pair_15/`为未校正输入诊断对照。两分支共30次推理，视差均为各自左图网格上的像素单位。\n\n" if filename == "REPORT_UNRECTIFIED.md" else ""
    start = """# 从这里开始

请先完整解压ZIP，再用支持Markdown的阅读器打开 [REPORT.md](REPORT.md)。流程图需要Mermaid支持，正文表格也说明了全部步骤。

本包为15组1280×720双目推理结果：模型输入、处理过程、输出、全组预览与数值文件均有相对链接。`source/`保留45张原始PNG，`provenance/`保存运行脚本来源快照。

BRANCH_NOTE

模型视差单位是像素；逆视差单位是1/像素。设备Raw的编码和单位尚未确认。本包没有米制深度、XYZ、点云或实物长宽高测量结果。不要把逆视差NPY当米，也不要默认设备Raw单位为毫米。

- [15组结果表](frames.csv)
- [全部15组总览](overview_all15.jpg)
- [运行记录](report.json)
- [文件和数组核验](verification.json)
- [来源PNG清单](source_manifest.json)
- [包内SHA-256清单](SHA256.json)

保持解压后的目录结构即可阅读。权重和完整运行环境不在包内；记录中的本机绝对路径用于追溯，不是查看报告的依赖。归档外`.zip.sha256`与`.archive_verification.json`分别记录整个ZIP的哈希及归档读回检查结果。
"""
    start = start.replace("BRANCH_NOTE\n", branch_note)
    if filename == "REPORT_UNRECTIFIED.md":
        start = start.replace("[15组结果表](frames.csv)", "[校正后15组结果表](rectified/frames.csv)")
        start = start.replace("[全部15组总览](overview_all15.jpg)", "[校正后全部15组总览](rectified/overview_all15.jpg)")
        start = start.replace("[运行记录](report.json)", "[校正后运行记录](rectified/report.json)")
    with (base / "START_HERE.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(start)


def write_rectified_report(base, baseline, rectified, rows, csv_rows, provenance):
    """Make the calibrated remapping branch the main report without hiding its limits."""
    branch = base / "rectified"
    with (branch / "frames.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    def link(row, key):
        return local_file(branch, row["files"][key], base).relative_to(base).as_posix()
    table = []
    for row in rows:
        p2, p50, p98 = row["disparity_percentiles"]
        table.append(f"| {row['number']:02d} | {row['inference_seconds']:.3f} | {row['positive_finite_ratio']:.2%} | {row['horizontal_overlap_ratio']:.2%} | {p2:.2f} / {p50:.2f} / {p98:.2f} | [预览]({link(row, 'preview')}) · [视差]({link(row, 'disparity')}) |")
    before_by_number = {r["number"]: r for r in baseline["results"]}
    comparison = []
    for row in rows:
        before = before_by_number[row["number"]]
        comparison.append(f"| {row['number']:02d} | {before['disparity_percentiles'][1]:.2f} | {row['disparity_percentiles'][1]:.2f} | {before['horizontal_overlap_ratio']:.2%} | {row['horizontal_overlap_ratio']:.2%} | [未校正]({before['files']['preview']}) · [校正后]({link(row, 'preview')}) |")
    times = [r["inference_seconds"] for r in rows]
    before_times = [r["inference_seconds"] for r in baseline["results"]]
    provenance_links = "、".join(f"[{Path(p['file']).name}]({p['file']})" for p in provenance)
    optional_links = []
    for name, description in (
        ("calibration.json", "恢复的相机参数"),
        ("calibration_recovered.txt", "从此前材料恢复的参数原文"),
        ("rectified/calibration.json", "本次采用的恢复相机参数"),
        ("rectified/calibration_recovered.txt", "从此前材料恢复的参数原文"),
        ("rectification_diagnostic.json", "校正前后稀疏匹配诊断"),
        ("rectified/rectification_diagnostic.json", "校正前后稀疏匹配诊断"),
        ("pair_diagnostic.json", "原始左右配对诊断"),
        ("rectified/left_source_support_mask.png", "校正左图来源支持掩膜"),
        ("rectified/right_source_support_mask.png", "校正右图来源支持掩膜"),
    ):
        if (base / name).is_file():
            optional_links.append(f"- [{description}]({name})")
    diagnostic_warning = "稀疏匹配垂直残差仅作诊断参考；执行校正映射不等于校正效果已验收。"
    calibration_caution = ""
    diagnostic_path = branch / "rectification_diagnostic.json"
    if diagnostic_path.is_file():
        diagnostic = read_json(diagnostic_path)
        complete = [r for r in diagnostic["rows"] if r.get("status") == "computed_fixed_original_matches"]
        if complete:
            larger = [r["number"] for r in complete if r["rectified_vertical_abs_pixels"]["p50"] > r["original_vertical_abs_pixels"]["p50"]]
            before = float(np.median([r["original_vertical_abs_pixels"]["p50"] for r in complete]))
            after = float(np.median([r["rectified_vertical_abs_pixels"]["p50"] for r in complete]))
            require(larger == diagnostic["summary"]["pairs_with_larger_rectified_median_abs_vertical_error"], "Diagnostic summary mismatch")
            pooled = {}
            for stage, prefix in (("original", "original"), ("rectified", "rectified")):
                dy_arrays = []
                for r in complete:
                    left = np.asarray(r[f"{prefix}_points_left"], dtype=np.float64)
                    right = np.asarray(r[f"{prefix}_points_right"], dtype=np.float64)
                    require(left.shape == right.shape == (r["fixed_match_count"], 2), "Diagnostic point shape mismatch")
                    dy = np.abs(left[:, 1] - right[:, 1])
                    require(np.isfinite(dy).all(), "Nonfinite diagnostic points")
                    near(float(np.median(dy)), r[f"{stage}_vertical_abs_pixels"]["p50"], f"Diagnostic {stage} pair {r['number']} P50", tolerance=1e-5)
                    dy_arrays.append(dy)
                pooled[stage] = np.percentile(np.concatenate(dy_arrays), [50, 90]).tolist()
            diagnostic_warning += f" 本次{len(complete)}组获得固定的稀疏对应点，共{sum(r['fixed_match_count'] for r in complete)}对。合并全部匹配点后，绝对垂直残差P50/P90从 **{pooled['original'][0]:.3f}/{pooled['original'][1]:.3f}px** 变为 **{pooled['rectified'][0]:.3f}/{pooled['rectified'][1]:.3f}px**；该统计从保存的对应点坐标重新计算。各组中位残差再取中位数从{before:.3f}px变为{after:.3f}px，其中{len(larger)}/{len(complete)}组的中位残差增大。"
            if larger:
                if pooled["rectified"][0] >= pooled["original"][0] and pooled["rectified"][1] >= pooled["original"][1]:
                    diagnostic_warning += " **给定参数校正后，整体垂直误差没有改善，参数适用关系待核对。**"
                    calibration_caution = f"**当前关键发现：给定参数校正后，整体垂直误差没有改善，参数适用关系待核对。** 全部{sum(r['fixed_match_count'] for r in complete)}个固定稀疏匹配的绝对垂直残差P50/P90，由{pooled['original'][0]:.3f}/{pooled['original'][1]:.3f}px变为{pooled['rectified'][0]:.3f}/{pooled['rectified'][1]:.3f}px；{len(larger)}/15组的中位残差增大。本包保留结果供诊断，不能把执行校正等同于几何效果已通过验收。"
                else:
                    diagnostic_warning += " **提供参数应用后，部分对应点的垂直对齐并未改善，标定适用性尚不能确认。**"
                diagnostic_warning += " 需进一步核对参数对应模式、图像裁剪/缩放与左右约定，当前校正后视差仍应作为诊断结果。"
            diagnostic_table = ["| 编号 | 固定匹配点对 | 未校正绝对垂直残差P50（px） | 校正后P50（px） |", "|---|---:|---:|---:|"]
            diagnostic_table += [f"| {r['number']:02d} | {r['fixed_match_count']} | {r['original_vertical_abs_pixels']['p50']:.3f} | {r['rectified_vertical_abs_pixels']['p50']:.3f} |" for r in complete]
            diagnostic_warning += "\n\n" + "\n".join(diagnostic_table)
            diagnostic_warning += "\n\n统计使用相同的原图SIFT互检匹配及F矩阵RANSAC内点，再把这些点按校正参数变换；没有在校正后重新挑点。这是稀疏特征诊断，不是深度或尺寸真值。"
    sample = rows[7]
    text = f"""# _Raw1：15组图像全量双目推理报告

本批次15组照片已全部运行。根据用户确认的“原图没有双目校正”，本包按之前提供的1280×720参数执行校正映射，再运行 FoundationStereo；同时保留未校正输入的15组诊断对照，合计 **30次双目推理**。两分支都使用1280×720输入、32次模型内部迭代。

**交付的是像素视差及诊断结果。** 参数平移 `T=(-32.4841,0,0)` 的单位尚未确认，设备Raw编码/单位也未确认，因此没有米制深度、XYZ、PLY或货物尺寸结果。校正操作和文件核验完成不代表几何精度验收完成。

{calibration_caution}

## 1. 输入是什么

源压缩包：`{baseline['source_zip']}`。全部45张PNG按编号1～15配对，每组包含：

| 文件 | 保存格式 | 本次用途 |
|---|---|---|
| `Image_ColorN.png` | 2560×720、uint8、三通道拼接图 | 左右切分，各1280×720，分别转灰度作为双目输入 |
| `Image_Depth_RawN.png` | 1280×720、uint16 | 原样保留设备数值，编码和单位待确认 |
| `Image_Depth_DispalyN.png` | 1280×720、uint8、三通道 | 原样保留设备显示图，仅供外观对照 |

用户说明相机都是同一个。本次采用此前提供的1280×720左右内参、畸变、旋转和平移参数；没有套用之前1280×800、约95mm基线的参数。参数来源恢复记录随包保留，具体适用关系仍需结合采集模式和诊断核对。

暂按拼接左半幅为物理左目、右半幅为右目处理。两半幅分别经 OpenCV `BGR2GRAY` 转灰度，再执行相应的双目校正重映射。原始图未校正是用户已确认的信息；左右物理顺序仍未获得独立验证。

输入灰度复制为三个通道后送网络。每幅1280×720图上下各补8像素，网络输入为 `[1,3,736,1280]`；网络输出去掉补边，恢复720×1280数组。没有外部缩小再放大。本次校正分支有几何重采样，不能称为原始像素不变。

## 2. 中间处理过程是什么

```mermaid
flowchart TD
    A[15张2560×720拼接图] --> B[切分左右半幅、分别转灰度]
    K[1280×720 K/D/R/T参数] --> C[stereoRectify与校正映射]
    B --> C
    C --> D[remap为1280×720校正左右图]
    D --> E[灰度复制三通道、补边、标准化]
    E --> F[共享局部和预训练视觉特征]
    F --> G[水平代价体、3D卷积和注意力聚合]
    G --> H[初始化视差、GRU细化32次]
    H --> I[学习上采样、去补边]
    I --> J[校正左图网格上的像素视差]
    J --> L[逆视差、数值掩膜、预览]
    B --> M[未校正输入对照分支，同一模型参数]
```

| 阶段 | 本次做法 |
|---|---|
| 几何校正 | `cv2.stereoRectify`，`CALIB_ZERO_DISPARITY`、`alpha=1`、`newImageSize=(1280,720)`；保存R1/R2/P1/P2/Q及ROI。 |
| 图像重映射 | 四张float32映射，`cv2.remap`使用双线性插值、越界常数0；原图范围外可能出现黑色边缘。 |
| 标准化 | 像素除255，按每通道mean `[0.485,0.456,0.406]`、std `[0.229,0.224,0.225]` 标准化。 |
| 提取特征 | EdgeNeXt局部特征与基于DINOv2的DepthAnything视觉特征融合；辅助单目结果不是最终深度。 |
| 建立匹配 | 在主要1/4网格上构造水平分组相关、特征拼接代价体，经3D hourglass与沿视差方向的注意力聚合。 |
| 初始化和细化 | 候选概率加权初始化视差，三尺度上下文及相关特征送入GRU，32次内部迭代更新；没有训练。 |
| 输出恢复 | 学习的邻域权重上采样，再裁掉上下补边，输出float32视差。 |

校正配置见 [rectification.json](rectified/rectification.json)，完整映射见 [rectification.npz](rectified/rectification.npz)。`Q`继承标定平移的原单位，不能将其重投影数值自动命名为“米”。本次未使用Q导出三维数据。

{diagnostic_warning}

{chr(10).join(optional_links)}

模型权重strict加载；注意力后端为 `{rectified['attention_backend']}`。Windows兼容层的运行记录不等于与另一FlashAttention环境的逐位数值一致性验收。来源代码快照：{provenance_links}。

## 3. 输出是什么

| 输出 | 格式与意义 | 第08组文件 |
|---|---|---|
| 模型视差 | 720×1280 float32，单位px，`u_right=u_left-d` | [视差NPY]({link(sample, 'disparity')}) |
| 逆视差 | 正有限位置为`1/d`，其余NaN，单位1/px，非米制深度 | [逆视差NPY]({link(sample, 'inverse_disparity')}) |
| 正有限掩膜 | `isfinite(d) and d>0` | [掩膜]({link(sample, 'positive_mask')}) |
| 水平重叠掩膜 | 再要求`0<=u-d<=1279` | [掩膜]({link(sample, 'overlap_mask')}) |
| 校正后灰度输入 | 实际送入模型、补边前的左右图 | [左图]({link(sample, 'left_input')}) · [右图]({link(sample, 'right_input')}) |
| 可视化 | 输入、模型视差、设备原Display对照 | [预览]({link(sample, 'preview')}) |

**校正分支的数组坐标属于校正后左图网格。** 虽然尺寸仍为1280×720，同一行列号不能当作原始左半幅的同一像素。设备Raw/Display保留原图网格，未与模型校正网格完成配准，预览中并列展示不构成逐像素精度对比。

正有限比例及水平重叠比例描述数值与横坐标范围，不是置信度、匹配正确率或测量精度，也不自动剔除所有黑边与遮挡。来源支持掩膜可辅助识别重映射越界区。精确视差读取NPY，图片颜色仅供观察；每图按各自分位数显示，不同图的同色不保证同一视差。

只有确认适用的校正内参、基线单位及对应关系后，才能按 `Z=fx×baseline/[d-(cx_left-cx_right)]` 转成有物理单位的光轴深度。本包未做这一步，也没有将设备Raw直接当作像素视差或毫米真值比较。

## 4. 校正后全部15组结果

校正后网络调用累计 **{sum(times):.3f}秒**，单组中位 **{float(np.median(times)):.3f}秒**；未校正对照累计 **{sum(before_times):.3f}秒**。耗时来自本次网络调用，不含模型加载、图片/磁盘读写、校正准备及打包，非独立性能评测或车端性能。

| 编号 | 网络秒 | 正有限比例 | 水平重叠比例 | 视差P2/P50/P98（px） | 文件 |
|---|---:|---:|---:|---:|---|
{chr(10).join(table)}

机器可读数据：[校正后frames.csv](rectified/frames.csv)、[校正后运行记录](rectified/report.json)。分位数仅统计正有限视差。

![校正后15组总览](rectified/overview_all15.jpg)

第08组代表预览：

![校正后第08组]({link(sample, 'preview')})

## 5. 未校正输入对照

用户确认原图未校正，模型的同一行匹配前提不满足。因此该分支作为流程与图像诊断对照保留，不作为正式几何结论。详细输入/过程/输出、设备Raw统计与全部原图预览见 [未校正对照报告](REPORT_UNRECTIFIED.md)、[对照CSV](frames.csv)及[对照运行记录](report.json)。

两分支的像素坐标不同，下表仅并列描述整体统计；P50或掩膜比例增减不代表精度改善或退化，不作逐像素差分。

| 编号 | 未校正P50（px） | 校正后P50（px） | 未校正水平重叠 | 校正后水平重叠 | 图像 |
|---|---:|---:|---:|---:|---|
{chr(10).join(comparison)}

## 6. 文件核验与当前边界

- 15个编号完整覆盖45张来源PNG；原始ZIP、源文件SHA-256、shape/dtype及解码检查通过。
- 未校正输入与原图切分后灰度逐像素一致；校正输入与保存映射按相同插值/边界规则重算的图像逐像素一致。
- 两分支30组720×1280 float32视差/逆视差、NaN规则、正有限和水平重叠掩膜及报告统计均已核验。
- 校正映射、参数、运行脚本和输入图随包保留；包内文档相对链接及归档全部文件的CRC、SHA-256读回检查通过。

核验证据：[verification.json](verification.json)；内容清单：[SHA256.json](SHA256.json)；来源图清单：[source_manifest.json](source_manifest.json)。压缩包旁`.zip.sha256`及`.archive_verification.json`记录整个ZIP的哈希和归档验收。

这些检查证明文件完整性和程序按记录执行，不能替代标定适用性、校正质量或实物测量精度验收。下一步需确认平移单位、物理左右顺序和设备Raw编码，核对校正诊断；明确测量对象、边与独立尺量真值后，再生成米制几何和尺寸误差。

## 7. 来源与复现

- 来源ZIP SHA-256：`{baseline['source_zip_sha256']}`
- 权重：`{rectified['checkpoint']}`
- 权重SHA-256：`{rectified['checkpoint_sha256']}`
- 两分支实际环境与参数分别记录于根目录及`rectified/`的`report.json`。

包内包含45张源PNG、两分支结果、校正参数与映射、文档和脚本快照；未包含外部模型权重和完整Python/GPU环境。记录中的本机绝对路径用于追溯，不是移机阅读报告的依赖。

已有环境中先运行随包的`run_stereo_zip_batch.py --help`和`prepare_raw1_rectified.py --help`查看相应入口。给每次实验使用 `E:\\gpt\\_release` 下的新目录，完成两分支后执行：

```powershell
python tools/build_raw1_stereo_report.py --input "E:\\gpt\\_release\\<新的结果目录>"
```

报告入口会核验所有结果并封包，拒绝覆盖已有报告和同名ZIP。
"""
    with (base / "REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def check_links(base):
    links = []
    for name in ("REPORT.md", "START_HERE.md", "REPORT_UNRECTIFIED.md"):
        if not (base / name).exists():
            continue
        text = (base / name).read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            target = unquote(target.strip("<>"))
            if target.startswith(("http:", "https:", "#")):
                continue
            # The self-referential hash manifest is created immediately after verification.
            if target not in ("verification.json", "SHA256.json"):
                local_file(base, target.split("#")[0])
            links.append({"document": name, "target": target})
    return links


def archive_release(base):
    archive = base.with_name(base.name + ".zip")
    partial = base.with_name(base.name + ".partial.zip")
    sidecar = archive.with_suffix(".zip.sha256")
    receipt = archive.with_suffix(".archive_verification.json")
    for path in (archive, partial, sidecar, receipt):
        require(not path.exists(), f"Existing release preserved: {path}")
    files = sorted(p for p in base.rglob("*") if p.is_file())
    require(all(not p.is_symlink() and base in p.resolve().parents for p in files), "Symlink or escaped content")
    require(not (base / "SHA256.json").exists(), "Existing SHA256 manifest preserved")
    source_bytes = sum(p.stat().st_size for p in files)
    require(shutil.disk_usage(base.parent).free >= source_bytes * 1.1, "Insufficient E-drive free space for conservative ZIP estimate")
    manifest = {p.relative_to(base).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files}
    write_json(base / "SHA256.json", manifest)
    files.append(base / "SHA256.json")
    prefix = base.name + "/"
    print(f"Archiving {len(files)} files ({source_bytes:,} content bytes)", flush=True)
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as output:
        for index, path in enumerate(files, 1):
            rel = path.relative_to(base).as_posix()
            info = zipfile.ZipInfo.from_file(path, prefix + rel)
            info.compress_type = zipfile.ZIP_STORED if path.suffix.lower() in (".png", ".jpg", ".jpeg") else zipfile.ZIP_DEFLATED
            info._compresslevel = 3
            before = path.stat()
            h, count = hashlib.sha256(), 0
            with path.open("rb") as inp, output.open(info, "w", force_zip64=True) as dest:
                for block in iter(lambda: inp.read(CHUNK), b""):
                    dest.write(block)
                    h.update(block)
                    count += len(block)
            after = path.stat()
            require(before.st_size == after.st_size == count and before.st_mtime_ns == after.st_mtime_ns, f"File changed during packaging: {rel}")
            if rel in manifest:
                require(h.hexdigest() == manifest[rel]["sha256"], f"File changed after manifest: {rel}")
            if index % 50 == 0 or index == len(files):
                print(f"Packed {index}/{len(files)}", flush=True)
    with zipfile.ZipFile(partial) as package:
        names = package.namelist()
        require(len(names) == len(set(names)) == len(manifest) + 1, "Unexpected ZIP count or duplicate entries")
        require(set(names) == {prefix + name for name in manifest} | {prefix + "SHA256.json"}, "Unexpected ZIP members")
        require(json.loads(package.read(prefix + "SHA256.json")) == manifest, "Archived manifest mismatch")
        for index, (rel, expected) in enumerate(manifest.items(), 1):
            h, count = hashlib.sha256(), 0
            # Reading each member to EOF also validates its ZIP CRC.
            with package.open(prefix + rel) as stream:
                for block in iter(lambda: stream.read(CHUNK), b""):
                    h.update(block)
                    count += len(block)
            require(h.hexdigest() == expected["sha256"] and count == expected["bytes"], f"Archive readback mismatch: {rel}")
            if index % 50 == 0 or index == len(manifest):
                print(f"Verified {index}/{len(manifest)}", flush=True)
    archive_hash = digest(partial)
    partial.rename(archive)
    with sidecar.open("x", encoding="ascii") as stream:
        stream.write(f"{archive_hash}  {archive.name}\n")
    evidence = {"status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
                "archive": str(archive), "archive_bytes": archive.stat().st_size, "archive_sha256": archive_hash,
                "source_directory": str(base), "hashed_content_files": len(manifest), "zip_entries": len(manifest) + 1,
                "all_entries_crc_and_sha256_verified": True, "duplicate_entries": False,
                "pairs": 15, "inference_branches": 2 if (base / "rectified" / "report.json").is_file() else 1,
                "network_calls": 30 if (base / "rectified" / "report.json").is_file() else 15,
                "source_png_count": 45, "metric_depth": False,
                "report": prefix + "REPORT.md", "start_here": prefix + "START_HERE.md"}
    write_json(receipt, evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", required=True, type=Path, help=r"Completed run directory under E:\gpt\_release")
    parser.add_argument("--runtime-source", type=Path, default=ROOT / "tools" / "foundationstereo_runtime.py",
                        help="Compatibility runtime source to snapshot (default: foundationstereo_runtime.py)")
    args = parser.parse_args()
    base = args.input.resolve()
    require(base.is_dir() and RELEASE_ROOT.resolve() in base.parents, "Input/staging must be under E:\\gpt\\_release")
    for name in ("REPORT.md", "REPORT_UNRECTIFIED.md", "START_HERE.md", "frames.csv", "overview_all15.jpg", "verification.json", "SHA256.json",
                 "rectified/frames.csv", "rectified/overview_all15.jpg"):
        require(not (base / name).exists(), f"Existing delivery preserved: {name}")
    for suffix in (".zip", ".partial.zip", ".zip.sha256", ".archive_verification.json"):
        require(not base.with_name(base.name + suffix).exists(), f"Existing archive preserved: {suffix}")
    report = read_json(base / "report.json")
    print("Verifying source files and all 15 result arrays", flush=True)
    rows, csv_rows, checks = verify_run(base, report)
    rectified_report = None
    rectified_checks = []
    if (base / "rectified" / "report.json").is_file():
        print("Verifying all 15 rectified result arrays and remapped input pixels", flush=True)
        rectified_report = read_json(base / "rectified" / "report.json")
        require(rectified_report["checkpoint_sha256"] == report["checkpoint_sha256"], "Branches used different model weights")
        rectified_rows, rectified_csv_rows, rectified_checks = verify_run(base / "rectified", rectified_report, base)
    provenance = copy_provenance(base, args.runtime_source)
    make_overview(base, rows)
    write_documents(base, report, rows, csv_rows, provenance, filename="REPORT_UNRECTIFIED.md" if rectified_report else "REPORT.md")
    if rectified_report:
        make_overview(base / "rectified", rectified_rows)
        write_rectified_report(base, report, rectified_report, rectified_rows, rectified_csv_rows, provenance)
    links = check_links(base)
    verification = {"status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
                    "pairs": len(rows), "source_png_count": 45, "source_zip_sha256_verified": True,
                    "source_png_sha256_verified": True, "source_coverage_exact": True,
                    "metric_depth": False, "physical_accuracy_validated": False,
                    "pair_checks": checks, "rectified_pair_checks": rectified_checks,
                    "inference_branches": 2 if rectified_report else 1,
                    "network_calls": len(checks) + len(rectified_checks), "provenance": provenance,
                    "relative_link_count": len(links), "relative_links": links,
                    "hash_manifest_created_during_archiving": True}
    write_json(base / "verification.json", verification)
    evidence = archive_release(base)
    require(all(local_file(base, item["target"]).is_file() for item in links), "Final relative-link check failed")
    print(json.dumps(evidence, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
