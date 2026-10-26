"""Build an evidence-linked Chinese Markdown report for the native stereo batch."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def image(path):
    result = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if result is None:
        raise ValueError(f"Cannot decode {path}")
    return result


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    report_path = out / "REPORT.md"
    if report_path.exists():
        raise ValueError("Refusing to overwrite the report")
    source = read_json(out / "orbbec_native47/report.json")
    provenance = read_json(out / "provenance.json")
    geometry_verification = read_json(out / "geometry/verification.json")
    assert geometry_verification["status"] == "passed" and geometry_verification["verified_pairs"] == 47
    assert geometry_verification["metric_pairs"] == 46
    assert source["scale"] == 1 and source["iterations"] == 32
    rows = source["results"]
    assert len(rows) == 47 and {r["number"] for r in rows} == set(range(1, 48))
    input_root = Path(rows[0]["file"]).parent.parent
    assert {Path(r["file"]).resolve() for r in rows} == set((input_root / "IR_LEFT").glob("*.png"))
    assert {Path(r["right_file"]).resolve() for r in rows} == set((input_root / "IR_RIGHT").glob("*.png"))
    assert len({r["right_file"] for r in rows}) == 47
    manifest = []
    table = []
    fractions = []
    metric_count = 0
    for row in rows:
        folder = f"left_{row['number']:02d}_doe{row['doe']}"
        src = out / "orbbec_native47" / folder
        geom = out / "geometry" / folder
        left, right = image(Path(row["file"])), image(Path(row["right_file"]))
        assert left.shape == right.shape == tuple(reversed(row["size"]))
        assert row["size"] == row["input_size"]
        assert np.array_equal(left, image(src / "left_input.png"))
        assert np.array_equal(right, image(src / "right_input.png"))
        disparity = np.load(src / "disparity_pixels.npy", allow_pickle=False)
        inverse = np.load(src / "inverse_disparity_nonmetric.npy", allow_pickle=False)
        assert disparity.dtype == np.float32 and disparity.shape == left.shape
        valid = np.isfinite(disparity) & (disparity > 0)
        overlap = valid & (np.arange(left.shape[1])[None] - disparity >= 0) & (np.arange(left.shape[1])[None] - disparity <= left.shape[1] - 1)
        assert np.array_equal(image(src / "positive_finite_mask.png") > 0, valid)
        assert np.array_equal(image(src / "horizontal_overlap_mask.png") > 0, overlap)
        assert np.allclose(disparity[valid] * inverse[valid], 1, atol=1e-6)
        assert np.isnan(inverse[~valid]).all()
        fraction = float(overlap.mean())
        fractions.append(fraction)
        metric_path = geom / "depth_m_provisional.npy"
        if row["number"] == 24:
            assert not metric_path.exists()
            z = "未换算"
        else:
            depth = np.load(metric_path, allow_pickle=False)
            assert depth.shape == left.shape
            metric_count += 1
            z = f"{np.nanmedian(depth):.3f}"
        note = "时间戳差1，暂定配对" if row["timestamp_delta_raw"] else ("缺该分辨率标定" if row["number"] == 24 else "精确时间戳配对")
        table.append(f"| {row['number']:02d} | {row['doe']} | {left.shape[1]}×{left.shape[0]} | {row['inference_seconds']:.3f} | {fraction:.2%} | {z} | [{note}](geometry/{folder}/preview.jpg) |")
        manifest.append(dict(number=row["number"], doe=row["doe"], width=left.shape[1], height=left.shape[0], left_file=row["file"], right_file=row["right_file"], left_sha256=digest(Path(row["file"])), right_sha256=digest(Path(row["right_file"])), timestamp_delta_raw=row["timestamp_delta_raw"], disparity_file=f"orbbec_native47/{folder}/disparity_pixels.npy", preview_file=f"geometry/{folder}/preview.jpg"))
    assert metric_count == 46
    with (out / "input_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    times = np.array([r["inference_seconds"] for r in rows])
    regular_times = np.array([r["inference_seconds"] for r in rows if r["number"] not in (1, 24)])
    calibration = read_json(out / "geometry/calibration.json")
    k = calibration["streams"]["ir_left"]["intrinsic"]
    baseline = -calibration["extrinsics"]["ir_left_to_ir_right"]["translation"][0] / 1000
    v, u = 400, 640
    sample_folder = out / "geometry/left_33_doe0"
    sample_disp = np.load(out / "orbbec_native47/left_33_doe0/disparity_pixels.npy")[v, u]
    sample_depth = np.load(sample_folder / "depth_m_provisional.npy")[v, u]
    sample_xyz = np.load(sample_folder / "xyz_m_provisional.npy")[v, u]
    source_dir = out / "provenance"
    source_dir.mkdir(exist_ok=True)
    for name in ["run_stereo_disparity_trial.py", "export_native_stereo_batch.py", "build_stereo_batch_markdown_report.py"]:
        shutil.copy2(ROOT / "tools" / name, source_dir / name)
    shutil.copy2(Path(source["checkpoint"]).with_name("cfg.yaml"), source_dir / "cfg.yaml")
    notes = ROOT / "docs/foundationstereo-process-notes-20260928.md"
    if notes.is_file():
        shutil.copy2(notes, source_dir / "model_process_source_audit.md")
    log_path = out.parent / f"{out.name}_inference.log"
    assert "DONE" in log_path.read_text(encoding="utf-8")
    shutil.copy2(log_path, out / "inference.log")
    checkpoint_hash = provenance["files"][0]["sha256"]
    md = f"""# FoundationStereo 全量图片推理报告：输入、模型处理过程与输出

日期：2026-09-28　｜　批次：`{out.name}`　｜　模型：标准 FoundationStereo，`23-51-11/model_best_bp2.pth`

## 1. 本次实际完成内容

对同一相机历史采集目录 `{input_root}` 的 **47 对左右 IR 图片（94 张）全部重新推理，47/47 成功**。本次使用原始分辨率输入，46 组为 1280×800，第 24 组为 848×480，未使用旧视差替代推理，也未采用半分辨率结果放大。

已生成 **47 组视差及预览、46 组条件性米制深度、46 组逐像素 XYZ 和 46 份全场景左 IR 灰度点云**。第 24 组没有匹配的 848×480 标定，只输出视差和几何掩膜；这组已完成网络推理，未被漏跑。

模型直接回答的是“左图每个像素在右图中横向移动了多少像素”，即**视差**。相机内参与双目基线用于网络之后的几何换算，得到深度和三维坐标。货物长宽高还需要确定测量对象、边界或平面，不是本次网络直接输出的字段。

## 2. 输入是什么

| 输入项 | 本次实际使用情况 | 作用 |
|---|---|---|
| 左右 IR 图 | 同一次采集的灰度 PNG，每组各一张 | 网络的直接图像输入 |
| DOE 分组 | DOE=6：1～23，共23组；DOE=0：24～47，共24组 | 结果分组元数据，不输入神经网络 |
| 模型权重 | 标准 `23-51-11/model_best_bp2.pth`，strict 加载成功 | 固定网络参数，本次不训练 |
| 左右内参和基线 | [本次采用的标定副本](geometry/calibration.json) | 视差之后的米制换算 |
| Color、设备 Depth PNG | 同目录分别47、46张，未送入该模型 | 属于辅助采集流，不是另一套双目输入 |

47张左图和47张右图均恰好使用一次，文件清单、设备时间戳差和 SHA-256 见 [input_manifest.csv](input_manifest.csv)。46对设备时间戳完全一致；第7对右图时间戳大1，原时间单位及容差未验证，保留暂定配对标记。

**输入数据形式：** 原图 `uint8[H,W]`，灰度值0～255。每幅图复制为三个相同通道，转换成 `float32[1,3,H,W]` 后送入 GPU。复制通道只为适配网络接口，不增加颜色信息。

标准1280×800图无需补边；848×480图左右各复制边缘8列，网络处理864×480，最终裁回848×480。全部图像保持原方向、原完整视场和左图像素网格。

## 3. 中间模型如何处理

```mermaid
flowchart TD
    A[左 IR 原图 + 右 IR 原图] --> B[配对检查 / 灰度复制三通道 / 必要补边]
    B --> C[按通道归一化]
    C --> D[卷积多尺度特征 + DINOv2 视觉特征融合]
    D --> E[沿水平方向建立候选视差代价体]
    E --> F[三维卷积与注意力聚合 / 初始视差]
    F --> G[多尺度 GRU 迭代修正视差 32 次]
    G --> H[学习权重上采样 / 去补边]
    H --> I[原尺寸左视图视差 d，单位像素]
    I --> J[正有限检查 + 右图坐标范围检查]
    K[适用的相机内参和双目基线] --> L[深度 Z 与逐像素 XYZ 几何换算]
    J --> L
    L --> M[深度 NPY / XYZ NPY / 灰度 PLY / 预览]
```

上图根据本机实际源码与调用路径整理；本批保存的是最终预测，没有把特征图或32次中间视差单独落盘。

| 阶段 | 实际计算 | 产生的内容 |
|---|---|---|
| 归一化 | 先除以255，再按 mean=(0.485,0.456,0.406)、std=(0.229,0.224,0.225) 处理三个通道 | 归一化左右图张量 |
| 特征提取 | 卷积编码局部纹理和边缘，DINOv2 ViT-L 分支提供视觉特征，进行多尺度融合 | 用于匹配的左右特征及左图上下文 |
| 建立代价体 | 在1/4网格沿水平候选位移比较左右特征；8组相关特征与左右各12通道拼接特征合并 | 32通道候选匹配信息，之后投影为28通道 |
| 代价聚合与初始化 | 三维卷积、多尺度结构和注意力聚合；对候选位移 softmax 后取加权期望 | 初始低分辨率视差 |
| 迭代更新 | 利用几何编码、相关信息和左图上下文，三层多尺度 GRU 反复预测视差增量 | 32次修正后的视差 |
| 输出恢复 | 先将1/4网格视差乘4恢复输入像素单位，再用学习的3×3邻域权重上采样，最后去除补边 | 与原始左图同宽高的最终视差 |

以1280×800图为例，主要匹配网格为320×200；`max_disp=416` 对应104个初始候选位移层。网络内部降低特征分辨率是模型结构的一部分，**本次外部输入和最终输出都保持1280×800**。104层是初始代价体的候选设置，不是最终视差的硬性裁剪范围。

运行设置：`batch=1`、`eval()`、`inference_mode()`、32次更新、CUDA自动混合精度FP16；结果存为float32。Windows环境以PyTorch SDPA适配原FlashAttention调用，权重strict加载成功；没有进行FlashAttention与SDPA的逐位一致性实验。

源码与配置证据：[推理入口](provenance/run_stereo_disparity_trial.py)、[配置](provenance/cfg.yaml)、[网络流程源码核对说明](provenance/model_process_source_audit.md)。

## 4. 视差怎样变成深度和三维点

视差定义为 `d = u_left - u_right`，单位是**原始输入图的像素**。已校正双目模型下：

```text
Z = fx × B / (d - (cx_left - cx_right))
X = (u - cx_left) × Z / fx
Y = (v - cy_left) × Z / fy
```

本次1280×800标定：`fx=fy={k['fx']:.9f} px`，`cx={k['cx']:g}`，`cy={k['cy']:g}`，基线 `B={baseline:.10f} m`。左右主点相同，因此 `Z={k['fx'] * baseline:.11f}/d`，单位米。

示例：第33组左图像素 `(u=640,v=400)`，本次视差 `{sample_disp:.6f} px`，换算 `Z={sample_depth:.6f} m`；`XYZ=({sample_xyz[0]:.6f}, {sample_xyz[1]:.6f}, {sample_xyz[2]:.6f}) m`。读取数组时顺序为 `depth[v,u]` 和 `xyz[v,u,:]`。

坐标原点为左IR相机光心，X向右、Y向下、Z向前；Z是光轴深度，点到光心的直线距离是 `sqrt(X²+Y²+Z²)`。无效像素在深度及XYZ中均保存为NaN。结果已是米，**不要再乘设备Depth PNG的单位比例**。

这里的米制输出带有条件：用户提供的历史标定须适用于这批左/右IR，并且原图已满足该已校正模型。标定呈现零畸变、单位旋转、纯水平平移形式，但历史IR是否校正尚未完成独立确认；本次没有额外对图像做校正重映射。因此深度与点云均标记为 `provisional`，没有宣称物理尺寸精度已验收。

## 5. 输出是什么、怎样使用

各帧以 `left_编号_doe分组` 命名，例如 `left_33_doe0`。

| 文件 | 数据形式 / 单位 | 用途 |
|---|---|---|
| `orbbec_native47/<帧>/disparity_pixels.npy` | float32，H×W，像素 | 模型直接预测，可供其他深度转换程序使用 |
| `orbbec_native47/<帧>/inverse_disparity_nonmetric.npy` | float32，H×W，1/像素 | 相对远近诊断，不是米制深度 |
| `orbbec_native47/<帧>/left_input.png`、`right_input.png` | uint8，原图尺寸 | 本次实际输入图 |
| `orbbec_native47/<帧>/positive_finite_mask.png` | 0/255 | 视差为正且有限的位置 |
| `orbbec_native47/<帧>/horizontal_overlap_mask.png` | 0/255 | 再检查 `0 ≤ u-d ≤ W-1` 的位置 |
| `geometry/<帧>/depth_m_provisional.npy` | float32，H×W，米 | 条件性光轴Z深度；第24组无此文件 |
| `geometry/<帧>/xyz_m_provisional.npy` | float32，H×W×3，米 | 与左图逐像素对应的XYZ；第24组无此文件 |
| `geometry/<帧>/*.ply` | 二进制PLY，米，左IR灰度着色 | 全场景可见表面点云；第24组无此文件 |
| `geometry/<帧>/preview.jpg` | 排版预览图 | 对照左右输入与输出，不能作为深度数组使用 |
| [geometry/frames.csv](geometry/frames.csv)、[geometry/summary.json](geometry/summary.json) | 逐帧指标和来源 | 程序读取与结果检索 |

几何掩膜只检查数值及图像范围，**不是网络置信度，也没有验证左右一致性或遮挡处匹配正确性**。视差伪彩按每帧2%～98%分位拉伸，跨帧颜色不可直接比距离；米制深度伪彩统一0.3～5米显示，超出范围只截断显示颜色，不截断数组数值。

以下为原图与输出的代表性对照；货物、白板和前景设备的距离结构可供目视检查，平滑轮廓不等同于真实边缘精度。

![第6组 DOE6 输入及输出](geometry/left_06_doe6/preview.jpg)

![第33组 DOE0 输入及输出](geometry/left_33_doe0/preview.jpg)

## 6. 全批结果与耗时

| 项目 | 本次结果 |
|---|---:|
| 推理成功 / 计划组数 | 47 / 47 |
| DOE6 / DOE0 | 23 / 24 |
| 原生1280×800 / 原生848×480 | 46 / 1 |
| 米制深度 / XYZ / PLY | 46 / 46 / 46 |
| 47次网络调用时间合计 | {times.sum():.3f} 秒 |
| 单组网络调用时间中位数 | {np.median(times):.3f} 秒 |
| 单组网络调用时间P95 | {np.percentile(times, 95):.3f} 秒 |
| 首组网络调用时间（含首次执行开销） | {times[0]:.3f} 秒 |
| 排除首组与小分辨率组后中位数 | {np.median(regular_times):.3f} 秒 |
| 几何范围掩膜占比范围 | {min(fractions):.2%}～{max(fractions):.2%} |

时间来自RTX 5070 Ti上本次单次顺序处理，计时前后进行CUDA同步；包含网络调用及必要执行等待，不包含权重加载、读图、落盘及XYZ/PLY后处理。没有独立预热、重复测速或Jetson实机测试，不能据此宣称车端实时性能。

逐组“Z中位数”统计整幅有效区域，不是货物距离、物体尺寸或误差。掩膜占比不表示预测准确率。

| 编号 | DOE | 输入/输出尺寸 | 网络秒 | 几何掩膜占比 | Z中位数 m | 结果预览与状态 |
|---:|---:|---|---:|---:|---:|---|
""" + "\n".join(table)
    md += "\n\n全47组总览：\n\n"
    overviews = sorted((out / "geometry").glob("overview*.jpg"))
    assert len(overviews) == 4
    for path in overviews:
        md += f"![{path.stem}](geometry/{path.name})\n\n"
    md += f"""## 7. 核验与结论边界

本次对47组原始输入覆盖率、保存输入与原始像素一致性、视差尺寸和逆视差关系独立核对；几何后处理对深度公式、NaN掩膜、XYZ的Z通道、像素回投、PLY读回和图像解码逐组核验。证据：[推理与报告核验](report_verification.json)、[几何输出核验](geometry/verification.json)。这些检查验证输出完整性与公式实现，不是实物测量准确率。

- 第7组左右时间戳差1，配对仍待核实；第24组缺848×480标定，未生成米制数值。
- 历史IR校正状态和标定适用关系仍是物理尺寸结论的前提；没有将设备Depth PNG当成独立真值。
- 本批没有覆盖47组的实物尺寸真值，也未对47组逐一标注货物边界或拟合货物尺寸，因此不报告全批毫米误差或合格率。此前第33组的人工辅助测量是另一项局部实验，不能推广为全批精度。
- XYZ和点云处于左IR相机坐标系，未做RGB配准，不包含遮挡背面、世界坐标或完整货物长宽高。

本次提供了每组可审查的输入、模型视差、适用时的深度和三维数据。下一步尺寸验收可在确认标定后，选定测量边或平面，并用独立尺量真值计算长度误差和重复性。

## 8. 运行记录与复现

Python `{provenance['python_version'].split('|')[0].strip()}`；PyTorch `{provenance['packages']['torch']}`；OpenCV `{provenance['packages']['opencv-python']}`；本机GPU为NVIDIA GeForce RTX 5070 Ti。

权重文件 SHA-256：`{checkpoint_hash}`。

```powershell
& 'D:\\softwares\\Anaconda3\\envs\\foundation_stereo\\python.exe' `
  tools/run_stereo_disparity_trial.py --all-pairs --scale 1 `
  --output outputs/depth/新的实验名/orbbec_native47

& 'D:\\softwares\\Anaconda3\\envs\\foundation_stereo\\python.exe' `
  tools/export_native_stereo_batch.py `
  --input outputs/depth/新的实验名/orbbec_native47 `
  --calibration outputs/depth/orbbec_cargo_pair2_metric_20260924/calibration.json `
  --output outputs/depth/新的实验名/geometry
```

复现须在项目根目录执行，并使用新目录以保留已有结果。该示例复现网络及几何数据；本报告生成器另读取环境来源记录。完整运行日志见 [inference.log](inference.log)，环境与源文件摘要见 [provenance.json](provenance.json)，几何后处理源码见 [export_native_stereo_batch.py](provenance/export_native_stereo_batch.py)。
"""
    report_path.write_text(md, encoding="utf-8")
    # Verify the Markdown's relative file links, including embedded figures.
    targets = re.findall(r"\]\(([^)]+)\)", md)
    missing = [p for p in targets if not (out / p).is_file()]
    if missing:
        # The verification file is written immediately below.
        assert missing == ["report_verification.json"], missing
    evidence = dict(status="passed", created_at=datetime.now().astimezone().isoformat(), pairs=47, metric_pairs=metric_count, left_source_coverage="exact_file_set", right_source_coverage="exact_file_set_each_used_once", native_inputs_equal_original_pixels=True, disparity_inverse_and_masks_checked=True, metric_geometry_verification="geometry/verification.json", markdown_links_checked=len(targets), missing_links=[], physical_accuracy_verified=False, network_seconds=dict(total=float(times.sum()), median=float(np.median(times)), p95=float(np.percentile(times,95))), checkpoint_sha256=checkpoint_hash)
    (out / "report_verification.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    assert all((out / p).is_file() for p in targets)
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
