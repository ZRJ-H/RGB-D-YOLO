"""Build a portable colleague-review package for the generated stereo XYZ sample."""
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NAME = "orbbec_xyz_sample25_review_20260924_v2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    frame = ROOT / "outputs/depth/orbbec_metric_native47_20260924/left_25_doe0"
    sample = ROOT / "outputs/depth/orbbec_xyz_sample25_20260924"
    bundle = Path(r"E:\gpt\_release") / NAME
    archive = bundle.with_suffix(".zip")
    if bundle.exists() or archive.exists():
        raise ValueError("Package already exists; choose a distinct package name")
    metadata = json.loads((frame / "result.json").read_text(encoding="utf-8"))
    report = json.loads((sample / "report.json").read_text(encoding="utf-8"))
    files = {
        "data/xyz_m_provisional.npy": sample / "xyz_m_provisional.npy",
        "data/cloud_ir_gray_provisional.ply": sample / "cloud_ir_gray_provisional.ply",
        "data/depth_m_provisional.npy": frame / "depth_m_provisional.npy",
        "data/disparity_pixels_native.npy": frame / "disparity_pixels_native.npy",
        "data/disparity_pixels_inference.npy": Path(metadata["raw_disparity_path"]),
        "images/left_ir.png": Path(metadata["source_left"]),
        "images/right_ir.png": Path(metadata["source_right"]),
        "images/depth_colour.png": frame / "depth_colour.png",
        "images/validity_mask.png": frame / "validity_mask.png",
        "images/preview.jpg": frame / "preview.jpg",
        "calibration/calibration.json": frame.parent / "calibration.json",
    }
    for source in files.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    bundle.mkdir(parents=True)
    origins = []
    for relative, source in files.items():
        target = bundle / relative
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise ValueError(f"Copy hash mismatch: {relative}")
        origins.append({"package_file": relative, "original_filename": source.name,
                        "sha256": digest(target), "bytes": target.stat().st_size})
    (bundle / "metadata").mkdir()
    (bundle / "tools").mkdir()
    report.update(source_frame="sample25_DOE0", xyz_file="data/xyz_m_provisional.npy",
                  pointcloud_file="data/cloud_ir_gray_provisional.ply",
                  paths_relative_to="package_root", depth_file="data/depth_m_provisional.npy",
                  calibration_file="calibration/calibration.json")
    write_json(bundle / "metadata/xyz_report.json", report)
    for key in ("source_result_path", "source_preview_path", "raw_disparity_href", "source_preview_href"):
        metadata.pop(key, None)
    metadata.update(source_left="images/left_ir.png", source_right="images/right_ir.png",
                    raw_disparity_path="data/disparity_pixels_inference.npy",
                    disparity_native_file="data/disparity_pixels_native.npy",
                    depth_file="data/depth_m_provisional.npy", validity_mask="images/validity_mask.png",
                    paths_relative_to="package_root")
    write_json(bundle / "metadata/frame_report.json", metadata)
    write_json(bundle / "metadata/provenance.json", {
        "sample": "25", "doe": 0, "camera_serial": "CPC93630004R",
        "model": "standard FoundationStereo", "checkpoint": "23-51-11/model_best_bp2.pth",
        "iterations": 32, "attention_backend": "PyTorch SDPA compatibility",
        "network_input_wh": [640, 400], "output_wh": [1280, 800],
        "restoration": "nearest neighbour; native disparity multiplied by 2; metric depth unchanged",
        "model_weights_included": False, "paired_device_timestamps_equal": True,
        "independent_valid_depth_samples_before_restoration": 248139,
        "pointcloud_vertex_count": report["valid_points"], "files": origins,
    })
    readme = """# 双目深度 / XYZ / 点云查看包

本包是第25组、DOE=0的结果样例。解压后双击 `START_HERE.html` 查看原图与深度对照；可将PLY导入你已有的点云查看软件。无需本机工程路径、模型权重或联网。

## 文件与单位

| 文件 | 内容 |
|---|---|
| data/xyz_m_provisional.npy | 800×1280×3、float32；xyz[v,u]=[X,Y,Z]，单位米 |
| data/depth_m_provisional.npy | 800×1280、float32光轴Z，单位米 |
| data/cloud_ir_gray_provisional.ply | 二进制PLY，992556点；颜色来自左红外灰度 |
| data/disparity_pixels_native.npy | 原图1280×800网格的像素视差 |
| data/disparity_pixels_inference.npy | 内部640×400推理的像素视差 |
| images/left_ir.png、right_ir.png | 原始左右红外图，1280×800 |
| images/depth_colour.png | 原尺寸深度伪彩色，统一0.3～5米色标 |
| images/validity_mask.png | 255可计算、0无效；不是精度或置信度 |
| images/preview.jpg | 左右对照排版预览，不能当作深度数据使用 |
| calibration/calibration.json | 提供的相机标定原字节副本 |
| metadata/ | 参数、示例坐标和来源摘要；所有路径以包根目录为基准 |

坐标原点在左IR相机光心，X向右、Y向下、Z沿光轴向前；像素(u,v)以左上角为(0,0)。Z不是欧氏距离，到光心距离为sqrt(X²+Y²+Z²)。无效XYZ的三个通道均NaN，PLY只包含有效位置，按图像行优先顺序排列。

本包深度NPY、XYZ和PLY已经是米，读取后不要再乘标定JSON中的`depth_unit_mm`。该字段是设备原始深度采集设置，本包的模型深度按双目焦距、基线和视差计算。

## 查询坐标

安装有Python和NumPy的环境中执行：

```text
python tools/read_xyz.py --u 640 --v 400
```

也可以读取数组：`xyz = numpy.load('data/xyz_m_provisional.npy')`，然后查询 `xyz[400,640]`。两点距离查询可加 `--u2 900 --v2 600`；应先确认选中的像素确实是待测对象的端点。

像素(640,400)的估计XYZ约为(0.008643,-0.020743,2.143970)m。无需NumPy的文件完整性检查：`python tools/verify_package.py`。

## 这份结果能说明什么

已验证数组尺寸、Z值与输入深度一致、NaN掩膜、PLY写入读回和坐标回投实现。数值回投误差约3.45e-5像素，只说明计算和文件处理一致，不是实物测量精度。

当前深度为640×400推理后最近邻恢复至1280×800；原始有效深度位置248139个，约99万点不代表新增4倍观测细节。米制结果仍以历史IR已正确校正、该标定适用为前提；没有尺量真值，尚未完成尺寸精度验收。

点云属于左IR坐标系，未配准RGB、未转换到地面或世界坐标，未重建遮挡背面。标定中的DEPTH→COLOR外参不能直接证明IR_LEFT与DEPTH坐标系相同；历史RGB1280×720与标定1280×800也尚待对应。本包未加入RGB图。

后续可指定可见边或平面，提供尺量尺寸，再比较长度绝对误差、相对误差及不同DOE条件。
"""
    (bundle / "README_先读我.md").write_text(readme, encoding="utf-8")
    (bundle / "START_HERE.html").write_text("""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>双目三维坐标样例 · 25 / DOE 0</title>
<style>body{font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif;background:#f4f6f3;color:#183b3f;max-width:1120px;margin:35px auto;padding:0 22px}h1{line-height:1.3}img{width:100%;border-radius:10px;background:white}a{color:#006c65}section{background:white;padding:18px 24px;border-radius:10px;margin:18px 0}code{background:#edf2f0;padding:3px 6px}li{margin:6px 0}.note{border-left:4px solid #b67b27}</style>
<h1>双目三维坐标样例</h1><p>第25组 · DOE=0 · CPC93630004R · 标准FoundationStereo</p>
<img src="images/preview.jpg" alt="左红外原图与暂定深度对照，色标0.3至5米">
<section><b>可直接查看或导入的文件</b><ul>
<li><a href="data/cloud_ir_gray_provisional.ply">PLY点云</a>：992556个可计算点，左IR灰度着色。</li>
<li><a href="data/xyz_m_provisional.npy">逐像素XYZ数组</a>：800×1280×3，单位米；<code>xyz[v,u]</code>查询。</li>
<li><a href="data/depth_m_provisional.npy">原尺寸深度数组</a> · <a href="images/depth_colour.png">深度伪彩色</a> · <a href="images/validity_mask.png">有效掩膜</a></li>
<li><a href="images/left_ir.png">原始左IR</a> · <a href="images/right_ir.png">原始右IR</a> · <a href="calibration/calibration.json">相机参数</a></li>
<li><a href="README_先读我.md">完整说明</a> · <a href="metadata/xyz_report.json">坐标报告</a> · <a href="metadata/provenance.json">来源与推理参数</a></li>
</ul></section>
<section><b>坐标定义</b><p>原点为左IR光心，X右、Y下、Z前，单位米；Z是光轴深度。像素(u=640,v=400)的估计坐标约为(0.008643, −0.020743, 2.143970)m。无效像素用NaN表示。</p><p>Python+NumPy查询：<code>python tools/read_xyz.py --u 640 --v 400</code></p></section>
<section class="note"><b>当前为条件性三维估计，尚无实物尺寸精度结论</b><p>内部640×400推理后恢复原尺寸1280×800，没有增加独立深度细节。历史IR校正和标定适用性仍待确认；目前验证的是计算与文件一致性，需尺量真值才能评估实际误差。有效点数量不表示准确率。未做RGB配准，也没有地面/世界坐标或遮挡背面。</p></section>
<p>解压后本页可离线使用。所有资料均在当前文件夹中；校验：<code>python tools/verify_package.py</code>。</p></html>""", encoding="utf-8")
    (bundle / "tools/read_xyz.py").write_text('''"""Query a pixel or two points. Requires NumPy; no model or GPU needed."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--u", type=int, default=640)
parser.add_argument("--v", type=int, default=400)
parser.add_argument("--u2", type=int)
parser.add_argument("--v2", type=int)
args = parser.parse_args()
if (args.u2 is None) != (args.v2 is None):
    parser.error("--u2 and --v2 must be given together")
root = Path(__file__).resolve().parents[1]
xyz = np.load(root / "data/xyz_m_provisional.npy", allow_pickle=False, mmap_mode="r")
def point(u, v):
    if not (0 <= v < xyz.shape[0] and 0 <= u < xyz.shape[1]):
        parser.error("Pixel outside image")
    p = xyz[v, u].astype(float)
    valid = bool(np.isfinite(p).all())
    return p, {"u": u, "v": v, "valid": valid,
               "xyz_m": p.tolist() if valid else None,
               "range_m": float(np.linalg.norm(p)) if valid else None}
a, first = point(args.u, args.v)
result = {"status": "provisional_metric_accuracy_unverified",
          "frame": "left_IR_optical_Xright_Ydown_Zforward", "point1": first}
if args.u2 is not None:
    b, second = point(args.u2, args.v2)
    result["point2"] = second
    result["point_distance_m"] = float(np.linalg.norm(a-b)) if first["valid"] and second["valid"] else None
print(json.dumps(result, indent=2, allow_nan=False))
''', encoding="utf-8")
    (bundle / "tools/verify_package.py").write_text('''"""Verify every packaged file using standard Python only."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "SHA256.json").read_text(encoding="utf-8"))
errors = []
for name, expected in manifest.items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        errors.append(name)
if errors:
    raise SystemExit("FAILED: " + ", ".join(errors))
print("PASS: " + str(len(manifest)) + " files verified")
''', encoding="utf-8")
    manifest = {p.relative_to(bundle).as_posix(): digest(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
    write_json(bundle / "SHA256.json", manifest)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file in sorted(bundle.rglob("*")):
            if file.is_file():
                z.write(file, f"{NAME}/{file.relative_to(bundle).as_posix()}")
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
        prefix = NAME + "/"
        for name, expected in manifest.items():
            if hashlib.sha256(z.read(prefix + name)).hexdigest() != expected:
                raise ValueError(f"Archive hash mismatch: {name}")
        xyz = np.load(io.BytesIO(z.read(prefix + "data/xyz_m_provisional.npy")), allow_pickle=False)
        depth = np.load(io.BytesIO(z.read(prefix + "data/depth_m_provisional.npy")), allow_pickle=False)
        np.testing.assert_array_equal(xyz[..., 2], depth)
        if xyz.shape != (800, 1280, 3) or int(np.isfinite(xyz).all(axis=2).sum()) != 992556:
            raise ValueError("Unexpected XYZ content")
        for name in (n for n in manifest if n.startswith("images/")):
            image = cv2.imdecode(np.frombuffer(z.read(prefix + name), np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None or (not name.endswith("preview.jpg") and image.shape[:2] != (800, 1280)):
                raise ValueError(f"Image verification failed: {name}")
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ("src", "href") and not (bundle / value).is_file():
                    raise ValueError(f"Nonportable HTML link: {value}")
    Links().feed((bundle / "START_HERE.html").read_text(encoding="utf-8"))
    archive_hash = digest(archive)
    archive.with_suffix(".zip.sha256").write_text(archive_hash + "  " + archive.name + "\n", encoding="utf-8")
    verification = {"status": "passed", "archive": str(archive), "archive_bytes": archive.stat().st_size,
                    "archive_sha256": archive_hash, "verified_content_files": len(manifest),
                    "zip_crc_verified": True, "manifest_verified_inside_zip": True,
                    "xyz_and_depth_verified_inside_zip": True, "images_decoded_inside_zip": True,
                    "html_links_portable": True, "metric_accuracy_verified": False}
    write_json(archive.with_suffix(".verification.json"), verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
